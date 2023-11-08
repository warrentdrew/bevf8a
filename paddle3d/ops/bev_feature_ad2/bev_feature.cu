/*
cnnseg feature generator
Written by Dong Jiarong
All Rights Reserved 2022.
*/

#include <vector>

#include "paddle/extension.h"

#define THREADS_PER_BLOCK 512
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

__device__ float atomic_exch(float* addr, float val) {
  return atomicExch(addr, (val));
}

__device__ void atomicMax(float* max_height_addr, float pz) {
  float old_pz = *max_height_addr;
  do {
    old_pz = atomic_exch(max_height_addr, (pz));
    if (pz < old_pz) {
      pz = old_pz;
    }
  } while (pz > (*max_height_addr));
}

__global__ void MapKernel(const int n, const int point_dim, const float* pc,
                          const int* point2grid, float* max_height_data,
                          float* mean_height_data, float* mean_intensity_data,
                          float* count_data) {
  CUDA_KERNEL_LOOP(i, n) {
    int idx = point2grid[i];
    if (idx == -1) {
      continue;
    }
    // point (x, y, z, i)
    float pz = pc[i * point_dim + 2];
    float pi = pc[i * point_dim + 3] / 255.0;
    atomicMax(&max_height_data[idx], pz);
    atomicAdd(&mean_height_data[idx], pz);
    if (mean_intensity_data != nullptr) {
      atomicAdd(&mean_intensity_data[idx], pi);
    }
    atomicAdd(&count_data[idx], (float)1);
  }
}

__global__ void TopIntensityKernel(const int n, const int point_dim,
                                   const float* pc, const int* point2grid,
                                   float* top_intensity,
                                   float* max_height_data) {
  CUDA_KERNEL_LOOP(i, n) {
    int idx = point2grid[i];
    if (idx == -1) {
      continue;
    }
    // printf("voxel idx: %d, point id: %d, cur_z: %f,  max_z: %f, eq: %d \n",
    // idx, i, pz, max_height_data[idx], static_cast<int>(pz ==
    // max_height_data[idx]));
    if (pc[i * point_dim + 2] == max_height_data[idx]) {
      top_intensity[idx] = pc[i * point_dim + 3] / 255.0;
    }
  }
}

// vis feature
// logodds func2
__global__ void VoxelTraversalKernel(
    const int cloud_size, const int point_dim, const float voxel_size_x,
    const float voxel_size_y, const float voxel_size_z, const float pxmin,
    const float pymin, const float pzmin, const int grid_size_x,
    const int grid_size_y, const int grid_size_z, const float ray_start_x,
    const float ray_start_y, const float ray_start_z, float lo_occupied,
    float lo_free, const float* pc, float* logodds) {
  CUDA_KERNEL_LOOP(i, cloud_size) {
    float ray_end_x = pc[i * point_dim + 0] - pxmin;
    float ray_end_y = pc[i * point_dim + 1] - pymin;
    float ray_end_z = pc[i * point_dim + 2] - pzmin;

    float ray_x = ray_end_x - ray_start_x;
    float ray_y = ray_end_y - ray_start_y;
    float ray_z = ray_end_z - ray_start_z;

    // This id of the first/current voxel hit by the ray.
    // Using floor (round down) is actually very important,
    // the implicit int-casting will round up for negative numbers.
    int current_voxel_x = static_cast<int>(floor(ray_start_x / voxel_size_x));
    int current_voxel_y = static_cast<int>(floor(ray_start_y / voxel_size_y));
    int current_voxel_z = static_cast<int>(floor(ray_start_z / voxel_size_z));

    // The id of the last voxel hit by the ray.
    // TODO: what happens if the end point is on a border?
    int last_voxel_x = static_cast<int>(floor(ray_end_x / voxel_size_x));
    int last_voxel_y = static_cast<int>(floor(ray_end_y / voxel_size_y));
    int last_voxel_z = static_cast<int>(floor(ray_end_z / voxel_size_z));

    // In which direction the voxel ids are incremented.
    int stepX = (ray_x >= 0) ? 1 : -1;  // correct
    int stepY = (ray_y >= 0) ? 1 : -1;  // correct
    int stepZ = (ray_z >= 0) ? 1 : -1;  // correct

    // Distance along the ray to the next voxel border from the current position
    // (tMaxX, tMaxY, tMaxZ).
    float next_voxel_boundary_x =
        (current_voxel_x + stepX) * voxel_size_x;  // correct
    float next_voxel_boundary_y =
        (current_voxel_y + stepY) * voxel_size_y;  // correct
    float next_voxel_boundary_z =
        (current_voxel_z + stepZ) * voxel_size_z;  // correct

    // tMaxX, tMaxY, tMaxZ -- distance until next intersection with voxel-border
    // the value of t at which the ray crosses the first vertical voxel boundary
    float tMaxX = (ray_x != 0) ? (next_voxel_boundary_x - ray_start_x) / ray_x
                               : FLT_MAX;  //
    float tMaxY = (ray_y != 0) ? (next_voxel_boundary_y - ray_start_y) / ray_y
                               : FLT_MAX;  //
    float tMaxZ = (ray_z != 0) ? (next_voxel_boundary_z - ray_start_z) / ray_z
                               : FLT_MAX;  //

    // tDeltaX, tDeltaY, tDeltaZ --
    // how far along the ray we must move for the horizontal component to equal
    // the width of a voxel the direction in which we traverse the grid can only
    // be FLT_MAX if we never go in that direction
    float tDeltaX = (ray_x != 0) ? voxel_size_x / ray_x * stepX : FLT_MAX;
    float tDeltaY = (ray_y != 0) ? voxel_size_y / ray_y * stepY : FLT_MAX;
    float tDeltaZ = (ray_z != 0) ? voxel_size_z / ray_z * stepZ : FLT_MAX;

    //
    int G1 = grid_size_x;
    int G2 = grid_size_y * G1;

    // Note: I am not sure why there is a need to do this, but I am keeping it
    // for now possibly explained by:
    // https://github.com/francisengelmann/fast_voxel_traversal/issues/6
    int diff_x = 0, diff_y = 0, diff_z = 0;
    bool neg_ray = false;
    if (current_voxel_x != last_voxel_x && ray_x < 0) {
      diff_x--;
      neg_ray = true;
    }
    if (current_voxel_y != last_voxel_y && ray_y < 0) {
      diff_y--;
      neg_ray = true;
    }
    if (current_voxel_z != last_voxel_z && ray_z < 0) {
      diff_z--;
      neg_ray = true;
    }

    // ray casting loop
    bool truncated = false;
    int voxel_idx =
        current_voxel_z * G2 + current_voxel_y * G1 + current_voxel_x;
    // debug qiaqia
    if (current_voxel_x < 0 || current_voxel_x >= grid_size_x ||
        current_voxel_y < 0 || current_voxel_y >= grid_size_y ||
        current_voxel_z < 0 || current_voxel_z >= grid_size_z) {
      truncated = true;
    } else if (logodds[voxel_idx] < lo_free) {
      atomicMax(&logodds[voxel_idx], lo_free);
    }

    if (neg_ray) {
      current_voxel_x += diff_x;
      current_voxel_y += diff_y;
      current_voxel_z += diff_z;
      voxel_idx = current_voxel_z * G2 + current_voxel_y * G1 + current_voxel_x;
      // debug qiaqia
      if (current_voxel_x < 0 || current_voxel_x >= grid_size_x ||
          current_voxel_y < 0 || current_voxel_y >= grid_size_y ||
          current_voxel_z < 0 || current_voxel_z >= grid_size_z) {
        truncated = true;
      } else if (logodds[voxel_idx] < lo_free) {
        atomicMax(&logodds[voxel_idx], lo_free);
      }
    }

    while (!truncated && !(current_voxel_x == last_voxel_x &&
                           current_voxel_y == last_voxel_y &&
                           current_voxel_z == last_voxel_z)) {
      if (tMaxX < tMaxY) {
        if (tMaxX < tMaxZ) {
          current_voxel_x += stepX;
          truncated = (current_voxel_x < 0 || current_voxel_x >= grid_size_x);
          tMaxX += tDeltaX;
        } else {
          current_voxel_z += stepZ;
          truncated = (current_voxel_z < 0 || current_voxel_z >= grid_size_z);
          tMaxZ += tDeltaZ;
        }
      } else {
        if (tMaxY < tMaxZ) {
          current_voxel_y += stepY;
          truncated = (current_voxel_y < 0 || current_voxel_y >= grid_size_y);
          tMaxY += tDeltaY;
        } else {
          current_voxel_z += stepZ;
          truncated = (current_voxel_z < 0 || current_voxel_z >= grid_size_z);
          tMaxZ += tDeltaZ;
        }
      }
      if (truncated) {
        break;
      }
      voxel_idx = current_voxel_z * G2 + current_voxel_y * G1 + current_voxel_x;
      // if (current_voxel_x < 0 || current_voxel_x >= grid_size_x ||
      //   current_voxel_y < 0 || current_voxel_y >= grid_size_y ||
      //   current_voxel_z < 0 || current_voxel_z >= grid_size_z) {
      //   printf("2 occupy voxel_idx %d, z %d, y %d, x %d \n",
      //         voxel_idx, current_voxel_z, current_voxel_y, current_voxel_x);
      // }
      if (logodds[voxel_idx] < lo_free) {
        // printf("voxel_idx %d, z %d, y %d, x %d \n", voxel_idx,
        // current_voxel_z, current_voxel_y, current_voxel_x);
        atomicMax(&logodds[voxel_idx], lo_free);
      }
    }

    if (!truncated) {
      voxel_idx = last_voxel_z * G2 + last_voxel_y * G1 + last_voxel_x;
      // if (current_voxel_x < 0 || current_voxel_x >= grid_size_x ||
      //   current_voxel_y < 0 || current_voxel_y >= grid_size_y ||
      //   current_voxel_z < 0 || current_voxel_z >= grid_size_z) {
      //   printf("3 occupy voxel_idx %d, z %d, y %d, x %d \n",
      //         voxel_idx, last_voxel_z, last_voxel_y, last_voxel_x);
      // }
      atomicMax(&logodds[voxel_idx], lo_occupied);
      // logodds[voxel_idx] = lo_occupied;
    }
  }
}

void mapfeatureLauncher(const cudaStream_t& stream, const int num,
                        const int point_dim, const float* points_data,
                        const int* points2grid_data, float* max_height_data,
                        float* mean_height_data, float* mean_intensity_data,
                        float* count_data) {
  dim3 blocks(
      DIVUP(num, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  MapKernel<<<blocks, threads, 0, stream>>>(
      num, point_dim, points_data, points2grid_data, max_height_data,
      mean_height_data, mean_intensity_data, count_data);
  // cudaStreamSynchronize(stream);
}

void TopIntensityLauncher(const cudaStream_t& stream, const int num,
                          const int point_dim, const float* points_data,
                          const int* points2grid_data,
                          float* top_intensity_data, float* max_height_data) {
  dim3 blocks(
      DIVUP(num, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);
  TopIntensityKernel<<<blocks, threads, 0, stream>>>(
      num, point_dim, points_data, points2grid_data, top_intensity_data,
      max_height_data);
  // cudaStreamSynchronize(stream);
}

// logodds func3
void ComputeLogOddsGPULauncher(
    const cudaStream_t& stream, const int num, const int point_dim,
    const float voxel_size_x, const float voxel_size_y,
    const float voxel_size_z, const float pxmin, const float pymin,
    const float pzmin, const int grid_size_x, const int grid_size_y,
    const int grid_size_z, const float offset_origin_x,
    const float offset_origin_y, const float offset_origin_z, float lo_occupied,
    float lo_free, const float* points_data, float* logodds_data) {
  dim3 blocks(
      DIVUP(num, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  VoxelTraversalKernel<<<blocks, threads, 0, stream>>>(
      num, point_dim, voxel_size_x, voxel_size_y, voxel_size_z, pxmin, pymin,
      pzmin, grid_size_x, grid_size_y, grid_size_z, offset_origin_x,
      offset_origin_y, offset_origin_z, lo_occupied, lo_free, points_data,
      logodds_data);
  // cudaStreamSynchronize(stream);
}