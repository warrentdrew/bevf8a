// Copyright 2023 Baidu Inc. All Rights Reserved.
// @author: Guojun Wang (wangguojun01@baidu.com)
// @file: roi_align_rotated_kernel.cu
// @brief: roi_align_rotated_kernel

// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
// #include <torch/extension.h>
// #include <torch/serialize/tensor.h>
// #include <torch/types.h>

// #include <ATen/cuda/CUDAApplyUtils.cuh>
// #include <ATen/cuda/detail/IndexUtils.cuh>
// #include <ATen/cuda/detail/TensorInfo.cuh>
#include <vector>
#include <limits>
#include <paddle/extension.h>
// using namespace at;

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

#define CHECK_CALL(call)                                                       \
  do {                                                                         \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess) {                                           \
      printf("CUDA Error:\n");                                                 \
      printf("    File:       %s\n", __FILE__);                                \
      printf("    Line:       %d\n", __LINE__);                                \
      printf("    Error code: %d\n", error_code);                              \
      printf("    Error text: %s\n", cudaGetErrorString(error_code));          \
      break;                                                                   \
    }                                                                          \
  } while (0)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return MIN(optimal_block_num, max_block_num);
}
template <typename T>
__device__ T bilinear_interpolate(const T* input, const int height,
                                  const int width, T y, T x,
                                  const int index /* index for debug only*/);

template <>
__device__ float bilinear_interpolate(
    const float* input, const int height, const int width, float y, float x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.f || y > height || x < -1.f || x > width) return 0.f;
  if (y <= 0.f) y = 0.f;
  if (x <= 0.f) x = 0.f;

  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x);
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = static_cast<float>(y_low);
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = static_cast<float>(x_low);
  } else {
    x_high = x_low + 1;
  }

  float ly = y - static_cast<float>(y_low);
  float lx = x - static_cast<float>(x_low);
  float hy = 1.f - ly, hx = 1.f - lx;
  // do bilinear interpolation
  float v1 = input[y_low * width + x_low];
  float v2 = input[y_low * width + x_high];
  float v3 = input[y_high * width + x_low];
  float v4 = input[y_high * width + x_high];
  float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <>
__device__ half bilinear_interpolate(const half* input, const int height,
                                     const int width, half y, half x,
                                     const int index) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
  // deal with cases that inverse elements are out of feature map boundary
  if (__hlt(y, __float2half(-1.f)) || __hgt(y, __int2half_rd(height)) ||
      __hlt(x, __float2half(-1.f)) || __hgt(x, __int2half_rd(width)))
    return __float2half(0.f);

  if (__hle(y, __float2half(0.f))) y = __float2half(0.f);
  if (__hle(x, __float2half(0.f))) x = __float2half(0.f);

  int y_low = __half2int_rd(y);
  int x_low = __half2int_rd(x);
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = __int2half_rd(y_low);
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = __int2half_rd(x_low);
  } else {
    x_high = x_low + 1;
  }

  half ly = y - __int2half_rd(y_low);
  half lx = x - __int2half_rd(x_low);
  half hy = __float2half(1.f) - ly, hx = __float2half(1.f) - lx;
  // do bilinear interpolation
  half v1 = input[y_low * width + x_low];
  half v2 = input[y_low * width + x_high];
  half v3 = input[y_high * width + x_low];
  half v4 = input[y_high * width + x_high];
  half w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  half val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
#else
  return 0;
#endif
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width, T y, T x, T& w1, T& w2, T& w3, T& w4,
    int& x_low, int& x_high, int& y_low, int& y_high,
    const int index /* index for debug only*/);

template <>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width, float y, float x, float& w1, float& w2,
    float& w3, float& w4, int& x_low, int& x_high, int& y_low, int& y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.f || y > static_cast<float>(height) || x < -1.f ||
      x > static_cast<float>(width)) {
    // empty
    w1 = w2 = w3 = w4 = 0.f;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0.f) y = 0.f;
  if (x <= 0.f) x = 0.f;

  y_low = static_cast<int>(y);
  x_low = static_cast<int>(x);

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = static_cast<float>(y_low);
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = static_cast<float>(x_low);
  } else {
    x_high = x_low + 1;
  }

  float ly = y - static_cast<float>(y_low);
  float lx = x - static_cast<float>(x_low);
  float hy = 1.f - ly, hx = 1.f - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

// template <>
// __device__ void bilinear_interpolate_gradient(
//     const int height, const int width, half y, half x, half& w1, half& w2,
//     half& w3, half& w4, int& x_low, int& x_high, int& y_low, int& y_high,
//     const int index /* index for debug only*/) {
//   // deal with cases that inverse elements are out of feature map boundary
//   if (__hlt(y, __float2half(-1.f)) || __hgt(y, __int2half_rd(height)) ||
//       __hlt(x, __float2half(-1.f)) || __hgt(x, __int2half_rd(width))) {
//     // empty
//     w1 = w2 = w3 = w4 = __float2half(0.f);
//     x_low = x_high = y_low = y_high = -1;
//     return;
//   }

//   if (__hle(y, __float2half(0.f))) y = __float2half(0.f);
//   if (__hle(x, __float2half(0.f))) x = __float2half(0.f);

//   y_low = __half2int_rd(y);
//   x_low = __half2int_rd(x);

//   if (y_low >= height - 1) {
//     y_high = y_low = height - 1;
//     y = __int2half_rd(y_low);
//   } else {
//     y_high = y_low + 1;
//   }

//   if (x_low >= width - 1) {
//     x_high = x_low = width - 1;
//     x = __int2half_rd(x_low);
//   } else {
//     x_high = x_low + 1;
//   }

//   half ly = y - __int2half_rd(y_low);
//   half lx = x - __int2half_rd(x_low);
//   half hy = __float2half(1.f) - ly, hx = __float2half(1.f) - lx;

//   w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

//   return;
// }

// /*** Forward ***/
template <typename scalar_t>
__global__ void RoIAlignRotatedForwardKernel(
    const int nthreads, const scalar_t* bottom_data,
    const scalar_t* bottom_rois, const scalar_t spatial_scale,
    const scalar_t pc_range_x, const scalar_t pc_range_y,
    const scalar_t voxel_x, const scalar_t voxel_y, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, scalar_t* top_data);

template <>
__global__ void RoIAlignRotatedForwardKernel(
    const int nthreads, const float* bottom_data, const float* bottom_rois,
    const float spatial_scale, const float pc_range_x, const float pc_range_y,
    const float voxel_x, const float voxel_y, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, float* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    // bottom_rois(batch, x, y, z, w, l, h, raw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const float* offset_bottom_rois = bottom_rois + n * 8;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    // float offset = aligned ? (float)0.5 : (float)0.0;
    float offset = 0.5f;
    float roi_center_w =
        (offset_bottom_rois[1] - pc_range_x) / voxel_x * spatial_scale - offset;
    ;
    float roi_center_h =
        (offset_bottom_rois[2] - pc_range_y) / voxel_y * spatial_scale - offset;
    float roi_width = offset_bottom_rois[4] / voxel_x * spatial_scale;
    float roi_height = offset_bottom_rois[5] / voxel_y * spatial_scale;
    float theta = offset_bottom_rois[7];

    float bin_size_h = roi_height / static_cast<float>(pooled_height);
    float bin_size_w = roi_width / static_cast<float>(pooled_width);

    const float* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    int roi_bin_grid_h = ceilf(bin_size_h);
    int roi_bin_grid_w = ceilf(bin_size_w);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    float roi_start_h = -roi_height / 2.f;
    float roi_start_w = -roi_width / 2.f;
    float cosscalar_theta = cos(theta);
    float sinscalar_theta = sin(theta);

    // We do average (integral) pooling inside a bin
    const float count = MAX(roi_bin_grid_h * roi_bin_grid_w, 1);

    float output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const float yy =
          roi_start_h + static_cast<float>(ph) * bin_size_h +
          static_cast<float>(iy + .5f) * bin_size_h /
              static_cast<float>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const float xx = roi_start_w + static_cast<float>(pw) * bin_size_w +
                         static_cast<float>(ix + .5f) * bin_size_w /
                             static_cast<float>(roi_bin_grid_w);

        // Rotate by theta (counterclockwise) around the center and translate
        float x = yy * sinscalar_theta + xx * cosscalar_theta + roi_center_w;
        float y = yy * cosscalar_theta - xx * sinscalar_theta + roi_center_h;

        float val = bilinear_interpolate<float>(offset_bottom_data, height,
                                                width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}

// template <>
// __global__ void RoIAlignRotatedForwardKernel(
//     const int nthreads, const at::Half* bottom_data,
//     const at::Half* bottom_rois, const at::Half spatial_scale,
//     const at::Half pc_range_x, const at::Half pc_range_y,
//     const at::Half voxel_x, const at::Half voxel_y, const int channels,
//     const int height, const int width, const int pooled_height,
//     const int pooled_width, at::Half* top_data) {
// #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
//   CUDA_1D_KERNEL_LOOP(index, nthreads) {
//     // (n, c, ph, pw) is an element in the pooled output
//     // bottom_rois(batch, x, y, z, w, l, h, raw)
//     int pw = index % pooled_width;
//     int ph = (index / pooled_width) % pooled_height;
//     int c = (index / pooled_width / pooled_height) % channels;
//     int n = index / pooled_width / pooled_height / channels;

//     const half* offset_bottom_rois = (half*)(bottom_rois + n * 8);
//     int roi_batch_ind = __half2int_rd(offset_bottom_rois[0]);

//     // Do not using rounding; this implementation detail is critical
//     // half offset = aligned ? (half)0.5 : (half)0.0;
//     half offset = __float2half(0.5f);
//     // half offset1 = static_cast<half>(0.5f);
//     // printf("float2half %f, static_cast %f", offset, offset1);
//     half roi_center_w = (offset_bottom_rois[1] - __half(pc_range_x)) /
//                             __half(voxel_x) * __half(spatial_scale) -
//                         offset;
//     ;
//     half roi_center_h = (offset_bottom_rois[2] - __half(pc_range_y)) /
//                             __half(voxel_y) * __half(spatial_scale) -
//                         offset;
//     half roi_width =
//         offset_bottom_rois[4] / __half(voxel_x) * __half(spatial_scale);
//     half roi_height =
//         offset_bottom_rois[5] / __half(voxel_y) * __half(spatial_scale);
//     half theta = offset_bottom_rois[7];

//     half bin_size_h = roi_height / __int2half_rd(pooled_height);
//     half bin_size_w = roi_width / __int2half_rd(pooled_width);

//     const half* offset_bottom_data =
//         (half*)(bottom_data + (roi_batch_ind * channels + c) * height * width);

//     int roi_bin_grid_h = __half2int_ru(hceil(bin_size_h));
//     int roi_bin_grid_w = __half2int_ru(hceil(bin_size_w));

//     // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
//     // Appropriate translation needs to be applied after.
//     half roi_start_h = -roi_height / __float2half(2.f);
//     half roi_start_w = -roi_width / __float2half(2.f);
//     half cosscalar_theta = hcos(theta);
//     half sinscalar_theta = hsin(theta);

//     // We do average (integral) pooling inside a bin
//     const half count = __int2half_rd(MAX(roi_bin_grid_h * roi_bin_grid_w, 1));

//     half output_val = __float2half(0.f);
//     for (int iy = 0; iy < roi_bin_grid_h; iy++) {
//       const half yy =
//           roi_start_h + __int2half_rd(ph) * bin_size_h +
//           __float2half(iy + .5f) * bin_size_h / __int2half_rd(roi_bin_grid_h);
//       for (int ix = 0; ix < roi_bin_grid_w; ix++) {
//         const half xx =
//             roi_start_w + __int2half_rd(pw) * bin_size_w +
//             __float2half(ix + .5f) * bin_size_w / __int2half_rd(roi_bin_grid_w);

//         // Rotate by theta (counterclockwise) around the center and translate
//         half x = yy * sinscalar_theta + xx * cosscalar_theta + roi_center_w;
//         half y = yy * cosscalar_theta - xx * sinscalar_theta + roi_center_h;

//         half val = bilinear_interpolate<half>(offset_bottom_data, height, width,
//                                               y, x, index);
//         output_val = output_val + val;
//       }
//     }
//     output_val = output_val / count;

//     ((half*)top_data)[index] = output_val;
//   }
// #endif
// }

template <typename scalar_t>
__global__ void RoIAlignRotatedBackwardKernel(
    const int nthreads, const scalar_t* top_diff, const scalar_t* bottom_rois,
    const scalar_t spatial_scale, const scalar_t pc_range_x,
    const scalar_t pc_range_y, const scalar_t voxel_x, const scalar_t voxel_y,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, scalar_t* bottom_diff);

// /*** Backward ***/
template <>
__global__ void RoIAlignRotatedBackwardKernel(
    const int nthreads, const float* top_diff, const float* bottom_rois,
    const float spatial_scale, const float pc_range_x, const float pc_range_y,
    const float voxel_x, const float voxel_y, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, float* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const float* offset_bottom_rois = bottom_rois + n * 8;
    int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);

    // Do not round
    float offset = 0.5f;
    float roi_center_w =
        (offset_bottom_rois[1] - pc_range_x) / voxel_x * spatial_scale - offset;
    ;
    float roi_center_h =
        (offset_bottom_rois[2] - pc_range_y) / voxel_y * spatial_scale - offset;
    float roi_width = offset_bottom_rois[4] / voxel_x * spatial_scale;
    float roi_height = offset_bottom_rois[5] / voxel_y * spatial_scale;
    float theta = offset_bottom_rois[7];
    float bin_size_h = roi_height / static_cast<float>(pooled_height);
    float bin_size_w = roi_width / static_cast<float>(pooled_width);

    float* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const float* offset_top_diff = top_diff + top_offset;
    const float top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = ceilf(bin_size_h);  // e.g., = 2
    int roi_bin_grid_w = ceilf(bin_size_w);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    float roi_start_h = -roi_height / 2.f;
    float roi_start_w = -roi_width / 2.f;
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);

    // We do average (integral) pooling inside a bin
    const float count =
        static_cast<float>(roi_bin_grid_h * roi_bin_grid_w);  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const float yy =
          roi_start_h + static_cast<float>(ph) * bin_size_h +
          (iy + .5f) * bin_size_h /
              static_cast<float>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const float xx =
            roi_start_w + static_cast<float>(pw) * bin_size_w +
            (ix + .5f) * bin_size_w / static_cast<float>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        float x = yy * sinTheta + xx * cosTheta + roi_center_w;
        float y = yy * cosTheta - xx * sinTheta + roi_center_h;

        float w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient<float>(height, width, y, x, w1, w2, w3,
                                             w4, x_low, x_high, y_low, y_high,
                                             index);

        float g1 = top_diff_this_bin * w1 / count;
        float g2 = top_diff_this_bin * w2 / count;
        float g3 = top_diff_this_bin * w3 / count;
        float g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
          atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
          atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
          atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
        }  // if
      }    // ix
    }      // iy
  }        // CUDA_1D_KERNEL_LOOP
}  // RoIAlignBackward

// template <>
// __global__ void RoIAlignRotatedBackwardKernel(
//     const int nthreads, const at::Half* top_diff, const at::Half* bottom_rois,
//     const at::Half spatial_scale, const at::Half pc_range_x,
//     const at::Half pc_range_y, const at::Half voxel_x, const at::Half voxel_y,
//     const int channels, const int height, const int width,
//     const int pooled_height, const int pooled_width, at::Half* bottom_diff) {
//   CUDA_1D_KERNEL_LOOP(index, nthreads) {
//     // (n, c, ph, pw) is an element in the pooled output
//     int pw = index % pooled_width;
//     int ph = (index / pooled_width) % pooled_height;
//     int c = (index / pooled_width / pooled_height) % channels;
//     int n = index / pooled_width / pooled_height / channels;

//     const half* offset_bottom_rois = (half*)(bottom_rois + n * 8);
//     int roi_batch_ind = __half2int_rd(offset_bottom_rois[0]);

//     // Do not round
//     half offset = __float2half(0.5f);
//     half roi_center_w = (offset_bottom_rois[1] - __half(pc_range_x)) /
//                             __half(voxel_x) * __half(spatial_scale) -
//                         offset;
//     ;
//     half roi_center_h = (offset_bottom_rois[2] - __half(pc_range_y)) /
//                             __half(voxel_y) * __half(spatial_scale) -
//                         offset;
//     half roi_width =
//         offset_bottom_rois[4] / __half(voxel_x) * __half(spatial_scale);
//     half roi_height =
//         offset_bottom_rois[5] / __half(voxel_y) * __half(spatial_scale);
//     half theta = offset_bottom_rois[7];
//     half bin_size_h = roi_height / __int2half_rd(pooled_height);
//     half bin_size_w = roi_width / __int2half_rd(pooled_width);

//     half* offset_bottom_diff =
//         (half*)(bottom_diff + (roi_batch_ind * channels + c) * height * width);

//     int top_offset = (n * channels + c) * pooled_height * pooled_width;
//     const half* offset_top_diff = (half*)(top_diff + top_offset);
//     const half top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

//     // We use roi_bin_grid to sample the grid and mimic integral
//     int roi_bin_grid_h = __half2int_ru(hceil(bin_size_h));
//     int roi_bin_grid_w = __half2int_ru(hceil(bin_size_w));

//     // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
//     // Appropriate translation needs to be applied after.
//     half roi_start_h = -roi_height / __float2half(2.f);
//     half roi_start_w = -roi_width / __float2half(2.f);
//     half cosTheta = hcos(theta);
//     half sinTheta = hsin(theta);

//     // We do average (integral) pooling inside a bin
//     const half count = __int2half_rd(roi_bin_grid_h * roi_bin_grid_w);

//     for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
//       const half yy = roi_start_h + __int2half_rd(ph) * bin_size_h +
//                       __float2half(iy + .5f) * bin_size_h /
//                           __int2half_rd(roi_bin_grid_h);  // e.g., 0.5, 1.5
//       for (int ix = 0; ix < roi_bin_grid_w; ix++) {
//         const half xx =
//             roi_start_w + __int2half_rd(pw) * bin_size_w +
//             __float2half(ix + .5f) * bin_size_w / __int2half_rd(roi_bin_grid_w);

//         // Rotate by theta around the center and translate
//         half x = yy * sinTheta + xx * cosTheta + roi_center_w;
//         half y = yy * cosTheta - xx * sinTheta + roi_center_h;

//         half w1, w2, w3, w4;
//         int x_low, x_high, y_low, y_high;

//         bilinear_interpolate_gradient<half>(height, width, y, x, w1, w2, w3, w4,
//                                             x_low, x_high, y_low, y_high,
//                                             index);

//         half g1 = top_diff_this_bin * w1 / count;
//         half g2 = top_diff_this_bin * w2 / count;
//         half g3 = top_diff_this_bin * w3 / count;
//         half g4 = top_diff_this_bin * w4 / count;

//         if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
//           atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
//           atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
//           atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
//           atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
//         }  // if
//       }    // ix
//     }      // iy
//   }        // CUDA_1D_KERNEL_LOOP
// }  // RoIAlignBackward


std::vector<paddle::Tensor> ROIAlignRotatedForwardCUDAKernelLauncher(
    const paddle::Tensor& input, const paddle::Tensor& rois, const float spatial_scale,
    const float pc_range_x, const float pc_range_y, const float voxel_x,
    const float voxel_y, const int channels, const int height, const int width,
    const int num_rois, const int pooled_height, const int pooled_width
    ) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  auto output = paddle::full({num_rois, channels, pooled_height, pooled_width}, 0, input.dtype(), paddle::GPUPlace()); 
  // if (input.scalar_type() == at::ScalarType::Half) {
  //   const at::Half* bottom_data = input.data_ptr<at::Half>();
  //   const at::Half* rois_data = rois.data_ptr<at::Half>();
  //   at::Half* top_data = output.data_ptr<at::Half>();

  //   RoIAlignRotatedForwardKernel<at::Half>
  //       <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
  //           output_size, bottom_data, rois_data, at::Half(spatial_scale),
  //           at::Half(pc_range_x), at::Half(pc_range_y), at::Half(voxel_x),
  //           at::Half(voxel_y), channels, height, width, pooled_height,
  //           pooled_width, top_data);
  // } else if (input.scalar_type() == at::ScalarType::Float) {
  const float* bottom_data = input.data<float>();
  const float* rois_data = rois.data<float>();
  float* top_data = output.data<float>();

  RoIAlignRotatedForwardKernel<float>
      <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_data, rois_data, spatial_scale, pc_range_x,
          pc_range_y, voxel_x, voxel_y, channels, height, width,
          pooled_height, pooled_width, top_data);
  // } else {
  //   printf("forward unsupport dtype!!!!");
  // }

  CHECK_CALL(cudaGetLastError());
  return {output};
}

std::vector<paddle::Tensor> ROIAlignRotatedBackwardCUDAKernelLauncher(
    const paddle::Tensor& top_grad, const paddle::Tensor& rois, const float spatial_scale,
    const float pc_range_x, const float pc_range_y, const float voxel_x,
    const float voxel_y, const int channels, const int height, const int width,
    const int num_rois, const int pooled_height, const int pooled_width, const int batch_size) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  auto bottom_grad = paddle::full({batch_size, channels, height, width}, 0, rois.dtype(), paddle::GPUPlace()); 
  // batch_size, num_channels, data_height,  #                             data_width

  // if (top_grad.scalar_type() == at::ScalarType::Half) {
  //   const at::Half* top_diff = top_grad.data_ptr<at::Half>();
  //   const at::Half* rois_data = rois.data_ptr<at::Half>();
  //   at::Half* bottom_diff = bottom_grad.data_ptr<at::Half>();
  //   RoIAlignRotatedBackwardKernel<at::Half>
  //       <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
  //           output_size, top_diff, rois_data, spatial_scale,
  //           at::Half(pc_range_x), at::Half(pc_range_y), at::Half(voxel_x),
  //           at::Half(voxel_y), channels, height, width, pooled_height,
  //           pooled_width, bottom_diff);
  // } else if (top_grad.scalar_type() == at::ScalarType::Float) {
    const float* top_diff = top_grad.data<float>();
    const float* rois_data = rois.data<float>();
    float* bottom_diff = bottom_grad.data<float>();
    RoIAlignRotatedBackwardKernel<float>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
            output_size, top_diff, rois_data, spatial_scale, pc_range_x,
            pc_range_y, voxel_x, voxel_y, channels, height, width,
            pooled_height, pooled_width, bottom_diff);
  // } else {
  //   printf("backward unsupport dtype!!!!");
  // }

  CHECK_CALL(cudaGetLastError());

  return {bottom_grad};
}
