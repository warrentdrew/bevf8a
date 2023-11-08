// Copyright 2023 Baidu Inc. All Rights Reserved.
// @author: Guojun Wang (wangguojun01@baidu.com)
// @file: dynamic_point_pool_kernel.cu
// @brief: dynamic_point_pool_kernel
#include <assert.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>
#include <vector>

#include "device_launch_parameters.h"
using namespace cooperative_groups;
#define THREADS_PER_BLOCK 512
#define LARGE_NEG -10000
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

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  // int max_block_num = 4096;
  // return MIN(optimal_block_num, max_block_num);
  return optimal_block_num;
}


__forceinline__ __device__ int atomicAggInc(int *ctr) {
  auto g = cooperative_groups::coalesced_threads();
  int warp_res;
  if (g.thread_rank() == 0)
    warp_res = atomicAdd(ctr, g.size());
  return g.shfl(warp_res, 0) + g.thread_rank();
}

__forceinline__ __device__ int
check_pt_in_box3d(const float x, const float y, const float z, const float cx,
                  const float cy, const float cz, const float w, const float l,
                  const float h, const float rz, const float *extra_wlh,
                  const float max_dist, float &local_x, float &local_y) {
  float large_w = w + extra_wlh[0];
  float large_l = l + extra_wlh[1];
  float large_h = h + extra_wlh[2];

  float local_z = z - cz;
  float offset_x = x - cx;
  float offset_y = y - cy;
  if ((local_z > large_h / 2.f) || (local_z < -large_h / 2.f) ||
      (offset_x > max_dist) || (offset_x < -max_dist) ||
      (offset_y > max_dist) || (offset_y < -max_dist)) {
    return 0;
  }

  float cosa, sina;
  cosa = cos(rz);
  sina = sin(rz);
  local_x = offset_x * cosa + offset_y * (-sina);
  local_y = offset_x * sina + offset_y * cosa;

  float in_flag = (local_x > -w / 2.f) && (local_x < w / 2.f) &&
                  (local_y > -l / 2.f) && (local_y < l / 2.f) &&
                  (local_z > -h / 2.f) && (local_z < h / 2.f);
  if (in_flag > 0) {
    return 2;
  }
  float in_large_flag = (local_x > -large_w / 2.f) &&
                        (local_x < large_w / 2.f) &&
                        (local_y > -large_l / 2.f) && (local_y < large_l / 2.f);
  if (in_large_flag > 0) {
    return 1;
  }
  return 0;

  // 0: out of large box
  // 1: in large box but out of small box
  // 2: in small box
}

__global__ void DynamicPointPoolKernel(
    const int num_threads, const int boxes_num, const int pts_num,
    const int max_pts_num, const float *rois, const float *pts,
    const int info_dim, const float *extra_wlh,
    float *out_pts_xyz, float *out_pts_feats, float *out_pts_info,
    long *out_roi_idx, int *global_counter, int *inbox_counter) {

  CUDA_1D_KERNEL_LOOP(idx, num_threads) {
    int box_idx = idx / pts_num;
    int pt_idx = idx - pts_num * box_idx;

    pts += pt_idx * 4;
    rois += box_idx * 7;
    float px = pts[0];
    float py = pts[1];
    float pz = pts[2];
    float pi = pts[3];

    float cx = rois[0];
    float cy = rois[1];
    float cz = rois[2];
    float w = rois[3];
    float l = rois[4];
    float h = rois[5];
    float rz = rois[6];

    float local_x = 0;
    float local_y = 0;

    int cur_in_flag = check_pt_in_box3d(px, py, pz, cx, cy, cz, w, l, h, rz,
                                        extra_wlh, 30.f, local_x, local_y);
    if (cur_in_flag > 0) {
      if (inbox_counter[box_idx] > max_pts_num) {
         return;
      }
      int cnt = atomicAggInc(&inbox_counter[box_idx]);
      if (cnt > max_pts_num) {
        return;
      }
      // int cnt = atomicAdd(&inbox_counter[box_idx], 1);
      float local_z = pz - cz; // to roi center
      int new_pt_idx = atomicAggInc(&global_counter[0]);
      int pts_offset = new_pt_idx * 3;
      out_pts_xyz[pts_offset] = px;
      out_pts_xyz[pts_offset + 1] = py;
      out_pts_xyz[pts_offset + 2] = pz;
      out_pts_feats[new_pt_idx] = pi;
      out_roi_idx[new_pt_idx] = box_idx;
      int info_offsets = new_pt_idx * info_dim;
      out_pts_info[info_offsets] = local_x;
      out_pts_info[info_offsets + 1] = local_y;
      out_pts_info[info_offsets + 2] = local_z;
      out_pts_info[info_offsets + 3] = local_x + w / 2;
      out_pts_info[info_offsets + 4] = local_y + l / 2;
      out_pts_info[info_offsets + 5] = local_z + h / 2;
      out_pts_info[info_offsets + 6] = -local_x + w / 2;
      out_pts_info[info_offsets + 7] = -local_y + l / 2;
      out_pts_info[info_offsets + 8] = -local_z + h / 2;
      out_pts_info[info_offsets + 9] = cur_in_flag == 1 ? 1 : 0;
    }
  }
}
void dynamic_point_pool_launcher(const int boxes_num, const int pts_num,
                                   const int max_pts_num, const float* rois,
                                   const float* pts, const int info_dim,
                                   const float* extra_wlh, float* out_pts_xyz,
                                   float* out_pts_feats, float* out_pts_info,
                                  long* out_roi_idx,
                                   int* global_counter) {
  int* inbox_counter = NULL;
  CHECK_CALL(cudaMalloc(&inbox_counter, boxes_num * sizeof(int)));
  CHECK_CALL(cudaMemset(inbox_counter, 0, boxes_num * sizeof(int)));
  int num_threads = boxes_num * pts_num;
  DynamicPointPoolKernel
      <<<GET_BLOCKS(num_threads, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
          num_threads, boxes_num, pts_num, 
          max_pts_num, rois,
          pts, info_dim, extra_wlh, out_pts_xyz,
          out_pts_feats, out_pts_info,
          out_roi_idx, 
          global_counter, inbox_counter);
  cudaFree(inbox_counter);
  return;
}