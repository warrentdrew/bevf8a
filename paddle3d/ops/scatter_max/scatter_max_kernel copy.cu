// Copyright 2023 Baidu Inc. All Rights Reserved.
// @author: Guojun Wang (wangguojun01@baidu.com)
// @file: scatter_max_kernel.cu
// @brief: scatter_max_kernel

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <limits>

#include "atomics.cuh"
// #define THREADS 512
// #define BLOCKS(N) (N + THREADS - 1) / THREADS

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define THREADS_PER_BLOCK 512
inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return MIN(optimal_block_num, max_block_num);
}

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

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void ScatterMaxForwardKernel(const scalar_t* src_data,
                                        const long* index_data,
                                        scalar_t* out_data, int C_in, int N_out,
                                        int numel);

template <>
__global__ void ScatterMaxForwardKernel(const float* src_data,
                                        const long* index_data, float* out_data,
                                        int C_in, int N_out, int numel) {
  CUDA_1D_KERNEL_LOOP(thread_idx, numel) {
    int index_idx = thread_idx / C_in;
    int channel_idx = thread_idx - C_in * index_idx;
    int out_idx = index_data[index_idx];
    atomicMax(out_data + out_idx * C_in + channel_idx, src_data[thread_idx]);
  }
}
template <>
__global__ void ScatterMaxForwardKernel(const half* src_data,
                                        const long* index_data, half* out_data,
                                        int C_in, int N_out, int numel) {
  CUDA_1D_KERNEL_LOOP(thread_idx, numel) {
    int index_idx = thread_idx / C_in;
    int channel_idx = thread_idx - C_in * index_idx;
    int out_idx = index_data[index_idx];
    atomicMax(out_data + out_idx * C_in + channel_idx, src_data[thread_idx]);
  }
}
// #endif

template <>
__global__ void ScatterMaxForwardKernel(const at::Half* src_data,
                                        const long* index_data,
                                        at::Half* out_data, int C_in, int N_out,
                                        int numel) {
  CUDA_1D_KERNEL_LOOP(thread_idx, numel) {
    int index_idx = thread_idx / C_in;
    int channel_idx = thread_idx - C_in * index_idx;
    int out_idx = index_data[index_idx];
    // half* out_data_ = reinterpret_cast<half*>(out_data);
    half* out_data_ = (half*)(out_data + out_idx * C_in + channel_idx);
    atomicMax(out_data_, src_data[thread_idx]);
  }
}
template <typename scalar_t>
__global__ void initKernel(const int size, const scalar_t f, scalar_t* ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    ptr[idx] = f;
  }
}

template <typename scalar_t>
__global__ void ScatterMaxArgKernel(const scalar_t* src_data,
                                    const long* index_data,
                                    const scalar_t* out_data,
                                    long* arg_out_data, int C_in, int N_out,
                                    int numel);

template <>
__global__ void ScatterMaxArgKernel(const float* src_data,
                                    const long* index_data,
                                    const float* out_data, long* arg_out_data,
                                    int C_in, int N_out, int numel) {
  CUDA_1D_KERNEL_LOOP(thread_idx, numel) {
    int index_idx = thread_idx / C_in;
    int channel_idx = thread_idx - C_in * index_idx;

    int out_idx = index_data[index_idx];  // 表示输出tensor的索引
    if (out_idx < 0) return;
    assert(out_idx < N_out);
    if (src_data[thread_idx] == out_data[out_idx * C_in + channel_idx]) {
      arg_out_data[out_idx * C_in + channel_idx] = index_idx;
    }
  }
}

template <>
__global__ void ScatterMaxArgKernel(const half* src_data,
                                    const long* index_data,
                                    const half* out_data, long* arg_out_data,
                                    int C_in, int N_out, int numel) {
  // #if __CUDA_ARCH__ >= 700 && !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(thread_idx, numel) {
    int index_idx = thread_idx / C_in;
    int channel_idx = thread_idx - C_in * index_idx;
    int out_idx = index_data[index_idx];  // 表示输出tensor的索引
    if (out_idx < 0) return;
    assert(out_idx < N_out);
    // if (src_data[thread_idx] == out_data[out_idx * C_in + channel_idx]) {
    //   arg_out_data[out_idx * C_in + channel_idx] = index_idx;
    // }
    if (__heq(src_data[thread_idx], out_data[out_idx * C_in + channel_idx])) {
      arg_out_data[out_idx * C_in + channel_idx] = index_idx;
    }
    //  if (	__half2float(src_data[thread_idx]) ==
    //  __half2float(out_data[out_idx * C_in + channel_idx])) {
    //   arg_out_data[out_idx * C_in + channel_idx] = index_idx;
    // }
  }
  // #endif
}

template <>
__global__ void ScatterMaxArgKernel(const at::Half* src_data,
                                    const long* index_data,
                                    const at::Half* out_data,
                                    long* arg_out_data, int C_in, int N_out,
                                    int numel) {
  // #if __CUDA_ARCH__ >= 700 && !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(thread_idx, numel) {
    int index_idx = thread_idx / C_in;
    int channel_idx = thread_idx - C_in * index_idx;
    int out_idx = index_data[index_idx];  // 表示输出tensor的索引
    if (out_idx < 0) return;
    assert(out_idx < N_out);
    if (__heq(static_cast<half>(src_data[thread_idx]),
              static_cast<half>(out_data[out_idx * C_in + channel_idx]))) {
      arg_out_data[out_idx * C_in + channel_idx] = index_idx;
    }
  }
}

// template <>
// __global__ void ScatterMaxArgKernel(const double* src_data,
//                                     const long* index_data,
//                                     const double* out_data, long*
//                                     arg_out_data, int C_in, int N_out, int
//                                     numel) {
// }

void scatter_max_launcher(at::Tensor src, at::Tensor index, int N_in, int C_in,
                          int N_out, at::Tensor out, at::Tensor arg_out) {
  const long* index_data = index.data_ptr<long>();
  long* arg_out_data = arg_out.data_ptr<long>();
  int numel_in = N_in * C_in;
  int numel_out = N_out * C_in;
  if (src.scalar_type() == at::ScalarType::Half) {
    // #if __CUDA_ARCH__ >= 700 && !defined(__CUDA_ARCH__)

    const at::Half* src_data = src.data_ptr<at::Half>();
    at::Half* out_data = out.data_ptr<at::Half>();
    initKernel<at::Half><<<GET_BLOCKS(numel_out), THREADS_PER_BLOCK>>>(
        numel_out, at::Half(-65504.f), out_data);
    ScatterMaxForwardKernel<at::Half>
        <<<GET_BLOCKS(numel_in), THREADS_PER_BLOCK>>>(
            src_data, index_data, out_data, C_in, N_out, numel_in);
    ScatterMaxArgKernel<at::Half><<<GET_BLOCKS(numel_in), THREADS_PER_BLOCK>>>(
        src_data, index_data, out_data, arg_out_data, C_in, N_out, numel_in);
    // #endif
  } else if (src.scalar_type() == at::ScalarType::Float) {
    const float* src_data = src.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    initKernel<float><<<GET_BLOCKS(numel_out), THREADS_PER_BLOCK>>>(
        numel_out, float(-FLT_MAX), out_data);
    ScatterMaxForwardKernel<float><<<GET_BLOCKS(numel_in), THREADS_PER_BLOCK>>>(
        src_data, index_data, out_data, C_in, N_out, numel_in);
    ScatterMaxArgKernel<float><<<GET_BLOCKS(numel_in), THREADS_PER_BLOCK>>>(
        src_data, index_data, out_data, arg_out_data, C_in, N_out, numel_in);
  }
  CHECK_CALL(cudaGetLastError());
}