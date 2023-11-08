#include <cuda_runtime.h>
#include <stdio.h>
#include <paddle/extension.h>

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.") //check input only check device on gpu

namespace {
int const threadsPerBlock = 512;
int const maxGridDim = 50000;
}  // namespace

__device__ __forceinline__ static void reduceMax(float *address, float val) {
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old || __int_as_float(old) < val);
}

__device__ __forceinline__ static void reduceMax(double *address, double val) {
  unsigned long long *address_as_ull =
      reinterpret_cast<unsigned long long *>(address);
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull, assumed,
        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
  } while (assumed != old || __longlong_as_double(old) < val);
}

// get rid of meaningless warnings when compiling host code
#ifdef __CUDA_ARCH__
__device__ __forceinline__ static void reduceAdd(float *address, float val) {
#if (__CUDA_ARCH__ < 200)
#warning \
    "compute capability lower than 2.x. fall back to use CAS version of atomicAdd for float32"
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(val + __int_as_float(assumed)));
  } while (assumed != old);
#else
  atomicAdd(address, val);
#endif
}

__device__ __forceinline__ static void reduceAdd(double *address, double val) {
#if (__CUDA_ARCH__ < 600)
#warning \
    "compute capability lower than 6.x. fall back to use CAS version of atomicAdd for float64"
  unsigned long long *address_as_ull =
      reinterpret_cast<unsigned long long *>(address);
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
#else
  atomicAdd(address, val);
#endif
}
#endif

template <typename data_t>
__global__ void
feats_reduce_kernel(const data_t *feats, const int32_t *coors_map,
                    data_t *reduced_feats, // shall be 0 at initialization
                    const int num_input, const int num_feats,
                    const int reduce_type) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num_input;
       x += gridDim.x * blockDim.x) {
    int32_t reduce_to = coors_map[x];
    if (reduce_to == -1) continue;

    const data_t *feats_offset = feats + x * num_feats;
    data_t *reduced_feats_offset = reduced_feats + reduce_to * num_feats;
    if (reduce_type == 2) {
      for (int i = 0; i < num_feats; i++) {
        reduceMax(&reduced_feats_offset[i], feats_offset[i]);
      }
    } else {
      for (int i = 0; i < num_feats; i++) {
        reduceAdd(&reduced_feats_offset[i], feats_offset[i]);
      }
    }
  }
}

template <typename data_t>
__global__ void add_reduce_traceback_grad_kernel(
    data_t *grad_feats, const data_t *grad_reduced_feats, const int32_t *coors_map,
    const int32_t *reduce_count, const int num_input, const int num_feats,
    const int reduce_type) {
  
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num_input;
       x += gridDim.x * blockDim.x) {
    

    // std::cout << " #### test!2" << std::endl;
    int32_t reduce_to = coors_map[x];
    if (reduce_to == -1) {
      continue;
    }

    const int input_offset = x * num_feats;
    data_t *grad_feats_offset = grad_feats + input_offset;
    const int reduced_offset = reduce_to * num_feats;
    const data_t *grad_reduced_feats_offset = grad_reduced_feats + reduced_offset;

    if (reduce_type == 0) {
      for (int i = 0; i < num_feats; i++) {
        grad_feats_offset[i] = grad_reduced_feats_offset[i];
      }
    } else if (reduce_type == 1) {
      for (int i = 0; i < num_feats; i++) {
        grad_feats_offset[i] = grad_reduced_feats_offset[i] /
                               static_cast<data_t>(reduce_count[reduce_to]);
      }
    }
  }
}

template <typename data_t>
__global__ void max_reduce_traceback_scatter_idx_kernel(
    const data_t *feats, const data_t *reduced_feats, int32_t *reduce_from,
    const int32_t *coors_map, const int num_input, const int num_feats) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num_input;
       x += gridDim.x * blockDim.x) {
    int32_t reduce_to = coors_map[x];

    const int input_offset = x * num_feats;
    const data_t *feats_offset = feats + input_offset;

    if (reduce_to == -1) {
      continue;
    }

    const int reduced_offset = reduce_to * num_feats;
    const data_t *reduced_feats_offset = reduced_feats + reduced_offset;
    int32_t *reduce_from_offset = reduce_from + reduced_offset;

    for (int i = 0; i < num_feats; i++) {
      if (feats_offset[i] == reduced_feats_offset[i]) {
        atomicMin(&reduce_from_offset[i], static_cast<int32_t>(x));
      }
    }
  }
}

template <typename data_t>
__global__ void max_reduce_scatter_grad_kernel(data_t *grad_feats,
                                               const data_t *grad_reduced_feats,
                                               const int32_t *reduce_from,
                                               const int num_reduced,
                                               const int num_feats) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num_reduced;
       x += gridDim.x * blockDim.x) {
    const int reduced_offset = x * num_feats;
    const int32_t *scatter_to_offset = reduce_from + reduced_offset;
    const data_t *grad_reduced_feats_offset = grad_reduced_feats + reduced_offset;

    for (int i = 0; i < num_feats; i++) {
      grad_feats[scatter_to_offset[i] * num_feats + i] =
          grad_reduced_feats_offset[i];
    }
  }
}

std::vector<paddle::Tensor> dynamic_point_to_voxel_forward_gpu(
    const paddle::Tensor &feats, const paddle::Tensor &coors,
    const int reduce_type) {
  CHECK_INPUT(feats);
  CHECK_INPUT(coors);
  const int num_input = feats.shape()[0]; //.size(0);
  const int num_feats = feats.shape()[1]; //.size(1);

  if (num_input == 0)
    return {feats,  
            coors, 
            paddle::full(coors.shape(), 0, paddle::DataType::INT32, paddle::GPUPlace()),
            paddle::full(coors.shape(), 0, paddle::DataType::INT32, paddle::GPUPlace())};

  paddle::Tensor out_coors;
  paddle::Tensor coors_map;
  paddle::Tensor reduce_count;
  paddle::Tensor _index;
  paddle::Tensor _unused_shape;
  paddle::Tensor fillvalue = paddle::full(coors.shape(), -1, coors.type(), paddle::GPUPlace());
  paddle::Tensor zerotensor = paddle::full(coors.shape(), 0, coors.type(), paddle::GPUPlace());
  paddle::Tensor cond = paddle::experimental::less_than(coors, zerotensor);
    cond = paddle::experimental::any(cond, {-1}, true); //tmp.any(-1, true);
    cond = paddle::tile(cond, {1, coors.shape()[1]});
    auto coors_clean = paddle::where(cond, fillvalue, coors); 
  std::tie(out_coors, _index, coors_map, reduce_count) = paddle::unique(coors_clean, true, true, true, {0});

  out_coors = out_coors.slice(1, out_coors.shape()[0]);
  reduce_count = reduce_count.slice(1, reduce_count.shape()[0]);
  coors_map = paddle::subtract(coors_map, paddle::full(coors_map.shape(), 1, coors_map.type(), paddle::GPUPlace())); //coors_map - 1;


  coors_map = coors_map.cast(paddle::DataType::INT32); 
  reduce_count = reduce_count.cast(paddle::DataType::INT32); 
  auto reduced_feats = paddle::empty({out_coors.shape()[0], num_feats}, feats.type(), paddle::GPUPlace());
  PD_DISPATCH_FLOATING_TYPES(
      feats.type(), "feats_reduce_kernel", ([&] {
    if (reduce_type == 2)
      reduced_feats = paddle::experimental::fill(reduced_feats, -std::numeric_limits<data_t>::infinity());
    else
      reduced_feats = paddle::experimental::fill(reduced_feats, static_cast<data_t>(0));

    dim3 blocks(std::min(DIVUP(num_input, threadsPerBlock), maxGridDim));
    dim3 threads(threadsPerBlock);

    feats_reduce_kernel<data_t><<<blocks, threads, 0, feats.stream()>>>(
        feats.data<data_t>(), coors_map.data<int32_t>(),
        reduced_feats.data<data_t>(), num_input, num_feats, reduce_type);
    if (reduce_type == 1) {

      reduced_feats = paddle::divide(reduced_feats, paddle::reshape(reduce_count, {reduce_count.shape()[0], 1}).cast(reduced_feats.type())); 
    }
  }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in dynamic_point_to_voxel_forward_gpu: %s\n", cudaGetErrorString(err));
  }
  return {reduced_feats, out_coors, coors_map, reduce_count};
}

std::vector<paddle::Tensor> dynamic_point_to_voxel_backward_gpu(//const paddle::Tensor grad_123,
                                      const paddle::Tensor &grad_reduced_feats,
                                         const paddle::Tensor &feats,
                                         const paddle::Tensor &reduced_feats,
                                         const paddle::Tensor &coors_map,
                                         const paddle::Tensor &reduce_count,
                                         const int reduce_type) {
  CHECK_INPUT(grad_reduced_feats);
  CHECK_INPUT(feats);
  CHECK_INPUT(reduced_feats);
  CHECK_INPUT(coors_map);
  CHECK_INPUT(reduce_count);

  const int num_input = feats.shape()[0]; 
  const int num_reduced = reduced_feats.shape()[0]; 
  const int num_feats = feats.shape()[1]; 

  auto grad_feats = paddle::full(feats.shape(), 0.0, grad_reduced_feats.type(), paddle::GPUPlace());

  if (num_input == 0 || num_reduced == 0) return {};
  if (reduce_type == 1 || reduce_type == 0) {
    PD_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.type(), "add_reduce_traceback_grad_kernel",
        ([&] {
          dim3 blocks(std::min(DIVUP(num_input, threadsPerBlock), maxGridDim));
          dim3 threads(threadsPerBlock);
          add_reduce_traceback_grad_kernel<data_t><<<blocks, threads, 0, grad_reduced_feats.stream()>>>(
              grad_feats.data<data_t>(),
              grad_reduced_feats.data<data_t>(),
              coors_map.data<int32_t>(), reduce_count.data<int32_t>(),
              num_input, num_feats, reduce_type);
        }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error in dynamic_point_to_voxel_forward_gpu: %s\n", cudaGetErrorString(err));
    }
  } else {
    auto reduce_from = paddle::full({num_reduced, num_feats}, num_input, paddle::DataType::INT32, paddle::GPUPlace());
    PD_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.type(), //scalar_type(),
        "max_reduce_traceback_scatter_idx_kernel", ([&] {
          dim3 blocks(std::min(DIVUP(num_input, threadsPerBlock), maxGridDim));
          dim3 threads(threadsPerBlock);
          max_reduce_traceback_scatter_idx_kernel<data_t><<<blocks, threads, 0, grad_reduced_feats.stream()>>>(
              feats.data<data_t>(), reduced_feats.data<data_t>(),
              reduce_from.data<int32_t>(), coors_map.data<int32_t>(),
              num_input, num_feats);
        }));
    
    // add to replace ad cuda check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error in dynamic_point_to_voxel_forward_gpu: %s\n", cudaGetErrorString(err));
    }
    PD_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.type(),
        "max_reduce_traceback_scatter_idx_kernel", ([&] {
          dim3 blocks(std::min(DIVUP(num_input, threadsPerBlock), maxGridDim));
          dim3 threads(threadsPerBlock);
          max_reduce_scatter_grad_kernel<data_t><<<blocks, threads>>>(
              grad_feats.data<data_t>(),
              grad_reduced_feats.data<data_t>(),
              reduce_from.data<int32_t>(), num_reduced, num_feats);
        }));

    // add to replace ad cuda check
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error in dynamic_point_to_voxel_forward_gpu: %s\n", cudaGetErrorString(err));
    }
  }
  return {grad_feats};

}
//}  // namespace voxelization
