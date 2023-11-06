#include <paddle/extension.h>
#include <stdio.h>
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define CHECK_INPUT(x) PD_CHECK(x.is_gpu() || x.is_gpu_pinned(), #x " must be a GPU Tensor.")

// namespace {
int const threadsPerBlock = sizeof(unsigned long long) * 8;
// }

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename data_t, typename T_int>
__global__ void dynamic_voxelize_kernel(
    const data_t* points, T_int* coors, const float voxel_x, const float voxel_y,
    const float voxel_z, const float coors_x_min, const float coors_y_min,
    const float coors_z_min, const float coors_x_max, const float coors_y_max,
    const float coors_z_max, const int grid_x, const int grid_y,
    const int grid_z, const int num_points, const int num_features,
    const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    // To save some computation
    auto points_offset = points + index * num_features;
    auto coors_offset = coors + index * NDim;
    int c_x = floor((points_offset[0] - coors_x_min) / voxel_x);
    if (c_x < 0 || c_x >= grid_x) {
      coors_offset[0] = -1;
      return;
    }

    int c_y = floor((points_offset[1] - coors_y_min) / voxel_y);
    if (c_y < 0 || c_y >= grid_y) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      return;
    }

    int c_z = floor((points_offset[2] - coors_z_min) / voxel_z);
    if (c_z < 0 || c_z >= grid_z) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      coors_offset[2] = -1;
    } else {
      coors_offset[0] = c_z;
      coors_offset[1] = c_y;
      coors_offset[2] = c_x;
    }
  }
}

// namespace voxelization {
std::vector<paddle::Tensor> dynamic_voxelize_gpu(const paddle::Tensor& points, // , paddle::Tensor& coors,
                          const std::vector<float> &voxel_size,
                          const std::vector<float> &coors_range,
                          const int NDim = 3) {
  // current version tooks about 0.04s for one frame on cpu
  // check device
  CHECK_INPUT(points);
  auto coors = paddle::full({points.shape()[0], 3}, 0.0, paddle::DataType::INT32, paddle::GPUPlace());
  const int num_points = points.shape()[0]; //.size(0);
  const int num_features = points.shape()[1]; //.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  const int col_blocks = DIVUP(num_points, threadsPerBlock); //at::cuda::ATenCeilDiv(num_points, threadsPerBlock);
  dim3 blocks(col_blocks);
  dim3 threads(threadsPerBlock);
  PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(points.type(), "dynamic_voxelize_kernel", [&] {
    dynamic_voxelize_kernel<data_t, int><<<blocks, threads, 0, points.stream()>>>(
        points.data<data_t>(),
        coors.data<int>(), voxel_x, voxel_y, voxel_z,
        coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
        coors_z_max, grid_x, grid_y, grid_z, num_points, num_features, NDim);
  });
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
        printf("error in dynamic_point_to_voxel_forward_gpu: %s\n", cudaGetErrorString(err));
    }

  return {coors};
}

// }  // namespace voxelization
