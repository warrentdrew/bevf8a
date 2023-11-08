// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "paddle/extension.h"

#define CHECK_INPUT_CUDA(x) \
  PD_CHECK(x.is_gpu() || x.is_gpu_pinned(), #x " must be a GPU Tensor.")

void mapfeatureLauncher(const cudaStream_t& stream, const int num,
                        const int point_dim, const float* points_data,
                        const int* points2grid_data, float* max_height_data,
                        float* mean_height_data, float* mean_intensity_data,
                        float* count_data);
void TopIntensityLauncher(const cudaStream_t& stream, const int num,
                          const int point_dim, const float* points_data,
                          const int* points2grid_data,
                          float* top_intensity_data, float* max_height_data);
void ComputeLogOddsGPULauncher(
    const cudaStream_t& stream, const int num, const int point_dim,
    const float voxel_size_x, const float voxel_size_y,
    const float voxel_size_z, const float pxmin, const float pymin,
    const float pzmin, const int grid_size_x, const int grid_size_y,
    const int grid_size_z, const float offset_origin_x,
    const float offset_origin_y, const float offset_origin_z, float lo_occupied,
    float lo_free, const float* points_data, float* logodds_data);

std::vector<paddle::Tensor> cnnseg_feature_generator_gpu(
    const paddle::Tensor& points, const paddle::Tensor& points2grid,
    const std::vector<int>& grid_size, const int cnnseg_feature_dim) {
  CHECK_INPUT_CUDA(points);
  CHECK_INPUT_CUDA(points2grid);  

  const int grid_size_x = grid_size[0];
  const int grid_size_y = grid_size[1];
  const int grid_size_z = grid_size[2];

  auto cnnseg_feature =
      paddle::full({cnnseg_feature_dim, grid_size_y, grid_size_x}, 0,
                   points.type(), paddle::GPUPlace());
  cnnseg_feature = paddle::scatter(cnnseg_feature, paddle::full({1}, 0, points2grid.type(), paddle::GPUPlace()), paddle::full({1, grid_size_y, grid_size_x}, -5.0, points.type(), paddle::GPUPlace()));

  const float* points_data = points.data<float>();
  const int* points2grid_data = points2grid.data<int>();
  float* cnnseg_feature_data = cnnseg_feature.data<float>();
  const int unit_offset = grid_size_x * grid_size_y;
  float* mean_height_data = cnnseg_feature_data + unit_offset * 0;
  float* max_height_data = cnnseg_feature_data + unit_offset * 1;
  float* count_data = cnnseg_feature_data + unit_offset * 2;
  float* top_intensity_data = cnnseg_feature_data + unit_offset * 3;
  float* mean_intensity_data = cnnseg_feature_data + unit_offset * 4;
  float* nonempty_data = cnnseg_feature_data + unit_offset * 5;
  const int num_point = points.shape()[0];
  const int point_dim = points.shape()[1];
  cudaStream_t stream = points.stream();
  mapfeatureLauncher(stream, num_point, point_dim, points_data,
                     points2grid_data, max_height_data, mean_height_data,
                     mean_intensity_data, count_data);

  TopIntensityLauncher(stream, num_point, point_dim, points_data,
                       points2grid_data, top_intensity_data, max_height_data);

  return {cnnseg_feature};
}

std::vector<paddle::Tensor> visibility_feature_gpu(
    const paddle::Tensor& points, const std::vector<float>& vis_voxel_size,
    const std::vector<float>& vis_range, const std::vector<int>& vis_grid_size,
    const std::vector<float>& vis_offset_origins, const float lo_occupied,
    const float lo_free) {
  CHECK_INPUT_CUDA(points);

  const float voxel_size_x = vis_voxel_size[0];
  const float voxel_size_y = vis_voxel_size[1];
  const float voxel_size_z = vis_voxel_size[2];
  const float pc_start_x = vis_range[0];
  const float pc_start_y = vis_range[1];
  const float pc_start_z = vis_range[2];
  const int grid_size_x = vis_grid_size[0];
  const int grid_size_y = vis_grid_size[1];
  const int grid_size_z = vis_grid_size[2];
  const float offset_origin_x = vis_offset_origins[0];
  const float offset_origin_y = vis_offset_origins[1];
  const float offset_origin_z = vis_offset_origins[2];

  auto logodds_feature =
      paddle::full({vis_grid_size[2], vis_grid_size[1], vis_grid_size[0]}, 0,
                   points.type(), paddle::GPUPlace());
  const float* points_data = points.data<float>();
  float* logodds_feature_data = logodds_feature.data<float>();
  int num_point = points.shape()[0];
  int point_dim = points.shape()[1];
  cudaStream_t stream = points.stream();
  ComputeLogOddsGPULauncher(
      stream, num_point, point_dim, voxel_size_x, voxel_size_y, voxel_size_z,
      pc_start_x, pc_start_y, pc_start_z, grid_size_x, grid_size_y, grid_size_z,
      offset_origin_x, offset_origin_y, offset_origin_z, lo_occupied, lo_free,
      points_data, logodds_feature_data);
  return {logodds_feature};
}

std::vector<std::vector<int64_t>> CnnSegFeatGenInferShape(
    std::vector<int64_t> points_shape, std::vector<int64_t> points2grid_shape,
    const std::vector<int>& grid_size, const int cnnseg_feature_dim) {
  return {{cnnseg_feature_dim, grid_size[1], grid_size[0]}};
}

std::vector<paddle::DataType> CnnSegFeatGenInferDtype(
    paddle::DataType points_dtype, paddle::DataType points2grid_dtype) {
  return {points_dtype};
}

std::vector<std::vector<int64_t>> VisFeatInferShape(
    std::vector<int64_t> points_shape, const std::vector<float>& vis_voxel_size,
    const std::vector<float>& vis_range, const std::vector<int>& vis_grid_size,
    const std::vector<float>& vis_offset_origins, const float lo_occupied,
    const float lo_free) {
  return {{vis_grid_size[2], vis_grid_size[1], vis_grid_size[0]}};
}

std::vector<paddle::DataType> VisFeatInferDtype(paddle::DataType points_dtype) {
  return {points_dtype};
}

PD_BUILD_OP(cnnseg_feature_generator_gpu)
    .Inputs({"points", "points2grid"})
    .Outputs({"cnnseg_feature"})
    .SetKernelFn(PD_KERNEL(cnnseg_feature_generator_gpu))
    .Attrs({"grid_size: std::vector<int>", "cnnseg_feature_dim: int"})
    .SetInferShapeFn(PD_INFER_SHAPE(CnnSegFeatGenInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(CnnSegFeatGenInferDtype));

PD_BUILD_OP(visibility_feature_gpu)
    .Inputs({"points"})
    .Outputs({"logodds_feature"})
    .SetKernelFn(PD_KERNEL(visibility_feature_gpu))
    .Attrs({"vis_voxel_size: std::vector<float>",
            "vis_range: std::vector<float>", "vis_grid_size: std::vector<int>",
            "vis_offset_origins: std::vector<float>", "lo_occupied: float",
            "lo_free: float"})
    .SetInferShapeFn(PD_INFER_SHAPE(VisFeatInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(VisFeatInferDtype));