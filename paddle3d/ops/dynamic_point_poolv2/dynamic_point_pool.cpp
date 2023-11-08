// Copyright 2023 Baidu Inc. All Rights Reserved.
// @author: Guojun Wang (wangguojun01@baidu.com)
// @file: dynamic_point_pool.cpp
// @brief: dynamic_point_pool

#include <assert.h>
// #include <torch/extension.h>
// #include <torch/serialize/tensor.h>
#include <paddle/extension.h>

#include <vector>

// #define CHECK_CUDA(x)                                                          \
//   TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
// #define CHECK_CONTIGUOUS(x)                                                    \
//   TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
// #define CHECK_INPUT(x)                                                         \
//   CHECK_CUDA(x);                                                               \
//   CHECK_CONTIGUOUS(x)


void dynamic_point_pool_launcher(const int boxes_num, const int pts_num,
                                   const int max_pts_num, const float* rois,
                                   const float* pts, const int info_dim,
                                   const float* extra_wlh, float* out_pts_xyz,
                                   float* out_pts_feats, float* out_pts_info,
                                   long* out_roi_idx,
                                   int* global_counter);

std::vector<paddle::Tensor> dynamic_point_pool_gpu(
                              const paddle::Tensor& rois, 
                              const paddle::Tensor& pts,
                              const paddle::Tensor& extra_wlh,
                              // paddle::Tensor out_pts_xyz,
                              // paddle::Tensor out_pts_feats, 
                              // paddle::Tensor out_pts_info,
                              // paddle::Tensor out_roi_idx,
                              // paddle::Tensor global_count,
                              const int max_pts_num, 
                              const int max_all_pts) {

  // auto cnnseg_feature = paddle::full({cnnseg_feature_dim, grid_size_y, grid_size_x}, 0.0, pts.type(), paddle::GPUPlace());
  auto out_pts_xyz = paddle::full({max_all_pts, 3}, 0.0, pts.type(), paddle::GPUPlace());  
  auto out_pts_feats = paddle::full({max_all_pts, 1}, 0.0, pts.type(), paddle::GPUPlace());  
  auto out_roi_idx = paddle::full({max_all_pts}, -1, paddle::DataType::INT64, paddle::GPUPlace());  
  auto out_pts_info = paddle::full({max_all_pts, 10}, 0.0, pts.type(), paddle::GPUPlace()); 
  auto global_count = paddle::full({1}, 0, paddle::DataType::INT32, paddle::GPUPlace()); 
  // out_pts_xyz = pts.new_zeros(max_all_pts, 3)  # 采样的每个点对应的原始点云的索引
  // out_pts_feats = pts.new_zeros(max_all_pts, 1)  # 采样的每个点对应的原始点云的索引
  // out_roi_idx = -1 * pts.new_ones(max_all_pts, dtype=torch.long)  # 采样的每个点对应的roi的索引
  // out_pts_info = pts.new_zeros(max_all_pts, 10)  # 采样的点特征
  // params rois: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coordinate
  // params pts: (npoints, 3) [x, y, z] in LiDAR coordinate

  // CHECK_INPUT(rois);
  // CHECK_INPUT(pts);
  //  CHECK_INPUT(out_pts_xyz);
  // CHECK_INPUT(out_pts_feats);
  // CHECK_INPUT(out_roi_idx);
  // CHECK_INPUT(out_pts_info);
  // CHECK_INPUT(global_count);

  int boxes_num = rois.shape()[0];
  int pts_num = pts.shape()[0];
  int info_dim = out_pts_info.shape()[1];
  assert(info_dim == 10);

  const float* rois_data = rois.data<float>();
  const float* pts_data = pts.data<float>();

  float* out_pts_xyz_data = out_pts_xyz.data<float>();
  float* out_pts_feats_data = out_pts_feats.data<float>();
  float* out_pts_info_data = out_pts_info.data<float>();
  long* out_roi_idx_data = out_roi_idx.data<long>();
  int* global_count_data = global_count.data<int>();
  const float* extra_wlh_data = extra_wlh.data<float>();

  dynamic_point_pool_launcher(boxes_num, pts_num, max_pts_num, rois_data,
                                pts_data, info_dim, extra_wlh_data,
                                out_pts_xyz_data, out_pts_feats_data,
                                out_pts_info_data, 
                                out_roi_idx_data, global_count_data);
  return {out_pts_xyz, out_pts_feats, out_pts_info, out_roi_idx, global_count};
}
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("forward", &dynamic_point_pool_gpu,
//         "dynamic_point_pool_gpu forward (CUDA)");
// }

PD_BUILD_OP(dynamic_point_pool_gpu)
    .Inputs({"rois", "pts", "extra_wlh"})
    .Outputs({"out_pts_xyz", "out_pts_feats", "out_pts_info", "out_roi_idx", "global_count"}) 
    .SetKernelFn(PD_KERNEL(dynamic_point_pool_gpu))
    .Attrs({"max_pts_num: int", "max_all_pts: int"});
    // .SetInferShapeFn(PD_INFER_SHAPE(DynamicPointToVoxelInferShape))
    // .SetInferDtypeFn(PD_INFER_DTYPE(DynamicPointToVoxelInferDtype));