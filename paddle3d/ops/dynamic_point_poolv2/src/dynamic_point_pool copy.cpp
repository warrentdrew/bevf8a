// Copyright 2023 Baidu Inc. All Rights Reserved.
// @author: Guojun Wang (wangguojun01@baidu.com)
// @file: dynamic_point_pool.cpp
// @brief: dynamic_point_pool

#include <assert.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)


void dynamic_point_pool_launcher(const int boxes_num, const int pts_num,
                                   const int max_pts_num, const float* rois,
                                   const float* pts, const int info_dim,
                                   const float* extra_wlh, float* out_pts_xyz,
                                   float* out_pts_feats, float* out_pts_info,
                                  long* out_roi_idx,
                                   int* global_counter);

void dynamic_point_pool_gpu(at::Tensor rois, at::Tensor pts,
                              at::Tensor extra_wlh, int max_pts_num,
                              at::Tensor out_pts_xyz,
                              at::Tensor out_pts_feats, at::Tensor out_pts_info,
                              at::Tensor out_roi_idx,
                              at::Tensor global_count) {
  // params rois: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coordinate
  // params pts: (npoints, 3) [x, y, z] in LiDAR coordinate

  CHECK_INPUT(rois);
  CHECK_INPUT(pts);
   CHECK_INPUT(out_pts_xyz);
  CHECK_INPUT(out_pts_feats);
  CHECK_INPUT(out_roi_idx);
  CHECK_INPUT(out_pts_info);
  CHECK_INPUT(global_count);

  int boxes_num = rois.size(0);
  int pts_num = pts.size(0);
  int info_dim = out_pts_info.size(1);
  assert(info_dim == 10);

  const float* rois_data = rois.data_ptr<float>();
  const float* pts_data = pts.data_ptr<float>();

  float* out_pts_xyz_data = out_pts_xyz.data_ptr<float>();
  float* out_pts_feats_data = out_pts_feats.data_ptr<float>();
  float* out_pts_info_data = out_pts_info.data_ptr<float>();
  long* out_roi_idx_data = out_roi_idx.data_ptr<long>();
  int* global_count_data = global_count.data_ptr<int>();
  const float* extra_wlh_data = extra_wlh.data_ptr<float>();

  dynamic_point_pool_launcher(boxes_num, pts_num, max_pts_num, rois_data,
                                pts_data, info_dim, extra_wlh_data,
                                out_pts_xyz_data, out_pts_feats_data,
                                out_pts_info_data, 
                                out_roi_idx_data, global_count_data);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dynamic_point_pool_gpu,
        "dynamic_point_pool_gpu forward (CUDA)");
}