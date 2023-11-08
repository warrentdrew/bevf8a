// Copyright 2023 Baidu Inc. All Rights Reserved.
// @author: Guojun Wang (wangguojun01@baidu.com)
// @file: roi_align_rotated.cpp
// @brief: roi_align_rotated
#include <assert.h>
#include <paddle/extension.h>

#include <vector>
// using namespace at;

std::vector<paddle::Tensor> ROIAlignRotatedForwardCUDAKernelLauncher(
    const paddle::Tensor& input, const paddle::Tensor& rois, const float spatial_scale,
    const float pc_range_x, const float pc_range_y,
    const float voxel_x, const float voxel_y, const int channels,
    const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width);

std::vector<paddle::Tensor> ROIAlignRotatedBackwardCUDAKernelLauncher(
    const paddle::Tensor& top_grad, const paddle::Tensor& rois, const float spatial_scale,
    const float pc_range_x, const float pc_range_y,
    const float voxel_x, const float voxel_y, const int channels,
    const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, int batch_size);

std::vector<paddle::Tensor> roi_align_rotated_forward(const paddle::Tensor& input, const paddle::Tensor& rois, 
                              //  Tensor output,
                               int aligned_height, int aligned_width,
                               float spatial_scale, float pc_range_x,
                               float pc_range_y, float voxel_size_x, float voxel_size_y) {
  // Number of ROIs
  int num_rois = rois.shape()[0];
  int size_rois = rois.shape()[1];

  if (size_rois != 8) {
    printf("wrong roi size");
  }

  int num_channels = input.shape()[1];
  int data_height = input.shape()[2];
  int data_width = input.shape()[3];
  auto output =  ROIAlignRotatedForwardCUDAKernelLauncher(
      input, rois, spatial_scale, pc_range_x, pc_range_y, voxel_size_x, voxel_size_y,
      num_channels, data_height, data_width, num_rois, aligned_height,
      aligned_width);
  return output;
}

std::vector<paddle::Tensor> roi_align_rotated_backward(const paddle::Tensor& top_grad, const paddle::Tensor& rois,
                                int aligned_height,
                                int aligned_width, float spatial_scale,
                                float pc_range_x, float pc_range_y,
                                float voxel_size_x, float voxel_size_y, int batch_size, int channels, int height, int width) {
  // Number of ROIs
  int num_rois = rois.shape()[0];
  int size_rois = rois.shape()[1];
  if (size_rois != 8) {
    printf("wrong roi size");
  }

  // int num_channels = bottom_grad.shape()[1];
  // int data_height = bottom_grad.shape()[2];
  // int data_width = bottom_grad.shape()[3];
  auto output = ROIAlignRotatedBackwardCUDAKernelLauncher(
      top_grad, rois, spatial_scale, pc_range_x, pc_range_y, voxel_size_x, voxel_size_y,
      channels, height, width, num_rois, aligned_height,
      aligned_width, batch_size);
  return {output};
}

PD_BUILD_OP(roi_align_rotated_forward)
    .Inputs({"input", "rois"})
    .Outputs({"output"}) 
    .SetKernelFn(PD_KERNEL(roi_align_rotated_forward))
    .Attrs({"aligned_height: int", 
            "aligned_width: int",
            "spatial_scale: float", 
            "pc_range_x: float",
            "pc_range_y: float",
            "voxel_size_x: float",
            "voxel_size_y: float"
            });

PD_BUILD_OP(roi_align_rotated_backward)
    .Inputs({"top_grad", "rois"})
    .Outputs({"output"}) 
    .SetKernelFn(PD_KERNEL(roi_align_rotated_backward))
    .Attrs({"aligned_height: int", 
            "aligned_width: int",
            "spatial_scale: float", 
            "pc_range_x: float",
            "pc_range_y: float",
            "voxel_size_x: float",
            "voxel_size_y: float",
            "batch_size: int",
            "channels: int",
            "height: int",
            "width: int",});
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("roi_align_rotated_forward", &roi_align_rotated_forward,
//         "roi_align_rotated forward", py::arg("input"), py::arg("rois"),
//         py::arg("output"), py::arg("pooled_height"), py::arg("pooled_width"),
//         py::arg("spatial_scale"), py::arg("pc_range_x"), py::arg("pc_range_y"),
//         py::arg("voxel_size_x"), py::arg("voxel_size_y"));
//   m.def("roi_align_rotated_backward", &roi_align_rotated_backward,
//         "roi_align_rotated backward", py::arg("rois"), py::arg("grad_input"),
//         py::arg("grad_output"), py::arg("pooled_height"),
//         py::arg("pooled_width"), py::arg("spatial_scale"),
//         py::arg("pc_range_x"), py::arg("pc_range_y"), py::arg("voxel_size_x"),
//         py::arg("voxel_size_y"));
// }

// static auto registry = torch::RegisterOperators("custom::RoIAlignRotated",
//                                                 &roi_align_rotated_forward);