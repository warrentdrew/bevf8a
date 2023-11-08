// Copyright 2023 Baidu Inc. All Rights Reserved.
// @author: Guojun Wang (wangguojun01@baidu.com)
// @file: scatter_max.cpp
// @brief: scatter_max

#include <assert.h>
#include <paddle/extension.h>
// #include <torch/extension.h>
// #include <torch/script.h>
// #include <torch/serialize/tensor.h>

#include <vector>

#define CHECK_CUDA(x)                                                          \
  PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")
  // PD_CHECK(x.device().is_gpu(), #x, " must be a CUDAtensor ")
// #define CHECK_CONTIGUOUS(x)                                                    \
//   TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
// #define CHECK_INPUT(x)                                                         \
//   CHECK_CUDA(x);                                                               \
//   CHECK_CONTIGUOUS(x)

std::vector<paddle::Tensor> scatter_max_launcher(const paddle::Tensor& src, const paddle::Tensor& index, 
                          int N_in, int C_in, int N_out);

std::vector<paddle::Tensor> scatter_max_gpu(const paddle::Tensor& src, const paddle::Tensor& index, int batch_rois_first_dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  // CHECK_CUDA(out);
  // CHECK_CUDA(arg_out);

  // CHECK_INPUT(src);
  // CHECK_INPUT(index);
  // CHECK_INPUT(out);
  // CHECK_INPUT(arg_out);
  int N_in = src.shape()[0];
  int C_in = src.shape()[1];
  const int N_out = batch_rois_first_dim;

  // auto out = paddle::full({batch_rois_first_dim, C_in}, 0.0, src.type(), paddle::GPUPlace());
  // auto arg_out = paddle::full({batch_rois_first_dim, C_in}, index.shape()[0], index.type(), paddle::GPUPlace());

  assert(out.shape()[0] == arg_out.shape()[0]);
  auto result = scatter_max_launcher(src, index, N_in, C_in, N_out);
  return result;
}

PD_BUILD_OP(scatter_max_gpu)
    .Inputs({"src", "index"})
    .Outputs({"out", "arg_out"}) 
    .SetKernelFn(PD_KERNEL(scatter_max_gpu))
    .Attrs({"batch_rois_first_dim: int"});

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("scatter_max_gpu", &scatter_max_gpu,
//         "dynamic_scatter_max_gpu forward (CUDA)");
// }

// static auto registry =
//     torch::RegisterOperators("custom::scatter_max", &scatter_max_gpu);