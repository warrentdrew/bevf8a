#include <paddle/extension.h>
#include <vector>

#define CHECK_CUDA(x)                                                          \
  PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")


#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x)                                                                
  // CHECK_CUDA(x);                                                               
  // CHECK_CONTIGUOUS(x)


void pointsiouLauncherV2(int pts_num, int boxes_num1, int boxes_num2,
                       const float *pts, const float *boxes3d1,
                       const float *boxes3d2, float *iou3d);

std::vector<paddle::Tensor> point_iou_gpuv2(const paddle::Tensor& pts, const paddle::Tensor& boxes3d1, 
                                            const paddle::Tensor& boxes3d2) {
  // params pts (N, 3)
  // params boxes3d1: (M, 7)
  // params boxes3d2: (N, 7)
  // params iou3d: (M, N)

  CHECK_INPUT(pts);
  CHECK_INPUT(boxes3d1);
  CHECK_INPUT(boxes3d2);
  // CHECK_INPUT(iou3d);

  int pts_num = pts.shape()[0];
  int boxes_num1 = boxes3d1.shape()[0];
  int boxes_num2 = boxes3d2.shape()[0];

  auto iou3d = paddle::full({boxes_num1, boxes_num2}, 0, boxes3d1.dtype(), paddle::GPUPlace());

  const float *pts_data = pts.data<float>();
  const float *boxes3d_data1 = boxes3d1.data<float>();
  const float *boxes3d_data2 = boxes3d2.data<float>();
  float *iou3d_data = iou3d.data<float>();

  pointsiouLauncherV2(pts_num, boxes_num1, boxes_num2, pts_data, boxes3d_data1,
                    boxes3d_data2, iou3d_data);
  return {iou3d};
}

PD_BUILD_OP(point_iou_gpuv2)
    .Inputs({"pts", "boxes3d1", "boxes3d2"})
    .Outputs({"iou3d"}) 
    .SetKernelFn(PD_KERNEL(point_iou_gpuv2));

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("pts_in_boxes3d_cpu", &pts_in_boxes3d_cpu, "pts_in_boxes3d_cpu");
//   m.def("roipool3d_cpu", &roipool3d_cpu, "roipool3d_cpu");
//   m.def("forward", &roipool3d_gpu, "roipool3d forward (CUDA)");
//   m.def("point_iou_gpu", &point_iou_gpu, "Point IoU for GPU (CUDA)");
//   m.def("point_iou_gpuv2", &point_iou_gpuv2, "Point IoU for GPU (CUDA)");
//   m.def("forward_slow", &roipool3d_gpu_slow, "roipool3d forward (CUDA)");
// }
