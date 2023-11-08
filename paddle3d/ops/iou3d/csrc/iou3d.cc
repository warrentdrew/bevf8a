#include <cuda.h>
#include <cuda_runtime_api.h>
#include <paddle/extension.h>
#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

const int THREADS_PER_BLOCK_NMS = sizeof(int64_t) * 8;

void NmsLauncher(const cudaStream_t &stream, const float *boxes, int64_t *mask,
                  int boxes_num, float nms_overlap_thresh);
void NmsNormalLauncher(const cudaStream_t &stream, const float *boxes,
                         int64_t *mask, int boxes_num,
                         float nms_overlap_thresh);
void AnchorsMaskLauncher(const cudaStream_t &stream, const int anchors_num,
                           const int *anchors, const int w,
                           const bool *voxel_mask, bool *anchors_mask);
void BoxesToParsingLauncher(const cudaStream_t &stream, const int box_num,
                               const float *boxes_data, const int w,
                               int *parsing_map_data);
void BoxesToParsingWithWeightLauncher(const cudaStream_t &stream,
                                           const int box_num,
                                           const float *boxes_data, const int w,
                                           float *parsing_map_data);
void ParsingToBoxesConfLauncher(const cudaStream_t &stream,
                                    const int box_num, const float *boxes_data,
                                    const int w, const float *conf_map_data,
                                    float *confs_data);

std::vector<paddle::Tensor>
anchors_mask_of_valid_voxels(const paddle::Tensor &anchors,
                             const paddle::Tensor &voxel_mask) {
  // params anchors: (N, 4) [x1, y1, x2, y2] int
  // params voxel_mask: (W*L) bool
  // params anchors_mask: (N) bool

  CHECK_INPUT(anchors);
  CHECK_INPUT(voxel_mask);

  auto anchors_mask = paddle::full({anchors.shape()[0]}, 0.0, paddle::DataType::BOOL, paddle::GPUPlace());
  int anchor_num = anchors.shape()[0];
  int w = voxel_mask.shape()[0]; // nx
  const int *anchors_data = anchors.data<int>();
  const bool *voxel_mask_data = voxel_mask.data<bool>();
  bool *anchors_mask_data = anchors_mask.data<bool>(); 
  AnchorsMaskLauncher(anchors.stream(), anchor_num, anchors_data, w,
                        voxel_mask_data, anchors_mask_data);
  return {anchors_mask};
}

// new version of boxes_to_parsing
std::vector<paddle::Tensor> boxes_to_parsing(const paddle::Tensor &boxes,
                                             std::vector<int> grid_size) {
  // params boxes: (N, 4, 2) [x1, y1, x2, y2] int
  // params parsing_map: (W, L) bool
  // struct timeval t0, t1, t2;
  CHECK_INPUT(boxes);

  auto parsing_map = paddle::full({grid_size[1], grid_size[0]}, 0, paddle::DataType::INT32, paddle::GPUPlace());

  int box_num = boxes.shape()[0];
  int w = parsing_map.shape()[0]; // nx
  const float *boxes_data = boxes.data<float>();
  int *parsing_map_data = parsing_map.data<int>();

  BoxesToParsingLauncher(boxes.stream(), box_num, boxes_data, w,
                            parsing_map_data);
  return {parsing_map};
}

std::vector<paddle::Tensor>
parsing_to_boxes_confidence(const paddle::Tensor &boxes,
                            const paddle::Tensor &confidence_map) {
  // params boxes: (N, 4, 2) [x1, y1, x2, y2] float
  // params parsing_map: (W, L) bool
  // params confidences: N

  CHECK_INPUT(boxes);
  CHECK_INPUT(confidence_map);

  int box_num = boxes.shape()[0];
  int w = confidence_map.shape()[0]; // nx
  const float *boxes_data = boxes.data<float>();
  const float *confidence_map_data = confidence_map.data<float>();
  auto confidences = paddle::full({box_num}, 0.0, paddle::DataType::FLOAT32, paddle::GPUPlace()); 
  float *confidences_data = confidences.data<float>();

  ParsingToBoxesConfLauncher(boxes.stream(), box_num, boxes_data, w,
                                 confidence_map_data, confidences_data);

  return {confidences};
}

/*new version*/
std::vector<paddle::Tensor> nms_normal_gpu(const paddle::Tensor &boxes,
                                           float nms_overlap_thresh) {
  // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
  // params keep: (N)

  auto keep = paddle::empty({boxes.shape()[0]}, paddle::DataType::INT32,
                            paddle::CPUPlace());
  auto num_to_keep_tensor =
      paddle::empty({1}, paddle::DataType::INT32, paddle::CPUPlace());
  int *num_to_keep_data = num_to_keep_tensor.data<int>();

  int boxes_num = boxes.shape()[0];
  const float *boxes_data = boxes.data<float>();
  int *keep_data = keep.data<int>();

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  auto mask = paddle::empty({boxes_num * col_blocks}, paddle::DataType::INT64,
                            paddle::GPUPlace());
  int64_t *mask_data = mask.data<int64_t>();
  NmsNormalLauncher(boxes.stream(), boxes_data, mask_data, boxes_num,
                      nms_overlap_thresh);

  const paddle::Tensor mask_cpu_tensor = mask.copy_to(paddle::CPUPlace(), true);
  const int64_t *mask_cpu = mask_cpu_tensor.data<int64_t>();

  int64_t remv_cpu[col_blocks];
  memset(remv_cpu, 0, col_blocks * sizeof(int64_t));

  int num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      const int64_t *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }

  num_to_keep_data[0] = num_to_keep;
  if (cudaSuccess != cudaGetLastError()) {
    printf("Error!\n");
  }
  return {keep, num_to_keep_tensor};
}

std::vector<paddle::DataType> NMSInferDtype(paddle::DataType boxes_dtype) {
  return {paddle::DataType::INT64, paddle::DataType::INT64};
}

std::vector<paddle::DataType>
NMSNORMALInferDtype(paddle::DataType boxes_dtype) {
  return {paddle::DataType::INT64, paddle::DataType::INT64};
}

std::vector<std::vector<int64_t>>
NMSNORMALInferShape(std::vector<int64_t> boxes_shape) {
  return {{boxes_shape[0]}, {1}};
}

std::vector<paddle::DataType> ANCHORInferDtype(paddle::DataType anchor_dtype,
                                               paddle::DataType mask_dtype) {
  return {paddle::DataType::BOOL};
}

std::vector<std::vector<int64_t>>
ANCHORInferShape(std::vector<int64_t> anchor_shape,
                 std::vector<int64_t> mask_shape) {
  return {{anchor_shape[0]}};
}

std::vector<paddle::DataType>
BOXESPARSINGInferDtype(paddle::DataType boxes_dtype) {
  return {paddle::DataType::INT32};
}

std::vector<paddle::DataType>
BOXESPARSINGWEIGHTInferDtype(paddle::DataType boxes_dtype) {
  return {paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>>
ParsingToBoxesConfidenceInferShape(std::vector<int64_t> boxes_shape,
                                   std::vector<int64_t> confidence_map_shape) {
  return {{boxes_shape[0]}};
}

std::vector<paddle::DataType>
ParsingToBoxesConfidenceInferDtype(paddle::DataType boxes_dtype,
                                   paddle::DataType confidence_map_dtype) {
  return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(nms_normal_gpu)
    .Inputs({"boxes"})
    .Outputs({"keep", "num_to_keep"})
    .Attrs({"nms_overlap_thresh: float"})
    .SetInferShapeFn(PD_INFER_SHAPE(NMSNORMALInferShape))
    .SetKernelFn(PD_KERNEL(nms_normal_gpu))
    .SetInferDtypeFn(PD_INFER_DTYPE(NMSNORMALInferDtype));

PD_BUILD_OP(anchors_mask_of_valid_voxels)
    .Inputs({"anchors", "voxel_mask"})
    .Outputs({"anchors_mask"})
    .SetInferShapeFn(PD_INFER_SHAPE(ANCHORInferShape))
    .SetKernelFn(PD_KERNEL(anchors_mask_of_valid_voxels))
    .SetInferDtypeFn(PD_INFER_DTYPE(ANCHORInferDtype));

PD_BUILD_OP(boxes_to_parsing)
    .Inputs({"boxes"})
    .Outputs({"parsing_map"})
    .Attrs({"grid_size: std::vector<int>"})
    .SetKernelFn(PD_KERNEL(boxes_to_parsing))
    .SetInferDtypeFn(PD_INFER_DTYPE(BOXESPARSINGInferDtype));

PD_BUILD_OP(parsing_to_boxes_confidence)
    .Inputs({"boxes", "confidence_map"})
    .Outputs({"confidence"})
    .SetKernelFn(PD_KERNEL(parsing_to_boxes_confidence))
    .SetInferShapeFn(PD_INFER_SHAPE(ParsingToBoxesConfidenceInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ParsingToBoxesConfidenceInferDtype));
