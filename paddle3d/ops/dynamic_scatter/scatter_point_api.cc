#include <paddle/extension.h>

// typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

std::vector<paddle::Tensor> dynamic_point_to_voxel_forward_gpu(
    const paddle::Tensor &feats, const paddle::Tensor &coors,
    const int reduce_type);

std::vector<paddle::Tensor> dynamic_point_to_voxel_backward_gpu( 
                                         const paddle::Tensor &grad_reduced_feats,
                                         const paddle::Tensor &feats,
                                         const paddle::Tensor &reduced_feats,
                                         const paddle::Tensor &coors_map,
                                         const paddle::Tensor &reduce_count,
                                         const int reduce_type);


inline int convert_reduce_type(const std::string &reduce_type) {
  if (reduce_type == "max")
    return 2;
  else if (reduce_type == "sum")
    return 0;
  else if (reduce_type == "mean")
    return 1;
  else 
    PD_THROW("do not support reduce type ", reduce_type);
  return 0;
}

std::vector<paddle::Tensor> dynamic_point_to_voxel_fwd(const paddle::Tensor &feats,
                                                                 const paddle::Tensor &coors,
                                                                 const std::string &reduce_type) {

// #ifdef WITH_CUDA
    return dynamic_point_to_voxel_forward_gpu(feats, coors, convert_reduce_type(reduce_type));

}

std::vector<paddle::Tensor> dynamic_point_to_voxel_bkwd(//const paddle::Tensor &grad_feats,
                                            const paddle::Tensor &grad_reduced_feats,
                                            const paddle::Tensor &feats,
                                            const paddle::Tensor &reduced_feats,
                                            const paddle::Tensor &coors_idx,
                                            const paddle::Tensor &reduce_count,
                                            const std::string &reduce_type) {
    return dynamic_point_to_voxel_backward_gpu(
         grad_reduced_feats, feats, reduced_feats, coors_idx, reduce_count,
        convert_reduce_type(reduce_type));

}

std::vector<paddle::DataType> DynamicPointToVoxelInferDtype(paddle::DataType feats_dtype, paddle::DataType coors_dtype) {
  return {feats_dtype, coors_dtype, paddle::DataType::INT32, paddle::DataType::INT32};
}

std::vector<std::vector<int64_t>> DynamicPointToVoxelInferShape(
    std::vector<int64_t> feats_shape, std::vector<int64_t> coors_shape) {
  return {{-1, feats_shape[1]}, {-1, coors_shape[1]}, {feats_shape[0]}, {-1}};
}

PD_BUILD_OP(dynamic_point_to_voxel_fwd)
    .Inputs({"feats", "coors"})
    .Outputs({"reduced_feats", "out_coors", "coors_map", "reduce_count"}) 
    .SetKernelFn(PD_KERNEL(dynamic_point_to_voxel_fwd))
    .Attrs({"reduce_type: std::string"})
    .SetInferShapeFn(PD_INFER_SHAPE(DynamicPointToVoxelInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DynamicPointToVoxelInferDtype));


PD_BUILD_OP(dynamic_point_to_voxel_bkwd)
    .Inputs({"grad_reduced_feats", 
            "feats", "reduced_feats", "coors_idx", "reduce_count"})
    .Outputs({"grad_feats"})
    .SetKernelFn(PD_KERNEL(dynamic_point_to_voxel_bkwd))
    .Attrs({"reduce_type: std::string"});
