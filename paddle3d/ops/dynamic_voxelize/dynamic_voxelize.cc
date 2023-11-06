#include <paddle/extension.h>

std::vector<paddle::Tensor> dynamic_voxelize_gpu(const paddle::Tensor &points,
                          const std::vector<float> &voxel_size,
                          const std::vector<float> &coors_range,
                          const int NDim);

std::vector<paddle::Tensor> dynamic_voxelize_fwd(const paddle::Tensor &points,
                             const std::vector<float> &voxel_size,
                             const std::vector<float> &coors_range
                            //  const int nDim
                             ) {
    auto coors = dynamic_voxelize_gpu(points, voxel_size, coors_range, 3);
    return coors;
}


std::vector<paddle::DataType> DynamicVoxelizeInferDtype(paddle::DataType points_dtype) {
  return {paddle::DataType::INT32};
}

std::vector<std::vector<int64_t>> DynamicVoxelizeInferShape(
    std::vector<int64_t> points_shape) {
  return {{points_shape[0], 3}};
}

PD_BUILD_OP(dynamic_voxelize_fwd)
    .Inputs({"points"})
    .Outputs({"coors"}) // {reduced_feats, out_coors, coors_map, reduce_count}    
    .Attrs({"voxel_size: std::vector<float>", "coors_range: std::vector<float>"})//, "nDim: int"})
    .SetKernelFn(PD_KERNEL(dynamic_voxelize_fwd))
    .SetInferShapeFn(PD_INFER_SHAPE(DynamicVoxelizeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DynamicVoxelizeInferDtype));
