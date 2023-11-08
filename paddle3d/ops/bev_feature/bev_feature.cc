#include <paddle/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.") //check input only check device on gpu

#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true){
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void mapfeatureLauncher(const cudaStream_t& stream,
                        const int num, 
                        const int point_dim,
                        const float* points_data,
                        const int* points2grid_data, 
                        float* max_height_data,
                        float* mean_height_data,
                        float* mean_intensity_data,
                        float* count_data);
void TopIntensityLauncher(const cudaStream_t& stream,
                         const int num, 
                         const int point_dim,
                         const float* points_data,
                         const int* points2grid_data, 
                         float* top_intensity_data,
                         float* max_height_data);
void ComputeLogOddsGPULauncher(const cudaStream_t& stream,
                                const int num, 
                                const int point_dim,
                                const float voxel_size_x,
                                const float voxel_size_y,
                                const float voxel_size_z,
                                const float pxmin,
                                const float pymin,
                                const float pzmin,
                                const int grid_size_x,
                                const int grid_size_y,
                                const int grid_size_z,
                                const float offset_origin_x,
                                const float offset_origin_y,
                                const float offset_origin_z,
                                float lo_occupied,
                                float lo_free,
                                const float* points_data,
                                float* logodds_data);

std::vector<paddle::Tensor> cnnseg_feature_generator_gpu(const paddle::Tensor &points,
                                const paddle::Tensor &points2grid,
                                //const paddle::Tensor cnnseg_feature,
                                const int cnnseg_feature_dim,
                                const int grid_size_x,
                                const int grid_size_y,
                                const int grid_size_z) {
    CHECK_INPUT(points);
    CHECK_INPUT(points2grid);
    auto cnnseg_feature = paddle::full({cnnseg_feature_dim, grid_size_y, grid_size_x}, 0.0, points.type(), paddle::GPUPlace());
    cnnseg_feature = paddle::scatter(cnnseg_feature, paddle::full({1}, 0, points2grid.type(), paddle::GPUPlace()), paddle::full({1, grid_size_y, grid_size_x}, -5.0, points.type(), paddle::GPUPlace())); // TODO1023
    const float* points_data = points.data<float>();
    const int* points2grid_data = points2grid.data<int>();
    float* cnnseg_feature_data = cnnseg_feature.data<float>();
    int unit_offset = grid_size_x * grid_size_y;
    float* max_height_data = cnnseg_feature_data + unit_offset * 0;
    float* mean_height_data = cnnseg_feature_data + unit_offset * 1;
    float* count_data = cnnseg_feature_data + unit_offset * 2;
    float* top_intensity_data = cnnseg_feature_data + unit_offset * 3;
    float* mean_intensity_data = cnnseg_feature_data + unit_offset * 4;
    float* nonempty_data = cnnseg_feature_data + unit_offset * 5;
    int num_point = points.shape()[0]; 
    int point_dim = points.shape()[1]; 
    mapfeatureLauncher(cnnseg_feature.stream(), 
                        num_point,
                        point_dim,
                        points_data,
                        points2grid_data,
                        max_height_data, 
                        mean_height_data,
                        mean_intensity_data, 
                        count_data);
	
    TopIntensityLauncher(cnnseg_feature.stream(), 
                        num_point,
                        point_dim,
                        points_data,
                        points2grid_data,
                        top_intensity_data,
                        max_height_data);
    cudaDeviceSynchronize();

    return {cnnseg_feature};
}

std::vector<paddle::Tensor> visibility_feature_gpu(const paddle::Tensor &points,
                            const float voxel_size_x,
                            const float voxel_size_y,
                            const float voxel_size_z,
                            const float pc_start_x,
                            const float pc_start_y,
                            const float pc_start_z,
                            const int grid_size_x,
                            const int grid_size_y,
                            const int grid_size_z,
                            const float offset_origin_x,
                            const float offset_origin_y,
                            const float offset_origin_z,
                            float lo_occupied,
                            float lo_free) {
    CHECK_INPUT(points);
    auto logodds_feature = paddle::full({grid_size_z, grid_size_y, grid_size_x}, 0.0, points.type(), paddle::GPUPlace());
    const float* points_data = points.data<float>();
    float* logodds_feature_data = logodds_feature.data<float>();
    int num_point = points.shape()[0]; 
    int point_dim = points.shape()[1]; 
    ComputeLogOddsGPULauncher(points.stream(), 
                              num_point,
                              point_dim,
                              voxel_size_x,
                              voxel_size_y,
                              voxel_size_z,
                              pc_start_x,
                              pc_start_y,
                              pc_start_z,
                              grid_size_x,
                              grid_size_y,
                              grid_size_z,
                              offset_origin_x,
                              offset_origin_y,
                              offset_origin_z,
                              lo_occupied,
                              lo_free,
                              points_data,
                              logodds_feature_data);
    cudaDeviceSynchronize();
    return {logodds_feature};//1;
}

std::vector<paddle::DataType> CnnsegFeatureGeneratorGpuInferDtype(paddle::DataType points_dtype, 
                                                                  paddle::DataType points2grid_dtype) {
  return {points_dtype}; //TODO
}

std::vector<std::vector<int64_t>>  CnnsegFeatureGeneratorGpuInferShape(std::vector<int64_t> points_shape, 
                                                                      std::vector<int64_t> points2grid_shape,
                                                                      const int cnnseg_feature_dim,
                                                                      const int grid_size_x,
                                                                      const int grid_size_y,
                                                                      const int grid_size_z) {
  return {{cnnseg_feature_dim, grid_size_y, grid_size_x}};
}


std::vector<paddle::DataType> VisibilityFeatureGpuInferDtype(paddle::DataType points_dtype) {
  return {points_dtype}; //TODO
}

std::vector<std::vector<int64_t>>  VisibilityFeatureGpuInferShape(std::vector<int64_t> points_shape,
                                                                    const float voxel_size_x,
                                                                    const float voxel_size_y,
                                                                    const float voxel_size_z,
                                                                    const float pc_start_x,
                                                                    const float pc_start_y,
                                                                    const float pc_start_z,
                                                                    const int grid_size_x,
                                                                    const int grid_size_y,
                                                                    const int grid_size_z,
                                                                    const float offset_origin_x,
                                                                    const float offset_origin_y,
                                                                    const float offset_origin_z,
                                                                    float lo_occupied,
                                                                    float lo_free) { //TODO

  return {{grid_size_z, grid_size_y, grid_size_x}};
}


PD_BUILD_OP(cnnseg_feature_generator_gpu)
    .Inputs({"points", "pounts2grid"})
    .Outputs({"cnnseg_feature"}) // {reduced_feats, out_coors, coors_map, reduce_count}
    .SetKernelFn(PD_KERNEL(cnnseg_feature_generator_gpu))
    .Attrs({"cnnseg_feature_dim: int", "grid_size_x: int", "grid_size_y: int", "grid_size_z: int"})
    .SetInferShapeFn(PD_INFER_SHAPE(CnnsegFeatureGeneratorGpuInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(CnnsegFeatureGeneratorGpuInferDtype));


PD_BUILD_OP(visibility_feature_gpu)
    .Inputs({"points"})
    .Outputs({"logodds_feature"})
    .SetKernelFn(PD_KERNEL(visibility_feature_gpu))
    .Attrs({"voxel_size_x: float", "voxel_size_y: float", "voxel_size_z: float",
            "pc_start_x: float", "pc_start_y: float", "pc_start_z: float",
            "grid_size_x: int", "grid_size_y: int", "grid_size_z: int",
            "offset_origin_x: float", "offset_origin_y: float", "offset_origin_z: float",
            "lo_occupied: float", "lo_free: float"})
    .SetInferShapeFn(PD_INFER_SHAPE(VisibilityFeatureGpuInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(VisibilityFeatureGpuInferDtype));
