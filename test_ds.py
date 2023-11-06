from paddle3d.ops import dynamic_point_to_voxel
import paddle

points = paddle.load('ad2badcase/points_rank_25.pdparams')
coors = paddle.load('ad2badcase/coors_rank_25.pdparams')

out = dynamic_point_to_voxel.dynamic_point_to_voxel_fwd(points, coors, 'mean')

print(out.shape)