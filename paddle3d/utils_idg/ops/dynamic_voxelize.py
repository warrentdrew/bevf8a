# !/usr/bin/env python3
"""
Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import paddle
from paddle import nn
from paddle.autograd import PyLayer
from paddle3d.ops import dynamic_voxelize_layer


class _Voxelization(PyLayer):

    @staticmethod
    def forward(ctx,
                points,
                voxel_size,
                coors_range,
                max_points=35,
                max_voxels=20000):
        """convert kitti points(N, >=3) to voxels.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.

        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        nDim = 3
        coors = paddle.zeros((points.shape[0], 3), dtype='int32') 

        coors = dynamic_voxelize_layer.dynamic_voxelize_fwd(points, voxel_size, coors_range) #, nDim)
        return coors

voxelization = _Voxelization.apply


class Voxelization(nn.Layer):

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = list(map(float, point_cloud_range))
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = (max_voxels, max_voxels) #_pair(max_voxels)

        point_cloud_range = paddle.to_tensor(point_cloud_range, dtype = 'float32')
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = paddle.to_tensor(voxel_size, dtype = 'float32')
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = paddle.round(grid_size).cast("int64")
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w]
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def forward(self, inputs):
        """
        Args:
            input: NC points, C=4
        """
        if getattr(self, 'export_model', False):
            coors = dynamic_voxelize_layer.dynamic_voxelize_fwd(inputs, self.voxel_size, self.point_cloud_range)
            self.pcd_shape = [1, 
                            round((self.point_cloud_range[4]-self.point_cloud_range[1])/self.voxel_size[0]),
                            round((self.point_cloud_range[3]-self.point_cloud_range[0])/self.voxel_size[0])]
            return coors
        else:
            if self.training:
                max_voxels = self.max_voxels[0]
            else:
                max_voxels = self.max_voxels[1]

            return voxelization(inputs, self.voxel_size, self.point_cloud_range,
                                self.max_num_points, max_voxels)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ')'
        return tmpstr


if __name__ == '__main__':
    import numpy as np

    voxel_size =  [0.2083, 0.2083, 10.]
    point_cloud_range = [-120, -120, -5, 120, 120, 5] #[-120., -120., -5., 120., 120., 5.]
    max_num_points = -1
    max_voxels = -1

    inputpts = np.load('opcheck_npy/input_dv.npy')
    vx = Voxelization(voxel_size, point_cloud_range, max_num_points, max_voxels)
    inputpts = paddle.to_tensor(inputpts)
    coors_out = vx(inputpts)

    print(coors_out.shape)
