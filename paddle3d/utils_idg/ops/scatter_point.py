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
import paddle.nn.functional as F
from paddle.autograd import PyLayer
from paddle3d.ops import dynamic_point_to_voxel
import numpy as np

class _dynamic_scatter(PyLayer):

    @staticmethod
    def forward(ctx, feats, coors, reduce_type='max'):
        """convert kitti points(N, >=3) to voxels.

        Args:
            feats: [N, C] float tensor. points features to be reduced
                into voxels.
            coors: [N, ndim] int tensor. corresponding voxel coordinates
                (specifically multi-dim voxel index) of each points.
            reduce_type: str. reduce op. support 'max', 'sum' and 'mean'
        Returns:
            tuple
            voxel_feats: [M, C] float tensor. reduced features. input features
                that shares the same voxel coordinates are reduced to one row
            coordinates: [M, ndim] int tensor, voxel coordinates.
        """
        results = dynamic_point_to_voxel.dynamic_point_to_voxel_fwd(feats, coors, reduce_type) #dynamic_point_to_voxel_forward(feats, coors, reduce_type)
        (voxel_feats, voxel_coors, point2voxel_map,
         voxel_points_count) = results
        ctx.reduce_type = reduce_type
        ctx.save_for_backward(feats, voxel_feats, point2voxel_map,
                              voxel_points_count)
        ctx.mark_non_differentiable(voxel_coors)
        return voxel_feats, voxel_coors

    @staticmethod
    def backward(ctx, grad_voxel_feats, grad_voxel_coors=None):
        (feats, voxel_feats, point2voxel_map,
         voxel_points_count) = ctx.saved_tensor()
        grad_feats = paddle.zeros_like(feats) 
        # TODO: whether to use index put or use cuda_backward
        # To use index put, need point to voxel index
        grad_feats = dynamic_point_to_voxel.dynamic_point_to_voxel_bkwd(grad_voxel_feats, feats,
                                        voxel_feats, point2voxel_map,
                                        voxel_points_count, ctx.reduce_type)
        return grad_feats, None 


dynamic_scatter = _dynamic_scatter.apply


class DynamicScatter(nn.Layer):

    def __init__(self, voxel_size, point_cloud_range, mode: str):
        super(DynamicScatter, self).__init__()
        """Scatters points into voxels, used in the voxel encoder with
           dynamic voxelization

        **Note**: The CPU and GPU implementation get the same output, but
        have numerical difference after summation and division (e.g., 5e-7).

        Args:
            average_points (bool): whether to use avg pooling to scatter
                points into voxel voxel_size (list): list [x, y, z] size
                of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.mode = mode

    def forward_single(self, points, coors):
        reduce = self.mode 
        result = dynamic_scatter(points, coors, reduce)
        return result

    def forward(self, points, coors):
        """
        Args:
            input: NC points
        """
        if coors.shape[-1] == 3:
            return self.forward_single(points, coors)
        else:
            batch_size = coors[-1, 0].cast('int32') + 1
            voxels, voxel_coors = [], []
            for i in range(batch_size):
                inds = coors[:, 0] == i 
                if getattr(self, 'export_model', False):
                    voxel, voxel_coor, _, _ = dynamic_point_to_voxel.dynamic_point_to_voxel_fwd(points[inds], coors[inds][:, 1:], self.mode)
                else:
                    voxel, voxel_coor = self.forward_single(
                        points[inds], coors[inds][:, 1:])
                tmp = paddle.full((voxel_coor.shape[0], 1), i, dtype = voxel_coor.dtype)
                coor_pad = paddle.concat([tmp, voxel_coor], axis = 1)
                voxel_coors.append(coor_pad)
                voxels.append(voxel)
            features = paddle.concat(voxels, axis = 0) 
            feature_coors = paddle.concat(voxel_coors, axis=0)
            
            return features, feature_coors
