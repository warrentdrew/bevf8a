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
from paddle3d.ops import bev_feature_layer

import numpy as np

class BevFeature(object):
    """ as name """
    def __init__(self,
                voxel_size,
                point_cloud_range,
                cnnseg_feature_dim=6,
                compute_logodds_cfg=None):
        """
        compute_logodds_cfg = {
            compute_logodds=True,
            vis_range=[-120, -120, -3, 120, 120, 2],
            vis_voxel_size=[0.2083, 0.2083, 1.25],
        }
        """
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.cnnseg_feature_dim = cnnseg_feature_dim
        self.bev_map_feature_dim = cnnseg_feature_dim
        grid_size = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        self.grid_size = np.round(grid_size).astype(np.int32)
        if compute_logodds_cfg is not None:
            self.compute_logodds = compute_logodds_cfg.get("compute_logodds", False)
            self.vis_range = np.array(compute_logodds_cfg.get("vis_range", 
                                    [-60, -60, -5, 60, 60, 5]), dtype = np.float64)
            self.vis_voxel_size = np.array(compute_logodds_cfg.get('vis_voxel_size', [0.2, 0.2, 10]))
            vxsize = int((self.vis_range[3] - self.vis_range[0]) / self.vis_voxel_size[0])
            vysize = int((self.vis_range[4] - self.vis_range[1]) / self.vis_voxel_size[1])
            vzsize = int((self.vis_range[5] - self.vis_range[2]) / self.vis_voxel_size[2])
            self.bev_map_feature_dim += vzsize
            self.vis_grid_size = np.array([vxsize, vysize, vzsize])
            self.vis_origins = np.array(compute_logodds_cfg.get('vis_origin', [0.0, 0.0, 0.0]))
            self.vis_offset_origins = self.vis_origins - self.vis_range[:3]
            self.lo_occupied = 0.7
            self.lo_free = 0.4
            
        else:
            self.compute_logodds = False

    @paddle.no_grad()     
    def __call__(self, points):
        # points has timestamp [x,y,z,i,t]
        if points.shape[1] == 5:
            cur_points_mask = points[:, -1] == 0
            cur_points = points[cur_points_mask, :]
        else:
            cur_points = points            
        
        bev_map_feature = paddle.zeros((int(self.cnnseg_feature_dim), int(self.grid_size[1]), int(self.grid_size[0])))
        bev_map_feature[0, :, :] = -5.
        # calculate points2grid
        pos_xf = (cur_points[:, 0] - self.point_cloud_range[0]) / float(self.voxel_size[0])
        pos_yf = (cur_points[:, 1] - self.point_cloud_range[1]) / float(self.voxel_size[1])
        pos_zf = (cur_points[:, 2] - self.point_cloud_range[2]) / float(self.voxel_size[2])
        
        points2grid = pos_yf.cast("int32") * self.grid_size[1] + pos_xf.cast("int32") 
        mask = (pos_xf < 0) | (pos_xf >= self.grid_size[0])
        mask |= ((pos_yf < 0) | (pos_yf >= self.grid_size[1]))
        mask |= ((cur_points[:, 2] < self.point_cloud_range[2]) | (cur_points[:, 2] > self.point_cloud_range[5]))
        points2grid = paddle.where(mask, paddle.full(points2grid.shape, -1, dtype = 'int32'), points2grid)
        points2grid = points2grid.cast("int32") 
        bev_map_feature = bev_feature_layer.cnnseg_feature_generator_gpu(cur_points, 
                                                            points2grid, 
                                                            self.cnnseg_feature_dim,
                                                            self.grid_size[0],
                                                            self.grid_size[1],
                                                            self.grid_size[2])
        # 0: max_height 1: mean_height 2: count 3: top_intensity 4: mean_intensity
        # 5: nonempty
        bev_map_feature = bev_map_feature.reshape((self.cnnseg_feature_dim, -1))
        nonempty_mask = bev_map_feature[2, ...]  > 0 
        
        bev_map_feature_tmp0 = paddle.where(~nonempty_mask, paddle.zeros(bev_map_feature[0].shape), bev_map_feature[0])
        bev_map_feature_tmp1 = paddle.where(nonempty_mask, (bev_map_feature[1] / (bev_map_feature[2] + 1e-8)), bev_map_feature[1])
        bev_map_feature_tmp4 = paddle.where(nonempty_mask, (bev_map_feature[4] / (bev_map_feature[2] + 1e-8)), bev_map_feature[4])
        bev_map_feature_tmp5 = paddle.where(nonempty_mask, paddle.ones(bev_map_feature[5].shape), bev_map_feature[5])

        bev_map_feature = bev_map_feature.put_along_axis(paddle.to_tensor([[0]]), bev_map_feature_tmp0, axis=0)
        bev_map_feature = bev_map_feature.put_along_axis(paddle.to_tensor([[1]]), bev_map_feature_tmp1, axis=0)
        bev_map_feature = bev_map_feature.put_along_axis(paddle.to_tensor([[4]]), bev_map_feature_tmp4, axis=0)
        bev_map_feature = bev_map_feature.put_along_axis(paddle.to_tensor([[5]]), bev_map_feature_tmp5, axis=0)

        bev_map_feature = bev_map_feature.put_along_axis(paddle.to_tensor([[2]]), 
                                        paddle.log(bev_map_feature[2, ...] + 1), axis=0)
        bev_map_feature = bev_map_feature.reshape((self.cnnseg_feature_dim, self.grid_size[1], self.grid_size[0]))
        if self.compute_logodds:
            logodds = paddle.zeros((int(self.vis_grid_size[2]), int(self.vis_grid_size[1]), int(self.vis_grid_size[0])))
            
            logodds = bev_feature_layer.visibility_feature_gpu(cur_points,
                                                        self.vis_voxel_size[0],
                                                        self.vis_voxel_size[1],
                                                        self.vis_voxel_size[2],
                                                        self.vis_range[0],
                                                        self.vis_range[1],
                                                        self.vis_range[2],
                                                        self.vis_grid_size[0],
                                                        self.vis_grid_size[1],
                                                        self.vis_grid_size[2],
                                                        self.vis_offset_origins[0],
                                                        self.vis_offset_origins[1],
                                                        self.vis_offset_origins[2],
                                                        self.lo_occupied,
                                                        self.lo_free)
            logodds = paddle.where(logodds <= 0, paddle.full(logodds.shape, 0.5, 'float32'), logodds)

            bev_map_feature = paddle.concat([bev_map_feature, logodds], axis=0)
        bev_map_feature = bev_map_feature.reshape((self.bev_map_feature_dim, self.grid_size[1], self.grid_size[0]))
        return bev_map_feature
