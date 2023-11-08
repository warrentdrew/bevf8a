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
from paddle3d.ops import dynamic_point_poolv2_op
import numpy as np

class DynamicPointPoolFunction(PyLayer):

    @staticmethod
    def forward(ctx, rois, pts, extra_wlh, max_box_pts=2048, max_all_pts=1000000):
        """convert kitti points(N, >=3) to voxels.

        Args:
            rois (paddle.Tensor): [N, 7], in LiDAR coordinate,
                (x, y, z) is the bottom center of rois
            pts (paddle.Tensor): [npoints, 3]
            pts_feature (paddle.Tensor): [npoints, C]
            out_size (int or tuple): n or [n1, n2, n3]
            max_pts_per_voxel (int): m
            mode (int): 0 (max pool) or 1 (average pool)
        """
        extra_wlh = paddle.to_tensor(extra_wlh)
        # out_pts_idx0 = -1 * pts.new_ones(max_all_pts, dtype=torch.long)  # 采样的每个点对应的原始点云的索引
        # out_roi_idx0 = -1 * pts.new_ones(max_all_pts, dtype=torch.long)  # 采样的每个点对应的roi的索引
        # out_pts_feats0 = pts.new_zeros(max_all_pts, 10, dtype=torch.float)  # 采样的点特征
        # global_count = pts.new_zeros(1, dtype=torch.int32)
        # global_count = paddle.zeros([1], dtype=np.int32)
        
        # out_pts_xyz = pts.new_zeros(max_all_pts, 3)  # 采样的每个点对应的原始点云的索引
        # out_pts_feats = pts.new_zeros(max_all_pts, 1)  # 采样的每个点对应的原始点云的索引
        # out_roi_idx = -1 * pts.new_ones(max_all_pts, dtype=torch.long)  # 采样的每个点对应的roi的索引
        # out_pts_info = pts.new_zeros(max_all_pts, 10)  # 采样的点特征

        results = dynamic_point_poolv2_op.dynamic_point_pool_gpu(
                rois, pts, extra_wlh, max_box_pts, max_all_pts) #dynamic_point_to_voxel_forward(feats, coors, reduce_type)
        (out_pts_xyz, out_pts_feats, out_pts_info, out_roi_idx, global_count) = results
        # ctx.reduce_type = reduce_type
        valid_mask = out_roi_idx >= 0 
        global_count = global_count.item()
        # global_count = inbox_counter[-1]
        # assert paddle.allclose(out_roi_idx[valid_mask].astype('float32'), out_roi_idx[:global_count].astype('float32'))

        out_pts_xyz = out_pts_xyz[:global_count]
        out_pts_feats = out_pts_feats[:global_count]
        out_pts_info = out_pts_info[:global_count]
        out_roi_idx = out_roi_idx[:global_count]
      
        ctx.mark_non_differentiable(out_pts_xyz)
        ctx.mark_non_differentiable(out_pts_feats)
        ctx.mark_non_differentiable(out_pts_info)
        ctx.mark_non_differentiable(out_roi_idx)

        return out_pts_xyz, out_pts_feats, out_pts_info, out_roi_idx

    @staticmethod
    def backward(ctx, g1, g2, g3, g4):

        return None, None#, None, None, None, None, None, None, None, None, None


dynamic_point_poolv2 = DynamicPointPoolFunction.apply
