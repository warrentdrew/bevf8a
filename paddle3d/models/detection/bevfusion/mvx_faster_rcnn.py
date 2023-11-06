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
import paddle.nn.functional as F
from paddle3d.apis import manager
from paddle3d.models.detection.bevfusion.mvx_two_stage import \
    MVXTwoStageDetector

__all__ = ['MVXFasterRCNN', 'DynamicMVXFasterRCNN']


@manager.MODELS.add_component
class MVXFasterRCNN(MVXTwoStageDetector):
    """Multi-modality model using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(MVXFasterRCNN, self).__init__(**kwargs)

@manager.MODELS.add_component
class DynamicMVXFasterRCNN(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN and dynamic voxelization."""

    def __init__(self, **kwargs):
        super(DynamicMVXFasterRCNN, self).__init__(**kwargs)

    @paddle.no_grad()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[paddle.Tensor]): Points of each sample.

        Returns:
            tuple[paddle.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        tmp_bev_features = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
            if self.compute_bev_feature:
                tmp_bev_feature = self.bev_feature_layer(res)
                tmp_bev_features.append(tmp_bev_feature.unsqueeze(0))
        points = paddle.concat(points, axis=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = paddle.concat([i * paddle.ones((coor.shape[0], 1), dtype = coor.dtype), coor], axis = 1)
            #F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = paddle.concat(coors_batch, axis=0)
        if self.compute_bev_feature:
            bev_features = paddle.concat(tmp_bev_features, axis=0)
        else:
            bev_features = None
            
        return points, coors_batch, bev_features

    def extract_pts_feat(self, points, img_feats, img_metas):
        """Extract point features."""
        if not self.with_pts_bbox:
            return None
        voxels, self.coors, bev_features = self.voxelize(points)
        voxel_features, feature_coors = self.pts_voxel_encoder(voxels, self.coors)
        batch_size = self.coors[-1, 0] + 1
        input_shape = self.pts_voxel_layer.pcd_shape[::-1]
        
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size, input_shape, bev_features)
        if self.with_pts_backbone:  
            x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

