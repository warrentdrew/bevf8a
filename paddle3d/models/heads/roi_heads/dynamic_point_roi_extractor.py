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
from paddle import nn as nn
from paddle3d.utils_idg.ops.dynamic_point_pool import dynamic_point_poolv2

class DynamicPointROIExtractor(nn.Layer):
    """Point-wise roi-aware Extractor.

    Extract Point-wise roi features.

    Args:
        roi_layer (dict): The config of roi layer.
    """

    def __init__(
        self,
        debug=False,
        extra_wlh=[0, 0, 0],
        grid_size=[20, 20, 10],
        max_inbox_point=1024,
        adaptive_extra=False,
        use_grid=(False, False),
    ):
        super().__init__()
        self.debug = debug
        self.extra_wlh = extra_wlh
        self.grid_size = grid_size
        self.use_grid = use_grid
        self.box_grid_num = int(grid_size[0] * grid_size[1] * grid_size[2])
        self.max_inbox_point = max_inbox_point
        self.adaptive_extra = adaptive_extra
        self.max_all_pts=1000000
        print("use adaptive extra ===============", self.adaptive_extra)
        print("use grid ==============", self.use_grid)

    def forward(self, pts, batch_pts_inds, rois, batch_roi_inds, max_inbox_point=None):
        # assert batch_inds is sorted
        assert len(pts) > 0
        assert len(batch_pts_inds) > 0
        assert len(rois) > 0
        assert len(batch_roi_inds) > 0

        if not (batch_pts_inds == 0).all():
            assert (batch_pts_inds.sort() == batch_pts_inds).all()  # 保证batch_id有序
        if not (batch_roi_inds == 0).all():
            assert (batch_roi_inds.sort() == batch_roi_inds).all()  # 保证roi_id有

        all_pts_xyz,  all_pts_feats, all_pts_info, all_roi_inds = [], [], [], []
        all_pts_batch_ids = []
        # batch_roi_empty_masks = rois.new_ones(rois.shape[0],dtype=paddle.bool)
        roi_inds_base = 0
        pts_inds_base = 0

        for batch_idx in range(int(batch_pts_inds.max()) + 1):
            roi_batch_mask = batch_roi_inds == batch_idx
            pts_batch_mask = batch_pts_inds == batch_idx

            num_roi_this_batch = roi_batch_mask.sum().item()
            num_pts_this_batch = pts_batch_mask.sum().item()
            if num_roi_this_batch == 0 or num_pts_this_batch == 0:
                ext_pts_xyz = paddle.zeros((0, 3), dtype=pts.dtype)  
                ext_pts_feats = paddle.zeros((0, 1), dtype=pts.dtype) 
                ext_pts_info = paddle.zeros((0, 10), dtype=pts.dtype)
                ext_roi_inds = -1 * paddle.ones((0,), dtype='int64')
                ext_pts_batch_ids = -1 * paddle.ones((0,), dtype='int64')
                print("roi num is zero!!!")

            else:
                ext_pts_xyz,  ext_pts_feats, ext_pts_info, ext_roi_inds = dynamic_point_poolv2(
                    rois[roi_batch_mask],
                    pts[pts_batch_mask],
                    self.extra_wlh,
                    self.max_inbox_point
                )
                
                ext_pts_batch_ids = paddle.ones((ext_pts_xyz.shape[0],), dtype='int64') *  batch_idx
                
            all_pts_xyz.append(ext_pts_xyz)
            all_pts_feats.append(ext_pts_feats)
            all_pts_info.append(ext_pts_info)
            all_roi_inds.append(ext_roi_inds + roi_inds_base)
            # all_roi_inds.append(ext_roi_inds + roi_inds_base)
            all_pts_batch_ids.append(ext_pts_batch_ids)
            pts_inds_base += num_pts_this_batch
            roi_inds_base += num_roi_this_batch

        all_pts_xyz = paddle.concat(all_pts_xyz, axis=0)
        all_pts_feats = paddle.concat(all_pts_feats, axis=0)
        all_pts_info = paddle.concat(all_pts_info, axis=0)
        all_roi_inds = paddle.concat(all_roi_inds, axis=0)
        all_pts_batch_ids = paddle.concat(all_pts_batch_ids, axis=0)

        return all_pts_xyz, all_pts_feats, all_pts_info, all_roi_inds, all_pts_batch_ids
