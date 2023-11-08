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

import logging
import numpy as np
import math
from collections import OrderedDict

import paddle
from paddle import nn
from paddle.nn import functional as F

from paddle3d.utils.logger import logger
from paddle3d.utils import get_bev_corners_paddle
from paddle3d.apis import manager
from paddle3d.utils_idg.box_paddle_ops import (center_to_corner_box2d, corner_to_standup_nd, 
                                        nms, nms_overlap, rotate_nms_overlap, center_to_corner_box3d)
from paddle3d.utils_idg.ops import iou3d_utils
from paddle3d.utils_idg.build_layer import build_norm_layer, get_activation_layer
from paddle3d.utils_idg.ops.roi_align_rotated import RoIAlignRotated
from paddle3d.utils_idg.proposal_target_layer import ProposalTargetLayer
from paddle3d.models.layers.param_init import constant_init, kaiming_normal_init, normal_init, init_weight
from .dynamic_point_roi_extractor import DynamicPointROIExtractor
from .sir_layerv2 import DynamicPointNetV2
from .confidence_map_head import one_hot_f, cat
from scipy import io


pcd_metadata = {'version': '0.7',
                'fields': ['x', 'y', 'z', 'intensity'],
                'size': [4, 4, 4, 1],
                'type': ['F', 'F', 'F', 'U'],
                'count': [1, 1, 1, 1],
                'width': 61755,
                'height': 1,
                'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                'points': 61755,
                'data': 'binary_compressed'}
dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),
               ('intensity', np.uint8)])

class NormedLinear(nn.Linear):
    """Normalized Linear Layer.

    Args:
        tempeature (float, optional): Tempeature term. Default to 20.
        power (int, optional): Power term. Default to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Default to 1e-6.
    """

    def __init__(self, *args, tempearture=20, power=1.0, eps=1e-6, **kwargs):
        super(NormedLinear, self).__init__(*args, **kwargs)
        self.tempearture = tempearture
        self.power = power
        self.eps = eps
        self.init_weights()

    def init_weights(self):
        normal_init(self.weight, loc=0.0, scale=0.01)
        if self.bias is not None:
            constant_init(self.bias, 0)

    def forward(self, x):
        weight_ = self.weight / \
            (paddle.linalg.norm(self.weight, axis=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x / (paddle.linalg.norm(x, axis=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x_ * self.tempearture

        return F.linear(x_, weight_, bias=self.bias)


class PointHead(nn.Layer):
    def __init__(
        self,
        inchannel=256,
        cls_num=5,
        pred_iou=False,
        norm_cfg=dict(type_name="LN", epsilon=1e-3),
        act="gelu",
        dropout=0,
        use_direction_classifier=False,
        use_norm=False,
    ):
        super(PointHead, self).__init__()
        self.pred_iou = pred_iou
        self.use_direction_classifier = use_direction_classifier
        self.cls_num = cls_num
        self.fc1 = nn.Sequential(
            nn.Linear(inchannel, 256, bias_attr=False),
            build_norm_layer(norm_cfg, 256)[1],
            get_activation_layer(act),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256, bias_attr=False),
            build_norm_layer(norm_cfg, 256)[1],
            get_activation_layer(act),
            nn.Dropout(dropout),
        )
        # NOTE: should there put a BN?
        self.cls_pred = nn.Sequential(
            nn.Linear(256, 256, bias_attr=False),
            build_norm_layer(norm_cfg, 256)[1],
            get_activation_layer(act),
            NormedLinear(256, cls_num, bias_attr=True) if use_norm else nn.Linear(
                256, cls_num, bias_attr=True),
        )
        self.center_pred = nn.Sequential(
            nn.Linear(256, 256, bias_attr=False),
            build_norm_layer(norm_cfg, 256)[1],
            get_activation_layer(act),
            nn.Linear(256, 3, bias_attr=True),
        )
        self.size_pred = nn.Sequential(
            nn.Linear(256, 256, bias_attr=False),
            build_norm_layer(norm_cfg, 256)[1],
            get_activation_layer(act),
            nn.Linear(256, 3, bias_attr=True),
        )

        self.heading_pred = nn.Sequential(
            nn.Linear(256, 256, bias_attr=False),
            build_norm_layer(norm_cfg, 256)[1],
            get_activation_layer(act),
            nn.Linear(256, 1, bias_attr=True),
        )
        if self.use_direction_classifier:
            self.dir_pred = nn.Sequential(
                nn.Linear(256, 256, bias_attr=False),
                build_norm_layer(norm_cfg, 256)[1],
                get_activation_layer(act),
                nn.Linear(256, 2, bias_attr=True),
            )

        self.iou_pred = nn.Sequential(
            nn.Linear(256, 256, bias_attr=False),
            build_norm_layer(norm_cfg, 256)[1],
            get_activation_layer(act),
            nn.Linear(256, 1, bias_attr=True),
        )
        self.init_weights()

    def forward(self, roi_feat):
        if roi_feat.shape[0] <= 0:
            logits = paddle.zeros([0, self.cls_num], dtype=roi_feat.dtype)
            centers = paddle.zeros([0, 3], dtype=roi_feat.dtype)
            sizes = paddle.zeros([0, 3], dtype=roi_feat.dtype)
            headings = paddle.zeros([0, 1], dtype=roi_feat.dtype)
            ious = paddle.zeros([0, 1], dtype=roi_feat.dtype)
            if self.use_direction_classifier:
                dirs = paddle.zeros([0, 2], dtype=roi_feat.dtype)
            return logits, centers, sizes, headings, ious, dirs if self.use_direction_classifier else None
        roi_feat = self.fc1(roi_feat)
        feat = self.fc2(roi_feat)
        logits = self.cls_pred(feat)
        centers = self.center_pred(feat)
        sizes = self.size_pred(feat)
        headings = self.heading_pred(feat)
        ious = self.iou_pred(feat)
        if self.use_direction_classifier:
            dirs = self.dir_pred(feat)

        return logits, centers, sizes, headings, ious, dirs if self.use_direction_classifier else None

    def init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                kaiming_normal_init(m.weight, reverse=True)
                if m.bias is not None:
                    constant_init(m.bias, value=0)
            elif isinstance(m, (paddle.nn.BatchNorm1D, paddle.nn.BatchNorm2D, paddle.nn.LayerNorm)):
                constant_init(m.weight, value = 1.0)
                constant_init(m.bias, value = 0.0)


@manager.HEADS.add_component
class LiDARRCNNHeadV2(nn.Layer):
    def __init__(
        self,
        num_blocks=3,
        in_channels=[17, 143, 143],
        feat_channels=[
            [64, 64],
        ]
        * 3,
        rel_mlp_hidden_dims=[
            [16, 32],
        ]
        * 3,
        rel_mlp_in_channels=[
            12,
        ]
        * 3,
        bev_channels=121,
        tasks=[],
        bbox_coder=None,
        rpn_cfg=None,
        loss_cls=None,
        loss_bbox=None,
        loss_iou_aware=None,
        loss_bev_cls=None,
        loss_bev_bbox=None,
        sample_pts_num=512,
        box_expand_ratio=[1.0, 1.0, 1.0],
        pc_range=[],
        voxel_size=[],
        roi_sampler_cfg=None,
        use_proposal_label=False,
        use_proposal_box=False,
        roi_grid_size=[3, 3],
        use_proposal_box_feat=False,
        use_proposal_label_feat=False,
        use_centerness_weight=False,
        use_corner_loss=False,
        corner_loss_weight=[1.0, 1.0, 1.0, 1.0, 1.0],
        adaptive_extra=False,
        roi_expand_ratio=[1.0, 1.0, 1.0],
        norm_cfg=dict(type_name="LN", epsilon=1e-3),
        center_scale=1.0,
        center_only_fg=True,
        pool_grid_size=[30, 30, 10],
        use_grid=[False, False],
        loss_pred_bctps=None,
        rel_dist_scaler=10,
        with_distance=False,
        box_refine=False,
        xyz_normalizer=[20, 20, 4],
        dir_loss_cls=[1, 2, 3, 4, 5],
        act="gelu",
        use_proposal_embedding=False,
        roi_bev_channel=256,
        loss_iou=None,
        dropout=0,
        voxel_size_downsample=None,
        loss_aux=None,
        loss_pts_reg=None,
        loss_pts_cls=None,
        pred_aware_assign=False,
        warm_epoch=20,
        iou_cls_weight=[1, 1, 1, 1, 1],
        use_empty_mask=False,
        use_cls_norm_layer=False,
        use_aspp=False,
        use_bev_features=False,
        reg_global=False,
    ):
        super(LiDARRCNNHeadV2, self).__init__()

        self.box_coder = bbox_coder
        self.tasks = tasks
        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.sample_pts_num = sample_pts_num
        self.in_channels = in_channels
        self.box_n_dim = self.box_coder.code_size
        self.rpn_cfg = rpn_cfg
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.loss_pred_bctps = None
        self.loss_iou_aware = loss_iou_aware
        if loss_pred_bctps is not None:
            self.loss_pred_bctps = loss_pred_bctps
        self.similarity_calc = iou3d_utils.RotateIou3dSimilarity()
        self.box_expand_ratio = box_expand_ratio

        self.voxel_size = voxel_size
        self.voxel_size_cuda = paddle.to_tensor(
            self.voxel_size, dtype='float32')
        self.pc_range = pc_range
        self.pc_range_cuda = paddle.to_tensor(
            self.pc_range, dtype='float32')
        self.roi_sampler_cfg = roi_sampler_cfg
        self.use_proposal_label = use_proposal_label
        self.use_proposal_box = use_proposal_box
        self.roi_grid_size = roi_grid_size
        self.bev_channels = bev_channels
        self.use_proposal_box_feat = use_proposal_box_feat
        self.use_proposal_label_feat = use_proposal_label_feat
        self.use_centerness_weight = use_centerness_weight
        self.use_corner_loss = use_corner_loss
        self.corner_loss_weight = paddle.to_tensor(
            corner_loss_weight, dtype='float32')
        self.iou_cls_weight = paddle.to_tensor(
            iou_cls_weight, dtype='float32')
        self.roi_expand_ratio = roi_expand_ratio
        self.center_scale = center_scale
        self.center_only_fg = center_only_fg
        self.pool_grid_size = pool_grid_size
        self.use_grid = use_grid
        self.fp16_enabled = False
        self.box_refine = box_refine
        self.dir_loss_cls = dir_loss_cls
        self.use_aspp = use_aspp
        self.max_distance = math.sqrt(
            self.pc_range[3] * self.pc_range[3] +
            self.pc_range[4] * self.pc_range[4]
        )
        # for sub region
        self.use_proposal_embedding = use_proposal_embedding
        self.loss_bev_cls = None
        self.loss_bev_bbox = None
        self.loss_iou = None
        self.pred_aware_assign = pred_aware_assign
        self.voxel_size_downsample = voxel_size_downsample
        self.epoch = 0
        self.warm_epoch = warm_epoch
        self.use_empty_mask = use_empty_mask
        if voxel_size_downsample:
            self.voxel_downsamplev2 = VoxelDownsampleV2(
                pc_range, voxel_size_downsample)
        if loss_iou:
            self.loss_iou = loss_iou
        if loss_bev_cls:
            self.loss_bev_cls = loss_bev_cls
            self.bev_cls = nn.Linear(
                roi_bev_channel, self.num_class_with_bg, bias_attr=True)
        if loss_bev_bbox:
            self.loss_bev_bbox = loss_bev_bbox
            self.bev_reg = nn.Linear(
                roi_bev_channel, self.box_coder.code_size, bias_attr=True)
        self.loss_aux = None
        if loss_aux is not None:
            self.loss_aux = loss_aux
        self.use_direction_classifier = loss_aux is not None
        self.use_bev_features = use_bev_features
        self.ori_class_names = []
        for i, cur_classe_name in enumerate(self.class_names):
            for cls_name in cur_classe_name:
                if cls_name.endswith('_sub'):
                    continue
                if cls_name == 'verybigMot':
                    continue
                self.ori_class_names.append(cls_name)

        self.ori_class_name_map = OrderedDict()
        flag = 0
        for i, cur_classe_name in enumerate(self.class_names):

            for cls_name in cur_classe_name:
                if cls_name.endswith('_sub'):
                    continue
                self.ori_class_name_map[cls_name] = flag
                flag += 1
                if cls_name == 'verybigMot':
                    self.ori_class_name_map[cls_name] = self.ori_class_name_map['bigMot']
                    flag -= 1
        # import pdb;pdb.set_trace()
        print("LidarRCNN: ori_class_names {} \n ori_class_name_map {}".format(
            self.ori_class_names, self.ori_class_name_map))
        # cat2label{'TrainedOthers': 0, 'smallMot': 1, 'bigMot': 2, 'nonMot': 3, 'pedestrian': 4, 'verybigMot': 5, 'pedestrian_sub': 6, 'accessory_main': 7, 'OnlyBicycle': 8, 'fog': 9, 'spike': 10}
        # ori_class_name_map OrderedDict([('TrainedOthers', 0), ('smallMot', 1), ('bigMot', 2), ('nonMot', 3), ('pedestrian', 4), ('verybigMot', 2), ('accessory_main', 5)])
        self.num_class_with_bg = len(self.ori_class_names) + 1
        self.reg_global = reg_global
        self.dynamic_pointnet = DynamicPointNetV2(
            num_blocks=3,
            in_channels=self.in_channels,
            feat_channels=feat_channels,
            rel_mlp_hidden_dims=rel_mlp_hidden_dims,
            rel_mlp_in_channels=rel_mlp_in_channels,
            norm_cfg=norm_cfg,
            rel_dist_scaler=rel_dist_scaler,
            with_distance=with_distance,
            xyz_normalizer=xyz_normalizer,
            act=act,
        )

        self.roi_align_rot = RoIAlignRotated(
            roi_grid_size, 0.5, pc_range=self.pc_range, voxel_size=self.voxel_size)

        self.roi_net_inchannels = bev_channels * self.roi_grid_size[0] * self.roi_grid_size[1]
        if self.use_proposal_label_feat:
            if self.use_proposal_embedding:
                self.roi_net_inchannels += 32
                self.score_embed = nn.Sequential(
                    nn.Linear(len(self.ori_class_names), 32, bias_attr=False),
                    build_norm_layer(
                        dict(type_name="BN1d", epsilon=1e-3, momentum=0.99), 32)[1],
                    get_activation_layer(act),
                )
            else:
                self.roi_net_inchannels += len(self.ori_class_names)
        if self.use_proposal_box_feat:
            if self.use_proposal_embedding:
                self.roi_net_inchannels += 32
                self.box_embed = nn.Sequential(
                    nn.Linear(self.box_coder.n_dim, 32, bias_attr=False),
                    build_norm_layer(
                        dict(type_name="BN1d", epsilon=1e-3, momentum=0.99), 32)[1],
                    get_activation_layer(act),
                )
            else:
                self.roi_net_inchannels += self.box_coder.n_dim
        self.roi_net = nn.Sequential(
            nn.Linear(self.roi_net_inchannels, roi_bev_channel, bias_attr=False),
            build_norm_layer(dict(type_name="BN1d", epsilon=1e-3,
                             momentum=0.99), roi_bev_channel)[1],
            get_activation_layer(act),
        )

        self.pts_cls_loss = None
        self.pts_reg_loss = None
        if loss_pts_cls:
            self.pts_cls = nn.Linear(
                feat_channels[-1][-1] * 2, self.num_class_with_bg, bias_attr=True)
            self.pts_cls_loss = loss_pts_cls
        if loss_pts_reg:
            self.pts_reg = nn.Linear(feat_channels[-1][-1] * 2, 3, bias_attr=True)
            self.pts_reg_loss = loss_pts_reg
        end_channel = 0
        for c in feat_channels:
            end_channel += sum(c)
        end_channel += roi_bev_channel
        if self.use_bev_features:
            end_channel += (64 + 32)
        self.point_head = PointHead(
            inchannel=end_channel,
            cls_num=len(self.ori_class_names),
            pred_iou=True,
            norm_cfg=dict(type_name="BN1d", epsilon=1e-3, momentum=0.99),
            act=act,
            dropout=dropout,
            use_direction_classifier=self.use_direction_classifier,
            use_norm=use_cls_norm_layer,
        )
        self.proposal_target_layer = ProposalTargetLayer(
            self.roi_sampler_cfg, similarity_calc=self.similarity_calc, num_class=len(
                self.ori_class_names)
        )
        self.roi_extractor = DynamicPointROIExtractor(
            extra_wlh=self.box_expand_ratio,
            max_inbox_point=self.sample_pts_num,
            adaptive_extra=adaptive_extra,
            grid_size=self.pool_grid_size,
            use_grid=self.use_grid,
        )
        self.bev_roi_extractor = None
        self.init_weights()
        logger.info("Finish LiDARRCNNHeadV2 Initialization")

    def init_weights(self):
        for m in self.roi_net.sublayers():
            init_weight(m)

    def forward(self, proposal_list, pts, pts_batch_ids, pts_rpn_feats, bev_features, img_metas):
        with paddle.no_grad():
            if self.voxel_size_downsample:
                pts, pts_batch_ids = self.downsample_points(
                    pts, pts_batch_ids)
            batch_rois, batch_roi_labels, batch_roi_scores, roi_batch_ids = self.merge_batch_roi_infos(
                proposal_list
            )
            pts_xyz, pts_feats, pts_info, pts_roi_inds, new_pts_batch_ids = self.batch_roi_aware_pooling(
                pts, pts_batch_ids, batch_rois, roi_batch_ids, img_metas
            )  # ([num_rois, num_pts, num_feats])
            batch_roi_nonempty_masks = paddle.zeros(
                batch_rois.shape[0], dtype='int32')
            if pts_roi_inds.shape[0] > 0:
                pts_roi_inds_mask = pts_roi_inds >= 0
                pts_roi_inds_ = pts_roi_inds.masked_select(pts_roi_inds_mask)
                # batch_roi_nonempty_masks[pts_roi_inds_] = 1
                if pts_roi_inds_.shape[0]>0:
                    pts_roi_inds_ = pts_roi_inds_ % batch_roi_nonempty_masks.shape[0]
                    batch_roi_nonempty_masks = paddle.scatter(batch_roi_nonempty_masks, pts_roi_inds_, paddle.ones([pts_roi_inds_.shape[0]], dtype='int32'))
            batch_roi_nonempty_masks = batch_roi_nonempty_masks.unsqueeze(
                -1)
            
        batch_rois_feats, pts_feats = self.dynamic_pointnet(
            pts_xyz, pts_feats, pts_info, pts_roi_inds, batch_rois, batch_roi_nonempty_masks
        )
        if self.use_bev_features:
            bev_infos, bev_roi_inds = self.batch_roi_bev_pooling(
                bev_features, batch_rois, roi_batch_ids)
            batch_roi_bev_nonempty_masks = paddle.zeros(
                [batch_rois.shape[0]], dtype=batch_rois.dtype)
            batch_roi_bev_nonempty_masks[bev_roi_inds[bev_roi_inds >= 0]] = 1.0
            batch_roi_bev_nonempty_masks = batch_roi_bev_nonempty_masks.unsqueeze(
                -1)
            bev_pts_feats, batch_rois_bev_feats = self.bev_encoder(
                bev_infos, bev_roi_inds, None, batch_rois, batch_roi_bev_nonempty_masks)
            batch_rois_feats = paddle.concat(
                [batch_rois_feats, batch_rois_bev_feats], axis=1)
        batch_bev_feats = self.batch_roi_grid_pooling(
            pts_rpn_feats,
            batch_rois,
            batch_roi_nonempty_masks,
            roi_batch_ids,
            batch_roi_labels,
            batch_roi_scores,
        )  # (num_rois, num_feats)
        if batch_bev_feats.shape[0] > 0:
            batch_bev_feats = self.roi_net(batch_bev_feats)
            batch_rois_feats = paddle.concat(
                [batch_rois_feats, batch_bev_feats], axis=-1)  # (num_rois, num_feat)
        rcnn_bev_cls_preds = None
        rcnn_bev_reg_preds = None
        if self.training and self.loss_bev_cls:
            rcnn_bev_cls_preds = self.bev_cls(batch_bev_feats)
        if self.training and self.loss_bev_bbox:
            rcnn_bev_reg_preds = self.bev_reg(batch_bev_feats)

        rcnn_cls_preds, centers, sizes, headings, rcnn_iou_preds, rcnn_dir_preds = self.point_head(
            batch_rois_feats)
        # (batch_size, num_rois, 7)
        rcnn_reg_preds = cat([centers, sizes, headings], axis=-1)
        return (
            rcnn_cls_preds,
            rcnn_reg_preds,
            rcnn_iou_preds,
            rcnn_dir_preds,
            rcnn_bev_cls_preds,
            rcnn_bev_reg_preds,
            batch_rois,
            batch_roi_labels,
            batch_roi_scores,
            roi_batch_ids,
            pts_xyz,
            new_pts_batch_ids,
            batch_roi_nonempty_masks.squeeze(-1)
        )

    def forward_onnx(
        self,
        batch_rois,
        batch_roi_nonempty_masks,
        pts_xyz,
        pts_feats,
        pts_info,
        pts_roi_inds,
        pts_rpn_feats0,
    ):

        batch_rois_feats, _ = self.dynamic_pointnet(
            pts_xyz, pts_feats, pts_info, pts_roi_inds, batch_rois, batch_roi_nonempty_masks
        )
        batch_bev_feats = self.batch_roi_grid_pooling(
            [pts_rpn_feats0],
            batch_rois,
            batch_roi_nonempty_masks,
            roi_batch_ids=paddle.zeros([batch_rois.shape[0]], dtype=batch_rois.dtype)
        )  # (num_rois, num_feats)
        batch_bev_feats = self.roi_net(batch_bev_feats)
        batch_rois_feats = paddle.concat(
            [batch_rois_feats, batch_bev_feats], axis=-1)  # (num_rois, num_feat)

        rcnn_cls_preds, centers, sizes, headings, rcnn_iou_preds, rcnn_dir_preds = self.point_head(
            batch_rois_feats)
        # (batch_size, num_rois, 7)
        rcnn_reg_preds = paddle.concat([centers, sizes, headings], axis=-1)
        rcnn_box_preds, rcnn_score_preds, rcnn_label_preds, rcnn_box_overlaps, rcnn_sorted_idx = \
            self.get_bboxes_onnx(rcnn_cls_preds, rcnn_reg_preds,
                                 rcnn_iou_preds, batch_rois, batch_roi_nonempty_masks)
        return rcnn_box_preds, rcnn_score_preds, rcnn_label_preds, rcnn_box_overlaps, rcnn_sorted_idx

    def get_bboxes(
        self,
        rcnn_cls_preds,
        rcnn_reg_preds,
        rcnn_iou_preds,
        rcnn_dir_preds,
        batch_rois,
        roi_batch_ids,
        batch_roi_nonempty_masks,
        test_cfg,
        batch_roi_scores=None,
        batch_roi_labels=None,
    ):
        batch_size = roi_batch_ids.max().item() + 1
        post_center_range = test_cfg['post_center_limit_range']
        batch_roi_nonempty_masks = batch_roi_nonempty_masks > 0
        rcnn_box_preds = self.decode_pred_bboxes(rcnn_reg_preds, batch_rois)
        if self.reg_global:
            rcnn_box_preds = self.decode(batch_rois, rcnn_reg_preds)
        if self.use_direction_classifier:
            dir_labels = paddle.max(rcnn_dir_preds, axis=-1)[1]
            opp_labels = (
                (rcnn_box_preds[..., -1]) > 0
            ) ^ dir_labels.bool()
            rcnn_box_preds[..., -1] += paddle.where(
                opp_labels,
                paddle.full(opp_labels.shape, np.pi).cast(rcnn_box_preds.dtype),
                paddle.full(opp_labels.shape, 0.0).cast(rcnn_box_preds.dtype),
            )
        if self.use_proposal_box:
            rcnn_box_preds = batch_rois

        if len(post_center_range) > 0:
            post_center_range = paddle.to_tensor(
                post_center_range,
                dtype=rcnn_reg_preds.dtype,
            )
        predictions_list = []
        for i in range(batch_size):
            batch_mask = roi_batch_ids == i
            box_preds = rcnn_box_preds[batch_mask]  # (num_rois, 7)
            nonempty_mask = batch_roi_nonempty_masks[batch_mask]
            cls_preds = rcnn_cls_preds[batch_mask]
            iou_preds = rcnn_iou_preds[batch_mask]
            if cls_preds.shape[0] == 0:
                dtype = box_preds.dtype
                predictions_dict = {
                    "box3d_lidar": paddle.zeros([0, self.box_n_dim], dtype=dtype),
                    "scores": paddle.zeros([0], dtype=dtype),
                    "label_preds": paddle.zeros([0], dtype=dtype),
                }
                predictions_list.append(predictions_dict)
                continue

            if self.use_empty_mask:
                box_preds = box_preds[nonempty_mask]
                cls_preds = cls_preds[nonempty_mask]
                iou_preds = iou_preds[nonempty_mask]
            if cls_preds.shape[0] == 0:
                dtype = box_preds.dtype
                predictions_dict = {
                    "box3d_lidar": paddle.zeros([0, self.box_n_dim], dtype=dtype),
                    "scores": paddle.zeros([0], dtype=dtype),
                    "label_preds": paddle.zeros([0], dtype=dtype),
                }
                predictions_list.append(predictions_dict)
                continue

            score_preds = paddle.nn.functional.sigmoid(iou_preds).squeeze(-1)
            cls_scores = F.softmax(cls_preds, axis=-1)
            cls_score_preds = cls_scores.max(axis=-1)
            label_preds = cls_scores.argmax(axis=-1)
            if self.use_proposal_label:
                assert not self.use_empty_mask
                assert batch_roi_labels is not None
                assert batch_roi_scores is not None
                label_preds = batch_roi_labels[batch_mask]
                score_preds = batch_roi_scores[batch_mask]
                if self.use_empty_mask:
                    label_preds = label_preds[nonempty_mask]
                    score_preds = score_preds[nonempty_mask]
                assert label_preds.shape[0] == box_preds.shape[0]
            if test_cfg['score_threshold'] > 0.0:
                thresh = paddle.to_tensor([test_cfg['score_threshold']]).cast(score_preds.dtype)
                scores_keep = score_preds >= thresh
                score_preds = score_preds[scores_keep]

            if score_preds.shape[0] != 0:
                if test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[scores_keep]
                    label_preds = label_preds[scores_keep]
                selected_boxes, selected_scores, selected_labels = self.nms_for_groups(
                    test_cfg, box_preds, score_preds, label_preds
                )  # n
            else:
                selected = []
                # if selected is not None:
                selected_boxes = box_preds[selected]
                selected_labels = label_preds[selected]
                selected_scores = score_preds[selected]

            if selected_boxes.shape[0] != 0:
                selected_boxes, selected_scores, selected_labels = self.nms_overlap_for_groups(
                    test_cfg, selected_boxes, selected_scores, selected_labels
                )  # n

            if selected_boxes.shape[0] != 0:
                final_box_preds = selected_boxes
                final_scores = selected_scores
                final_labels = selected_labels

                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": final_labels[mask],
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": final_labels,
                    }
            else:
                dtype = box_preds.dtype
                predictions_dict = {
                    "box3d_lidar": paddle.zeros([0, self.box_n_dim], dtype=dtype),
                    "scores": paddle.zeros([0], dtype=dtype),
                    "label_preds": paddle.zeros([0], dtype='int64'),
                }

            predictions_list.append(predictions_dict)
        if 'accessory_main' in self.ori_class_names:
            for i, ret in enumerate(predictions_list):
                if ret['box3d_lidar'].shape[0] != 0:
                    bigmot_inds = ret['label_preds'] == self.ori_class_name_map['bigMot']
                    head_inds = ret['label_preds'] == self.ori_class_name_map['accessory_main']

                    bigmot_boxes = ret['box3d_lidar'][bigmot_inds]
                    head_boxes = ret['box3d_lidar'][head_inds]
                    head_scores = ret['scores'][head_inds]

                    if bigmot_boxes.shape[0] == 0 or head_boxes.shape[0] == 0:
                        selected_combo_uid = None
                        ret['combo_uid'] = paddle.zeros(
                            [0], dtype='int64')
                    else:
                        head_scores_matrix = head_scores.unsqueeze(0).repeat_interleave(
                            bigmot_boxes.shape[0], 0)

                        bev_iom = iou3d_utils.boxes_iom_bev(
                            bigmot_boxes, head_boxes)

                        neg_head_inds = bev_iom <= 0.7
                        bev_iom = paddle.where(neg_head_inds, 
                                      -1 * paddle.ones_like(bev_iom), 
                                      bev_iom)

                        head_scores_matrix = paddle.where(neg_head_inds, 
                                      -1 * paddle.ones_like(head_scores_matrix), 
                                      head_scores_matrix)

                        neg_head_inds = head_scores_matrix <= 0.2
                        head_scores_matrix = paddle.where(neg_head_inds, 
                                      -1 * paddle.ones_like(head_scores_matrix), 
                                      head_scores_matrix)

                        selected_combo_uid = head_scores_matrix.argmax(1)
                        selected_combo_uid = paddle.where(head_scores_matrix.max(1) < 0, 
                                            -1 * paddle.ones_like(selected_combo_uid),
                                            selected_combo_uid)
                        ret['combo_uid'] = selected_combo_uid
                else:
                    ret['combo_uid'] = paddle.zeros(
                        [0], dtype='int64')

        bboxes = []
        scores = []
        labels = []
        combo_uids = []
        for i, ret in enumerate(predictions_list):
            bboxes.append(ret["box3d_lidar"])
            scores.append(ret["scores"])
            labels.append(ret["label_preds"])
            if 'accessory_main' in self.ori_class_name_map:
                combo_uids.append(ret['combo_uid'])

        bboxes = paddle.concat(bboxes, axis=0)
        scores = paddle.concat(scores, axis=0)
        labels = paddle.concat(labels, axis=0)
        if 'accessory_main' in self.ori_class_name_map:
            combo_uids = cat(combo_uids, axis=0)
        else:
            combo_uids = paddle.zeros(
                [0], dtype='int64')

        return [[bboxes, scores, labels, combo_uids]]

    def get_bboxes_onnx(
        self,
        rcnn_cls_preds,
        rcnn_reg_preds,
        rcnn_iou_preds,
        batch_rois,
        batch_roi_nonempty_masks=None
    ):
        rcnn_score_preds = paddle.nn.functional.sigmoid(rcnn_iou_preds)
        rcnn_label_preds = rcnn_cls_preds.max(axis=-1, keepdim=True)[1]
        rcnn_box_preds = self.decode_pred_bboxes(rcnn_reg_preds, batch_rois)
        rcnn_box_preds[:, 2] = rcnn_box_preds[:, 2] + \
            rcnn_box_preds[:, 5] * 0.5
        rcnn_box_overlaps = self.rotate_overlaps(rcnn_box_preds)
        rcnn_sorted_idx = paddle.topk(
            rcnn_score_preds, k=rcnn_score_preds.shape[0], axis=0, largest=True, sorted=True)[1].squeeze(1)
        return rcnn_score_preds, rcnn_label_preds, rcnn_box_preds, rcnn_box_overlaps, rcnn_sorted_idx

    def loss(
        self,
        rcnn_cls_preds,
        rcnn_reg_preds,
        rcnn_iou_preds,
        rcnn_dir_preds,
        rcnn_bev_cls_preds,
        rcnn_bev_reg_preds,
        batch_rois,
        roi_batch_ids,
        pts,
        pts_batch_ids,
        batch_roi_nonempty_masks,
        gt_boxes,
        gt_classes,
    ):
        batch_roi_empty_masks = batch_roi_nonempty_masks <= 0
        ret = {}
        dtype = rcnn_cls_preds.dtype
        if roi_batch_ids.shape[0] > 0:
            batch_size = roi_batch_ids.max().item() + 1
        else:
            batch_size = 1
        gt_boxes_ = [gt_box.clone()
                     for gt_box in gt_boxes]  # 底面中心
        gt_classes_ = []
        for i, gt_class in enumerate(gt_classes):
            gt_class_ = gt_class.clone()
            gt_class_[gt_class_ == 5] = 2  # 合并verybigmot和bigmot
            gt_class_[gt_class_ == 7] = 5  # 重映射accessory_main 7-->5
            gt_classes_.append(gt_class_)
        del gt_boxes
        del gt_classes

        if rcnn_cls_preds.shape[0] == 0:
            ret["rcnn_loss"] = paddle.zeros([1], dtype=dtype)
            ret["rcnn_loss"].stop_gradient = False
            ret["rcnn_cls_pos_loss"] = paddle.zeros([1], dtype=dtype)
            ret["rcnn_cls_neg_loss"] = paddle.zeros([1], dtype=dtype)

            ret["rcnn_cls_loss"] = paddle.zeros([1], dtype=dtype)
            ret["rcnn_reg_loss"] = paddle.zeros([1], dtype=dtype)
            ret["rcnn_reg_num"] = paddle.zeros([1], dtype=dtype)
            ret["rcnn_num_pos"] = paddle.zeros([1], dtype=dtype)
            ret["rcnn_num_ignore"] = paddle.zeros([1], dtype=dtype)
            ret["rcnn_iou_loss_aware"] = paddle.zeros([1], dtype=dtype)
            if self.use_corner_loss:
                ret["rcnn_corner_loss"] = paddle.zeros([1], dtype=dtype)
            if self.loss_pred_bctps:
                ret["rcnn_pred_bctps_loss"] = paddle.zeros([1], dtype=dtype)
            if self.loss_bev_cls:
                ret["rcnn_bev_cls_loss"] = paddle.zeros([1], dtype=dtype)
                ret["rcnn_bev_cls_pos_loss"] = paddle.zeros([1], dtype=dtype)
                ret["rcnn_bev_cls_neg_loss"] = paddle.zeros([1], dtype=dtype)
            if self.loss_bev_bbox:
                ret["rcnn_bev_reg_loss"] = paddle.zeros([1], dtype=dtype)
                # ret["rcnn_bev_reg_elem_loss"] = reg_elem_loss
            if self.loss_iou:
                ret["rcnn_iou_loss"] = paddle.zeros([1], dtype=dtype)
            return ret

        rcnn_box_preds = self.decode_pred_bboxes(rcnn_reg_preds, batch_rois)
        if self.reg_global:
            rcnn_box_preds = self.decode(batch_rois, rcnn_reg_preds)
        assign_output = self.assign_targets(
            pts, pts_batch_ids, batch_rois, rcnn_box_preds, roi_batch_ids, gt_boxes_, gt_classes_
        )
        del pts
        del pts_batch_ids
        (
            batch_roi_ious,
            batch_roi_rcnn_labels,
            batch_roi_gt_boxes,
            batch_roi_gt_classes,
            batch_roi_border_masks,
            batch_roi_reg_mask,
            batch_roi_soft_labels,
            batch_roi_sampled_mask,
            batch_roi_gt_boxes_src,
        ) = assign_output

        centerness_weight = self.get_centerness_weight(
            batch_rois) if self.use_centerness_weight else 1
        if self.center_only_fg:
            # 提升远距离正样本的权重
            centerness_weight = paddle.where(batch_roi_rcnn_labels < 0, 
                                paddle.ones_like(centerness_weight), centerness_weight)
        # cls loss
        # 没有点的proposal不参与loss计算，间接减少副样本
        batch_roi_rcnn_labels = paddle.where(batch_roi_empty_masks, 
                                            -2 * paddle.ones_like(batch_roi_rcnn_labels),
                                            batch_roi_rcnn_labels)
        batch_roi_reg_mask = paddle.where(batch_roi_empty_masks, 
                            paddle.zeros(batch_roi_reg_mask.shape, dtype='int64'), 
                            batch_roi_reg_mask.cast('int64')).cast('bool')
        batch_roi_sampled_mask = paddle.where(batch_roi_empty_masks, 
                            paddle.zeros(batch_roi_reg_mask.shape, dtype='int64'), 
                            batch_roi_sampled_mask.cast('int64')).cast('bool')
        cls_valid_mask = batch_roi_rcnn_labels >= 0  # 只会正样本计算分类损失
        cls_one_hot_target = F.one_hot(
            (batch_roi_rcnn_labels * cls_valid_mask).cast('int64'), num_classes=len(self.ori_class_names)
        )
        cls_weights = cls_valid_mask.cast('float32') * centerness_weight
        batch_loss_cls = self.loss_cls(
            rcnn_cls_preds, cls_one_hot_target, weights=cls_weights)  # (num_rois)
        rcnn_loss_cls = batch_loss_cls.sum() / paddle.clip(cls_valid_mask.sum(), min=1.0)
        rcnn_loss_cls = rcnn_loss_cls * self.loss_cls._loss_weight
        cls_pos_loss, cls_neg_loss = self.get_pos_neg_loss(
            batch_loss_cls, batch_roi_rcnn_labels)
        num_pos = (batch_roi_rcnn_labels >= 0).sum() / batch_size
        num_neg = (batch_roi_rcnn_labels == -1).sum() / batch_size
        num_ignore = (batch_roi_rcnn_labels == -2).sum() / batch_size

        ret["rcnn_cls_loss"] = rcnn_loss_cls
        ret["rcnn_cls_pos_loss"] = cls_pos_loss
        ret["rcnn_cls_neg_loss"] = cls_neg_loss
        ret["rcnn_num_pos"] = num_pos
        ret["rcnn_num_neg"] = num_neg
        ret["rcnn_num_ignore"] = num_ignore

        # reg loss
        fg_sum = batch_roi_reg_mask.cast('int64').sum()
        reg_weights = batch_roi_reg_mask.cast('float32') * centerness_weight
        reg_targets = self.encode_pred_bboxes(batch_roi_gt_boxes, batch_rois)
        if self.reg_global:
            reg_targets = self.encode(batch_rois, batch_roi_gt_boxes_src)
            rcnn_reg_preds, reg_targets = self.add_sin_difference(
                rcnn_reg_preds, reg_targets)
        batch_loss_reg = self.loss_bbox(
            rcnn_reg_preds, reg_targets, weights=reg_weights)

        rcnn_loss_reg = batch_loss_reg.sum() / max(fg_sum, 1.0) * \
            self.loss_bbox._loss_weight
        loss = rcnn_loss_cls + rcnn_loss_reg
        ret["rcnn_loss"] = loss
        ret["rcnn_reg_loss"] = rcnn_loss_reg
        ret["rcnn_reg_num"] = batch_roi_reg_mask.sum() / batch_size

        # iou aware loss
        batch_loss_iou = F.binary_cross_entropy_with_logits(
            rcnn_iou_preds, batch_roi_soft_labels.unsqueeze(-1), reduction="none"
        )
        batch_roi_gt_classes_ = batch_roi_gt_classes % self.iou_cls_weight.shape[0]
        iou_cls_weight = self.iou_cls_weight.index_select(batch_roi_gt_classes_, axis=0)
        iou_cls_weight = paddle.where(batch_roi_gt_classes< 0, paddle.ones_like(iou_cls_weight), iou_cls_weight)
        batch_loss_iou = batch_roi_sampled_mask.cast(batch_loss_iou.dtype).unsqueeze(
            -1) * batch_loss_iou * iou_cls_weight.unsqueeze(-1)
        rcnn_loss_iou = (
            batch_loss_iou.sum()
            / max(batch_roi_sampled_mask.sum().cast('float32'), 1.0)
            * self.loss_iou_aware._loss_weight
        )
        ret["rcnn_iou_loss_aware"] = rcnn_loss_iou
        ret["rcnn_loss"] = ret["rcnn_loss"] + rcnn_loss_iou

        if self.use_corner_loss:
            # 根据类别设置corner_loss权重
            batch_roi_gt_classes_ = batch_roi_gt_classes % self.corner_loss_weight.shape[0]
            corner_weights = self.corner_loss_weight.index_select(batch_roi_gt_classes_, axis=0)
            corner_weights = paddle.where(batch_roi_gt_classes< 0, paddle.ones_like(corner_weights), corner_weights)
            corner_weights = corner_weights * batch_roi_reg_mask.cast('float32')
            corner_loss = self.get_corner_loss(
                rcnn_box_preds, batch_roi_gt_boxes_src, weights=corner_weights)
            corner_loss = corner_loss.sum() / max(fg_sum, 1.0)
            ret["rcnn_corner_loss"] = corner_loss
            ret["rcnn_loss"] = ret["rcnn_loss"] + corner_loss

        if self.loss_pred_bctps: 
            batch_roi_border_masks = paddle.stack([batch_roi_border_masks, batch_roi_border_masks], 2).reshape(
                [batch_roi_border_masks.shape[0], 8]
            )
            reg_weights_ = reg_weights.unsqueeze(-1).repeat_interleave(8, 1)
            batch_roi_border_masks *= reg_weights_

            batch_bev_gts = paddle.concat([batch_roi_gt_boxes_src[:, 0:2], 
                                            batch_roi_gt_boxes_src[:, 3:5], 
                                            batch_roi_gt_boxes_src[:, 6:7]], axis=1)
            
            batch_bev_preds = paddle.concat([rcnn_box_preds[:, 0:2], 
                                            rcnn_box_preds[:, 3:5], 
                                            rcnn_box_preds[:, 6:7]], axis=1)
            bev_gt_corners = get_bev_corners_paddle(batch_bev_gts)
            bev_pred_corners = get_bev_corners_paddle(batch_bev_preds)

            bev_gt_corners_cat = paddle.concat(
                [bev_gt_corners, bev_gt_corners[:, :1, :]], 1)
            bev_pred_corners_cat = paddle.concat(
                [bev_pred_corners, bev_pred_corners[:, :1, :]], 1)

            bev_gt_bctps = (
                bev_gt_corners_cat[:, :-1, :] + bev_gt_corners_cat[:, 1:, :]) / 2
            bev_pred_bctps = (
                bev_pred_corners_cat[:, :-1, :] + bev_pred_corners_cat[:, 1:, :]) / 2

            theta_diff = paddle.abs(batch_bev_gts[:, 4] - batch_bev_preds[:, 4])
            la_pi_inds = theta_diff > np.pi
            theta_diff[la_pi_inds] = 2 * np.pi - theta_diff[la_pi_inds]
            la_half_pi_inds = theta_diff > (np.pi / 2)
            le_half_pi_inds = theta_diff <= (np.pi / 2)
            bev_pred_bctps_ = paddle.zeros_like(bev_pred_bctps)
            bev_pred_bctps_tmp = paddle.concat([bev_pred_bctps[:, 2:3, :],
                                                bev_pred_bctps[:, 3:4, :],
                                                bev_pred_bctps[:, 0:1, :],
                                                bev_pred_bctps[:, 1:2, :],], 1)
            bev_pred_bctps_ = paddle.where(la_half_pi_inds.unsqueeze(-1).unsqueeze(-1), bev_pred_bctps_tmp, bev_pred_bctps_)
            bev_pred_bctps_ = paddle.where(le_half_pi_inds.unsqueeze(-1).unsqueeze(-1), bev_pred_bctps, bev_pred_bctps_)
            
            bev_gt_bctps = bev_gt_bctps.reshape([-1, 8])
            bev_pred_bctps = bev_pred_bctps_.reshape([-1, 8])

            pred_bctps_loss = self.loss_pred_bctps(
                bev_pred_bctps, bev_gt_bctps, weights=batch_roi_border_masks
            )
            pred_bctps_loss_reduced = (
                pred_bctps_loss.sum() / max(fg_sum, 1.0) * self.loss_pred_bctps._loss_weight
            )
            ret["rcnn_pred_bctps_loss"] = pred_bctps_loss_reduced
            ret["rcnn_loss"] = ret["rcnn_loss"] + pred_bctps_loss_reduced

        if self.loss_bev_cls:
            bev_cls_valid_mask = batch_roi_rcnn_labels >= -1  # 背景是-1
            bev_cls_weights = bev_cls_valid_mask.cast('float32') * centerness_weight
            bev_cls_one_hot_target = F.one_hot(
                ((batch_roi_rcnn_labels + 1) * bev_cls_valid_mask).cast('int64'),
                num_classes=len(self.ori_class_names) + 1,
            )
            batch_bev_loss_cls = self.loss_bev_cls(
                rcnn_bev_cls_preds, bev_cls_one_hot_target, weights=bev_cls_weights
            )  # (num_rois)
            rcnn_loss_bev_cls = batch_bev_loss_cls.sum(
            ) / paddle.clip(bev_cls_valid_mask.sum(), min=1.0)
            rcnn_loss_bev_cls = rcnn_loss_bev_cls * self.loss_bev_cls._loss_weight
            cls_pos_loss, cls_neg_loss = self.get_pos_neg_loss(
                batch_bev_loss_cls, batch_roi_rcnn_labels)
            ret["rcnn_bev_cls_loss"] = rcnn_loss_bev_cls
            ret["rcnn_bev_cls_pos_loss"] = cls_pos_loss
            ret["rcnn_bev_cls_neg_loss"] = cls_neg_loss
            ret["rcnn_loss"] = ret["rcnn_loss"] + rcnn_loss_bev_cls
        if self.loss_bev_bbox:
            batch_loss_bev_reg = self.loss_bev_bbox(
                rcnn_bev_reg_preds, reg_targets, weights=reg_weights)
            rcnn_loss_bev_reg = batch_loss_bev_reg.sum() / max(fg_sum, 1.0) * \
                self.loss_bev_bbox._loss_weight
            # reg_elem_loss_bev = batch_loss_bev_reg.sum(axis=0) / max(fg_sum, 1.0)

            ret["rcnn_loss"] = ret["rcnn_loss"] + rcnn_loss_bev_reg
            ret["rcnn_bev_reg_loss"] = rcnn_loss_bev_reg
            # ret["rcnn_bev_reg_elem_loss"] = reg_elem_loss_bev

        if self.loss_iou:
            batch_loss_iou_ = self.loss_iou(
                rcnn_reg_preds, reg_targets, weights=reg_weights)
            rcnn_loss_iou_ = batch_loss_iou_.sum() / max(fg_sum, 1.0) * \
                self.loss_iou._loss_weight
            ret["rcnn_iou_loss"] = rcnn_loss_iou_
            ret["rcnn_loss"] = ret["rcnn_loss"] + rcnn_loss_iou_
        if self.use_direction_classifier:
            dir_targets = self.get_direction_target(
                batch_roi_gt_boxes_src
            )
            dir_logits = rcnn_dir_preds
            # weights = (labels > 0).type_as(dir_logits)
            # weights /= paddle.clip(weights.sum(-1, keepdim=True), min=1.0)
            rcnn_dir_loss = self.loss_aux(
                dir_logits, dir_targets, weights=reg_weights)
            rcnn_dir_loss = rcnn_dir_loss.sum() / max(fg_sum, 1.0) * \
                self.loss_aux._loss_weight
            ret["rcnn_dir_loss"] = rcnn_dir_loss
            ret["rcnn_loss"] = ret["rcnn_loss"] + rcnn_dir_loss
        return ret

    @paddle.no_grad()
    def batch_roi_aware_pooling(self, pts, pts_batch_ids, batch_rois, roi_batch_ids, img_metas):
        pts_xyz = pts[:, :3]
        pts_feats = pts[:, 3:]

        if batch_rois.shape[0] == 0:
            new_pts_xyz = paddle.zeros((0, 3), dtype=pts_xyz.dtype)
            new_pts_feats = paddle.zeros((0, pts_feats.shape[1]), dtype=pts_feats.dtype)
            ext_pts_roi_inds = paddle.full(
                (0,), -1, dtype='int64')
            ext_pts_info = paddle.zeros((0, 10), dtype=pts_feats.dtype)
            new_pts_batch_ids = paddle.full(
                (0,), -1, dtype='int64')
        else:
            batch_rois_ = batch_rois.clone().detach()
            batch_rois_[:, 2] = batch_rois_[:, 2] + \
                batch_rois_[:, 5] * 0.5  # 变换到几何中心
            ext_pts_xyz, ext_pts_feats, ext_pts_info, ext_pts_roi_inds, ext_pts_batch_ids = self.roi_extractor(
                pts, pts_batch_ids, batch_rois_, roi_batch_ids  # batch concat point xyz
            )
            del batch_rois_
            new_pts_feats = ext_pts_feats
            new_pts_xyz = ext_pts_xyz
            new_pts_batch_ids = ext_pts_batch_ids

        return new_pts_xyz, new_pts_feats, ext_pts_info, ext_pts_roi_inds, new_pts_batch_ids

    def batch_roi_bev_pooling(self, bev_features, batch_rois, roi_batch_ids):

        if batch_rois.shape[0] == 0:
            new_bev_info = paddle.zeros(
                (0, bev_features.shape[1] + 9), dtype=bev_features.dtype)
            ext_bev_roi_inds = paddle.full(
                (0,), -1, dtype='int64')
        else:
            new_bev_info, ext_bev_roi_inds = self.bev_roi_extractor(
                bev_features, batch_rois, roi_batch_ids  # batch concat point xyz
            )

        return new_bev_info, ext_bev_roi_inds

    @paddle.no_grad()
    def assign_targets(self, pts, pts_batch_ids, batch_rois, rcnn_box_preds, roi_batch_ids, gt_boxes, gt_classes):
        with paddle.no_grad():
            assign_out = self.proposal_target_layer.forward(
                pts, pts_batch_ids, batch_rois, roi_batch_ids, gt_boxes, gt_classes,
                pred_aware_assign=(self.pred_aware_assign and self.epoch > self.warm_epoch), rcnn_box_preds=rcnn_box_preds
            )
        (
            batch_roi_ious,
            batch_roi_rcnn_labels,
            batch_roi_gt_boxes,
            batch_roi_gt_classes,
            batch_roi_border_masks,
            batch_roi_reg_mask,
            batch_roi_soft_labels,
            batch_roi_sampled_mask,
        ) = assign_out

        num_rois = batch_rois.shape[0]
        batch_roi_gt_boxes_src = batch_roi_gt_boxes.clone().detach()
        if num_rois == 0:
            return (
                batch_roi_ious,
                batch_roi_rcnn_labels,
                batch_roi_gt_boxes,
                batch_roi_gt_classes,
                batch_roi_border_masks,
                batch_roi_reg_mask,
                batch_roi_soft_labels,
                batch_roi_sampled_mask,
                batch_roi_gt_boxes_src,
            )
        # canonical transformation
        roi_center = batch_rois[:, 0:3]
        roi_ry = batch_rois[:, 6] % (2 * np.pi)
        batch_roi_gt_boxes[:, 0:3] = batch_roi_gt_boxes[:, 0:3] - roi_center
        batch_roi_gt_boxes[:, 6] = batch_roi_gt_boxes[:, 6] - roi_ry

        # transfer LiDAR coords to local coords
        batch_roi_gt_boxes[:, :3] = self.rotation_3d_in_axis(
            points=batch_roi_gt_boxes[:, :3].reshape([-1, 1, 3]), angles=-roi_ry.reshape([-1])
        ).reshape([num_rois, 3])

        # flip orientation if rois have opposite orientation
        heading_label = batch_roi_gt_boxes[:, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi *
                         0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (
            2 * np.pi
        )  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = paddle.clip(
            heading_label, min=-np.pi / 2, max=np.pi / 2)
        batch_roi_gt_boxes[:, 6] = heading_label

        return (
            batch_roi_ious,
            batch_roi_rcnn_labels,
            batch_roi_gt_boxes,
            batch_roi_gt_classes,
            batch_roi_border_masks,
            batch_roi_reg_mask,
            batch_roi_soft_labels,
            batch_roi_sampled_mask,
            batch_roi_gt_boxes_src,
        )

    def merge_batch_roi_infos(self, proposal_list):
        batch_size = len(proposal_list)
        # rois, roi_scores, roi_labels,
        batch_rois = cat([ret["rois"] for ret in proposal_list], axis=0)
        batch_roi_scores = cat(
            [ret["roi_scores"] for ret in proposal_list], axis=0)
        batch_roi_labels = cat(
            [ret["roi_labels"] for ret in proposal_list], axis=0)

        roi_batch_ids = []
        for batch_id in range(batch_size):
            ret = proposal_list[batch_id]
            cur_ids = paddle.full(
                (ret["rois"].shape[0],), batch_id, dtype='int64')
            roi_batch_ids.append(cur_ids)
        roi_batch_ids = cat(roi_batch_ids, axis=0)  # (num_rois)

        return batch_rois, batch_roi_labels, batch_roi_scores, roi_batch_ids

    def get_pos_neg_loss(self, cls_loss, labels):
        batch_size = cls_loss.shape[0]
        neg_num = max((labels == -1).sum().item(), 1)
        pos_num = max((labels >= 0).sum().item(), 1)
        cls_pos_loss = (labels >= 0).cast(cls_loss.dtype) * cls_loss
        cls_neg_loss = (labels == -1).cast(cls_loss.dtype) * cls_loss
        cls_pos_loss = cls_pos_loss.sum() / pos_num
        cls_neg_loss = cls_neg_loss.sum() / neg_num

        return cls_pos_loss, cls_neg_loss

    def batch_roi_grid_pooling(
        self, pts_feats, batch_rois, batch_nonempty_masks, roi_batch_ids, batch_roi_labels=None, batch_roi_scores=None,
    ):
        bev_features = pts_feats[0]
        batch_rois_scaled = cat(
            [roi_batch_ids.unsqueeze(1).cast('float32'), batch_rois], axis=1)

        pooled_pooled_features = self.roi_align_rot(
            bev_features, batch_rois_scaled)  # (num_rois, C, h, w)
        num_rois, C, h, w = pooled_pooled_features.shape
        pooled_pooled_features = pooled_pooled_features.reshape(
            [num_rois, C * h * w])
        if self.use_proposal_label_feat:
            pooled_label_features = one_hot_f(
                batch_roi_labels, len(self.ori_class_names))
            pooled_label_features = pooled_label_features * \
                batch_roi_scores.unsqueeze(1)  # (num_rois, 6)
            if self.use_proposal_embedding:
                pooled_label_features = self.score_embed(pooled_label_features)
            pooled_pooled_features = paddle.concat(
                [pooled_pooled_features, pooled_label_features], axis=1)
        if batch_nonempty_masks is not None and pooled_pooled_features.shape[0] > 0:
            pooled_pooled_features = pooled_pooled_features * batch_nonempty_masks
        return pooled_pooled_features

    def get_centerness_weight(self, batch_rois):
        assert batch_rois.shape[1] == 7
        distance = paddle.linalg.norm(batch_rois[:, :2], axis=1)
        centerness = 1 + distance / (self.max_distance) * self.center_scale
        return centerness

    def get_corner_loss(self, pred_bbox3d, gt_bbox3d, weights=None, delta=1):
        """Calculate corner loss of given boxes.

        Args:
            pred_bbox3d (paddle.FloatTensor): Predicted boxes in shape (N, 7).
            gt_bbox3d (paddle.FloatTensor): Ground truth boxes in shape (N, 7).

        Returns:
            paddle.FloatTensor: Calculated corner loss in shape (N).
        """
        assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

        # This is a little bit hack here because we assume the box for
        # Part-A2 is in LiDAR coordinates
        gt_box_corners = center_to_corner_box3d(
            gt_bbox3d[:, :3], gt_bbox3d[:, 3:6], gt_bbox3d[:, 6], origin=(0.5, 0.5, 0.5), axis=2
        )
        pred_box_corners = center_to_corner_box3d(
            pred_bbox3d[:, :3], pred_bbox3d[:, 3:6], pred_bbox3d[:, 6], origin=(0.5, 0.5, 0.5), axis=2
        )

        # This flip only changes the heading direction of GT boxes
        gt_bbox3d_flip = gt_bbox3d.clone()
        gt_bbox3d_flip[:, 6] += np.pi
        gt_box_corners_flip = center_to_corner_box3d(
            gt_bbox3d_flip[:, :3],
            gt_bbox3d_flip[:, 3:6],
            gt_bbox3d_flip[:, 6],
            origin=(0.5, 0.5, 0.5),
            axis=2,
        )

        corner_dist = paddle.minimum(
            paddle.linalg.norm(pred_box_corners - gt_box_corners, axis=2),
            paddle.linalg.norm(pred_box_corners - gt_box_corners_flip, axis=2),
        )  # (N, 8)
        # huber loss
        abs_error = paddle.abs(corner_dist)
        quadratic = paddle.clip(abs_error, max=delta)
        linear = abs_error - quadratic
        corner_loss = 0.5 * quadratic**2 + delta * linear
        corner_loss = corner_loss * weights.unsqueeze(1)
        corner_loss = corner_loss.mean(axis=1)
        return corner_loss

    def nms_for_groups(
        self,
        test_cfg,
        box3d_lidar,
        scores,
        label_preds,
    ):
        nms_groups = test_cfg['nms'].get("nms_groups", [])
        if not isinstance(nms_groups[0][0], int):
            for i, group in enumerate(nms_groups):
                for j, name in enumerate(group):
                    nms_groups[i][j] = self.ori_class_name_map[name]

        group_nms_pre_max_size = test_cfg['nms'].get("group_nms_pre_max_size", [])
        group_nms_post_max_size = test_cfg['nms'].get(
            "group_nms_post_max_size", [])
        group_nms_iou_threshold = test_cfg['nms'].get(
            "group_nms_iou_threshold", [])
        add_iou_edge = test_cfg['nms'].get("add_iou_edge", 0.0)
        if len(group_nms_pre_max_size) == 0:
            group_nms_pre_max_size = [
                test_cfg['nms']['nms_pre_max_size']] * len(nms_groups)
        if len(group_nms_post_max_size) == 0:
            group_nms_post_max_size = [
                test_cfg['nms']['nms_post_max_size']] * len(nms_groups)
        if len(group_nms_iou_threshold) == 0:
            group_nms_iou_threshold = [
                test_cfg['nms']['nms_iou_threshold']] * len(nms_groups)

        assert len(group_nms_pre_max_size) == len(nms_groups)
        assert len(group_nms_post_max_size) == len(nms_groups)
        assert len(group_nms_iou_threshold) == len(nms_groups)

        nms_func = nms

        boxes_for_nms = paddle.concat([box3d_lidar[:, 0:2], box3d_lidar[:, 3:5], box3d_lidar[:, -1:]], axis=1)
        if not test_cfg['nms']['use_rotate_nms']:
            box_preds_corners = center_to_corner_box2d(
                boxes_for_nms[:, :2],
                boxes_for_nms[:, 2:4],
                boxes_for_nms[:, 4],
            )
            boxes_for_nms = corner_to_standup_nd(
                box_preds_corners)

        for group_id, nms_group in enumerate(nms_groups):
            selecteds = label_preds >= 0
            if len(nms_group) == 0:
                continue
            mask = label_preds == nms_group[0]
            for label_id in nms_group:
                mask |= label_preds == label_id
            indices = paddle.where(mask)[0]

            if indices.shape[0] != 0:
                group_boxes_for_nms = paddle.index_select(boxes_for_nms, indices, axis=0)
                group_scores = paddle.index_select(scores, indices, axis=0)
                # the nms in 3d detection just remove overlap boxes.
                selected = nms_func(
                    group_boxes_for_nms,
                    group_scores,
                    pre_max_size=group_nms_pre_max_size[group_id],
                    post_max_size=group_nms_post_max_size[group_id],
                    iou_threshold=group_nms_iou_threshold[group_id],
                    add_iou_edge=add_iou_edge,
                )
                selected_indices = indices[selected]
                selecteds[indices] = False
                selecteds[selected_indices] = True

                boxes_for_nms = boxes_for_nms[selecteds]
                box3d_lidar = box3d_lidar[selecteds]
                label_preds = label_preds[selecteds]
                scores = scores[selecteds]

        return box3d_lidar, scores, label_preds

    @paddle.no_grad()
    def downsample_points(self, pts, pts_batch_ids):
        """Apply dynamic voxelization to points.

        Args:
            points (list[paddle.Tensor]): Points of each sample.

        Returns:
            tuple[paddle.Tensor]: Concatenated points and coordinates.
        """
        points = pts
        batch_ids = pts_batch_ids
        # coords = example['coordinates']
        new_points = paddle.concat([batch_ids.unsqueeze(1), points], axis=-1)

        ds_new_points = self.voxel_downsamplev2(new_points)
        ds_points = ds_new_points[:, 1:]
        ds_batch_ids = ds_new_points[:, 0]

        return ds_points, ds_batch_ids

    def decode_pred_bboxes(self, rcnn_reg_preds, rois):
        roi_ry = rois[:, 6].reshape([-1])
        roi_xyz = rois[:, 0:3].reshape([-1, 3])
        local_rois = rois.clone().detach()
        local_rois[:, 0:3] = 0
        rcnn_box_preds = self.box_coder.decode(local_rois, rcnn_reg_preds)
        rcnn_center_preds = self.rotation_3d_in_axis(
            rcnn_box_preds[:, :3].unsqueeze(axis=1), angles=roi_ry
        ).squeeze(axis=1)
        rcnn_center_preds = rcnn_center_preds + roi_xyz
        rcnn_box_preds = paddle.concat(
            [rcnn_center_preds, rcnn_box_preds[:, 3:]], axis=-1)
        return rcnn_box_preds

    def rotate_overlaps(self, rcnn_box_preds):
        rotate_overlaps = iou3d_utils.boxes_iou_bev(rcnn_box_preds, rcnn_box_preds)
        return paddle.stack([rotate_overlaps, rotate_overlaps], axis=0)  # 2*n*n

    def encode_pred_bboxes(self, rcnn_gt_boxes, rois):
        rois_anchor = rois.clone().detach()
        rois_anchor[:, 0:3] = 0
        rois_anchor[:, 6] = 0
        reg_targets = self.box_coder.encode(rois_anchor, rcnn_gt_boxes)
        return reg_targets

    def rotation_3d_in_axis(self, points, angles):
        # points: [N, point_size, 3]
        # angles: [N]
        rot_sin = paddle.sin(angles)
        rot_cos = paddle.cos(angles)
        ones = paddle.ones_like(rot_cos)
        zeros = paddle.zeros_like(rot_cos)
        rot_mat_T_ = paddle.stack(
            [
                paddle.stack([rot_cos, rot_sin, zeros], axis=-1),
                paddle.stack([-rot_sin, rot_cos, zeros], axis=-1),
                paddle.stack([zeros, zeros, ones], axis=-1),
            ],
            axis=-1
        )
        return paddle.matmul(points, rot_mat_T_)

    def nms_overlap_for_groups(
        self,
        test_cfg,
        box3d_lidar,
        scores,
        label_preds
    ):

        nms_overlap_groups = test_cfg['nms'].get('nms_overlap_groups', [])

        if len(nms_overlap_groups) != 0:
            # change class names in nms_group to id
            if not isinstance(nms_overlap_groups[0][0], int):
                class_names = []
                for names in self.class_names:
                    class_names.extend(names)
                # class2id = {token: i for i, token in enumerate(class_names)}
                for i, group in enumerate(nms_overlap_groups):
                    for j, name in enumerate(group):
                        nms_overlap_groups[i][j] = self.ori_class_name_map[name]
        group_nms_overlap_pre_max_size = test_cfg['nms'].get(
            'group_nms_overlap_pre_max_size', [])
        group_nms_overlap_post_max_size = test_cfg['nms'].get(
            'group_nms_overlap_post_max_size', [])
        group_nms_overlap_iou_threshold = test_cfg['nms'].get(
            'group_nms_overlap_iou_threshold', [])
        if len(group_nms_overlap_pre_max_size) == 0:
            group_nms_overlap_pre_max_size = [
                test_cfg['nms']['nms_pre_max_size']] * len(nms_overlap_groups)
        if len(group_nms_overlap_post_max_size) == 0:
            group_nms_overlap_post_max_size = [
                test_cfg['nms']['nms_post_max_size']] * len(nms_overlap_groups)
        if len(group_nms_overlap_iou_threshold) == 0:
            group_nms_overlap_iou_threshold = [
                test_cfg['nms']['nms_iou_threshold']] * len(nms_overlap_groups)

        assert len(group_nms_overlap_pre_max_size) == len(nms_overlap_groups)
        assert len(group_nms_overlap_post_max_size) == len(nms_overlap_groups)
        assert len(group_nms_overlap_iou_threshold) == len(nms_overlap_groups)

        if (not hasattr(test_cfg['nms'], "use_rotate_nms_overlap")) or (not test_cfg['nms']['use_rotate_nms_overlap']):
            nms_overlap_func = nms_overlap
        else:
            nms_overlap_func = rotate_nms_overlap

        boxes_for_nms = paddle.concat([box3d_lidar[:, 0:2], box3d_lidar[:, 3:5], box3d_lidar[:, -1:]], axis=1)

        if (not hasattr(test_cfg['nms'], "use_rotate_nms_overlap")) or (not test_cfg['nms']['use_rotate_nms_overlap']):
            box_preds_corners = center_to_corner_box2d(
                boxes_for_nms[:, :2],
                boxes_for_nms[:, 2:4],
                boxes_for_nms[:, 4],
            )
            boxes_for_nms = corner_to_standup_nd(
                box_preds_corners
            )

        for group_id, nms_group in enumerate(nms_overlap_groups):
            selecteds = label_preds >= 0
            if len(nms_group) == 0:
                continue
            mask = label_preds == nms_group[0]
            for label_id in nms_group:
                mask |= label_preds == label_id
            indices = paddle.where(mask)[0]

            if indices.shape[0] != 0:
                group_boxes_for_nms = paddle.index_select(boxes_for_nms, indices, axis=0)
                group_scores = paddle.index_select(scores, indices, axis=0)

                # the nms in 3d detection just remove overlap boxes.
                selected = nms_overlap_func(
                    group_boxes_for_nms,
                    group_scores,
                    pre_max_size=group_nms_overlap_pre_max_size[group_id],
                    post_max_size=group_nms_overlap_post_max_size[group_id],
                    overlap_threshold=group_nms_overlap_iou_threshold[group_id],
                )
                selected_indices = indices[selected]
                selecteds[indices] = False
                selecteds[selected_indices] = True

                boxes_for_nms = boxes_for_nms[selecteds]
                box3d_lidar = box3d_lidar[selecteds]
                label_preds = label_preds[selecteds]
                scores = scores[selecteds]
        return box3d_lidar, scores, label_preds

    def decode(self, anchors, deltas):
        """Apply transformation `deltas` (dx, dy, dz, dw, dh, dl, dr, dv*) to
        `boxes`.

        Args:
            anchors (paddle.Tensor): Parameters of anchors with shape (N, 7).
            deltas (paddle.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, w, l, h, r, velo*].

        Returns:
            paddle.Tensor: Decoded boxes.
        """
        xa, ya, za, wa, la, ha, ra = paddle.split(anchors, 7, axis=-1)
        xt, yt, zt, wt, lt, ht, rt = paddle.split(deltas, 7, axis=-1)
        diagonal = paddle.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za
        lg = paddle.exp(lt) * la
        wg = paddle.exp(wt) * wa
        hg = paddle.exp(ht) * ha
        rg = rt + ra
        return paddle.concat([xg, yg, zg, wg, lg, hg, rg], axis=-1)

    def encode(self, src_boxes, dst_boxes):
        """Get box regression transformation deltas (dx, dy, dz, dw, dh, dl,
        dr, dv*) that can be used to transform the `src_boxes` into the
        `target_boxes`.

        Args:
            src_boxes (paddle.Tensor): source boxes, e.g., object proposals.
            dst_boxes (paddle.Tensor): target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            paddle.Tensor: Box transformation deltas.
        """
        src_boxes = paddle.concat([src_boxes[..., 0:3], paddle.clip(
            src_boxes[..., 3:6], min=1e-5), src_boxes[..., 6:]], axis=-1)
        dst_boxes = paddle.concat([dst_boxes[..., 0:3], paddle.clip(
            dst_boxes[..., 3:6], min=1e-5), dst_boxes[..., 6:]], axis=-1)
        xa, ya, za, wa, la, ha, ra = paddle.split(src_boxes, 7, axis=-1)
        xg, yg, zg, wg, lg, hg, rg = paddle.split(dst_boxes, 7, axis=-1)
        diagonal = paddle.sqrt(la**2 + wa**2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = paddle.log(lg / la)
        wt = paddle.log(wg / wa)
        ht = paddle.log(hg / ha)
        rt = rg - ra
        return paddle.concat([xt, yt, zt, wt, lt, ht, rt], axis=-1)

    def add_sin_difference(self, boxes1, boxes2):
        rad_pred_encoding = paddle.sin(
            boxes1[..., -1:]) * paddle.cos(boxes2[..., -1:])
        rad_tg_encoding = paddle.cos(
            boxes1[..., -1:]) * paddle.sin(boxes2[..., -1:])
        boxes1 = paddle.concat([boxes1[..., :-1], rad_pred_encoding], axis=-1)
        boxes2 = paddle.concat([boxes2[..., :-1], rad_tg_encoding], axis=-1)
        return boxes1, boxes2

    def get_direction_target(self, gt_boxes, one_hot=True):
        rot_gt = gt_boxes[:, 6]
        rot_gt[rot_gt > np.pi] -= 2 * np.pi
        rot_gt[rot_gt < -np.pi] += 2 * np.pi
        dir_cls_targets = (rot_gt > 0).cast('int64')
        dir_cls_targets = one_hot_f(dir_cls_targets, 2, dtype=gt_boxes.dtype)
        return dir_cls_targets
