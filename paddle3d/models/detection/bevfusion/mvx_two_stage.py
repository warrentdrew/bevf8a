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

# This code is based on https://github.com/ADLab-AutoDrive/BEVFusion/blob/main/mmdet3d/models/detectors/mvx_two_stage.py

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.geometries import BBoxes3D
from paddle3d.sample import Sample, SampleMeta
from paddle3d.utils import checkpoint
from paddle3d.utils_idg.ops.dynamic_voxelize import Voxelization
from paddle3d.utils_idg.ops.bev_feature import BevFeature
from paddle3d.utils_idg import target_ops
from paddle3d.utils_idg.ops.iou3d_utils import boxes3d_to_near_torch, anchors_match_valid_voxels


# 8A used
def calculate_anchor_masks_torch(anchors, coordinates, grid_size, voxel_size, pc_range):
    # grid_size:[600, 600, 1]
    # voxel_size:[0.2, 0.2, 10]
    # pc_range:[-60, -60, -5, 60, 60, 5]
    # coordinates:[batch_id, 0, y, x]
    # anchors_near: x1, y1, x2, y2
    if isinstance(anchors, list):
        batch_size = anchors[0].shape[0]
        anchors_mask_list = []
        voxel_masks = []
        for i in range(batch_size):
            batch_mask = paddle.where(coordinates[:, 0] == i)[0].squeeze(1)
            this_coords = paddle.index_select(coordinates, batch_mask, axis=0)
            indices = this_coords[:, 2] * grid_size[0] + this_coords[:, 3] # coord(y, x)
            indices = indices.cast('int64')
            voxel_mask = paddle.zeros((grid_size[1] * grid_size[0], )).cast('int32')
            # voxel_mask = paddle.zeros((grid_size[1] * grid_size[0], )).cast('float32')
            if indices.min() < 0:
                indices = indices % (grid_size[1] * grid_size[0])
            voxel_mask = paddle.scatter(voxel_mask, indices, paddle.ones(indices.shape, dtype='int32')).cast('bool')
            # voxel_mask = paddle.put_along_axis(voxel_mask, indices, 1., 0).cast('bool')
            voxel_mask = voxel_mask.reshape([grid_size[1], grid_size[0]])
            voxel_masks.append(voxel_mask)
        # b = aa
        for task_id in range(len(anchors)):
            anchors_near = boxes3d_to_near_torch(anchors[task_id][0])
            anchors_near_0 = paddle.clip(paddle.floor((anchors_near[:, 0] - pc_range[0]) / voxel_size[0]),
                                        min=0, max=grid_size[0] - 1)
            anchors_near_1 = paddle.clip(paddle.floor((anchors_near[:, 1] - pc_range[1]) / voxel_size[1]),
                                        min=0, max=grid_size[1] - 1)
                                        
            anchors_near_2 = paddle.clip(paddle.floor((anchors_near[:, 2] - pc_range[0]) / voxel_size[0] + 1),
                                        min=0, max=grid_size[0] - 1)
                                        
            anchors_near_3 = paddle.clip(paddle.floor((anchors_near[:, 3] - pc_range[1]) / voxel_size[1] + 1),
                                        min=0, max=grid_size[1] - 1)
            anchors_near = paddle.stack([anchors_near_0, anchors_near_1, anchors_near_2, anchors_near_3], axis=1).cast('int32')
            anchors_mask = []
            for i in range(batch_size):
                anchor_mask = anchors_match_valid_voxels(anchors_near, voxel_masks[i])
                anchors_mask.append(anchor_mask.cast('int32'))
            anchors_mask = paddle.stack(anchors_mask, axis=0).cast('bool')

            anchors_mask_list.append(anchors_mask)
        return anchors_mask_list
    else:
        raise NotImplementedError

def cat(tensors, axis):
    """
    Efficient version of paddle.concat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    sum_c = sum([tensor.shape[0] for tensor in tensors])
    if sum_c==0:
        return tensors[0]
    return paddle.concat(tensors, axis)
    

class MVXTwoStageDetector(nn.Layer):
    """Base class of Multi-modality"""

    def __init__(
            self,
            sync_bn=False,
            freeze_img=False,
            pts_voxel_layer=None,
            bev_feature_layer=None,
            pts_voxel_encoder=None,
            pts_middle_encoder=None,
            pts_fusion_layer=None,
            img_backbone=None,
            pts_backbone=None,
            img_neck=None,
            pts_neck=None,
            pts_bbox_head=None,
            pts_roi_head=None,
            img_roi_head=None,
            img_rpn_head=None,
            train_cfg=None,
            test_cfg=None,
            cam_only=False,
            tasks=None,
            need_convert_gt_format=False,
            use_bbox_used_in_mainline=False,
            pretrained=None,
    ):
        super(MVXTwoStageDetector, self).__init__()

        self.sync_bn = sync_bn
        self.freeze_img = freeze_img
        self.test_cfg = test_cfg
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer) 
        if bev_feature_layer:
            self.compute_bev_feature = True
            self.bev_feature_layer = BevFeature(**bev_feature_layer)
        else:
            self.compute_bev_feature = False
        if pts_voxel_encoder:
            self.pts_voxel_encoder = pts_voxel_encoder
        if pts_middle_encoder:
            self.pts_middle_encoder = pts_middle_encoder
        if pts_backbone:
            self.pts_backbone = pts_backbone
        if pts_fusion_layer:
            self.pts_fusion_layer = pts_fusion_layer
        if pts_neck:
            self.pts_neck = pts_neck
        if pts_bbox_head:
            self.pts_bbox_head = pts_bbox_head

        if img_backbone:
            self.img_backbone = img_backbone
        if img_neck is not None:
            self.img_neck = img_neck
        if img_rpn_head is not None:
            self.img_rpn_head = img_rpn_head
        if img_roi_head is not None:
            self.img_roi_head = img_roi_head
        if pts_roi_head is not None:
            self.pts_roi_head = pts_roi_head
        self.cam_only = cam_only
        self.use_bbox_used_in_mainline = use_bbox_used_in_mainline
        self.need_convert_gt_format = need_convert_gt_format
        self.tasks = tasks
        if tasks is not None:
            self.num_classes = [len(t["class_names"]) for t in tasks]

        # self.init_weights()
        self.coors = None

    def init_weights(self, pretrained=None):
        """Initialize model weights."""

        if self.with_img_backbone:
            self.img_backbone.init_weights()
        if self.with_pts_backbone:
            self.pts_backbone.init_weights()
        if self.with_img_neck:
            if isinstance(self.img_neck, nn.Sequential):
                for m in self.img_neck:
                    m.init_weights()
            else:
                self.img_neck.init_weights()
        
        if pretrained is not None:
            checkpoint.load_pretrained_model(self, pretrained)

    @property
    def with_pts_roi_head(self):
        """bool: Whether the detector has a roi head in pts branch."""
        return hasattr(self, 'pts_roi_head') and self.pts_roi_head is not None

    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self,
                       'img_shared_head') and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self, 'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self,
                       'pts_fusion_layer') and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_pts_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(
            self, 'pts_voxel_encoder') and self.pts_voxel_encoder is not None

    @property
    def with_pts_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(
            self, 'pts_middle_encoder') and self.pts_middle_encoder is not None

    def train(self):
        super(MVXTwoStageDetector, self).train()
        if self.with_pts_neck:
            self.pts_neck.train()

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.shape[0] == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.shape[0] > 1:
                B, N, C, H, W = img.shape
                img = img.reshape([B * N, C, H, W])
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, self.coors, bev_features = self.voxelize(pts)
        voxel_features, feature_coors = self.pts_voxel_encoder(voxels, self.coors)
        batch_size = self.coors[-1, 0] + 1
        input_shape = self.pts_voxel_layer.pcd_shape[::-1]
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size, input_shape, bev_features)
        if self.with_pts_backbone:  
            x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        return x

    def extract_feat(self, points, img, img_metas, gt_bboxes_3d=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    def forward_train(self,
                      sample=None,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      roi_regions=None):
        """Forward training function.

        Args:
            points (list[paddle.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[paddle.Tensor], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[paddle.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[paddle.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[paddle.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (paddle.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[paddle.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[paddle.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        if sample is not None:
            img = sample.get('img', None)
            img_metas = sample['img_metas']
            gt_bboxes_3d = sample['gt_bboxes_3d']
            gt_labels_3d = sample['gt_labels_3d']
            points = sample.get('points', None)
            roi_regions = sample.get('roi_regions', None)

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore=gt_bboxes_ignore, gt_border_masks=gt_border_masks, roi_regions=roi_regions)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          valid_flag=None,
                          gt_bboxes_ignore=None,
                          gt_border_masks=None,
                          roi_regions=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[paddle.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[paddle.Tensor]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[paddle.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[paddle.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        # pts_feats = paddle.load("pts_feats1.pdt")
        num_points_in_gts = None
        # outs = self.pts_bbox_head(pts_feats)
        outs, head_layers = self.pts_bbox_head(pts_feats) # TODO1023 8A type
        if self.with_pts_roi_head: # TODO1023
            roi_preds = self.pts_roi_head(pts_feats)
        # outs = paddle.load("outs.pdt")
        # head_layers = paddle.load("head_layers.pdt")
        # roi_preds = paddle.load("roi_preds.pdt")
        if self.need_convert_gt_format:
            gt_bboxes_3d, gt_labels_3d, num_points_in_gts, gt_border_masks = self.convert_gt_format(gt_bboxes_3d, gt_labels_3d, num_points_in_gts=num_points_in_gts, gt_border_masks=gt_border_masks)
        
        
        roi_inputs = outs + (img_metas, self.coors)

        if self.cam_only:
            loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
            losses = self.pts_bbox_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        else:
            loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas, self.coors)

            # =====================
            # 8A   
            # roi_inputs = outs + (img_metas, self.coors)
            if roi_regions is not None:
                # print("===========test2.5=============")
                self.expand_roi_regions_by_pred(img_metas, loss_inputs, roi_inputs, roi_preds, roi_regions)
                # print("===========test2.6=============")

            # =====================
            losses, anchors_mask, batch_anchors, labels = self.pts_bbox_head.loss(
                *loss_inputs, self.test_cfg, None, gt_bboxes_ignore, gt_border_masks, roi_regions)
        # losses = {}

        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas, self.coors)
        if self.with_pts_roi_head:  # TODO1023
            roi_head_loss = self.pts_roi_head.loss(roi_preds, *loss_inputs, batch_anchors, anchors_mask, labels, num_points_in_gts=num_points_in_gts)
            losses.update(roi_head_loss)
        return losses

    # =============
    # 8A
    # def expand_roi_regions_by_pred(self, img_metas, loss_inputs, roi_inputs, roi_preds, roi_regions):
    #     featmap_sizes = np.array(self.pts_bbox_head.grid_size)[:2] // self.pts_bbox_head.downsample
    #     # device = loss_inputs[0][0].device
    #     batch_anchors = self.pts_bbox_head.get_anchors(featmap_sizes, img_metas)
    #     #
    #     if self.pts_bbox_head.cal_anchor_mask or self.pts_bbox_head.assign_cfg.use_anchor_mask:
    #         anchors_mask = target_ops.calculate_anchor_masks_torch(batch_anchors, self.coors, self.pts_bbox_head.grid_size,
    #                                                                self.pts_bbox_head.voxel_size,
    #                                                                self.pts_bbox_head.pc_range)
    #     else:
    #         anchors_mask = [None for _ in range(len(batch_anchors))]
    #     nms_rets = self.pts_roi_head.get_bboxes(roi_preds, *roi_inputs, batch_anchors, anchors_mask, self.test_cfg,
    #                                             is_training=True)
    #     for batch_id in range(len(roi_preds)):

    #         label_preds = nms_rets[2][batch_id]
    #         bigmot_cls_id = self.pts_roi_head.class2id['bigMot']
    #         selected_labels_inds = label_preds == bigmot_cls_id
    #         selected_boxes = nms_rets[0][batch_id][selected_labels_inds]

    #         for region in roi_regions[batch_id]:
    #             if region['type'] == 3:
    #                 region['region'] = paddle.concat([region['region'], selected_boxes], axis=0)

		
    def expand_roi_regions_by_pred(self, img_metas, loss_inputs, roi_inputs, roi_preds, roi_regions):
        featmap_sizes = np.array(self.pts_bbox_head.grid_size)[:2] // self.pts_bbox_head.downsample
        batch_anchors = self.pts_bbox_head.get_anchors(featmap_sizes, img_metas)
        # print("cond: ", self.pts_bbox_head.cal_anchor_mask or self.pts_bbox_head.assign_cfg['use_anchor_mask'])
        if self.pts_bbox_head.cal_anchor_mask or self.pts_bbox_head.assign_cfg['use_anchor_mask']:
            anchors_mask = calculate_anchor_masks_torch(batch_anchors, self.coors, self.pts_bbox_head.grid_size,
                                                                   self.pts_bbox_head.voxel_size,
                                                                   self.pts_bbox_head.pc_range)
        else:
            anchors_mask = [None for _ in range(len(batch_anchors))]
        # nms_rets = self.pts_roi_head.get_bboxes(roi_preds, *roi_inputs, batch_anchors, anchors_mask, self.test_cfg,
        #                                         is_training=True)
        # for batch_id in range(len(roi_preds)):    # TODO yipin change back

        #     label_preds = nms_rets[2][batch_id]
        #     bigmot_cls_id = self.pts_roi_head.class2id['bigMot']
        #     if label_preds.shape[0]>0:
        #         selected_labels_inds = paddle.where(label_preds == bigmot_cls_id)[0].squeeze(1)
        #         if selected_labels_inds.shape[0]>0:
        #             selected_boxes = paddle.index_select(nms_rets[0][batch_id], selected_labels_inds, axis=0)

        #             for region in roi_regions[batch_id]:
        #                 if region['type'] == 3:
        #                     region['region'] = paddle.concat([region['region'], selected_boxes], axis=0)


    def forward_img_train(self,
                          x,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          proposals=None,
                          **kwargs):
        """Forward function for image branch.

        This function works similar to the forward function of Faster R-CNN.

        Args:
            x (list[paddle.Tensor]): Image features of shape (B, C, H, W)
                of multiple levels.
            img_metas (list[dict]): Meta information of images.
            gt_bboxes (list[paddle.Tensor]): Ground truth boxes of each image
                sample.
            gt_labels (list[paddle.Tensor]): Ground truth labels of boxes.
            gt_bboxes_ignore (list[paddle.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            proposals (list[paddle.Tensor], optional): Proposals of each sample.
                Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        losses = dict()
        # RPN forward and loss
        if self.with_img_rpn:
            rpn_outs = self.img_rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                          self.train_cfg.img_rpn)
            rpn_losses = self.img_rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('img_rpn_proposal',
                                              self.test_cfg.img_rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # bbox head forward and loss
        if self.with_img_bbox:
            # bbox head forward and loss
            img_roi_losses = self.img_roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore, **kwargs)
            losses.update(img_roi_losses)

        return losses

    def simple_test_img(self, x, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_metas,
                                                 self.test_cfg.img_rpn)
        else:
            proposal_list = proposals

        return self.img_roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
    
    def simple_test_rpn(self, x, img_metas, rpn_test_cfg):
        """RPN test function."""
        rpn_outs = self.img_rpn_head(x)
        proposal_inputs = rpn_outs + (img_metas, rpn_test_cfg)
        proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

        
    def forward(self, sample, **kwargs):
        # sample = paddle.load("test_sample.pdt")
        if self.training:
            loss_dict = self.forward_train(sample, **kwargs)
            return {'loss': loss_dict}
        else:
            preds = self.forward_test(sample, **kwargs)
            return preds

    def forward_test(self, sample, **kwargs):
        """
        as name
        """
        img = sample.get('img', None)
        points = sample.get('points', None)
        img_metas = sample['img_metas']
        num_augs = len(points)
        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
        else:
            return self.aug_test(points, img_metas, img, **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=True):
        """Test function of point cloud branch."""
        # outs = self.pts_bbox_head(x)
        outs, head_layers = self.pts_bbox_head(x)
        if self.with_pts_roi_head:
            roi_preds = self.pts_roi_head(x)
        has_two_stage = True

        if self.cam_only:
            bbox_list = self.pts_bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale)
        else:
            loss_inputs = outs + (img_metas, self.coors)
            bbox_list, anchors_mask, batch_anchors = self.pts_bbox_head.get_bboxes(
                *loss_inputs, self.test_cfg, has_two_stage=has_two_stage)
        
        if self.with_pts_roi_head and has_two_stage:
            bbox_list = self.pts_roi_head.get_bboxes(roi_preds, *loss_inputs, batch_anchors, anchors_mask, self.test_cfg) 
        
        if self.use_bbox_used_in_mainline:
            bbox_list[0][0].tensor[:, 2] -= 0.5 * bbox_list[0][0].tensor[:, 5]

        bbox_results = [
            bbox3d2result_combouid(bboxes, scores, labels, combo_uids)
            for bboxes, scores, labels, combo_uids in bbox_list
        ]
        return bbox_results

    def convert_gt_format(self, gt_bboxes, gt_labels, num_points_in_gts=None, gt_border_masks=None):
        """Convert gt infos format from batch priority to task priority, which is used in ppdetfusion

        Args:
            gt_bboxes (list[paddle.Tensor(float)]): format: [bs1:tensor[num_gt_bboxes1, box_code_size], ..., bsN:tensor[num_gt_bboxesN, box_code_size]].
            gt_labels (list[paddle.Tensor(int)]): format: [bs1:tensor[num_gt_bboxes1], ..., bsN:tensor[num_gt_bboxesN]].
            num_points_in_gts (list[paddle.Tensor(int)]): format: [bs1:tensor[num_gt_bboxes1], ..., bsN:tensor[num_gt_bboxesN]].

        Returns:
            gt_bboxes: format: [task1:[bs1:tensor[num_gt_bboxes11+...+num_gt_bboxes1cls, box_code_size], ...], ...]
            gt_labels: format: [task1:[bs1:tensor[num_gt_bboxes11+...+num_gt_bboxes1cls], ...], ...]
            num_points_in_gts: format: [task1:[bs1:[tensor[num_gt_bboxes11+...+num_gt_bboxes1cls], ...], ...]
        """
        batch_size = len(gt_bboxes)
        gt_bboxes_by_task = []
        gt_labels_by_task = []
        if num_points_in_gts is not None:
            num_points_in_gts_by_task = []
    		
        if gt_border_masks is not None:
            gt_border_masks_by_task = []

        pre_cumsum_num_classes = 0
        for task_id in range(len(self.tasks)):
            gt_bboxes_per_task = []
            gt_labels_per_task = []
            if num_points_in_gts is not None:
                num_points_in_gts_per_task = []
            
            if gt_border_masks is not None:
                gt_border_masks_per_task = []
            
            for batch_id in range(batch_size):
                gt_bboxes_per_batch = []
                gt_labels_per_batch = []
                if num_points_in_gts is not None:
                    num_points_in_gts_per_batch = []
                if gt_border_masks is not None:
                    gt_border_masks_per_batch = []
                for class_id in range(self.num_classes[task_id]):
                    mask = gt_labels[batch_id] == pre_cumsum_num_classes + class_id
                    gt_bboxes_per_batch.append(paddle.to_tensor(gt_bboxes[batch_id][mask]))
                    gt_labels_per_batch.append(paddle.full_like(gt_labels[batch_id][mask], class_id + 1, dtype=gt_labels[batch_id][mask].dtype))
                    if num_points_in_gts is not None:
                        num_points_in_gts_per_batch.append(num_points_in_gts[batch_id][mask])

                    if gt_border_masks is not None:
                        gt_border_masks_per_batch.append(paddle.to_tensor(gt_border_masks[batch_id][mask])) #torch.tensor(gt_border_masks[batch_id][mask], device=device))

                # gt_bboxes_per_batch = LiDARInstance3DBoxes.cat(gt_bboxes_per_batch)
                if gt_bboxes_per_batch[0].shape[0] == 0:
                    gt_bboxes_per_batch = paddle.empty(gt_bboxes_per_batch[0].shape)
                else:
                    gt_bboxes_per_batch = paddle.concat(gt_bboxes_per_batch, axis=0)
                
                if gt_labels_per_batch[0].shape[0] == 0:
                    gt_labels_per_batch = paddle.empty(gt_labels_per_batch[0].shape)
                else:
                    gt_labels_per_batch = paddle.concat(gt_labels_per_batch, axis=0)
                # gt_bboxes_per_batch = paddle.concat(gt_bboxes_per_batch, axis=0)
                # gt_labels_per_batch = paddle.concat(gt_labels_per_batch, axis=0)
                if num_points_in_gts is not None:
                    num_points_in_gts_per_batch = paddle.concat(num_points_in_gts_per_batch, axis=0)
                if gt_border_masks is not None:
                    gt_border_masks_per_batch = cat(gt_border_masks_per_batch, axis = 0) #paddle.concat(gt_border_masks_per_batch, axis = 0) #torch.cat(gt_border_masks_per_batch, dim=0)
                gt_bboxes_per_task.append(gt_bboxes_per_batch)
                gt_labels_per_task.append(gt_labels_per_batch)
                if num_points_in_gts is not None:
                    num_points_in_gts_per_task.append(num_points_in_gts_per_batch)

                if gt_border_masks is not None:
                    gt_border_masks_per_task.append(gt_border_masks_per_batch)

            gt_bboxes_by_task.append(gt_bboxes_per_task)
            gt_labels_by_task.append(gt_labels_per_task)
            if num_points_in_gts is not None:
                num_points_in_gts_by_task.append(num_points_in_gts_per_task)
            if gt_border_masks is not None:
                gt_border_masks_by_task.append(gt_border_masks_per_task)
            pre_cumsum_num_classes += self.num_classes[task_id]
        if num_points_in_gts is None:
            num_points_in_gts_by_task = None
        if gt_border_masks is None:
            gt_border_masks_by_task = None
        return gt_bboxes_by_task, gt_labels_by_task, num_points_in_gts_by_task, gt_border_masks_by_task


    def simple_test(self, points, img_metas, img=None, rescale=True):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def _parse_results_to_sample(self, results: dict, sample: dict):
        num_samples = len(results)
        new_results = []
        for i in range(num_samples):
            data = Sample(None, sample["modality"][i])
            bboxes_3d = results[i]['pts_bbox']["boxes_3d"].numpy()
            labels = results[i]['pts_bbox']["labels_3d"].numpy()
            confidences = results[i]['pts_bbox']["scores_3d"].numpy()
            bottom_center = bboxes_3d[:, :3]
            gravity_center = np.zeros_like(bottom_center)
            gravity_center[:, :2] = bottom_center[:, :2]
            gravity_center[:, 2] = bottom_center[:, 2] + bboxes_3d[:, 5] * 0.5
            bboxes_3d[:, :3] = gravity_center
            data.bboxes_3d = BBoxes3D(bboxes_3d[:, 0:7])
            data.bboxes_3d.coordmode = 'Lidar'
            data.bboxes_3d.origin = [0.5, 0.5, 0.5]
            data.bboxes_3d.rot_axis = 2
            data.bboxes_3d.velocities = bboxes_3d[:, 7:9]
            data['bboxes_3d_numpy'] = bboxes_3d[:, 0:7]
            data['bboxes_3d_coordmode'] = 'Lidar'
            data['bboxes_3d_origin'] = [0.5, 0.5, 0.5]
            data['bboxes_3d_rot_axis'] = 2
            data['bboxes_3d_velocities'] = bboxes_3d[:, 7:9]
            data.labels = labels
            data.confidences = confidences
            data.meta = SampleMeta(id=sample["meta"][i]['id'])
            if "calibs" in sample:
                calib = [calibs.numpy()[i] for calibs in sample["calibs"]]
                data.calibs = calib
            new_results.append(data)
        return new_results

    def collate_fn(self, batch):
        sample = batch[0]
        collated_batch = {}
        collated_fields = [
            'img', 'points', 'img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_border_masks', 'roi_regions', 
            'modality', 'meta', 'img_depth', 'sample_idx' 
        ]
        for k in list(sample.keys()):
            if k not in collated_fields:
                continue
            if k == 'img':
                collated_batch[k] = np.stack([elem[k] for elem in batch],
                                             axis=0)
            elif k == 'img_depth':
                collated_batch[k] = paddle.stack(
                    [paddle.stack(elem[k], axis=0) for elem in batch], axis=0)
            else:
                collated_batch[k] = [elem[k] for elem in batch]
        return collated_batch


def bbox3d2result(bboxes, scores, labels):
    """Convert detection results to a list of numpy arrays.
    """
    return dict(
        boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)

# ===========
# 8A
def bbox3d2result_combouid(bboxes, scores, labels, combo_uids):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
    """
    return dict(
        boxes_3d=bboxes,
        scores_3d=scores,
        labels_3d=labels,
        combo_uid=combo_uids
        )