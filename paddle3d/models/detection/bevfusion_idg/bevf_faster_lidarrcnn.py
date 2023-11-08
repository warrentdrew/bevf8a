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
import numpy as np
import time
from collections import OrderedDict

import paddle
import paddle.nn as nn
from paddle.nn import functional as F

from paddle3d.apis import manager
from paddle3d.utils.logger import logger
from paddle3d.utils.checkpoint import load_pretrained_model
from paddle3d.models.detection.bevfusion.mvx_faster_rcnn import MVXFasterRCNN
from paddle3d.models.detection.bevfusion.conv_module import ConvModule
from paddle3d.models.detection.bevfusion.mvx_two_stage import bbox3d2result_combouid
from .bevf_faster_rcnn_v1 import SEBlock 

@manager.MODELS.add_component
class BEVFFasterLiDARRCNN(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""

    def __init__(self, lc_fusion=False,  num_views=6, se=False, imc=256, lic=384,    
                norm_cfg=dict(type='BatchNorm2D', epsilon=1e-5, momentum=1-0.1), cam2bev_modules=None,
                train_cam2bev=False, train_mghead=False, train_lidar=False,
                only_rcnn=True, pretrained=None, **kwargs):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            lc_fusion (bool): fusing multi-modalities of camera and LiDAR in BEVFusion.
            camera_stream (bool): using camera stream.
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            grid, num_views, final_dim, pc_range, downsample: args for LSS, see cam_stream_lss.py.
            imc (int): channel dimension of camera BEV feature.
            lic (int): channel dimension of LiDAR BEV feature.
            use_cam2bev_transformer: using Transformer-based cam2bev Transform (From BEVFormer)

        """
        self.pretrained = pretrained
        super(BEVFFasterLiDARRCNN, self).__init__(**kwargs)
        self.num_views = num_views
        self.lc_fusion = lc_fusion
        self.se = se
        self.cam2bev_modules = None
        if cam2bev_modules is not None:
            self.cam2bev_modules = cam2bev_modules
        
        if lc_fusion:
            if se:
                self.seblock = SEBlock(lic)
            self.reduc_conv = ConvModule(
                lic + imc,
                lic,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'))
            
        self.freeze_img = kwargs.get('freeze_img', False)
        self.only_rcnn = only_rcnn
        self.train_cam2bev = train_cam2bev
        self.train_mghead = train_mghead
        self.train_lidar = train_lidar
        logger.info(f"In BEVFFasterLiDARRCNN, self.freeze_img: {self.freeze_img}")
        logger.info(f"In BEVFFasterLiDARRCNN, self.only_rcnn: {self.only_rcnn}")
        logger.info(f"In BEVFFasterLiDARRCNN, self.pretrained: {self.pretrained}")
        logger.info(f"In BEVFFasterLiDARRCNN, self.train_cam2bev: {self.train_cam2bev}")
        logger.info(f"In BEVFFasterLiDARRCNN, self.train_mghead: {self.train_mghead}")
        logger.info(f"In BEVFFasterLiDARRCNN, self.train_lidar: {self.train_lidar}")
        # if self.only_rcnn:
            # assert self.pretrained is not None
        self.init_weights(pretrained=pretrained)
        self.freeze()
    def init_weights(self, pretrained=None):
        if pretrained is not None:
            load_pretrained_model(self, pretrained, verbose=False)
    def freeze(self):
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.trainable = False
                self.img_backbone.eval()
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.trainable = False
                self.img_neck.eval()
        if self.only_rcnn:
            for param in self.parameters():
                param.trainable = False
            self.eval() #先冻结所有参数
            for param in self.pts_roi_head.parameters():
                param.trainable = True #然后打开roi_head的参数
            self.pts_roi_head.train() #更改bn drop等层的状态
            if self.train_cam2bev:
                for param in self.cam2bev_modules.parameters():
                    param.trainable = True #然后打开cam2bev的参数
                self.cam2bev_modules.train()

                for param in self.seblock.parameters():
                    param.trainable = True #然后打开seblock的参数
                self.seblock.train()

                for param in self.reduc_conv.parameters():
                    param.trainable = True #然后打开reduc_conv的参数
                self.reduc_conv.train()
                for param in self.pts_bbox_head.parameters():
                    param.trainable = True #然后打开roi_head的参数
                self.pts_bbox_head.train()
            if self.train_mghead:
                for param in self.pts_bbox_head.parameters():
                    param.trainable = True #然后打开roi_head的参数
                self.pts_bbox_head.train()
            if self.train_lidar:
                # assert self.train_mghead
                for param in self.pts_voxel_encoder.parameters():
                    param.trainable = True #然后打开pts_voxel_encoder的参数
                self.pts_voxel_encoder.train()
                for param in self.pts_middle_encoder.parameters():
                    param.trainable = True #然后打开roi_head的参数
                self.pts_middle_encoder.train()
                for param in self.pts_neck.parameters():
                    param.trainable = True #然后打开roi_head的参数
                self.pts_neck.train()
                for param in self.seblock.parameters():
                    param.trainable = True #然后打开seblock的参数
                self.seblock.train()
                for param in self.reduc_conv.parameters():
                    param.trainable = True #然后打开reduc_conv的参数
                self.reduc_conv.train()
                for param in self.pts_bbox_head.parameters():
                    param.trainable = True #然后打开roi_head的参数
                self.pts_bbox_head.train()

                
            for nm, param in self.named_parameters():
                if param.trainable:
                    logger.info(f"required_grad param: {nm}")

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
            coor_pad = F.pad(coor.unsqueeze(0), (1, 0), mode='constant', value=i, data_format='NCL').squeeze(0)
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
        return x, voxels, self.coors, bev_features


    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if getattr(self, 'export_model', False):
                img = img.squeeze(0)
            else:
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

    def extract_feat(self, points, img, img_metas, gt_bboxes_3d=None, gt_labels_3d=None, lidar_aug_matrix=None):
        """Extract features from images and points."""
        with paddle.set_grad_enabled((not self.only_rcnn) and self.training):
            img_feats = self.extract_img_feat(img, img_metas)
            if len(img_feats) == 4:
                mlvl_feats = list(img_feats)[1:]
            else:
                mlvl_feats = list(img_feats)
            for i in range(len(mlvl_feats)):
                BN, C, H, W = mlvl_feats[i].shape
                B = BN//self.num_views
                mlvl_feats[i] = mlvl_feats[i].reshape([B, int(BN / B), C, H, W])
        
        with_grad=(not self.only_rcnn) or (self.only_rcnn and self.train_cam2bev)
        with paddle.set_grad_enabled(with_grad and self.training):
            img_bev_feat = self.cam2bev_modules(mlvl_feats, img_metas, gt_bboxes_3d, gt_labels_3d, lidar_aug_matrix=lidar_aug_matrix)
        
        with_grad=(not self.only_rcnn) or (self.only_rcnn and self.train_lidar)
        with paddle.set_grad_enabled(with_grad and self.training):
            pts_feats, points, coords, bev_features = self.extract_pts_feat(points, img_feats, img_metas)
        
        with_grad=(not self.only_rcnn) or (self.only_rcnn and (self.train_lidar or self.train_cam2bev))
        with paddle.set_grad_enabled(with_grad and self.training):
            depth_dist = None
            if self.lc_fusion:
                if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                    img_bev_feat = F.interpolate(img_bev_feat, pts_feats[0].shape[2:], mode='bilinear', align_corners=True)
                pts_feats = [self.reduc_conv(paddle.concat([img_bev_feat, pts_feats[0]], axis=1))]
                if self.se:
                    pts_feats = [self.seblock(pts_feats[0])]

        return dict(
            img_feats = img_feats,
            pts_feats = pts_feats,
            depth_dist = depth_dist,
            pts=points,
            pts_batch_ids=coords[:, 0],
            bev_features=bev_features
        )
    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats'] 
        depth_dist = feature_dict['depth_dist']
        pts = feature_dict['pts']
        pts_batch_ids = feature_dict['pts_batch_ids']
        bev_features = feature_dict['bev_features']

        bbox_list = [dict() for i in range(len(img_metas))]
        
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, pts, pts_batch_ids, img_metas, rescale=rescale, bev_features=bev_features)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def simple_test_pts(self, x, pts, pts_batch_ids, img_metas, rescale=False, bev_features=None):
        """Test function of point cloud branch."""
        #提取proposal
        bbox_head_outs, head_layers = self.pts_bbox_head(x)
        proposal_list = self.pts_bbox_head.get_proposal_for_rcnn(*bbox_head_outs, img_metas, self.coors, self.pts_roi_head.rpn_cfg['test']) 
        #rcnn前向推理
        roi_head_output = self.pts_roi_head(proposal_list, pts, pts_batch_ids, [head_layers[0], head_layers[1], head_layers[3]], bev_features, img_metas)
        rcnn_cls_preds, rcnn_reg_preds, rcnn_iou_preds, rcnn_dir_preds, rcnn_bev_cls_preds, rcnn_bev_reg_preds, batch_rois, batch_roi_labels, batch_roi_scores, \
            roi_batch_ids, pts, pts_batch_ids, batch_roi_empty_masks = roi_head_output
        #rcnn后处理
        bbox_list = self.pts_roi_head.get_bboxes(rcnn_cls_preds, rcnn_reg_preds,
            rcnn_iou_preds,
            rcnn_dir_preds,
            batch_rois,
            roi_batch_ids,
            batch_roi_empty_masks,
            self.test_cfg,
            batch_roi_scores,
            batch_roi_labels
        )  

        if self.use_bbox_used_in_mainline:
            bbox_list[0][0].tensor[:, 2] -= 0.5 * bbox_list[0][0].tensor[:, 5]
        bbox_results = [
            bbox3d2result_combouid(bboxes, scores, labels, combo_uids)
            for bboxes, scores, labels, combo_uids in bbox_list
        ]
        return bbox_results

    def forward_train(self,
                      sample=None,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_border_masks=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      proposals=None,
                      lidar_aug_matrix=None,
                      valid_flag=None,
                      gt_bboxes_ignore=None,
                      roi_regions=None):
        if sample is not None:
            img = sample['img']
            img_metas = sample['img_metas']
            gt_bboxes_3d = sample['gt_bboxes_3d']
            gt_labels_3d = sample['gt_labels_3d']
            points = sample.get('points', None)
            img_depth = sample.get('img_depth', None)
            lidar_aug_matrix = sample.get('lidar_aug_matrix', None)
            valid_flag = sample.get('valid_flag', None)
            gt_border_masks = sample.get('gt_border_masks', None)
            roi_regions = sample.get('roi_regions', None)
            if roi_regions is not None:
                for i in range(len(roi_regions)):
                    for j in range(len(roi_regions[i])):
                        if roi_regions[i][j]['is_emtpy']:
                            roi_regions[i][j]['region'] = paddle.ones((0, 7), dtype='float32')


        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, lidar_aug_matrix=lidar_aug_matrix)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']
        bev_features= feature_dict['bev_features']
        # depth_dist = feature_dict['depth_dist']
        pts = feature_dict['pts']
        pts_batch_ids = feature_dict['pts_batch_ids']

        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(
                pts_feats, pts, pts_batch_ids, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore, gt_border_masks, roi_regions, bev_features)
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
                          pts, 
                          pts_batch_ids,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          gt_border_masks=None,
                          roi_regions=None,
                          bev_features=None
                          ):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[paddle.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[paddle.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[paddle.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        num_points_in_gts = None
        losses = dict()
        with_grad=(not self.only_rcnn) or (self.only_rcnn and (self.train_mghead or self.train_lidar or self.train_cam2bev))
        with paddle.set_grad_enabled(with_grad and self.training):
            bbox_head_outs, head_layers = self.pts_bbox_head(pts_feats)
        if with_grad:
            if self.need_convert_gt_format:
                gt_bboxes_3d_list, gt_labels_3d_list, num_points_in_gts_list, gt_border_masks_list = self.convert_gt_format(gt_bboxes_3d, gt_labels_3d, num_points_in_gts=num_points_in_gts,
                                                                                             gt_border_masks=gt_border_masks)
                loss_inputs = bbox_head_outs + (gt_bboxes_3d_list, gt_labels_3d_list, img_metas, self.coors)
            bbox_head_loss, anchors_mask, batch_anchors, labels = self.pts_bbox_head.loss(
                *loss_inputs, self.test_cfg, None, gt_bboxes_ignore, gt_border_masks_list, roi_regions)
            losses['bbox_head_loss'] = bbox_head_loss
        if self.with_pts_roi_head:
            proposal_list = self.pts_bbox_head.get_proposal_for_rcnn(*bbox_head_outs, img_metas, self.coors, self.pts_roi_head.rpn_cfg['train'], roi_regions, gt_bboxes_3d,
                        gt_labels_3d,)
            roi_head_output = self.pts_roi_head(proposal_list, pts, pts_batch_ids, [head_layers[0], head_layers[1], head_layers[3]], bev_features, img_metas)
            rcnn_cls_preds, rcnn_reg_preds, rcnn_iou_preds, rcnn_dir_preds, rcnn_bev_cls_preds, rcnn_bev_reg_preds, batch_rois, batch_roi_labels, batch_roi_scores, \
            roi_batch_ids, pts, pts_batch_ids, batch_roi_empty_masks = roi_head_output
            rcnn_loss_input = (rcnn_cls_preds, rcnn_reg_preds, rcnn_iou_preds, rcnn_dir_preds, rcnn_bev_cls_preds, rcnn_bev_reg_preds, batch_rois, \
            roi_batch_ids, pts, pts_batch_ids, batch_roi_empty_masks)
            roi_head_loss = self.pts_roi_head.loss(*rcnn_loss_input, gt_bboxes_3d, gt_labels_3d,)
            losses['roi_head_loss'] = roi_head_loss
        return losses

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        bbox_head_loss = losses.get('bbox_head_loss', None)
        roi_head_loss = losses.get('roi_head_loss')
        if bbox_head_loss is not None:
            for loss_name, loss_value in bbox_head_loss.items():
                if isinstance(loss_value, paddle.Tensor):
                    log_vars[loss_name] = loss_value.mean()
                elif isinstance(loss_value, list):
                    log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')
            loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
            log_vars['rpn_loss']=loss
        else:
            loss = 0
        
        loss = loss + roi_head_loss['rcnn_loss']
        log_vars['loss'] = loss
        
        for loss_name, loss_value in roi_head_loss.items():
            if isinstance(loss_value, paddle.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
        
        
        for loss_name, loss_value in log_vars.items():
            log_vars[loss_name] = loss_value.detach().cpu().item()
      
        return loss, log_vars

    def forward(self, sample, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        paddle.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[paddle.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if self.training:
            loss_dict = self.forward_train(sample, **kwargs)
            return {'loss': loss_dict}
        else:
            preds = self.forward_test(sample, **kwargs)
            return preds

