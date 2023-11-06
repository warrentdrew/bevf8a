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

# This code is based on https://github.com/ADLab-AutoDrive/BEVFusion/blob/main/mmdet3d/models/detectors/bevf_faster_rcnn.py

import os
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.detection.bevfusion.cam_stream_lss import LiftSplatShoot
from paddle3d.models.detection.bevfusion.mvx_faster_rcnn import MVXFasterRCNN
from paddle3d.models.layers.param_init import reset_parameters
from paddle3d.models.detection.bevfusion.conv_module import ConvModule
from paddle3d.utils.checkpoint import load_pretrained_model
from paddle3d.utils.logger import logger
from paddle3d.utils import checkpoint
from .merge_augs import merge_aug_bboxes_3d

__all__ = ['BEVFFasterRCNNV1']


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

class SEBlock(nn.Layer):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2D(1), nn.Conv2D(c, c, kernel_size=1, stride=1),
            nn.Sigmoid())
        self.init_weights()

    def forward(self, x):
        return x * self.att(x)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2D):
                reset_parameters(m)

        self.apply(_init_weights)


@manager.MODELS.add_component
class BEVFFasterRCNNV1(MVXFasterRCNN):
    """Multi-modality BEVFusion."""

    def __init__(self,
                 lss=False,
                 lc_fusion=False,
                 camera_stream=False,
                 camera_depth_range=[4.0, 45.0, 1.0],
                 img_depth_loss_weight=1.0,
                 img_depth_loss_method='kld',
                 grid=0.6,
                 num_views=6,
                 se=False,
                 final_dim=(900, 1600),
                 pc_range=[-50, -50, -5, 50, 50, 3],
                 downsample=4,
                 imc=256,
                 lic=384,
                 dla_add_extra_conv=False,
                 norm_cfg=dict(type='BatchNorm2D', epsilon=1e-5, momentum=1-0.1),
                 dla_input_dim=128,
                 bev_h=200,
                 bev_w=200,
                 use_cam2bev_transformer=False,
                 cam2bev_transformer=None,
                 cam2bev_head=None,
                 cam2bev_modules=None,
                 use_valid_flag=True,
                 pretrained=None,
                 norm_decay=0.0,
                 load_cam_from=None,
                 load_lidar_from=None,
                 bias_lr_factor=1.0,
                 **kwargs):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            lc_fusion (bool): fusing multi-modalities of camera and LiDAR in BEVFusion.
            camera_stream (bool): using camera stream.
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            grid, num_views, final_dim, pc_range, downsample: args for LSS, see cam_stream_lss.py.
            imc (int): channel dimension of camera BEV feature.
            lic (int): channel dimension of LiDAR BEV feature.

        """
        super(BEVFFasterRCNNV1, self).__init__(**kwargs)
        self.num_views = num_views
        self.lc_fusion = lc_fusion
        self.img_depth_loss_weight = img_depth_loss_weight
        self.img_depth_loss_method = img_depth_loss_method
        self.camera_depth_range = camera_depth_range
        self.lift = camera_stream
        self.dla_add_extra_conv = dla_add_extra_conv
        self.use_cam2bev_transformer = use_cam2bev_transformer
        self.se = se
        if self.dla_add_extra_conv:
            bias_flag=True
            self.conv4dla = nn.Sequential(nn.Conv2D(dla_input_dim, 256, 3, stride=1, padding=1, bias_attr=bias_flag),
                                          nn.ReLU())
        self.use_valid_flag = use_valid_flag
        self.cam2bev_modules = None
        if cam2bev_modules is not None:
            self.cam2bev_modules = cam2bev_modules
        if camera_stream:
            if self.use_cam2bev_transformer:
                self.cam2bev_transformer = cam2bev_transformer
                
            # elif self.use_cam2bev_transformer==False and cam2bev_modules is None:
            #     self.lift_splat_shot_vis = LiftSplatShoot(lss=lss, grid=grid, inputC=imc, camC=64, 
            #         pc_range=pc_range, final_dim=final_dim, downsample=downsample)
        if lc_fusion:
            if se:
                self.seblock = SEBlock(lic)
            self.reduc_conv = ConvModule(
                lic + imc,
                lic,
                3,
                padding=1,
                conv_cfg=None,
                norm_decay=norm_decay,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'))

        self.freeze_img = kwargs.get('freeze_img', False)
        print("In BEVF_FasterRCNN_V1, self.freeze_img: ", self.freeze_img)
        self.init_weights(pretrained=pretrained)
        self.freeze()
        self.init_lr(bias_lr_factor=bias_lr_factor)
        if load_cam_from is not None:
            logger.info("load cam weight from {}".format(load_cam_from))
            load_pretrained_model(self, load_cam_from, verbose=False)
        if load_lidar_from is not None:
            logger.info("load lidar weight from {}".format(load_lidar_from))
            load_pretrained_model(self, load_lidar_from, verbose=False)

    def init_lr(self, bias_lr_factor=1.0):
        for name, param in self.img_backbone.named_parameters():
            param.optimize_attr['learning_rate'] = bias_lr_factor

    def freeze(self):
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.trainable = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.trainable = False
            if self.lift:
                for param in self.lift_splat_shot_vis.parameters():
                    param.trainable = False

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

        # bev_features = paddle.to_tensor(np.load("/mnt/zhuyipin/idg/lidarrcnn/BEVFusion/bev_features.npy")) 
        return points, coors_batch, bev_features

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
        if self.dla_add_extra_conv:
            img_feat = self.conv4dla(img_feats[0])
            img_feats = (img_feat,)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_feat(self, points, img, img_metas, gt_bboxes_3d=None, gt_labels_3d=None, lidar_aug_matrix=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        if self.cam_only:
            pts_feats = None
        else:
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        # pts_feats = paddle.load("pts_feats2.pdt")
        depth_dist = None

        if self.cam2bev_modules is not None:
            if len(img_feats) == 4:
                mlvl_feats = list(img_feats)[1:]
            else:
                mlvl_feats = list(img_feats)
            for i in range(len(mlvl_feats)):
                BN, C, H, W = mlvl_feats[i].shape
                B = BN//self.num_views
                mlvl_feats[i] = mlvl_feats[i].reshape([B, int(BN / B), C, H, W])
        
            img_bev_feat = self.cam2bev_modules(mlvl_feats, img_metas, gt_bboxes_3d, gt_labels_3d, lidar_aug_matrix=lidar_aug_matrix)
            # img_bev_feat = paddle.load("img_bev_feat.pdt")
            if pts_feats is None:
                pts_feats = [img_bev_feat] ####cam stream only
            else:
                if self.lc_fusion==True:
                    if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                        img_bev_feat = F.interpolate(img_bev_feat, pts_feats[0].shape[2:], mode='bilinear', align_corners=True)
                    pts_feats = [self.reduc_conv(paddle.concat([img_bev_feat, pts_feats[0]], axis=1))]
                    # pts_feats = paddle.load("pts_feats4.pdt")
                    if self.se:
                        pts_feats = [self.seblock(pts_feats[0])]
                    # pts_feats = paddle.load("pts_feats5.pdt")

        elif self.lift:
            BN, C, H, W = img_feats[0].shape
            batch_size = BN // self.num_views
            img_feats_view = img_feats[0].reshape(
                [batch_size, self.num_views, C, H, W])
            rots = []
            trans = []
            for sample_idx in range(batch_size):
                rot_list = []
                trans_list = []
                for mat in img_metas[sample_idx]['lidar2img']:
                    mat = mat.astype('float32')
                    rot_list.append(mat.inverse()[:3, :3])
                    trans_list.append(mat.inverse()[:3, 3].reshape([-1]))
                rot_list = paddle.stack(rot_list, axis=0)
                trans_list = paddle.stack(trans_list, axis=0)
                rots.append(rot_list)
                trans.append(trans_list)
            rots = paddle.stack(rots)
            trans = paddle.stack(trans)
            lidar2img_rt = img_metas[sample_idx]['lidar2img']

            img_bev_feat, depth_dist = self.lift_splat_shot_vis(
                img_feats_view,
                rots,
                trans,
                lidar2img_rt=lidar2img_rt,
                img_metas=img_metas)
            if pts_feats is None:
                pts_feats = [img_bev_feat]
            else:
                if self.lc_fusion:
                    if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                        img_bev_feat = F.interpolate(
                            img_bev_feat,
                            pts_feats[0].shape[2:],
                            mode='bilinear',
                            align_corners=True)
                    pts_feats = [
                        self.reduc_conv(
                            paddle.concat([img_bev_feat, pts_feats[0]], axis=1))
                    ]
                    if self.se:
                        pts_feats = [self.seblock(pts_feats[0])]
        # pts_feats = paddle.load("pts_feats3.pdt")
        return dict(
            img_feats=img_feats, pts_feats=pts_feats, depth_dist=depth_dist)

    def export_forward(self, points, img, img_metas):
        """
        Args:
            points: paddle.Tensor, [num_points, 4]
            img: paddle.Tensor, [1, 6, 3, 480, 800]
            img_meats: List[]
        """
        # only for bs=1 forward
        # img_metas and points should be a list
        img_metas = [img_metas]
        points = [points]
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']
        depth_dist = feature_dict['depth_dist']
        if pts_feats and self.with_pts_bbox:
            bbox_preds = self.pts_bbox_head(pts_feats)
            if self.with_pts_roi_head:
                roi_preds = self.pts_roi_head(pts_feats)
        return bbox_preds, roi_preds

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']
        depth_dist = feature_dict['depth_dist']


        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            if self.test_cfg['use_benchmar_format']:
                return bbox_pts
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]
    
    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs, img_metas)
        return img_feats, pts_feats
    
    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
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

            # bbox_list[0][0].tensor[:, 6] = -bbox_list[0][0].tensor[:, 6] + math.pi / 2
            if self.use_bbox_used_in_mainline:
                bbox_list[0][0].tensor[:, 2] -= 0.5 * bbox_list[0][0].tensor[:, 5]

            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.test_cfg)
        return merged_bboxes
    
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
                        if roi_regions[i][j]['is_empty']:
                            roi_regions[i][j]['region'] = paddle.ones((0, 7), dtype='float32')

        # print("========test1================")
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, lidar_aug_matrix=lidar_aug_matrix)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']
        depth_dist = feature_dict['depth_dist']

        
        # img_feats = (paddle.to_tensor(np.load('/mnt/zhuyipin/idg/lidarrcnn/BEVFusion/img_feats[0].npy')),)
        # pts_feats = [paddle.to_tensor(np.load('/mnt/zhuyipin/idg/lidarrcnn/BEVFusion/pts_feats2.npy'))]
        losses = dict()

        # img_feats = paddle.load("img_feats.pdt")
        # pts_feats = paddle.load("pts_feats.pdt")
        if pts_feats:
            if self.use_valid_flag == True:
                # print("========test2================")
                losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas, valid_flag, 
                                                gt_bboxes_ignore=gt_bboxes_ignore, 
                                                gt_border_masks=gt_border_masks, 
                                                roi_regions=roi_regions)
                # print("========test3================")
            else:

                losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas, None, 
                                                gt_bboxes_ignore=gt_bboxes_ignore, 
                                                gt_border_masks=gt_border_masks, 
                                                roi_regions=roi_regions)

            losses.update(losses_pts)
        if img_feats:
            # print("========test4================")
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            # print("========test5================")
            if img_depth is not None:
                loss_depth = self.depth_dist_loss(
                    depth_dist,
                    img_depth,
                    loss_method=self.img_depth_loss_method,
                    img=img) * self.img_depth_loss_weight
                losses.update(img_depth_loss=loss_depth)
            losses.update(losses_img)
        return losses

    def depth_dist_loss(self,
                        predict_depth_dist,
                        gt_depth,
                        loss_method='kld',
                        img=None):
        # predict_depth_dist: B, N, D, H, W
        # gt_depth: B, N, H', W'
        B, N, D, H, W = predict_depth_dist.shape
        guassian_depth, min_depth = gt_depth[..., 1:], gt_depth[..., 0]
        mask = (min_depth >= self.camera_depth_range[0]) & (
            min_depth <= self.camera_depth_range[1])
        mask = mask.reshape([-1])
        guassian_depth = guassian_depth.reshape([-1, D])[mask]
        predict_depth_dist = predict_depth_dist.transpose(
            [0, 1, 3, 4, 2]).reshape([-1, D])[mask]
        if loss_method == 'kld':
            loss = F.kl_div(
                paddle.log(predict_depth_dist),
                guassian_depth,
                reduction='mean')
        elif loss_method == 'mse':
            loss = F.mse_loss(predict_depth_dist, guassian_depth)
        else:
            raise NotImplementedError
        return loss
    

    def export(self, save_dir, **kwargs):
        self.export_model = True
        self.forward = self.export_forward
        self.pts_voxel_layer.export_model = True
        self.pts_voxel_encoder.pfn_scatter.export_model = True
        self.pts_voxel_encoder.cluster_scatter.export_model = True
        self.pts_middle_encoder.export_model = True
        self.cam2bev_modules.transformer.encoder.export_model = True
        img_spec = paddle.static.InputSpec(
            shape=[1, 6, 3, 768, 1152], dtype='float32', name='img')
        pts_spec = paddle.static.InputSpec(
            shape=[None, 4], dtype='float32', name='pts')
        img_metas_spec = {
            'lidar2img': paddle.static.InputSpec(
                    shape=[1, 6, 4, 4], dtype='float32', name='lidar2img'),
            'pad_shape': [768, 1152, 3]
        }
        input_spec = [pts_spec, img_spec, img_metas_spec]

        save_path = os.path.join(save_dir, 'bevfusion')
        paddle.jit.to_static(self, input_spec=input_spec)
        paddle.jit.save(self, save_path)
        logger.info("Exported model is saved in {}".format(save_path))


# if __name__ == '__main__':
#     pts_feats = paddle.load("pts_feats4.pdt")
#     lic = 384
#     self.seblock = SEBlock(lic)
#     pts_feats = [self.seblock(pts_feats[0])]
#     print(pts_feats)