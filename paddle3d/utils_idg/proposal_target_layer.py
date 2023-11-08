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
import paddle
import paddle.nn as nn

from paddle3d.utils_idg.ops.roi_pool3d_utils import point_iou_gpuv2

class ProposalTargetLayer(nn.Layer):
    def __init__(self, roi_sampler_cfg, similarity_calc, num_class):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

        if isinstance(self.roi_sampler_cfg['reg_fg_thresh'], list):
            self.reg_fg_thresh = paddle.to_tensor(self.roi_sampler_cfg['reg_fg_thresh'])
        if isinstance(self.roi_sampler_cfg['cls_fg_thresh'], list):
            self.cls_fg_thresh = paddle.to_tensor(self.roi_sampler_cfg['cls_fg_thresh'])
        if isinstance(self.roi_sampler_cfg['cls_fg_thresh_low'], list):
            self.cls_fg_thresh_low = paddle.to_tensor(self.roi_sampler_cfg['cls_fg_thresh_low'])
        if isinstance(self.roi_sampler_cfg['cls_bg_thresh'], list):
            self.cls_bg_thresh = paddle.to_tensor(self.roi_sampler_cfg['cls_bg_thresh'])

        self.similarity_calc = similarity_calc
        self.num_class = num_class

    def forward(self, pts, pts_batch_ids, batch_rois, roi_batch_ids, gt_boxes, gt_classes, pred_aware_assign=False, rcnn_box_preds=None):
        """
        Args:
            example:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            example:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """

        return self.sample_rois_for_rcnn(pts, pts_batch_ids, batch_rois, roi_batch_ids, gt_boxes, gt_classes, pred_aware_assign=pred_aware_assign, rcnn_box_preds=rcnn_box_preds)

    def sample_rois_for_rcnn(self, pts, pts_batch_ids, batch_rois, roi_batch_ids, gt_boxes, gt_classes, pred_aware_assign=False, rcnn_box_preds=None):
        """
        Args:
            example:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        #batch_roi_rcnn_labels,-2 ignore, -1 neg, >=0 pos
        batch_size = int(pts_batch_ids.max().item()) + 1
        batch_roi_ious = paddle.zeros((batch_rois.shape[0],), dtype='float32')
        batch_roi_rcnn_labels = paddle.full((batch_rois.shape[0],), -2, dtype='int64')
        batch_roi_gt_boxes = paddle.zeros((batch_rois.shape[0], 7), dtype='float32')
        batch_roi_gt_classes = paddle.full((batch_rois.shape[0],), -2, dtype='int64')
        batch_roi_border_masks = paddle.zeros((batch_rois.shape[0], 4), dtype='float32')
        batch_roi_reg_mask = paddle.zeros((batch_rois.shape[0],), dtype='bool')
        batch_roi_soft_labels = paddle.zeros((batch_rois.shape[0],), dtype='float32')
        batch_roi_sampled_mask = paddle.zeros((batch_rois.shape[0],), dtype='bool')
        
        for index in range(batch_size):
            roi_batch_mask = roi_batch_ids == index
            # pts_batch_mask = pts_batch_ids == index
            pts_batch_indx = paddle.nonzero(pts_batch_ids == index).squeeze(-1)
            cur_roi = batch_rois[roi_batch_mask]
            if pred_aware_assign and (rcnn_box_preds is not None):
                cur_roi = rcnn_box_preds[roi_batch_mask].detach()
            # cur_gt_mask = ret['roi_is_gt'] > 0
            
            cur_gt_box = gt_boxes[index]
            
            cur_gt_cls = gt_classes[index]
            # cur_gt_cls[cur_gt_cls == 6] = 3 #merge verybigmot and bigmot
            valid_gt_mask = (cur_gt_cls >=0) & (cur_gt_cls < self.num_class)
            cur_gt_box = cur_gt_box[valid_gt_mask]
            cur_gt_cls = cur_gt_cls[valid_gt_mask]
            cur_gt_cls = cur_gt_cls.cast('int64')

            # cur_feats = pts[pts_batch_mask, :]
            if pts_batch_indx.shape[0] > 0:
                cur_feats = paddle.index_select(pts, pts_batch_indx, axis=0)
                cur_pts = cur_feats[:, 0:3]
            else:
                cur_feats = paddle.ones([0, pts.shape[1]], dtype=pts.dtype)
                cur_pts = paddle.ones([0, 3], dtype=pts.dtype)

            if len(cur_roi) == 0:
                print("current batch roi num 0")
                continue
            if cur_pts.shape[0] == 0:
                print("current pts num 0, cur roi num {}".format(len(cur_roi)))
                continue

            if len(cur_gt_box) == 0:
                cur_gt_box = paddle.zeros((1, cur_gt_box.shape[1]), dtype=cur_gt_box.dtype)
                cur_gt_cls = paddle.full((1,), -1, dtype='int64')

            cur_gt_border_mask = self.generate_border_masks(cur_gt_box)  # (N, 4)
            cur_gt_border_mask = cur_gt_border_mask.cast('float32')

            if self.roi_sampler_cfg.get("point_iou", False):
                cur_gt_box_ = cur_gt_box.clone()
                cur_gt_box_[:, 2] = cur_gt_box_[:, 2] + cur_gt_box_[:, 5] * 0.5  #变换到几何中心
                cur_roi_ = cur_roi.clone()
                cur_roi_[:, 2] = cur_roi_[:, 2] + cur_roi_[:, 5] * 0.5  #变换到几何中心
                iou3d = point_iou_gpuv2(cur_pts, cur_roi_, cur_gt_box_[:, 0:7])
                del cur_gt_box_
                del cur_roi_
                # print("cur roi {}, cur gt {}, cur pts {}".format(cur_roi_.shape, cur_gt_box_.shape, cur_pts.shape))
            else:
                iou3d = self.similarity_calc(cur_roi, cur_gt_box[:, 0:7])  # (M, N)
            # print("gtclass, ", cur_gt_cls)
            assign_gt_labels, assign_soft_labels, assign_gt_ids, assign_overlaps, assign_reg_mask, sampled_mask \
                = self.subsample_rois(iou3d, cur_gt_cls)

            roi_batch_mask_indx = paddle.nonzero(roi_batch_mask).squeeze(-1)
            batch_roi_ious = paddle.scatter(batch_roi_ious, roi_batch_mask_indx, assign_overlaps)
            batch_roi_rcnn_labels = paddle.scatter(batch_roi_rcnn_labels, roi_batch_mask_indx, assign_gt_labels)
            batch_roi_gt_boxes = paddle.scatter(batch_roi_gt_boxes, roi_batch_mask_indx, cur_gt_box.index_select(assign_gt_ids, axis=0))
            batch_roi_gt_classes = paddle.scatter(batch_roi_gt_classes, roi_batch_mask_indx, cur_gt_cls.index_select(assign_gt_ids, axis=0))
            batch_roi_border_masks = paddle.scatter(batch_roi_border_masks, roi_batch_mask_indx, cur_gt_border_mask.index_select(assign_gt_ids, axis=0))
            batch_roi_reg_mask = paddle.scatter(batch_roi_reg_mask.cast('int64'), roi_batch_mask_indx, assign_reg_mask.cast('int64')).cast('bool')
            batch_roi_soft_labels = paddle.scatter(batch_roi_soft_labels, roi_batch_mask_indx, assign_soft_labels)
            batch_roi_sampled_mask = paddle.scatter(batch_roi_sampled_mask.cast('int64'), roi_batch_mask_indx, sampled_mask.cast('int64')).cast('bool')
        return batch_roi_ious, batch_roi_rcnn_labels, batch_roi_gt_boxes, batch_roi_gt_classes, batch_roi_border_masks, \
            batch_roi_reg_mask, batch_roi_soft_labels, batch_roi_sampled_mask
    def subsample_rois(self, iou3d, cur_gt_cls):
        # sample fg, easy_bg, hard_bg
        paddle.device.cuda.synchronize(paddle.CUDAPlace(0))
        max_overlaps = paddle.max(iou3d, axis=1)
        paddle.device.cuda.synchronize(paddle.CUDAPlace(0))
        argmax_overlaps = paddle.argmax(iou3d, axis=1)
        assign_gt_cls = cur_gt_cls[argmax_overlaps] #for compute class specific threshold
        assign_gt_labels = paddle.full((iou3d.shape[0],), -2, dtype=assign_gt_cls.dtype)
        assign_gt_ids = argmax_overlaps.clone().detach()
        assign_overlaps = max_overlaps.clone().detach()
        paddle.device.cuda.synchronize(paddle.CUDAPlace(0))

        if isinstance(self.roi_sampler_cfg['reg_fg_thresh'], list):
            # get cls specific threshold
            leng = self.reg_fg_thresh.shape[0]
            assign_gt_cls_ = assign_gt_cls % leng
            reg_fg_thresh = self.reg_fg_thresh[assign_gt_cls_]
            # get cls specific threshold
            cls_fg_thresh = self.cls_fg_thresh[assign_gt_cls_]
            cls_bg_thresh = self.cls_bg_thresh[assign_gt_cls_]
            fg_thresh = paddle.minimum(reg_fg_thresh, cls_fg_thresh)
            # get cls specific threshold
        else:
            reg_fg_thresh = self.roi_sampler_cfg['reg_fg_thresh']
            cls_fg_thresh = self.roi_sampler_cfg['cls_fg_thresh']
            cls_bg_thresh = self.roi_sampler_cfg['cls_bg_thresh']
            fg_thresh = min(reg_fg_thresh, cls_fg_thresh)

        
        # fg_inds = (max_overlaps >= cls_fg_thresh).nonzero().view(-1)
        # bg_inds = (max_overlaps < cls_bg_thresh).nonzero().view(-1)
        # fg_inds_reg = (max_overlaps >= reg_fg_thresh).nonzero().view(-1)
        assign_bg_mask = max_overlaps <= cls_bg_thresh
        assign_reg_mask = max_overlaps >= reg_fg_thresh
        # assign_gt_labels[assign_reg_mask] = assign_gt_cls[assign_reg_mask]
        assign_gt_labels = paddle.where(assign_reg_mask, assign_gt_cls, assign_gt_labels)
        # assign_gt_labels[assign_bg_mask] = -1
        assign_gt_labels = paddle.where(assign_bg_mask, -1 * paddle.ones(assign_gt_labels.shape, dtype=assign_gt_labels.dtype), assign_gt_labels)
        assign_soft_labels = (max_overlaps >= cls_fg_thresh).cast('float32')
        interval_mask = (max_overlaps < cls_fg_thresh) & (max_overlaps >= cls_bg_thresh)
        # assign_soft_labels[interval_mask] = (max_overlaps[interval_mask] - cls_bg_thresh[interval_mask]) / (cls_fg_thresh[interval_mask] - cls_bg_thresh[interval_mask])
        assign_soft_labels_temp = (max_overlaps - cls_bg_thresh) / (cls_fg_thresh - cls_bg_thresh)
        assign_soft_labels = paddle.where(interval_mask, assign_soft_labels_temp, assign_soft_labels)
        if not self.roi_sampler_cfg.get("use_sample", True):
            sampled_mask = paddle.ones_like(assign_gt_labels, dtype='bool')
            return assign_gt_labels, assign_soft_labels, assign_gt_ids, assign_overlaps, assign_reg_mask, sampled_mask

        fg_inds = (max_overlaps >= fg_thresh).nonzero().reshape([-1])
        easy_bg_inds = (max_overlaps < self.roi_sampler_cfg['cls_bg_thresh_lo']).nonzero().reshape([-1])
        hard_bg_inds = ((max_overlaps < fg_thresh) & (max_overlaps > self.roi_sampler_cfg['cls_bg_thresh_lo'])).nonzero().reshape([-1])
        # 采样样本
        fg_num_rois = int(fg_inds.numel())
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        roi_this_image = iou3d.shape[0]
        fg_rois_this_image = int(np.round(self.roi_sampler_cfg['fg_ratio'] * roi_this_image))
        bg_rois_this_image = int(roi_this_image * (1 - self.roi_sampler_cfg['fg_ratio']))
        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_sampled_num = min(fg_rois_this_image, fg_num_rois)

            # wangna11
            rand_num = paddle.to_tensor(np.random.permutation(fg_num_rois)).cast('int64')
            fg_inds = fg_inds[rand_num[:fg_rois_sampled_num]]
            # fg_inds = fg_inds[:fg_rois_sampled_num]

            # sampling bg
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_this_image, self.roi_sampler_cfg['hard_bg_ratio']
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(roi_this_image) * fg_num_rois)
            rand_num = paddle.to_tensor(rand_num).cast('int64')
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0]  # yield empty tensor

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_this_image, self.roi_sampler_cfg['hard_bg_ratio']
            )
        else:
            print("maxoverlaps:(min=%f, max=%f)" % (max_overlaps.min().item(), max_overlaps.max().item()))
            print("ERROR: FG=%d, BG=%d" % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_mask = paddle.zeros_like(assign_gt_labels, dtype='bool')
        if fg_inds.shape[0] > 0:
            sampled_mask[fg_inds] = True
        if bg_inds.shape[0] > 0:
            sampled_mask[bg_inds] = True
        return assign_gt_labels, assign_soft_labels, assign_gt_ids, assign_overlaps, assign_reg_mask, sampled_mask

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if bg_rois_per_this_image <= 0:
            bg_inds = paddle.ones([0,], dtype='int64')
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            # wangna11
            if hard_bg_rois_num > 0:
                rand_idx = paddle.randint(low=0, high=hard_bg_inds.numel(), shape=(hard_bg_rois_num,)).cast('int64')
                # rand_idx = paddle.arange(hard_bg_rois_num)
                hard_bg_inds = hard_bg_inds.index_select(rand_idx, axis=0)

            # sampling easy bg
            # wangna11
            if easy_bg_rois_num > 0:
                rand_idx = paddle.randint(low=0, high=easy_bg_inds.numel(), shape=(easy_bg_rois_num,)).cast('int64')
                # rand_idx = paddle.arange(easy_bg_rois_num)
                easy_bg_inds = easy_bg_inds.index_select(rand_idx, axis=0)

            if hard_bg_rois_num > 0 and easy_bg_rois_num > 0:
                bg_inds = paddle.concat([hard_bg_inds, easy_bg_inds], axis=0)
            elif hard_bg_rois_num > 0 and easy_bg_rois_num <= 0:
                bg_inds = hard_bg_inds
            elif hard_bg_rois_num <= 0 and easy_bg_rois_num > 0:
                bg_inds = easy_bg_inds
            else:
                bg_inds = paddle.ones([0,], dtype='int64')
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = paddle.randint(low=0, high=hard_bg_inds.numel(), shape=(hard_bg_rois_num,)).cast('int64')
            bg_inds = hard_bg_inds.index_select(rand_idx, axis=0)
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = paddle.randint(low=0, high=easy_bg_inds.numel(), shape=(easy_bg_rois_num,)).cast('int64')
            bg_inds = easy_bg_inds.index_select(rand_idx, axis=0)
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def sample_bg_indsv2(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        hard_bg_num = hard_bg_inds.numel()
        easy_bg_num = easy_bg_inds.numel()
        total_bg_num = hard_bg_num + easy_bg_num
        if total_bg_num < bg_rois_per_this_image:
            bg_inds = paddle.concat([hard_bg_inds, easy_bg_inds], axis=0)
        elif hard_bg_num > 0 and easy_bg_num > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), hard_bg_num)

            # sampling hard bg
            rand_num = paddle.to_tensor(np.random.permutation(len(hard_bg_inds))).cast('int64')
            rand_idx = rand_num[:hard_bg_rois_num]
            # rand_idx = paddle.randint(
            #     low=0, high=hard_bg_inds.numel(), shape=(hard_bg_rois_num,)).cast('int64')
            hard_bg_inds = hard_bg_inds[rand_idx]

            easy_bg_rois_num = min(bg_rois_per_this_image - hard_bg_rois_num, easy_bg_num)

            # sampling easy bg
            rand_num = paddle.to_tensor(np.random.permutation(len(easy_bg_inds))).cast('int64')
            rand_idx = rand_num[:easy_bg_rois_num]
            # rand_idx = paddle.randint(
            #     low=0, high=easy_bg_inds.numel(), shape=(easy_bg_rois_num,)).cast('int64')
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = paddle.concat([hard_bg_inds, easy_bg_inds], axis=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = min(bg_rois_per_this_image, hard_bg_num)
            # sampling hard bg
            # rand_idx = paddle.randint(
            #     low=0, high=hard_bg_inds.numel(), shape=(hard_bg_rois_num,)).cast('int64')
            rand_num = paddle.to_tensor(np.random.permutation(len(hard_bg_inds))).cast('int64')
            rand_idx = rand_num[:hard_bg_rois_num]
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = min(bg_rois_per_this_image, easy_bg_num)
            # sampling easy bg
            # rand_idx = paddle.randint(
            #     low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).cast('int64')
            rand_num = paddle.to_tensor(np.random.permutation(len(easy_bg_inds))).cast('int64')
            rand_idx = rand_num[:easy_bg_rois_num]
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    def generate_border_masks(self, boxes):
        """
                        t
        corner0 __________________corner3
                |                 |
                |                 |
            l   |                 |  r
                |                 |
        corner1 |_________________| corner2
                        b
        details: https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/38aiIFhMSE/l3w02XqQgMpnXx
        """
        assert len(boxes.shape) == 2
        bev_boxes = paddle.index_select(boxes, 
                                    paddle.to_tensor([0, 1, 3, 4, 6], dtype='int64'),
                                    axis=1)
        # corner0, corner1, corner2, corner3
        bev_corners = self.get_box_corners(bev_boxes)
        bev_origins = paddle.zeros_like(bev_corners)

        borders_start = paddle.concat([bev_corners, bev_corners[:, 0:2]], axis=1)  # (N, 6, 2)
        borders_end = paddle.concat([bev_corners[:, 1:], bev_corners[:, :3]], axis=1)  # (N, 6, 2)

        cross_results = []
        for i in range(2):
            cross_result = self.cross(
                bev_corners,
                bev_origins,
                borders_start[:, (i + 1) : (i + 5)],
                borders_end[:, (i + 1) : (i + 5)],
            )
            cross_results.append(cross_result)
        cross_results = paddle.stack(cross_results, axis=2)
        # true: non-cross, false: cross
        cross_results = ~((cross_results[..., 0].cast('int64') + cross_results[..., 1].cast('int64')).cast('bool'))
        pos_masks = paddle.concat([cross_results, cross_results[:, :1]], axis=1)

        # l, b, r, t
        border_masks = pos_masks[:, :4] * pos_masks[:, 1:]

        # self.draw_boxes(bev_corners, border_masks)
        return border_masks

    def get_box_corners(self, boxes):
        anglePis = boxes[:, 4]
        cxs, cys = boxes[:, 0], boxes[:, 1]
        ws, ls = boxes[:, 2], boxes[:, 3]

        rxs0 = cxs - (ws / 2) * paddle.cos(anglePis) + (ls / 2) * paddle.sin(anglePis)
        rys0 = cys - (ws / 2) * paddle.sin(anglePis) - (ls / 2) * paddle.cos(anglePis)

        rxs1 = cxs - (ws / 2) * paddle.cos(anglePis) - (ls / 2) * paddle.sin(anglePis)
        rys1 = cys - (ws / 2) * paddle.sin(anglePis) + (ls / 2) * paddle.cos(anglePis)

        rxs2 = cxs + (ws / 2) * paddle.cos(anglePis) - (ls / 2) * paddle.sin(anglePis)
        rys2 = cys + (ws / 2) * paddle.sin(anglePis) + (ls / 2) * paddle.cos(anglePis)

        rxs3 = cxs + (ws / 2) * paddle.cos(anglePis) + (ls / 2) * paddle.sin(anglePis)
        rys3 = cys + (ws / 2) * paddle.sin(anglePis) - (ls / 2) * paddle.cos(anglePis)

        rcorners0 = paddle.stack([rxs0, rys0], axis=1)
        rcorners1 = paddle.stack([rxs1, rys1], axis=1)
        rcorners2 = paddle.stack([rxs2, rys2], axis=1)
        rcorners3 = paddle.stack([rxs3, rys3], axis=1)
        rcorners = paddle.stack([rcorners0, rcorners1, rcorners2, rcorners3], axis=1)
        return rcorners

    def cross(self, s_p1, s_p2, d_p1, d_p2):
        v1 = s_p1 - d_p1
        v2 = s_p2 - d_p1
        vm = d_p2 - d_p1
        flag1 = self.cross_multi(v1, vm) * self.cross_multi(v2, vm) <= 0

        v1 = d_p1 - s_p1
        v2 = d_p2 - s_p1
        vm = s_p2 - s_p1
        flag2 = self.cross_multi(v1, vm) * self.cross_multi(v2, vm) <= 0

        flag = flag1 * flag2
        return flag

    def cross_multi(self, v1, v2):
        # calculate cross multiply
        return v1[:, :, 0] * v2[:, :, 1] - v2[:, :, 0] * v1[:, :, 1]

    def downsample_points(self, example):
        """Apply dynamic voxelization to points.

        Args:
            points (list[paddle.Tensor]): Points of each sample.

        Returns:
            tuple[paddle.Tensor]: Concatenated points and coordinates.
        """
        batch_anchors = example["anchors"]
        batch_size = batch_anchors[0].shape[0]
        points = example['pts_features']
        coords = example['coordinates']

        new_points = paddle.concat([coords[:,0:1], points, coords], axis=-1)
        ds_new_points = self.voxel_downsample(new_points)
        ds_points1 = ds_new_points[:, 1:-4]
        ds_coors1 = ds_new_points[:, -4:].cast('int64')

        return ds_points1, ds_coors1