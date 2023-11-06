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
import logging
from collections import defaultdict, OrderedDict
from enum import Enum
import numpy as np
from paddle3d.apis import manager
import paddle.nn as nn
from paddle3d.utils import DeltaXYZWLHRBBoxCoderIDG
from paddle3d.utils_idg.sub_region_utils import get_class2id
from paddle3d.models.losses import CrossEntropyLossIDG
from paddle3d.utils_idg.build_layer import build_norm_layer
from paddle3d.utils_idg.box_paddle_ops import center_to_corner_box2d, corner_to_standup_nd, nms, nms_overlap, rotate_nms_overlap # rotate_nms
from paddle3d.models.layers.layer_libs import rotate_nms_pcdet # use this rotate nms instead
from paddle3d.utils_idg.ops import iou3d_utils
from paddle3d.utils_idg.target_ops import assign_weight_to_voxel, assign_label_to_voxel, assign_label_to_box
from paddle3d.geometries import BBoxes3D
from paddle3d.models.layers import param_init, reset_parameters, constant_init, kaiming_normal_init

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
    
def build_bbox_coder(bbox_coder):
    assert bbox_coder['type_name'] 
    bbox_coder.pop("type_name")
    return DeltaXYZWLHRBBoxCoderIDG(**bbox_coder)

def build_loss_cls(loss_cls):
    assert loss_cls['type_name'] 
    loss_cls.pop("type_name")
    return CrossEntropyLossIDG(**loss_cls)

def one_hot_f(tensor, depth, dim=-1, on_value=1.0, dtype='float32'):
    tensor_onehot = paddle.zeros(shape=[*list(tensor.shape), depth], dtype=
        dtype)
    tensor_onehot.put_along_axis(indices = tensor.unsqueeze(axis=dim).cast('int64'), values = on_value, axis = dim)
    return tensor_onehot


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = paddle.sin(x=boxes1[(...), -1:]) * paddle.cos(x=
        boxes2[(...), -1:])
    rad_tg_encoding = paddle.cos(x=boxes1[(...), -1:]) * paddle.sin(x=
        boxes2[(...), -1:])
    boxes1 = paddle.concat(x=[boxes1[(...), :-1], rad_pred_encoding], axis=-1)
    boxes2 = paddle.concat(x=[boxes2[(...), :-1], rad_tg_encoding], axis=-1)
    return boxes1, boxes2


def _get_pos_neg_loss(cls_loss, labels):
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).cast(dtype=cls_loss.dtype
            ) * cls_loss.reshape((batch_size, -1))
        cls_neg_loss = (labels == 0).cast(dtype=cls_loss.dtype
            ) * cls_loss.reshape((batch_size, -1)) 
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0.0):
    batch_size = reg_targets.shape[0]
    anchors = anchors.reshape((batch_size, -1, anchors.shape[-1]))
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = (rot_gt - dir_offset > 0).cast(dtype='int64')
    if one_hot:
        dir_cls_targets = one_hot_f(dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets


def smooth_l1_loss(pred, gt, sigma):

    def _smooth_l1_loss(pred, gt, sigma):
        sigma2 = sigma ** 2
        cond_point = 1 / sigma2
        x = pred - gt
        abs_x = paddle.abs(x=x)
        in_mask = abs_x < cond_point
        out_mask = 1 - in_mask
        in_value = 0.5 * (sigma * x) ** 2
        out_value = abs_x - 0.5 / sigma2
        value = in_value * in_mask.astype(dtype=in_value.dtype
            ) + out_value * out_mask.astype(dtype=out_value.dtype)
        return value
    value = _smooth_l1_loss(pred, gt, sigma)
    loss = value.mean(axis=1).sum()
    return loss


def smooth_l1_loss_detectron2(input, target, beta: float, reduction: str='none'
    ):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    Smooth L1 loss is related to Huber loss, which is defined as:
                | 0.5 * x ** 2                  if abs(x) < beta
     huber(x) = |
                | beta * (abs(x) - 0.5 * beta)  otherwise
    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:
     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.
    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.
    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        The loss with the reduction option applied.
     """
    if beta < 1e-05:
        loss = paddle.abs(x=input - target)
    else:
        n = paddle.abs(x=input - target)
        cond = n < beta
        loss = paddle.where(condition=cond, x=0.5 * n ** 2 / beta, y=n - 0.5 * beta)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


class LossNormType(Enum):
    NormByNumPoints = 'norm_by_num_points'
    NormByNumProposals = 'norm_by_num_proposals'
    NormByNumVoxels = 'norm_by_num_voxels'
    DontNorm = 'dont_norm'


@manager.HEADS.add_component
class ConfidenceHead(paddle.nn.Layer):

    def __init__(self, mode='3d', 
                        in_channels=[128], 
                        norm_cfg=None, 
                        tasks=[], 
                        weights=[], 
                        bbox_coder=None,
                        with_cls=True,
                        encode_background_as_zeros=True, 
                        grid_size=[600, 600, 1],
                        voxel_size=[0.2, 0.2, 10], 
                        pc_range=[-60, -60, -5, 60, 60, 5],
                        spatial_scale=1.0, 
                        rpn_cfg=None, 
                        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), 
                        loss_norm=dict(type='NormByNumVoxels'), 
                        ignore_invalid_voxel=False,
                        start_epoch=0, 
                        loss_weight=1.0, 
                        direction_offset=0.0,
                        use_sigmoid_score=True, 
                        use_direction_classifier=True, 
                        name='confmap', 
                        logger=None, 
                        use_sub_region_head=False, 
                        sub_region_attr=None):
        super(ConfidenceHead, self).__init__()
        assert with_cls
        self.tasks = tasks
        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.box_coder = build_bbox_coder(bbox_coder)
        box_code_sizes = [self.box_coder.code_size] * len(num_classes)
        self.with_cls = with_cls
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encode_background_as_zeros = encode_background_as_zeros
        self.use_sigmoid_score = use_sigmoid_score
        self.box_n_dim = self.box_coder.code_size
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.spatial_scale = spatial_scale
        self.rpn_cfg = rpn_cfg
        self.loss_norm = loss_norm
        self.start_epoch = start_epoch
        self.loss_weight = loss_weight
        self.ignore_invalid_voxel = ignore_invalid_voxel
        self.direction_offset = direction_offset
        self.use_direction_classifier = use_direction_classifier
        self.use_sub_region_head = use_sub_region_head
        if self.use_sub_region_head:
            assert sub_region_attr is not None, 'ValueError: sub_region_attr should not be none when use sub_region head!'
            self.sub_region_class = sub_region_attr.get('sub_region_class', None)
            self.sub_region_range = sub_region_attr.get('sub_region_range', None)
            self.sub_region_postfix = sub_region_attr.get('sub_region_postfix', None)
            assert self.sub_region_class is not None, 'ValueError: sub_region_class should not be none when use sub_region head!'
            assert self.sub_region_range is not None, 'ValueError: sub_region_range should not be none when use sub_region head!'
            assert self.sub_region_postfix is not None, 'ValueError: sub_region_postfix should not be none when use sub_region head!'
            self.class2id = get_class2id(self.tasks, self.sub_region_postfix)
        else:
            self.class2id = get_class2id(self.tasks)
        loss_cls['reduction'] = 'none'
        self.loss_cls = build_loss_cls(loss_cls)
        if not logger:
            logger = logging.getLogger('ConfidenceHead')
        self.logger = logger
        self.dcn = None
        self.zero_init_residual = False
        if norm_cfg is None:
            norm_cfg = dict(type='BN', eps=0.001, momentum=1-0.01)
        self._norm_cfg = norm_cfg
        self.last_layer = paddle.nn.Sequential(
                    paddle.nn.Conv2DTranspose(in_channels=in_channels, 
                        out_channels=in_channels // 2,kernel_size=2, stride=2, bias_attr=False), 
                    build_norm_layer(self._norm_cfg, self.in_channels // 2)[1], 
                    paddle.nn.ReLU(),
                    paddle.nn.Conv2D(in_channels=in_channels // 2, out_channels=1,
                        kernel_size=1, stride=1, padding=0))
        self.init_weights()
        logger.info('Finish ConfidenceHead Initialization')

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            state_dict = paddle.load(pretrained)
            self.set_dict(state_dict)
        elif pretrained is None:
            for m in self.sublayers():
                if isinstance(m, paddle.nn.Conv2D) or isinstance(m, paddle.nn.Conv2DTranspose):
                    kaiming_normal_init(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        constant_init(m.bias, value=0.0)
                elif isinstance(m, (nn.layer.norm._BatchNormBase, nn.layer.norm.LayerNorm, nn.layer.norm._InstanceNormBase)):
                    constant_init(m.weight, value = 1)
                    constant_init(m.bias, value = 0)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x):
        x_score = self.last_layer(x[0])
        return x_score

    def proposal_layer(self, batch_anchors, cls_scores, bbox_preds,
        gt_bboxes, gt_labels, anchors_mask, labels):
        batch_size = batch_anchors[0].shape[0]
        proposals = [[] for i in range(batch_size)]
        for task_id, batch_cls_preds in enumerate(cls_scores):
            num_class = self.num_classes[task_id]
            batch_task_anchors = batch_anchors[task_id].reshape((batch_size, -1, self.box_n_dim))
            batch_task_labels = labels[task_id]
            # if anchors_mask[task_id] is None:
            #     batch_anchors_mask = [None] * batch_size
            # else:
            #     batch_anchors_mask = anchors_mask[task_id].reshape((batch_size, -1))
            batch_anchors_mask = (batch_task_labels >=0).reshape((batch_size, -1)) # .view(batch_size, -1)

            batch_box_preds = bbox_preds[task_id]
            box_ndim = self.box_n_dim
            batch_box_preds = batch_box_preds.reshape((batch_size, -1, box_ndim)) 
            num_class_with_bg = self.num_classes[task_id]
            if not self.encode_background_as_zeros:
                num_class_with_bg = self.num_classes[task_id] + 1
            batch_cls_preds = batch_cls_preds.reshape((batch_size, -1, num_class_with_bg))
            batch_reg_preds = self.box_coder.decode(batch_task_anchors,
                batch_box_preds[:, :, :self.box_coder.code_size])
            for batch_id in range(batch_size):
                box_preds = batch_reg_preds[batch_id]
                cls_preds = batch_cls_preds[batch_id]
                a_mask = batch_anchors_mask[batch_id]
                if a_mask is not None:
                    # to avoid empty tensor by bool mask
                    # box_preds = box_preds[a_mask]   
                    # cls_preds = cls_preds[a_mask]
                    a_index = paddle.where(a_mask)[0].squeeze(-1)
                    box_preds = box_preds.index_select(a_index)
                    cls_preds = cls_preds.index_select(a_index)
                box_preds = box_preds.astype(dtype='float32')
                cls_preds = cls_preds.astype(dtype='float32')
                if self.encode_background_as_zeros:
                    assert self.use_sigmoid_score is True
                    total_scores = nn.functional.sigmoid(x=cls_preds)
                elif self.use_sigmoid_score:
                    total_scores = nn.functional.sigmoid(x=cls_preds)[..., 1:]
                else:
                    total_scores = nn.functional.softmax(x=cls_preds, axis=-1)[..., 1:]
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(axis=-1)
                else:
                    top_scores = paddle.max(total_scores, axis = -1)
                if self.rpn_cfg['score_threshold'] > 0.0:
                    thresh = paddle.to_tensor(data=[self.rpn_cfg['score_threshold']], dtype=total_scores.dtype)
                    top_scores_keep = top_scores >= thresh
                    top_scores = top_scores.masked_select(mask=top_scores_keep)
                if top_scores.shape[0] != 0:
                    if self.rpn_cfg['score_threshold'] > 0.0:
                        box_preds = box_preds[top_scores_keep]
                    if self.rpn_cfg['nms_pre_max_size'] is not None:
                        top_scores, indices = paddle.topk(x=top_scores, k = min(self.rpn_cfg['nms_pre_max_size'], top_scores.shape[0]))
                        # print("box_preds shape 0: ", box_preds.shape)
                        # print("indices: ", indices.shape)
                        if indices.shape[0]==1:
                            box_preds = box_preds[indices].unsqueeze(0)
                        else:
                            box_preds = box_preds[indices]
                        # box_preds = box_preds[indices]  # TODO adapt2dev yipin
                        if self.rpn_cfg['use_nms'] != -1:
                            inds = paddle.to_tensor([0, 1, 3, 4, 6]) %  box_preds.shape[1]
                            # print("inds: ", inds)

                            # print("box_preds: ", box_preds.shape)
                            boxes_for_nms = box_preds.index_select(inds, axis = 1) # box_preds[:, ([0, 1, 3, 4, -1])]
                            # print("boxes_for_nms", boxes_for_nms.shape)
                            box_preds_corners = (center_to_corner_box2d(boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:,4]))
                            boxes_for_nms = corner_to_standup_nd(box_preds_corners)
                            boxes_for_nms = paddle.concat([boxes_for_nms, box_preds[:, -1:]], axis=-1)
                            keep_idx = iou3d_utils.nms_normal_gpu_nosort(boxes_for_nms, top_scores, self.rpn_cfg['nms_iou_threshold'])  # TODO: zhuyipin sort cause invalid config problem
                            selected = keep_idx[:self.rpn_cfg['nms_post_max_size']]
                            if selected.shape[0]==1:
                                box_preds = box_preds[selected].unsqueeze(0)
                            else:
                                box_preds = box_preds[selected]
                            # box_preds = box_preds[selected] # TODO adapt2dev yipin
                    proposals[batch_id].append(box_preds)
        if self.rpn_cfg['use_gt_boxes']:
            for task_id, task in enumerate(self.tasks):
                gt_boxes_task = gt_bboxes[task_id]
                gt_classes_task = gt_labels[task_id]
                for batch_id in range(batch_size):
                    if gt_classes_task[batch_id].shape[0] > 0:
                        mask = gt_classes_task[batch_id] > 0
                        proposals[batch_id].append(gt_boxes_task[batch_id][mask])
                    else:
                        proposals[batch_id].append(paddle.empty(gt_boxes_task[batch_id].shape, gt_boxes_task[batch_id].dtype))
        proposals = [cat(proposal, axis=0) for proposal in proposals]
        return proposals

    def assign_gtbox_to_mask(self, batch_anchors, gt_bboxes, gt_labels,
        num_points_in_gts=None):
        batch_size = batch_anchors[0].shape[0]
        if not hasattr(self, 'corners_norm'):
            self.corners_norm = paddle.to_tensor(data=[[-0.5, -0.5], [-0.5,
                0.5], [0.5, -0.5], [0.5, 0.5]], dtype=batch_anchors[0].dtype)
            self.corners_norm = self.corners_norm[[0, 1, 3, 2]]
        batch_pos_gt_boxes = [[] for i in range(batch_size)]
        batch_ignore_gt_boxes = [[] for i in range(batch_size)]
        for task_id, task in enumerate(self.tasks):
            gt_boxes_task = gt_bboxes[task_id]
            gt_classes_task = gt_labels[task_id]
            if num_points_in_gts is not None:
                num_points_in_gts_task = num_points_in_gts[task_id]
            for batch_id in range(batch_size):
                if gt_classes_task[batch_id].shape[0] > 0:
                    mask = gt_classes_task[batch_id] > 0
                    if num_points_in_gts is not None:
                        pos_mask = num_points_in_gts_task[batch_id] > 0
                        ignore_mask = num_points_in_gts_task[batch_id] <= 0
                        mask_pos = mask & pos_mask
                        mask_ignore = mask & ignore_mask
                    else:
                        mask_pos = mask
                        mask_ignore = paddle.full(shape=mask.shape, fill_value=
                            False).astype(mask.dtype)
                    batch_pos_gt_boxes[batch_id].append(gt_boxes_task[batch_id][mask_pos])
                    batch_ignore_gt_boxes[batch_id].append(gt_boxes_task[batch_id][mask_ignore])
                else:
                    batch_pos_gt_boxes[batch_id].append(gt_boxes_task[batch_id])
                    batch_ignore_gt_boxes[batch_id].append(gt_boxes_task[batch_id])

        batch_pos_gt_boxes = [cat(batch_gt_box, axis=0) for batch_gt_box in batch_pos_gt_boxes] 
        batch_ignore_gt_boxes = [cat(batch_gt_box, axis=0) for batch_gt_box in batch_ignore_gt_boxes] 

        pos_target_masks = []
        ignore_target_masks = []
        bev_pos_gt_boxes = [batch_gt_box.index_select(paddle.to_tensor([0, 1, 3, 4, 6]), axis = 1) for batch_gt_box in batch_pos_gt_boxes]  #[batch_gt_box[:, ([0, 1, 3, 4, 6])] for batch_gt_box in batch_pos_gt_boxes]
        bev_ignore_gt_boxes = [batch_gt_box.index_select(paddle.to_tensor([0, 1, 3, 4, 6]), axis = 1) for batch_gt_box in batch_ignore_gt_boxes]  #[batch_gt_box[:, ([0, 1, 3, 4, 6])] for batch_gt_box in batch_ignore_gt_boxes]
        for i in range(batch_size):
            pos_target_mask = assign_label_to_voxel(bev_pos_gt_boxes[i], self.grid_size,
                self.voxel_size, self.pc_range, self.corners_norm)
            ignore_target_mask = assign_label_to_voxel(bev_ignore_gt_boxes[i], 
                                        self.grid_size, self.voxel_size, self.pc_range, self.corners_norm)
            pos_target_masks.append(pos_target_mask.reshape((self.grid_size[1], self.grid_size[0])))
            ignore_target_masks.append(ignore_target_mask.reshape((self.grid_size[1], self.grid_size[0])))
            
        pos_target_masks = paddle.stack(x=pos_target_masks, axis=0).reshape((batch_size, -1, 1))
        ignore_target_masks = paddle.stack(x=ignore_target_masks, axis=0).reshape((batch_size, -1, 1))
        return pos_target_masks, ignore_target_masks

    def assign_proposal_to_mask(self, proposals):
        batch_size = len(proposals)
        if not hasattr(self, 'corners_norm'):
            self.corners_norm = paddle.to_tensor(data=[[-0.5, -0.5], [-0.5,
                0.5], [0.5, -0.5], [0.5, 0.5]], dtype=proposals[0].dtype)
            self.corners_norm = self.corners_norm[[0, 1, 3, 2]]
        loss_norm_type = getattr(LossNormType, self.loss_norm['type_name'])
        proposal_masks = []
        bev_proposal_boxes = [proposal_boxes.index_select(paddle.to_tensor([0, 1, 3, 4, 6]), axis = 1) for proposal_boxes in proposals] #[proposal_boxes[:, ([0, 1, 3, 4, 6])] for proposal_boxes in proposals]
        for i in range(batch_size):
            if loss_norm_type == LossNormType.NormByNumProposals:
                proposal_mask = assign_weight_to_voxel(bev_proposal_boxes[i], self.grid_size, self.voxel_size, self.pc_range, self.
                    corners_norm, expand_ratio=1.0)
                proposal_mask = proposal_mask / max(bev_proposal_boxes[i].
                    shape[0], 1.0)
            else:
                proposal_mask = assign_label_to_voxel(bev_proposal_boxes[i], self.grid_size, self.voxel_size, 
                                                    self.pc_range, self.corners_norm, expand_ratio=1.0)
                if loss_norm_type == LossNormType.NormByNumPoints:
                    proposal_mask = proposal_mask.cast(dtype='float32') / paddle.clip(x=paddle.sum(x=proposal_mask.reshape((-1, )) > 0), min=1.0)
                else:
                    proposal_mask = proposal_mask.astype(dtype='float32') / (self.grid_size[1] * self.grid_size[0])
            proposal_masks.append(proposal_mask.reshape((self.grid_size[0],
                self.grid_size[1])))
        proposal_masks = paddle.stack(x=proposal_masks, axis=0)
        return proposal_masks

    def get_confidences(self, confidence_map, box3d_lidar):
        # print("p5: ", box3d_lidar)
        bev_box = box3d_lidar.index_select(paddle.to_tensor([0, 1, 3, 4, 6]), axis = 1)#box3d_lidar[:, ([0, 1, 3, 4, 6])]
        # print("p6: ", box3d_lidar)
        confidences = assign_label_to_box(bev_box, confidence_map, self.grid_size, self.voxel_size, self.
            pc_range)
        # print("p7: ", box3d_lidar)
        return confidences

    def loss(self, 
            roi_preds, 
            cls_scores, 
            bbox_preds, 
            dir_cls_preds,
            iou_preds,
            bctp_preds,
            gt_bboxes, 
            gt_labels, 
            input_metas, 
            coors, 
            batch_anchors,
            anchors_mask,
            labels, 
            num_points_in_gts=None, 
            gt_bboxes_ignore=None):

        batch_size = batch_anchors[0].shape[0]
        proposals = self.proposal_layer(batch_anchors, cls_scores,
            bbox_preds, gt_bboxes, gt_labels, anchors_mask, labels)
        proposals_masks = self.assign_proposal_to_mask(proposals).reshape((batch_size, -1, 1))
        confidences_map = roi_preds
        confidences_map = confidences_map.reshape((batch_size, -1, 1)) 
        with paddle.no_grad():
            target_masks, ignore_target_masks = self.assign_gtbox_to_mask(
                batch_anchors, gt_bboxes, gt_labels, num_points_in_gts)
        for i in range(batch_size):
            ignore_target_masks = ignore_target_masks.astype(dtype='int64')
            target_masks[i][ignore_target_masks[i] == True] = 0
            proposals_masks[i][ignore_target_masks[i] == True] = 0
            confidences_map[i][ignore_target_masks[i] == True] = -100
            if self.ignore_invalid_voxel:
                batch_mask = coors[:, 0] == i
                this_coords = coors[(batch_mask), :]
                indices = this_coords[:, 2] * self.grid_size[0] + this_coords[:, 3]
                indices = indices.cast('int64')
                voxel_mask = paddle.zeros(shape=(self.grid_size[1] * self.grid_size[0],), dtype='bool')
                voxel_mask[indices] = True
                target_masks[i][~voxel_mask] = 0
                proposals_masks[i][~voxel_mask] = 0
                confidences_map[i][~voxel_mask] = -100
        cls_losses = self.loss_cls(confidences_map, target_masks, weight=
            proposals_masks)
        cls_losses = cls_losses.sum() / batch_size
        roi_losses = {}
        roi_losses['loss_roi_head'] = cls_losses * self.loss_weight
        return roi_losses

    @paddle.jit.not_to_static
    def list_rets(self, rets, ori_class_name_map):
        num_samples = len(rets[0])
        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ['box3d_lidar', 'scores', 'iou_scores']:
                    if rets[0][i][k] is not None:
                        retlist = []
                        for re in rets:
                            if not re[0][k].shape[0] == 0:
                                retlist.append(re)
                        
                        if len(retlist) > 0:
                            ret[k] = paddle.concat([re[i][k] for re in retlist])
                        else:
                            ret[k] = paddle.empty((0, 7))

                    else:
                        ret[k] = None
                elif k in ['label_preds']:
                    flag = 0
                    # for j, num_class in enumerate(ori_class_name):
                    #     if self.use_sub_region_head and self.class_names[j][0].endswith(self.sub_region_postfix):
                    #         continue

                    #     # added
                    #     if self.class_names[j][0] == 'verybigMot':
                    #         rets[j][i][k] += 2 # merge 'verybigMot' and 'bigMot'
                    #     else:
                    #         rets[j][i][k] += flag

                    #     flag += num_class

                    for j, (cls_name,cls_id) in enumerate(ori_class_name_map.items()):
                        rets[j][i][k] += cls_id  # recover class id, such as 4th class

                    retlist = []
                    for re in rets:
                        if not re[0][k].shape[0] == 0:
                            retlist.append(re)
                    
                    if len(retlist) > 0:
                        ret[k] = paddle.concat([re[i][k] for re in retlist])
                    else:
                        ret[k] = paddle.empty((0, 7))

                elif k == 'metadata':
                    ret[k] = rets[0][i][k]
            ret_list.append(ret)
        return ret_list

    def get_bboxes(self, 
                   roi_preds, 
                   cls_scores, 
                   bbox_preds,
                   dir_cls_preds, 
                   iou_preds,
                   bctp_preds,
                   input_metas, 
                   coors, 
                   batch_anchors, 
                   anchors_mask,
                   test_cfg, 
                   **kwargs):
        """start with v1.6.0, this function don't contain any kitti-specific code.
        Returns:
            predict: list of pred_dict.
            pred_dict: {
                box3d_lidar: [N, 7] 3d box.
                scores: [N]
                label_preds: [N]
                metadata: meta-data which contains dataset-specific information.
                    for kitti, it contains image idx (label idx),
                    for nuscenes, sample_token is saved in it.
            }
        """
        # assert len(anchors_mask) == len(cls_scores)
        if anchors_mask is None: 
            anchors_mask = [None for _ in range(len(self.tasks))] 
           
        assert (len(anchors_mask) == len(cls_scores)) or ((len(anchors_mask) == len(cls_scores) + 1))

        batch_size_device = batch_anchors[0].shape[0]
        rets = []
        if self.use_sub_region_head:
            sub_region_result = dict()
            out_region_result = dict()
        for task_id, cls_score in enumerate(cls_scores):
            batch_size = batch_anchors[task_id].shape[0]
            batch_task_anchors = batch_anchors[task_id].reshape((batch_size, -1, self.box_n_dim))
            if anchors_mask[task_id] is None:
                batch_anchors_mask = [None] * batch_size
            else:
                batch_anchors_mask = anchors_mask[task_id].reshape((batch_size, -1)) #.view(batch_size, -1)
            batch_box_preds = bbox_preds[task_id]
            batch_cls_preds = cls_score
            box_ndim = self.box_n_dim
            if kwargs.get('mode', False):
                batch_box_preds_base = batch_box_preds.reshape((batch_size, -1, box_ndim))
                batch_box_preds = batch_task_anchors # .clone()
                batch_box_preds[:, :, ([0, 1, 3, 4, 6])] = batch_box_preds_base
            else:
                batch_box_preds = batch_box_preds.reshape((batch_size, -1, box_ndim))
            num_class_with_bg = self.num_classes[task_id]
            if not self.encode_background_as_zeros:
                num_class_with_bg = self.num_classes[task_id] + 1
            batch_cls_preds = batch_cls_preds.reshape((batch_size, -1, num_class_with_bg))
            batch_reg_preds = self.box_coder.decode(batch_task_anchors, batch_box_preds[:, :, :self.box_coder.code_size])
            if self.use_direction_classifier:
                batch_dir_preds = dir_cls_preds[task_id]
                batch_dir_preds = batch_dir_preds.reshape((batch_size, -1, 2)) #.view(batch_size, -1, 2)
            else:
                batch_dir_preds = [None] * batch_size
            if None not in iou_preds:
                batch_iou_preds = iou_preds[task_id]
            else:
                batch_iou_preds = [None] * batch_size
            if batch_iou_preds[0] is not None:
                batch_iou_preds = batch_iou_preds.reshape((batch_size, -1, 1)) #.view(batch_size, -1, 1)
            if self.use_sub_region_head:
                cur_class_name = self.class_names[task_id]
                if cur_class_name in self.sub_region_class:
                    sub_region_mask = batch_anchors[task_id][:, :, (0)
                        ] > self.sub_region_range[0][0]
                    sub_region_mask = sub_region_mask & (batch_anchors[
                        task_id][:, :, (0)] < self.sub_region_range[0][1])
                    sub_region_mask = sub_region_mask & (batch_anchors[
                        task_id][:, :, (1)] > self.sub_region_range[1][0])
                    sub_region_mask = sub_region_mask & (batch_anchors[
                        task_id][:, :, (1)] < self.sub_region_range[1][1])
                    sub_region_mask = ~sub_region_mask
                    batch_reg_preds = batch_reg_preds[sub_region_mask].reshape((batch_size, -1, 7))
                    batch_cls_preds = batch_cls_preds[sub_region_mask].reshape((batch_size, -1, 1))
                    batch_dir_preds = batch_dir_preds[sub_region_mask].reshape((batch_size, -1, 2))
                    batch_anchors_mask = batch_anchors_mask[sub_region_mask].reshape((batch_size, -1))
                    if batch_iou_preds[0] is not None:
                        batch_iou_preds = batch_iou_preds[sub_region_mask].reshape((batch_size, -1, 1))
                        out_region_result[cur_class_name[0]] = [batch_reg_preds
                            , batch_cls_preds, batch_dir_preds,
                            batch_anchors_mask, batch_iou_preds]
                    else:
                        out_region_result[cur_class_name[0]] = [batch_reg_preds
                            , batch_cls_preds, batch_dir_preds,
                            batch_anchors_mask]
                    if not sub_region_result.get(cur_class_name[0]):
                        continue
                if cur_class_name[0].endswith(self.sub_region_postfix):
                    if batch_iou_preds[0] is not None:
                        sub_region_result[cur_class_name[0][0:-len(self.
                            sub_region_postfix)]] = [batch_reg_preds,
                            batch_cls_preds, batch_dir_preds,
                            batch_anchors_mask, batch_iou_preds]
                    else:
                        sub_region_result[cur_class_name[0][0:-len(self.
                            sub_region_postfix)]] = [batch_reg_preds,
                            batch_cls_preds, batch_dir_preds,
                            batch_anchors_mask]
                    if not out_region_result.get(cur_class_name[0][0:-len(
                        self.sub_region_postfix)]):
                        continue
                task_id = self.class2id.get(cur_class_name[0])
                ori_class_name = self.class_names[task_id][0]
                if sub_region_result.get(ori_class_name
                    ) and out_region_result.get(ori_class_name):
                    batch_reg_preds = paddle.concat(x=(out_region_result.
                        get(ori_class_name)[0], sub_region_result.get(
                        ori_class_name)[0]), axis=1)
                    batch_cls_preds = paddle.concat(x=(out_region_result.
                        get(ori_class_name)[1], sub_region_result.get(
                        ori_class_name)[1]), axis=1)
                    batch_dir_preds = paddle.concat(x=(out_region_result.
                        get(ori_class_name)[2], sub_region_result.get(
                        ori_class_name)[2]), axis=1)
                    batch_anchors_mask = paddle.concat(x=(out_region_result
                        .get(ori_class_name)[3], sub_region_result.get(
                        ori_class_name)[3]), axis=1)
                    if batch_iou_preds[0] is not None:
                        batch_iou_preds = paddle.concat(x=(
                            out_region_result.get(ori_class_name)[4],
                            sub_region_result.get(ori_class_name)[4]), axis=1)
            rets.append(self.get_task_detections(task_id, num_class_with_bg,
                test_cfg, batch_cls_preds, batch_reg_preds, batch_dir_preds,
                batch_iou_preds, batch_anchors_mask))
        
        if ['pedestrian_sub'] in self.class_names and ['verybigMot'] in self.class_names:
            if self.class_names.index(['pedestrian_sub']) > self.class_names.index(['verybigMot']):
                ped_pred = rets[5]
                verybigmot_pred = rets[4]
                rets[4] = ped_pred
                rets[5] = verybigmot_pred

        num_tasks = len(rets)
        num_preds = len(rets)

        # ori_class_name = []
        # for cur_classe_name in self.class_names:
        #     if cur_classe_name[0].endswith('_sub'):
        #         continue
        #     ori_class_name.append(1)

        ori_class_name_map = OrderedDict()
        flag = 0
        for i, cur_classe_name in enumerate(self.class_names):
            
            for cls_name in cur_classe_name:
                if cls_name.endswith('_sub'):
                    continue
                ori_class_name_map[cls_name] = flag
                flag += 1
                if cls_name =='verybigMot':
                    ori_class_name_map[cls_name] = ori_class_name_map['bigMot']
                    flag -= 1

        assert len(ori_class_name_map) == len(rets)

        ret_list = self.list_rets(rets, ori_class_name_map)
        
        confidences_map = roi_preds
        confidences_map = paddle.nn.functional.sigmoid(x=confidences_map.squeeze())
        if self.ignore_invalid_voxel:
            confidences_map = confidences_map.reshape((batch_size, -1)) 
            for i in range(batch_size):
                batch_mask = coors[:, (0)] == i
                this_coords = coors[(batch_mask), :]
                indices = this_coords[:, (2)] * self.grid_size[0
                    ] + this_coords[:, (3)]
                indices = indices.cast('int64')
                voxel_mask = paddle.zeros(shape=(self.grid_size[1] * self.
                    grid_size[0],), dtype='bool')
                voxel_mask[indices] = True
                confidences_map[i][~voxel_mask] = -1
            confidences_map = confidences_map.reshape((batch_size, self.grid_size[1], self.grid_size[0]))

        for i, ret in enumerate(ret_list):
            # print("p3: ", ret['box3d_lidar'])
            box3d_lidar = ret['box3d_lidar']
            confidences = self.get_confidences(confidences_map[i], box3d_lidar)
            # print("p5: ", box3d_lidar)
            # print("p4: ", ret['box3d_lidar'])
            ret['confidences'] = confidences
        nms_groups = test_cfg["nms"].get('nms_groups', [])
        if len(nms_groups) != 0:
            if not isinstance(nms_groups[0][0], int):
                class_names = []
                for names in self.class_names:
                    class_names.extend(names)
                for i, group in enumerate(nms_groups):
                    for j, name in enumerate(group):
                        nms_groups[i][j] = self.class2id[name]
            for i, ret in enumerate(ret_list):
                # print("pr2: ", ret['box3d_lidar'])
                if ret['box3d_lidar'].shape[0] == 0:
                    continue
                # print("pr0: ", ret['box3d_lidar']) # TODO
                # print("pr0: type", type(ret['box3d_lidar'])) # TODO
                (box3d_lidar, scores, confidences, iou_scores, label_preds) = (
                    self.nms_for_groups(test_cfg, nms_groups, ret['box3d_lidar'], ret['scores'], ret['iou_scores'], ret['confidences'], ret['label_preds']))
                ret_list[i]['box3d_lidar'] = box3d_lidar
                ret_list[i]['scores'] = scores
                ret_list[i]['iou_scores'] = iou_scores
                ret_list[i]['confidences'] = confidences
                ret_list[i]['label_preds'] = label_preds
                # ret_list[i]['box3d_lidar'] = ret['box3d_lidar']
                # ret_list[i]['scores'] = ret['scores']
                # ret_list[i]['iou_scores'] = ret['iou_scores']
                # ret_list[i]['confidences'] = ret['confidences']
                # ret_list[i]['label_preds'] = ret['label_preds']

        nms_overlap_groups = test_cfg['nms'].get('nms_overlap_groups', [])
        if len(nms_overlap_groups) != 0:
            if not isinstance(nms_overlap_groups[0][0], int):
                class_names = []
                for names in self.class_names:
                    class_names.extend(names)
                for i, group in enumerate(nms_overlap_groups):
                    for j, name in enumerate(group):
                        nms_overlap_groups[i][j] = self.class2id[name]
            for i, ret in enumerate(ret_list):
                if ret['box3d_lidar'].shape[0] == 0:
                    continue
                (box3d_lidar, scores, confidences, iou_scores, label_preds) = (
                    self.nms_overlap_for_groups(test_cfg,
                    nms_overlap_groups, ret['box3d_lidar'], ret['scores'],
                    ret['iou_scores'], ret['confidences'], ret['label_preds']))
                ret_list[i]['box3d_lidar'] = box3d_lidar
                ret_list[i]['scores'] = scores
                ret_list[i]['iou_scores'] = iou_scores
                ret_list[i]['confidences'] = confidences
                ret_list[i]['label_preds'] = label_preds

                # ret_list[i]['box3d_lidar'] = ret['box3d_lidar']
                # ret_list[i]['scores'] = ret['scores']
                # ret_list[i]['iou_scores'] = ret['iou_scores']
                # ret_list[i]['confidences'] = ret['confidences']
                # ret_list[i]['label_preds'] = ret['label_preds']

        # ====================
        # add 8A
        if 'accessory_main' in ori_class_name_map:
            for i, ret in enumerate(ret_list):
                if ret['box3d_lidar'].shape[0] == 0:
                    ret_list[i]['combo_uid'] = paddle.zeros(
                        [0], dtype='int64')
                    continue

                bigmot_inds = ret['label_preds'] == ori_class_name_map['bigMot']
                head_inds = ret['label_preds'] == ori_class_name_map['accessory_main']

                bigmot_boxes = ret['box3d_lidar'][bigmot_inds]
                head_boxes = ret['box3d_lidar'][head_inds]
                head_scores = ret['scores'][head_inds]

                if bigmot_boxes.shape[0] == 0:
                    ret_list[i]['combo_uid'] = paddle.zeros(
                        [0], dtype='int64')
                    continue
                if head_boxes.shape[0] == 0:
                    ret_list[i]['combo_uid'] = paddle.zeros(
                        [0], dtype='int64')
                    continue

                head_scores_matrix = head_scores.tile((bigmot_boxes.shape[0], 1))
                #head_scores.repeat(
                #    bigmot_boxes.size(0), 1)

                bev_iou = iou3d_utils.boxes_iou_bev(bigmot_boxes, head_boxes)

                neg_head_inds = bev_iou <= 0.2
                bev_iou[neg_head_inds] = -1

                bev_iom = iou3d_utils.boxes_iom_bev(bigmot_boxes, head_boxes)

                neg_head_inds = bev_iom <= 0.7
                bev_iom[neg_head_inds] = -1

                head_scores_matrix[neg_head_inds] = -1

                neg_head_inds = head_scores_matrix <= 0.2
                head_scores_matrix[neg_head_inds] = -1

                ret_list[i]['combo_uid'] = head_scores_matrix.argmax(axis = 1)# [1] # TODO yipin check
                ret_list[i]['combo_uid'][head_scores_matrix.max(axis = 1) < 0] = -1
        
        # ====================
            
        bboxes = []
        scores = []
        labels = []
        combo_uids = []

        for i, ret in enumerate(ret_list):
            bboxes.append(ret_list[i]['box3d_lidar'])
            scores.append(ret_list[i]['scores'])
            labels.append(ret_list[i]['label_preds'])
            if 'accessory_main' in ori_class_name_map:    # add 8A
                combo_uids.append(ret_list[i]['combo_uid'])
        

        if kwargs.get('is_training'):
            return [bboxes, scores, labels]

        boxlist = []
        scorelist = []
        labellist = []
        for bbox, score, label in zip(bboxes, scores, labels):
            assert bbox.shape[0] == score.shape[0] == label.shape[0]
            if bbox.shape[0] != 0:
                boxlist.append(bbox)
                scorelist.append(score)
                labellist.append(label)
        
        assert len(boxlist) ==  len(scorelist) == len(labellist)
        if len(boxlist) != 0:
            bboxes = paddle.concat(boxlist, axis = 0)
            scores = paddle.concat(scorelist, axis = 0)
            labels = paddle.concat(labellist, axis = 0)

        else:
            bboxes = paddle.empty((0, 7))
            scores = paddle.empty((0,))
            labels = paddle.empty((0,))
    
        # add 8A
        if 'accessory_main' in ori_class_name_map:
            combo_uids = cat(combo_uids, axis=0) # paddle.concat(combo_uids, axis=0)
        else:
            combo_uids = paddle.zeros(
                [0], dtype=scores.dtype)
        # bboxes = BBoxes3D(bboxes)
        return [[bboxes, scores, labels, combo_uids]]

    def get_task_detections(self, task_id, num_class_with_bg, test_cfg,
        batch_cls_preds, batch_reg_preds, batch_dir_preds=None,
        batch_iou_preds=None, batch_anchors_mask=None):
        predictions_dicts = []
        post_center_range = test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = paddle.to_tensor(data=post_center_range,
                dtype=batch_reg_preds.dtype)
        for box_preds, cls_preds, dir_preds, iou_preds, a_mask in zip(
            batch_reg_preds, batch_cls_preds, batch_dir_preds,
            batch_iou_preds, batch_anchors_mask):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.astype(dtype='float32')
            cls_preds = cls_preds.astype(dtype='float32')
            if self.use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = paddle.argmax(dir_preds, axis=-1) 
            if iou_preds is not None:
                iou_preds = iou_preds[a_mask].astype(dtype='float32')
                iou_scores = paddle.nn.functional.sigmoid(x=iou_preds).squeeze(
                    axis=-1)
            if self.encode_background_as_zeros:
                assert self.use_sigmoid_score is True
                total_scores = paddle.nn.functional.sigmoid(x=cls_preds)
            elif self.use_sigmoid_score:
                total_scores = paddle.nn.functional.sigmoid(x=cls_preds)[(
                    ...), 1:]
            else:
                total_scores = paddle.nn.functional.softmax(x=cls_preds,
                    axis=-1)[(...), 1:]
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(axis=-1)
                top_labels = paddle.zeros((total_scores.shape[0], ), dtype='int64')
            else:
                top_scores = paddle.max(total_scores, axis=-1)
                top_labels = paddle.argmax(total_scores, axis=-1)
            if test_cfg['score_threshold'] > 0.0:
                thresh = paddle.to_tensor(data=[test_cfg['score_threshold']]).astype(dtype=total_scores.dtype)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(mask=top_scores_keep)
            dtype = batch_reg_preds.dtype
            if top_scores.shape[0] != 0:
                if test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    if self.use_direction_classifier:
                        dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                if self.use_direction_classifier:
                    opp_labels = (box_preds[..., -1] - self.direction_offset > 0) ^ dir_labels.astype(dtype='bool')
                    box_preds[..., -1] += paddle.where(condition=opp_labels, 
                                                        x = paddle.to_tensor(data=np.pi).astype(dtype=box_preds.dtype), 
                                                        y=paddle.to_tensor(data=0.0).astype(dtype=box_preds.dtype))

                if iou_preds is not None:
                    iou_scores = iou_scores[top_scores_keep]
                if post_center_range is not None:
                    mask = (box_preds[:, :3] >= post_center_range[:3]).all(axis=1)
                    mask &= (box_preds[:, :3] <= post_center_range[3:]).all(axis=1)
                    predictions_dict = {'box3d_lidar': box_preds[mask],
                                        'scores': top_scores[mask], 
                                        'label_preds': top_labels[mask], 
                                        'iou_scores': iou_scores[mask] if iou_preds is not None else None}
                else:
                    predictions_dict = {'box3d_lidar': box_preds, 
                                        'scores': top_scores, 
                                        'label_preds': top_labels, 
                                        'iou_scores': iou_scores if iou_preds is not None else None}
            else:
                predictions_dict = {'box3d_lidar': paddle.zeros(shape=[0, self.box_n_dim], dtype=dtype), 
                                    'scores': paddle.zeros(shape=[0], dtype=dtype), 
                                    'label_preds': paddle.zeros(shape=[0], dtype=top_labels.dtype), 
                                    'iou_scores': paddle.zeros(shape=[0], dtype=dtype) if iou_preds is not None else None}
            predictions_dicts.append(predictions_dict)
        return predictions_dicts

    def nms_for_groups(self, test_cfg, nms_groups, box3d_lidar, scores,
        iou_scores, confidences, label_preds):
        # print("pr1: ", box3d_lidar)
        group_nms_pre_max_size = test_cfg['nms'].get('group_nms_pre_max_size', [])
        group_nms_post_max_size = test_cfg['nms'].get('group_nms_post_max_size',
            [])
        group_nms_iou_threshold = test_cfg['nms'].get('group_nms_iou_threshold',
            [])
        add_iou_edge = test_cfg['nms'].get('add_iou_edge', 1.0)
        if len(group_nms_pre_max_size) == 0:
            group_nms_pre_max_size = [test_cfg['nms']['nms_pre_max_size']] * len(
                nms_groups)
        if len(group_nms_post_max_size) == 0:
            group_nms_post_max_size = [test_cfg['nms']['nms_post_max_size']] * len(
                nms_groups)
        if len(group_nms_iou_threshold) == 0:
            group_nms_iou_threshold = [test_cfg['nms']['nms_iou_threshold']] * len(
                nms_groups)
        assert len(group_nms_pre_max_size) == len(nms_groups)
        assert len(group_nms_post_max_size) == len(nms_groups)
        assert len(group_nms_iou_threshold) == len(nms_groups)
        nms_func = nms
        # print('box3d_lidar', box3d_lidar.shape)
        # boxes_for_nms = box3d_lidar.index_select(paddle.to_tensor([0, 1, 3, 4, -1]), axis = 1) #box3d_lidar[:, ([0, 1, 3, 4, -1])]
        # print('box3d_lidar', box3d_lidar.shape)
        # boxes_for_nms = box3d_lidar.index_select(paddle.to_tensor(np.array([0, 1, 3, 4, 6]).astype('int32')), axis = 1) #box3d_lidar[:, ([0, 1, 3, 4, -1])]
        # print("@@shape: ", box3d_lidar.shape)
        # print("@@box3d_lidar: ", box3d_lidar)
        # boxes_for_nms = box3d_lidar.index_select(paddle.to_tensor([0, 1, 3, 4, 6]), axis = 1)
        boxes_for_nms = paddle.gather(box3d_lidar, paddle.to_tensor([0, 1, 3, 4, 6]), axis = 1)
        
        if not test_cfg['nms']['use_rotate_nms']:
            box_preds_corners = center_to_corner_box2d(boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, (4)])
            boxes_for_nms = corner_to_standup_nd(box_preds_corners)
        for group_id, nms_group in enumerate(nms_groups):
            selecteds = label_preds >= 0
            if len(nms_group) == 0:
                continue
            mask = label_preds == nms_group[0]
            for label_id in nms_group:
                mask |= label_preds == label_id
            indices = paddle.where(mask)[0].squeeze(axis = -1)
            if indices.shape[0] != 0:
                group_boxes_for_nms = boxes_for_nms.index_select(indices) 
                group_scores = scores.index_select(indices) 
                group_iou_scores = iou_scores.index_select(indices) if iou_scores is not None else None 
                if group_iou_scores is not None:
                    group_scores = group_scores * group_iou_scores
                selected = nms_func(group_boxes_for_nms, group_scores,    # TODO
                                    pre_max_size=group_nms_pre_max_size[group_id],
                                    post_max_size=group_nms_post_max_size[group_id],
                                    iou_threshold=group_nms_iou_threshold[group_id],
                                    add_iou_edge=add_iou_edge)
                selected_indices = indices[selected]
                selecteds = paddle.scatter(selecteds.cast('int32'), 
                                            indices, 
                                            paddle.zeros(indices.shape, dtype='int32'), 
                                            overwrite=True)
                selecteds = paddle.scatter(selecteds, 
                                            selected_indices, 
                                            paddle.ones(selected_indices.shape, dtype='int32'), 
                                            overwrite=True).cast('bool')
                boxes_for_nms = boxes_for_nms[selecteds]
                box3d_lidar = box3d_lidar[selecteds]
                label_preds = label_preds[selecteds]
                scores = scores[selecteds]
                confidences = confidences[selecteds]
                iou_scores = iou_scores[selecteds
                    ] if iou_scores is not None else None
        return box3d_lidar, scores, confidences, iou_scores, label_preds

    def nms_overlap_for_groups(self, test_cfg, nms_overlap_groups,
        box3d_lidar, scores, iou_scores, confidences, label_preds):
        group_nms_overlap_pre_max_size = test_cfg['nms'].get(
            'group_nms_overlap_pre_max_size', [])
        group_nms_overlap_post_max_size = test_cfg['nms'].get(
            'group_nms_overlap_post_max_size', [])
        group_nms_overlap_iou_threshold = test_cfg['nms'].get(
            'group_nms_overlap_iou_threshold', [])
        if len(group_nms_overlap_pre_max_size) == 0:
            group_nms_overlap_pre_max_size = [test_cfg['nms']['nms_pre_max_size']] * len(nms_overlap_groups)
        if len(group_nms_overlap_post_max_size) == 0:
            group_nms_overlap_post_max_size = [test_cfg['nms']['nms_post_max_size']] * len(nms_overlap_groups)
        if len(group_nms_overlap_iou_threshold) == 0:
            group_nms_overlap_iou_threshold = [test_cfg['nms']['nms_iou_threshold']] * len(nms_overlap_groups)
        assert len(group_nms_overlap_pre_max_size) == len(nms_overlap_groups)
        assert len(group_nms_overlap_post_max_size) == len(nms_overlap_groups)
        assert len(group_nms_overlap_iou_threshold) == len(nms_overlap_groups)
        if not ('use_rotate_nms_overlap' in test_cfg['nms']) or not test_cfg['nms']['use_rotate_nms_overlap']:
            nms_overlap_func = nms_overlap
        else:
            nms_overlap_func = rotate_nms_overlap
            # raise NotImplementedError
            # print("rotate_nms_overlap uses numba implementation!!")
        # print("@@t1")
        # print(box3d_lidar.shape)
        # boxes_for_nms = box3d_lidar.index_select(paddle.to_tensor([0, 1, 3, 4, -1]), axis = 1)
        boxes_for_nms = box3d_lidar.index_select(paddle.to_tensor([0, 1, 3, 4, 6]), axis = 1)
        if not ('use_rotate_nms_overlap' in test_cfg['nms']) or not test_cfg['nms']['use_rotate_nms_overlap']:
            box_preds_corners = center_to_corner_box2d(boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4])
            boxes_for_nms = corner_to_standup_nd(box_preds_corners)
        
        # print("@@t2")
        for group_id, nms_group in enumerate(nms_overlap_groups):
            selecteds = label_preds >= 0
            if len(nms_group) == 0:
                continue
            mask = label_preds == nms_group[0]
            for label_id in nms_group:
                mask |= label_preds == label_id
            # indices = paddle.nonzero(x=mask, as_tuple=True)[0]
            indices = paddle.where(mask)[0].squeeze(axis = -1) #paddle.nonzero(x=mask, as_tuple=True)[0]
            # print("@@t3")
            if indices.shape[0] != 0:
                group_boxes_for_nms = boxes_for_nms.index_select(indices) #boxes_for_nms[indices]
                group_scores = scores.index_select(indices) #scores[indices]
                group_iou_scores = iou_scores.index_select(indices) if iou_scores is not None else None 
                if group_iou_scores is not None:
                    group_scores = group_scores * group_iou_scores
                selected = nms_overlap_func(group_boxes_for_nms,
                                            group_scores, 
                                            pre_max_size=group_nms_overlap_pre_max_size[group_id], 
                                            post_max_size=group_nms_overlap_post_max_size[group_id],
                                            overlap_threshold=group_nms_overlap_iou_threshold[group_id])
                selected_indices = indices[selected] # TODO1023
                selecteds = paddle.scatter(selecteds.cast('int32'), 
                                            indices, 
                                            paddle.zeros(indices.shape, dtype='int32'), 
                                            overwrite=True)
                selecteds = paddle.scatter(selecteds, 
                                            selected_indices, 
                                            paddle.ones(selected_indices.shape, dtype='int32'), 
                                            overwrite=True).cast('bool')
                boxes_for_nms = boxes_for_nms[selecteds]
                box3d_lidar = box3d_lidar[selecteds]
                label_preds = label_preds[selecteds]
                scores = scores[selecteds]
                confidences = confidences[selecteds]
                iou_scores = iou_scores[selecteds] if iou_scores is not None else None
        # print("@@t4")
        return box3d_lidar, scores, confidences, iou_scores, label_preds
