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
import pickle
import logging
from enum import Enum
from typing import List

from collections import OrderedDict

import numpy as np
from functools import partial
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle3d.models.heads.dense_heads.target_assigner import AlignedAnchor3DRangeGenerator
from paddle3d.apis import manager
from paddle3d.utils import DeltaXYZWLHRBBoxCoderIDG, get_bev_corners_paddle
from paddle3d.models.losses import WeightedSigmoidLoss, WeightedSmoothL1LossIDG, WeightedSoftmaxClassificationLossIDG
from paddle3d.utils_idg.build_layer import build_norm_layer
from paddle3d.utils_idg.target_assigner_torch import AssignTargetTorch, build_torch_similarity_metric
from paddle3d.utils_idg.target_ops import calculate_anchor_masks_paddle
from paddle3d.geometries import BBoxes3D
from paddle3d.models.layers.layer_libs import rotate_nms_pcdet
from paddle3d.utils_idg.ops.nms_gpu import nms, nms_overlap, rotate_nms_overlap
from paddle3d.models.layers import param_init, reset_parameters, constant_init, kaiming_normal_init, normal_init
from paddle3d.utils_idg.box_utils import bbox_overlaps_nearest_3d
from paddle3d.utils_idg.ops import iou3d_utils
from paddle3d.models.heads.roi_heads.confidence_map_head import cat

from paddle3d.utils_idg.sub_region_utils import get_class2id
from paddle3d.utils_idg.preprocess import noise_gt_bboxesv2_


def bias_init_with_prob(prior_prob: float) -> float:
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

def normal_init_mgl(layers,
                mean: float = 0,
                std: float = 1,
                bias: float = 0) -> None:
    if hasattr(layers, 'weight') and layers.weight is not None:
        normal_init(layers.weight, mean = mean, std = std)
    if hasattr(layers, 'bias') and layers.bias is not None:
        constant_init(layers.bias, value = bias)


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

def build_anchor_generator(cfg): 
    # TODO support other anchor generators
    assert cfg['type_name'] 
    cfg.pop('type_name')
    return AlignedAnchor3DRangeGenerator(**cfg)
    
def build_bbox_coder(bbox_coder):
    assert bbox_coder['type_name'] 
    bbox_coder.pop("type_name")
    return DeltaXYZWLHRBBoxCoderIDG(**bbox_coder)

def build_loss_cls(loss_cls):
    assert loss_cls['type_name'] 
    loss_cls.pop("type_name")
    return WeightedSigmoidLoss(**loss_cls)


def build_loss_bbox(loss_bbox):
    assert loss_bbox['type_name'] 
    loss_bbox.pop("type_name")
    return WeightedSmoothL1LossIDG(**loss_bbox)


def build_loss_aux(loss_aux):
    assert loss_aux['type_name'] 
    loss_aux.pop("type_name")
    return WeightedSoftmaxClassificationLossIDG(**loss_aux)

def cal_sub_feature_map_range(sub_region_range, full_region_range, map_size=576):
    """
    @param sub_region_range: 
    @param total_scope:
    @param map_size: feature map size

    return [grid_start, grid_end]
    """
    # x_range = [-40, 40], y_range = [-40, 40]
    total_scope = full_region_range[1] - full_region_range[0]
    scope = (sub_region_range[1] - sub_region_range[0]) / total_scope * map_size
    start = int((total_scope / 2 + sub_region_range[0]) / total_scope * map_size)
    end = int(start + scope)
    return start, end



def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2 ** ndim, ndim])
    return corners


def rotation_2d(points, angles):
    """Rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (np.ndarray): Points to be rotated with shape \
            (N, point_size, 2).
        angles (np.ndarray): Rotation angle with shape (N).

    Returns:
        np.ndarray: Same shape as points.
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """Convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 2).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 2).
        angles (np.ndarray): Rotation_y in kitti label file with shape (N).

    Returns:
        np.ndarray: Corners with the shape of (N, 4, 2).
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def corner_to_standup_nd(boxes_corner):
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(paddle.min(boxes_corner[:, :, i], dim=1))
    for i in range(ndim):
        standup_boxes.append(paddle.max(boxes_corner[:, :, i], dim=1))
    return paddle.stack(standup_boxes, dim=1)


def one_hot_f(tensor, depth, dim=-1, on_value=1.0, dtype='float32'):
    tensor_onehot = paddle.zeros(shape=[*list(tensor.shape), depth], dtype=dtype)
    tensor_onehot = tensor_onehot.put_along_axis(indices = tensor.unsqueeze(axis=dim).cast('int64'), values = on_value, axis = dim)
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
        cls_pos_loss = (labels > 0).astype(dtype=cls_loss.dtype) * cls_loss.reshape((batch_size, -1))
        cls_neg_loss = (labels == 0).astype(dtype=cls_loss.dtype) * cls_loss.reshape((batch_size, -1))
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[(...), 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0.0):
    batch_size = reg_targets.shape[0]
    anchors = anchors.reshape((batch_size, -1, anchors.shape[-1]))
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = (rot_gt - dir_offset > 0).astype(dtype='int64')
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
    Note:
        's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
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


def create_loss(loc_loss_ftor, cls_loss_ftor, box_preds, cls_preds,
    cls_targets, cls_weights, reg_targets, reg_weights, num_class,
    encode_background_as_zeros=True, encode_rad_error_by_sin=True, bev_only
    =False, box_code_size=7):
    batch_size = int(box_preds.shape[0])
    if bev_only:
        box_preds = box_preds.reshape((batch_size, -1, box_code_size))
    else:
        box_preds = box_preds.reshape((batch_size, -1, box_code_size))
    if encode_background_as_zeros:
        cls_preds = cls_preds.reshape((batch_size, -1, num_class))
    else:
        cls_preds = cls_preds.reshape((batch_size, -1, num_class + 1))
    cls_targets = cls_targets.squeeze(axis=-1)
    one_hot_targets = one_hot_f(cls_targets, depth=num_class + 1, dtype=
        box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(box_preds, reg_targets, weights=reg_weights)
    cls_losses = cls_loss_ftor(cls_preds, one_hot_targets, weights=cls_weights)
    return loc_losses, cls_losses


class LossNormType(Enum):
    NormByNumPositives = 'norm_by_num_positives'
    NormByNumExamples = 'norm_by_num_examples'
    NormByNumPosNeg = 'norm_by_num_pos_neg'
    DontNorm = 'dont_norm'


@manager.HEADS.add_component
class Head(paddle.nn.Layer):

    def __init__(self, 
                num_input, 
                num_pred, 
                num_cls, 
                use_dir=False, 
                num_dir=0, 
                use_iou=False,
                num_iou=0,
                use_bctp=False,
                num_bctp_pred=0,
                header=True, 
                name='', 
                focal_loss_init=False,
                **kwargs):

        super(Head, self).__init__(**kwargs)
        self.use_dir = use_dir
        self.use_iou = use_iou
        self.use_bctp = use_bctp

        self.conv_reg = paddle.nn.Conv2D(in_channels=num_input, out_channels=num_pred, kernel_size=1)
        self.conv_cls = paddle.nn.Conv2D(in_channels=num_input, out_channels=num_cls, kernel_size=1)
        if self.use_dir:
            self.conv_dir = paddle.nn.Conv2D(in_channels=num_input,
                out_channels=num_dir, kernel_size=1)
        if self.use_iou:
            self.conv_iou = paddle.nn.Conv2D(in_channels=num_input,
                out_channels=num_iou, kernel_size=1)
        
        if self.use_bctp:
            self.conv_bctp = nn.Conv2D(num_input, num_bctp_pred, 1)

    def forward(self, x):
        bbox_pred = self.conv_reg(x)
        cls_score = self.conv_cls(x)
        if self.use_dir:
            dir_preds = self.conv_dir(x)
        else:
            dir_preds = None
        if self.use_iou:
            iou_preds = self.conv_iou(x)
        else:
            iou_preds = None
        
        if self.use_bctp:
            bctp_preds = self.conv_bctp(x)
        else:
            bctp_preds = None

        return cls_score, bbox_pred, dir_preds, iou_preds, bctp_preds


@manager.HEADS.add_component
class MultiGroupHead(nn.Layer):

    def __init__(self, 
                mode='3d', 
                in_channels=[128], 
                norm_cfg=None, 
                tasks=[], 
                weights=[], 
                num_classes=[1], 
                num_anchor_per_locs=[], 
                ds_head_filters=[], 
                ds_head_strides=[], 
                verybigMot_index = [],
                use_bigMot_subhead=False,
                head_layer_index=[],
                bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
                anchor_cfg=None, 
                with_cls=True, 
                with_reg=True, 
                reg_class_agnostic = False, 
                encode_background_as_zeros=True,
                anchor_bctps=None, 
                pred_bctps=None, 
                cal_anchor_mask=False,
                downsample=1,
                grid_size=[600, 600, 1], 
                voxel_size=[0.2, 0.2, 10],
                pc_range=[-60, -60, -5, 60, 60, 5], 
                assign_cfg=None, 
                loss_norm=dict(type_name= 'NormByNumPositives', pos_class_weight=1.0, neg_class_weight=1.0),
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), 
                use_sigmoid_score=True, 
                loss_bbox=dict(type= 'SmoothL1Loss', beta=1.0, loss_weight=1.0),
                loss_anchor_bctps = None, 
                loss_pred_bctps = None, 
                encode_rad_error_by_sin =True, 
                loss_aux=None, 
                loss_iou=None,
                loss_pull_push=None, 
                direction_offset=0.0, 
                name='rpn',
                logger=None, 
                use_sub_region_head= False, 
                sub_region_attr=None, 
                sub_region_head_index=None, 
                train_cfg =None, 
                test_cfg=None,                		
                onnx_export=False,
                bg_iof_threshold=0.2,
                fg_iof_threshold=0.2,
                bg_cls_loss_weight=1.0):
        super(MultiGroupHead, self).__init__()
        assert with_cls or with_reg
        self.tasks = tasks
        self.num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.name2class = {x[0]:i for i,x in enumerate(self.class_names)}  # add 8a
            
        self.tasks_weight = [float(t.get("weights", 1.0)) for t in tasks]
        print(f"use tasks weight {self.tasks_weight}")
        anchor_generators = anchor_cfg['anchor_generators']
        flag = 0
        self.anchor_generators_by_task = []
        for i, task in enumerate(anchor_cfg["tasks"]):
            anchor_generators_per_task = []
            for j in range(task['num_class']):
                anchor_generators_per_task.append(build_anchor_generator(anchor_generators[flag + j])) 
            self.anchor_generators_by_task.append(anchor_generators_per_task)
            flag += task['num_class']
        if len(num_anchor_per_locs) == 0:
            self.num_anchor_per_locs = [(2 * n) for n in self.num_classes]
        else:
            assert len(num_anchor_per_locs) == len(self.num_classes)
            self.num_anchor_per_locs = num_anchor_per_locs
        self.box_coder = build_bbox_coder(bbox_coder)
        box_code_sizes = [self.box_coder.code_size] * len(self.num_classes)
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.in_channels = in_channels
        self.reg_class_agnostic = reg_class_agnostic
        self.encode_rad_error_by_sin = encode_rad_error_by_sin
        self.encode_background_as_zeros = encode_background_as_zeros
        self.use_sigmoid_score = use_sigmoid_score
        self.box_n_dim = self.box_coder.code_size
        self.cal_anchor_mask = cal_anchor_mask
        self.downsample = downsample
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.pc_range = pc_range

        self.onnx_export = onnx_export
        self.anchor_bctps = anchor_bctps
        self.pred_bctps = pred_bctps
        self.bg_iof_threshold = bg_iof_threshold
        self.fg_iof_threshold = fg_iof_threshold
        self.bg_cls_loss_weight = bg_cls_loss_weight

        self.loss_anchor_bctps = build_loss_bbox(loss_anchor_bctps) if loss_anchor_bctps else None   # build WeightedSmoothL1LossIDG loss
        self.loss_pred_bctps = build_loss_bbox(loss_pred_bctps) if loss_pred_bctps else None # build WeightedSmoothL1LossIDG loss

        self.assign_cfg = assign_cfg
        if assign_cfg:
            self.assign_target = AssignTargetTorch(cfg=assign_cfg)
        self.use_sub_region_head = use_sub_region_head
        if self.use_sub_region_head:
            assert sub_region_attr is not None, 'ValueError: sub_region_attr should not be none when use sub_region head!'
            self.sub_region_postfix = sub_region_attr.get('sub_region_postfix',
                '_sub')

            self.sub_region_class = sub_region_attr.get('sub_region_class', None)
            self.sub_region_range = sub_region_attr.get('sub_region_range',
                [[0, 60], [-30, 30]])
            self.full_region_range = sub_region_attr.get('full_region_range',
                [[-120, 120], [-120, 120]])
            self.sub_region_head_index = sub_region_attr.get(
                'sub_region_head_index', [1])
            self.sub_region_grid_hrange = None
            self.sub_region_grid_vrange = None

            # add 8a
            self.class2id = get_class2id(self.tasks, self.sub_region_postfix)
        else:
            self.class2id = get_class2id(self.tasks)
        
        
        self.use_bigMot_subhead = use_bigMot_subhead
        if self.use_bigMot_subhead:
            self.verybigMot_index = verybigMot_index.index(1)
        
        self.loss_cls = build_loss_cls(loss_cls)
        self.loss_reg = build_loss_bbox(loss_bbox)

        if loss_aux is not None:
            if 'task_loss_aux_weight' in loss_aux:
                self.task_loss_aux_weight = loss_aux.pop('task_loss_aux_weight')
            else:
                self.task_loss_aux_weight = [1] * len(tasks)
            self.loss_aux = build_loss_aux(loss_aux)
        self.use_direction_classifier = loss_aux is not None
        if loss_aux:
            self.direction_offset = direction_offset
        self.use_iou_score = loss_iou is not None
        self.use_loss_pull_push = loss_pull_push is not None
        self.loss_norm = loss_norm
        if not logger:
            logger = logging.getLogger('MultiGroupHead')
        self.logger = logger
        self.dcn = None
        self.zero_init_residual = False
        self.bev_only = True if mode == 'bev' else False
        num_clss = []
        num_preds = []
        num_dirs = []
        num_ious = []
        num_bctp_preds = []
        for num_c, num_a, box_cs in zip(self.num_classes, self.
            num_anchor_per_locs, box_code_sizes):
            if self.encode_background_as_zeros:
                num_cls = num_a * num_c
            else:
                num_cls = num_a * (num_c + 1)
            num_clss.append(num_cls)
            if self.bev_only:
                num_pred = num_a * box_cs
            else:
                num_pred = num_a * box_cs
            num_preds.append(num_pred)
            if self.use_direction_classifier:
                num_dir = num_a * 2
                num_dirs.append(num_dir)
            else:
                num_dir = None
            if self.use_iou_score:
                num_iou = num_a
                num_ious.append(num_iou)
        		
            if self.anchor_bctps is not None and self.anchor_bctps['mode']:
                num_bctp_pred = num_a * 8
                num_bctp_preds.append(num_bctp_pred)

        logger.info(
            f'num_classes: {self.num_classes}, num_preds: {num_preds}, num_dirs: {num_dirs}'
            )
        assert len(ds_head_filters) == len(ds_head_strides)
        if len(head_layer_index) == 0:
            head_layer_index = [0] * len(tasks)
        assert len(head_layer_index) == len(tasks)
        self.ds_head_filters = ds_head_filters
        self.ds_head_strides = ds_head_strides
        self.head_layer_index = head_layer_index
        if self.use_sub_region_head:
            assert len(self.ds_head_filters) == len(self.ds_head_strides)
        self._norm_cfg = norm_cfg
        self.ds_heads = paddle.nn.LayerList()
        num_input = self.in_channels
        for i, _ in enumerate(self.ds_head_filters):
            if self.use_bigMot_subhead and i == self.verybigMot_index:
                self.ds_heads.append(self._make_downsample_head(self.ds_head_filters[0],
                                                                self.ds_head_filters[i],
                                                                self.ds_head_strides[i]))
            else:
                self.ds_heads.append(self._make_downsample_head(num_input,
                                                                self.ds_head_filters[i],
                                                                self.ds_head_strides[i]))

                   
                            
        out_channels = [in_channels]
        out_channels.extend(self.ds_head_filters)
        self.tasks = paddle.nn.LayerList()
        for task_id, (num_pred, num_cls) in enumerate(zip(num_preds, num_clss)):
            cur_out_channel = out_channels[self.head_layer_index[task_id]]
            if self.onnx_export == False:
                self.tasks.append(
                    Head(
                        cur_out_channel,
                        num_pred,
                        num_cls,
                        use_dir=self.use_direction_classifier,
                        num_dir=num_dirs[task_id] if self.use_direction_classifier else None,
                        header=False,
                        use_iou=self.use_iou_score,
                        num_iou=num_ious[task_id] if self.use_iou_score else None,

                        use_bctp=False if self.anchor_bctps is None else self.anchor_bctps['mode'],
                        num_bctp_pred = num_bctp_preds[task_id] if self.anchor_bctps is not None and self.anchor_bctps['mode'] else None,
                    )
                )
            else:
                raise NotImplementedError('ONNX export is not implemented yet.')

        self.init_weights()
        logger.info('Finish MultiGroupHead Initialization')

    def _make_downsample_head(self, inchannel, outchannel, stride=2):
        if stride < 1:
            stride = round(1 / stride)
            ds_head = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=inchannel, 
                out_channels=outchannel, kernel_size=1, stride=1, bias_attr=False), 
                build_norm_layer(self._norm_cfg, outchannel)[1], 
                paddle.nn.ReLU(), 
                paddle.nn.Conv2DTranspose(in_channels=outchannel, out_channels=outchannel,
                kernel_size=2, stride=stride, bias_attr=False),
                build_norm_layer(self._norm_cfg, outchannel)[1], 
                paddle.nn.ReLU())
        else:
            ds_head = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=inchannel, 
                out_channels=outchannel, kernel_size=3, stride=stride, padding=1, bias_attr=False), 
                build_norm_layer(self._norm_cfg, outchannel)[1], paddle.nn.ReLU())
        return ds_head

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            state_dict = paddle.load(pretrained)
            self.set_dict(state_dict)
        elif pretrained is None:
            bias_cls = bias_init_with_prob(0.01)
            for i in range(len(self.tasks)):
                normal_init_mgl(self.tasks[i].conv_reg, std=0.01)
                normal_init_mgl(self.tasks[i].conv_dir, std=0.01)
                normal_init_mgl(self.tasks[i].conv_cls, std=0.01, bias=bias_cls)
                if self.anchor_bctps is not None and self.anchor_bctps['mode']:
                    normal_init_mgl(self.tasks[i].conv_bctp, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_single(self, task):
        return task['net'](task['featmap'])

    def forward(self, x):
        head_layers = x
        for idx, ds_head in enumerate(self.ds_heads):
            if self.use_sub_region_head and idx in self.sub_region_head_index:
                if (self.sub_region_grid_hrange is None or self.
                    sub_region_grid_vrange is None):
                    self.sub_region_grid_hrange = cal_sub_feature_map_range(self.sub_region_range[0], 
                        self.full_region_range[0], map_size=head_layers[0].shape[3])
                    self.sub_region_grid_vrange = cal_sub_feature_map_range(self.sub_region_range[1], 
                        self.full_region_range[1], map_size=head_layers[0].shape[2])
                sub_region_feature = head_layers[0][:, :, 
                    self.sub_region_grid_vrange[0]:self.sub_region_grid_vrange[1], 
                    self.sub_region_grid_hrange[0]:self.sub_region_grid_hrange[1]]
                head_layers.append(ds_head(sub_region_feature))
            		
            elif self.use_bigMot_subhead and idx == self.verybigMot_index:
                head_layers.append(ds_head(head_layers[1]))

            else:
                head_layers.append(ds_head(head_layers[0]))
        task_list = []
        for task_id, task in enumerate(self.tasks):
            task_list.append(dict(net=task, featmap=head_layers[self.
                head_layer_index[task_id]]))
        return multi_apply(self.forward_single, task_list), head_layers

    def get_proposals(self, batch_anchors, batch_box_preds, batch_cls_preds,
        pos_inds, pos_gt_inds, num_class_with_bg, conf_thr=0.05):
        batch_size = batch_anchors.shape[0]
        batch_anchors = batch_anchors.reshape((batch_size, -1, self.box_n_dim)) 
        batch_cls_preds = batch_cls_preds.reshape((batch_size, -1, num_class_with_bg)) 
        batch_reg_preds = self.box_coder.decode(batch_anchors,
            batch_box_preds[:, :, :self.box_coder.code_size])
        proposals = []
        for batch_id in range(batch_size):
            box_preds = batch_reg_preds[batch_id][pos_inds[batch_id]]
            cls_preds = batch_cls_preds[batch_id][pos_inds[batch_id]]
            batch_pos_gt_inds = pos_gt_inds[batch_id]
            if self.encode_background_as_zeros:
                assert self.use_sigmoid_score is True
                total_scores = paddle.nn.functional.sigmoid(x=cls_preds)
            elif self.use_sigmoid_score:
                total_scores = paddle.nn.functional.sigmoid(x=cls_preds)[(...), 1:]
            else:
                total_scores = paddle.nn.functional.softmax(x=cls_preds,
                    axis=-1)[(...), 1:]
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(axis=-1)
            else:
                paddle.max(total_scores, axis=-1)
            if conf_thr > 0.0:
                top_scores_keep = top_scores >= conf_thr
                top_scores = top_scores.masked_select(mask=top_scores_keep)
                box_preds = box_preds[top_scores_keep]
                batch_pos_gt_inds = batch_pos_gt_inds[top_scores_keep]
            assert box_preds.shape[0] == top_scores.shape[0] \
                and box_preds.shape[0] == batch_pos_gt_inds.shape[0]
            proposals.append([box_preds, top_scores, batch_pos_gt_inds])
        return proposals

    def get_anchors(self, featmap_sizes, input_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            input_metas (list[dict]): contain pcd and img's meta info.

        Returns:
            list[list[paddle.Tensor]]: Anchors of each image, valid flags                 
            of each image.
        """
        num_imgs = len(input_metas)
        multi_level_anchors_by_task = []
        for task_id, anchor_generators_per_task in enumerate(self.anchor_generators_by_task):
            anchors_per_task = []
            for anchor_generator in anchor_generators_per_task:
                if isinstance(featmap_sizes, List):
                    anchors_per_task.append(anchor_generator.grid_anchors([featmap_sizes[task_id]])[0])
                elif isinstance(featmap_sizes, np.ndarray):
                    anchors_per_task.append(anchor_generator.grid_anchors([featmap_sizes])[0])
                else:
                    raise NotImplementedError

            anchors_per_task = paddle.concat(x=anchors_per_task, axis=0)[None, ...].tile(repeat_times=[num_imgs, 1, 1])
            multi_level_anchors_by_task.append(anchors_per_task)
        return multi_level_anchors_by_task

    def assign_target_and_mask(self, coors, batch_anchors, gt_bboxes,
        gt_labels, box_preds, num_points_in_gts=None, gt_border_masks=None, roi_regions=None, test_mode=False):
        result = self.assign_target(coors, batch_anchors, gt_bboxes,
            gt_labels, box_preds, num_points_in_gts, gt_border_masks, roi_regions, test_mode)
        return result['labels'], result['reg_targets'], result['reg_weights'], \
                result['positive_gt_id'], result['anchors_mask'], result['bctp_targets'],\
                result['border_mask_weights'], result['regions_mask']

    def prepare_loss_weights(self, 
                            labels,
                            loss_norm=dict(type='NormByNumPositives', pos_cls_weight=1.0, neg_cls_weight=1.0), 
                            dtype='float32',
                            bg_weights=None):

        loss_norm_type = getattr(LossNormType, loss_norm['type_name'])
        pos_cls_weight = loss_norm['pos_cls_weight']
        neg_cls_weight = loss_norm['neg_cls_weight']
        cared = labels >= 0
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.cast(dtype) * neg_cls_weight
        if bg_weights is not None:
            negative_cls_weights *= bg_weights
        cls_weights = negative_cls_weights + pos_cls_weight * positives.cast(dtype)
        reg_weights = positives.cast(dtype)
        if loss_norm_type == LossNormType.NormByNumExamples:
            num_examples = cared.cast(dtype).sum(axis=1, keepdim=True)
            num_examples = paddle.clip(x=num_examples, min=1.0)
            cls_weights /= num_examples
            bbox_normalizer = positives.sum(axis=1, keepdim=True).cast(dtype)
            reg_weights /= paddle.clip(x=bbox_normalizer, min=1.0)
        elif loss_norm_type == LossNormType.NormByNumPositives:
            pos_normalizer = positives.sum(axis=1, keepdim=True).cast(dtype)
            reg_weights /= paddle.clip(x=pos_normalizer, min=1.0)
            cls_weights /= paddle.clip(x=pos_normalizer, min=1.0)
        elif loss_norm_type == LossNormType.NormByNumPosNeg:
            pos_neg = paddle.stack(x=[positives, negatives], axis=-1).cast(dtype)
            normalizer = pos_neg.sum(axis=1, keepdim=True)
            cls_normalizer = (pos_neg * normalizer).sum(axis=-1)
            cls_normalizer = paddle.clip(x=cls_normalizer, min=1.0)
            normalizer = paddle.clip(x=normalizer, min=1.0)
            reg_weights /= normalizer[:, 0:1, (0)]
            cls_weights /= cls_normalizer
        elif loss_norm_type == LossNormType.DontNorm:
            pos_normalizer = positives.sum(axis=1, keepdim=True).cast(dtype)
            reg_weights /= paddle.clip(x=pos_normalizer, min=1.0)
        else:
            raise ValueError(
                f'unknown loss norm type. available: {list(LossNormType)}')
        return cls_weights, reg_weights, cared

    def loss_single(self, 
                    bbox_preds, 
                    cls_scores,
                    bctp_preds,
                    gt_bboxes, 
                    gt_labels,
                    dir_cls_preds, 
                    iou_preds, 
                    labels, 
                    reg_targets,
                    bctp_targets,
                    border_mask_weights,
                    positive_gt_id,
                    num_classes, 
                    anchors_mask, 
                    batch_anchors,
                    task_loss_aux_weight_per_task,
                    task_weight,
                    batch_size_device,
                    task_id,
                    bg_bboxes, 
                    bg_labels, 
                    region_mask):
        if self.cal_anchor_mask:
            labels[~anchors_mask] = -1
        
        		
        bg_weights = paddle.ones_like(batch_anchors[:, :, 0])  # shape: (batch_size, n_anchors)
        if region_mask is not None:
            bg_weights *= region_mask

        cls_weights, reg_weights, cared = self.prepare_loss_weights(labels,
            loss_norm=self.loss_norm, dtype='float32', bg_weights=bg_weights)
        cls_targets = labels * cared.astype(dtype=labels.dtype)
        cls_targets = cls_targets.unsqueeze(axis=-1)
        loc_loss, cls_loss = create_loss(self.loss_reg, self.loss_cls,
            bbox_preds, cls_scores, cls_targets, cls_weights, reg_targets,
            reg_weights, num_classes, self.encode_background_as_zeros, 
            self.encode_rad_error_by_sin, bev_only=self.bev_only, 
            box_code_size=self.box_n_dim)

        		
        bg_cls_losses = []
        # process background
        cls_prob = F.sigmoid(cls_scores)
        with paddle.no_grad():
            batch_size = batch_anchors.shape[0]
            for batch_id, (anchors_, gt_boxes_, box_preds_, cls_prob_) in enumerate(
                zip(batch_anchors, gt_bboxes, 
                    bbox_preds.reshape((batch_size, -1, self.box_n_dim)),
                    cls_prob.reshape((batch_size, -1, num_classes)))):
                box_preds_decode = self.box_coder.decode(box_preds_, anchors_)
                if bg_bboxes is not None:
                    bg_boxes_ = bg_bboxes[batch_id]
                    bg_classes_ = bg_labels[batch_id]
                    valid = bg_classes_ > 0
                    bg_boxes_ = bg_boxes_[valid]
                    if len(bg_boxes_) and len(gt_boxes_):
                        bg_box_iou = bbox_overlaps_nearest_3d(box_preds_decode, bg_boxes_ , 'iof')
                        bg_box_iou[bg_box_iou.isnan()] = 0 
                        bg_box_iou = bg_box_iou.max(axis=1)

                        fg_box_iou = bbox_overlaps_nearest_3d(box_preds_decode, gt_boxes_ , 'iof')
                        fg_box_iou[bg_box_iou.isnan()] = 0 
                        fg_box_iou = fg_box_iou.max(axis=1)

                        valid = (bg_box_iou >= self.bg_iof_threshold) & (fg_box_iou <= self.fg_iof_threshold)
                        bg_cls = cls_prob_[valid]
                        if len(bg_cls) > 0:
                            bg_cls_loss = F.binary_cross_entropy(bg_cls, paddle.zeros_like(bg_cls), reduction='none').mean(axis=-1)
                            bg_cls_losses.append(bg_cls_loss)

        if len(bg_cls_losses) > 0:
            bg_cls_loss = paddle.concat(bg_cls_losses).mean() * self.bg_cls_loss_weight
        else:
            bg_cls_loss = paddle.to_tensor([0]).cast(cls_prob.dtype)

        
        loc_loss_reduced = loc_loss.sum() / batch_size_device
        loc_loss_reduced *= self.loss_reg._loss_weight
        cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
        cls_pos_loss /= self.loss_norm['pos_cls_weight']
        cls_neg_loss /= self.loss_norm['neg_cls_weight']
        cls_loss_reduced = cls_loss.sum() / batch_size_device
        cls_loss_reduced *= self.loss_cls._loss_weight

        an_bctps_loss_reduced = None
        if self.anchor_bctps and self.anchor_bctps['mode']:
            border_mask_weights_an = paddle.stack([border_mask_weights, border_mask_weights], 3).reshape((batch_size_device, -1, 8))

            reg_weights_ = reg_weights.unsqueeze(-1).tile([1, 1, 8])
            border_mask_weights_an *= reg_weights_

            bctp_preds = bctp_preds.reshape((batch_size_device, -1, 8))
            bctp_targets = bctp_targets.reshape((batch_size_device, bctp_targets.shape[1], -1))

            if self.anchor_bctps['near_bcpts']:
                batch_gts = self.box_coder.decode(batch_anchors, reg_targets)
                theta_diff = abs(batch_gts[:, :, 6] - batch_anchors[:, :, 6])
    
                la_pi_inds = theta_diff > np.pi
                theta_diff[la_pi_inds] = 2 * np.pi - theta_diff[la_pi_inds]

                la_half_pi_inds = theta_diff > (np.pi / 2)
                le_half_pi_inds = theta_diff <= (np.pi / 2)

                bctp_preds_ = paddle.zeros(bctp_preds.shape)
                bctp_preds_[la_half_pi_inds][:, 0:2] = bctp_preds[la_half_pi_inds][:, 4:6]
                bctp_preds_[la_half_pi_inds][:, 2:4] = bctp_preds[la_half_pi_inds][:, 6:8]
                bctp_preds_[la_half_pi_inds][:, 4:6] = bctp_preds[la_half_pi_inds][:, 0:2]
                bctp_preds_[la_half_pi_inds][:, 6:8] = bctp_preds[la_half_pi_inds][:, 2:4]

                bctp_preds_[le_half_pi_inds] = bctp_preds[le_half_pi_inds]
            else:
                bctp_preds_ = bctp_preds

            an_bctps_loss = self.loss_anchor_bctps(bctp_preds_, bctp_targets, weights=border_mask_weights_an)
            an_bctps_loss_reduced = an_bctps_loss.sum() / batch_size_device
            an_bctps_loss_reduced *= self.loss_anchor_bctps._loss_weight

        pred_bctps_loss_reduced = None
        if self.pred_bctps and self.pred_bctps['mode']:
            border_mask_weights_pr = paddle.stack([border_mask_weights, border_mask_weights], 3).reshape((batch_size_device, -1, 8))
            
            reg_weights_ = reg_weights.unsqueeze(-1).tile((1, 1, 8))
            border_mask_weights_pr *= reg_weights_
            
            bbox_preds = bbox_preds.reshape((batch_size_device, -1, self.box_n_dim)) 
            
            batch_gts = self.box_coder.decode(batch_anchors, reg_targets)
            batch_preds = self.box_coder.decode(batch_anchors, bbox_preds)
            batch_gts = batch_gts.reshape((-1, self.box_n_dim)) 
            batch_preds = batch_preds.reshape((-1, self.box_n_dim)) 
            batch_bev_gts = batch_gts.index_select(paddle.to_tensor([0, 1, 3, 4, 6]), axis = 1) 
            batch_bev_preds = batch_preds.index_select(paddle.to_tensor([0, 1, 3, 4, 6]), axis = 1) 
            bev_gt_corners = get_bev_corners_paddle(batch_bev_gts)
            bev_pred_corners = get_bev_corners_paddle(batch_bev_preds)

            bev_gt_corners_cat = paddle.concat([bev_gt_corners, bev_gt_corners[:, :1, :]], 1) 
            bev_pred_corners_cat = paddle.concat([bev_pred_corners, bev_pred_corners[:, :1, :]], 1) 

            bev_gt_bctps = (bev_gt_corners_cat[:, :-1, :] + bev_gt_corners_cat[:, 1:, :])/2
            bev_pred_bctps = (bev_pred_corners_cat[:, :-1, :] + bev_pred_corners_cat[:, 1:, :])/2

            if self.pred_bctps['near_bcpts']:
                theta_diff = paddle.abs(batch_bev_gts[:, 4] - batch_bev_preds[:, 4])

                la_pi_inds = theta_diff > np.pi
                theta_diff[la_pi_inds] = 2 * np.pi - theta_diff[la_pi_inds]

                la_half_pi_inds = theta_diff > (np.pi / 2)
                le_half_pi_inds = theta_diff <= (np.pi / 2)
                bev_pred_bctps_ = paddle.zeros(bev_pred_bctps.shape)

                bev_pred_bctps_tmp = paddle.concat([bev_pred_bctps[:, 2:3, :],
                                                    bev_pred_bctps[:, 3:4, :],
                                                    bev_pred_bctps[:, 0:1, :],
                                                    bev_pred_bctps[:, 1:2, :],], 1)
                bev_pred_bctps_ = paddle.where(la_half_pi_inds.unsqueeze(-1).unsqueeze(-1), bev_pred_bctps_tmp, bev_pred_bctps_)
                bev_pred_bctps_ = paddle.where(le_half_pi_inds.unsqueeze(-1).unsqueeze(-1), bev_pred_bctps, bev_pred_bctps_)

            else:
                bev_pred_bctps_ = bev_pred_bctps

            bev_gt_bctps = bev_gt_bctps.reshape((batch_size_device, -1, 8))
            bev_pred_bctps = bev_pred_bctps_.reshape((batch_size_device, -1, 8))

            pred_bctps_loss = self.loss_pred_bctps(bev_pred_bctps, bev_gt_bctps, weights=border_mask_weights_pr)

            pred_bctps_loss_reduced = pred_bctps_loss.sum() / batch_size_device

            if isinstance(self.loss_pred_bctps._loss_weight, list):
                pred_bctps_loss_reduced *= self.loss_pred_bctps._loss_weight[task_id]
            else:
                pred_bctps_loss_reduced *= self.loss_pred_bctps._loss_weight

        if self.use_direction_classifier:
            dir_targets = get_direction_target(batch_anchors, reg_targets,
                dir_offset=self.direction_offset)
            dir_logits = dir_cls_preds.reshape((batch_size_device, -1, 2))
            weights = (labels > 0).astype(dtype=dir_logits.dtype)
            weights /= paddle.clip(x=weights.sum(axis=-1, keepdim=True),
                min=1.0)
            dir_loss = self.loss_aux(dir_logits, dir_targets, weights=weights)
            dir_loss = (dir_loss.sum() / batch_size_device) * task_loss_aux_weight_per_task
            dir_loss = dir_loss * self.loss_aux._loss_weight
        pull_push_sum_loss = None
        bbox_preds = bbox_preds.reshape((batch_size_device, -1, self.box_n_dim))
        if self.use_loss_pull_push:
            zero_loss = paddle.mean(x=bbox_preds).astype(dtype='float32') * 0
            batch_proposals = self.get_proposals(batch_anchors, bbox_preds,
                cls_scores, labels > 0, positive_gt_id[0], num_classes)
            pull_push_loss = self.loss_pull_push(gt_bboxes, gt_labels,
                batch_proposals, zero_loss)
            pull_push_sum_loss = pull_push_loss['pull_loss'] + pull_push_loss[
                'push_loss']
        iou_sum_loss = None
        if self.use_iou_score:
            zero_loss = paddle.to_tensor(data=0) 
            pos_pred_mask = labels > 0
            pos_box_preds_decode = self.box_coder.decode(batch_anchors[pos_pred_mask], 
                bbox_preds[pos_pred_mask])
            pos_reg_targets_decode = self.box_coder.decode(batch_anchors[pos_pred_mask], 
                reg_targets[pos_pred_mask])
            xy = paddle.clip(x=pos_box_preds_decode[:, :2], min=-60, max=60)
            z = paddle.clip(x=pos_box_preds_decode[:, 2:3], min=-7, max=7)
            wlh = paddle.clip(x=pos_box_preds_decode[:, 3:6], min=1e-06, max=50
                )
            yaw = paddle.clip(x=pos_box_preds_decode[:, 6:7], min=-31.4,
                max=31.4)
            pos_box_preds_decode = paddle.concat(x=[xy, z, wlh, yaw], axis=1)
            if iou_preds is not None:
                iou_preds = iou_preds.reshape((batch_size_device, -1))[pos_pred_mask]
            iou_loss = self.loss_iou(pos_box_preds_decode,
                pos_reg_targets_decode, iou_preds, zero_loss)
            iou_sum_loss = iou_loss['iou_score_loss'] + iou_loss['iou_loss']
        
        if loc_loss_reduced is not None:
            loc_loss_reduced *= task_weight
        if cls_loss_reduced is not None:
            cls_loss_reduced *= task_weight
        if dir_loss is not None:
            dir_loss *= task_weight
        if pull_push_sum_loss is not None:
            pull_push_sum_loss *= task_weight
        if iou_sum_loss is not None:
            iou_sum_loss *= task_weight

        return (loc_loss_reduced, cls_loss_reduced, dir_loss,
            pull_push_sum_loss, iou_sum_loss, an_bctps_loss_reduced, pred_bctps_loss_reduced, bg_cls_loss)

    def loss(self, 
            cls_scores,
            bbox_preds, 
            dir_cls_preds, 
            iou_preds,
            bctp_preds,
            gt_bboxes, 
            gt_labels, 
            input_metas, 
            coors, 
            test_cfg,
            num_points_in_gts=None, 
            gt_bboxes_ignore=None,
            gt_border_masks=None,
            roi_regions=None):
        featmap_sizes = np.array(self.grid_size)[:2] // self.downsample
        batch_anchors = self.get_anchors(featmap_sizes, input_metas, )
        for i in range(len(cls_scores)):
            if cls_scores[i] is not None:
                cls_scores[i] = cls_scores[i].transpose(perm=[0, 2, 3, 1])
            if bbox_preds[i] is not None:
                bbox_preds[i] = bbox_preds[i].transpose(perm=[0, 2, 3, 1])
            if dir_cls_preds[i] is not None:
                dir_cls_preds[i] = dir_cls_preds[i].transpose(perm=[0, 2, 3, 1])
            if iou_preds[i] is not None:
                iou_preds[i] = iou_preds[i].transpose(perm=[0, 2, 3, 1])
            if bctp_preds[i] is not None:
                bctp_preds[i] = bctp_preds[i].transpose([0, 2, 3, 1])
        # if (["pedestrian_sub"] in self.class_names) and len(gt_bboxes) == 6:
        # 8A
        # print("====test1 mgl==============") # TODO TODO TODO
        if self.use_sub_region_head:
            # gt_bboxes[5] = gt_bboxes[4]
            # gt_labels[5] = gt_labels[4]
            for class_name in self.class_names:
                if class_name[0].endswith(self.sub_region_postfix):
                    org_name = class_name[0][:-len(self.sub_region_postfix)]
                    org_idx = self.name2class[org_name]
                    sub_idx = self.name2class[class_name[0]]
                    # print("sub_idx: ", sub_idx)
                    # print("org_idx: ", org_idx)
                    # print("gt bboxes: ", gt_bboxes)
                    # print("gt labels: ", gt_labels)
                    gt_bboxes[sub_idx] = gt_bboxes[org_idx]
                    gt_labels[sub_idx] = gt_labels[org_idx]
                    gt_border_masks[sub_idx] = gt_border_masks[org_idx]
        # print("====test2 mgl==============")
        # elif (["pedestrian_sub"] in self.class_names) and (len(gt_bboxes) == 7 or len(gt_bboxes) == 8):
        #     gt_bboxes[6] = gt_bboxes[4]
        #     gt_labels[6] = gt_labels[4]
        #     gt_border_masks[6] = gt_border_masks[4]

        if self.assign_cfg:
            labels, reg_targets, reg_weights, positive_gt_id, anchors_mask, bctp_targets, border_mask_weights, regions_mask = self.assign_target_and_mask(coors, batch_anchors,
                gt_bboxes, gt_labels, bbox_preds, num_points_in_gts, gt_border_masks, roi_regions)
        batch_size_device = batch_anchors[0].shape[0]
        batch_size_devices = [batch_size_device for _ in range(len(gt_bboxes))]
        anchors_mask = [None for _ in range(len(gt_bboxes))]
        if self.cal_anchor_mask:
            anchors_mask = calculate_anchor_masks_paddle(batch_anchors, coors, 
                                                        self.grid_size, self.voxel_size, self.pc_range)
        
        if bctp_targets is None:
            bctp_targets = [None] * len(gt_bboxes)

        if border_mask_weights is None:
            border_mask_weights = [None] * len(gt_bboxes)     
        
        # obtain background objects
        num_task = len(self.tasks)
        if len(gt_bboxes) > num_task:
            assert len(gt_bboxes) == (num_task + 1)
            bg_bboxes = gt_bboxes[-1]
            bg_labels = gt_labels[-1]
        else:
            bg_bboxes = None
            bg_labels = None
        task_ids = [i for i in range(len(gt_bboxes))]

        (bbox_losses, cls_losses, dir_losses, pull_push_losses, iou_losses, an_bctps_loss, pred_bctps_loss, bg_cls_loss) = \
            multi_apply(self.loss_single, 
                        bbox_preds, 
                        cls_scores,
                        bctp_preds,
                        gt_bboxes[:num_task], 
                        gt_labels[:num_task], 
                        dir_cls_preds, 
                        iou_preds, 
                        labels[:num_task],
                        reg_targets[:num_task],
                        bctp_targets[:num_task], 
                        border_mask_weights[:num_task],
                        positive_gt_id[:num_task], 
                        self.num_classes, 
                        anchors_mask[:num_task],
                        batch_anchors[:num_task], 
                        self.task_loss_aux_weight,
                        self.tasks_weight,
                        batch_size_devices[:num_task],
                        task_ids[:num_task],
                        [bg_bboxes] * num_task,
                        [bg_labels] * num_task,
                        regions_mask[:num_task] if regions_mask is not None else [None] * num_task)

        losses = {'loss_bbox': bbox_losses, 'loss_cls': cls_losses}
        if self.use_direction_classifier:
            losses['loss_dir'] = dir_losses
        if self.use_loss_pull_push:
            losses['loss_pullpush'] = pull_push_losses
        if self.use_iou_score:
            losses['loss_iou'] = iou_losses
        if self.anchor_bctps and self.anchor_bctps['mode']:
            losses["an_bctps_loss"] = an_bctps_loss
        if self.pred_bctps and self.pred_bctps['mode']:
            losses["pred_bctps_loss"] = pred_bctps_loss
        if bg_bboxes is not None:
            losses["bg_cls_loss"] = bg_cls_loss

        return losses, anchors_mask, batch_anchors, labels

    def get_bboxes(self, 
                   cls_scores, 
                   bbox_preds, 
                   dir_cls_preds, 
                   iou_preds,
                   bctp_preds,
                   input_metas, 
                   coors, 
                   test_cfg, 
                   has_two_stage=True, 
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
        featmap_sizes = np.array(self.grid_size)[:2] // self.downsample
        batch_anchors = self.get_anchors(featmap_sizes, input_metas)
        for i in range(len(cls_scores)):
            if cls_scores[i] is not None:
                cls_scores[i] = cls_scores[i].transpose(perm=[0, 2, 3, 1])
            if bbox_preds[i] is not None:
                bbox_preds[i] = bbox_preds[i].transpose(perm=[0, 2, 3, 1])
            if dir_cls_preds[i] is not None:
                dir_cls_preds[i] = dir_cls_preds[i].transpose(perm=[0, 2, 3, 1]
                    )
            if iou_preds[i] is not None:
                iou_preds[i] = iou_preds[i].transpose(perm=[0, 2, 3, 1])
        rets = []
        batch_size_device = batch_anchors[0].shape[0]
        anchors_mask = [None for _ in range(len(self.tasks))]
        if self.cal_anchor_mask:
            anchors_mask = calculate_anchor_masks_paddle(batch_anchors, coors, 
                                                self.grid_size, self.voxel_size, self.pc_range)
        if self.assign_cfg:
            anchors_mask = self.assign_target_and_mask(coors, batch_anchors,
                None, None, bbox_preds, None, None, test_mode=True)
        if has_two_stage:
            return None, anchors_mask, batch_anchors
        for task_id, cls_score in enumerate(cls_scores):
            batch_size = batch_anchors[task_id].shape[0]
            if self.bev_only:
                batch_task_anchors = batch_anchors[task_id].reshape((batch_size, -1, self.box_n_dim + 2))
            else:
                batch_task_anchors = batch_anchors[task_id].reshape((batch_size, -1, self.box_n_dim))
            if anchors_mask[task_id] is not None:
                batch_anchors_mask = anchors_mask[task_id].reshape((batch_size, -1)) 
            batch_box_preds = bbox_preds[task_id]
            batch_cls_preds = cls_score
            if self.bev_only:
                box_ndim = self.box_n_dim
            else:
                box_ndim = self.box_n_dim
            if kwargs.get('mode', False):
                batch_box_preds_base = batch_box_preds.reshape((batch_size, -1, box_ndim))
                batch_box_preds = batch_task_anchors.clone()
                batch_box_preds[:, :, ([0, 1, 3, 4, 6])] = batch_box_preds_base
            else:
                batch_box_preds = batch_box_preds.reshape((batch_size, -1, box_ndim))
                    
            num_class_with_bg = self.num_classes[task_id]
            if not self.encode_background_as_zeros:
                num_class_with_bg = self.num_classes[task_id] + 1
            batch_cls_preds = batch_cls_preds.reshape((batch_size, -1, num_class_with_bg))
            batch_reg_preds = self.box_coder.decode(batch_task_anchors,
                batch_box_preds[:, :, :self.box_coder.code_size])
            if self.use_direction_classifier:
                batch_dir_preds = dir_cls_preds
                batch_dir_preds = batch_dir_preds.reshape((batch_size, -1, 2)) 
            else:
                batch_dir_preds = [None] * batch_size
            if len(test_cfg.nms.get('nms_groups', [])) > 0:
                rets.append(self.get_task_detections(task_id,
                    num_class_with_bg, test_cfg, batch_cls_preds,
                    batch_reg_preds, batch_dir_preds, batch_anchors_mask))
            else:
                rets.append(self.get_task_detections_with_nms(task_id,
                    num_class_with_bg, test_cfg, batch_cls_preds,
                    batch_reg_preds, batch_dir_preds, batch_anchors_mask))
        num_tasks = len(rets)
        num_preds = len(rets)
        num_samples = len(rets[0])
        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ['box3d_lidar', 'scores']:
                    ret[k] = paddle.concat(x=[ret[i][k] for ret in rets])
                elif k in ['label_preds']:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = paddle.concat(x=[ret[i][k] for ret in rets])
            ret_list.append(ret)
        nms_groups = test_cfg.nms.get('nms_groups', [])
        if len(nms_groups) != 0:
            if not isinstance(nms_groups[0][0], int):
                class_names = []
                for names in self.class_names:
                    class_names.extend(names)
                class2id = {token: i for i, token in enumerate(class_names)}
                for i, group in enumerate(nms_groups):
                    for j, name in enumerate(group):
                        nms_groups[i][j] = class2id[name]
            for i, ret in enumerate(ret_list):
                if ret['box3d_lidar'].shape[0] == 0:
                    continue
                box3d_lidar, scores, label_preds = self.nms_for_groups(test_cfg
                    , nms_groups, ret['box3d_lidar'], ret['scores'], ret[
                    'label_preds'])
                ret_list[i]['box3d_lidar'] = box3d_lidar
                ret_list[i]['scores'] = scores
                ret_list[i]['label_preds'] = label_preds
        nms_overlap_groups = test_cfg.nms.get('nms_overlap_groups', [])
        if len(nms_overlap_groups) != 0:
            if not isinstance(nms_overlap_groups[0][0], int):
                class_names = []
                for names in self.class_names:
                    class_names.extend(names)
                class2id = {token: i for i, token in enumerate(class_names)}
                for i, group in enumerate(nms_overlap_groups):
                    for j, name in enumerate(group):
                        nms_overlap_groups[i][j] = class2id[name]
            for i, ret in enumerate(ret_list):
                if ret['box3d_lidar'].shape[0] == 0:
                    continue
                box3d_lidar, scores, label_preds = self.nms_overlap_for_groups(
                    test_cfg, nms_overlap_groups, ret['box3d_lidar'], ret[
                    'scores'], ret['label_preds'])
                ret_list[i]['box3d_lidar'] = box3d_lidar
                ret_list[i]['scores'] = scores
                ret_list[i]['label_preds'] = label_preds
        bboxes = []
        scores = []
        labels = []
        for i, ret in enumerate(ret_list):
            bboxes.append(ret_list[i]['box3d_lidar'])
            scores.append(ret_list[i]['scores'])
            labels.append(ret_list[i]['label_preds'])
        bboxes = paddle.concat(x=bboxes, axis=0)
        bboxes = BBoxes3D(bboxes)
        scores = paddle.concat(x=scores, axis=0)
        labels = paddle.concat(x=labels, axis=0)
        return [[bboxes, scores, labels]], anchors_mask, batch_anchors

    def get_task_detections(self, task_id, num_class_with_bg, test_cfg,
        batch_cls_preds, batch_reg_preds, batch_dir_preds=None,
        batch_iou_preds=None, # add8a
        batch_anchors_mask=None):
        predictions_dicts = []
        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = paddle.to_tensor(data=post_center_range,
                dtype=batch_reg_preds.dtype, place=batch_reg_preds.place)
        for box_preds, cls_preds, dir_preds, iou_preds, a_mask in zip(batch_reg_preds,
            batch_cls_preds, batch_dir_preds, batch_iou_preds, batch_anchors_mask):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.astype(dtype='float32')
            cls_preds = cls_preds.astype(dtype='float32')
            if self.use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = paddle.argmax(dir_preds, axis=-1) 
            
            # =========
            # add 8a TODO1023
            if iou_preds is not None:
                iou_preds = iou_preds[a_mask].cast("float32") #.float()
                iou_scores = F.sigmoid(iou_preds).squeeze(-1)
            
            # =========
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
                top_labels = paddle.zeros(shape=total_scores.shape[0],
                    dtype='int64')
            else:
                top_scores = paddle.max(total_scores, axis = -1)
                top_labels = paddle.argmax(total_scores, axis = -1)   

            # if test_cfg.score_threshold > 0.0:
            
            if isinstance(test_cfg.score_threshold, list):
                score_threshold = test_cfg.score_threshold[task_id]
            else:
                score_threshold = test_cfg.score_threshold

            if score_threshold > 0.0:  
                thresh = paddle.to_tensor(data=[score_threshold],
                    place=total_scores.place).astype(dtype=total_scores.dtype)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(mask=top_scores_keep)

            if top_scores.shape[0] != 0:
                # if test_cfg.score_threshold > 0.0:
                if score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    if self.use_direction_classifier:
                        dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                if self.use_direction_classifier:
                    opp_labels = (box_preds[..., -1] - self.
                        direction_offset > 0) ^ dir_labels.astype(dtype='bool')
                    box_preds[..., -1] += paddle.where(condition=opp_labels, x=
                        paddle.to_tensor(data=np.pi).astype(dtype=box_preds
                        .dtype), y=paddle.to_tensor(data=0.0).astype(dtype=
                        box_preds.dtype))

                if iou_preds is not None:
                    iou_scores = iou_scores[top_scores_keep]

                if post_center_range is not None:
                    mask = (box_preds[:, :3] >= post_center_range[:3]).all(axis
                        =1)
                    mask &= (box_preds[:, :3] <= post_center_range[3:]).all(
                        axis=1)
                    predictions_dict = {
                        'box3d_lidar': box_preds[mask],
                        'scores': top_scores[mask], 
                        'label_preds': top_labels[mask],
                        "iou_scores": iou_scores[mask] if iou_preds is not None else None} # 8a
                else:
                    predictions_dict = {
                        'box3d_lidar': box_preds, 
                        'scores': top_scores, 
                        'label_preds': top_labels,
                        "iou_scores": iou_scores if iou_preds is not None else None} # 8a
            else:
                dtype = batch_reg_preds.dtype
                # box_ndim = (self.box_n_dim + 2 if self.bev_only else self.box_n_dim)
                predictions_dict = {
                    'box3d_lidar': paddle.zeros(shape=[0, self.box_n_dim], dtype=dtype), 
                    'scores': paddle.zeros(shape=[0], dtype=dtype), 
                    'label_preds': paddle.zeros(shape=[0], dtype=top_labels.dtype),
                    "iou_scores": torch.zeros([0], dtype=dtype, device=device) if iou_preds is not None else None}
            predictions_dicts.append(predictions_dict)
        return predictions_dicts

    def get_task_detections_with_nms(self, task_id, num_class_with_bg,
        test_cfg, batch_cls_preds, batch_reg_preds, batch_dir_preds=None,
        batch_anchors_mask=None):
        predictions_dicts = []
        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = paddle.to_tensor(data=post_center_range,
                dtype=batch_reg_preds.dtype, place=batch_reg_preds.place)
        for box_preds, cls_preds, dir_preds, a_mask in zip(batch_reg_preds,
            batch_cls_preds, batch_dir_preds, batch_anchors_mask):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.astype(dtype='float32')
            cls_preds = cls_preds.astype(dtype='float32')
            if self.use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = paddle.argmax(dir_preds, axis=-1) 
            if self.encode_background_as_zeros:
                assert self.use_sigmoid_score is True
                total_scores = paddle.nn.functional.sigmoid(x=cls_preds)
            elif self.use_sigmoid_score:
                total_scores = paddle.nn.functional.sigmoid(x=cls_preds)[(
                    ...), 1:]
            else:
                total_scores = paddle.nn.functional.softmax(x=cls_preds,
                    axis=-1)[(...), 1:]
            if test_cfg.nms.use_rotate_nms:
                nms_func = rotate_nms_pcdet
            else:
                nms_func = nms
            feature_map_size_prod = batch_reg_preds.shape[1] // self.num_anchor_per_locs[task_id]
            if test_cfg.nms.use_multi_class_nms:
                assert self.encode_background_as_zeros is True
                boxes_for_nms = box_preds[:, ([0, 1, 3, 4, -1])]
                if not test_cfg.nms.use_rotate_nms:
                    box_preds_corners = center_to_corner_box2d(boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, (4)])
                    boxes_for_nms = corner_to_standup_nd(box_preds_corners)
                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []
                scores = total_scores
                boxes = boxes_for_nms
                selected_per_class = []
                score_threshs = [test_cfg.score_threshold] * self.num_classes[
                    task_id]
                pre_max_sizes = [test_cfg.nms.nms_pre_max_size
                    ] * self.num_classes[task_id]
                post_max_sizes = [test_cfg.nms.nms_post_max_size
                    ] * self.num_classes[task_id]
                iou_thresholds = [test_cfg.nms.nms_iou_threshold
                    ] * self.num_classes[task_id]
                add_iou_edge = test_cfg.nms.get('add_iou_edge', 1.0)
                for class_idx, score_thresh, pre_ms, post_ms, iou_th in zip(
                    range(self.num_classes[task_id]), score_threshs,
                    pre_max_sizes, post_max_sizes, iou_thresholds):
                    self._nms_class_agnostic = False
                    if self._nms_class_agnostic:
                        class_scores = total_scores.reshape((feature_map_size_prod, -1, self.num_classes[task_id]))[..., class_idx]
                        class_scores = class_scores.reshape((-1))
                        class_boxes_nms = boxes.reshape((-1, boxes_for_nms.shape[-1]))
                        class_boxes = box_preds
                        class_dir_labels = dir_labels
                    else:
                        anchors_range = self.target_assigners[task_id
                            ].anchors_range
                        class_scores = total_scores.reshape((-1, self._num_classes[task_id]))[anchors_range[0]:anchors_range[1], (class_idx)]
                        class_boxes_nms = boxes.reshape((-1, boxes_for_nms.shape[-1]))[anchors_range[0]:anchors_range[1], :]
                        class_scores = class_scores.reshape(-1) 
                        class_boxes_nms = class_boxes_nms.reshape((-1, boxes_for_nms.shape[-1]))
                        class_boxes = box_preds.reshape((-1, box_preds.shape[-1]))[anchors_range[0]:anchors_range[1], :]
                        class_boxes = class_boxes.reshape((-1, box_preds.shape[-1])) 
                        if self.use_direction_classifier:
                            class_dir_labels = dir_labels.reshape((-1))[anchors_range[0]:anchors_range[1]]
                            class_dir_labels = class_dir_labels.reshape((-1))
                    if score_thresh > 0.0:
                        class_scores_keep = class_scores >= score_thresh
                        if class_scores_keep.shape[0] == 0:
                            selected_per_class.append(None)
                            continue
                        class_scores = class_scores[class_scores_keep]
                    if class_scores.shape[0] != 0:
                        if score_thresh > 0.0:
                            class_boxes_nms = class_boxes_nms[class_scores_keep
                                ]
                            class_boxes = class_boxes[class_scores_keep]
                            class_dir_labels = class_dir_labels[
                                class_scores_keep]
                        keep = nms_func(class_boxes_nms, class_scores,
                            pre_ms, post_ms, iou_th, add_iou_edge=add_iou_edge)
                        if keep.shape[0] != 0:
                            selected_per_class.append(keep)
                        else:
                            selected_per_class.append(None)
                    else:
                        selected_per_class.append(None)
                    selected = selected_per_class[-1]
                    if selected is not None:
                        selected_boxes.append(class_boxes[selected])
                        selected_labels.append(paddle.full(shape=[
                            class_boxes[selected].shape[0]], fill_value=
                            class_idx).astype('int64'))
                        if self.use_direction_classifier:
                            selected_dir_labels.append(class_dir_labels[
                                selected])
                        selected_scores.append(class_scores[selected])
                selected_boxes = paddle.concat(x=selected_boxes, axis=0)
                selected_labels = paddle.concat(x=selected_labels, axis=0)
                selected_scores = paddle.concat(x=selected_scores, axis=0)
                if self.use_direction_classifier:
                    selected_dir_labels = paddle.concat(x=
                        selected_dir_labels, axis=0)
            else:
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(axis=-1)
                    top_labels = paddle.zeros(shape=total_scores.shape[0],
                        dtype='int64')
                else:
                    top_scores = paddle.max(total_scores, axis=-1)
                    top_labels = paddle.argmax(total_scores, axis=-1) 
                if test_cfg.score_threshold > 0.0:
                    thresh = paddle.to_tensor(data=[test_cfg.
                        score_threshold], place=total_scores.place).astype(
                        dtype=total_scores.dtype)
                    top_scores_keep = top_scores >= thresh
                    top_scores = top_scores.masked_select(mask=top_scores_keep)
                if top_scores.shape[0] != 0:
                    if test_cfg.score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self.use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, ([0, 1, 3, 4, -1])]
                    if not test_cfg.nms.use_rotate_nms:
                        box_preds_corners = center_to_corner_box2d(boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], 
                            boxes_for_nms[:, (4)])
                        boxes_for_nms = corner_to_standup_nd(box_preds_corners)
                    selected = nms_func(boxes_for_nms, top_scores,
                        pre_max_size=test_cfg.nms.nms_pre_max_size,
                        post_max_size=test_cfg.nms.nms_post_max_size,
                        iou_threshold=test_cfg.nms.nms_iou_threshold,
                        add_iou_edge=add_iou_edge)
                else:
                    selected = []
                selected_boxes = box_preds[selected]
                if self.use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self.use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] - self.direction_offset > 0) ^ dir_labels.astype(dtype='bool')
                    box_preds[..., -1] += paddle.where(condition=opp_labels, 
                                                    x=paddle.to_tensor(data=np.pi).astype(dtype=box_preds.dtype), 
                                                    y=paddle.to_tensor(data=0.0).astype(dtype=box_preds.dtype))
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >= post_center_range[:3]).all(axis=1)
                    mask &= (final_box_preds[:, :3] <= post_center_range[3:]).all(axis=1)
                    predictions_dict = {'box3d_lidar': final_box_preds[mask], 
                                        'scores': final_scores[mask], 
                                        'label_preds': label_preds[mask]}
                else:
                    predictions_dict = {'box3d_lidar': final_box_preds,
                        'scores': final_scores, 'label_preds': label_preds}
            else:
                dtype = batch_reg_preds.dtype
                box_ndim = (self.box_n_dim + 2 if self.bev_only else self.box_n_dim)
                predictions_dict = {'box3d_lidar': paddle.zeros(shape=[0, box_ndim], dtype=dtype), 
                                    'scores': paddle.zeros(shape=[0], dtype=dtype), 
                                    'label_preds': paddle.zeros(shape=[0], dtype=top_labels.dtype)}
            predictions_dicts.append(predictions_dict)
        return predictions_dicts

    def nms_for_groups(self, test_cfg, nms_groups, box3d_lidar, scores,
                        iou_scores,
                        label_preds):
        group_nms_pre_max_size = test_cfg.nms.get('group_nms_pre_max_size', [])
        group_nms_post_max_size = test_cfg.nms.get('group_nms_post_max_size', [])
        group_nms_iou_threshold = test_cfg.nms.get('group_nms_iou_threshold', [])
        # add_iou_edge = test_cfg.nms.get('add_iou_edge', 1.0)
        add_iou_edge = test_cfg.nms.get('add_iou_edge', 0.0)

        if len(group_nms_pre_max_size) == 0:
            group_nms_pre_max_size = [test_cfg.nms.nms_pre_max_size] * len(
                nms_groups)
        if len(group_nms_post_max_size) == 0:
            group_nms_post_max_size = [test_cfg.nms.nms_post_max_size] * len(
                nms_groups)
        if len(group_nms_iou_threshold) == 0:
            group_nms_iou_threshold = [test_cfg.nms.nms_iou_threshold] * len(
                nms_groups)
        assert len(group_nms_pre_max_size) == len(nms_groups)
        assert len(group_nms_post_max_size) == len(nms_groups)
        assert len(group_nms_iou_threshold) == len(nms_groups)
        if test_cfg.nms.use_rotate_nms:
            nms_func = rotate_nms_pcdet
        else:
            nms_func = nms
        boxes_for_nms = box3d_lidar[:, ([0, 1, 3, 4, -1])]
        if not test_cfg.nms.use_rotate_nms:
            box_preds_corners = center_to_corner_box2d(boxes_for_nms[:, :2], 
                                                    boxes_for_nms[:, 2:4], 
                                                    boxes_for_nms[:, (4)])
            boxes_for_nms = corner_to_standup_nd(box_preds_corners)
        for group_id, nms_group in enumerate(nms_groups):
            selecteds = label_preds >= 0
            if len(nms_group) == 0:
                continue
            mask = label_preds == nms_group[0]
            for label_id in nms_group:
                mask |= label_preds == label_id
            indices = paddle.nonzero(x=mask, as_tuple=True)[0]
            if indices.shape[0] != 0:
                group_boxes_for_nms = boxes_for_nms[indices]
                group_scores = scores[indices]

                group_iou_scores = iou_scores[indices] if iou_scores is not None else None
                if group_iou_scores is not None:
                    group_scores = group_scores * group_iou_scores
                    #group_scores = group_iou_scores

                selected = nms_func(group_boxes_for_nms, group_scores,
                    pre_max_size=group_nms_pre_max_size[group_id],
                    post_max_size=group_nms_post_max_size[group_id],
                    iou_threshold=group_nms_iou_threshold[group_id],
                    add_iou_edge=add_iou_edge)
                selected_indices = indices[selected]
                selecteds[indices] = False
                selecteds[selected_indices] = True
                boxes_for_nms = boxes_for_nms[selecteds]
                box3d_lidar = box3d_lidar[selecteds]
                label_preds = label_preds[selecteds]
                scores = scores[selecteds]
                iou_scores = iou_scores[selecteds] if iou_scores is not None else None

        return box3d_lidar, scores, iou_scores, label_preds

    def nms_overlap_for_groups(self, 
                                test_cfg, 
                                nms_overlap_groups,
                                box3d_lidar, 
                                scores, 
                                iou_scores,
                                label_preds):
        group_nms_overlap_pre_max_size = test_cfg.nms.get(
            'group_nms_overlap_pre_max_size', [])
        group_nms_overlap_post_max_size = test_cfg.nms.get(
            'group_nms_overlap_post_max_size', [])
        group_nms_overlap_iou_threshold = test_cfg.nms.get(
            'group_nms_overlap_iou_threshold', [])
        if len(group_nms_overlap_pre_max_size) == 0:
            group_nms_overlap_pre_max_size = [test_cfg.nms.nms_pre_max_size
                ] * len(nms_overlap_groups)
        if len(group_nms_overlap_post_max_size) == 0:
            group_nms_overlap_post_max_size = [test_cfg.nms.nms_post_max_size
                ] * len(nms_overlap_groups)
        if len(group_nms_overlap_iou_threshold) == 0:
            group_nms_overlap_iou_threshold = [test_cfg.nms.nms_iou_threshold
                ] * len(nms_overlap_groups)
        assert len(group_nms_overlap_pre_max_size) == len(nms_overlap_groups)
        assert len(group_nms_overlap_post_max_size) == len(nms_overlap_groups)
        assert len(group_nms_overlap_iou_threshold) == len(nms_overlap_groups)
        if not hasattr(test_cfg.nms, 'use_rotate_nms_overlap') or not test_cfg.nms.use_rotate_nms_overlap:
            nms_overlap_func = nms_overlap 
        else:
            nms_overlap_func = rotate_nms_overlap
        boxes_for_nms = box3d_lidar[:, ([0, 1, 3, 4, -1])]
        if not hasattr(test_cfg.nms, 'use_rotate_nms_overlap'
            ) or not test_cfg.nms.use_rotate_nms_overlap:
            box_preds_corners = center_to_corner_box2d(boxes_for_nms[:, :2], boxes_for_nms[
                :, 2:4], boxes_for_nms[:, (4)])
            boxes_for_nms = corner_to_standup_nd(box_preds_corners)
        for group_id, nms_group in enumerate(nms_overlap_groups):
            selecteds = label_preds >= 0
            if len(nms_group) == 0:
                continue
            mask = label_preds == nms_group[0]
            for label_id in nms_group:
                mask |= label_preds == label_id
            indices = paddle.nonzero(x=mask, as_tuple=True)[0]
            if indices.shape[0] != 0:
                group_boxes_for_nms = boxes_for_nms[indices]
                group_scores = scores[indices]

                # 8A
                group_iou_scores = iou_scores[indices] if iou_scores is not None else None
                if group_iou_scores is not None:
                    group_scores = group_scores * group_iou_scores

                selected = nms_overlap_func(group_boxes_for_nms,
                    group_scores, pre_max_size = group_nms_overlap_pre_max_size[group_id],
                    post_max_size=group_nms_overlap_post_max_size[group_id],
                    overlap_threshold=group_nms_overlap_iou_threshold[group_id]
                    )
                selected_indices = indices[selected]
                selecteds[indices] = False
                selecteds[selected_indices] = True
                boxes_for_nms = boxes_for_nms[selecteds]
                box3d_lidar = box3d_lidar[selecteds]
                label_preds = label_preds[selecteds]
                scores = scores[selecteds]
                iou_scores = iou_scores[selecteds] if iou_scores is not None else None
        return box3d_lidar, scores, iou_scores, label_preds



'''
# TODO torch get_proposal_for_rcnn 8A
    @torch.no_grad()
	
    def get_proposal_for_rcnn(self, 
                cls_scores,
                bbox_preds,
                dir_cls_preds,
                iou_preds,
                bctp_preds,
                input_metas,
                coors,
                test_cfg,
                roi_regions=None,
                gt_boxes=None,
                gt_labels=None,
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
        featmap_sizes = np.array(self.grid_size)[:2] // self.downsample

        device = cls_scores[0].device
        batch_anchors = self.get_anchors(
            featmap_sizes, input_metas, device=device)
        batch_size_device = batch_anchors[0].shape[0]
        for i in range(len(cls_scores)):
            if cls_scores[i] is not None:
                cls_scores[i] = cls_scores[i].permute(0, 2, 3, 1).contiguous()
            if bbox_preds[i] is not None:
                bbox_preds[i] = bbox_preds[i].permute(0, 2, 3, 1).contiguous()
            if dir_cls_preds[i] is not None:
                dir_cls_preds[i] = dir_cls_preds[i].permute(0, 2, 3, 1).contiguous()
            if iou_preds[i] is not None:
                iou_preds[i] = iou_preds[i].permute(0, 2, 3, 1).contiguous()

        
        anchors_mask = [None for _ in range(len(self.tasks))] 
        if self.assign_cfg:
            anchors_mask = self.assign_target_and_mask(coors, batch_anchors, None, None, bbox_preds, None, None, test_mode=True)
        assert (len(anchors_mask) == len(cls_scores)) or ((len(anchors_mask) == len(cls_scores) + 1))

        rets = []

        if self.use_sub_region_head:
            # {class_name: class_result}
            sub_region_result = dict()
            out_region_result = dict()

        for task_id, cls_score in enumerate(cls_scores):
            batch_size = batch_anchors[task_id].shape[0]
            batch_task_anchors = batch_anchors[task_id].view(
                batch_size, -1, self.box_n_dim
            )

            if anchors_mask[task_id] is None:
                batch_anchors_mask = [None] * batch_size
            else:
                batch_anchors_mask = anchors_mask[task_id].view(
                    batch_size, -1
                )

            batch_box_preds = bbox_preds[task_id]
            batch_cls_preds = cls_score

            box_ndim = self.box_n_dim

            if kwargs.get("mode", False):
                batch_box_preds_base = batch_box_preds.view(batch_size, -1, box_ndim)
                batch_box_preds = batch_task_anchors.clone()
                batch_box_preds[:, :, [0, 1, 3, 4, 6]] = batch_box_preds_base
            else:
                batch_box_preds = batch_box_preds.view(batch_size, -1, box_ndim)

            num_class_with_bg = self.num_classes[task_id]

            if not self.encode_background_as_zeros:
                num_class_with_bg = self.num_classes[task_id] + 1

            batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)

            batch_reg_preds = self.box_coder.decode(
                batch_task_anchors, batch_box_preds[:, :, : self.box_coder.code_size]
            )

            if self.use_direction_classifier:
                batch_dir_preds = dir_cls_preds[task_id]
                batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
            else:
                batch_dir_preds = [None] * batch_size


            if None not in iou_preds:
                batch_iou_preds = iou_preds[task_id]
            else:
                batch_iou_preds = [None] * batch_size

            if batch_iou_preds[0] is not None:
                batch_iou_preds = batch_iou_preds.view(batch_size, -1, 1)

            if self.use_sub_region_head:
                cur_class_name = self.class_names[task_id]

                if cur_class_name in self.sub_region_class:  # for ori_head
                    # calculate out_range anchor_mask, drop low-pixel region result
                    sub_region_mask = batch_anchors[task_id][:, :, 0] > self.sub_region_range[0][0]
                    sub_region_mask = sub_region_mask & (batch_anchors[task_id][:, :, 0] < self.sub_region_range[0][1])
                    sub_region_mask = sub_region_mask & (batch_anchors[task_id][:, :, 1] > self.sub_region_range[1][0])
                    sub_region_mask = sub_region_mask & (batch_anchors[task_id][:, :, 1] < self.sub_region_range[1][1])
                    sub_region_mask = ~sub_region_mask

                    # drop the raw_head results which are in sub_region via sub_region_mask
                    batch_reg_preds = batch_reg_preds[sub_region_mask].reshape(batch_size, -1, 7)
                    batch_cls_preds = batch_cls_preds[sub_region_mask].reshape(batch_size, -1, 1)
                    batch_dir_preds = batch_dir_preds[sub_region_mask].reshape(batch_size, -1, 2)
                    batch_anchors_mask = batch_anchors_mask[sub_region_mask].reshape(batch_size, -1)

                    if batch_iou_preds[0] is not None:
                        batch_iou_preds = batch_iou_preds[sub_region_mask].reshape(batch_size, -1, 1)
                        out_region_result[cur_class_name[0]] = [batch_reg_preds, batch_cls_preds,
                                                                batch_dir_preds, batch_anchors_mask, 
                                                                batch_iou_preds]
                    else:
                        out_region_result[cur_class_name[0]] = [batch_reg_preds, batch_cls_preds,
                                                                batch_dir_preds, batch_anchors_mask]
                    if not sub_region_result.get(cur_class_name[0]):
                        continue

                if cur_class_name[0].endswith(self.sub_region_postfix):  # for sub_head
                    # get sub_region_head prediction result
                    if batch_iou_preds[0] is not None:
                        sub_region_result[cur_class_name[0][0: -len(self.sub_region_postfix)]] = [batch_reg_preds, batch_cls_preds,
                                                                                                  batch_dir_preds, batch_anchors_mask,
                                                                                                  batch_iou_preds]
                    else:
                        sub_region_result[cur_class_name[0][0: -len(self.sub_region_postfix)]] = [batch_reg_preds, batch_cls_preds,
                                                                                                  batch_dir_preds, batch_anchors_mask]

                    if not out_region_result.get(cur_class_name[0][0: -len(self.sub_region_postfix)]):
                        continue

                ori_task_id = self.class2id.get(cur_class_name[0])
                ori_class_name = self.class_names[ori_task_id][0]

                # reset the task_id and get original class name if out_region_result and sub_region_result is filled by current class
                if sub_region_result.get(ori_class_name) and out_region_result.get(ori_class_name):
                    # merge sub_region and out_region
                    batch_reg_preds = torch.cat((out_region_result.get(ori_class_name)[0], 
                                                 sub_region_result.get(ori_class_name)[0]), dim=1)
                    batch_cls_preds = torch.cat((out_region_result.get(ori_class_name)[1], 
                                                 sub_region_result.get(ori_class_name)[1]), dim=1)
                    batch_dir_preds = torch.cat((out_region_result.get(ori_class_name)[2], 
                                                 sub_region_result.get(ori_class_name)[2]), dim=1)
                    batch_anchors_mask = torch.cat((out_region_result.get(ori_class_name)[3], 
                                                    sub_region_result.get(ori_class_name)[3]), dim=1)
                    if batch_iou_preds[0] is not None:
                        batch_iou_preds = torch.cat((out_region_result.get(ori_class_name)[4], 
                                                    sub_region_result.get(ori_class_name)[4]), dim=1)

            rets.append(
                self.get_task_detections(
                    task_id,
                    num_class_with_bg,
                    test_cfg,
                    batch_cls_preds,
                    batch_reg_preds,
                    batch_dir_preds,
                    batch_iou_preds,
                    batch_anchors_mask
                )
            )            
        if ['pedestrian_sub'] in self.class_names and ['verybigMot'] in self.class_names:
            if self.class_names.index(['pedestrian_sub']) > self.class_names.index(['verybigMot']):
                ped_pred = rets[5]
                verybigmot_pred = rets[4]
                rets[4] = ped_pred
                rets[5] = verybigmot_pred

        # Merge branches results
        num_tasks = len(rets)
        ret_list = []
        # len(rets) == task num
        # len(rets[0]) == batch_size
        num_preds = len(rets) # task num
        num_samples = len(rets[0]) #batch_size

        # ori_class_name = []
        # ori_class_name2id = {}
        # for i, cur_classe_name in enumerate(self.class_names):
        #     if cur_classe_name[0].endswith('_sub'):
        #         continue
        #     ori_class_name.append(1)
        #     ori_class_name2id[cur_classe_name[0]] = i

        # ret_list = []
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
        #OrderedDict([('TrainedOthers', 0), ('smallMot', 1), ('bigMot', 2), ('nonMot', 3), ('pedestrian', 4), ('verybigMot', 2), ('accessory_main', 5)])
        assert len(ori_class_name_map) == len(rets)

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores", "iou_scores"]:
                    if rets[0][i][k] is not None:
                        ret[k] = torch.cat([ret[i][k] for ret in rets])
                    else:
                        ret[k] = None
                elif k in ["label_preds"]:
                    flag = 0
                    for j, (cls_name,cls_id) in enumerate(ori_class_name_map.items()):
                       
                        rets[j][i][k] += cls_id  # recover class id, such as 4th class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k == "metadata":
                    # metadata
                    ret[k] = rets[0][i][k]
            ret_list.append(ret)

        
        proposal_list = []
        nms_groups = test_cfg.nms.get('nms_groups', [])
        if len(nms_groups) != 0:
            # change class names in nms_group to id
            if not isinstance(nms_groups[0][0], int):
                class_names = []
                for names in self.class_names:
                    class_names.extend(names)
                # class2id = {token: i for i, token in enumerate(class_names)}
                for i, group in enumerate(nms_groups):
                    for j, name in enumerate(group):
                        nms_groups[i][j] = self.class2id[name]
            for i, ret in enumerate(ret_list):
                proposal_dict=dict()
                if ret['box3d_lidar'].shape[0] == 0:
                    # continue
                    proposal_dict['rois'] = ret['box3d_lidar']
                    proposal_dict['roi_scores'] = ret['scores']
                    proposal_dict['roi_labels'] = ret['label_preds']
                else:
                    box3d_lidar, scores, iou_scores, label_preds = self.nms_for_groups(
                                                                test_cfg,
                                                                nms_groups,
                                                                ret['box3d_lidar'], # n * [x,y,z,w,l,h,yaw]
                                                                ret['scores'], # n ret['scores']
                                                                ret['iou_scores'],
                                                                ret['label_preds']) # n
                
                    proposal_dict['rois'] = box3d_lidar
                    proposal_dict['roi_scores'] = scores
                    proposal_dict['roi_labels'] = label_preds
                proposal_list.append(proposal_dict)
        
        if 'accessory_main' in ori_class_name_map:
            
            for i, ret in enumerate(proposal_list):
                if ret['rois'].shape[0] == 0:
                    continue
                no_head_inds = ret['roi_labels'] != ori_class_name_map['accessory_main']
                rois = ret['rois'][no_head_inds]
                roi_scores = ret['roi_scores'][no_head_inds]
                roi_labels = ret['roi_labels'][no_head_inds]

                head_inds = ret['roi_labels'] == ori_class_name_map['accessory_main']
                bigmot_inds = ret['roi_labels'] == ori_class_name_map['bigMot']
                
                bigmot_boxes = ret['rois'][bigmot_inds]
                head_boxes = ret['rois'][head_inds]
                head_scores = ret['roi_scores'][head_inds]
                head_labels = ret['roi_labels'][head_inds]

                if bigmot_boxes.shape[0] == 0:
                    continue
                if head_boxes.shape[0] == 0:
                    continue

                head_scores_matrix = head_scores.repeat(bigmot_boxes.size(0), 1)
                bev_iom = iou3d_utils.boxes_iom_bev(bigmot_boxes, head_boxes)
                neg_head_inds = bev_iom <= 0.5
                head_scores_matrix[neg_head_inds] = -1
                neg_head_inds = head_scores_matrix <= 0.2
                head_scores_matrix[neg_head_inds] = -1

                combo_uid = head_scores_matrix.max(1)[1]
                combo_uid[head_scores_matrix.max(1)[0] < 0] = -1
                valid_head_inds = combo_uid[combo_uid>=0]
                head_boxes = head_boxes[valid_head_inds]
                head_scores = head_scores[valid_head_inds]
                head_labels = head_labels[valid_head_inds]
                proposal_list[i]['rois'] = torch.cat([rois, head_boxes],dim=0)
                proposal_list[i]['roi_scores'] = torch.cat([roi_scores, head_scores],dim=0)
                proposal_list[i]['roi_labels'] = torch.cat([roi_labels, head_labels],dim=0)
        # for i, ret in enumerate(proposal_list):
        #     unique_value, unique_counts = torch.unique(proposal_list[i]['roi_labels'], return_counts=True)
        #     print("unique_value {} unique_counts {}".format(unique_value, unique_counts))
        if roi_regions is not None:
            for batch_id in range(batch_size):  #rcnn阶段只考虑标注区域内的proposal
                rois = proposal_list[batch_id]['rois']
                roi_scores = proposal_list[batch_id]['roi_scores']
                roi_labels = proposal_list[batch_id]['roi_labels']
                if rois.shape[0] == 0:
                    continue
                mask = torch.zeros_like(rois[:, 0], dtype=torch.bool)
                regions = roi_regions[batch_id]
                xy_a = rois[:, [0, 1]]
                regions_ = []
                for region in regions:
                    if region['type'] == 2:
                        regions_.append(region)
                if len(regions_) == 0:
                    mask[:] = True
                else:
                    for region in regions_:
                        if region['type'] == 2:
                            center_xf = region['region'][0]
                            center_yf = region['region'][1]
                            radius = region['region'][3]
                            center = torch.tensor(
                                [center_xf, center_yf],
                                dtype=rois.dtype,
                                device=rois.device).view(-1, 2)
                            dist = torch.norm(xy_a - center, p=2, dim=1)
                            mask[dist <= radius] = True
                        
                proposal_list[batch_id]['rois'] = rois[mask]
                proposal_list[batch_id]['roi_scores'] = roi_scores[mask]
                proposal_list[batch_id]['roi_labels'] = roi_labels[mask]
        
        
        gt_aug_factor = test_cfg.get("gt_aug_factor", [1, 1, 1, 1, 1, 1])  # 每个类的增加倍数
        for batch_id in range(len(proposal_list)):
            ret = proposal_list[batch_id]
            valid_mask = (ret["rois"][:, 3:6] > 1e-5).all(dim=1)
            ret["rois"] = ret["rois"][valid_mask]
            # if self.roi_aug:
            #     ret['rois'] = noise_gt_bboxes_(ret['rois'].unsqueeze(0)).squeeze(0)
            ret["roi_scores"] = ret["roi_scores"][valid_mask]
            ret["roi_labels"] = ret["roi_labels"][valid_mask]
            # print("mghead training ", self.training)
            # if ret["rois"].shape[0] > test_cfg.nms.nms_post_max_size and self.training:
            #     topk_idx = ret["roi_scores"].topk(
            #         test_cfg.nms.nms_post_max_size, dim=0)[1].view(-1)
            #     ret["rois"] = ret["rois"][topk_idx]
            #     ret["roi_scores"] = ret["roi_scores"][topk_idx]
            #     ret["roi_labels"] = ret["roi_labels"][topk_idx]
            if test_cfg.use_gt_boxes:
                assert gt_boxes is not None
                # mask = (gt_classes[batch_id] > 0)
                cur_gt_boxes = gt_boxes[batch_id].tensor.clone().to(device)
                cur_gt_labels = gt_labels[batch_id].clone()
                cur_gt_labels[cur_gt_labels==5] = 2 #合并verybigmot和bigmot
                cur_gt_labels[cur_gt_labels==7] = 5 # 重映射accessory_main 7-->5
                
                # mask = torch.zeros_like(gt_cls_list[batch_id]).bool()
                for cls_id in test_cfg.gt_classes:
                    mask = cur_gt_labels == cls_id
                    aug_factor = gt_aug_factor[cls_id]
                    extra_rois = cur_gt_boxes[mask].repeat(aug_factor, 1)
                    extra_roi_labels = cur_gt_labels[mask].repeat(aug_factor)
                    extra_roi_scores = extra_rois.new_ones((extra_rois.shape[0],)) * 0.01
                    extra_rois = noise_gt_bboxesv2_(extra_rois)
                    ret["rois"] = torch.cat([extra_rois, ret["rois"]], dim=0)
                    ret["roi_scores"] = torch.cat([extra_roi_scores, ret["roi_scores"]], dim=0)
                    ret["roi_labels"] = torch.cat([extra_roi_labels, ret["roi_labels"]], dim=0)
                # ret['roi_is_gt'] = ret['roi_is_gt'][topk_idx]
            proposal_list[batch_id] = ret
        
        return proposal_list

'''
