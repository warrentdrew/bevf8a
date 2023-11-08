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

from paddle3d.apis import manager
from paddle3d.models.losses.utils import weight_reduce_loss
from paddle3d.utils.box import bbox_overlaps


class IOULoss(nn.Layer):
    """
    Intersetion Over Union (IoU) loss
    This code is based on https://github.com/aim-uofa/AdelaiDet/blob/master/adet/layers/iou_loss.py
    """

    def __init__(self, loc_loss_type='iou'):
        """
        Args:
            loc_loss_type: str, supports three IoU computations: 'iou', 'linear_iou', 'giou'.
        """
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: Nx4 predicted bounding boxes
            target: Nx4 target bounding boxes
            weight: N loss weight for each instance
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = paddle.minimum(pred_left, target_left) + \
                      paddle.minimum(pred_right, target_right)
        h_intersect = paddle.minimum(pred_bottom, target_bottom) + \
                      paddle.minimum(pred_top, target_top)

        g_w_intersect = paddle.maximum(pred_left, target_left) + \
                        paddle.maximum(pred_right, target_right)
        g_h_intersect = paddle.maximum(pred_bottom, target_bottom) + \
                        paddle.maximum(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'iou':
            losses = -paddle.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()


def giou_loss(pred, target, weight, eps=1e-7, reduction='mean',
              avg_factor=None):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    This function is modified from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/iou_loss.py#L102

    Args:
        pred (paddle.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (paddle.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@manager.LOSSES.add_component
class GIoULoss(nn.Layer):
    """
    This class is modified from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/iou_loss.py#L358
    """

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """ as name """
        if weight is not None and not paddle.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze([1])
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


def get_rdiou(bboxes1, bboxes2):
    x1u, y1u, z1u = bboxes1[..., 0], bboxes1[..., 1], bboxes1[..., 2]
    l1, w1, h1 = paddle.clip(bboxes1[..., 3], max=5), paddle.clip(bboxes1[..., 4], max=5), paddle.clip(bboxes1[..., 5], max=5)
    l1, w1, h1 = paddle.exp(l1), paddle.exp(w1), paddle.exp(h1)
    t1 = paddle.sin(bboxes1[..., 6]) * paddle.cos(bboxes2[..., 6])
    x2u, y2u, z2u = bboxes2[..., 0], bboxes2[..., 1], bboxes2[..., 2]
    l2, w2, h2 = paddle.exp(bboxes2[..., 3]), paddle.exp(
        bboxes2[..., 4]), paddle.exp(bboxes2[..., 5])
    t2 = paddle.cos(bboxes1[..., 6]) * paddle.sin(bboxes2[..., 6])

    # we emperically scale the y/z to make their predictions more sensitive.
    x1 = x1u
    y1 = y1u
    z1 = z1u
    x2 = x2u
    y2 = y2u
    z2 = z2u

    # clamp is necessray to aviod inf.
    l1, w1, h1 = paddle.clip(l1, max=10), paddle.clip(
        w1, max=10), paddle.clip(h1, max=10)
    j1, j2 = paddle.ones_like(h2), paddle.ones_like(h2)

    volume_1 = l1 * w1 * h1 * j1
    volume_2 = l2 * w2 * h2 * j2

    inter_l = paddle.maximum(x1 - l1 / 2, x2 - l2 / 2)
    inter_r = paddle.minimum(x1 + l1 / 2, x2 + l2 / 2)
    inter_t = paddle.maximum(y1 - w1 / 2, y2 - w2 / 2)
    inter_b = paddle.minimum(y1 + w1 / 2, y2 + w2 / 2)
    inter_u = paddle.maximum(z1 - h1 / 2, z2 - h2 / 2)
    inter_d = paddle.minimum(z1 + h1 / 2, z2 + h2 / 2)
    inter_m = paddle.maximum(t1 - j1 / 2, t2 - j2 / 2)
    inter_n = paddle.minimum(t1 + j1 / 2, t2 + j2 / 2)

    inter_volume = paddle.clip((inter_r - inter_l), min=0) * paddle.clip((inter_b - inter_t), min=0) \
        * paddle.clip((inter_d - inter_u), min=0) * paddle.clip((inter_n - inter_m), min=0)

    c_l = paddle.minimum(x1 - l1 / 2, x2 - l2 / 2)
    c_r = paddle.maximum(x1 + l1 / 2, x2 + l2 / 2)
    c_t = paddle.minimum(y1 - w1 / 2, y2 - w2 / 2)
    c_b = paddle.maximum(y1 + w1 / 2, y2 + w2 / 2)
    c_u = paddle.minimum(z1 - h1 / 2, z2 - h2 / 2)
    c_d = paddle.maximum(z1 + h1 / 2, z2 + h2 / 2)
    c_m = paddle.minimum(t1 - j1 / 2, t2 - j2 / 2)
    c_n = paddle.maximum(t1 + j1 / 2, t2 + j2 / 2)

    inter_diag = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 + (t2 - t1)**2
    c_diag = paddle.clip((c_r - c_l), min=0)**2 + paddle.clip((c_b - c_t), min=0)**2 + \
        paddle.clip((c_d - c_u), min=0)**2 + \
        paddle.clip((c_n - c_m), min=0)**2

    union = volume_1 + volume_2 - inter_volume
    
    u = (inter_diag) / c_diag
    rdiou = inter_volume / union
    return u, rdiou

@manager.LOSSES.add_component
class RDIoULoss(nn.Layer):
    def __init__(
        self,
        loss_weight=1.0,
    ):
        super(RDIoULoss, self).__init__()
        self._loss_weight = loss_weight

    def forward(self, pred, target, weights=None):

        u, rdiou = get_rdiou(pred, target)
        rdiou_loss_n = rdiou - u
        rdiou_loss_n = paddle.clip(rdiou_loss_n, min=-1.0, max=1.0)
        rdiou_loss_m = 1 - rdiou_loss_n
        rdiou_loss_m = rdiou_loss_m.unsqueeze(-1)
        if weights is not None:
            rdiou_loss_m = rdiou_loss_m * weights.unsqueeze(-1)
        return rdiou_loss_m