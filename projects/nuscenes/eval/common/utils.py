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
# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

from typing import List, Dict, Any

import numpy as np
from pyquaternion import Quaternion

from nuscenes.eval.common.data_classes import EvalBox
from nuscenes.utils.data_classes import Box

# from mmdet3d.core import xywhr2xyxyr
# from mmdet3d.ops.iou3d import iou3d_cuda

from paddle3d.models.heads.roi_heads.target_assigner.iou3d_nms_utils import boxes_iou3d_gpu
import paddle

DetectionBox = Any  # Workaround as direct imports lead to cyclic dependencies.


def center_distance(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2]))

def rel_dist_distance(pred_box: np.array, gt_box: np.array) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    pred_box = np.swapaxes(pred_box, 0, 1)
    gt_box = np.swapaxes(gt_box, 0, 1)

    pred_box = pred_box[:, :, np.newaxis].repeat(gt_box.shape[1], axis=2)
    gt_box = gt_box[:, np.newaxis, :].repeat(pred_box.shape[1], axis=1)
    ab_dist = np.linalg.norm(pred_box[:2, ...]-gt_box[:2, ...], axis=0)
    rel_dist = ab_dist/np.linalg.norm(gt_box[:2, ...], axis=0)
     
    ab_dist = ab_dist[np.newaxis, ...] 
    rel_dist = rel_dist[np.newaxis, ...] 
    dist_metric = np.concatenate([ab_dist, rel_dist], axis=0)
    return dist_metric

def iou_calculate(pred_box: EvalBox, gt_box: EvalBox) -> float:
    """
    3D IoU between the box.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: 3D IoU.
    """
    gt_box_tensor = check_is_paddle_cuda(gt_box)
    pred_box_tensor = check_is_paddle_cuda(pred_box)

    # gt_box_xywhr   = gt_box_tensor[:, [0, 1, 3, 4, 6]]
    # pred_box_xywhr = pred_box_tensor[:, [0, 1, 3, 4, 6]]
 
    # gt_box_xyxyr   = xywhr2xyxyr(gt_box_xywhr)
    # pred_box_xyxyr = xywhr2xyxyr(pred_box_xywhr)

    # bev_iou = pred_box_xyxyr.new_zeros(paddle.Size((pred_box_xyxyr.shape[0], gt_box_xyxyr.shape[0])))
    # bev_overlap = pred_box_xyxyr.new_zeros(paddle.Size((pred_box_xyxyr.shape[0], gt_box_xyxyr.shape[0])))
    # iou3d_cuda.boxes_iou_bev_gpu(pred_box_xyxyr.contiguous(), gt_box_xyxyr.contiguous(), bev_iou)
    iou3d, ioubev = boxes_iou3d_gpu(pred_box_tensor, gt_box_tensor, return_bev=True)
    return iou3d, ioubev


def velocity_l2(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the velocity vectors (xy only).
    If the predicted velocities are nan, we return inf, which is subsequently clipped to 1.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.velocity) - np.array(gt_box.velocity))


def yaw_diff(yaw_gt: float, yaw_est: float, period: float = 2*np.pi) -> float:
    """
    Returns the yaw angle difference between the orientation of two boxes.
    :param gt_box: Ground truth box.
    :param eval_box: Predicted box.
    :param period: Periodicity in radians for assessing angle difference.
    :return: Yaw angle difference in radians in [0, pi].
    """
    # yaw_gt = quaternion_yaw(Quaternion(gt_box.rotation))
    # yaw_est = quaternion_yaw(Quaternion(eval_box.rotation))

    return abs(angle_diff(yaw_gt, yaw_est, period))


def angle_diff(x: float, y: float, period: float) -> float:
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


def attr_acc(gt_box: DetectionBox, pred_box: DetectionBox) -> float:
    """
    Computes the classification accuracy for the attribute of this class (if any).
    If the GT class has no attributes or the annotation is missing attributes, we assign an accuracy of nan, which is
    ignored later on.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Attribute classification accuracy (0 or 1) or nan if GT annotation does not have any attributes.
    """
    if gt_box.attribute_name == '':
        # If the class does not have attributes or this particular sample is missing attributes, return nan, which is
        # ignored later. Note that about 0.4% of the sample_annotations have no attributes, although they should.
        acc = np.nan
    else:
        # Check that label is correct.
        acc = float(gt_box.attribute_name == pred_box.attribute_name)
    return acc


def scale_iou(sample_annotation: EvalBox, sample_result: EvalBox) -> float:
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    sa_size = sample_annotation[3:6]
    sr_size = sample_result[3:6]
    assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
    assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

    # Compute IOU.
    min_wlh = np.minimum(sa_size, sr_size)
    volume_annotation = np.prod(sa_size)
    volume_result = np.prod(sr_size)
    intersection = np.prod(min_wlh)  # type: float
    union = volume_annotation + volume_result - intersection  # type: float
    iou = intersection / union

    return iou


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def boxes_to_sensor(boxes: List[EvalBox], pose_record: Dict, cs_record: Dict):
    """
    Map boxes from global coordinates to the vehicle's sensor coordinate system.
    :param boxes: The boxes in global coordinates.
    :param pose_record: The pose record of the vehicle at the current timestamp.
    :param cs_record: The calibrated sensor record of the sensor.
    :return: The transformed boxes.
    """
    boxes_out = []
    for box in boxes:
        # Create Box instance.
        box = Box(box.translation, box.size, Quaternion(box.rotation))

        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes_out.append(box)

    return boxes_out


def cummean(x: np.array) -> np.array:
    """
    Computes the cumulative mean up to each position in a NaN sensitive way
    - If all values are NaN return an array of ones.
    - If some values are NaN, accumulate arrays discording those entries.
    """
    if sum(np.isnan(x)) == len(x):
        # Is all numbers in array are NaN's.
        return np.ones(len(x))  # If all errors are NaN set to error to 1 for all operating points.
    else:
        # Accumulate in a nan-aware manner.
        sum_vals = np.nancumsum(x.astype(float))  # Cumulative sum ignoring nans.
        count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
        return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)


def check_is_paddle_cuda(box):
    box_tensor = paddle.to_tensor(box).cast('float32')
    return box_tensor
    
    