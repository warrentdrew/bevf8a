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

from paddle3d.ops.iou3d_nms_cuda import nms_gpu

def horizontal(bboxes):
    """ as name """
    bboxes[:, 1::7] = -bboxes[:, 1::7]
    bboxes[:, 6] = -bboxes[:, 6] + np.pi
    return bboxes

def vertical(bboxes):
    """ as name """
    bboxes[:, 0::7] = -bboxes[:, 0::7]
    bboxes[:, 6] = -bboxes[:, 6]
    return bboxes

def bbox3d_mapping_back(bboxes, scale_factor, flip_horizontal, flip_vertical):
    """Map bboxes from testing scale to original image scale.

    Args:
        bboxes (:obj:`BaseInstance3DBoxes`): Boxes to be mapped back.
        scale_factor (float): Scale factor.
        flip_horizontal (bool): Whether to flip horizontally.
        flip_vertical (bool): Whether to flip vertically.

    Returns:
        :obj:`BaseInstance3DBoxes`: Boxes mapped back.
    """
    new_bboxes = bboxes.clone()
    if flip_horizontal:
        new_bboxes = horizontal(new_bboxes)
    if flip_vertical:
        new_bboxes = vertical(new_bboxes)
    new_bboxes.scale(1 / scale_factor)

    return new_bboxes


def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (paddle.Tensor): Rotated boxes in XYWHR format.

    Returns:
        paddle.Tensor: Converted boxes in XYXYR format.
    """
    boxes = paddle.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes

def bbox3d2result(bboxes, scores, labels):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (paddle.Tensor): Bounding boxes with shape of (n, 5).
        labels (paddle.Tensor): Labels with shape of (n, ).
        scores (paddle.Tensor): Scores with shape of (n, ).

    Returns:
        dict[str, paddle.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (paddle.Tensor): 3D boxes.
            - scores (paddle.Tensor): Prediction scores.
            - labels_3d (paddle.Tensor): Box labels.
    """
    return dict(
        boxes_3d=bboxes.numpy(),
        scores_3d=scores.numpy(),
        labels_3d=labels.numpy())

def merge_aug_bboxes_3d(aug_results, img_metas, test_cfg):
    """Merge augmented detection 3D bboxes and scores.

    Args:
        aug_results (list[dict]): The dict of detection results.
            The dict contains the following keys

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (paddle.Tensor): Detection scores.
            - labels_3d (paddle.Tensor): Predicted box labels.
        img_metas (list[dict]): Meta information of each sample.
        test_cfg (dict): Test config.

    Returns:
        dict: Bounding boxes results in cpu mode, containing merged results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Merged detection bbox.
            - scores_3d (paddle.Tensor): Merged detection scores.
            - labels_3d (paddle.Tensor): Merged predicted box labels.
    """

    assert len(aug_results) == len(img_metas), \
        '"aug_results" should have the same length as "img_metas", got len(' \
        f'aug_results)={len(aug_results)} and len(img_metas)={len(img_metas)}'

    recovered_bboxes = []
    recovered_scores = []
    recovered_labels = []

    for bboxes, img_info in zip(aug_results, img_metas):
        scale_factor = img_info[0]['pcd_scale_factor']
        pcd_horizontal_flip = img_info[0]['pcd_horizontal_flip']
        pcd_vertical_flip = img_info[0]['pcd_vertical_flip']
        recovered_scores.append(bboxes['scores_3d'])
        recovered_labels.append(bboxes['labels_3d'])
        bboxes = bbox3d_mapping_back(bboxes['boxes_3d'], scale_factor,
                                     pcd_horizontal_flip, pcd_vertical_flip)
        recovered_bboxes.append(bboxes)

    aug_bboxes = recovered_bboxes[0].cat(recovered_bboxes)
    aug_bboxes_for_nms = xywhr2xyxyr(aug_bboxes.bev)
    aug_scores = paddle.concat(recovered_scores, axis=0)
    aug_labels = paddle.concat(recovered_labels, axis=0)

    # TODO: use a more elegent way to deal with nms
    if test_cfg.use_rotate_nms:
        nms_func = nms_gpu
    else:
        raise TypeError('not support other nms')

    merged_bboxes = []
    merged_scores = []
    merged_labels = []

    # Apply multi-class nms when merge bboxes
    if len(aug_labels) == 0:
        return bbox3d2result(aug_bboxes, aug_scores, aug_labels)

    for class_id in range(paddle.max(aug_labels).item() + 1):
        class_inds = (aug_labels == class_id)
        bboxes_i = aug_bboxes[class_inds]
        bboxes_nms_i = aug_bboxes_for_nms[class_inds, :]
        scores_i = aug_scores[class_inds]
        labels_i = aug_labels[class_inds]
        if len(bboxes_nms_i) == 0:
            continue
        selected = nms_func(bboxes_nms_i, scores_i, test_cfg.nms_thr)

        merged_bboxes.append(bboxes_i[selected, :])
        merged_scores.append(scores_i[selected])
        merged_labels.append(labels_i[selected])

    merged_bboxes = merged_bboxes[0].cat(merged_bboxes)
    merged_scores = paddle.concat(merged_scores, axis=0)
    merged_labels = paddle.concat(merged_labels, axis=0)

    _, order = merged_scores.sort(0, descending=True)
    num = min(test_cfg.max_num, len(aug_bboxes))
    order = order[:num]

    merged_bboxes = merged_bboxes[order]
    merged_scores = merged_scores[order]
    merged_labels = merged_labels[order]

    return bbox3d2result(merged_bboxes, merged_scores, merged_labels)
