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
import math
from paddle3d.ops import iou3d_nms_cuda, iou3d_idg

def limit_period(val, offset=0.5, period=math.pi):
    """limit_period
    """
    return val - paddle.floor(val / period + offset) * period

@paddle.jit.not_to_static
def boxes3d_to_near_torch(boxes3d):
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        boxes_near: [N, 4(xmin, ymin, xmax, ymax)] nearest boxes
    """
    rboxes = boxes3d.index_select(paddle.to_tensor([0, 1, 3, 4, 6]), axis= 1) 
    # rboxes = paddle.concat([boxes3d[:, 0:2], boxes3d[:, 3:5], boxes3d[:, 6:7]], axis=1)
    rots = rboxes[..., -1]
    rots_0_pi_div_2 = paddle.abs(limit_period(rots, 0.5, math.pi))
    cond = (rots_0_pi_div_2 > math.pi / 4)[..., None]
    rboxes_ = paddle.concat([rboxes[:, 0:2], rboxes[:, 3:4], rboxes[:, 2:3]], axis=1)
    boxes_center = paddle.where(cond, rboxes_, rboxes[:, :4])
    boxes_near = paddle.concat([boxes_center[:, :2] - boxes_center[:, 2:] / 2, 
                                boxes_center[:, :2] + boxes_center[:, 2:] / 2], axis=-1)
    return boxes_near

def boxes_iou(bboxes1, bboxes2, mode='iou', eps=0.0):
    """limit_period
    """
    assert mode in ['iou', 'iof']

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0] 

    if rows * cols == 0:
        return bboxes1.new(rows, cols)

    lt = paddle.maximum(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
    rb = paddle.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]
    wh = (rb - lt + eps).clip(min = 0) #clamp(min=0)  # [rows, cols, 2]
    overlap = wh[:, :, 0] * wh[:, :, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + eps) * (
        bboxes1[:, 3] - bboxes1[:, 1] + eps)
    if mode == 'iou':
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + eps) * (
            bboxes2[:, 3] - bboxes2[:, 1] + eps)
        ious = overlap / (area1[:, None] + area2 - overlap)
    else:
        ious = overlap / (area1[:, None])
    return ious

def boxes_distance(points,
                    qpoints,
                    dist_norm,
                    with_rotation=False,
                    rot_alpha=0.5):
    """boxes_distance
    """
    N = points.shape[0] 
    K = qpoints.shape[0] 
    points = points[:, None, :] # N * 1 * 2 
    rot_alpha_1 = 1 - rot_alpha
    dists = paddle.abs(points[:, :, :2] - qpoints[:, :2])  # N * K * 2
    dists = dists.reshape((-1, 2))
    dists_mask = dists > dist_norm
    dists_mask = dists_mask.sum(axis=1) # (N*K)
    invalid_dists_mask = paddle.nonzero(dists_mask)[:, 0]
    valid_dists_mask = paddle.where(dists_mask == 0)[0].squeeze(-1)
    if valid_dists_mask.shape[0] != 0:
        valid_dists = dists.index_select(valid_dists_mask, axis = 0) #dists[valid_dists_mask]
        valid_dists = paddle.pow(valid_dists, 2).sum(1) # N'* 2
        valid_dists = paddle.clip(valid_dists / dist_norm, max=dist_norm)
    else:
        valid_dists = paddle.empty((0, 3, 2))
    if with_rotation:
        pass
    else:
        valid_dists = 1 - valid_dists

    dists = dists.sum(axis=1) #(N * K)
    # TODO yipin: change to paddle where to avoid a[mask], cause of speed problem

    if valid_dists_mask.shape[0] != 0:
        dists = dists.put_along_axis(indices = valid_dists_mask, values = valid_dists, axis = 0)
    if invalid_dists_mask.shape[0] != 0:
        dists = dists.put_along_axis(indices = invalid_dists_mask, values = 0.0, axis = 0)

    dists = dists.reshape((N, K))

    return dists 

def boxes3d_to_bev_torch(boxes3d):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = paddle.empty((boxes3d.shape[0], 5), dtype = boxes3d.dtype) 

    cu, cv = boxes3d[:, 0], boxes3d[:, 1]
    half_l, half_w = boxes3d[:, 3] / 2, boxes3d[:, 4] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev

def boxes_iou_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 7)
    :param boxes_b: (N, 7)
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = boxes3d_to_bev_torch(boxes_b)

    ans_iou = iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a_bev, boxes_b_bev) 
    #  custom op already implemented in paddle3d

    return ans_iou

def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = boxes3d_to_bev_torch(boxes_b)

    # bev overlap
    if boxes_a_bev.shape[-1] == 7:
        overlaps_bev = iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a_bev, boxes_b_bev)
    elif boxes_a_bev.shape[-1] == 5:
        overlaps_bev = iou3d_nms_cuda.boxes_overlap_bev_v2_gpu(boxes_a_bev, boxes_b_bev)
    else:
        raise NotImplementedError

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5]).reshape((-1, 1)) 
    boxes_a_height_min = boxes_a[:, 2].reshape((-1, 1)) 
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5]).reshape((-1, 1)) 
    boxes_b_height_min = boxes_b[:, 2].reshape((1, -1)) 

    max_of_min = paddle.maximum(boxes_a_height_min, boxes_b_height_min) 
    min_of_max = paddle.minimum(boxes_a_height_max, boxes_b_height_max) 
    overlaps_h = paddle.clip(min_of_max - max_of_min, min=0) 

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).reshape((-1, 1))
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).reshape((1, -1))

    iou3d = overlaps_3d / paddle.clip(vol_a + vol_b - overlaps_3d, min=1e-7) 

    return iou3d

def boxes_iou_bev_align(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7)
    :param boxes_b: (N, 7)
    :return:
        ans_iou: (N)
    """
    boxes_a_bev = boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = boxes3d_to_bev_torch(boxes_b)
    area_a = (boxes_a[:, 3] * boxes_a[:, 4]).reshape((-1))
    area_b = (boxes_b[:, 3] * boxes_b[:, 4]).reshape((-1))

    overlaps_bev = paddle.zeros((boxes_a_bev.shape[0], boxes_b_bev.shape[0])) 
    iou3d_nms_cuda.boxes_overlap_bev_align_gpu(boxes_a_bev, boxes_b_bev, overlaps_bev)

    bev_iou = overlaps_bev / paddle.clip(area_a + area_b - overlaps_bev, min=1e-7) 

    return bev_iou

# def nms_normal_gpu(boxes, scores, thresh):
#     """
#     :param boxes: (N, 5) [x1, y1, x2, y2, ry]
#     :param scores: (N)
#     :param thresh:
#     :return:
#     """
#     # areas = (x2 - x1) * (y2 - y1)
#     order = scores.argsort(axis=0, descending=True)

#     boxes = boxes[order] 

#     keep = paddle.zeros(boxes.shape[0], dtype = 'int64')
#     num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
#     return order[keep[:num_out]]

def nms_normal_gpu_nosort(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    order = paddle.arange(0, scores.shape[0], step = 1)
    keep, num_out = iou3d_idg.nms_normal_gpu(boxes, thresh)
    return order[keep[:num_out]]

def anchors_match_valid_voxels(anchors, voxel_mask):
    """
    :param anchors: (N, 4) [x1, y1, x2, y2] int
    :param voxel_mask: (W * L) bool
    :return: anchors_mask
    """
    anchors_mask = iou3d_idg.anchors_mask_of_valid_voxels(anchors, voxel_mask) 
    ## TODO custom op not implemented, add it into iou3d_nms_cuda
    return anchors_mask 

def boxes_to_parsing(boxes, grid_size):
    """
    :param corners: (N, 4, 2) in voxel coords type
    :param parsing_map: (W * L) float
    :return: parsing_map
    """
    parsing_map = iou3d_idg.boxes_to_parsing(boxes, grid_size) ## TODO custom op not implemented 
    return parsing_map

def parsing_to_boxes_confidence(boxes, confidence_map):
    """
    :param corners: (N, 4, 2) in voxel coords type
    :param confidence_map: (W * L) float
    :return: confidences
    """
    confidences = iou3d_idg.parsing_to_boxes_confidence(boxes, confidence_map)
    return confidences


class RotateIou2dSimilarity(object):
    """Class to compute similarity based on Intersection over Union (IOU) metric.

    This class computes pairwise similarity between two BoxLists based on IOU.
    """
    def __call__(self, boxes1, boxes2, is_aligned=False):
        if is_aligned:
            assert boxes1.shape == boxes2.shape
            return boxes_iou_bev_align(boxes1, boxes2)
        else:
            return boxes_iou_bev(boxes1, boxes2)

class RotateIou3dSimilarity(object):
    """Class to compute similarity based on Intersection over Union (IOU) metric.

    This class computes pairwise similarity between two BoxLists based on IOU.
    """
    def __call__(self, boxes1, boxes2):
        return boxes_iou3d_gpu(boxes1, boxes2)


class NearestIouSimilarity(object):
    """Class to compute similarity based on the squared distance metric.

    This class computes pairwise similarity between two BoxLists based on the
    negative squared distance metric.
    """

    def __call__(self, boxes1, boxes2):
        """Compute matrix of (negated) sq distances.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.

        Returns:
          A tensor with shape [N, M] representing negated pairwise squared distance.
        """

        boxes1_near = boxes3d_to_near_torch(boxes1)
        boxes2_near = boxes3d_to_near_torch(boxes2)
        return boxes_iou(boxes1_near, boxes2_near)

class DistanceSimilarity(object):
    """Class to compute similarity based on the squared distance metric.

    This class computes pairwise similarity between two BoxLists based on the
    negative squared distance metric.
    """
    def __init__(self, distance_norm, with_rotation=False, rotation_alpha=0.5):
        self._distance_norm = distance_norm
        self._with_rotation = with_rotation
        self._rotation_alpha = rotation_alpha

    def __call__(self, boxes1, boxes2):
        """Compute matrix of (negated) sq distances.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.

        Returns:
          A tensor with shape [N, M] representing negated pairwise squared distance.
        """
        return boxes_distance(
            boxes1.index_select(paddle.to_tensor([0, 1, boxes1.shape[1] - 1]), axis = 1),
            boxes2.index_select(paddle.to_tensor([0, 1, boxes2.shape[1] - 1]), axis = 1),
            dist_norm=self._distance_norm,
            with_rotation=self._with_rotation,
            rot_alpha=self._rotation_alpha)

def boxes_iou_bev_v2(boxes_a, boxes_b, mode='iou'):
    """
    :param boxes_a: (M, 7)
    :param boxes_b: (N, 7)
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = boxes3d_to_bev_torch(boxes_b)

    # bev overlap
    if boxes_a_bev.shape[-1] == 7:
        overlaps_bev = iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a_bev, boxes_b_bev)
    elif boxes_a_bev.shape[-1] == 5:
        overlaps_bev = iou3d_nms_cuda.boxes_overlap_bev_v2_gpu(boxes_a_bev, boxes_b_bev)
    else:
        raise NotImplementedError

    area_a = (boxes_a[:, 3] * boxes_a[:, 4]).reshape([-1, 1])
    area_b = (boxes_b[:, 3] * boxes_b[:, 4]).reshape([1, -1])
    if mode == 'iou':
        iou3d = overlaps_bev / paddle.clip(area_a + area_b - overlaps_bev, min=1e-7)
    elif mode == 'iof':
        iou3d = overlaps_bev / paddle.clip(area_a, min=1e-7)
    else:
        raise NotImplementedError

    return iou3d

def boxes_iom_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 7)
    :param boxes_b: (N, 7)
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = boxes3d_to_bev_torch(boxes_b)

    if boxes_a_bev.shape[-1] == 7:
        overlaps_bev = iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a_bev, boxes_b_bev)
    elif boxes_a_bev.shape[-1] == 5:
        overlaps_bev = iou3d_nms_cuda.boxes_overlap_bev_v2_gpu(boxes_a_bev, boxes_b_bev)
    else:
        raise NotImplementedError

    area_a = (boxes_a[:, 3] * boxes_a[:, 4]).reshape([-1, 1]).repeat_interleave(boxes_b.shape[0], 1)
    area_b = (boxes_b[:, 3] * boxes_b[:, 4]).reshape([1, -1]).repeat_interleave(boxes_a.shape[0], 0)

    # area_a[area_a > area_b] = area_b[area_a > area_b]
    area_a = paddle.where(area_a > area_b, area_b, area_a)

    iom_bev = overlaps_bev / paddle.clip(area_a, min=1e-7)

    return iom_bev

if __name__ == '__main__':
    pass