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
"""Geometry transforms for box implemented by paddle."""

import numpy as np

import paddle
from paddle3d.utils_idg.ops.nms_gpu import (nms_overlap_gpu, rotate_nms_overlap_gpu, nms_gpu)
import paddle3d.utils_idg.box_np_ops as box_np_ops

def check_numpy_to_paddle(x):
    if isinstance(x, np.ndarray):
        return paddle.to_tensor(x).cast('float32'), True
    if isinstance(x, np.float64) or isinstance(x, np.float32):
        return paddle.to_tensor([x]).cast('float32'), True
    return x, False


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin)
    corners_norm = paddle.to_tensor(corners_norm, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2 ** ndim, ndim])
    return corners


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = paddle.sin(angles)
    rot_cos = paddle.cos(angles)
    rot_mat_T = paddle.stack(
        [paddle.stack([rot_cos, -rot_sin]), paddle.stack([rot_sin, rot_cos])])
    # print("points shape: ", points.shape)
    # print('rot_mat_T shape: ', rot_mat_T.shape)
    return paddle.einsum("aij,jka->aik", points, rot_mat_T)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
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
    """Convert box corner to the box represente by the minimun and maximum point."""
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(paddle.min(boxes_corner[:, :, i], axis=1))
    for i in range(ndim):
        standup_boxes.append(paddle.max(boxes_corner[:, :, i], axis=1))
    return paddle.stack(standup_boxes, axis=1)


def second_box_encode(boxes,
                      anchors,
                      encode_angle_to_vector=False,
                      smooth_dim=False,
                      norm_velo=False):
    """box encode for VoxelNet
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, l, w, h, r
        anchors ([N, 7] Tensor): anchors
    """
    box_ndim = anchors.shape[-1]

    if box_ndim == 7:
        xa, ya, za, wa, la, ha, ra = paddle.split(anchors, 7, axis=-1)
        xg, yg, zg, wg, lg, hg, rg = paddle.split(boxes, 7, axis=-1)
    else:
        xa, ya, za, wa, la, ha, vxa, vya, ra = paddle.split(
            anchors, 7, axis=-1)
        xg, yg, zg, wg, lg, hg, vxg, vyg, rg = paddle.split(boxes, 7, axis=-1)

    diagonal = paddle.sqrt(la ** 2 + wa ** 2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha

    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = paddle.log(lg / la)
        wt = paddle.log(wg / wa)
        ht = paddle.log(hg / ha)

    ret = [xt, yt, zt, wt, lt, ht]

    if box_ndim > 7:
        if norm_velo:
            vxt = (vxg - vxa) / diagonal
            vyt = (vyg - vya) / diagonal
        else:
            vxt = vxg - vxa
            vyt = vyg - vya
        ret.extend([vxt, vyt])

    if encode_angle_to_vector:
        rgx = paddle.cos(rg)
        rgy = paddle.sin(rg)
        rax = paddle.cos(ra)
        ray = paddle.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        ret.extend([rtx, rty])
    else:
        rt = rg - ra
        ret.append(rt)

    return paddle.concat(ret, axis=-1)


def second_box_decode(
        box_encodings,
        anchors,
        encode_angle_to_vector=False,
        bin_loss=False,
        smooth_dim=False,
        norm_velo=False, ):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    box_ndim = anchors.shape[-1]

    if box_ndim == 9:
        xa, ya, za, wa, la, ha, vxa, vya, ra = paddle.split(
            anchors, 9, axis=-1)
        if encode_angle_to_vector:
            xt, yt, zt, wt, lt, ht, vxt, vyt, rtx, rty = paddle.split(
                box_encodings, 9, axis=-1)
        else:
            xt, yt, zt, wt, lt, ht, vxt, vyt, rt = paddle.split(
                box_encodings, 9, axis=-1)
    elif box_ndim == 7:
        xa, ya, za, wa, la, ha, ra = paddle.split(anchors, 7, axis=-1)
        if encode_angle_to_vector:
            xt, yt, zt, wt, lt, ht, rtx, rty = paddle.split(
                box_encodings, 7, axis=-1)
        else:
            xt, yt, zt, wt, lt, ht, rt = paddle.split(
                box_encodings, 7, axis=-1)

    diagonal = paddle.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    ret = [xg, yg, zg]

    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:

        lg = paddle.exp(lt) * la
        wg = paddle.exp(wt) * wa
        hg = paddle.exp(ht) * ha
    ret.extend([wg, lg, hg])

    if encode_angle_to_vector:
        rax = paddle.cos(ra)
        ray = paddle.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = paddle.atan2(rgy, rgx)
    else:
        rg = rt + ra

    if box_ndim > 7:
        if norm_velo:
            vxg = vxt * diagonal + vxa
            vyg = vyt * diagonal + vya
        else:
            vxg = vxt + vxa
            vyg = vyt + vya
        ret.extend([vxg, vyg])

    ret.append(rg)
    return paddle.concat(ret, axis=-1)


def nms(bboxes, scores, pre_max_size=None, post_max_size=None, iou_threshold=0.5, add_iou_edge=1):
    """ as name """
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = paddle.topk(scores, k=pre_max_size)
        bboxes = bboxes.index_select(indices)#bboxes[indices]
        # bboxes = bboxes.reshape([-1, box_ndim])
    dets = paddle.concat([bboxes, scores.unsqueeze(-1)], axis=1)
    dets_np = dets.numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(nms_gpu(dets_np, iou_threshold, add_iou_edge), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return paddle.zeros([0], dtype='int64')
    if pre_max_size is not None:
        keep = paddle.to_tensor(keep, dtype='int64')
        return indices[keep]
    else:
        return paddle.to_tensor(keep, dtype='int64')



def nms_overlap(bboxes,
                scores,
                pre_max_size=None,
                post_max_size=None,
                overlap_threshold=0.5):
    """Non-maximum suppression."""
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = paddle.topk(scores, k=pre_max_size)
        bboxes = bboxes.index_select(indices) # bboxes[indices]
        # TODO(luoqianhui): when indices.shape is (1,),
        # bboxes[indices].shape is (box_ndim,) but supposed to be (1, box_ndim),
        # so we add a reshape op
        #bboxes = bboxes.reshape([-1, box_ndim])
    dets = paddle.concat([bboxes, scores.unsqueeze(-1)], axis=1)
    dets_np = dets.numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(nms_overlap_gpu(dets_np, overlap_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return paddle.zeros([0], dtype='int64')
    if pre_max_size is not None:
        keep = paddle.to_tensor(keep)
        return indices[keep]
    else:
        return paddle.to_tensor(keep)


def rotate_nms_overlap(bboxes,
                       scores,
                       pre_max_size=None,
                       post_max_size=None,
                       overlap_threshold=0.5):
    """Non-maximum suppression for rotate box."""
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = paddle.topk(scores, k=pre_max_size)
        ndim = bboxes.shape[-1]
        bboxes = bboxes[indices]
        # TODO(luoqianhui): when indices.reshape is (1,),
        # bboxes[indices].shape is (5,) but supposed to be (1, 5),
        # so we add a reshape op
        bboxes = bboxes.reshape([-1, ndim])

    dets = paddle.concat([bboxes, scores.unsqueeze(-1)], axis=1)
    dets_np = dets.numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(rotate_nms_overlap_gpu(dets_np, overlap_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return paddle.zeros([0], dtype='int64')
    if pre_max_size is not None:
        keep = paddle.to_tensor(keep, dtype='int64')
        return indices[keep]
    else:
        return paddle.to_tensor(keep)



# ==========
# 8A
def enlarge_box3d_by_ratio(boxes3d, extra_ratio=0.5):
    """
    Args:
        boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        extra_width: [extra_x, extra_y, extra_z]

    Returns:

    """
    boxes3d, is_numpy = check_numpy_to_paddle(boxes3d)
    large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += boxes3d[:, 3:6] * extra_ratio
    return large_boxes3d



# ==========
# 8A
def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    """
    Args:
        boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        extra_width: [extra_x, extra_y, extra_z]

    Returns:

    """
    boxes3d, is_numpy = check_numpy_to_paddle(boxes3d)
    large_boxes3d = boxes3d.clone()
    if isinstance(extra_width, paddle.Tensor):
        large_boxes3d[:, 3:6] += extra_width
    else:
        large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
    return large_boxes3d