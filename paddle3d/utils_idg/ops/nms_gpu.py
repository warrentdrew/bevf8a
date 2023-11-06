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
import numba
from numba import cuda 
import numpy as np
import math

@cuda.jit("(float32[:], float32[:], float32)", device=True, inline=True)
def iou_device(a, b, add_iou_edge):
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    top = max(a[1], b[1])
    bottom = min(a[3], b[3])
    width = max(right - left + add_iou_edge, 0.0)
    height = max(bottom - top + add_iou_edge, 0.0)
    interS = width * height
    Sa = (a[2] - a[0] + add_iou_edge) * (a[3] - a[1] + add_iou_edge)
    Sb = (b[2] - b[0] + add_iou_edge) * (b[3] - b[1] + add_iou_edge)
    return interS / (Sa + Sb - interS)


@cuda.jit("(int64, float32, float32, float32[:], uint64[:])")
def nms_kernel(n_boxes, nms_overlap_thresh, add_iou_edge, dev_boxes, dev_mask):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)
    dev_box_idx = threadsPerBlock * col_start + tx
    if tx < col_size:
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        cur_box_idx = threadsPerBlock * row_start + tx
        # cur_box = dev_boxes + cur_box_idx * 5;
        t = 0
        start = 0
        if row_start == col_start:
            start = tx + 1
        for i in range(start, col_size):
            iou = iou_device(
                dev_boxes[cur_box_idx * 5 : cur_box_idx * 5 + 4],
                block_boxes[i * 5 : i * 5 + 4],
                add_iou_edge
            )
            if iou > nms_overlap_thresh:
                t |= 1 << i
        col_blocks = (n_boxes) // (threadsPerBlock) + (
            (n_boxes) % (threadsPerBlock) > 0
        )
        dev_mask[cur_box_idx * col_blocks + col_start] = t

@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)

@numba.jit(nopython=True)
def nms_postprocess(keep_out, mask_host, boxes_num):
    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    remv = np.zeros((col_blocks), dtype=np.uint64)
    num_to_keep = 0
    for i in range(boxes_num):
        nblock = i // threadsPerBlock
        inblock = i % threadsPerBlock
        mask = np.array(1 << inblock, dtype=np.uint64)
        if not (remv[nblock] & mask):
            keep_out[num_to_keep] = i
            num_to_keep += 1
            # unsigned long long *p = &mask_host[0] + i * col_blocks;
            for j in range(nblock, col_blocks):
                remv[j] |= mask_host[i * col_blocks + j]
                # remv[j] |= p[j];
    return num_to_keep

def nms_gpu(dets, nms_overlap_thresh, add_iou_edge=1, device_id=0):
    """nms in gpu.
    Args:
        dets ([type]): [description]
        nms_overlap_thresh ([type]): [description]
        device_id ([type], optional): Defaults to 0. [description]

    Returns:
        [type]: [description]
    """
    boxes_num = dets.shape[0]
    keep_out = np.zeros([boxes_num], dtype=np.int32)
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)
    boxes_host = dets[order, :]

    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    cuda.select_device(device_id)
    mask_host = np.zeros((boxes_num * col_blocks,), dtype=np.uint64)
    blockspergrid = (
        div_up(boxes_num, threadsPerBlock),
        div_up(boxes_num, threadsPerBlock),
    )
    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes_host.reshape([-1]), stream)
        mask_dev = cuda.to_device(mask_host, stream)
        nms_kernel[blockspergrid, threadsPerBlock, stream](
            boxes_num, nms_overlap_thresh, add_iou_edge, boxes_dev, mask_dev
        )
        mask_dev.copy_to_host(mask_host, stream=stream)
    # stream.synchronize()
    num_out = nms_postprocess(keep_out, mask_host, boxes_num)
    keep = keep_out[:num_out]
    return list(order[keep])


def nms(bboxes, scores, pre_max_size=None, post_max_size=None, iou_threshold=0.5, add_iou_edge=1):
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = paddle.topk(scores, k=pre_max_size) 
        bboxes = bboxes[indices]
    dets = paddle.concate([bboxes, scores.unsqueeze(-1)], axis=1) 
    dets_np = dets.numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(nms_gpu(dets_np, iou_threshold, add_iou_edge, bboxes.device.index), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return paddle.zeros([0]).cast("int64") 
    if pre_max_size is not None:
        keep = paddle.to_tensor(keep).cast("int64")
        return indices[keep]
    else:
        return paddle.to_tensor(keep).cast("int64") 


@cuda.jit("(float32[:], float32[:])", device=True, inline=True)
def overlap_device(a, b):
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    top = max(a[1], b[1])
    bottom = min(a[3], b[3])
    width = max(right - left + 1, 0.0)
    height = max(bottom - top + 1, 0.0)
    interS = width * height
    Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return interS / min(Sa, Sb)


@cuda.jit("(int64, float32, float32[:], uint64[:])")
def nms_overlap_kernel(n_boxes, overlap_thresh, dev_boxes, dev_mask):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)
    dev_box_idx = threadsPerBlock * col_start + tx
    if tx < col_size:
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        cur_box_idx = threadsPerBlock * row_start + tx
        # cur_box = dev_boxes + cur_box_idx * 5;
        t = 0
        start = 0
        if row_start == col_start:
            start = tx + 1
        for i in range(start, col_size):
            iou = overlap_device(
                dev_boxes[cur_box_idx * 5 : cur_box_idx * 5 + 4],
                block_boxes[i * 5 : i * 5 + 4],
            )
            if iou > overlap_thresh:
                t |= 1 << i
        col_blocks = (n_boxes) // (threadsPerBlock) + (
            (n_boxes) % (threadsPerBlock) > 0
        )
        dev_mask[cur_box_idx * col_blocks + col_start] = t


def nms_overlap_gpu(dets, overlap_thresh, device_id=0):
    """nms in gpu.

    Args:
        dets ([type]): [description]
        nms_overlap_thresh ([type]): [description]
        device_id ([type], optional): Defaults to 0. [description]

    Returns:
        [type]: [description]
    """

    boxes_num = dets.shape[0]
    keep_out = np.zeros([boxes_num], dtype=np.int32)
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)
    boxes_host = dets[order, :]

    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    cuda.select_device(device_id)
    mask_host = np.zeros((boxes_num * col_blocks,), dtype=np.uint64)
    blockspergrid = (
        div_up(boxes_num, threadsPerBlock),
        div_up(boxes_num, threadsPerBlock),
    )
    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes_host.reshape([-1]), stream)
        mask_dev = cuda.to_device(mask_host, stream)
        nms_overlap_kernel[blockspergrid, threadsPerBlock, stream](
            boxes_num, overlap_thresh, boxes_dev, mask_dev
        )
        mask_dev.copy_to_host(mask_host, stream=stream)
    # stream.synchronize()
    num_out = nms_postprocess(keep_out, mask_host, boxes_num)
    keep = keep_out[:num_out]
    return list(order[keep])


def nms_overlap(bboxes, scores, pre_max_size=None, post_max_size=None, overlap_threshold=0.5):
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = paddle.topk(scores, k=pre_max_size)
        bboxes = bboxes[indices]
    dets = paddle.concat([bboxes, scores.unsqueeze(-1)], axis=1)
    dets_np = dets.numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(nms_overlap_gpu(dets_np, overlap_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return paddle.zeros([0]).cast("int64") #.long().to(bboxes.device)
    if pre_max_size is not None:
        keep = paddle.to_tensor(keep).cast("int64") #.long().to(bboxes.device)
        return indices[keep]
    else:
        return paddle.to_tensor(keep).cast("int64") #.long().to(bboxes.device)



@cuda.jit("(float32[:], float32[:], float32[:])", device=True, inline=True)
def trangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0


@cuda.jit("(float32[:], int32)", device=True, inline=True)
def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(
                int_pts[:2],
                int_pts[2 * i + 2 : 2 * i + 4],
                int_pts[2 * i + 4 : 2 * i + 6],
            )
        )
    return area_val

@cuda.jit("(float32[:], int32)", device=True, inline=True)
def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = cuda.local.array((2,), dtype=numba.float32)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = cuda.local.array((2,), dtype=numba.float32)
        vs = cuda.local.array((16,), dtype=numba.float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty



@cuda.jit(
    "(float32[:], float32[:], int32, int32, float32[:])", device=True, inline=True
)
def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = cuda.local.array((2,), dtype=numba.float32)
    B = cuda.local.array((2,), dtype=numba.float32)
    C = cuda.local.array((2,), dtype=numba.float32)
    D = cuda.local.array((2,), dtype=numba.float32)

    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


@cuda.jit("(float32, float32, float32[:])", device=True, inline=True)
def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


@cuda.jit("(float32[:], float32[:], float32[:])", device=True, inline=True)
def quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = cuda.local.array((2,), dtype=numba.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


@cuda.jit("(float32[:], float32[:])", device=True, inline=True)
def rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = cuda.local.array((4,), dtype=numba.float32)
    corners_y = cuda.local.array((4,), dtype=numba.float32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y

@cuda.jit("(float32[:], float32[:])", device=True, inline=True)
def inter(rbbox1, rbbox2):
    corners1 = cuda.local.array((8,), dtype=numba.float32)
    corners2 = cuda.local.array((8,), dtype=numba.float32)
    intersection_corners = cuda.local.array((16,), dtype=numba.float32)

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(
        corners1, corners2, intersection_corners
    )
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    # print(intersection_corners.reshape([-1, 2])[:num_intersection])

    return area(intersection_corners, num_intersection)

@cuda.jit("(float32[:], float32[:])", device=True, inline=True)
def devRotateOverlap(rbox1, rbox2):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    return area_inter / min(area1, area2)


@cuda.jit("(int64, float32, float32[:], uint64[:])")
def rotate_nms_overlap_kernel(n_boxes, nms_overlap_thresh, dev_boxes, dev_mask):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 6,), dtype=numba.float32)
    dev_box_idx = threadsPerBlock * col_start + tx
    if tx < col_size:
        block_boxes[tx * 6 + 0] = dev_boxes[dev_box_idx * 6 + 0]
        block_boxes[tx * 6 + 1] = dev_boxes[dev_box_idx * 6 + 1]
        block_boxes[tx * 6 + 2] = dev_boxes[dev_box_idx * 6 + 2]
        block_boxes[tx * 6 + 3] = dev_boxes[dev_box_idx * 6 + 3]
        block_boxes[tx * 6 + 4] = dev_boxes[dev_box_idx * 6 + 4]
        block_boxes[tx * 6 + 5] = dev_boxes[dev_box_idx * 6 + 5]
    cuda.syncthreads()
    if tx < row_size:
        cur_box_idx = threadsPerBlock * row_start + tx
        # cur_box = dev_boxes + cur_box_idx * 5;
        t = 0
        start = 0
        if row_start == col_start:
            start = tx + 1
        for i in range(start, col_size):
            overlap = devRotateOverlap(
                dev_boxes[cur_box_idx * 6 : cur_box_idx * 6 + 5],
                block_boxes[i * 6 : i * 6 + 5],
            )
            # print('iou', iou, cur_box_idx, i)
            if overlap > nms_overlap_thresh:
                t |= 1 << i
        col_blocks = (n_boxes) // (threadsPerBlock) + (
            (n_boxes) % (threadsPerBlock) > 0
        )
        dev_mask[cur_box_idx * col_blocks + col_start] = t


def rotate_nms_overlap_gpu(dets, nms_overlap_thresh, device_id=0):
    """nms in gpu. WARNING: this function can provide right result
    but its performance isn't be tested

    Args:
        dets ([type]): [description] [N, 6] 
        nms_overlap_thresh ([type]): [description]
        device_id ([type], optional): Defaults to 0. [description]

    Returns:
        [type]: [description]
    """
    dets = dets.astype(np.float32)
    boxes_num = dets.shape[0]
    keep_out = np.zeros([boxes_num], dtype=np.int32)
    scores = dets[:, 5]
    order = scores.argsort()[::-1].astype(np.int32)
    boxes_host = dets[order, :]

    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    cuda.select_device(device_id)
    # mask_host shape: boxes_num * col_blocks * sizeof(np.uint64)
    mask_host = np.zeros((boxes_num * col_blocks,), dtype=np.uint64)
    blockspergrid = (
        div_up(boxes_num, threadsPerBlock),
        div_up(boxes_num, threadsPerBlock),
    )
    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes_host.reshape([-1]), stream)
        mask_dev = cuda.to_device(mask_host, stream)
        rotate_nms_overlap_kernel[blockspergrid, threadsPerBlock, stream](
            boxes_num, nms_overlap_thresh, boxes_dev, mask_dev
        )
        mask_dev.copy_to_host(mask_host, stream=stream)
    num_out = nms_postprocess(keep_out, mask_host, boxes_num)
    keep = keep_out[:num_out]
    return list(order[keep])



def rotate_nms_overlap(bboxes, scores, pre_max_size=None, post_max_size=None, overlap_threshold=0.5):
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = paddle.topk(scores, k=pre_max_size)
        bboxes = bboxes[indices]
    dets = paddle.concat([bboxes, scores.unsqueeze(-1)], axis=1)
    dets_np = dets.data.cpu().numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(rotate_nms_overlap_gpu(dets_np, overlap_threshold, bboxes.device.index), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return paddle.zeros([0]).cast("int64") 
    if pre_max_size is not None:
        keep = paddle.to_tensor(keep).cast("int64") 
        return indices[keep]
    else:
        return paddle.to_tensor(keep).cast("int64") 