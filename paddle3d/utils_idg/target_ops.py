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
"""Operations related to target."""

import numpy as np
import paddle

from paddle3d.utils_idg import box_np_ops
from paddle3d.utils_idg.ops import iou3d_utils


def calculate_anchor_masks_paddle(anchors, coordinates, grid_size, voxel_size,
                                  pc_range):
    """Calculate anchor masks by paddle.
    """
    if isinstance(anchors, list):
        coord_dim = coordinates.shape[-1]
        batch_size = anchors[0].shape[0]
        anchors_mask_list = []
        voxel_masks = []
        for i in range(batch_size):
            batch_mask = (coordinates[:, 0] == i)
            this_coords = coordinates[batch_mask == True]
            voxel_mask = paddle.zeros(
                (grid_size[1] * grid_size[0], ), dtype='bool')
            indices = this_coords[:, 2] * grid_size[0] + this_coords[:, 3]  # coord(y, x)
            indices = indices.cast("int64") 

            # ===============================================
            # TODO(luoqianhui): setitem cost too much time
            # so we replace it with scatter temporarily
            # voxel_mask[indices] = True
            indices = indices % voxel_mask.shape[0] # TODO zhuyipin: to make sure that all the indices are positive
            updates = paddle.ones(indices.shape, dtype='int32')
            voxel_mask = paddle.scatter(
                voxel_mask.astype('int32'), index=indices,
                updates=updates).astype('bool')

            # ===============================================

            voxel_mask = voxel_mask.reshape([grid_size[1], grid_size[0]])
            voxel_masks.append(voxel_mask)

        for task_id in range(len(anchors)):
            anchors_near = iou3d_utils.boxes3d_to_near_torch(anchors[task_id][0])
            num_eps = -1e-8
            anchors_near[:, 0] = paddle.clip(
                paddle.floor((anchors_near[:, 0] - pc_range[0]) /
                             (voxel_size[0] + num_eps)),
                min=0,
                max=grid_size[0] - 1)
            anchors_near[:, 1] = paddle.clip(
                paddle.floor((anchors_near[:, 1] - pc_range[1]) /
                             (voxel_size[1] + num_eps)),
                min=0,
                max=grid_size[1] - 1)

            anchors_near[:, 2] = paddle.clip(
                paddle.floor((anchors_near[:, 2] - pc_range[0]) / (voxel_size[
                    0] + num_eps) + 1),
                min=0,
                max=grid_size[0] - 1)

            anchors_near[:, 3] = paddle.clip(
                paddle.floor((anchors_near[:, 3] - pc_range[1]) / (voxel_size[
                    1] + num_eps) + 1),
                min=0,
                max=grid_size[1] - 1)
            anchors_near = paddle.cast(anchors_near, 'int32')

            anchors_mask = []
            for i in range(batch_size):
                anchor_mask = iou3d_utils.anchors_match_valid_voxels(
                    anchors_near, voxel_masks[i])
                anchors_mask.append(anchor_mask.unsqueeze(0))
            # TODO(luoqianhui): paddle.stack does not support bool
            # so we use concat and unsqueeze the input before concat
            anchors_mask = paddle.concat(anchors_mask, axis=0)
            anchors_mask_list.append(anchors_mask)
        return anchors_mask_list


def assign_weight_to_voxel(boxes,
                           grid_size,
                           voxel_size,
                           pc_range,
                           corners_norm,
                           expand_ratio=0.0):
    """
    input:
        boxes: n * [x, y, w, l, yaw]
        grid_size:[600, 600, 1]
        voxel_size:[0.2, 0.2, 10]
        pc_range:[-60, -60, -5, 60, 60, 5]
    temp:
        corners: [n, 4, 2(x, y)]
    output:
        mask[grid_size[0], grid_size[1]]
    """
    # data type int or float?
    if boxes.shape[0] == 0:
        target_weight = paddle.zeros(
            (grid_size[1], grid_size[0]),
            dtype=paddle.float32, )
        return target_weight

    centers = boxes[:, 0:2]  #(n,2)
    dims = boxes[:, 2:4]
    angles = boxes[:, 4]

    if expand_ratio > 1.0:
        dims = dims * expand_ratio

    corners = dims.reshape([-1, 1, 2]) * corners_norm.reshape(
        [1, -1, 2])  #(n,4,2)

    # rotated
    rot_sin = paddle.sin(angles)
    rot_cos = paddle.cos(angles)
    rot_mat_0 = paddle.concat(
        [rot_cos.unsqueeze(0), -rot_sin.unsqueeze(0)], axis=0)
    rot_mat_1 = paddle.concat(
        [rot_sin.unsqueeze(0), rot_cos.unsqueeze(0)], axis=0)
    rot_mat_T = paddle.concat(
        [rot_mat_0.unsqueeze(0), rot_mat_1.unsqueeze(0)], axis=0)  #(2,2,n)
    corners = paddle.einsum('aij, jka -> aik', corners, rot_mat_T)  #(n,4,2)
    corners += centers.reshape([-1, 1, 2])  #(n, 4, 2)

    # to voxel coord
    corners[:, :, 0] = paddle.clip(
        (corners[:, :, 0] - pc_range[0]) / voxel_size[0],
        min=0,
        max=grid_size[0] - 1)
    corners[:, :, 1] = paddle.clip(
        (corners[:, :, 1] - pc_range[1]) / voxel_size[1],
        min=0,
        max=grid_size[1] - 1)

    corners = corners.reshape([corners.shape[0], -1])

    target_weight = iou3d_utils.boxes_to_parsing_with_weight(corners,
                                                             grid_size)

    return target_weight


def assign_label_to_voxel(boxes,
                          grid_size,
                          voxel_size,
                          pc_range,
                          corners_norm,
                          expand_ratio=0.0):
    """
    input:
        boxes: n * [x, y, w, l, yaw]
        grid_size:[600, 600, 1]
        voxel_size:[0.2, 0.2, 10]
        pc_range:[-60, -60, -5, 60, 60, 5]
    temp:
        corners: [n, 4, 2(x, y)]
    output:
        mask[grid_size[0], grid_size[1]]
    """
    # data type int or float?
    if boxes.shape[0] == 0:
        target_mask = paddle.zeros(
            (grid_size[1], grid_size[0]),
            dtype=paddle.int32, )
        return target_mask

    centers = boxes[:, 0:2]  #(n,2)
    dims = boxes[:, 2:4]
    angles = boxes[:, 4]

    if expand_ratio > 1.0:
        dims = dims * expand_ratio

    corners = dims.reshape([-1, 1, 2]) * corners_norm.reshape(
        [1, -1, 2])  #(n,4,2)

    rot_sin = paddle.sin(angles)
    rot_cos = paddle.cos(angles)
    rot_mat_0 = paddle.concat(
        [rot_cos.unsqueeze(0), -rot_sin.unsqueeze(0)], axis=0)
    rot_mat_1 = paddle.concat(
        [rot_sin.unsqueeze(0), rot_cos.unsqueeze(0)], axis=0)
    rot_mat_T = paddle.concat(
        [rot_mat_0.unsqueeze(0), rot_mat_1.unsqueeze(0)], axis=0)  #(2,2,n)
    corners = paddle.einsum('aij, jka -> aik', corners, rot_mat_T)  #(n,4,2)
    corners += centers.reshape([-1, 1, 2])  #(n, 4, 2)

    corners[:, :, 0] = paddle.clip(
        (corners[:, :, 0] - pc_range[0]) / voxel_size[0],
        min=0,
        max=grid_size[0] - 1)
    corners[:, :, 1] = paddle.clip(
        (corners[:, :, 1] - pc_range[1]) / voxel_size[1],
        min=0,
        max=grid_size[1] - 1)

    corners = corners.reshape([corners.shape[0], -1])

    target_mask = iou3d_utils.boxes_to_parsing(corners, grid_size)

    return target_mask


def assign_label_to_box(boxes, confidence_map, grid_size, voxel_size,
                        pc_range):
    """
    input:
        boxes: n * [x, y, w, l, yaw]
        confidece_map: [W, L]
        grid_size:[600, 600, 1]
        voxel_size:[0.2, 0.2, 10]
        pc_range:[-60, -60, -5, 60, 60, 5]
    temp:
        corners: [n, 4, 2(x, y)]
    output:
        confidences: n
    """
    # data type float
    if boxes.shape[0] == 0:
        confidences = paddle.zeros(
            [boxes.shape[0], ],
            dtype=boxes.dtype, )
        return confidences
    # change boxes to corners
    corners_norm = paddle.to_tensor(
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]],
        dtype=boxes.dtype)
    corners_norm = corners_norm[
        [0, 1, 3, 2]]  # left-up, left-bottom, right-bottom, right-up

    centers = boxes[:, 0:2]  #(n,2)
    dims = boxes[:, 2:4]
    angles = boxes[:, 4]
    """
    #expand
    dims[:, 0] = dims[:, 0] + voxel_size[0]
    dims[:, 1] = dims[:, 1] + voxel_size[1]
    """
    corners = dims.reshape([-1, 1, 2]) * corners_norm.reshape(
        [1, -1, 2])  #(n,4,2)

    # rotated
    rot_sin = paddle.sin(angles)
    rot_cos = paddle.cos(angles)
    rot_mat_0 = paddle.concat(
        [rot_cos.unsqueeze(0), -rot_sin.unsqueeze(0)], axis=0)
    rot_mat_1 = paddle.concat(
        [rot_sin.unsqueeze(0), rot_cos.unsqueeze(0)], axis=0)
    rot_mat_T = paddle.concat(
        [rot_mat_0.unsqueeze(0), rot_mat_1.unsqueeze(0)], axis=0)  #(2,2,n)
    corners = paddle.einsum('aij, jka -> aik', corners, rot_mat_T)  #(n,4,2)
    corners += centers.reshape([-1, 1, 2])  #(n, 4, 2)

    # to voxel coord
    corners[:, :, 0] = paddle.clip(
        (corners[:, :, 0] - pc_range[0]) / voxel_size[0],
        min=0,
        max=grid_size[0] - 1)
    corners[:, :, 1] = paddle.clip(
        (corners[:, :, 1] - pc_range[1]) / voxel_size[1],
        min=0,
        max=grid_size[1] - 1)

    corners = corners.reshape([corners.shape[0], -1])
    # print("p8: ", corners.shape)
    # print("p8 cm: ", confidence_map.shape)
    # paddle.save(corners, "ptbc_corners.pdt")
    # paddle.save(confidence_map, "confidence_map.pdt")
    confidences = iou3d_utils.parsing_to_boxes_confidence(corners, confidence_map)
    # print("p9: ", confidences)
    return confidences

# new feature
# def cal_task_roi_region_mask(anchors, roi_regions):
#     batch_size = len(roi_regions)
#     mask = paddle.zeros(anchors[:, :, 0].shape)
#     for batch_id in range(batch_size):
#         regions = roi_regions[batch_id]
#         if len(regions) == 0:
#             mask[batch_id, :] = 1.0 # TODO
#         else:
#             for region in regions:
#                 if region['type'] == 2:
#                     xy_a = anchors[batch_id, :, [0, 1]] # TODO
#                     center_xf = region['region'][0]
#                     center_yf = region['region'][1]
#                     radius = region['region'][3]
#                     center = paddle.to_tensor([center_xf, center_yf], dtype=anchors.dtype).reshape((-1, 2))
#                     dist = paddle.linalg.norm(xy_a - center, p=2, axis=1) 
#                     mask[batch_id, dist <= radius] = 1.0    #TODO
#                 else:
#                     raise NotImplementedError
#     return mask > 0

# 8A
def cal_task_roi_region_mask(anchors, roi_regions, type3_roi_regions, task_idx):
    batch_size = len(roi_regions)
    mask = paddle.zeros(anchors[:, :, 0].shape)
    assert type3_roi_regions[0][0]['task_of_interest'] ==7
    if len(type3_roi_regions[0]) > 0 and task_idx == type3_roi_regions[0][0]['task_of_interest']:
        assert all([x[0]['task_of_interest'] == type3_roi_regions[0][0]['task_of_interest'] for x in type3_roi_regions[1:]])
        # type3 regions only concern task of `task_of_interest`
        for batch_id in range(batch_size):
            type3_regions = type3_roi_regions[batch_id][0] # type3 region only has one region dict
            B, N, C = anchors.shape
            if len(type3_regions['region']) > 0:
                regions = type3_regions['region']
                regions[:, 3:5] += 0.5
                iof = iou3d_utils.boxes_iou_bev_v2(regions, anchors[batch_id], mode='iof')
                tmp = (iof.max(0) >= type3_regions['threshold']).cast('float32')
                mask[batch_id, :] = tmp
                # mask[batch_id, np.random.choice(mask.shape[1],200, replace=mask.shape[1] < 200)] = 1.0
    else:
        for batch_id in range(batch_size):
            regions = roi_regions[batch_id]
            if len(regions) == 0:
                mask[batch_id, :] = 1.0 # TODO
            else:
                for region in regions:
                    if region['type'] == 2:
                        xy_a = anchors[batch_id, :, [0, 1]] # TODO
                        center_xf = region['region'][0]
                        center_yf = region['region'][1]
                        radius = region['region'][3]
                        center = paddle.to_tensor([center_xf, center_yf], dtype=anchors.dtype).reshape((-1, 2))
                        dist = paddle.linalg.norm(xy_a - center, p=2, axis=1) 
                        # mask[batch_id, dist <= radius] = 1.0    #TODO
                        tmp = paddle.where(dist <= radius, paddle.ones([mask.shape[1]]), mask[batch_id, :])
                        mask[batch_id, :] = tmp
                    # else:
                    #     raise NotImplementedError
    return mask > 0

# new feature
def cal_roi_region_mask(anchors_list, roi_regions):
    region_mask_list = []
    # for anchors in anchors_list:
        # mask = cal_task_roi_region_mask(anchors, roi_regions)
    common_regions = [[] for _ in range(len(roi_regions))]
    type3_regions = [[] for _ in range(len(roi_regions))]
    for batch_id, batch_roi_regions in enumerate(roi_regions):
        for region in batch_roi_regions:
            if region['type'] == 3:
                type3_regions[batch_id].append(region)
            else:
                common_regions[batch_id].append(region)

    for task_idx, anchors in enumerate(anchors_list):
        mask = cal_task_roi_region_mask(anchors, common_regions, type3_regions, task_idx)      
        region_mask_list.append(mask)
    return region_mask_list


def create_target_paddle(all_anchors,
                         gt_boxes,
                         similarity_fn,
                         box_encoding_fn,
                         bctp_encoding_fn,
                         prune_anchor_fn=None,
                         gt_classes=None,
                         num_points_in_gts=None,
                         anchor_mask=None,
                         border_masks=None,
                         min_points_num_in_gt=-1,
                         matched_threshold=0.6,
                         unmatched_threshold=0.45,
                         positive_fraction=None,
                         rpn_batch_size=300,
                         norm_by_num_examples=False,
                         box_code_size=7,
                         near_bcpts=False,
                         regions_mask=None):
    """Calculate the target for each sample.
    """

    def _unmap(data, count, inds, fill=0):
        """ Unmap a subset of item (data) back to the original set of items (of
        size count) """

        if data.dim() == 1:
            ret = paddle.full((count, ), fill, dtype=data.dtype)
            idx = paddle.where(inds)[0].squeeze(-1)
            ret = ret.put_along_axis(idx, data, axis = 0)

        else:
            new_size = (count, *data.shape[1:]) 
            ret = paddle.full(new_size, fill, dtype=data.dtype)
            idx = paddle.where(inds)[0].squeeze()
            ret = paddle.scatter(ret, idx, data)

        return ret

    if not isinstance(matched_threshold, float):
        matched_threshold = matched_threshold[0].item()
    if not isinstance(unmatched_threshold, float):
        unmatched_threshold = unmatched_threshold[0].item()
    total_anchors = all_anchors.shape[0]


    if (regions_mask is not None) and (anchor_mask is not None):
        mask = regions_mask & anchor_mask
    elif anchor_mask is not None:
        mask = anchor_mask
    elif regions_mask is not None:
        mask = regions_mask
    else:
        mask = None

    if mask is not None:
        index = paddle.nonzero(mask > 0)
        # TODO(luoqianhui): index_select cannot support zero-shape index
        if 0 in index.shape:
            anchors = paddle.zeros(
                [0] + all_anchors.shape[1:], dtype=all_anchors.dtype)
        else:
            anchors = paddle.index_select(all_anchors, index, axis=0)


    
        if not isinstance(matched_threshold, float):
            matched_threshold = matched_threshold[mask]
        if not isinstance(unmatched_threshold, float):
            unmatched_threshold = unmatched_threshold[mask]
    else:
        anchors = all_anchors
    num_inside = len(paddle.nonzero(mask)) if mask is not None else total_anchors
    if gt_classes is None:
        gt_classes = paddle.ones([gt_boxes.shape[0]], dtype=paddle.int32)

    if num_points_in_gts is not None and min_points_num_in_gt >= 0:
        # TODO(luoqianhui): less_than does not support zero-shape
        if 0 not in num_points_in_gts.shape:
            points_num_mask = num_points_in_gts < min_points_num_in_gt
            gt_classes[points_num_mask] = -1

    # Compute anchor labels:
    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = paddle.full((num_inside, ), -1, dtype=paddle.int32)
    gt_ids = paddle.full((num_inside, ), -1, dtype=paddle.int64)


    # new feature
    bctp_targets = paddle.zeros((num_inside, 4, 2), dtype=all_anchors.dtype)
    border_mask_weights = paddle.zeros((num_inside, 4), dtype=all_anchors.dtype)
    bbox_outside_weights = paddle.zeros((num_inside,), dtype=all_anchors.dtype)

    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)
        # Map from anchor to gt box that has highest overlap
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
        # For each anchor, amount of overlap with most overlapping gt box
        anchor_to_gt_max = anchor_by_gt_overlap[paddle.arange(num_inside),
                                                anchor_to_gt_argmax]
        # Map from gt box to an anchor that has highest overlap
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
        # For each gt box, amount of overlap with most overlapping anchor
        gt_to_anchor_max = anchor_by_gt_overlap[
            gt_to_anchor_argmax, paddle.arange(anchor_by_gt_overlap.shape[1])]
        # must remove gt which doesn't match any anchor.

        empty_gt_idx = paddle.where(gt_to_anchor_max == 0)[0].squeeze(-1)
        if empty_gt_idx.shape[0] > 0:
            gt_to_anchor_max = gt_to_anchor_max.put_along_axis(empty_gt_idx, paddle.to_tensor(
                -1, dtype=gt_to_anchor_max.dtype), axis = 0)


        anchors_with_max_overlap = paddle.nonzero(
            anchor_by_gt_overlap == gt_to_anchor_max)[:, 0]

        # Fg label: for each gt use anchors with highest overlap
        # (including ties)
        if 0 not in anchors_with_max_overlap.shape:
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels = paddle.scatter(labels, anchors_with_max_overlap,
                                    gt_classes[gt_inds_force].cast('int32'))# .cast("int32")
            gt_ids = paddle.scatter(gt_ids, anchors_with_max_overlap,
                                    gt_inds_force)
        else:
            gt_inds_force = paddle.zeros([0], dtype='int32')
        # Fg label: above threshold IOU
        pos_inds = anchor_to_gt_max >= matched_threshold
        gt_inds = anchor_to_gt_argmax[pos_inds]
        # TODO(luoqianhui): length of the input(index) of index_select
        # can not be 0.
        if 0 not in gt_inds.shape:
            pos_idx = paddle.where(pos_inds)[0].squeeze(-1)
            labels = labels.put_along_axis(pos_idx, gt_classes[gt_inds].cast("int32"), axis = 0)
            gt_ids = gt_ids.put_along_axis(pos_idx, gt_inds, axis = 0)

        bg_inds = paddle.nonzero(anchor_to_gt_max < unmatched_threshold)[:, 0]
    else:
        bg_inds = paddle.arange(num_inside)

    fg_inds = paddle.nonzero(labels > 0)[:, 0]
    fg_max_overlap = None
    if len(gt_boxes) > 0:
        # TODO(luoqianhui): paddle.index_select cannot support zero-shape input
        # so we add this condition
        if fg_inds.shape[0] > 0:
            fg_max_overlap = paddle.gather(anchor_to_gt_max, fg_inds)
        else:
            fg_max_overlap = paddle.zeros(
                shape=[0], dtype=anchor_to_gt_max.dtype)
    # TODO(luoqianhui): paddle.index_select cannot support zero-shape input
    # so we add this condition
    if fg_inds.shape[0] > 0: 
        gt_pos_ids = paddle.gather(gt_ids, fg_inds)
    else:
        gt_pos_ids = paddle.zeros(shape=[0], dtype=gt_ids.dtype)

    # subsample positive labels if we have too many 
    if positive_fraction is not None:
        num_fg = int(positive_fraction * rpn_batch_size)
        if len(fg_inds) > num_fg:
            # np.random.seed(0) # TODO yipin
            disable_inds = np.random.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            disable_label_init = paddle.full(
                disable_inds.shape, -1, dtype=labels.dtype)
            labels = paddle.scatter(
                labels, disable_inds, updates=disable_label_init)
            fg_inds = paddle.nonzero(labels > 0)[:, 0]

        # subsample negative labels if we have too many
        # (samples with replacement, but since the set of bg inds is large most
        # samples will not have repeats)
        num_bg = rpn_batch_size - np.sum(labels > 0)
        if len(bg_inds) > num_bg:
            # np.random.seed(0)
            enable_inds = bg_inds[np.random.randint(len(bg_inds), size=num_bg)]
            enable_label_init = paddle.zeros(
                enable_inds.shape, dtype=labels.dtype)
            labels = paddle.scatter(
                labels, enable_inds, updates=enable_label_init)
    else:
        if len(gt_boxes) == 0 or anchors.shape[0] == 0:
            labels[:] = 0
        else:
            # TODO(luoqianhui): setitem cost too much time
            # so we replace it with scatter temporarily
            bg_label_init = paddle.zeros(bg_inds.shape, dtype=labels.dtype)
            labels = paddle.scatter(labels, bg_inds, updates=bg_label_init)
            # re-enable anchors_with_max_overlap
            if 0 not in gt_inds_force.shape:
                labels = paddle.scatter(labels, anchors_with_max_overlap,
                                        gt_classes[gt_inds_force].cast("int32"))
    bbox_targets = paddle.zeros(
        (num_inside, box_code_size), dtype=all_anchors.dtype)

    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        if 0 not in fg_inds.shape:
            bbox_targets[fg_inds, :] = box_encoding_fn(
                gt_boxes[anchor_to_gt_argmax[fg_inds]], anchors[fg_inds])
            
            if border_masks is not None:
                bctp_targets[fg_inds, :] = bctp_encoding_fn(paddle.index_select(gt_boxes, anchor_to_gt_argmax[fg_inds], axis=0),
                                                            paddle.index_select(anchors, fg_inds, axis=0), 
                                                            near_bcpts=near_bcpts)                                            
                border_mask_weights[fg_inds, :] = paddle.index_select(border_masks.cast("float32"), anchor_to_gt_argmax[fg_inds], axis=0)
    
    # uniform weighting of examples (given non-uniform sampling)

    if norm_by_num_examples:
        num_examples = paddle.sum(labels >= 0)  # neg + pos
        num_examples = paddle.maximum(
            paddle.to_tensor(1, num_examples.dtype), num_examples)
        
        if bbox_outside_weights[labels > 0].shape[0] > 0:   # add to avoid 0 tensor
            bbox_outside_weights[labels > 0] = paddle.to_tensor(1.0 / num_examples, dtype=bbox_outside_weights.dtype)
    else:
        if bbox_outside_weights[labels > 0].shape[0] > 0:
            bbox_outside_weights[labels > 0] = paddle.to_tensor(1.0, dtype=bbox_outside_weights.dtype)

    # Map up to original set of anchors

    if mask is not None:
        labels = _unmap(labels, total_anchors, mask, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, mask, fill=0)
        bbox_outside_weights = _unmap(
            bbox_outside_weights, total_anchors, mask, fill=0)
    		
        if border_masks is not None:
            bctp_targets = _unmap(bctp_targets,
                                total_anchors, mask, fill=0)
            border_mask_weights = _unmap(border_mask_weights,
                                     total_anchors, mask, fill=0)

    ret = {
        "labels": labels,
        "bbox_targets": bbox_targets,
        "bbox_outside_weights": bbox_outside_weights,
        "assigned_anchors_overlap": fg_max_overlap,
        "positive_gt_id": gt_pos_ids,
    }

    if border_masks is not None:
        ret["bctp_targets"] = bctp_targets
        ret["border_mask_weights"] = border_mask_weights

    if mask is not None:
        # TODO(luoqianhui): paddle.index_select cannot support zero-shape input
        # so we add this condition
        if fg_inds.shape[0] > 0:
            # TODO(luoqianhui): paddle.index_select cannot support bool tensor
            # so we cast it to int32 before and convert back after the index_select
            select_anchor_mask = mask.cast('int32')[fg_inds]
            ret["assigned_anchors_inds"] = select_anchor_mask.cast('bool')
        else:
            ret["assigned_anchors_inds"] = paddle.zeros(
                shape=[0], dtype=mask.dtype)
    else:
        ret["assigned_anchors_inds"] = fg_inds
    return ret


