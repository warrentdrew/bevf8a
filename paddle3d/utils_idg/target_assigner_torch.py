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
import itertools

from paddle3d.utils import DeltaXYZWLHRBBoxCoderIDG
from paddle3d.utils_idg.target_ops import calculate_anchor_masks_paddle, create_target_paddle, cal_roi_region_mask
from paddle3d.utils_idg.ops.iou3d_utils import RotateIou2dSimilarity, NearestIouSimilarity, DistanceSimilarity

def build_bbox_coder(bbox_coder):
    """ as name """
    assert bbox_coder['type_name'] 
    bbox_coder.pop("type_name")
    return DeltaXYZWLHRBBoxCoderIDG(**bbox_coder)

def build_torch_similarity_metric(similarity_config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    similarity_type = similarity_config['type_name']

    if similarity_type == "rotate_iou_similarity":
        return RotateIou2dSimilarity()
    elif similarity_type == "nearest_iou_similarity":
        return NearestIouSimilarity()
    elif similarity_type == "distance_similarity":
        cfg = similarity_config['distance_similarity']
        return DistanceSimilarity(
            distance_norm=cfg['distance_norm'],
            with_rotation=cfg['with_rotation'],
            rotation_alpha=cfg['rotation_alpha'])
    else:
        raise ValueError("unknown similarity type")


class AssignTargetTorch(object):
    """ as name """
    def __init__(
        self,
        cfg=None,
        **kwargs
    ):

        tasks = cfg['tasks']
        class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))
        positive_fraction = cfg['sample_positive_fraction']
        if positive_fraction < 0:
            positive_fraction = None
    
        self.box_coder = build_bbox_coder(cfg['bbox_coder'])
        
        target_assigners_config = cfg['target_assigners']
        assert len(target_assigners_config) == len(class_names)
        target_assigners = []
        for i, target_assigner_cfg in enumerate(target_assigners_config):
            assert target_assigner_cfg['class_name'] == class_names[i]
            similarity_calc = build_torch_similarity_metric(
                target_assigner_cfg['region_similarity_calculator']
            )
            # iou, atss, topk
            assigner_type = target_assigner_cfg.get("type", "iou")
            if assigner_type == "iou":
                min_points_num_in_gt = target_assigner_cfg.get("min_points_num_in_gt", -1)
                target_assigner = TargetAssignerTorch(
                    box_coder=self.box_coder,
                    match_threshold=target_assigner_cfg['matched_threshold'],
                    unmatch_threshold=target_assigner_cfg['unmatched_threshold'],
                    class_name=target_assigner_cfg['class_name'],
                    region_similarity_calculator=similarity_calc,
                    positive_fraction=positive_fraction,
                    sample_size=512,
                    min_points_num_in_gt=min_points_num_in_gt,
                )
            elif assigner_type == "atss":
                # TODO support atss
                raise NotImplementedError
            elif assigner_type == "topk":
                # TODO support topk
                raise NotImplementedError             

            target_assigners.append(target_assigner)
        
        self.target_assigners = target_assigners
        self.tasks = tasks
        self.class_names = class_names
        self.class_index_by_task = [range(len(task['class_names'])) for task in tasks]
        self.use_anchor_mask = cfg['use_anchor_mask']
        self.voxel_size = cfg['voxel_size']
        self.pc_range = cfg['pc_range']
        self.grid_size = []
        for i in range(3):
            self.grid_size.append(int((self.pc_range[i + 3] - self.pc_range[i]) / self.voxel_size[i]))
        self.num_anchor_per_locs_by_class = cfg['num_anchor_per_locs_by_class']
        
        self.quantize = cfg.get('quantize', -1)
        self.box_min_size = cfg.get('box_min_size', -1)
        self.near_bcpts = cfg.get('near_bcpts', False)

    def __call__(self, coors, batch_anchors, gt_boxes, gt_classes, box_preds, num_points_in_gts=None, 
                     border_masks=None, roi_regions=None, test_mode=False):
        """ as name """
        if test_mode:
            if self.use_anchor_mask:
                batch_anchors_mask_by_task = calculate_anchor_masks_paddle(
                    batch_anchors,
                    coors,
                    self.grid_size,
                    self.voxel_size,
                    self.pc_range
                )
            else:
                batch_anchors_mask_by_task = None
            return batch_anchors_mask_by_task
            # return {
            #     'labels': batch_anchors_mask_by_task[0],
            #     'reg_targets': batch_anchors_mask_by_task[1],
            #     'reg_weights': batch_anchors_mask_by_task[2],
            #     'positive_gt_id': batch_anchors_mask_by_task[3],
            #     'anchors_mask': batch_anchors_mask_by_task[4],
            #     'bctp_targets': batch_anchors_mask_by_task[5],
            #     'border_mask_weights': batch_anchors_mask_by_task[6],
            #     'regions_mask': batch_anchors_mask_by_task[7],
            # }
        batch_size_device = batch_anchors[0].shape[0]

        batch_anchors_by_class = []
        batch_gt_boxes_by_class = []
        batch_gt_classes_by_class = []
        if num_points_in_gts is not None:
            batch_num_points_in_gts_by_class = []
        batch_pred_boxes_by_class = []

        batch_border_masks_by_class = []
        batch_roi_regions = roi_regions
        if batch_roi_regions is not None:
            batch_region_masks_by_class = cal_roi_region_mask(batch_anchors, batch_roi_regions)


        for task_id, task in enumerate(self.tasks):
            gt_boxes_task = gt_boxes[task_id]
            gt_classes_task = gt_classes[task_id]
            if num_points_in_gts is not None:
                num_points_in_gts_task = num_points_in_gts[task_id]
            batch_anchors_task = batch_anchors[task_id]
            pred_boxes_task = box_preds[task_id].reshape((batch_size_device, -1, self.box_coder.code_size)) #.view(batch_size_device, -1, self.box_coder.code_size)
            
            if border_masks is not None:
                border_masks_task = border_masks[task_id] 
            
            anchors_num = batch_anchors_task.shape[-2]
            feature_map_size = anchors_num / sum(self.num_anchor_per_locs_by_class[task_id])
            anchor_loc_idx = 0 
            for class_id in self.class_index_by_task[task_id]:
                batch_gt_boxes_class = []
                batch_gt_classes_class = []
                if num_points_in_gts is not None:
                    batch_num_points_in_gts = []
                
                if border_masks is not None:
                    batch_border_masks_class = []
                for batch_id in range(batch_size_device):
                    if gt_classes_task[batch_id].shape[0] == 0:
                        mask = paddle.empty((0, )).cast("bool")
                    else:
                        mask = (gt_classes_task[batch_id] == (class_id + 1)) # default: encode background by zero

                        
                    batch_gt_boxes_class.append(gt_boxes_task[batch_id][mask])
                    batch_gt_classes_class.append(gt_classes_task[batch_id][mask])
                    if num_points_in_gts is not None:
                        batch_num_points_in_gts.append(num_points_in_gts_task[batch_id][mask])
                    if border_masks is not None:
                        batch_border_masks_class.append(border_masks_task[batch_id][mask])
                batch_gt_boxes_by_class.append(batch_gt_boxes_class)
                batch_gt_classes_by_class.append(batch_gt_classes_class)
                if num_points_in_gts is not None:
                    batch_num_points_in_gts_by_class.append(batch_num_points_in_gts)

                if border_masks is not None:
                    batch_border_masks_by_class.append(batch_border_masks_class)

                class_anchors_num = int(self.num_anchor_per_locs_by_class[task_id][class_id] * feature_map_size)
                batch_anchors_by_class.append(batch_anchors_task[:, anchor_loc_idx: (anchor_loc_idx + \
                                                                class_anchors_num)])
                batch_pred_boxes_by_class.append(pred_boxes_task[:, anchor_loc_idx: (anchor_loc_idx + \
                                                                class_anchors_num)])
                anchor_loc_idx = anchor_loc_idx + class_anchors_num

        if self.use_anchor_mask:
            batch_anchors_mask_by_class = calculate_anchor_masks_paddle(
                batch_anchors_by_class,
                coors,
                self.grid_size,
                self.voxel_size,
                self.pc_range
            )
        targets_list_by_class = []
        for class_id in range(len(batch_anchors_by_class)):
            targets_list_by_batch = []
            for batch_id in range(batch_size_device):
                if self.use_anchor_mask:
                    anchor_mask = batch_anchors_mask_by_class[class_id][batch_id]
                else:
                    anchor_mask = None

                if batch_roi_regions is not None:
                    region_mask = batch_region_masks_by_class[class_id][batch_id]
                else:
                    region_mask = None

                if len(batch_gt_boxes_by_class[class_id]) > 0:
                    gt_boxes_class = batch_gt_boxes_by_class[class_id][batch_id]
                    gt_classes_class = batch_gt_classes_by_class[class_id][batch_id]
                    if num_points_in_gts is not None:
                        num_points_in_gts_class = batch_num_points_in_gts_by_class[class_id][batch_id]
                    border_masks_class = batch_border_masks_by_class[class_id][batch_id] if border_masks is not None else None
                else:
                    gt_boxes_class = paddle.zeros((0, self.box_coder.code_size), dtype='float32')
                    gt_classes_class = paddle.zeros((0), dtype='int64')
                    if num_points_in_gts is not None:
                        num_points_in_gts_class = paddle.zeros((0), dtype='int64')
                    border_masks_class = paddle.zeros((0, 4), dtype='float32') if border_masks is not None else None
                # quantize
                if self.quantize > 0:
                    gt_boxes_class[:, 3:6] = paddle.ceil(gt_boxes_class[:, 3:6] / self.quantize) * self.quantize

                if self.box_min_size > 0:
                    gt_boxes_class[:, 3:6] = paddle.clip(gt_boxes_class[:, 3:6], min = self.box_min_size) 

                if num_points_in_gts is None:
                    num_points_in_gts_class = None

                targets = self.target_assigners[class_id].assign(
                    batch_pred_boxes_by_class[class_id][batch_id],
                    batch_anchors_by_class[class_id][batch_id],
                    gt_boxes_class,
                    anchors_mask=anchor_mask,
                    gt_classes=gt_classes_class, 
                    num_points_in_gts=num_points_in_gts_class,
                    border_masks_class = border_masks_class,
                    near_bcpts = self.near_bcpts,
                    regions_mask = region_mask
                )
                targets_list_by_batch.append(targets)
            
            targets_dict = {
                "labels": [t["labels"] for t in targets_list_by_batch],
                "bbox_targets": [t["bbox_targets"] for t in targets_list_by_batch],
                "bbox_outside_weights": [t["bbox_outside_weights"] for t in targets_list_by_batch],
                "positive_gt_id": [t["positive_gt_id"] for t in targets_list_by_batch]
            }
        		
            if border_masks is not None:
                targets_dict["bctp_targets"] = [t["bctp_targets"] for t in targets_list_by_batch]
                targets_dict["border_mask_weights"] = [t["border_mask_weights"] for t in targets_list_by_batch]

            targets_dict["labels"] = paddle.stack(targets_dict["labels"], axis=0) 
            targets_dict["bbox_targets"] = paddle.stack(targets_dict["bbox_targets"], axis=0) 
            targets_dict["bbox_outside_weights"] = paddle.stack(targets_dict["bbox_outside_weights"], axis=0)

            if border_masks is not None:
                targets_dict["bctp_targets"] = paddle.stack(targets_dict["bctp_targets"], axis=0) #.contiguous()
                targets_dict["border_mask_weights"] = paddle.stack(targets_dict["border_mask_weights"], axis=0) #.contiguous()

            targets_list_by_class.append(targets_dict)

        # merge class to task
        class_idx = 0
        targets_dict_list = []
        anchors_mask_list = []
        regions_mask_list = []
        for task_id, task in enumerate(self.tasks):
            targets_dict = {
                "labels": [targets_list_by_class[i]['labels'] for i in range(class_idx, 
                                                            (class_idx + task['num_class']))],
                "bbox_targets": [targets_list_by_class[i]['bbox_targets'] for i in range(class_idx, 
                                                            (class_idx + task['num_class']))],
                "bbox_outside_weights": [targets_list_by_class[i]['bbox_outside_weights'] for i in range(class_idx, 
                                                            (class_idx + task['num_class']))],
                "positive_gt_id": [targets_list_by_class[i]['positive_gt_id'] for i in range(class_idx, 
                                                            (class_idx + task['num_class']))],
            }
            if border_masks is not None:
                targets_dict["bctp_targets"] = [targets_list_by_class[i]['bctp_targets'] for i in range(class_idx, (class_idx + task['num_class']))]
                targets_dict["border_mask_weights"] = [targets_list_by_class[i]['border_mask_weights'] for i in range(class_idx, (class_idx + task['num_class']))]

            targets_dict["labels"] = paddle.concat(targets_dict["labels"], axis=1)
            targets_dict["bbox_targets"] = paddle.concat(targets_dict["bbox_targets"], axis=1)
            targets_dict["bbox_outside_weights"] = paddle.concat(targets_dict["bbox_outside_weights"], axis=1)
        		
            if border_masks is not None:
                targets_dict["bctp_targets"] = paddle.concat(targets_dict["bctp_targets"], axis=1) #.contiguous()
                targets_dict["border_mask_weights"] = paddle.concat(targets_dict["border_mask_weights"], axis=1) #.contiguous()

            targets_dict_list.append(targets_dict)
            if self.use_anchor_mask:
                batch_anchors_list = []
                for t in batch_anchors_mask_by_class[class_idx: class_idx + task['num_class']]:
                    batch_anchors_list.append(t.cast('int32'))
                
                anchors_mask_task = paddle.concat(batch_anchors_list, axis = 1) #.stack(batch_anchors_list, axis = 1)
                anchors_mask_task = anchors_mask_task.cast("bool")
                anchors_mask_list.append(anchors_mask_task)
            		
            if batch_roi_regions is not None:
                regions_mask_task = paddle.concat(batch_region_masks_by_class[class_idx: class_idx + task['num_class']], axis=1) # .contiguous()
                regions_mask_list.append(regions_mask_task)

            class_idx += task['num_class']
        
        labels = [targets_dict["labels"] for targets_dict in targets_dict_list]
        reg_targets = [targets_dict["bbox_targets"] for targets_dict in targets_dict_list]
        reg_weights = [targets_dict["bbox_outside_weights"] for targets_dict in targets_dict_list]
        positive_gt_id = [targets_dict["positive_gt_id"] for targets_dict in targets_dict_list]
		
        if border_masks is not None:
            bctp_targets = [targets_dict["bctp_targets"] for targets_dict in targets_dict_list]
            border_mask_weights = [targets_dict["border_mask_weights"] for targets_dict in targets_dict_list]
        else:
            bctp_targets = None
            border_mask_weights = None
            

        if self.use_anchor_mask:
            anchors_mask = anchors_mask_list
        else:
            anchors_mask = None
        
        if batch_roi_regions is not None:
            regions_mask = regions_mask_list
        else:
            regions_mask = None

        return labels, reg_targets, reg_weights, positive_gt_id, anchors_mask, bctp_targets, border_mask_weights, regions_mask
        # return {
        #     'labels': labels,
        #     'reg_targets': reg_targets,
        #     'reg_weights': reg_weights,
        #     'positive_gt_id': positive_gt_id,
        #     'anchors_mask': anchors_mask,
        #     'bctp_targets': bctp_targets, 
        #     'border_mask_weights': border_mask_weights, 
        #     'regions_mask': regions_mask
        # }

   

class TargetAssignerTorch(object):
    """ as name """
    def __init__(
        self,
        box_coder,
        match_threshold=None,
        unmatch_threshold=None,
        class_name=None,
        region_similarity_calculator=None,
        positive_fraction=None,
        sample_size=512,
        topk=9,
        min_points_num_in_gt=-1,
    ):
        self._region_similarity_calculator = region_similarity_calculator
        self._box_coder = box_coder
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold
        self._positive_fraction = positive_fraction
        self._sample_size = sample_size
        self._class_name = class_name
        self._min_points_num_in_gt = min_points_num_in_gt

    @property
    def box_coder(self):
        """ as name """
        return self._box_coder

    @property
    def classes(self):
        """ as name """
        return [self._class_name]

    def assign(
        self,
        pred_boxes,
        anchors,
        gt_boxes,
        anchors_mask=None,
        gt_classes=None,
        num_points_in_gts=None,
        border_masks_class = None,
        near_bcpts = None, 
        regions_mask = None
    ):
        """ as name """

        def similarity_fn(anchors, gt_boxes):
            return self._region_similarity_calculator(anchors, gt_boxes)

        def box_encoding_fn(boxes, anchors):
            return self._box_coder.encode(anchors, boxes)
        
        def bctp_encoding_fn(boxes, anchors, near_bcpts = None):
            return self._box_coder.bctp_encode_paddle(boxes, anchors, near_bcpts = near_bcpts)


        return create_target_paddle(
            anchors,
            gt_boxes,
            similarity_fn,
            box_encoding_fn,
            bctp_encoding_fn, 
            gt_classes=gt_classes,
            num_points_in_gts=num_points_in_gts,
            anchor_mask=anchors_mask,
            border_masks = border_masks_class, 
            min_points_num_in_gt=self._min_points_num_in_gt,
            matched_threshold=self._match_threshold,
            unmatched_threshold=self._unmatch_threshold,
            positive_fraction=self._positive_fraction,
            rpn_batch_size=self._sample_size,
            norm_by_num_examples=False,
            box_code_size=self.box_coder.code_size,	
            near_bcpts = near_bcpts,
            regions_mask = regions_mask
        )
