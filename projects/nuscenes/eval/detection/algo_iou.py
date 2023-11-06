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
# !/usr/bin/env python3
# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

import pickle
from typing import Callable
from operator import add
import numpy as np
from pyquaternion import Quaternion

from projects.nuscenes.eval.common.data_classes import EvalBoxes
from projects.nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, iou_calculate, cummean, quaternion_yaw
from projects.nuscenes.eval.detection.data_classes import DetectionMetricData, DetectionConfig
from projects.nuscenes.eval.detection.algo_score import geo_iou_match, geo_iou_match_bi, pts_iou_match, pts_iou_match_bi

def accumulate(sensor: str,
               gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               accumulate_dict: dict,
               class_name: str,
               cfg: DetectionConfig, 
               distance: list,
               pred_taken: dict, 
               verbose: bool = False):
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    if cfg.dist_type == 'pts_dist':
        assert cfg.pts_dist_path is not None
        dist_infos = pickle.load(open(cfg.pts_dist_path, 'rb'), encoding='iso-8859-1') 
    else:
        dist_infos = None
    
    # Count the positives. #计算该类别下的有效gt_box个数
    if cfg.ignore_gt_valid:
        if cfg.dist_type == 'pts_dist':
            npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name and \
                        dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]>=distance[0] and\
                        dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]<=distance[1]])
        else:
            npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name and gt_box.ego_dist>=distance[0] and gt_box.ego_dist<=distance[1]])
    else:
        if cfg.dist_type == 'pts_dist':
            npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name and gt_box.valid == True and \
                         dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]>=distance[0] and \
                         dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]<=distance[1]])
        else:
            npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name and gt_box.valid == True and gt_box.ego_dist>=distance[0] and gt_box.ego_dist<=distance[1]])
        
    # Organize the predictions in a single list.
    inds_detname_dict = {}
    if cfg.cross_cls_match:
        pred_boxes_list = []
        for box in pred_boxes.all:
            pred_boxes_list.append(box)
            if box.sample_token not in inds_detname_dict:
                inds_detname_dict.update({box.sample_token: {box.index: box.detection_name}})
            else:
                inds_detname_dict[box.sample_token].update({box.index: box.detection_name})
    else:
        pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]

    if accumulate_dict['is_finished'] == False:
        if cfg.iou_type == 'pts_iou':
            accumulate_dict = pts_iou_match(accumulate_dict, gt_boxes, pred_boxes_list,
                                            class_name, cfg)
        else:
            accumulate_dict = geo_iou_match(accumulate_dict, gt_boxes, pred_boxes_list,
                                            class_name, cfg)
    
    # Do the actual matching.
    tp = [0]*10001  # Accumulator of true positives.
    fp = [0]*10001  # Accumulator of false positives.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'iou3d_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'ioubev_err': [],
                  'npos': npos}

    for sample_token in accumulate_dict['sample_token'].keys():
        if class_name not in accumulate_dict['sample_token'][sample_token].keys():
            continue
        pred_matrix = accumulate_dict['sample_token'][sample_token][class_name]['pred_matrix'][:,:7]
        gt_matrix   = accumulate_dict['sample_token'][sample_token][class_name]['gt_matrix']
        iou_metric  = accumulate_dict['sample_token'][sample_token][class_name]['iou_metric']
        dist_metric = accumulate_dict['sample_token'][sample_token][class_name]['dist_metric']
        det_score   = accumulate_dict['sample_token'][sample_token][class_name]['pred_matrix'][:,7] 
        global_inds = accumulate_dict['sample_token'][sample_token][class_name]['pred_matrix'][:,8] 
        det_mask = accumulate_dict['sample_token'][sample_token][class_name]['det_mask']
        
        if len(iou_metric.shape) < 3:  #该时间段没有GT
            if cfg.cross_cls_match:
                pos_ind = [i for i, key in enumerate(inds_detname_dict[sample_token].keys())
                           if inds_detname_dict[sample_token][key] == class_name]
                if len(pos_ind) == 0:
                    continue
                if sensor == 'cam':
                    if class_name in ['bigMot', 'smallMot', 'Mot']:
                        if distance[-1] == cfg.rel_ab_th:
                            if cfg.iou_type == 'pts_iou':
                                ori_iou_metric = 1-accumulate_dict['sample_token'][sample_token][class_name]['ori_iou_metric']
                                best_match_indx = list(ori_iou_metric.argmin(1))
                                best_match_value = list(ori_iou_metric.min(1))
                                pop_list = []
                                for i in range(len(pos_ind)):
                                    if det_mask[int(global_inds[pos_ind[i]])] == False:
                                        pop_list.append(i)
                                for pop_idx in reversed(pop_list):
                                    pos_ind.pop(pop_idx)
                            else:
                                ori_iou_metric = iou_metric
                                best_match_local_indx = list(ori_iou_metric.argmin(0))
                                best_match_value = list(ori_iou_metric.min(0))
                                best_match_indx = [global_inds[i] for i in best_match_local_indx]
                            threshold = 1 - cfg.iou_th_tp
                        else:
                            ori_dist_metric = dist_metric
                            best_match_local_indx = list(ori_dist_metric.argmin(0))
                            best_match_value = list(ori_dist_metric.min(0))
                            best_match_indx = [global_inds[i] for i in best_match_local_indx]
                            threshold = cfg.rel_dist_th_tp
                    else:
                        if distance[-1] == cfg.rel_ab_th:
                            ori_dist_metric = dist_metric
                            best_match_local_indx = list(ori_dist_metric.argmin(0))
                            best_match_value = list(ori_dist_metric.min(0))
                            best_match_indx = [global_inds[i] for i in best_match_local_indx]
                            threshold = cfg.ab_dist_th_tp
                        else:
                            ori_dist_metric = dist_metric
                            best_match_local_indx = list(ori_dist_metric.argmin(0))
                            best_match_value = list(ori_dist_metric.min(0))
                            best_match_indx = [global_inds[i] for i in best_match_local_indx]
                            threshold = cfg.rel_dist_th_tp
                else:
                    if cfg.iou_type == 'pts_iou':
                        ori_iou_metric = 1-accumulate_dict['sample_token'][sample_token][class_name]['ori_iou_metric']
                        best_match_indx = list(ori_iou_metric.argmin(1))
                        best_match_value = list(ori_iou_metric.min(1))
                        pop_list = []
                        for i in range(len(pos_ind)):
                            if det_mask[int(global_inds[pos_ind[i]])] == False:
                                pop_list.append(i)
                        for pop_idx in reversed(pop_list):
                            pos_ind.pop(pop_idx)
                    else:
                        ori_iou_metric = iou_metric
                        best_match_local_indx = list(ori_iou_metric.argmin(0))
                        best_match_value = list(ori_iou_metric.min(0))
                        best_match_indx = [global_inds[i] for i in best_match_local_indx]
                    threshold = 1-cfg.iou_th_tp
                
                pop_list = []
                for i in range(len(pos_ind)):
                    if global_inds[pos_ind[i]] in best_match_indx and best_match_value[best_match_indx.index(global_inds[pos_ind[i]])] <= threshold:
                        pop_list.append(i)
                
                for pop_idx in reversed(pop_list):
                    pos_ind.pop(pop_idx)

                if len(pos_ind) == 0:
                    continue
                else:
                    pos_ind = np.array(pos_ind)

                if cfg.dist_type == 'pts_dist':
                    pos_inds = global_inds[pos_ind.astype(np.int)].astype(np.int)
                    try:
                        pred_box_dist = dist_infos[sample_token]['pred_dists'][pos_inds]
                    except:
                        pred_box_dist = np.linalg.norm(pred_matrix[pos_ind, :2], axis=1)
                else:
                    pred_box_dist = np.linalg.norm(pred_matrix[pos_ind, :2], axis=1)
                det_score = det_score[pos_ind]
            else:
                if cfg.dist_type == 'pts_dist':
                    try:
                        pred_box_dist = dist_infos[sample_token]['pred_dists'][global_inds.astype(np.int)]
                    except:
                        pred_box_dist = np.linalg.norm(pred_matrix[:, :2], axis=1)
                else:
                    pred_box_dist = np.linalg.norm(pred_matrix[:, :2], axis=1)
            
            fp_min = pred_box_dist >= distance[0]
            fp_max = pred_box_dist < distance[1]
            num_fps = pred_box_dist[np.where((fp_min & fp_max) == True)].shape[0]
            det_score = det_score[np.where((fp_min & fp_max) == True)]
            if num_fps > 0:
                det_index = det_score / 0.0001
                det_index = det_index.astype(int)
                for i, index in enumerate(det_index):
                    fp[index] += 1
        else:
            gt_taken = []  # Initially no gt bounding box is matched.
            for gt_idx in range(gt_matrix.shape[0]):
                gt_box = gt_matrix[gt_idx]
                min_metric = np.inf
                match_pred_idx = None
                match_pred_box = None
                for pred_idx in range(pred_matrix.shape[0]):
                    global_ind = global_inds[pred_idx]
                    if sample_token in pred_taken.keys() and global_ind in pred_taken[sample_token]:
                        continue
                    pred_box = pred_matrix[pred_idx]
                    if sensor == 'cam':
                        if class_name in ['bigMot', 'smallMot', 'Mot']:
                            if distance[-1] == cfg.rel_ab_th:
                                if cfg.iou_type == 'pts_iou':
                                    this_metric = iou_metric[0, int(gt_box[7]), int(global_ind)]
                                else:
                                    this_metric = iou_metric[1, pred_idx, gt_idx] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0, pred_idx, gt_idx]
                            else:
                                this_metric = dist_metric[1, pred_idx, gt_idx]
                        else:
                            if distance[-1] == cfg.rel_ab_th:
                                this_metric = dist_metric[0, pred_idx, gt_idx]
                            else:
                                this_metric = dist_metric[1, pred_idx, gt_idx]
                    else:
                        if cfg.iou_type == 'pts_iou':
                            this_metric = iou_metric[0, int(gt_box[7]), int(global_ind)]
                        else:
                            this_metric = iou_metric[1, pred_idx, gt_idx] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0, pred_idx, gt_idx]
                    
                    if this_metric < min_metric:
                        min_metric = this_metric
                        match_pred_idx = pred_idx
                        match_pred_box = pred_box

                #If the closest match is close enough according to threshold we have a match!
                if sensor == 'cam':
                    if class_name in ['bigMot', 'smallMot', 'Mot']:
                        if distance[-1] == cfg.rel_ab_th:
                            is_match = min_metric < 1 - cfg.iou_th_tp
                        else:
                            is_match = min_metric < cfg.rel_dist_th_tp
                    else:
                        if distance[-1] == cfg.rel_ab_th:
                            is_match = min_metric < cfg.ab_dist_th_tp
                        else:
                            is_match = min_metric < cfg.rel_dist_th_tp  
                else:
                    is_match = min_metric < 1 - cfg.iou_th_tp   
                
                if is_match:
                    det_index = (det_score[match_pred_idx] / 0.0001).astype(int)
                    if cfg.dist_type == 'pts_dist':
                        matched_gt_dist = dist_infos[sample_token]['gt_dists'][int(gt_box[7])]
                    else:
                        matched_gt_dist = np.linalg.norm(np.array(gt_box[:2])) 
                    if matched_gt_dist >= distance[0] and matched_gt_dist < distance[1]:
                        gt_taken.append(gt_idx)
                        if sample_token in pred_taken.keys():
                            pred_taken[sample_token].append(global_inds[match_pred_idx])
                        else:
                            pred_taken.update({sample_token: [global_inds[match_pred_idx]]})

                        tp[det_index]+=1

                        # Since it is a match, update match data also.
                        match_data['trans_err'].append(dist_metric[0, match_pred_idx, gt_idx])
                        match_data['scale_err'].append(1 - scale_iou(gt_box[:7], match_pred_box[:7]))
                        period = np.pi if class_name == 'barrier' else 2 * np.pi
                        match_data['orient_err'].append(yaw_diff(gt_box[6], match_pred_box[6], period=period))
                        if cfg.iou_type == 'pts_iou':
                            match_data['iou3d_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
                            match_data['ioubev_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
                        else:
                            match_data['iou3d_err'].append(iou_metric[0, match_pred_idx, gt_idx])
                            match_data['ioubev_err'].append(iou_metric[1, match_pred_idx, gt_idx])

            for pred_idx in range(pred_matrix.shape[0]):
                global_ind = global_inds[pred_idx]
                if cfg.iou_type == 'pts_iou' and det_mask[int(global_ind)] == False:
                    continue
                if sample_token in pred_taken.keys() and global_ind in pred_taken[sample_token]:
                    continue
                if cfg.cross_cls_match and inds_detname_dict[sample_token][global_ind] != class_name:
                    continue
                if cfg.cross_cls_match:
                    if sensor == 'cam':
                        if class_name in ['bigMot', 'smallMot', 'Mot']:
                            if distance[-1] == cfg.rel_ab_th:
                                if cfg.iou_type == 'pts_iou':
                                    ori_iou_metric = 1-accumulate_dict['sample_token'][sample_token][class_name]['ori_iou_metric']
                                    best_match_indx = list(ori_iou_metric.argmin(1))
                                    best_match_value = list(ori_iou_metric.min(1))
                                else:
                                    ori_iou_metric = iou_metric[1] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0]
                                    best_match_local_indx = list(ori_iou_metric.argmin(0))
                                    best_match_value = list(ori_iou_metric.min(0))
                                    best_match_indx = [global_inds[i] for i in best_match_local_indx]
                                threshold = 1 - cfg.iou_th_tp
                            else:
                                ori_dist_metric = dist_metric[1]
                                best_match_local_indx = list(ori_dist_metric.argmin(0))
                                best_match_value = list(ori_dist_metric.min(0))
                                best_match_indx = [global_inds[i] for i in best_match_local_indx]
                                threshold = cfg.rel_dist_th_tp
                        else:
                            if distance[-1] == cfg.rel_ab_th:
                                ori_dist_metric = dist_metric[0]
                                best_match_local_indx = list(ori_dist_metric.argmin(0))
                                best_match_value = list(ori_dist_metric.min(0))
                                best_match_indx = [global_inds[i] for i in best_match_local_indx]
                                threshold = cfg.ab_dist_th_tp
                            else:
                                ori_dist_metric = dist_metric[1]
                                best_match_local_indx = list(ori_dist_metric.argmin(0))
                                best_match_value = list(ori_dist_metric.min(0))
                                best_match_indx = [global_inds[i] for i in best_match_local_indx]
                                threshold = cfg.rel_dist_th_tp
                    else:
                        if cfg.iou_type == 'pts_iou':
                            ori_iou_metric = 1-accumulate_dict['sample_token'][sample_token][class_name]['ori_iou_metric']
                            best_match_indx = list(ori_iou_metric.argmin(1))
                            best_match_value = list(ori_iou_metric.min(1))
                        else:
                            ori_iou_metric = iou_metric[1] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0]
                            best_match_local_indx = list(ori_iou_metric.argmin(0))
                            best_match_value = list(ori_iou_metric.min(0))
                            best_match_indx = [global_inds[i] for i in best_match_local_indx]
                        threshold = 1-cfg.iou_th_tp
                
                    if global_ind in best_match_indx and best_match_value[best_match_indx.index(global_ind)] <= threshold:
                        continue

                pred_box = pred_matrix[pred_idx]

                if cfg.dist_type == 'pts_dist':
                    pred_box_dist = dist_infos[sample_token]['pred_dists'][int(global_ind)]
                else:
                    pred_box_dist = np.linalg.norm(pred_box[:2])
                    
                if pred_box_dist>= distance[0] and pred_box_dist < distance[1]:
                    det_index = (det_score[pred_idx]/ 0.0001).astype(int)
                    fp[det_index] += 1
    
    pred_num = sum(tp)+sum(fp)
    if pred_num == 0 and npos == 0:
        return DetectionMetricData.no_gt_predictions()
    elif npos == 0:
        return DetectionMetricData.no_gts(fp)
    else:
        if len(match_data['trans_err']) == 0:
            return DetectionMetricData.no_pos_predictions(npos, fp)

        # ---------------------------------------------
        # Calculate and interpolate precision and recall
        # ---------------------------------------------
        # Accumulate.

        cls_recall = sum(tp)/npos

        ap = 0
        tp_ = np.cumsum(tp[::-1])[::-1]
        fp_ = np.cumsum(fp[::-1])[::-1]

        current_recall = 0.0
        recall_dim = 101
        recall_step = 1.0 / (recall_dim - 1)
        precision_list = []
        recall_list = []
        for i in range(len(tp_)-1, -1, -1):
            total_det_num = tp_[i] + fp_[i]
            total_tp_num = tp_[i]
            left_recall = total_tp_num / npos
            right_recall = 0.0

            if i > 0:
                right_recall = tp_[i - 1] / npos
            else:
                right_recall = left_recall
        
            if (right_recall - current_recall) < (current_recall - left_recall) and i > 0:
                continue
            precision = total_tp_num/ total_det_num if total_det_num > 0 else 1.0
            current_recall += recall_step
            ap += precision

            precision_list.append(precision)
            recall_list.append(current_recall)

        ap /= recall_dim

        # ---------------------------------------------
        # Re-sample the match-data to match, prec, recall and conf.
        # ---------------------------------------------

        for key in match_data.keys():
            if key =='npos':
                continue  # Confidence is used as reference to align with fp and tp. So skip in this step.
            else:
                match_data[key] = np.array(match_data[key])

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(ap=ap,
                               recall=recall_list,
                               precision=precision_list,
                               trans_err=match_data['trans_err'],
                               iou3d_err=match_data['iou3d_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               ioubev_err=match_data['ioubev_err'],
                               confusion_matrix = 0, # not used
                               cls_recall = cls_recall,
                               pred_num = pred_num,
                               tp = np.array(tp),
                               fp = np.array(fp), 
                               npos = match_data['npos'])

def accumulate_all(sensor: str,
                   gt_boxes: EvalBoxes,
                   pred_boxes: EvalBoxes,
                   accumulate_dict: dict,
                   class_name: str,
                   cfg: DetectionConfig,
                   distance: list,
                   pred_taken: dict, 
                   verbose: bool = False):

    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    if cfg.dist_type == 'pts_dist':
        assert cfg.pts_dist_path is not None
        dist_infos = pickle.load(open(cfg.pts_dist_path, 'rb'), encoding='iso-8859-1') 
    else:
        dist_infos = None

    # Count the positives. #计算该类别下的有效gt_box个数
    if cfg.ignore_gt_valid:
        if cfg.dist_type == 'pts_dist':
            npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name and \
                        dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]>=distance[0] and\
                        dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]<=distance[1]])
        else:
            npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name and gt_box.ego_dist>=distance[0] and gt_box.ego_dist<=distance[1]])
    else:
        if cfg.dist_type == 'pts_dist':
            npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name and gt_box.valid == True and \
                         dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]>=distance[0] and \
                         dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]<=distance[1]])
        else:
            npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name and gt_box.valid == True and gt_box.ego_dist>=distance[0] and gt_box.ego_dist<=distance[1]])
        
    
    # Organize the predictions in a single list.
    inds_detname_dict = {}
    if cfg.cross_cls_match:
        pred_boxes_list = []
        for box in pred_boxes.all:
            pred_boxes_list.append(box)
            if box.sample_token not in inds_detname_dict:
                inds_detname_dict.update({box.sample_token: {box.index: box.detection_name}})
            else:
                inds_detname_dict[box.sample_token].update({box.index: box.detection_name})
    else:
        pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]

    if accumulate_dict['is_finished'] == False:
        if cfg.iou_type == 'pts_iou':
            accumulate_dict = pts_iou_match(accumulate_dict, gt_boxes, pred_boxes_list,
                                            class_name, cfg)
        else:
            accumulate_dict = geo_iou_match(accumulate_dict, gt_boxes, pred_boxes_list,
                                            class_name, cfg)

    # Do the actual matching.
    tp = [0]*10001  # Accumulator of true positives.
    fp = [0]*10001  # Accumulator of false positives.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'iou3d_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'ioubev_err': [],
                  'conf': [],
                  'npos': npos}

    for sample_token in accumulate_dict['sample_token'].keys():
        if class_name not in accumulate_dict['sample_token'][sample_token].keys():
            continue
        pred_matrix = accumulate_dict['sample_token'][sample_token][class_name]['pred_matrix'][:,:7]
        gt_matrix   = accumulate_dict['sample_token'][sample_token][class_name]['gt_matrix']
        iou_metric  = accumulate_dict['sample_token'][sample_token][class_name]['iou_metric']
        dist_metric = accumulate_dict['sample_token'][sample_token][class_name]['dist_metric']
        det_score   = accumulate_dict['sample_token'][sample_token][class_name]['pred_matrix'][:,7] #已排序
        global_inds = accumulate_dict['sample_token'][sample_token][class_name]['pred_matrix'][:,8] 
        det_mask = accumulate_dict['sample_token'][sample_token][class_name]['det_mask']

        if len(iou_metric.shape) < 3:  #该区间没有GT
            if cfg.cross_cls_match:
                pos_ind = [i for i, key in enumerate(inds_detname_dict[sample_token].keys())
                           if inds_detname_dict[sample_token][key] == class_name]
                if len(pos_ind) == 0:
                    continue
                if sensor == 'cam':
                    if class_name in ['bigMot', 'smallMot', 'Mot']:
                        if distance[-1] == cfg.rel_ab_th:
                            if cfg.iou_type == 'pts_iou':
                                ori_iou_metric = 1-accumulate_dict['sample_token'][sample_token][class_name]['ori_iou_metric']
                                best_match_indx = list(ori_iou_metric.argmin(1))
                                best_match_value = list(ori_iou_metric.min(1))
                                pop_list = []
                                for i in range(len(pos_ind)):
                                    if det_mask[int(global_inds[pos_ind[i]])] == False:
                                        pop_list.append(i)
                                for pop_idx in reversed(pop_list):
                                    pos_ind.pop(pop_idx)
                            else:
                                ori_iou_metric = iou_metric
                                best_match_local_indx = list(ori_iou_metric.argmin(0))
                                best_match_value = list(ori_iou_metric.min(0))
                                best_match_indx = [global_inds[i] for i in best_match_local_indx]
                            threshold = 1 - cfg.iou_th_tp
                        else:
                            ori_dist_metric = dist_metric
                            best_match_local_indx = list(ori_dist_metric.argmin(0))
                            best_match_value = list(ori_dist_metric.min(0))
                            best_match_indx = [global_inds[i] for i in best_match_local_indx]
                            threshold = cfg.rel_dist_th_tp
                    else:
                        if distance[-1] == cfg.rel_ab_th:
                            ori_dist_metric = dist_metric
                            best_match_local_indx = list(ori_dist_metric.argmin(0))
                            best_match_value = list(ori_dist_metric.min(0))
                            best_match_indx = [global_inds[i] for i in best_match_local_indx]
                            threshold = cfg.ab_dist_th_tp
                        else:
                            ori_dist_metric = dist_metric
                            best_match_local_indx = list(ori_dist_metric.argmin(0))
                            best_match_value = list(ori_dist_metric.min(0))
                            best_match_indx = [global_inds[i] for i in best_match_local_indx]
                            threshold = cfg.rel_dist_th_tp
                else:
                    if cfg.iou_type == 'pts_iou':
                        ori_iou_metric = 1-accumulate_dict['sample_token'][sample_token][class_name]['ori_iou_metric']
                        best_match_indx = list(ori_iou_metric.argmin(1))
                        best_match_value = list(ori_iou_metric.min(1))
                        pop_list = []
                        for i in range(len(pos_ind)):
                            if det_mask[int(global_inds[pos_ind[i]])] == False:
                                pop_list.append(i)
                        for pop_idx in reversed(pop_list):
                            pos_ind.pop(pop_idx)
                    else:
                        ori_iou_metric = iou_metric
                        best_match_local_indx = list(ori_iou_metric.argmin(0))
                        best_match_value = list(ori_iou_metric.min(0))
                        best_match_indx = [global_inds[i] for i in best_match_local_indx]
                    threshold = 1-cfg.iou_th_tp

                pop_list = []
                for i in range(len(pos_ind)):
                    if global_inds[pos_ind[i]] in best_match_indx and best_match_value[best_match_indx.index(global_inds[pos_ind[i]])] <= threshold:
                        pop_list.append(i)
                
                for pop_idx in reversed(pop_list):
                    pos_ind.pop(pop_idx)

                if len(pos_ind) == 0:
                    continue
                else:
                    pos_ind = np.array(pos_ind)

                if cfg.dist_type == 'pts_dist':
                    pos_inds = global_inds[pos_ind.astype(np.int)].astype(np.int)
                    try:
                        pred_box_dist = dist_infos[sample_token]['pred_dists'][pos_inds]
                    except:
                        pred_box_dist = np.linalg.norm(pred_matrix[pos_ind, :2], axis=1)
                else:
                    pred_box_dist = np.linalg.norm(pred_matrix[pos_ind, :2], axis=1)
                det_score = det_score[pos_ind]
            else:
                if cfg.dist_type == 'pts_dist':
                    try:
                        pred_box_dist = dist_infos[sample_token]['pred_dists'][global_inds.astype(np.int)]
                    except:
                        pred_box_dist = np.linalg.norm(pred_matrix[:, :2], axis=1)
                else:
                    pred_box_dist = np.linalg.norm(pred_matrix[:, :2], axis=1)
            
            fp_min = pred_box_dist >= distance[0]
            fp_max = pred_box_dist < distance[1]
            num_fps = pred_box_dist[np.where((fp_min & fp_max) == True)].shape[0]
            det_score = det_score[np.where((fp_min & fp_max) == True)]
            if num_fps > 0:
                det_index = det_score / 0.0001
                det_index = det_index.astype(int)
                for i, index in enumerate(det_index):
                    fp[index] += 1
        else:
            gt_taken = []  # Initially no gt bounding box is matched.
            for gt_idx in range(gt_matrix.shape[0]):
                gt_box = gt_matrix[gt_idx]
                min_metric_lid = np.inf
                min_metric_cam1 = np.inf
                min_metric_cam2 = np.inf
                match_pred_idx_lid = None
                match_pred_idx_cam1 = None
                match_pred_idx_cam2 = None
                match_pred_box_lid = None
                match_pred_box_cam1 = None
                match_pred_box_cam2 = None
                for pred_idx in range(pred_matrix.shape[0]):
                    global_ind = global_inds[pred_idx]
                    if sample_token in pred_taken.keys() and global_ind in pred_taken[sample_token]:
                        continue
                    pred_box = pred_matrix[pred_idx]
                    if sensor == 'cam':
                        if class_name in ['bigMot', 'smallMot', 'Mot']:
                            # if distance[-1] == rel_ab_th:
                            if cfg.iou_type == 'pts_iou':
                                this_metric_cam1 = iou_metric[0, int(gt_box[7]), int(global_ind)]
                            else:
                                this_metric_cam1 = iou_metric[1, pred_idx, gt_idx] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0, pred_idx, gt_idx]
                            this_metric_cam2 = dist_metric[1, pred_idx, gt_idx]
                        else:
                            this_metric_cam1 = dist_metric[0, pred_idx, gt_idx]
                            this_metric_cam2 = dist_metric[1, pred_idx, gt_idx]

                        if this_metric_cam1 < min_metric_cam1:
                            min_metric_cam1 = this_metric_cam1
                            match_pred_idx_cam1 = pred_idx
                            match_pred_box_cam1 = pred_box.copy()

                        if this_metric_cam2 < min_metric_cam2:
                            min_metric_cam2 = this_metric_cam2
                            match_pred_idx_cam2 = pred_idx
                            match_pred_box_cam2 = pred_box.copy()
                    else:
                        if cfg.iou_type == 'pts_iou':
                            this_metric_lid = iou_metric[0, int(gt_box[7]), int(global_ind)]
                        else:
                            this_metric_lid = iou_metric[1, pred_idx, gt_idx] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0, pred_idx, gt_idx]
                    
                        if this_metric_lid < min_metric_lid:
                            min_metric_lid = this_metric_lid 
                            match_pred_idx_lid = pred_idx
                            match_pred_box_lid = pred_box

                #If the closest match is close enough according to threshold we have a match!
                if sensor == 'cam':
                    if class_name in ['bigMot', 'smallMot', 'Mot']:
                        is_match_cam1 = min_metric_cam1 < 1 - cfg.iou_th_tp
                        is_match_cam2 = min_metric_cam2 < cfg.rel_dist_th_tp
                    else:
                        is_match_cam1 = min_metric_cam1 < cfg.ab_dist_th_tp
                        is_match_cam2 = min_metric_cam2 < cfg.rel_dist_th_tp  
                    match_process_cam(sample_token, pred_matrix, gt_idx, gt_box, is_match_cam1, is_match_cam2, match_pred_idx_cam1, match_pred_box_cam1, 
                                      match_pred_idx_cam2, match_pred_box_cam2, iou_metric, dist_metric, distance, cfg, gt_taken, pred_taken, 
                                      tp, fp, match_data, class_name, det_score, global_inds, inds_detname_dict, dist_infos)
                else:
                    is_match_lid = min_metric_lid < 1 - cfg.iou_th_tp  
                    match_process_lid(sample_token, pred_matrix, gt_idx, gt_box, is_match_lid, match_pred_idx_lid, match_pred_box_lid, 
                                      iou_metric, dist_metric, distance, cfg, gt_taken, pred_taken, tp, fp,match_data, 
                                      class_name, det_score, global_inds, inds_detname_dict, dist_infos)

            for pred_idx in range(pred_matrix.shape[0]):
                global_ind = global_inds[pred_idx]
                if cfg.iou_type == 'pts_iou' and det_mask[int(global_ind)] == False:
                    continue
                if sample_token in pred_taken.keys() and global_ind in pred_taken[sample_token]:
                    continue
                if cfg.cross_cls_match and inds_detname_dict[sample_token][global_ind] != class_name:
                    continue
                if cfg.cross_cls_match:
                    if sensor == 'cam':
                        if class_name in ['bigMot', 'smallMot', 'Mot']:
                            if distance[-1] == cfg.rel_ab_th:
                                if cfg.iou_type == 'pts_iou':
                                    ori_iou_metric = 1-accumulate_dict['sample_token'][sample_token][class_name]['ori_iou_metric']
                                    best_match_indx = list(ori_iou_metric.argmin(1))
                                    best_match_value = list(ori_iou_metric.min(1))
                                else:
                                    ori_iou_metric = iou_metric[1] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0]
                                    best_match_local_indx = list(ori_iou_metric.argmin(0))
                                    best_match_value = list(ori_iou_metric.min(0))
                                    best_match_indx = [global_inds[i] for i in best_match_local_indx]
                                threshold = 1 - cfg.iou_th_tp
                            else:
                                ori_dist_metric = dist_metric[1]
                                best_match_local_indx = list(ori_dist_metric.argmin(0))
                                best_match_value = list(ori_dist_metric.min(0))
                                best_match_indx = [global_inds[i] for i in best_match_local_indx]
                                threshold = cfg.rel_dist_th_tp
                        else:
                            if distance[-1] == cfg.rel_ab_th:
                                ori_dist_metric = dist_metric[0]
                                best_match_local_indx = list(ori_dist_metric.argmin(0))
                                best_match_value = list(ori_dist_metric.min(0))
                                best_match_indx = [global_inds[i] for i in best_match_local_indx]
                                threshold = cfg.ab_dist_th_tp
                            else:
                                ori_dist_metric = dist_metric[1]
                                best_match_local_indx = list(ori_dist_metric.argmin(0))
                                best_match_value = list(ori_dist_metric.min(0))
                                best_match_indx = [global_inds[i] for i in best_match_local_indx]
                                threshold = cfg.rel_dist_th_tp
                    else:
                        if cfg.iou_type == 'pts_iou':
                            ori_iou_metric = 1-accumulate_dict['sample_token'][sample_token][class_name]['ori_iou_metric']
                            best_match_indx = list(ori_iou_metric.argmin(1))
                            best_match_value = list(ori_iou_metric.min(1))
                        else:
                            ori_iou_metric = iou_metric[1] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0]
                            best_match_local_indx = list(ori_iou_metric.argmin(0))
                            best_match_value = list(ori_iou_metric.min(0))
                            best_match_indx = [global_inds[i] for i in best_match_local_indx]
                        threshold = 1-cfg.iou_th_tp
                
                    if global_ind in best_match_indx and best_match_value[best_match_indx.index(global_ind)] <= threshold:
                        continue

                pred_box = pred_matrix[pred_idx]

                if cfg.dist_type == 'pts_dist':
                    pred_box_dist = dist_infos[sample_token]['pred_dists'][int(global_ind)]
                else:
                    pred_box_dist = np.linalg.norm(pred_box[:2])
                    
                if pred_box_dist>= distance[0] and pred_box_dist < distance[1]:
                    det_index = (det_score[pred_idx]/ 0.0001).astype(int)
                    fp[det_index] += 1
    
    pred_num = sum(tp)+sum(fp)
    if pred_num == 0 and npos == 0:
        return DetectionMetricData.no_gt_predictions()
    elif npos == 0:
        return DetectionMetricData.no_gts(fp)
    else:
        if len(match_data['trans_err']) == 0:
            return DetectionMetricData.no_pos_predictions(npos, fp)
        # ---------------------------------------------
        # Calculate and interpolate precision and recall
        # ---------------------------------------------
        # Accumulate.
        cls_recall = sum(tp)/npos

        ap = 0
        tp_ = np.cumsum(tp[::-1])[::-1]
        fp_ = np.cumsum(fp[::-1])[::-1]

        current_recall = 0.0
        recall_dim = 101
        recall_step = 1.0 / (recall_dim - 1)

        precision_list = [] 
        recall_list = []
        for i in range(len(tp_)-1, -1, -1):
            total_det_num = tp_[i] + fp_[i]
            total_tp_num = tp_[i]
            left_recall =total_tp_num / npos
            right_recall = 0.0

            if i > 0:
                right_recall = tp_[i - 1] / npos
            else:
                right_recall = left_recall
        
            if (right_recall - current_recall) < (current_recall - left_recall) and i > 0:
                continue
            precision = total_tp_num/ total_det_num if total_det_num > 0 else 1.0
            current_recall += recall_step
            ap += precision

            precision_list.append(precision)
            recall_list.append(current_recall)

        ap /= recall_dim

        # ---------------------------------------------
        # Re-sample the match-data to match, prec, recall and conf.
        # ---------------------------------------------

        for key in match_data.keys():
            if key == "conf" or key =='npos':
                continue  # Confidence is used as reference to align with fp and tp. So skip in this step.
            
            else:
                match_data[key] = np.array(match_data[key])

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(ap=ap,
                               recall=recall_list,
                               precision=precision_list, 
                               trans_err=match_data['trans_err'],
                               iou3d_err=match_data['iou3d_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               ioubev_err=match_data['ioubev_err'],
                               confusion_matrix = 0, # not used
                               cls_recall = cls_recall,
                               pred_num = pred_num,
                               tp = np.array(tp),
                               fp = np.array(fp), 
                               npos = match_data['npos'])

def match_process_cam(sample_token: str, 
                      pred_matrix: np.array,
                      gt_idx: int, 
                      gt_box: np.array, 
                      is_match_cam1: bool, 
                      is_match_cam2: bool, 
                      match_pred_idx_cam1: int, 
                      match_pred_box_cam1: np.array, 
                      match_pred_idx_cam2: int, 
                      match_pred_box_cam2: np.array, 
                      iou_metric: np.array, 
                      dist_metric: np.array, 
                      distance: list, 
                      cfg: DetectionConfig, 
                      gt_taken: list, 
                      pred_taken: dict,
                      tp: list, 
                      fp: list, 
                      match_data:dict, 
                      class_name: str,
                      det_score: np.array,
                      global_inds: np.array,
                      inds_detname_dict: dict,
                      dist_infos: dict):
    match_pred_idx = None
    if is_match_cam1 and is_match_cam2:
        if cfg.dist_type == 'pts_dist':
            matched_gt_dist_cam1 = dist_infos[sample_token]['gt_dists'][int(gt_box[7])]
            matched_gt_dist_cam2 = dist_infos[sample_token]['gt_dists'][int(gt_box[7])]
        else:
            matched_gt_dist_cam1 = np.linalg.norm(np.array(gt_box[:2]))
            matched_gt_dist_cam2 = np.linalg.norm(np.array(gt_box[:2]))

        if (matched_gt_dist_cam1 <= cfg.rel_ab_th and matched_gt_dist_cam2 > cfg.rel_ab_th)\
            or (matched_gt_dist_cam1 > cfg.rel_ab_th and matched_gt_dist_cam2 <= cfg.rel_ab_th):
            ab_dist1 = dist_metric[0, match_pred_idx_cam1, gt_idx]
            ab_dist2 = dist_metric[0, match_pred_idx_cam2, gt_idx]
            if ab_dist1 <= ab_dist2:
                match_pred_idx = match_pred_idx_cam1
                match_pred_box = match_pred_box_cam1
            else:
                match_pred_idx = match_pred_idx_cam2
                match_pred_box = match_pred_box_cam2

            gt_taken.append(gt_idx)
            if sample_token in pred_taken.keys():
                pred_taken[sample_token].append(global_inds[match_pred_idx])
            else:
                pred_taken.update({sample_token: [global_inds[match_pred_idx]]})

            det_index = (det_score[match_pred_idx] / 0.0001).astype(int)
            tp[det_index]+=1

            # Since it is a match, update match data also.
            match_data['trans_err'].append(dist_metric[0, match_pred_idx, gt_idx])
            match_data['scale_err'].append(1 - scale_iou(gt_box[:7], match_pred_box[:7]))
            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box[6], match_pred_box[6], period=period))

            if cfg.iou_type == 'pts_iou':
                match_data['iou3d_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
                match_data['ioubev_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
            else:
                match_data['iou3d_err'].append(iou_metric[0, match_pred_idx, gt_idx])
                match_data['ioubev_err'].append(iou_metric[1, match_pred_idx, gt_idx])

    if match_pred_idx is None:
        if is_match_cam1:
            if cfg.dist_type == 'pts_dist':
                matched_gt_dist_cam1 = dist_infos[sample_token]['gt_dists'][int(gt_box[7])]
            else:
                matched_gt_dist_cam1 = np.linalg.norm(np.array(gt_box[:2])) 
                
            if matched_gt_dist_cam1 <= cfg.rel_ab_th:
                match_pred_idx = match_pred_idx_cam1
                match_pred_box = match_pred_box_cam1

                gt_taken.append(gt_idx)
                if sample_token in pred_taken.keys():
                    pred_taken[sample_token].append(global_inds[match_pred_idx])
                else:
                    pred_taken.update({sample_token: [global_inds[match_pred_idx]]})

                det_index = (det_score[match_pred_idx] / 0.0001).astype(int)
                tp[det_index]+=1

                # Since it is a match, update match data also.
                match_data['trans_err'].append(dist_metric[0, match_pred_idx, gt_idx])
                match_data['scale_err'].append(1 - scale_iou(gt_box[:7], match_pred_box[:7]))

                # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
                period = np.pi if class_name == 'barrier' else 2 * np.pi
                match_data['orient_err'].append(yaw_diff(gt_box[6], match_pred_box[6], period=period))

                if cfg.iou_type == 'pts_iou':
                    match_data['iou3d_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
                    match_data['ioubev_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
                else:
                    match_data['iou3d_err'].append(iou_metric[0, match_pred_idx, gt_idx])
                    match_data['ioubev_err'].append(iou_metric[1, match_pred_idx, gt_idx])

        if is_match_cam2:  
            if cfg.dist_type == 'pts_dist':
                matched_gt_dist_cam2 = dist_infos[sample_token]['gt_dists'][int(gt_box[7])]
            else:
                matched_gt_dist_cam2 = np.linalg.norm(np.array(gt_box[:2])) 

            if matched_gt_dist_cam2 > cfg.rel_ab_th and matched_gt_dist_cam2 <= distance[1]:
                match_pred_idx = match_pred_idx_cam2
                match_pred_box = match_pred_box_cam2

                gt_taken.append(gt_idx)

                if sample_token in pred_taken.keys():
                    pred_taken[sample_token].append(global_inds[match_pred_idx])
                else:
                    pred_taken.update({sample_token: [global_inds[match_pred_idx]]})

                det_index = (det_score[match_pred_idx] / 0.0001).astype(int)
                tp[det_index]+=1

                # Since it is a match, update match data also.
                match_data['trans_err'].append(dist_metric[0, match_pred_idx, gt_idx])
                match_data['scale_err'].append(1 - scale_iou(gt_box[:7], match_pred_box[:7]))

                # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
                period = np.pi if class_name == 'barrier' else 2 * np.pi
                match_data['orient_err'].append(yaw_diff(gt_box[6], match_pred_box[6], period=period))

                if cfg.iou_type == 'pts_iou':
                    match_data['iou3d_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
                    match_data['ioubev_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
                else:
                    match_data['iou3d_err'].append(iou_metric[0, match_pred_idx, gt_idx])
                    match_data['ioubev_err'].append(iou_metric[1, match_pred_idx, gt_idx])

def match_process_lid(sample_token: str,
                      pred_matrix: np.array,   
                      gt_idx: int, 
                      gt_box: np.array, 
                      is_match_lid: bool, 
                      match_pred_idx: int, 
                      match_pred_box: np.array, 
                      iou_metric: np.array, 
                      dist_metric: np.array, 
                      distance: list, 
                      cfg: DetectionConfig, 
                      gt_taken: list, 
                      pred_taken: dict, 
                      tp: list, 
                      fp: list, 
                      match_data:dict, 
                      class_name: str,
                      det_score: np.array, 
                      global_inds: np.array,
                      inds_detname_dict: dict,
                      dist_infos: dict):
                      
    if is_match_lid:
        if cfg.dist_type == 'pts_dist':
            matched_gt_dist = dist_infos[sample_token]['gt_dists'][int(gt_box[7])]
        else:
            matched_gt_dist = np.linalg.norm(np.array(gt_box[:2])) 

        if matched_gt_dist >= distance[0] and matched_gt_dist < distance[1]:
            gt_taken.append(gt_idx)
            if sample_token in pred_taken.keys():
                pred_taken[sample_token].append(global_inds[match_pred_idx])
            else:
                pred_taken.update({sample_token: [global_inds[match_pred_idx]]})

            det_index = (det_score[match_pred_idx] / 0.0001).astype(int)
            tp[det_index]+=1

            # Since it is a match, update match data also.
            match_data['trans_err'].append(dist_metric[0, match_pred_idx, gt_idx])
            match_data['scale_err'].append(1 - scale_iou(gt_box[:7], match_pred_box[:7]))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box[6], match_pred_box[6], period=period))

            if cfg.iou_type == 'pts_iou':
                match_data['iou3d_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
                match_data['ioubev_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
            else:
                match_data['iou3d_err'].append(iou_metric[0, match_pred_idx, gt_idx])
                match_data['ioubev_err'].append(iou_metric[1, match_pred_idx, gt_idx])


def binary_accumulate(sensor: str,
                      gt_boxes: EvalBoxes,
                      pred_boxes: EvalBoxes,
                      binary_accumulate_dict: dict,
                      class_inds: dict, 
                      inds_class: dict,
                      cfg: DetectionConfig, 
                      distance: list,
                      pred_taken: dict, 
                      verbose: bool = False):

    if cfg.dist_type == 'pts_dist':
        assert cfg.pts_dist_path is not None
        dist_infos = pickle.load(open(cfg.pts_dist_path, 'rb'), encoding='iso-8859-1') 
    else:
        dist_infos = None

    if cfg.ignore_gt_valid:
        if cfg.dist_type == 'pts_dist':
            npos = len([1 for gt_box in gt_boxes.all if  \
                        dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]>=distance[0] and\
                        dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]<=distance[1]])
        else:
            npos = len([1 for gt_box in gt_boxes.all if gt_box.ego_dist>=distance[0] and gt_box.ego_dist<=distance[1]])
    else:
        if cfg.dist_type == 'pts_dist':
            npos = len([1 for gt_box in gt_boxes.all if gt_box.valid == True and \
                         dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]>=distance[0] and \
                         dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]<=distance[1]])
        else:
            npos = len([1 for gt_box in gt_boxes.all if gt_box.valid == True and gt_box.ego_dist>=distance[0] and gt_box.ego_dist<=distance[1]])

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all]

    if binary_accumulate_dict['is_finished'] == False:
        if cfg.iou_type == 'pts_iou':
            binary_accumulate_dict = pts_iou_match_bi(npos, binary_accumulate_dict, gt_boxes, 
                                                      pred_boxes_list, class_inds, cfg)
        else:
            binary_accumulate_dict = geo_iou_match_bi(npos, binary_accumulate_dict, gt_boxes, 
                                                      pred_boxes_list, class_inds, cfg)

    # Do the actual matching.
    tp = [0]*10001  # Accumulator of true positives.
    fp = [0]*10001  # Accumulator of false positives.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'iou3d_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'ioubev_err': [],
                  'confusion_matrix': np.zeros([len(class_inds), len(class_inds)]),
                  'npos': npos}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------
    for sample_token in binary_accumulate_dict['sample_token'].keys():
        pred_matrix  = binary_accumulate_dict['sample_token'][sample_token]['pred_matrix'][:,:7]
        gt_matrix    = binary_accumulate_dict['sample_token'][sample_token]['gt_matrix']
        iou_metric   = binary_accumulate_dict['sample_token'][sample_token]['iou_metric']
        dist_metric  = binary_accumulate_dict['sample_token'][sample_token]['dist_metric']
        det_score    = binary_accumulate_dict['sample_token'][sample_token]['pred_matrix'][:,7] #已排序
        det_class_id = binary_accumulate_dict['sample_token'][sample_token]['pred_matrix'][:,8]
        global_inds  = binary_accumulate_dict['sample_token'][sample_token]['pred_matrix'][:,9] 
        det_mask     = binary_accumulate_dict['sample_token'][sample_token]['det_mask']

        if len(iou_metric.shape) < 3:  #该区间没有GT
            if cfg.dist_type == 'pts_dist':
                try:
                    pred_box_dist = dist_infos[sample_token]['pred_dists'][global_inds.astype(np.int)]
                except:
                    pred_box_dist = np.linalg.norm(pred_matrix[:, :2], axis=1)
            else:
                pred_box_dist = np.linalg.norm(pred_matrix[:, :2], axis=1)
            fp_min = pred_box_dist >= distance[0]
            fp_max = pred_box_dist < distance[1]
            num_fps = pred_box_dist[np.where((fp_min & fp_max) == True)].shape[0]
            det_score = det_score[np.where((fp_min & fp_max) == True)]

            if num_fps > 0:
                det_index = det_score / 0.0001
                det_index = det_index.astype(int)
                for index in det_index:
                    fp[index] += 1
        else:
            gt_class_id = gt_matrix[:, 8]
            gt_matrix = gt_matrix[:, :-1]

            if distance[-1] == cfg.rel_ab_th:
                binary_match_tpfps(sample_token, tp, fp, match_data, pred_matrix, gt_matrix, 
                            sensor, distance, cfg, iou_metric, dist_metric, det_score, det_class_id, 
                            gt_class_id, class_inds, inds_class, global_inds, pred_taken, ['bigMot', 'smallMot', 'Mot'],
                            dist_infos, det_mask, binary_accumulate_dict)
                binary_match_tpfps(sample_token, tp, fp, match_data, pred_matrix, gt_matrix, 
                            sensor, distance, cfg, iou_metric, dist_metric, det_score, det_class_id, 
                            gt_class_id, class_inds, inds_class, global_inds, pred_taken, ['nonMot', 'pedestrian', 'TrainedOthers'],
                            dist_infos, det_mask, binary_accumulate_dict)
            else:
                binary_match_tpfps(sample_token, tp, fp, match_data, pred_matrix, gt_matrix, 
                            sensor, distance, cfg, iou_metric, dist_metric, det_score, det_class_id, 
                            gt_class_id, class_inds, inds_class, global_inds, pred_taken,
                            ['Mot', 'bigMot', 'smallMot', 'nonMot', 'pedestrian', 'TrainedOthers'],
                             dist_infos, det_mask, binary_accumulate_dict)
        
    pred_num = sum(tp)+sum(fp)
    if pred_num == 0 and npos == 0:
        return DetectionMetricData.no_gt_predictions()
    elif npos == 0:
        return DetectionMetricData.no_gts(fp)
    else:
        if len(match_data['trans_err']) == 0:
            return DetectionMetricData.no_pos_predictions(npos, fp)

        ap = 0
        tp_ = np.cumsum(tp[::-1])[::-1]
        fp_ = np.cumsum(fp[::-1])[::-1]
        current_recall = 0.0
        recall_dim = 101
        recall_step = 1.0 / (recall_dim - 1)

        precision_list = []
        recall_list = []
        for i in range(len(tp_)-1, -1, -1):
            total_det_num = tp_[i] + fp_[i]
            total_tp_num = tp_[i]
            left_recall =total_tp_num / npos
            right_recall = 0.0

            if i > 0:
                right_recall = tp_[i - 1] / npos
            else:
                right_recall = left_recall
        
            if (right_recall - current_recall) < (current_recall - left_recall) and i > 0:
                continue
            precision = total_tp_num/ total_det_num if total_det_num > 0 else 1.0
            current_recall += recall_step
            ap += precision

            precision_list.append(precision)
            recall_list.append(current_recall)

        ap /= recall_dim
        # ---------------------------------------------
        # Re-sample the match-data to match, prec, recall and conf.
        # ---------------------------------------------

        for key in match_data.keys():
            if key =='npos':
                continue  # Confidence is used as reference to align with fp and tp. So skip in this step.
            else:
                match_data[key] = np.array(match_data[key])

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(ap=ap,
                               recall=recall_list,
                               precision=precision_list,
                               trans_err=match_data['trans_err'],
                               iou3d_err=match_data['iou3d_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               ioubev_err=match_data['ioubev_err'],
                               confusion_matrix = match_data['confusion_matrix'],
                               cls_recall = 0,  #not used
                               pred_num = pred_num,
                               tp = np.array(tp),
                               fp = np.array(fp), 
                               npos = match_data['npos'])

def binary_accumulate_all(sensor: str,
                          gt_boxes: EvalBoxes,
                          pred_boxes: EvalBoxes,
                          binary_accumulate_dict: dict,
                          class_inds: dict, 
                          inds_class: dict,
                          cfg: DetectionConfig, 
                          distance: list,
                          pred_taken: dict, 
                          verbose: bool = False):

    if cfg.dist_type == 'pts_dist':
        assert cfg.pts_dist_path is not None
        dist_infos = pickle.load(open(cfg.pts_dist_path, 'rb'), encoding='iso-8859-1') 
    else:
        dist_infos = None
    
    if cfg.ignore_gt_valid:
        if cfg.dist_type == 'pts_dist':
            npos = len([1 for gt_box in gt_boxes.all if  \
                        dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]>=distance[0] and\
                        dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]<=distance[1]])
        else:
            npos = len([1 for gt_box in gt_boxes.all if gt_box.ego_dist>=distance[0] and gt_box.ego_dist<=distance[1]])
    else:
        if cfg.dist_type == 'pts_dist':
            npos = len([1 for gt_box in gt_boxes.all if gt_box.valid == True and \
                         dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]>=distance[0] and \
                         dist_infos[gt_box.sample_token]['gt_dists'][[int(gt_box.index)]]<=distance[1]])
        else:
            npos = len([1 for gt_box in gt_boxes.all if gt_box.valid == True and gt_box.ego_dist>=distance[0] and gt_box.ego_dist<=distance[1]])

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all]

    if binary_accumulate_dict['is_finished'] == False:
        if cfg.iou_type == 'pts_iou':
            binary_accumulate_dict = pts_iou_match_bi(npos, binary_accumulate_dict, gt_boxes, 
                                                      pred_boxes_list, class_inds, cfg)
        else:
            binary_accumulate_dict = geo_iou_match_bi(npos, binary_accumulate_dict, gt_boxes, 
                                                      pred_boxes_list, class_inds, cfg)

    # Do the actual matching.
    tp = [0]*10001  # Accumulator of true positives.
    fp = [0]*10001  # Accumulator of false positives.
    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'iou3d_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'ioubev_err': [],
                  'confusion_matrix': np.zeros([len(class_inds), len(class_inds)]),
                  'npos': npos}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------
    for sample_token in binary_accumulate_dict['sample_token'].keys():
        pred_matrix  = binary_accumulate_dict['sample_token'][sample_token]['pred_matrix'][:,:7]
        gt_matrix    = binary_accumulate_dict['sample_token'][sample_token]['gt_matrix']
        iou_metric   = binary_accumulate_dict['sample_token'][sample_token]['iou_metric']
        dist_metric  = binary_accumulate_dict['sample_token'][sample_token]['dist_metric']
        det_score    = binary_accumulate_dict['sample_token'][sample_token]['pred_matrix'][:,7] #已排序
        det_class_id = binary_accumulate_dict['sample_token'][sample_token]['pred_matrix'][:,8]
        global_inds  = binary_accumulate_dict['sample_token'][sample_token]['pred_matrix'][:,9] 
        det_mask     = binary_accumulate_dict['sample_token'][sample_token]['det_mask']
        
        if len(iou_metric.shape) < 3:  #该区间没有GT
            if cfg.dist_type == 'pts_dist':
                try:
                    pred_box_dist = dist_infos[sample_token]['pred_dists'][global_inds.astype(np.int)]
                except:
                    pred_box_dist = np.linalg.norm(pred_matrix[:, :2], axis=1)
            else:
                pred_box_dist = np.linalg.norm(pred_matrix[:, :2], axis=1)
            fp_min = pred_box_dist >= distance[0]
            fp_max = pred_box_dist < distance[1]
            num_fps = pred_box_dist[np.where((fp_min & fp_max) == True)].shape[0]
            det_score = det_score[np.where((fp_min & fp_max) == True)]

            if num_fps > 0:
                det_index = det_score / 0.0001
                det_index = det_index.astype(int)
                for index in det_index:
                    fp[index] += 1
        else:
            gt_taken = []  # Initially no gt bounding box is matched.
            gt_class_id  = gt_matrix[:,8]
            gt_matrix = gt_matrix[:, :-1]

            for gt_idx in range(gt_matrix.shape[0]):
                gt_box = gt_matrix[gt_idx]
                min_metric_lid = np.inf
                min_metric_abd = np.inf
                min_metric_iou = np.inf
                match_pred_idx_lid = None
                match_pred_idx_abd = None
                match_pred_idx_iou = None
                match_pred_box_lid = None
                match_pred_box_abd = None
                match_pred_box_iou = None
                for pred_idx in range(pred_matrix.shape[0]):
                    global_ind = global_inds[pred_idx]
                    if sample_token in pred_taken.keys() and global_ind in pred_taken[sample_token]:
                        continue
                    pred_box = pred_matrix[pred_idx]
                    if sensor == 'cam':
                        this_metric_abd = dist_metric[0, pred_idx, gt_idx]
                        if cfg.iou_type == 'pts_iou':
                            this_metric_iou = iou_metric[0, int(gt_box[7]), int(global_ind)]
                        else:
                            this_metric_iou = iou_metric[1, pred_idx, gt_idx] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0, pred_idx, gt_idx]

                        if this_metric_abd < min_metric_abd:
                            min_metric_abd = this_metric_abd
                            match_pred_idx_abd = pred_idx
                            match_pred_box_abd = pred_box.copy()

                        if this_metric_iou < min_metric_iou:
                            min_metric_iou = this_metric_iou
                            match_pred_idx_iou = pred_idx
                            match_pred_box_iou = pred_box.copy()
                    else:
                        if cfg.iou_type == 'pts_iou':
                            this_metric_lid = iou_metric[0, int(gt_box[7]), int(global_ind)]
                        else:
                            this_metric_lid = iou_metric[1, pred_idx, gt_idx] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0, pred_idx, gt_idx]
                    
                        if this_metric_lid < min_metric_lid:
                            min_metric_lid = this_metric_lid 
                            match_pred_idx_lid = pred_idx
                            match_pred_box_lid = pred_box
                    
                if sensor == 'cam':
                    is_match_iou = min_metric_iou < 1
                    is_match = False
                    if is_match_iou:
                        is_match = binary_match_tpfps_all_iou(sample_token, min_metric_iou, match_pred_idx_iou, gt_idx, 
                                                            match_pred_box_iou, gt_box, tp, fp, match_data, distance, 
                                                            cfg, iou_metric, dist_metric, det_score, gt_taken, pred_taken, 
                                                            det_class_id, gt_class_id, inds_class, global_inds, dist_infos,
                                                            det_mask)
                    if is_match is False:
                        binary_match_tpfps_all_abd(sample_token, min_metric_abd, match_pred_idx_abd, gt_idx, match_pred_box_abd, 
                                                gt_box, tp, fp, match_data, distance, cfg, iou_metric, dist_metric, det_score, gt_taken,
                                                pred_taken, det_class_id, gt_class_id, inds_class, global_inds, dist_infos,
                                                det_mask)
                else:
                    is_match = min_metric_lid < 1 - cfg.iou_th_tp
                    binary_match_tpfps_all_lid(sample_token, min_metric_lid, match_pred_idx_lid, gt_idx, match_pred_box_lid, 
                                            gt_box, tp, fp, match_data, distance, cfg, iou_metric, 
                                            dist_metric, det_score, gt_taken, pred_taken, det_class_id, gt_class_id, 
                                            inds_class, is_match, global_inds, dist_infos, det_mask)
            
            for pred_idx in range(pred_matrix.shape[0]):
                global_ind = global_inds[pred_idx]
                if sample_token in pred_taken.keys() and global_ind in pred_taken[sample_token]:
                    continue
                if cfg.iou_type == 'pts_iou' and det_mask[int(global_ind)] == False:
                    continue

                pred_box = pred_matrix[pred_idx]

                if cfg.dist_type == 'pts_dist':
                    pred_box_dist = dist_infos[sample_token]['pred_dists'][int(global_ind)]
                else:
                    pred_box_dist = np.linalg.norm(pred_box[:2])
                    
                if pred_box_dist>= distance[0] and pred_box_dist < distance[1]:
                    det_index = (det_score[pred_idx]/ 0.0001).astype(int)
                    fp[det_index] += 1

    pred_num = sum(tp)+sum(fp)
    if pred_num == 0 and npos == 0:
        return DetectionMetricData.no_gt_predictions()
    elif npos == 0:
        return DetectionMetricData.no_gts(fp)
    else:
        if len(match_data['trans_err']) == 0:
            return DetectionMetricData.no_pos_predictions(npos, fp)

        ap = 0
        tp_ = np.cumsum(tp[::-1])[::-1]
        fp_ = np.cumsum(fp[::-1])[::-1]
        current_recall = 0.0
        recall_dim = 101
        confidence_dim = 10001
        recall_step = 1.0 / (recall_dim - 1)
  
        precision_list = []
        recall_list = []

        for i in range(len(tp_)-1, -1, -1):
            total_det_num = tp_[i] + fp_[i]
            total_tp_num = tp_[i]
            left_recall =total_tp_num / npos
            right_recall = 0.0

            if i > 0:
                right_recall = tp_[i - 1] / npos
            else:
                right_recall = left_recall
        
            if (right_recall - current_recall) < (current_recall - left_recall) and i > 0:
                continue
            precision = total_tp_num/ total_det_num if total_det_num > 0 else 1.0
            current_recall += recall_step
            ap += precision
            precision_list.append(precision)
            recall_list.append(current_recall)

        ap /= recall_dim

        # ---------------------------------------------
        # Re-sample the match-data to match, prec, recall and conf.
        # ---------------------------------------------

        for key in match_data.keys():
            if key =='npos':
                continue  # Confidence is used as reference to align with fp and tp. So skip in this step.
            
            else:
                match_data[key] = np.array(match_data[key])

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(ap=ap,
                               recall=recall_list, 
                               precision=precision_list, 
                               trans_err=match_data['trans_err'],
                               iou3d_err=match_data['iou3d_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               ioubev_err=match_data['ioubev_err'],
                               confusion_matrix = match_data['confusion_matrix'],
                               cls_recall = 0,  #not used
                               pred_num = pred_num,
                               tp = np.array(tp),
                               fp = np.array(fp), 
                               npos = match_data['npos'])


def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1
    if md.npos == -1:
        return -1
    else:
        prec = np.copy(md.precision)
        prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
        prec -= min_precision  # Clip low precision
        prec[prec < 0] = 0
        return float(np.mean(prec)) / (1.0 - min_precision)

def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """
    if md.npos == -1:
        error_infos = {'mean_err': -1, 'p90': -1, 'p99': -1, 'pos_1std': -1, 'pos_2std': -1}
    else:
        sorted_metric_data = sorted(getattr(md, metric_name))

        error_infos = {}
        mean_error = float(np.mean(sorted_metric_data))
        p90 = float(sorted_metric_data[int(len(sorted_metric_data)*0.9)])
        p99 = float(sorted_metric_data[int(len(sorted_metric_data)*0.99)])

        pos_1std = float(sorted_metric_data[int(len(sorted_metric_data)*0.68268949)])
        pos_2std = float(sorted_metric_data[int(len(sorted_metric_data)*0.95449974)])

        error_infos = {'mean_err': mean_error, 'p90': p90, 'p99': p99, 'pos_1std': pos_1std, 'pos_2std': pos_2std}
    return error_infos  # +1 to include error at max recall.

def binary_match_tpfps(sample_token: str,
                        tp: list, 
                        fp: list, 
                        match_data: dict, 
                        pred_matrix: np.array, 
                        gt_matrix: np.array, 
                        sensor: str, 
                        distance: list, 
                        cfg: DetectionConfig,
                        iou_metric: np.array, 
                        dist_metric: np.array, 
                        det_score: np.array, 
                        det_class_id: np.array, 
                        gt_class_id: np.array, 
                        class_inds: dict, 
                        inds_class: dict, 
                        global_inds: np.array,
                        pred_taken: dict, 
                        match_type: list,
                        dist_infos: dict,
                        det_mask: np.array,
                        binary_accumulate_dict: dict):

    gt_taken = []  # Initially no gt bounding box is matched.

    for gt_idx in range(gt_matrix.shape[0]):
        gt_box = gt_matrix[gt_idx]
        min_metric = np.inf
        match_pred_idx = None
        match_pred_box = None
        gt_class_name = inds_class[gt_class_id[gt_idx]]
        for pred_idx in range(pred_matrix.shape[0]):
            global_ind = global_inds[pred_idx]
            if sample_token in pred_taken.keys() and global_ind in pred_taken[sample_token]:
                continue
            pred_box = pred_matrix[pred_idx]
            det_class_name = inds_class[det_class_id[pred_idx]]
            if det_class_name not in match_type:
                continue
            if gt_class_name in match_type:
                if sensor == 'cam':
                    if det_class_name in ['bigMot', 'smallMot', 'Mot']:
                        if distance[-1] == cfg.rel_ab_th:
                            if cfg.iou_type == 'pts_iou':
                                this_metric = iou_metric[0, int(gt_box[7]), int(global_ind)]
                            else:
                                this_metric = iou_metric[1, pred_idx, gt_idx] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0, pred_idx, gt_idx]
                        else:
                            this_metric = dist_metric[1, pred_idx, gt_idx]
                    else:
                        if distance[-1] == cfg.rel_ab_th:
                            this_metric = dist_metric[0, pred_idx, gt_idx]
                        else:
                            this_metric = dist_metric[1, pred_idx, gt_idx]
                else:
                    if cfg.iou_type == 'pts_iou':
                        this_metric = iou_metric[0, int(gt_box[7]), int(global_ind)]
                    else:
                        this_metric = iou_metric[1, pred_idx, gt_idx] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0, pred_idx, gt_idx]
        
                if this_metric < min_metric:
                    min_metric = this_metric
                    match_pred_idx = pred_idx
                    match_pred_box = pred_box

        # If the closest match is close enough according to threshold we have a match!
        if sensor == 'cam':
            if match_type == ['bigMot', 'smallMot', 'Mot']:
                if distance[-1] == cfg.rel_ab_th:
                    is_match = min_metric < 1 - cfg.iou_th_tp
                else:
                    is_match = min_metric < cfg.rel_dist_th_tp
            else:
                if distance[-1] == cfg.rel_ab_th:
                    is_match = min_metric < cfg.ab_dist_th_tp
                else:
                    is_match = min_metric < cfg.rel_dist_th_tp  
        else:
            is_match = min_metric < 1 - cfg.iou_th_tp   

        if is_match:
            det_index = (det_score[match_pred_idx] / 0.0001).astype(int)
            if cfg.dist_type == 'pts_dist':
                matched_gt_dist = dist_infos[sample_token]['gt_dists'][int(gt_box[7])]
            else:
                matched_gt_dist = np.linalg.norm(np.array(gt_box[:2])) 
            if matched_gt_dist >= distance[0] and matched_gt_dist < distance[1]:
                gt_taken.append(gt_idx)
                if sample_token in pred_taken.keys():
                    pred_taken[sample_token].append(global_inds[match_pred_idx])
                else:
                    pred_taken.update({sample_token: [global_inds[match_pred_idx]]})

                tp[det_index]+=1

                # Since it is a match, update match data also.
                match_data['trans_err'].append(dist_metric[0, match_pred_idx, gt_idx])
                match_data['scale_err'].append(1 - scale_iou(gt_box[:7], match_pred_box[:7]))
                # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
                period = 2*np.pi 
                match_data['orient_err'].append(yaw_diff(gt_box[6], match_pred_box[6], period=period))
                match_data['confusion_matrix'][int(det_class_id[pred_idx]), int(gt_class_id[gt_idx])] += 1

                if cfg.iou_type == 'pts_iou':
                    match_data['iou3d_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
                    match_data['ioubev_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
                else:
                    match_data['iou3d_err'].append(iou_metric[0, match_pred_idx, gt_idx])
                    match_data['ioubev_err'].append(iou_metric[1, match_pred_idx, gt_idx])

    for pred_idx in range(pred_matrix.shape[0]):
        global_ind = global_inds[pred_idx]
        if cfg.iou_type == 'pts_iou' and det_mask[int(global_ind)] == False:
            continue
        if sample_token in pred_taken.keys() and global_ind in pred_taken[sample_token]:
            continue
        if match_type == ['bigMot', 'smallMot', 'Mot']:
            continue
        if sensor == 'cam':
            if gt_class_name in ['bigMot', 'smallMot', 'Mot']:
                if distance[-1] == cfg.rel_ab_th:
                    if cfg.iou_type == 'pts_iou':
                        ori_iou_metric = 1-binary_accumulate_dict['sample_token'][sample_token]['ori_iou_metric']
                        best_match_indx = list(ori_iou_metric.argmin(1))
                        best_match_value = list(ori_iou_metric.min(1))
                    else:
                        ori_iou_metric = iou_metric[1] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0]
                        best_match_local_indx = list(ori_iou_metric.argmin(0))
                        best_match_value = list(ori_iou_metric.min(0))
                        best_match_indx = [global_inds[i] for i in best_match_local_indx]
                    threshold = 1 - cfg.iou_th_tp
                else:
                    ori_dist_metric = dist_metric[1]
                    best_match_local_indx = list(ori_dist_metric.argmin(0))
                    best_match_value = list(ori_dist_metric.min(0))
                    best_match_indx = [global_inds[i] for i in best_match_local_indx]
                    threshold = cfg.rel_dist_th_tp
            else:
                if distance[-1] == cfg.rel_ab_th:
                    ori_dist_metric = dist_metric[0]
                    best_match_local_indx = list(ori_dist_metric.argmin(0))
                    best_match_value = list(ori_dist_metric.min(0))
                    best_match_indx = [global_inds[i] for i in best_match_local_indx]
                    threshold = cfg.ab_dist_th_tp
                else:
                    ori_dist_metric = dist_metric[1]
                    best_match_local_indx = list(ori_dist_metric.argmin(0))
                    best_match_value = list(ori_dist_metric.min(0))
                    best_match_indx = [global_inds[i] for i in best_match_local_indx]
                    threshold = cfg.rel_dist_th_tp
        else:
            if cfg.iou_type == 'pts_iou':
                ori_iou_metric = 1-binary_accumulate_dict['sample_token'][sample_token]['ori_iou_metric']
                best_match_indx = list(ori_iou_metric.argmin(1))
                best_match_value = list(ori_iou_metric.min(1))
            else:
                ori_iou_metric = iou_metric[1] if cfg.geo_iou_type == 'bev_iou' else iou_metric[0]
                best_match_local_indx = list(ori_iou_metric.argmin(0))
                best_match_value = list(ori_iou_metric.min(0))
                best_match_indx = [global_inds[i] for i in best_match_local_indx]
            threshold = 1-cfg.iou_th_tp
        
        if global_ind in best_match_indx and best_match_value[best_match_indx.index(global_ind)] <= threshold:
            continue
        
        pred_box = pred_matrix[pred_idx]

        if cfg.dist_type == 'pts_dist':
            pred_box_dist = dist_infos[sample_token]['pred_dists'][int(global_ind)]
        else:
            pred_box_dist = np.linalg.norm(pred_box[:2])
            
        if pred_box_dist>= distance[0] and pred_box_dist < distance[1]:
            det_index = (det_score[pred_idx]/ 0.0001).astype(int)
            fp[det_index] += 1
    

def binary_match_tpfps_all_iou(sample_token, min_metric, match_pred_idx, gt_idx, match_pred_box, gt_box, 
                               tp, fp, match_data, distance, cfg, iou_metric, dist_metric, det_score, gt_taken,
                               pred_taken, det_class_id, gt_class_id, inds_class, global_inds, dist_infos, det_mask):
    if cfg.dist_type == 'pts_dist':
        matched_gt_dist = dist_infos[sample_token]['gt_dists'][int(gt_box[7])]
    else:
        matched_gt_dist = np.linalg.norm(np.array(gt_box[:2])) 
    if matched_gt_dist <= cfg.rel_ab_th:
        if inds_class[gt_class_id[gt_idx]] in ['bigMot', 'smallMot', 'Mot']:
            is_match = min_metric < 1 - cfg.iou_th_tp
        else:
            is_match = dist_metric[0, match_pred_idx, gt_idx] < cfg.ab_dist_th_tp \
                                          if match_pred_idx is not None else False
    elif matched_gt_dist <= distance[1]:
        is_match = dist_metric[1, match_pred_idx, gt_idx] < cfg.rel_dist_th_tp \
                                          if match_pred_idx is not None else False
    else:
        is_match = False

    det_index = (det_score[match_pred_idx] / 0.0001).astype(int)

    if is_match:
        gt_taken.append(gt_idx)
        if sample_token in pred_taken.keys():
            pred_taken[sample_token].append(global_inds[match_pred_idx])
        else:
            pred_taken.update({sample_token: [global_inds[match_pred_idx]]})

        #  Update tp, fp and confs.
        tp[det_index]+=1

        # Since it is a match, update match data also.
        match_data['trans_err'].append(dist_metric[0, match_pred_idx, gt_idx])
        match_data['scale_err'].append(1 - scale_iou(gt_box[:7], match_pred_box[:7 ]))

        # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
        period = 2*np.pi 
        match_data['orient_err'].append(yaw_diff(gt_box[6], match_pred_box[6], period=period))
        match_data['confusion_matrix'][int(det_class_id[match_pred_idx]), int(gt_class_id[gt_idx])] += 1

        if cfg.iou_type == 'pts_iou':
            match_data['iou3d_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
            match_data['ioubev_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
        else:
            match_data['iou3d_err'].append(iou_metric[0, match_pred_idx, gt_idx])
            match_data['ioubev_err'].append(iou_metric[1, match_pred_idx, gt_idx])
    return is_match

def binary_match_tpfps_all_abd(sample_token, min_metric, match_pred_idx, gt_idx, match_pred_box, gt_box, 
                               tp, fp, match_data, distance, cfg, iou_metric, dist_metric, det_score, gt_taken,
                               pred_taken, det_class_id, gt_class_id, inds_class, global_inds, dist_infos, det_mask):
    if cfg.dist_type == 'pts_dist':
        matched_gt_dist = dist_infos[sample_token]['gt_dists'][int(gt_box[7])]
    else:
        matched_gt_dist = np.linalg.norm(np.array(gt_box[:2])) 
    
    det_index = (det_score[match_pred_idx] / 0.0001).astype(int)
    if matched_gt_dist <= cfg.rel_ab_th:
        if inds_class[gt_class_id[gt_idx]] in ['bigMot', 'smallMot', 'Mot']:
            is_match = min_metric < 1 - cfg.iou_th_tp
        else:
            is_match = dist_metric[0, match_pred_idx, gt_idx] < cfg.ab_dist_th_tp \
                                        if match_pred_idx is not None else False

        if is_match:
            gt_taken.append(gt_idx)
            if sample_token in pred_taken.keys():
                pred_taken[sample_token].append(global_inds[match_pred_idx])
            else:
                pred_taken.update({sample_token: [global_inds[match_pred_idx]]})
            #  Update tp, fp and confs.
            tp[det_index]+=1

            # Since it is a match, update match data also.
            match_data['trans_err'].append(dist_metric[0, match_pred_idx, gt_idx])
            match_data['scale_err'].append(1 - scale_iou(gt_box[:7], match_pred_box[:7]))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = 2*np.pi 
            match_data['orient_err'].append(yaw_diff(gt_box[6], match_pred_box[6], period=period))
            match_data['confusion_matrix'][int(det_class_id[match_pred_idx]), int(gt_class_id[gt_idx])] += 1
            if cfg.iou_type == 'pts_iou':
                match_data['iou3d_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
                match_data['ioubev_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
            else:
                match_data['iou3d_err'].append(iou_metric[0, match_pred_idx, gt_idx])
                match_data['ioubev_err'].append(iou_metric[1, match_pred_idx, gt_idx])
                
    elif matched_gt_dist <= distance[1]:
        is_match = dist_metric[1, match_pred_idx, gt_idx] < cfg.rel_dist_th_tp \
                                     if match_pred_idx is not None else False
        if is_match:
            gt_taken.append(gt_idx)
            if sample_token in pred_taken.keys():
                pred_taken[sample_token].append(global_inds[match_pred_idx])
            else:
                pred_taken.update({sample_token: [global_inds[match_pred_idx]]})
            #  Update tp, fp and confs.
            tp[det_index]+=1

            # Since it is a match, update match data also.
            match_data['trans_err'].append(dist_metric[0, match_pred_idx, gt_idx])
            match_data['scale_err'].append(1 - scale_iou(gt_box[:7], match_pred_box[:7]))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = 2*np.pi 
            match_data['orient_err'].append(yaw_diff(gt_box[6], match_pred_box[6], period=period))
            match_data['confusion_matrix'][int(det_class_id[match_pred_idx]), int(gt_class_id[gt_idx])] += 1
            if cfg.iou_type == 'pts_iou':
                match_data['iou3d_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
                match_data['ioubev_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
            else:
                match_data['iou3d_err'].append(iou_metric[0, match_pred_idx, gt_idx])
                match_data['ioubev_err'].append(iou_metric[1, match_pred_idx, gt_idx])

def binary_match_tpfps_all_lid(sample_token, min_metric, match_pred_idx, gt_idx, match_pred_box, gt_box, 
                               tp, fp, match_data, distance, cfg, iou_metric, dist_metric, det_score, gt_taken,
                               pred_taken, det_class_id, gt_class_id, inds_class, is_match, global_inds, dist_infos, det_mask):
    det_index = (det_score[match_pred_idx] / 0.0001).astype(int)
    if is_match:
        if cfg.dist_type == 'pts_dist':
            matched_gt_dist = dist_infos[sample_token]['gt_dists'][int(gt_box[7])]
        else:
            matched_gt_dist = np.linalg.norm(np.array(gt_box[:2])) 
        if matched_gt_dist >= distance[0] and matched_gt_dist < distance[1]:
            gt_taken.append(gt_idx)
            if sample_token in pred_taken.keys():
                pred_taken[sample_token].append(global_inds[match_pred_idx])
            else:
                pred_taken.update({sample_token: [global_inds[match_pred_idx]]})

            tp[det_index]+=1

            # Since it is a match, update match data also.
            match_data['trans_err'].append(dist_metric[0, match_pred_idx, gt_idx])
            match_data['scale_err'].append(1 - scale_iou(gt_box[:7], match_pred_box[:7]))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = 2*np.pi 
            match_data['orient_err'].append(yaw_diff(gt_box[6], match_pred_box[6], period=period))
            match_data['confusion_matrix'][int(det_class_id[match_pred_idx]), int(gt_class_id[gt_idx])] += 1
            if cfg.iou_type == 'pts_iou':
                match_data['iou3d_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
                match_data['ioubev_err'].append(iou_metric[0, int(gt_box[7]), int(global_inds[match_pred_idx])])
            else:
                match_data['iou3d_err'].append(iou_metric[0, match_pred_idx, gt_idx])
                match_data['ioubev_err'].append(iou_metric[1, match_pred_idx, gt_idx])
