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
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np

from projects.nuscenes import NuScenes
from projects.nuscenes.eval.common.config import config_factory
from projects.nuscenes.eval.common.data_classes import EvalBoxes
from projects.nuscenes.eval.common.loaders import load_prediction, load_gt, load_my_gt, add_center_dist, \
     filter_eval_boxes, my_filter_eval_boxes
from projects.nuscenes.eval.detection import algo_score, algo_iou
from projects.nuscenes.eval.detection.constants import TP_METRICS
from projects.nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
     DetectionMetricDataList
from projects.nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample

def get_eval_distance(distances):
    """
    input:
        self.cfg.distances = [0,50,20]
    return:
        eval_distance = [[0, 20], [20, 40], [40, 50], [0, 50]]
    """
    start_dis, end_dis, inter =  distances[:]
    start_tmp = start_dis
    eval_distance = []
    num_interval = (end_dis - start_dis) // inter if (end_dis - start_dis) % inter == 0 else (end_dis - start_dis) // inter + 1
    for i in range(num_interval):
        if start_dis + inter <= end_dis:
            eval_distance.append([start_dis, start_dis + inter])
            start_dis += inter
        else:
            eval_distance.append([start_dis, end_dis])
    eval_distance.append([start_tmp, end_dis])
    return eval_distance


class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        consider_class_list = [i for i in self.cfg.class_range.keys()]
        self.pred_boxes, self.meta = load_prediction(self.result_path, consider_class_list, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     self.cfg.score_thr, self.cfg.fuse_mot, verbose=verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self, sensor = 'cam', relaxation = 'easy') -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()
        if relaxation == 'easy':
            self.cfg.iou_th_tp = self.cfg.iou_th_easy_tp
        else:
            self.cfg.iou_th_tp = self.cfg.iou_th_hard_tp

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')

        metric_data_list_distance = {}
        metrics_distance = {}
        configs_distance = {}
        #self.cfg.distances = [[0, 20], [20, 40], [40, 60], [0, 60]]
        if sensor == 'cam' and relaxation == 'hard':
            eval_distance = [self.cfg.distances[0]]
        else:
            eval_distance = self.cfg.distances  #get_eval_distance(self.cfg.distances) #self.cfg.distances = [0,50,20],

        assert eval_distance[0][-1] == self.cfg.rel_ab_th

        consider_class_list = [i for i in self.cfg.class_range.keys()]
        self.cfg.consider_class_list = consider_class_list
        
        pred_taken1 = {}
        pred_taken2 = {}
        for distance in eval_distance:
            configs = {}
            metric_data_list = DetectionMetricDataList()
            gt_sum_num = 0
            tp_sum_num = 0
            pred_sum_num = 0
            for class_name in self.cfg.class_names: #['bigMot', 'smallMot', 'nonMot', 'pedestrian', 'TrainedOthers']
                if self.cfg.score_first:
                    md = algo_score.accumulate(sensor, self.gt_boxes, self.pred_boxes, self.accumulate_dict,
                                    class_name, self.cfg, distance, pred_taken1)
                else:
                    md = algo_iou.accumulate(sensor, self.gt_boxes, self.pred_boxes, self.accumulate_dict,
                                    class_name, self.cfg, distance, pred_taken1)
                metric_data_list.set(class_name, self.cfg.rel_dist_th_tp, md)
                if md.npos > 0:
                    gt_sum_num = gt_sum_num + md.npos
                if not (md.trans_err == 1).all():
                    tp_sum_num = tp_sum_num + len(md.trans_err)
                if md.pred_num > 0: 
                    pred_sum_num = pred_sum_num + md.pred_num

            configs = {'pred': pred_sum_num, 'gt': gt_sum_num, 'recall': tp_sum_num}

            #binary class accumulate
            if self.cfg.score_first:
                md = algo_score.binary_accumulate(sensor, self.gt_boxes, self.pred_boxes, self.binary_accumulate_dict, 
                                    self.class_inds, self.inds_class, self.cfg, distance, pred_taken2)
            else:
                md = algo_iou.binary_accumulate(sensor, self.gt_boxes, self.pred_boxes, self.binary_accumulate_dict, 
                                    self.class_inds, self.inds_class, self.cfg, distance, pred_taken2)

            configs.update({'binary_pred': md.pred_num, 
                            'binary_recall': len(md.trans_err)
            })
            metric_data_list.set('binary_class', self.cfg.rel_dist_th_tp, md)
            
            self.accumulate_dict['is_finished'] =True
            self.binary_accumulate_dict['is_finished'] =True

            # -----------------------------------
            # Step 2: Calculate metrics from the data.
            # -----------------------------------
            if self.verbose:
                print('Calculating metrics...')
            metrics = DetectionMetrics(self.cfg)
            tp_list = np.zeros(10001) 
            fp_list = np.zeros(10001)
            npos_all = 0
            for class_name in self.cfg.class_names:
                # Compute APs.
                metric_data = metric_data_list[(class_name, self.cfg.rel_dist_th_tp)]
                metrics.add_label_ap(class_name, self.cfg.rel_dist_th_tp, metric_data.ap)
                if metric_data.npos != -1:  
                    npos_all += metric_data.npos
                    tp_list += metric_data.tp
                    fp_list += metric_data.fp

                # Compute TP metrics.
                for metric_name in TP_METRICS: 
                    metric_data = metric_data_list[(class_name, self.cfg.rel_dist_th_tp)]
                    error_infos = algo_score.calc_tp(metric_data, self.cfg.min_recall, metric_name)
                    metrics.add_label_tp(class_name, metric_name, error_infos['mean_err'])
                
                metrics.add_label_tp(class_name, 'cls_recall', metric_data.cls_recall)

            metrics.add_label_ap('overall_class', 'tp_list', [tp_list, npos_all])
            metrics.add_label_ap('overall_class', 'fp_list', [fp_list, npos_all])

            # Compute binary APs.
            metric_data = metric_data_list[('binary_class', self.cfg.rel_dist_th_tp)]
            metrics.add_label_ap('binary_class', self.cfg.rel_dist_th_tp, metric_data.ap)

            # Compute binary TP metrics.
            for metric_name in TP_METRICS: 
                metric_data = metric_data_list[('binary_class', self.cfg.rel_dist_th_tp)]
                error_infos = algo_score.calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp('binary_class', metric_name, error_infos['mean_err'])
            
            for cls_ind, class_name in enumerate(self.cfg.class_names):
                if type(metric_data.confusion_matrix) == type(0):
                     metrics.add_label_tp(class_name, 'cls_acc', metric_data.confusion_matrix)
                else:
                    row_sum = metric_data.confusion_matrix[cls_ind].sum()
                    col_sum = metric_data.confusion_matrix[:, cls_ind].sum()
                    numerator = metric_data.confusion_matrix[cls_ind, cls_ind]
                    denominator = row_sum + col_sum - numerator
                    accuracy = numerator / denominator
                    metrics.add_label_tp(class_name, 'cls_acc', accuracy)
            
            # Compute evaluation time.
            metrics.add_runtime(time.time() - start_time)
            
            metric_data_list_distance[str(distance[0]) + '--' + str(distance[1])] = metric_data_list
            metrics_distance[str(distance[0]) + '--' + str(distance[1])] = metrics
            configs_distance[str(distance[0]) + '--' + str(distance[1])] = configs

        # overall distance
        eval_distance = [[self.cfg.distances[0][0], self.cfg.distances[-1][-1]]] 
        for distance in eval_distance:
            configs = {}
            metric_data_list = DetectionMetricDataList()
            gt_sum_num = 0
            tp_sum_num = 0
            pred_sum_num = 0
            pred_taken1 = {}
            pred_taken2 = {}
            for class_name in self.cfg.class_names: #['bigMot', 'smallMot', 'nonMot', 'pedestrian', 'TrainedOthers']
                if self.cfg.score_first:
                    md = algo_score.accumulate_all(sensor, self.gt_boxes, self.pred_boxes, self.accumulate_dict, 
                                        class_name, self.cfg, distance, pred_taken1)
                else:
                    md = algo_iou.accumulate_all(sensor, self.gt_boxes, self.pred_boxes, self.accumulate_dict, 
                                        class_name, self.cfg, distance, pred_taken1)
                metric_data_list.set(class_name, self.cfg.rel_dist_th_tp, md)
                if md.npos > 0:
                    gt_sum_num = gt_sum_num + md.npos
                if not (md.trans_err == 1).all():
                    tp_sum_num = tp_sum_num + len(md.trans_err)
                if md.pred_num > 0:
                        pred_sum_num = pred_sum_num + md.pred_num
            
            configs = {'pred': pred_sum_num, 'gt': gt_sum_num, 'recall': tp_sum_num}

            #binary class accumulate
            if self.cfg.score_first:
                md = algo_score.binary_accumulate_all(sensor, self.gt_boxes, self.pred_boxes, self.binary_accumulate_dict, 
                                            self.class_inds, self.inds_class, self.cfg, distance, pred_taken2)
            else:
                md = algo_iou.binary_accumulate_all(sensor, self.gt_boxes, self.pred_boxes, self.binary_accumulate_dict, 
                                            self.class_inds, self.inds_class, self.cfg, distance, pred_taken2)

            metric_data_list.set('binary_class', self.cfg.rel_dist_th_tp, md)
            
            configs.update({'binary_pred': md.pred_num, 
                            'binary_recall': len(md.trans_err)
            })

            # -----------------------------------
            # Step 2: Calculate metrics from the data.
            # -----------------------------------
            if self.verbose:
                print('Calculating metrics...')
            metrics = DetectionMetrics(self.cfg)
            tp_list = np.zeros(10001) 
            fp_list = np.zeros(10001)
            npos_all = 0
            for class_name in self.cfg.class_names:
                # Compute APs.
                metric_data = metric_data_list[(class_name, self.cfg.rel_dist_th_tp)]
                metrics.add_label_ap(class_name, self.cfg.rel_dist_th_tp, metric_data.ap)
                
                if metric_data.npos != -1:  
                    npos_all += metric_data.npos
                    tp_list += metric_data.tp
                    fp_list += metric_data.fp

                # Compute TP metrics.
                for metric_name in TP_METRICS: 
                    metric_data = metric_data_list[(class_name, self.cfg.rel_dist_th_tp)]
                    error_infos = algo_score.calc_tp(metric_data, self.cfg.min_recall, metric_name)
                    metrics.add_label_tp(class_name, metric_name, error_infos['mean_err'])

                metrics.add_label_tp(class_name, 'cls_recall', metric_data.cls_recall)

            metrics.add_label_ap('overall_class', 'tp_list', [tp_list, npos_all])
            metrics.add_label_ap('overall_class', 'fp_list', [fp_list, npos_all])

            # Compute binary APs.
            metric_data = metric_data_list[('binary_class', self.cfg.rel_dist_th_tp)]
            metrics.add_label_ap('binary_class', self.cfg.rel_dist_th_tp, metric_data.ap)

            # Compute binary TP metrics.
            for metric_name in TP_METRICS: 
                metric_data = metric_data_list[('binary_class', self.cfg.rel_dist_th_tp)]
                error_infos = algo_score.calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp('binary_class', metric_name, error_infos['mean_err'])

            metrics.add_label_tp('binary_class', 'confusion_matrix', metric_data.confusion_matrix)
            for cls_ind, class_name in enumerate(self.cfg.class_names):
                if type(metric_data.confusion_matrix) == type(0):
                     metrics.add_label_tp(class_name, 'cls_acc', metric_data.confusion_matrix)
                else:
                    row_sum = metric_data.confusion_matrix[cls_ind].sum()
                    col_sum = metric_data.confusion_matrix[:, cls_ind].sum()
                    numerator = metric_data.confusion_matrix[cls_ind, cls_ind]
                    denominator = row_sum + col_sum - numerator
                    accuracy = numerator / denominator
                    metrics.add_label_tp(class_name, 'cls_acc', accuracy)

            # Compute evaluation time.
            metrics.add_runtime(time.time() - start_time)
            
            metric_data_list_distance[str(distance[0]) + '--' + str(distance[1])] = metric_data_list
            metrics_distance[str(distance[0]) + '--' + str(distance[1])] = metrics
            configs_distance[str(distance[0]) + '--' + str(distance[1])] = configs

        return metrics_distance, metric_data_list_distance, configs_distance

    def render(self, metrics: dict, md_list: dict) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        class_names = [ key for key in self.cfg.class_names]
        class_names.append('binary_class')

        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.rel_dist_th_tp, savepath=savepath('summary'), class_names = class_names)

        for detection_name in class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            # class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.rel_dist_th_tp,
            #                savepath=savepath(detection_name + '_tp'))

        for dist_inteval in metrics.keys():
            dist_pr_curve(md_list, metrics, dist_inteval, self.cfg.min_precision, self.cfg.min_recall,
                          class_names, savepath=savepath('dist_pr_' + str(dist_inteval)))

    def print_info(self, f_result, metrics_distance, metric_data_list_distance, configs_distance, key, sensor = 'cam', relaxation = 'easy', print_matrix = False):
        print("*********************** %s_%s ***********************" % (sensor, relaxation))
        f_result.write("*********************** %s_%s *********************** \n" % (sensor, relaxation))

        metrics = metrics_distance[key]
        metric_data_list = metric_data_list_distance[key]
        configs = configs_distance[key]

        print("pred: %d   gt: %d   recall: %d\n" %(configs['pred'], configs['gt'], configs['recall']))
        print("binary_pred: %d   binary_recall: %d\n" %(configs['binary_pred'], configs['binary_recall']))
        f_result.write("pred: %d   gt: %d   recall: %d\n" %(configs['pred'], configs['gt'], configs['recall']))
        f_result.write("binary_pred: %d   binary_recall: %d\n" %(configs['binary_pred'], configs['binary_recall']))
        f_result.write("\n")

        bap_vals = metrics.pop_label_ap('binary_class')
        btp_vals = metrics.pop_label_tp('binary_class')
     
        
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()

        metrics_summary['binary_mean_ap'] = bap_vals[self.cfg.rel_dist_th_tp]

       
        # with open(os.path.join(self.output_dir, 'metrics_summary_{}.json'.format(key)), 'w') as f:
        #     json.dump(metrics_summary, f, indent=2)
        
        # with open(os.path.join(self.output_dir, 'metrics_details_{}.json'.format(key)), 'w') as f:
        #     json.dump(metric_data_list.serialize(), f, indent=2)

        
        bap_val = bap_vals[self.cfg.rel_dist_th_tp]
        print('bmAP:       {:.4f}       mAP:       {:.4f}'.format(bap_val, metrics_summary['mean_ap']))
        f_result.write('bmAP:       {:.4f}       mAP:       {:.4f}\n'.format(bap_val, metrics_summary['mean_ap']))
            
        for tp_name in TP_METRICS:
            btp_val = btp_vals[tp_name]
            tp_val  = metrics_summary['tp_errors'][tp_name]
            print('b{}:{}{:.4f}       {}:{}{:.4f}'.format(self.err_name_mapping[tp_name], self.space_mapping[tp_name], btp_val,
                                              self.err_name_mapping[tp_name], self.space_mapping[tp_name], tp_val))
            
            f_result.write('b{}:{}{:.4f}       {}:{}{:.4f}\n'.format(self.err_name_mapping[tp_name], self.space_mapping[tp_name], btp_val,
                                              self.err_name_mapping[tp_name], self.space_mapping[tp_name], tp_val))

        if print_matrix == True:
            confusion_matrix = btp_vals['confusion_matrix']
            if type(confusion_matrix) == type(0):
                confusion_matrix = np.ones([len(self.cfg.class_range), len(self.cfg.class_range)])*confusion_matrix
            else:
                confusion_matrix_row = confusion_matrix/confusion_matrix.sum(1).reshape(confusion_matrix.shape[0],1)
                confusion_matrix_clo = confusion_matrix.T/confusion_matrix.T.sum(1).reshape(confusion_matrix.shape[0],1)

        print()
        f_result.write("\n")
        print('Eval time: %.1fs' % metrics_summary['eval_time'])
        f_result.write('Eval time: %.1fs \n' % metrics_summary['eval_time'])
        print()
        f_result.write("\n")

        # Print per-class metrics.
        print('Per-class results:')
        f_result.write('Per-class results: \n')
        print('Object Class\tAP\tATE\tASE\tAOE\tAiou3dE\t  AioubevE\tClsRecall\tClsAcc')
        f_result.write('Object Class\tAP\tATE\tASE\tAOE\tAiou3dE\t  AioubevE\tClsRecall\tClsAcc \n')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']

        for class_name in class_aps.keys():
            cls_recall = metrics.get_label_tp(class_name,'cls_recall')
            cls_acc = metrics.get_label_tp(class_name,'cls_acc')
            print('{}{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t  {:.3f}\t        {:.3f}\t        {:.3f}'.format(
                    class_name, self.space_mapping_class['print'][class_name], 
                    class_aps[class_name],
                    class_tps[class_name]['trans_err'],
                    class_tps[class_name]['scale_err'],
                    class_tps[class_name]['orient_err'],
                    class_tps[class_name]['iou3d_err'],
                    class_tps[class_name]['ioubev_err'],
                    cls_recall,
                    cls_acc))
            f_result.write('{}{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t  {:.3f}\t        {:.3f}\t        {:.3f}\n'.format(
                    class_name, self.space_mapping_class['write'][class_name], 
                    class_aps[class_name],
                    class_tps[class_name]['trans_err'],
                    class_tps[class_name]['scale_err'],
                    class_tps[class_name]['orient_err'],
                    class_tps[class_name]['iou3d_err'],
                    class_tps[class_name]['ioubev_err'],
                    cls_recall,
                    cls_acc))

        print()
        f_result.write("\n")

        if print_matrix == True:
            print('Classification confusion matrix (Row: prediction, Col: groundtruth)')
            f_result.write('Classification confusion matrix (Row: prediction, Col: groundtruth)\n')
            for cls_ind, class_name in enumerate(class_aps.keys()):
                print('{}:{}{}'.format(class_name, self.space_mapping_matrix[class_name], np.round(confusion_matrix_row[cls_ind], 4)))
                f_result.write('{}:{}{}\n'.format(class_name, self.space_mapping_matrix[class_name], np.round(confusion_matrix_row[cls_ind], 4)))
     
            print()
            f_result.write("\n")
       
            print('Classification confusion matrix (Row: groundtruth, Col: prediction)')
            f_result.write('Classification confusion matrix (Row: groundtruth, Col: prediction)\n')
            for cls_ind, class_name in enumerate(class_aps.keys()):
                print('{}:{}{}'.format(class_name, self.space_mapping_matrix[class_name], np.round(confusion_matrix_clo[cls_ind], 4)))
                f_result.write('{}:{}{}\n'.format(class_name, self.space_mapping_matrix[class_name], np.round(confusion_matrix_clo[cls_ind], 4)))
            print()
            f_result.write("\n")

    def main(self,
             render_curves: bool = False,
             lidar_metric: bool = False,
             camera_metric: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        # Run evaluation.
        if camera_metric:
            cam_metrics_distance_easy, cam_metric_data_list_distance_easy, cam_configs_distance_easy = self.evaluate(sensor = 'cam', relaxation = 'easy')
            cam_metrics_distance_hard, cam_metric_data_list_distance_hard, cam_configs_distance_hard = self.evaluate(sensor = 'cam', relaxation = 'hard')

            # Render PR and TP curves.
            if render_curves:
                #self.render(cam_metrics_distance_easy, cam_metric_data_list_distance_easy) 
                self.render(cam_metrics_distance_hard, cam_metric_data_list_distance_hard) 

            # Dump the metric data, meta and metrics to disk.
            if self.verbose:
                print('Saving metrics to: %s' % self.output_dir)
            
            f_result = open(os.path.join(self.output_dir, 'result.txt'), 'w')

            for dist_ind, key in enumerate(cam_metrics_distance_easy): # or lidar_metrics_distance_easy
                print("*********************** %s ***********************" % (key))
                f_result.write("*********************** %s *********************** \n" % (key))
                if dist_ind == 0:
                    self.print_info(f_result, cam_metrics_distance_easy, cam_metric_data_list_distance_easy, cam_configs_distance_easy, key, sensor='cam', relaxation='easy')
                    self.print_info(f_result, cam_metrics_distance_hard, cam_metric_data_list_distance_hard, cam_configs_distance_hard, key, sensor='cam', relaxation='hard')
                else:
                    self.print_info(f_result, cam_metrics_distance_easy, cam_metric_data_list_distance_easy, cam_configs_distance_easy, key, sensor='cam', relaxation='easy',
                                    print_matrix = True if dist_ind == len(cam_metrics_distance_easy)-1 else False)
            self.print_info(f_result, cam_metrics_distance_hard, cam_metric_data_list_distance_hard, cam_configs_distance_hard, key, sensor='cam', relaxation='hard', print_matrix = True)

        if lidar_metric:
            lidar_metrics_distance_easy, lidar_metric_data_list_distance_easy, lidar_configs_distance_easy = self.evaluate(sensor = 'lidar', relaxation = 'easy')
            lidar_metrics_distance_hard, lidar_metric_data_list_distance_hard, lidar_configs_distance_hard = self.evaluate(sensor = 'lidar', relaxation = 'hard')

            # Render PR and TP curves.
            if render_curves:
                #self.render(lidar_metrics_distance_easy, lidar_metric_data_list_distance_easy) 
                self.render(lidar_metrics_distance_hard, lidar_metric_data_list_distance_hard) 

            f_result = open(os.path.join(self.output_dir, 'result.txt'), 'w')

            for dist_ind, key in enumerate(lidar_metrics_distance_easy): # or lidar_metrics_distance_easy
                print("*********************** %s ***********************" % (key))
                f_result.write("*********************** %s *********************** \n" % (key))
                self.print_info(f_result, lidar_metrics_distance_easy, lidar_metric_data_list_distance_easy, lidar_configs_distance_easy, key, sensor='lidar', relaxation='easy',
                                print_matrix = True if dist_ind == len(lidar_metrics_distance_easy)-1 else False)
                self.print_info(f_result, lidar_metrics_distance_hard, lidar_metric_data_list_distance_hard, lidar_configs_distance_hard, key, sensor='lidar', relaxation='hard',
                                print_matrix = True if dist_ind == len(lidar_metrics_distance_easy)-1 else False)
        

class NuScenesEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """


class MyDataEval(DetectionEval):
    def __init__(self,
                 nusc,
                 result_path,
                 eval_set,
                 output_dir,
                 verbose,
                 config):
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        self.accumulate_dict = {'is_finished': False,
                                'sample_token': {}}
        self.binary_accumulate_dict = {'is_finished': False,
                                       'sample_token': {}}
        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        consider_class_list = [i for i in self.cfg.class_range.keys()]
        self.pred_boxes, self.meta = load_prediction(self.result_path, consider_class_list, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     self.cfg.score_thr, self.cfg.fuse_mot, verbose=verbose)
        self.gt_boxes = load_my_gt(self.eval_set, consider_class_list, DetectionBox, self.cfg.fuse_mot, verbose=verbose)
        # Filter boxes (distance, points per box, etc.).
        if self.cfg.fuse_mot:
            assert 'bigMot' in self.cfg.class_range.keys()
            self.cfg.class_range.update({'Mot': self.cfg.class_range['bigMot']})
            self.cfg.class_range.pop('smallMot')
            self.cfg.class_range.pop('bigMot')

        self.class_inds = {}
        self.inds_class = {}
        for ind, class_name in enumerate(self.cfg.class_range.keys()):
            self.class_inds.update({class_name: ind})
            self.inds_class.update({ind: class_name})

        if verbose:
            print('Filtering predictions')

        self.pred_boxes = my_filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = my_filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)
        self.sample_tokens = self.gt_boxes.sample_tokens

        self.err_name_mapping = {
                'trans_err': 'mATE',
                'scale_err': 'mASE',
                'orient_err': 'mAOE',
                'iou3d_err': 'mAiou3dE',
                'ioubev_err': 'mAioubevE',
                'cls_err': 'mACE',
            }

        self.space_mapping = {
            'iou3d_err': '  ',
            'ioubev_err': ' ',
            'trans_err': '      ',
            'scale_err': '      ',
            'orient_err': '      ',
            'cls_err': '      ',
        }

        self.space_mapping_class = {
            'print':{
                'Mot': '            ',
                'bigMot': '         ',
                'smallMot': '  ',
                'nonMot': '      ',
                'pedestrian': '   ',
                'TrainedOthers': '  '
                 },
            'write':{
                'Mot': '            ',
                'bigMot': '         ',
                'smallMot': '      ',
                'nonMot': '      ',
                'pedestrian': '   ',
                'TrainedOthers': '  '
                 }    
            }

        self.space_mapping_matrix = {
            'Mot': '            ',
            'bigMot': '         ',
            'smallMot': '       ',
            'nonMot': '         ',
            'pedestrian': '     ',
            'TrainedOthers': '  ',
        }        

if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
