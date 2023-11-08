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
# Code written by Oscar Beijbom, 2019.

from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np

from projects.nuscenes.eval.common.data_classes import MetricData, EvalBox
from projects.nuscenes.eval.common.utils import center_distance, rel_dist_distance
from projects.nuscenes.eval.detection.constants import DETECTION_NAMES, ATTRIBUTE_NAMES, TP_METRICS


class DetectionConfig:
    """ Data class that specifies the detection evaluation settings. """

    def __init__(self,
                 class_range: Dict[str, int],
                 distances: List[int],
                 rel_dist_fcn: str,
                 rel_dist_th_tp: float,
                 ab_dist_th_tp: float,
                 iou_th_hard_tp:float,
                 iou_th_easy_tp:float,
                 rel_ab_th:float,
                 iou_type: str,
                 iou_matrix_path: str,
                 geo_iou_type: str,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: int,
                 ignore_gt_valid: bool,
                 score_thr: float,
                 fuse_mot: bool,
                 cross_cls_match: bool,
                 score_first: bool,
                 dist_type: str,
                 pts_dist_path: str):
        for key in class_range.keys():
            assert key in DETECTION_NAMES, "Class mismatch."

        self.class_range = class_range
        self.distances = distances
        self.rel_dist_th_tp = rel_dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.ab_dist_th_tp = ab_dist_th_tp
        self.iou_th_hard_tp = iou_th_hard_tp
        self.iou_th_easy_tp = iou_th_easy_tp
        self.rel_ab_th = rel_ab_th
        self.iou_type = iou_type
        self.iou_matrix_path = iou_matrix_path
        self.geo_iou_type = geo_iou_type
        self.rel_dist_fcn = rel_dist_fcn
        self.ignore_gt_valid = ignore_gt_valid
        self.score_thr = score_thr
        self.fuse_mot = fuse_mot
        self.cross_cls_match = cross_cls_match
        self.score_first = score_first
        self.dist_type = dist_type
        self.pts_dist_path = pts_dist_path

        self.class_names = self.class_range.keys()

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'class_range': self.class_range,
            'distances': self.distances,
            'rel_dist_fcn': self.rel_dist_fcn,
            'rel_dist_th_tp': self.rel_dist_th_tp,
            'ab_dist_th_tp': self.ab_dist_th_tp,
            'iou_th_hard_tp': self.iou_th_hard_tp,
            'iou_th_easy_tp': self.iou_th_easy_tp,
            'rel_ab_th': self.rel_ab_th,
            'iou_type' : self.iou_type,
            'iou_matrix_path': self.iou_matrix_path,
            'geo_iou_type': self.geo_iou_type,
            'min_recall': self.min_recall,
            'min_precision': self.min_precision,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'ignore_gt_valid': self.ignore_gt_valid,
            'score_thr': self.score_thr,
            'fuse_mot': self.fuse_mot,
            'cross_cls_match': self.cross_cls_match,
            'score_first': self.score_first, 
            'dist_type': self.dist_type,
            'pts_dist_path': self.pts_dist_path

        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """
        return cls(content['class_range'],
                   content['distances'],
                   content['rel_dist_fcn'],
                   content['rel_dist_th_tp'],
                   content['ab_dist_th_tp'], 
                   content['iou_th_hard_tp'],
                   content['iou_th_easy_tp'],
                   content['rel_ab_th'],
                   content['iou_type'],
                   content['iou_matrix_path'],
                   content['geo_iou_type'],
                   content['min_recall'],
                   content['min_precision'],
                   content['max_boxes_per_sample'],
                   content['ignore_gt_valid'],
                   content['score_thr'],
                   content['fuse_mot'],
                   content['cross_cls_match'],
                   content['score_first'], 
                   content['dist_type'],
                   content['pts_dist_path'])

    @property
    def dist_fcn_callable(self):
        """ Return the distance function corresponding to the dist_fcn string. """
        if self.dist_fcn == 'center_distance':
            return center_distance
        else:
            raise Exception('Error: Unknown distance function %s!' % self.dist_fcn)

    @property
    def rel_dist_fcn_callable(self):
        """ Return the distance function corresponding to the dist_fcn string. """
        if self.rel_dist_fcn == 'relative_center_distance':
            return rel_dist_distance
        else:
            raise Exception('Error: Unknown iou function %s!' % self.rel_dist_fcn)


class DetectionMetricData(MetricData):
    """ This class holds accumulated and interpolated data required to calculate the detection metrics. """

    nelem = 101

    def __init__(self,
                 ap: np.float,
                 recall: np.array,
                 precision: np.array,
                 trans_err: np.array,
                 iou3d_err: np.array,
                 scale_err: np.array,
                 orient_err: np.array,
                 ioubev_err: np.array,
                 cls_recall: float,
                 confusion_matrix: np.array,
                 pred_num: int,
                 tp: list,
                 fp: list,
                 npos: int):

        # Set attributes explicitly to help IDEs figure out what is going on.
        self.ap = ap
        self.recall = recall
        self.precision = precision
        self.trans_err = trans_err
        self.iou3d_err = iou3d_err
        self.scale_err = scale_err
        self.orient_err = orient_err
        self.ioubev_err = ioubev_err
        self.cls_recall = cls_recall
        self.confusion_matrix = confusion_matrix
        self.npos = npos
        self.pred_num = pred_num
        self.tp = tp 
        self.fp = fp

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    @property
    def max_recall_ind(self):
        """ Returns index of max recall achieved. """
        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(self.confidence)[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        return max_recall_ind

    @property
    def max_recall(self):
        """ Returns max recall achieved. """

        return self.recall[self.max_recall_ind]

    def serialize(self):
        """ Serialize instance into json-friendly format. """
        if isinstance(self.trans_err, np.ndarray):
            self.trans_err =  self.trans_err.tolist()
            self.iou3d_err =  self.iou3d_err.tolist()
            self.scale_err = self.scale_err.tolist()
            self.orient_err =  self.orient_err.tolist()
            self.ioubev_err = self.ioubev_err.tolist()
        return {
            'ap': self.ap,
            'recall': self.recall,
            'precision': self.precision,
            'trans_err': self.trans_err,
            'iou3d_err': self.iou3d_err,
            'scale_err': self.scale_err,
            'orient_err': self.orient_err,
            'ioubev_err': self.ioubev_err
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(ap = content['ap'],
                   recall=np.array(content['recall']),
                   precision=np.array(content['precision']),
                   trans_err=np.array(content['trans_err']),
                   iou3d_err=np.array(content['iou3d_err']),
                   scale_err=np.array(content['scale_err']),
                   orient_err=np.array(content['orient_err']),
                   ioubev_err=np.array(content['ioubev_err']),
                   cls_recall=float(content['cls_recall']),
                   confusion_matrix=np.array(content['confusion_matrix']),
                   pred_num=int(content['pred_num']),
                   tp=list(content['tp']),
                   fp=list(content['fp']),
                   npos=int(content['npos']))

    @classmethod
    def no_predictions(cls,npos):
        """ Returns a md instance corresponding to having no predictions. """
        return cls(ap=0, 
                   recall=np.linspace(0, 1, cls.nelem),
                   precision=np.zeros(cls.nelem),
                   trans_err=np.ones(cls.nelem),
                   iou3d_err=np.ones(cls.nelem),
                   scale_err=np.ones(cls.nelem),
                   orient_err=np.ones(cls.nelem),
                   ioubev_err=np.ones(cls.nelem),
                   confusion_matrix=0,
                   cls_recall = 0,
                   pred_num = 0, 
                   tp = np.zeros(10001),
                   fp = np.zeros(10001),
                   npos=npos)

    @classmethod
    def no_gt_predictions(cls):
        """ Returns a md instance corresponding to having no predictions and gts. """
        return cls(ap=-1, 
                   recall=np.linspace(0, 1, cls.nelem),
                   precision=np.zeros(cls.nelem),
                   trans_err=np.ones(cls.nelem),
                   iou3d_err=np.ones(cls.nelem),
                   scale_err=np.ones(cls.nelem),
                   orient_err=np.ones(cls.nelem),
                   ioubev_err=np.ones(cls.nelem),
                   confusion_matrix=-1,
                   cls_recall=-1,
                   pred_num = 0,
                   tp = np.zeros(10001),
                   fp = np.zeros(10001),
                   npos=-1)

    @classmethod
    def no_pos_predictions(cls, npos, fp):
        """ Returns a md instance corresponding to having no predictions and gts. """
        return cls(ap=0, 
                   recall=np.linspace(0, 1, cls.nelem),
                   precision=np.zeros(cls.nelem),
                   trans_err=np.ones(cls.nelem),
                   iou3d_err=np.ones(cls.nelem),
                   scale_err=np.ones(cls.nelem),
                   orient_err=np.ones(cls.nelem),
                   ioubev_err=np.ones(cls.nelem),
                   confusion_matrix=-1,
                   cls_recall=0,
                   pred_num = sum(fp),
                   tp = np.zeros(10001),
                   fp = np.array(fp),
                   npos=npos)

    @classmethod
    def no_gts(cls, fp):
        """ Returns a md instance corresponding to having no predictions and gts. """
        return cls(ap=0, 
                   recall=np.linspace(0, 1, cls.nelem),
                   precision=np.zeros(cls.nelem),
                   trans_err=np.ones(cls.nelem),
                   iou3d_err=np.ones(cls.nelem),
                   scale_err=np.ones(cls.nelem),
                   orient_err=np.ones(cls.nelem),
                   ioubev_err=np.ones(cls.nelem),
                   confusion_matrix=-1,
                   cls_recall=0,
                   pred_num = sum(fp),
                   tp = np.zeros(10001),
                   fp = np.array(fp),
                   npos=0)

class DetectionMetrics:
    """ Stores average precision and true positive metric results. Provides properties to summarize. """

    def __init__(self, cfg: DetectionConfig):

        self.cfg = cfg
        self._label_aps = defaultdict(lambda: defaultdict(float))
        self._label_tp_errors = defaultdict(lambda: defaultdict(float))
        self.eval_time = None

    def add_label_ap(self, detection_name: str, dist_th: float, ap: float) -> None:
        self._label_aps[detection_name][dist_th] = ap

    def pop_label_ap(self, detection_name: str) -> None:
        return self._label_aps.pop(detection_name)

    def get_label_ap(self, detection_name: str, dist_th: float) -> float:
        return self._label_aps[detection_name][dist_th]

    def add_label_tp(self, detection_name: str, metric_name: str, tp: float):
        self._label_tp_errors[detection_name][metric_name] = tp

    def get_label_tp(self, detection_name: str, metric_name: str) -> float:
        return self._label_tp_errors[detection_name][metric_name]

    def pop_label_tp(self, detection_name: str) -> float:
        return self._label_tp_errors.pop(detection_name)

    def add_runtime(self, eval_time: float) -> None:
        self.eval_time = eval_time

    @property
    def mean_dist_aps(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_aps.items()}

    @property
    def mean_ap(self) -> float:
        """ Calculates the mean AP by averaging over distance thresholds and classes. """
        ap = 0
        overall_class = self.pop_label_ap('overall_class')
        tp = overall_class['tp_list'][0]
        fp = overall_class['fp_list'][0]
        npos = overall_class['tp_list'][1]
        if npos == 0:
            return -1
        tp_ = np.cumsum(tp[::-1])[::-1]
        fp_ = np.cumsum(fp[::-1])[::-1]
        current_recall = 0.0
        recall_dim = 101
        recall_step = 1.0 / (recall_dim - 1)
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

        ap /= recall_dim
        return ap

    @property
    def tp_errors(self) -> Dict[str, float]:
        """ Calculates the mean true positive error across all classes for each metric. """
        errors = {}
        for metric_name in TP_METRICS:
            class_errors = []
            for detection_name in self.cfg.class_names:
                err = self.get_label_tp(detection_name, metric_name)
                if err != -1:
                    class_errors.append(err)
            errors[metric_name] = float(np.nanmean(class_errors))  if len(class_errors) > 0 else -1

        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        scores = {}
        tp_errors = self.tp_errors
        for metric_name in TP_METRICS:

            # We convert the true positive errors to "scores" by 1-error.
            score = 1.0 - tp_errors[metric_name]

            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)

            scores[metric_name] = score

        return scores

    @property
    def nd_score(self) -> float:
        """
        Compute the nuScenes detection score (NDS, weighted sum of the individual scores).
        :return: The NDS.
        """
        # Summarize.
        total = float(self.cfg.mean_ap_weight * self.mean_ap + np.sum(list(self.tp_scores.values())))

        # Normalize.
        total = total / float(self.cfg.mean_ap_weight + len(self.tp_scores.keys()))

        return total

    def serialize(self):
        return {
            'label_aps': self._label_aps,
            'mean_ap': self.mean_ap,
            'mean_dist_aps': self.mean_dist_aps,
            'label_tp_errors': self._label_tp_errors,
            'tp_errors': self.tp_errors,
            'tp_scores': self.tp_scores,
            #'nd_score': self.nd_score,
            'eval_time': self.eval_time,
            'cfg': self.cfg.serialize()
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """

        cfg = DetectionConfig.deserialize(content['cfg'])

        metrics = cls(cfg=cfg)
        metrics.add_runtime(content['eval_time'])

        for detection_name, label_aps in content['label_aps'].items():
            for dist_th, ap in label_aps.items():
                metrics.add_label_ap(detection_name=detection_name, dist_th=float(dist_th), ap=float(ap))

        for detection_name, label_tps in content['label_tp_errors'].items():
            for metric_name, tp in label_tps.items():
                metrics.add_label_tp(detection_name=detection_name, metric_name=metric_name, tp=float(tp))

        return metrics

    def __eq__(self, other):
        eq = True
        eq = eq and self._label_aps == other._label_aps
        eq = eq and self._label_tp_errors == other._label_tp_errors
        eq = eq and self.eval_time == other.eval_time
        eq = eq and self.cfg == other.cfg

        return eq


class DetectionBox(EvalBox):
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 valid = True, 
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: [float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 detection_name: str = 'car',  # The class name used in the detection challenge.
                 detection_score: float = -1.0,  # GT samples do not have a score.
                 attribute_name: str = '', 
                 index = -1):  # Box attribute. Each box can have at most 1 attribute.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name

        assert attribute_name in ATTRIBUTE_NAMES or attribute_name == '', \
            'Error: Unknown attribute_name %s' % attribute_name

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name
        self.valid = valid
        self.index = index

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.valid == other.valid and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.detection_name == other.detection_name and
                self.detection_score == other.detection_score and
                self.attribute_name == other.attribute_name and 
                self.index == other.index)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'valid': self.valid,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name
        }

    @classmethod
    def deserialize(cls, content: dict, index: int):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   attribute_name=content['attribute_name'],
                   index=index)


class DetectionMetricDataList:
    """ This stores a set of MetricData in a dict indexed by (name, match-distance). """

    def __init__(self):
        self.md = {}

    def __getitem__(self, key):
        return self.md[key]

    def __eq__(self, other):
        eq = True
        for key in self.md.keys():
            eq = eq and self[key] == other[key]
        return eq

    def get_class_data(self, detection_name: str) -> List[Tuple[DetectionMetricData, float]]:
        """ Get all the MetricData entries for a certain detection_name. """
        return [(md, dist_th) for (name, dist_th), md in self.md.items() if name == detection_name]

    def get_dist_data(self, dist_th: float) -> List[Tuple[DetectionMetricData, str]]:
        """ Get all the MetricData entries for a certain match_distance. """
        return [(md, detection_name) for (detection_name, dist), md in self.md.items() if dist == dist_th]

    def set(self, detection_name: str, match_distance: float, data: DetectionMetricData):
        """ Sets the MetricData entry for a certain detection_name and match_distance. """
        self.md[(detection_name, match_distance)] = data

    def serialize(self) -> dict:
        #for key, value in self.md.items():
            #print("key: ", key)
            #print("value: ", value)
            #print("value.serialize(): ", value.serialize())
        return {key[0] + ':' + str(key[1]): value.serialize() for key, value in self.md.items()}

    @classmethod
    def deserialize(cls, content: dict):
        mdl = cls()
        for key, md in content.items():
            name, distance = key.split(':')
            mdl.set(name, float(distance), DetectionMetricData.deserialize(md))
        return mdl

