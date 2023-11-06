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

__all__ = ["SamplingDatabase"]

import os.path as osp
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np

from paddle3d.apis import manager
from paddle3d.geometries.bbox import BBoxes3D, box_collision_test
from paddle3d.geometries.pointcloud import PointCloud
from paddle3d.sample import Sample
from paddle3d.transforms.base import TransformABC
from paddle3d.utils.logger import logger


@manager.TRANSFORMS.add_component
class SamplingDatabase(TransformABC):
    """
    Sample objects from ground truth database and paste on current scene.

    Args:
        min_num_points_in_box_per_class (Dict[str, int]): Minimum number of points in sampled object for each class.
        max_num_samples_per_class (Dict[str, int]): Maximum number of objects sampled from each class.
        database_anno_path (str): Path to database annotation file (.pkl).
        database_root (str): Path to database root directory.
        class_names (List[str]): List of class names.
        ignored_difficulty (List[int]): List of difficulty levels to be ignored.
    """

    def __init__(self,
                 min_num_points_in_box_per_class: Dict[str, int],
                 max_num_samples_per_class: Dict[str, int],
                 database_anno_path: str,
                 database_root: str,
                 class_names: List[str],
                 ignored_difficulty: List[int] = None):
        self.min_num_points_in_box_per_class = min_num_points_in_box_per_class
        self.max_num_samples_per_class = max_num_samples_per_class
        self.database_anno_path = database_anno_path
        with open(database_anno_path, "rb") as f:
            database_anno = pickle.load(f)
        if not osp.exists(database_root):
            raise ValueError(
                f"Database root path {database_root} does not exist!!!")
        self.database_root = database_root
        self.class_names = class_names
        self.database_anno = self._filter_min_num_points_in_box(database_anno)
        self.ignored_difficulty = ignored_difficulty
        if ignored_difficulty is not None:
            self.database_anno = self._filter_ignored_difficulty(
                self.database_anno)

        self.sampler_per_class = dict()
        for cls_name, annos in self.database_anno.items():
            self.sampler_per_class[cls_name] = Sampler(cls_name, annos)

    def _filter_min_num_points_in_box(self, database_anno: Dict[str, list]):
        new_database_anno = defaultdict(list)
        for cls_name, annos in database_anno.items():
            if cls_name not in self.class_names or cls_name not in self.min_num_points_in_box_per_class:
                continue
            logger.info("Load {} {} database infos".format(
                len(annos), cls_name))
            for anno in annos:
                if anno["num_points_in_box"] >= self.min_num_points_in_box_per_class[
                        cls_name]:
                    new_database_anno[cls_name].append(anno)
        logger.info("After filtering min_num_points_in_box:")
        for cls_name, annos in new_database_anno.items():
            logger.info("Load {} {} database infos".format(
                len(annos), cls_name))
        return new_database_anno

    def _filter_ignored_difficulty(self, database_anno: Dict[str, list]):
        new_database_anno = defaultdict(list)
        for cls_name, annos in database_anno.items():
            if cls_name not in self.class_names or cls_name not in self.min_num_points_in_box_per_class:
                continue
            for anno in annos:
                if anno["difficulty"] not in self.ignored_difficulty:
                    new_database_anno[cls_name].append(anno)
        logger.info("After filtering ignored difficulty:")
        for cls_name, annos in new_database_anno.items():
            logger.info("Load {} {} database infos".format(
                len(annos), cls_name))
        return new_database_anno

    def _convert_box_format(self, bboxes_3d):
        # convert to [x,y,z,l,w,h,heading], original is [x,y,z,w,l,h,yaw]
        bboxes_3d[:, 2] += bboxes_3d[:, 5] / 2
        bboxes_3d[:, 3:6] = bboxes_3d[:, [4, 3, 5]]
        bboxes_3d[:, 6] = -(bboxes_3d[:, 6] + np.pi / 2)
        return bboxes_3d

    def _convert_box_format_back(self, bboxes_3d):
        bboxes_3d[:, 2] -= bboxes_3d[:, 5] / 2
        bboxes_3d[:, 3:6] = bboxes_3d[:, [4, 3, 5]]
        bboxes_3d[:, 6] = -(bboxes_3d[:, 6] + np.pi / 2)
        return bboxes_3d

    def _lidar_to_rect(self, pts_lidar, R0, V2C):
        pts_lidar_hom = self._cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(V2C.T, R0.T))
        return pts_rect

    def _rect_to_lidar(self, pts_rect, R0, V2C):
        pts_rect_hom = self._cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4),
                                             dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1
        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(
            np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def _cart_to_hom(self, pts):
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def _put_boxes_on_road_planes(self, sampled_boxes, road_planes, calibs):
        a, b, c, d = road_planes
        R0, V2C = calibs[4], calibs[5]
        sampled_boxes = self._convert_box_format(sampled_boxes)
        center_cam = self._lidar_to_rect(sampled_boxes[:, 0:3], R0, V2C)
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = self._rect_to_lidar(center_cam, R0, V2C)[:, 2]
        mv_height = sampled_boxes[:,
                                  2] - sampled_boxes[:, 5] / 2 - cur_lidar_height
        sampled_boxes[:, 2] -= mv_height
        sampled_boxes = self._convert_box_format_back(sampled_boxes)
        return sampled_boxes, mv_height

    def sampling(self, sample: Sample, num_samples_per_class: Dict[str, int]):
        existing_bboxes_3d = sample.bboxes_3d.copy()
        existing_velocities = None
        if sample.bboxes_3d.velocities is not None:
            existing_velocities = sample.bboxes_3d.velocities.copy()
        existing_labels = sample.labels.copy()
        existing_data = sample.data.copy()
        existing_difficulties = getattr(sample, "difficulties", None)
        ignored_bboxes_3d = getattr(
            sample, "ignored_bboxes_3d",
            np.zeros([0, existing_bboxes_3d.shape[1]],
                     dtype=existing_bboxes_3d.dtype))
        avoid_coll_bboxes_3d = np.vstack(
            [existing_bboxes_3d, ignored_bboxes_3d])

        for cls_name, num_samples in num_samples_per_class.items():
            if num_samples > 0:
                sampling_annos = self.sampler_per_class[cls_name].sampling(
                    num_samples)
                num_sampling = len(sampling_annos)
                indices = np.arange(num_sampling)
                sampling_bboxes_3d = np.vstack(
                    [sampling_annos[i]["bbox_3d"] for i in range(num_sampling)])

                sampling_bboxes = BBoxes3D(
                    sampling_bboxes_3d,
                    coordmode=sample.bboxes_3d.coordmode,
                    origin=sample.bboxes_3d.origin)
                avoid_coll_bboxes = BBoxes3D(
                    avoid_coll_bboxes_3d,
                    coordmode=sample.bboxes_3d.coordmode,
                    origin=sample.bboxes_3d.origin)
                s_bboxes_bev = sampling_bboxes.corners_2d
                e_bboxes_bev = avoid_coll_bboxes.corners_2d
                # filter the sampling bboxes which cross over the existing bboxes
                total_bv = np.concatenate([e_bboxes_bev, s_bboxes_bev], axis=0)
                coll_mat = box_collision_test(total_bv, total_bv)
                diag = np.arange(total_bv.shape[0])
                coll_mat[diag, diag] = False
                idx = e_bboxes_bev.shape[0]
                mask = []
                for num in range(num_sampling):
                    if coll_mat[idx + num].any():
                        coll_mat[idx + num] = False
                        coll_mat[:, idx + num] = False
                        mask.append(False)
                    else:
                        mask.append(True)
                indices = indices[mask]

                # put all boxes(without filter) on road plane
                sampling_bboxes_3d_copy = sampling_bboxes_3d.copy()
                if hasattr(sample, "road_plane"):
                    sampling_bboxes_3d, mv_height = self._put_boxes_on_road_planes(
                        sampling_bboxes_3d, sample.road_plane, sample.calibs)

                if len(indices) > 0:
                    sampling_data = []
                    sampling_labels = []
                    sampling_velocities = []
                    sampling_difficulties = []
                    label = self.class_names.index(cls_name)
                    for i in indices:
                        if existing_velocities is not None:
                            sampling_velocities.append(
                                sampling_annos[i]["velocity"])
                        if existing_difficulties is not None:
                            sampling_difficulties.append(
                                sampling_annos[i]["difficulty"])

                        sampling_labels.append(label)
                        lidar_data = np.fromfile(
                            osp.join(self.database_root,
                                     sampling_annos[i]["lidar_file"]),
                            "float32").reshape(
                                [-1, sampling_annos[i]["lidar_dim"]])
                        lidar_data[:, 0:3] += sampling_bboxes_3d_copy[i, 0:3]
                        if hasattr(sample, "road_plane"):
                            lidar_data[:, 2] -= mv_height[i]
                        sampling_data.append(lidar_data)

                    existing_bboxes_3d = np.vstack(
                        [existing_bboxes_3d, sampling_bboxes_3d[indices]])
                    avoid_coll_bboxes_3d = np.vstack(
                        [avoid_coll_bboxes_3d, sampling_bboxes_3d[indices]])
                    if sample.bboxes_3d.velocities is not None:
                        existing_velocities = np.vstack(
                            [existing_velocities, sampling_velocities])
                    existing_labels = np.hstack(
                        [existing_labels, sampling_labels])
                    existing_data = np.vstack(
                        [np.vstack(sampling_data), existing_data])
                    if existing_difficulties is not None:
                        existing_difficulties = np.hstack(
                            [existing_difficulties, sampling_difficulties])

        result = {
            "bboxes_3d": existing_bboxes_3d,
            "data": existing_data,
            "labels": existing_labels
        }
        if existing_velocities is not None:
            result.update({"velocities": existing_velocities})
        if existing_difficulties is not None:
            result.update({"difficulties": existing_difficulties})
        return result

    def _cal_num_samples_per_class(self, sample: Sample):
        labels = sample.labels
        num_samples_per_class = dict()
        for cls_name, max_num_samples in self.max_num_samples_per_class.items():
            label = self.class_names.index(cls_name)
            if label in labels:
                num_existing = np.sum([int(label) == int(l) for l in labels])
                num_samples = 0 if num_existing > max_num_samples else max_num_samples - num_existing
                num_samples_per_class[cls_name] = num_samples
            else:
                num_samples_per_class[cls_name] = max_num_samples
        return num_samples_per_class

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError(
                "Sampling from a database only supports lidar data!")

        num_samples_per_class = self._cal_num_samples_per_class(sample)
        samples = self.sampling(sample, num_samples_per_class)

        sample.bboxes_3d = BBoxes3D(
            samples["bboxes_3d"],
            coordmode=sample.bboxes_3d.coordmode,
            origin=sample.bboxes_3d.origin)
        sample.labels = samples["labels"]
        if "velocities" in samples:
            sample.bboxes_3d.velocities = samples["velocities"]
        if "difficulties" in samples:
            sample.difficulties = samples["difficulties"]
        sample.data = PointCloud(samples["data"])
        return sample


class Sampler(object):
    def __init__(self, cls_name: str, annos: List[dict], shuffle: bool = True):
        self.shuffle = shuffle
        self.cls_name = cls_name
        self.annos = annos
        self.idx = 0
        self.length = len(annos)
        self.indices = np.arange(len(annos))
        if shuffle:
            np.random.shuffle(self.indices)

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.idx = 0

    def sampling(self, num_samples):
        if self.idx + num_samples >= self.length:
            indices = self.indices[self.idx:].copy()
            self.reset()
        else:
            indices = self.indices[self.idx:self.idx + num_samples]
            self.idx += num_samples

        sampling_annos = [self.annos[i] for i in indices]
        return sampling_annos


# class BatchSampler:
#     """Class for sampling specific category of ground truths.

#     Args:
#         sample_list (list[dict]): List of samples.
#         name (str | None): The category of samples. Default: None.
#         epoch (int | None): Sampling epoch. Default: None.
#         shuffle (bool): Whether to shuffle indices. Default: False.
#         drop_reminder (bool): Drop reminder. Default: False.
#     """

#     def __init__(self,
#                  sampled_list,
#                  name=None,
#                  epoch=None,
#                  shuffle=True,
#                  drop_reminder=False):
#         self._sampled_list = sampled_list
#         self._indices = np.arange(len(sampled_list))
#         if shuffle:
#             np.random.shuffle(self._indices)
#         self._idx = 0
#         self._example_num = len(sampled_list)
#         self._name = name
#         self._shuffle = shuffle
#         self._epoch = epoch
#         self._epoch_counter = 0
#         self._drop_reminder = drop_reminder

#     def _sample(self, num):
#         """Sample specific number of ground truths and return indices.

#         Args:
#             num (int): Sampled number.

#         Returns:
#             list[int]: Indices of sampled ground truths.
#         """
#         if self._idx + num >= self._example_num:
#             ret = self._indices[self._idx:].copy()
#             self._reset()
#         else:
#             ret = self._indices[self._idx:self._idx + num]
#             self._idx += num
#         return ret

#     def _reset(self):
#         """Reset the index of batchsampler to zero."""
#         assert self._name is not None
#         # print("reset", self._name)
#         if self._shuffle:
#             np.random.shuffle(self._indices)
#         self._idx = 0

#     def sample(self, num):
#         """Sample specific number of ground truths.

#         Args:
#             num (int): Sampled number.

#         Returns:
#             list[dict]: Sampled ground truths.
#         """
#         indices = self._sample(num)
#         return [self._sampled_list[i] for i in indices]



# @manager.TRANSFORMS.add_component
# class DataBaseSampler(TransformABC):
#     """Class for sampling data from the ground truth database.

#     Args:
#         info_path (str): Path of groundtruth database info.
#         data_root (str): Path of groundtruth database.
#         rate (float): Rate of actual sampled over maximum sampled number.
#         prepare (dict): Name of preparation functions and the input value.
#         sample_groups (dict): Sampled classes and numbers.
#         classes (list[str]): List of classes. Default: None.
#         points_loader(dict): Config of points loader. Default: dict(
#             type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
#     """

#     def __init__(self,
#                  info_path,
#                  data_root,
#                  rate,
#                  prepare,
#                  sample_groups,
#                  classes=None,
#                  points_loader=dict(
#                      type='LoadPointsFromFile',
#                      coord_type='LIDAR',
#                      load_dim=4,
#                      use_dim=[0, 1, 2, 3])):
#         super().__init__()
#         self.data_root = data_root
#         self.info_path = info_path
#         self.rate = rate
#         self.prepare = prepare
#         self.classes = classes
#         self.cat2label = {name: i for i, name in enumerate(classes)}
#         self.label2cat = {i: name for i, name in enumerate(classes)}
#         self.points_loader = LoadPointsFromFile(coord_type='LIDAR',
#                                                 load_dim=4,
#                                                 use_dim=[0, 1, 2, 3]) 
#         #mmcv.build_from_cfg(points_loader, PIPELINES)

#         db_infos = mmcv.load(info_path)

#         # filter database infos
#         # from mmdet3d.utils import get_root_logger
#         # logger = get_root_logger()
#         # for k, v in db_infos.items():
#         #     logger.info(f'load {len(v)} {k} database infos')
#         # for prep_func, val in prepare.items():
#         #     db_infos = getattr(self, prep_func)(db_infos, val)
#         # logger.info('After filter database:')
#         # for k, v in db_infos.items():
#         #     logger.info(f'load {len(v)} {k} database infos')

#         self.db_infos = db_infos

#         # load sample groups
#         # TODO: more elegant way to load sample groups
#         self.sample_groups = []
#         for name, num in sample_groups.items():
#             self.sample_groups.append({name: int(num)})

#         self.group_db_infos = self.db_infos  # just use db_infos
#         self.sample_classes = []
#         self.sample_max_nums = []
#         for group_info in self.sample_groups:
#             self.sample_classes += list(group_info.keys())
#             self.sample_max_nums += list(group_info.values())

#         self.sampler_dict = {}
#         for k, v in self.group_db_infos.items():
#             self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
#         # TODO: No group_sampling currently
#         print("dbsample.py, self.sample_classes: ", self.sample_classes)
#         print("dbsample.py, self.cat2label: ", self.cat2label)

#     @staticmethod
#     def filter_by_difficulty(db_infos, removed_difficulty):
#         """Filter ground truths by difficulties.

#         Args:
#             db_infos (dict): Info of groundtruth database.
#             removed_difficulty (list): Difficulties that are not qualified.

#         Returns:
#             dict: Info of database after filtering.
#         """
#         new_db_infos = {}
#         for key, dinfos in db_infos.items():
#             new_db_infos[key] = [
#                 info for info in dinfos
#                 if info['difficulty'] not in removed_difficulty
#             ]
#         return new_db_infos

#     @staticmethod
#     def filter_by_min_points(db_infos, min_gt_points_dict):
#         """Filter ground truths by number of points in the bbox.

#         Args:
#             db_infos (dict): Info of groundtruth database.
#             min_gt_points_dict (dict): Different number of minimum points
#                 needed for different categories of ground truths.

#         Returns:
#             dict: Info of database after filtering.
#         """
#         for name, min_num in min_gt_points_dict.items():
#             min_num = int(min_num)
#             if min_num > 0:
#                 filtered_infos = []
#                 for info in db_infos[name]:
#                     if info.get('sampled_points_num', info['num_points_in_gt']) >= min_num:
#                         filtered_infos.append(info)
#                 db_infos[name] = filtered_infos
#         return db_infos

#     def sample_all(self, gt_bboxes, gt_names, img=None, ground_plane=None, noise_classes=None):
#         """Sampling all categories of bboxes.

#         Args:
#             gt_bboxes (np.ndarray): Ground truth bounding boxes.
#             gt_labels (np.ndarray): Ground truth labels of boxes.
#             img (np.ndarray, optional): Image array. Defaults to None.
#             ground_plane (np.ndarray, optional): Ground plane information.
#                 Defaults to None.

#         Returns:
#             dict: Dict of sampled 'pseudo ground truths'.

#                 - gt_labels_3d (np.ndarray): ground truths labels
#                   of sampled objects.
#                 - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`):
#                   sampled ground truth 3D bounding boxes
#                 - points (np.ndarray): sampled points
#                 - group_ids (np.ndarray): ids of sampled ground truths
#         """
#         sampled_num_dict = {}
#         sample_num_per_class = []
#         if noise_classes is None:
#             noise_classes = set()
#         for class_name, max_sample_num in zip(self.sample_classes,
#                                               self.sample_max_nums):
#             # class_label = self.cat2label[class_name]
#             sampled_num = int(max_sample_num -
#                               np.sum([n == class_name for n in gt_names]))
#             # sampled_num = int(max_sample_num -
#             #                   np.sum([n == class_label for n in gt_labels]))
#             sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
#             sampled_num_dict[class_name] = sampled_num
#             sample_num_per_class.append(sampled_num)

#         sampled = []
#         sampled_gt_bboxes = []
#         avoid_coll_boxes = gt_bboxes

#         for class_name, sampled_num in zip(self.sample_classes,
#                                            sample_num_per_class):
#             if sampled_num > 0:
#                 sampled_cls = self.sample_class_v2(class_name, sampled_num,
#                                                    avoid_coll_boxes)

#                 sampled += sampled_cls
#                 if len(sampled_cls) > 0:
#                     sampled_gt_box = np.concatenate(
#                         [s["box3d_lidar"].reshape((-1, 7)) for s in sampled_cls], axis=0
#                     )
#                     collision_box = np.concatenate(
#                         [s.get('collision_box', s["box3d_lidar"]).reshape((-1, 7)) for s in sampled_cls], axis=0
#                     )

#                     sampled_gt_bboxes += [sampled_gt_box]
#                     avoid_coll_boxes = np.concatenate(
#                         [avoid_coll_boxes, collision_box], axis=0)

#         ret = None
#         if len(sampled) > 0:
#             sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
#             num_sampled = len(sampled_gt_bboxes)
#             s_points_list = []
#             num_points_list = []
#             gt_names = []
#             difficulty = []
#             sampled_noise_mask = []
            
#             count = 0
#             for info in sampled:
#                 file_path = os.path.join(
#                     self.data_root,
#                     info['path']) if self.data_root else info['path']
#                 results = dict(pts_filename=file_path)
#                 s_points = self.points_loader(results)['points']
#                 center = info.get('center', info['box3d_lidar'][:3])  # weakly labeled area
#                 s_points.translate(center)

#                 count += 1

#                 s_points_list.append(s_points)
                
#                 if isinstance(info['name'], np.ndarray):  # weakly labeled area
#                     gt_names.extend(info['name'])
#                     # difficulty.extend(info['difficulty'])
#                     # num_points_list.extend(info['num_points_in_gt'])
#                     sampled_noise_mask.append(np.zeros(len(s_points), dtype=np.bool))
#                 else:
#                     gt_names.append(info['name'])
#                     # difficulty.append(info['difficulty'])
#                     # num_points_list.append(s_points.shape[0])
#                     if info['name'] in noise_classes:
#                         sampled_noise_mask.append(np.ones(len(s_points), dtype=np.bool))
#                     else:
#                         sampled_noise_mask.append(np.zeros(len(s_points), dtype=np.bool))

#             gt_labels = np.array([self.cat2label[cat] if cat in self.classes else -1 for cat in gt_names],
#                                  dtype=np.long)

#             if ground_plane is not None:
#                 xyz = sampled_gt_bboxes[:, :3]
#                 dz = (ground_plane[:3][None, :] *
#                       xyz).sum(-1) + ground_plane[3]
#                 sampled_gt_bboxes[:, 2] -= dz
#                 for i, s_points in enumerate(s_points_list):
#                     s_points.tensor[:, 2].sub_(dz[i])

#             ret = {
#                 # 'gt_names_3d': np.array(gt_names),
#                 # 'difficulty': np.array(difficulty),
#                 'gt_labels_3d':
#                 gt_labels,
#                 'gt_bboxes_3d':
#                 sampled_gt_bboxes,
#                 'points':
#                 s_points_list[0].cat(s_points_list),
#                 'group_ids':
#                 np.arange(gt_bboxes.shape[0],
#                           gt_bboxes.shape[0] + len(sampled)),
#                 # 'point_num': np.array([num for num in num_points_list]),
#                 'collision_boxes': avoid_coll_boxes[gt_bboxes.shape[0]:],
#                 'sampled_noise_mask': np.concatenate(sampled_noise_mask, 0),
#             }

#         return ret

#     def sample_class_v2(self, name, num, gt_bboxes):
#         """Sampling specific categories of bounding boxes.

#         Args:
#             name (str): Class of objects to be sampled.
#             num (int): Number of sampled bboxes.
#             gt_bboxes (np.ndarray): Ground truth boxes.

#         Returns:
#             list[dict]: Valid samples after collision test.
#         """
#         sampled = self.sampler_dict[name].sample(num)
#         sampled = copy.deepcopy(sampled)
#         num_gt = gt_bboxes.shape[0]
#         num_sampled = len(sampled)
#         gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
#             gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6])

#         # sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
#         sp_boxes = np.concatenate(
#             [s.get('collision_box', s["box3d_lidar"]).reshape((-1, 7)) for s in sampled], axis=0)
#         boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

#         sp_boxes_new = boxes[gt_bboxes.shape[0]:]
#         sp_boxes_bv = box_np_ops.center_to_corner_box2d(
#             sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

#         total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
#         coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
#         diag = np.arange(total_bv.shape[0])
#         coll_mat[diag, diag] = False

#         valid_samples = []
#         for i in range(num_gt, num_gt + num_sampled):
#             if coll_mat[i].any():
#                 coll_mat[i] = False
#                 coll_mat[:, i] = False
#             else:
#                 valid_samples.append(sampled[i - num_gt])
#         return valid_samples
