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

import os
from pathlib import Path
from typing import List, Union
import zipfile
from pypcd import pypcd

import cv2
import paddle
import numpy as np
from PIL import Image

from paddle3d.apis import manager
from paddle3d.datasets.at128.core import check_file_exist, imfrombytes
from paddle3d.geometries import PointCloud
from paddle3d.geometries.bbox import points_in_convex_polygon_3d_jit
from paddle3d.models.detection.bevfusion.utils import generate_guassian_depth_target, map_pointcloud_to_image
from paddle3d.sample import Sample
from paddle3d.transforms import functional as F
from paddle3d.transforms.base import TransformABC
from paddle3d.utils.logger import logger

__all__ = [
    "LoadImage", "LoadPointCloud"
]


@manager.TRANSFORMS.add_component
class LoadPointsFromZipFile(TransformABC):
    """
    Load point cloud.

    Args:
        dim: The dimension of each point.
        use_dim: The dimension of each point to use.
        use_time_lag: Whether to use time lag.
        sweep_remove_radius: The radius within which points are removed in sweeps.
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim = [0, 1, 2],
                 shift_height=False,
                 pad_empty_sweeps=False,
                 use_nsweeps_points=False,
                 remove_close=False,
                 test_mode=False,
                 downsample_ratio=1.0,
                 nsweeps=1,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.use_nsweeps_points = use_nsweeps_points
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.nsweeps = nsweeps
        self.downsample_ratio = downsample_ratio

    def find_points_path(self, zip_file_list):
        for each_file in zip_file_list:
            if each_file.endswith('.pcd'):
                return each_file
    
    def _load_points(self, zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_obj:
            zip_file_list = zip_obj.namelist()
            #print("zip_file_list: ", zip_file_list)   # 'velodyne_points/at128_fusion.pcd'
            # if 'at128_fusion' in zip_path:
            #     points_path = 'velodyne_points/at128_fusion.pcd'   
            # elif 'hesai90' in zip_path:
            #     points_path = 'velodyne_points/hesai90.pcd' 
            # elif 'hesai128' in zip_path:
            #     points_path = 'velodyne_points/hesai128.pcd' 
            points_path = self.find_points_path(zip_file_list)
            if points_path in zip_file_list:
                points_obj = zip_obj.open(points_path, 'r')
                points_pcd = pypcd.PointCloud.from_fileobj(points_obj)
                x = points_pcd.pc_data["x"]
                y = points_pcd.pc_data["y"]
                z = points_pcd.pc_data["z"]
                intensity = points_pcd.pc_data["intensity"]
                mask = np.isnan(x) | np.isnan(y) | np.isnan(z) | np.isnan(intensity)
                points = np.c_[x, y, z, intensity]
                points = points[~mask]
                #print("points.shape: ", points.shape)

        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        return points
    
    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]
    
    def _random_sample_points(self, points, drop_ratio=1.0):
        if drop_ratio < 0.9999:
            sample_pointnum_ratio = random.random() * (1 - drop_ratio) + drop_ratio
            points_num_src = points.shape[0]
            points_num_tgt = min(int((points_num_src + 1 / sample_pointnum_ratio - 1) * sample_pointnum_ratio),
                                points_num_src)
            sample_indexes = random.sample(range(points_num_src), points_num_tgt)
            points = points[sample_indexes, ...]
        return points
    
    def _transform_points(self, points, transform_matrix):
        points = points.T
        nbr_points = points.shape[1]
        points[:3, :] = transform_matrix.dot(
            np.vstack((points[:3, :], np.ones(nbr_points)))
        )[:3, :]
        return points.T
    
    def _load_nsweeps_points(self, points, results):
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]
        sweeps = results['sweeps']

        if len(sweeps) > 0:
            if self.test_mode:
                choices = []
                choices_list = list(range(len(sweeps)))
                if self.nsweeps > 1:
                    # sort choices
                    choices = sorted(choices_list, key=lambda _i: sweeps[_i]['time_lag'], reverse=False)
                    if (self.nsweeps - 1) < len(choices):
                        choices = choices[:self.nsweeps - 1]
            else: # train
                # assert (self.nsweeps - 1) <= len(sweeps
                #     ), "nsweeps {} should not greater than list length {}.".format(self.nsweeps, len(sweeps))
                # choices = sorted(choices_list, key=lambda _i: sweeps[_i]['time_lag'], reverse=False) #wangna11
                choices = np.random.choice(min(len(sweeps), (self.nsweeps - 1) * 2), self.nsweeps - 1, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['zip_filename'])
                points_sweep = self._random_sample_points(points_sweep, self.downsample_ratio)
                sweep_points_list.append(self._transform_points(points_sweep, sweep["transform_matrix"]))
                times_sweep = np.zeros((points_sweep.shape[0], 1), dtype=points_sweep.dtype) + sweep['time_lag']
                sweep_times_list.append(times_sweep)
        # step 3. concatenate
        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)
        points_pcd = np.hstack([points, times]).astype(np.float32)
        # step 4. filter nan and nearby points
        nan_mask = np.isnan(points_pcd).any(axis=1)
        points_pcd = points_pcd[~nan_mask]
        if self.remove_close:
            points_pcd = self._remove_close(points_pcd)

        return points_pcd 
    
    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (np.ndarray): Point clouds data.
        """
        #pts_filename = results['pts_filename']
        #print("results.keys(): ", results.keys())
        filename = results['img_filename']
        results['filename'] = filename
        # Load from ZIP
        zip_path = filename[0].split('+')[0]
        #print("zip_path: ", zip_path)
        # 'records/kitti_data/at128_fusion/files/MKZ223_495_1649836079_1649836919/408083/408083.zip'

        #(num_points, 4): x, y, z, intensity
        try:
            points = self._load_points(zip_path)
        except:
            print('bad zip file,', zip_path)
            return None

        if self.use_nsweeps_points:
            #(num_points, 5): x, y, z, intensity, time_lag
            points = self._load_nsweeps_points(points, results)
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points, np.expand_dims(height, 1)], 1)
            attribute_dims = dict(height=3)

        # points_class = get_points_type(self.coord_type)
        # points = points_class(
        #     points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

@manager.TRANSFORMS.add_component
class LoadMultiViewImageFromZipFiles(TransformABC):
    """
    load multi-view image from zip files

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Default: False.
        color_type (str): Color type of the file. Default: -1.
            - -1: cv2.IMREAD_UNCHANGED
            -  0: cv2.IMREAD_GRAYSCALE
            -  1: cv2.IMREAD_COLOR
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, sample):
        """
        Call function to load multi-view image from files.
        """
        filename = sample['img_filename']
        # Load from ZIP
        zip_path = filename[0].split('+')[0]

        with zipfile.ZipFile(zip_path, 'r') as zip_obj:
            zip_file_list = zip_obj.namelist()
            #print("zip_file_list: ", zip_file_list)   # 'velodyne_points/at128_fusion.pcd'

            # load surrounding image
            imgs = []
            height, width = -1, -1
            for img_filename in sample['img_filename']:
                if img_filename is None:
                    imgs.append(None)
                    continue
                img_filename = img_filename.split('+')[1]
                if img_filename is not None and img_filename in zip_file_list:
                    with zip_obj.open(img_filename, 'r') as img_obj:
                        img = imfrombytes(img_obj.read())
                        if height <= 0 or width <= 0:
                           height, width, _ = img.shape
                        if self.to_float32:
                            img = img.astype(np.float32)
                else:
                    img = np.zeros((1080, 1920, 3), dtype=np.float32)
                imgs.append(img)
            for i in range(len(imgs)):
                if imgs[i] is None: 
                    imgs[i] = np.zeros((height, width, 3), dtype=np.float32)
                    print('add zeros image: ({}, {}, 3)'.format(height, width))
            img = np.stack(imgs, axis=-1)
        
        # Load from image_path
        sample['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        sample['img'] = [img[..., i] for i in range(img.shape[-1])]
        sample['img_shape'] = img.shape
        sample['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        sample['pad_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        sample['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        sample['img_fields'] = ['img']
        return sample

@manager.TRANSFORMS.add_component
class LoadAnnotations3D(TransformABC):
    """
    load annotation
    """

    def __init__(
            self,
            with_bbox_3d=True,
            with_label_3d=True,
            with_attr_label=False,
            with_mask_3d=False,
            with_seg_3d=False,
            with_name_3d=False,
            with_weakly_roi=False,
            classes=None,
    ):
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.with_name_3d = with_name_3d
        self.with_weakly_roi = with_weakly_roi
        self.classes = classes

    def _load_names_3d(self, results):
        """Private function to load label name annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label name annotations.
        """
        results['gt_names_3d'] = results['ann_info']['gt_names_3d']
        return results
    
    def _load_bboxes_3d(self, sample) -> Sample:
        """
        as name
        """
        sample['gt_bboxes_3d'] = sample['ann_info']['gt_bboxes_3d']
        sample['bbox3d_fields'].append('gt_bboxes_3d')
        return sample

    def _load_labels_3d(self, sample) -> Sample:
        """
        as name
        """
        sample['gt_labels_3d'] = sample['ann_info']['gt_labels_3d']
        return sample

    def _load_attr_labels(self, sample) -> Sample:
        """
        as name
        """
        sample['attr_labels'] = sample['ann_info']['attr_labels']
        return sample
    
    def _load_weakly_roi_data(self, results):
        """Private function to load roi regions in weakly labeled data.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded roi regions.
        """
        roi_regions = []
        if 'accessory_main' in self.classes:
            gt_names_3d = np.asarray(results["gt_names_3d"])
            mot_mask = ((gt_names_3d == 'bigMot') | (gt_names_3d == 'smallMot')) | (gt_names_3d == 'verybigMot')
            mot_mask = np.array(mot_mask)
            roi_regions = [
                {
                    'type': 3,
                    'region': results['gt_bboxes_3d'][mot_mask],
                    'threshold': 0.3,
                    'task_of_interest': self.classes.index('accessory_main')
                }
            ]
        roi_infos = results['ann_info'].get('roi_data', None)
        if roi_infos is not None and 'result' in roi_infos:
            roi_infos = roi_infos['result']
        else:
            roi_infos = []
        for roi_info in roi_infos:
            for region in roi_info['content']:
                if roi_info['type'] == 2:
                    cx, cy, cz, r = region['x'], region['y'], region['z'], region['r']
                    roi_regions.append(
                        {
                            'region': np.array([cx, cy, cz, r]),
                            'type': roi_info['type']
                        }
                    )
                else:
                    raise NotImplementedError
            
        results['roi_regions'] = roi_regions
        return results

    def __call__(self, sample) -> Sample:
        """Call function to load multiple types annotations.
        """
        if self.with_bbox_3d:
            sample = self._load_bboxes_3d(sample)
            if sample is None:
                return None

        if self.with_label_3d:
            sample = self._load_labels_3d(sample)

        if self.with_attr_label:
            sample = self._load_attr_labels(sample)
        
        if self.with_name_3d:
            sample = self._load_names_3d(sample)
        if self.with_weakly_roi:
            sample = self._load_weakly_roi_data(sample)

        return sample
