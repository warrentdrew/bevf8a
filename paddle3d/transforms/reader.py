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
class LoadImage(TransformABC):
    """
    as name
    """
    _READER_MAPPER = {"cv2": cv2.imread, "pillow": Image.open}

    def __init__(self,
                 to_chw: bool = True,
                 to_rgb: bool = True,
                 reader: str = "cv2"):
        if reader not in self._READER_MAPPER.keys():
            raise ValueError('Unsupported reader {}'.format(reader))

        self.reader = reader
        self.to_rgb = to_rgb
        self.to_chw = to_chw

    def __call__(self, sample: Sample) -> Sample:
        """
        as name
        """
        sample.data = np.array(self._READER_MAPPER[self.reader](sample.path))

        sample.meta.image_reader = self.reader
        sample.meta.image_format = "bgr" if self.reader == "cv2" else "rgb"
        sample.meta.channel_order = "hwc"

        if sample.meta.image_format != "rgb" and self.to_rgb:
            if sample.meta.image_format == "bgr":
                sample.data = cv2.cvtColor(sample.data, cv2.COLOR_BGR2RGB)
                sample.meta.image_format = "rgb"
            else:
                raise RuntimeError('Unsupported image format {}'.format(
                    sample.meta.image_format))
        elif sample.meta.image_format != "bgr" and (self.to_rgb is False):
            if sample.meta.image_format == "rgb":
                sample.data = sample.data[:, :, ::-1]
                sample.meta.image_format = "bgr"
            else:
                raise RuntimeError('Unsupported image format {}'.format(
                    sample.meta.image_format))

        if self.to_chw:
            sample.data = sample.data.transpose((2, 0, 1))
            sample.meta.channel_order = "chw"

        return sample


@manager.TRANSFORMS.add_component
class LoadPointCloud(TransformABC):
    """
    Load point cloud.

    Args:
        dim: The dimension of each point.
        use_dim: The dimension of each point to use.
        use_time_lag: Whether to use time lag.
        sweep_remove_radius: The radius within which points are removed in sweeps.
    """

    def __init__(self,
                 dim,
                 use_dim: Union[int, List[int]] = None,
                 use_time_lag: bool = False,
                 sweep_remove_radius: float = 1):
        self.dim = dim
        self.use_dim = range(use_dim) if isinstance(use_dim, int) else use_dim
        self.use_time_lag = use_time_lag
        self.sweep_remove_radius = sweep_remove_radius

    def __call__(self, sample: Sample):
        """
        as name
        """
        if sample.modality != "lidar":
            raise ValueError('{} Only Support samples in modality lidar'.format(
                self.__class__.__name__))

        if sample.data is not None:
            raise ValueError(
                'The data for this sample has been processed before.')

        data = np.fromfile(sample.path, np.float32).reshape(-1, self.dim)

        if self.use_dim is not None:
            data = data[:, self.use_dim]

        if self.use_time_lag:
            time_lag = np.zeros((data.shape[0], 1), dtype=data.dtype)
            data = np.hstack([data, time_lag])

        if len(sample.sweeps) > 0:
            data_sweep_list = [
                data,
            ]
            # np.random.seed(0)
            for i in np.random.choice(
                    len(sample.sweeps), len(sample.sweeps), replace=False):
                sweep = sample.sweeps[i]
                sweep_data = np.fromfile(sweep.path, np.float32).reshape(
                    -1, self.dim)
                if self.use_dim:
                    sweep_data = sweep_data[:, self.use_dim]
                sweep_data = sweep_data.T

                # Remove points that are in a certain radius from origin.
                x_filter_mask = np.abs(
                    sweep_data[0, :]) < self.sweep_remove_radius
                y_filter_mask = np.abs(
                    sweep_data[1, :]) < self.sweep_remove_radius
                not_close = np.logical_not(
                    np.logical_and(x_filter_mask, y_filter_mask))
                sweep_data = sweep_data[:, not_close]

                # Homogeneous transform of current sample to reference coordinate
                if sweep.meta.ref_from_curr is not None:
                    sweep_data[:3, :] = sweep.meta.ref_from_curr.dot(
                        np.vstack((sweep_data[:3, :],
                                   np.ones(sweep_data.shape[1]))))[:3, :]
                sweep_data = sweep_data.T
                if self.use_time_lag:
                    curr_time_lag = sweep.meta.time_lag * np.ones(
                        (sweep_data.shape[0], 1)).astype(sweep_data.dtype)
                    sweep_data = np.hstack([sweep_data, curr_time_lag])
                data_sweep_list.append(sweep_data)
            data = np.concatenate(data_sweep_list, axis=0)

        sample.data = PointCloud(data)
        return sample

@manager.TRANSFORMS.add_component
class LoadPointsFromFile(TransformABC):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type='LIDAR',
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
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

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        check_file_exist(pts_filename)
        if pts_filename.endswith('.npy'):
            points = np.load(pts_filename)
        else:
            points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (np.ndarray): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)  # (173760,)
        points = points.reshape(-1, self.load_dim) # (34752, 5)
        points = points[:, self.use_dim]  # (34752, 5)

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
            random.seed(0) # TODO zhuyipin remove in training 
            sample_pointnum_ratio = random.random() * (1 - drop_ratio) + drop_ratio
            points_num_src = points.shape[0]
            points_num_tgt = min(int((points_num_src + 1 / sample_pointnum_ratio - 1) * sample_pointnum_ratio),
                                points_num_src)
            random.seed(0) # TODO zhuyipin remove in training 
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
            if True: #self.test_mode: #TODO zhuyipin change to test mode
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
                np.random.seed(0) # TODO zhuyipin
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
        # points = self._load_points(zip_path)
        # 8A
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
class LoadMultiViewImageFromFiles(TransformABC):
    """
    load multi-view image from files

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
                 project_pts_to_img_depth=False,
                 cam_depth_range=[4.0, 45.0, 1.0],
                 constant_std=0.5,
                 imread_flag=-1):
        self.to_float32 = to_float32
        self.project_pts_to_img_depth = project_pts_to_img_depth
        self.cam_depth_range = cam_depth_range
        self.constant_std = constant_std
        self.imread_flag = imread_flag

    def __call__(self, sample):
        """
        Call function to load multi-view image from files.
        """
        filename = sample['img_filename']

        img = np.stack(
            [cv2.imread(name, self.imread_flag) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        sample['filename'] = filename

        sample['img'] = [img[..., i] for i in range(img.shape[-1])]
        sample['img_shape'] = img.shape
        sample['ori_shape'] = img.shape

        sample['pad_shape'] = img.shape
        # sample['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]

        sample['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        sample['img_fields'] = ['img']

        if self.project_pts_to_img_depth:
            sample['img_depth'] = []
            for i in range(len(sample['img'])):
                depth = map_pointcloud_to_image(
                    sample['points'],
                    sample['img'][i],
                    sample['caminfo'][i]['sensor2lidar_rotation'],
                    sample['caminfo'][i]['sensor2lidar_translation'],
                    sample['caminfo'][i]['cam_intrinsic'],
                    show=False)
                guassian_depth, min_depth, std_var = generate_guassian_depth_target(
                    paddle.to_tensor(depth).unsqueeze(0),
                    stride=8,
                    cam_depth_range=self.cam_depth_range,
                    constant_std=self.constant_std)
                depth = paddle.concat(
                    [min_depth[0].unsqueeze(-1), guassian_depth[0]], axis=-1)
                sample['img_depth'].append(depth)
        return sample


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

        # 8A
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
            classes=None, # 8A
    ):
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.with_name_3d = with_name_3d
        self.with_weakly_roi = with_weakly_roi
        self.classes = classes # 8A

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
        
        # ================
        # 8A 
        if 'accessory_main' in self.classes:
            gt_names_3d = np.asarray(results["gt_names_3d"])
            mot_mask = ((gt_names_3d == 'bigMot') | (gt_names_3d == 'smallMot')) | (gt_names_3d == 'verybigMot')
            # mot_mask = paddle.to_tensor(mot_mask) #torch.from_numpy(mot_mask)
            roi_regions = [
                {
                    'type': 3,
                    'region': results['gt_bboxes_3d'][mot_mask],
                    'threshold': 0.3,
                    'task_of_interest': self.classes.index('accessory_main')
                }
            ]
        # ================
            
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

@manager.TRANSFORMS.add_component
class LoadAnnotations3D_valid(TransformABC):
    """
    load annotation
    """

    def __init__(
            self,
            with_bbox_3d=True,
            with_label_3d=True,
            with_mask_3d=False,
            with_seg_3d=False):
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d

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
        sample['valid_flag'] = sample['ann_info']['valid_flag']
        return sample

    def __call__(self, sample) -> Sample:
        """Call function to load multiple types annotations.
        """
        if self.with_bbox_3d:
            sample = self._load_bboxes_3d(sample)
            if sample is None:
                return None

        if self.with_label_3d:
            sample = self._load_labels_3d(sample)

        return sample

@manager.TRANSFORMS.add_component
class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            sweeps_num=5,
            to_float32=False,
            pad_empty_sweeps=False,
            sweep_range=[3, 27],
            sweeps_id=None,
            imread_flag=-1,  #'unchanged'
            sensors=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            test_mode=True,
            prob=1.0,
    ):

        self.sweeps_num = sweeps_num
        self.to_float32 = to_float32
        self.imread_flag = imread_flag
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.prob = prob
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def __call__(self, sample):
        """Call function to load multi-view sweep image from filenames.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = sample['img']
        img_timestamp = sample['img_timestamp']
        lidar_timestamp = sample['timestamp']
        img_timestamp = [
            lidar_timestamp - timestamp for timestamp in img_timestamp
        ]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)
        nums = len(imgs)
        if self.pad_empty_sweeps and len(sample['sweeps']) == 0:
            for i in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                mean_time = (
                    self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend(
                    [time + mean_time for time in img_timestamp])
                for j in range(nums):
                    sample['filename'].append(sample['filename'][j])
                    sample['lidar2img'].append(np.copy(sample['lidar2img'][j]))
                    sample['intrinsics'].append(
                        np.copy(sample['intrinsics'][j]))
                    sample['extrinsics'].append(
                        np.copy(sample['extrinsics'][j]))
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(sample['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(sample['sweeps']))
            elif self.test_mode:
                choices = [
                    int((self.sweep_range[0] + self.sweep_range[1]) / 2) - 1
                ]
            else:
                # np.random.seed(0)
                if np.random.random() < self.prob:
                    if self.sweep_range[0] < len(sample['sweeps']):
                        sweep_range = list(
                            range(
                                self.sweep_range[0],
                                min(self.sweep_range[1],
                                    len(sample['sweeps']))))
                    else:
                        sweep_range = list(
                            range(self.sweep_range[0], self.sweep_range[1]))
                    # np.random.seed(0)
                    choices = np.random.choice(
                        sweep_range, self.sweeps_num, replace=False)
                else:
                    choices = [
                        int((self.sweep_range[0] + self.sweep_range[1]) / 2) - 1
                    ]

            for idx in choices:
                sweep_idx = min(idx, len(sample['sweeps']) - 1)
                sweep = sample['sweeps'][sweep_idx]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = sample['sweeps'][sweep_idx - 1]
                sample['filename'].extend(
                    [sweep[sensor]['data_path'] for sensor in self.sensors])

                img = np.stack([
                    cv2.imread(sweep[sensor]['data_path'], self.imread_flag)
                    for sensor in self.sensors
                ],
                               axis=-1)

                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                sweep_imgs_list.extend(img)
                sweep_ts = [
                    lidar_timestamp - sweep[sensor]['timestamp'] / 1e6
                    for sensor in self.sensors
                ]
                timestamp_imgs_list.extend(sweep_ts)
                for sensor in self.sensors:
                    sample['lidar2img'].append(sweep[sensor]['lidar2img'])
                    sample['intrinsics'].append(sweep[sensor]['intrinsics'])
                    sample['extrinsics'].append(sweep[sensor]['extrinsics'])
        sample['img'] = sweep_imgs_list
        sample['timestamp'] = timestamp_imgs_list

        return sample
