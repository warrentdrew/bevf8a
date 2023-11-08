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

import numbers
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from copy import deepcopy
from numpy import random

import paddle
import paddle.nn as nn
from paddle3d.apis import manager
from paddle3d.geometries.bbox import BBoxes3D, CoordMode, points_in_convex_polygon_3d_jit
from paddle3d.sample import Sample
from paddle3d import transforms as T
from paddle3d.datasets.at128.core import is_list_of
from paddle3d.transforms import functional as F
from paddle3d.transforms.base import TransformABC
from paddle3d.transforms.functional import points_to_voxel
from paddle3d.utils import box_utils
from paddle3d.utils_idg.box_np_ops import points_in_rbbox

__all__ = [
    "RandomHorizontalFlip", "RandomVerticalFlip", "GlobalRotate", "GlobalScale",
    "GlobalTranslate", "ShufflePoint", "SamplePoint",
    "FilterPointOutsideRange", "FilterBBoxOutsideRange", "HardVoxelize",
    "RandomObjectPerturb", "ConvertBoxFormat", "ResizeShortestEdge",
    "RandomContrast", "RandomBrightness", "RandomSaturation",
    "ToVisionBasedBox", "PhotoMetricDistortionMultiViewImage",
    "RandomScaleImageMultiViewImage"
]


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (paddle.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        paddle.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period]
    """
    return val - np.floor(val / period + offset) * period


@manager.TRANSFORMS.add_component
class SampleRangeFilter(object):
    """
    Filter samples by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def in_range_bev(self, box_range, gt_bboxes_3d):
        """
        Check whether the boxes are in the given range.
        """
        in_range_flags = ((gt_bboxes_3d[:, 0] > box_range[0])
                          & (gt_bboxes_3d[:, 1] > box_range[1])
                          & (gt_bboxes_3d[:, 0] < box_range[2])
                          & (gt_bboxes_3d[:, 1] < box_range[3]))
        return in_range_flags

    def limit_yaw(self, gt_bboxes_3d, offset=0.5, period=np.pi):
        """Limit the yaw to a given period and offset.

        Args:
            offset (float): The offset of the yaw.
            period (float): The expected period.
        """
        gt_bboxes_3d[:, 6] = limit_period(gt_bboxes_3d[:, 6], offset, period)
        return gt_bboxes_3d

    def __call__(self, sample):
        """Call function to filter objects by the range.

        Args:
            sample (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the Sample.
        """
        if isinstance(sample['gt_bboxes_3d'], (BBoxes3D, np.ndarray)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        else:
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = sample['gt_bboxes_3d']
        gt_labels_3d = sample['gt_labels_3d']
        gt_names_3d = sample['gt_names_3d']

        mask = self.in_range_bev(bev_range, gt_bboxes_3d)
        gt_bboxes_3d = gt_bboxes_3d[mask]

        gt_labels_3d = gt_labels_3d[mask.astype(np.bool_)]
        gt_names_3d = np.array(gt_names_3d)[mask.astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d = self.limit_yaw(
            gt_bboxes_3d, offset=0.5, period=2 * np.pi)
        sample['gt_bboxes_3d'] = gt_bboxes_3d
        sample['gt_labels_3d'] = gt_labels_3d
        sample['gt_names_3d'] = gt_names_3d

        return sample


@manager.TRANSFORMS.add_component
class SampleNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, sample):
        """Call function to filter objects by their names.

        Args:
            sample (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the Sample.
        """
        gt_labels_3d = sample['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        sample['gt_bboxes_3d'] = sample['gt_bboxes_3d'][gt_bboxes_mask]
        sample['gt_labels_3d'] = sample['gt_labels_3d'][gt_bboxes_mask]

        return sample



@manager.TRANSFORMS.add_component
class MyNormalize(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, sample):
        """Call function to normalize images.
        Args:
            sample (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized sample, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in sample.get('img_fields', ['img']):
            if key =='img_depth':
                continue
            for idx in range(len(sample['img'])):
                sample[key][idx] = F.normalize_use_cv2(sample[key][idx], self.mean, self.std,
                                                     self.to_rgb)
        sample['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return sample




@manager.TRANSFORMS.add_component
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    This class is modified from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L289
    Args:
        scales
    """

    def __init__(self, scales=[], fix_size=False):
        self.scales = scales
        self.fix_size = fix_size
        # assert len(self.scales) == 1

    def __call__(self, sample):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            sample (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if self.fix_size == False:
            if len(self.scales) == 2:
                rand_scale = np.random.uniform(low=self.scales[0], high=self.scales[1])
                y_size = [int(img.shape[0] * rand_scale) for img in sample['img']]
                x_size = [int(img.shape[1] * rand_scale) for img in sample['img']]
            elif len(self.scales) == 1:
                rand_scale = self.scales[0]
                y_size = [int(img.shape[0] * rand_scale) for img in sample['img']]
                x_size = [int(img.shape[1] * rand_scale) for img in sample['img']]

            scale_factor = np.eye(4)
            scale_factor[0, 0] *= rand_scale
            scale_factor[1, 1] *= rand_scale
            sample['img'] = [
                imresize(img, (x_size[idx], y_size[idx]), return_scale=False)
                for idx, img in enumerate(sample['img'])
            ]
            lidar2img = [scale_factor @ l2i for l2i in sample['lidar2img']]
            sample['lidar2img'] = lidar2img
            sample['img_shape'] = [img.shape for img in sample['img']]
            sample['ori_shape'] = [img.shape for img in sample['img']]
        else:
            y_size = [ 768 for _ in sample['img']]
            x_size = [ 1152 for _ in sample['img']]
            scale_factors = []
            for idx in range(len(x_size)):
                scale_factor_h = 1.0*y_size[idx] / sample['img'][idx].shape[0]
                scale_factor_w = 1.0*x_size[idx] / sample['img'][idx].shape[1]
                scale_factor = np.eye(4)
                scale_factor[0, 0] *= scale_factor_w
                scale_factor[1, 1] *= scale_factor_h
                scale_factors.append(scale_factor)

            sample['img'] = [imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                               enumerate(sample['img'])]

            lidar2img = []
            idx=0
            for l2i in sample['lidar2img']:
                lidar2img.append(scale_factors[idx] @ l2i)
                idx += 1
            sample['lidar2img'] = lidar2img
            sample['img_shape'] = [img.shape for img in sample['img']]
            sample['ori_shape'] = [img.shape for img in sample['img']]

        # # For Ego3RT 
        # for i in range(len(sample['img'])):
        #     cam_intrinsic = sample['cam_intrinsic'][i]
        #     cam_intrinsic[0, 2] = cam_intrinsic[0, 2] * rand_scale
        #     cam_intrinsic[1, 2] = cam_intrinsic[1, 2] * rand_scale
        #     cam_intrinsic[0, 0] = cam_intrinsic[0, 0] * rand_scale
        #     cam_intrinsic[1, 1] = cam_intrinsic[1, 1] * rand_scale
        #     sample['cam_intrinsic'][i] = cam_intrinsic

        return sample



@manager.TRANSFORMS.add_component
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = imgs
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = img
        if 'img_depth' in results:
            if isinstance(results['img_depth'], list):
                # process multiple imgs in single frame
                imgs = np.ascontiguousarray(np.stack(results['img_depth'], axis=0))
                results['img_depth'] = imgs
            else:
                img = np.ascontiguousarray(results['img_depth'])
                results['img_depth'] = img
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = results['gt_semantic_seg'][None, ...]

        return results



@manager.TRANSFORMS.add_component
class DefaultFormatBundle3D(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(DefaultFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D 
        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'gt_border_masks' in results:
                    results['gt_border_masks'] = results['gt_border_masks'][
                        gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ], dtype=np.int64)
        results = super(DefaultFormatBundle3D, self).__call__(results)
        return results



@manager.TRANSFORMS.add_component
class MultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions
            for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool): Whether apply horizontal flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
        pcd_vertical_flip (bool): Whether apply vertical flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 pts_scale_ratio,
                 flip=False,
                 flip_direction='horizontal',
                 pcd_horizontal_flip=False,
                 pcd_vertical_flip=False):
        if isinstance(transforms, list):
            transforms = T.Compose(transforms)

        self.transforms = transforms
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio \
            if isinstance(pts_scale_ratio, list) else[float(pts_scale_ratio)]

        # assert is_list_of(self.img_scale, tuple)
        # assert is_list_of(self.pts_scale_ratio, float)

        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip

        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]

    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with \
                different scales and flips.
        """
        aug_data = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] \
            if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] \
            if self.flip and self.pcd_vertical_flip else [False]
        for scale in self.img_scale:
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results)
                                _results['scale'] = scale
                                _results['flip'] = flip
                                _results['pcd_scale_factor'] = \
                                    pts_scale_ratio
                                _results['flip_direction'] = direction
                                _results['pcd_horizontal_flip'] = \
                                    pcd_horizontal_flip
                                _results['pcd_vertical_flip'] = \
                                    pcd_vertical_flip
                                data = self.transforms(_results)
                                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

@manager.TRANSFORMS.add_component
class PointShuffle(TransformABC):
    """Shuffle input points."""

    def __call__(self, input_dict):
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' keys are updated \
                in the result dict.
        """
        input_dict['points'] = np.random.permutation(input_dict['points'])

        return input_dict



@manager.TRANSFORMS.add_component
class CustomRandomFlip3D(TransformABC):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 **kwargs):
        super(CustomRandomFlip3D, self).__init__()
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def flip(self, tensor, bev_direction='horizontal', points=None):
        """Flip the boxes in BEV along given BEV direction.

        In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, None):
                Points to flip. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or None: Flipped points.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            tensor[:, 1::7] = -tensor[:, 1::7]
            if tensor.shape[-1] > 6:
                tensor[:, 6] = -tensor[:, 6] + np.pi
        elif bev_direction == 'vertical':
            tensor[:, 0::7] = -tensor[:, 0::7]
            if tensor.shape[-1] > 6:
                tensor[:, 6] = -tensor[:, 6]

        if points is not None:
            if bev_direction == 'horizontal':
                points[:, 1] = -points[:, 1]
            elif bev_direction == 'vertical':
                points[:, 0] = -points[:, 0]
            return tensor, points
        return tensor

        
    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = np.array([], dtype=np.float32)
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                input_dict[key], input_dict['points'] = self.flip(input_dict[key], direction, points=input_dict['points'])
            else:
                input_dict[key] = self.flip(input_dict[key], direction)
            # if 'radar' in input_dict:
            #     input_dict['radar'].flip(direction)
                
        if 'roi_regions' in input_dict and len(input_dict['roi_regions']) > 0:
            for region in input_dict['roi_regions']:
                if region['type'] == 1:
                    raise NotImplementedError
                elif region['type'] == 2:
                    if direction == 'horizontal':
                        region['region'][1] = -region['region'][1] # x, y, z, radius
                    elif direction == 'vertical':
                        region['region'][0] = -region['region'][0] # x, y, z, radius
                elif region['type'] == 3:
                    self.flip(region['region'], direction)
                else:
                    raise NotImplementedError

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        if 'pcd_horizontal_flip' not in input_dict:
            flip_horizontal = True if np.random.rand(
            ) < self.flip_ratio_bev_horizontal else False
            input_dict['pcd_horizontal_flip'] = flip_horizontal
        if 'pcd_vertical_flip' not in input_dict:
            flip_vertical = True if np.random.rand(
            ) < self.flip_ratio_bev_vertical else False
            input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        flip_mat = np.eye(4)
        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
            flip_mat[1, 1] = -1
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])
            flip_mat[0, 0] = -1
        for view in range(len(input_dict["lidar2img"])):
            input_dict["lidar2img"][view] = input_dict["lidar2img"][view] @ flip_mat
            input_dict["lidar2cam"][view] = input_dict["lidar2cam"][view] @ flip_mat
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str


@manager.TRANSFORMS.add_component
class CalculateGTBoxVisibleBorders(TransformABC):
    # def __init__(self, **kwargs):

    def __call__(self, res):
        '''              
                         t                                        
        corner0 __________________corner3                                       
                |                 |                                                
                |                 |                                            
            l   |                 |  r                                            
                |                 |                                                 
        corner1 |_________________| corner2                                        
                        b                                                                    
        '''

        # print('#################-{}-################'.format(res['gt_bboxes_3d'].tensor.shape))

        boxes = res['gt_bboxes_3d']
        assert len(boxes.shape) == 2

        bev_boxes = boxes[:, [0, 1, 3, 4, 6]]
        bev_corners = self.get_box_corners(bev_boxes)
        bev_origins = np.zeros_like(bev_corners)
        
        #re_bev_corners_cat = np.concatenate([re_bev_corners, re_bev_corners[:, 0:1]], axis = 1)
        borders_start = np.concatenate([bev_corners, bev_corners[:, 0:2]], axis=1)
        borders_end   = np.concatenate([bev_corners[:, 1:], bev_corners[:, :3]], axis=1)

        cross_results = []
        for i in range(2):
            cross_result = self.cross(bev_corners, bev_origins, 
                                      borders_start[:, (i+1):(i+5)], borders_end[:, (i+1):(i+5)])
            cross_results.append(cross_result)
        cross_results = np.stack(cross_results, axis=2)
        cross_results = ~(cross_results[..., 0] + cross_results[..., 1]) # true: non-cross, false: cross
        pos_masks = np.concatenate([cross_results, cross_results[:, :1]], axis=1)

        # t, l, b, r
        border_masks = pos_masks[:, :4] * pos_masks[:, 1:]

        #self.draw_boxes(bev_corners, border_masks)
        res['gt_border_masks'] = border_masks

        return res

    def get_box_corners(self, boxes):
        anglePis = boxes[:, 4]
        cxs, cys = boxes[:, 0], boxes[:, 1]
        ws, ls   = boxes[:, 2], boxes[:, 3]

        rxs0 = cxs - (ws/2)*np.cos(anglePis) + (ls/2)*np.sin(anglePis)
        rys0 = cys - (ws/2)*np.sin(anglePis) - (ls/2)*np.cos(anglePis)

        rxs1 = cxs - (ws/2)*np.cos(anglePis) - (ls/2)*np.sin(anglePis)
        rys1 = cys - (ws/2)*np.sin(anglePis) + (ls/2)*np.cos(anglePis)
         
        rxs2 = cxs + (ws/2)*np.cos(anglePis) - (ls/2)*np.sin(anglePis)
        rys2 = cys + (ws/2)*np.sin(anglePis) + (ls/2)*np.cos(anglePis)

        rxs3 = cxs + (ws/2)*np.cos(anglePis) + (ls/2)*np.sin(anglePis)
        rys3 = cys + (ws/2)*np.sin(anglePis) - (ls/2)*np.cos(anglePis)

        rcorners0 = np.stack([rxs0, rys0], axis=1)
        rcorners1 = np.stack([rxs1, rys1], axis=1)
        rcorners2 = np.stack([rxs2, rys2], axis=1)
        rcorners3 = np.stack([rxs3, rys3], axis=1)
        rcorners  = np.stack([rcorners0, rcorners1, 
                              rcorners2, rcorners3], axis=1)
        return rcorners
    
    def cross(self, s_p1, s_p2, d_p1, d_p2):
        v1 = s_p1 - d_p1
        v2 = s_p2 - d_p1
        vm = d_p2 - d_p1
        flag1 = self.cross_multi(v1, vm) * self.cross_multi(v2, vm) <= 0
        
        v1 = d_p1 - s_p1
        v2 = d_p2 - s_p1
        vm = s_p2 - s_p1
        flag2 = self.cross_multi(v1, vm) * self.cross_multi(v2, vm) <= 0

        flag = flag1 * flag2
        return flag

    def cross_multi(self, v1, v2):
        # calculate cross multiply
        return v1[:, :, 0] * v2[:, :, 1] - v2[:, :, 0] * v1[:, :, 1]
    
    def draw_boxes(self, corners, masks):
        import matplotlib.pyplot as plt
        import matplotlib
        import random

        image=np.ones((240,240,3),np.uint8)*255
        im      = image[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12)) 
        fig     = ax.imshow(im, aspect='equal')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.scatter(120, 120, s=0.1, c='red')

        for i in range(corners.shape[0]):
            corner = corners[i]
            for j in range(4):
                plt.scatter(120+corner[j, 0], 120+corner[j, 1], s=0.1, c='black')
            corner_for_plot = np.concatenate([corner, corner[:1, :]], axis=0)
            plt.plot(120+corner_for_plot[:, 0], 120+corner_for_plot[:, 1], linewidth=0.1, color='black') 
            mask = masks[i]
            for j in range(4):
                if mask[j]:
                    plt.plot(120+corner_for_plot[j:j+2, 0], 120+corner_for_plot[j:j+2, 1], 
                             linewidth=0.2, color='red')

        plt.savefig('output/{}.pdf'.format(random.randint(0,1000000)))
        plt.close()

@manager.TRANSFORMS.add_component
class ObjectSample(TransformABC):
    """Sample GT objects to the data.
    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images.
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
        use_ground_plane (bool): Whether to use ground plane to adjust the
            3D labels. Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False, use_ground_plane=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        self.db_sampler = db_sampler
        self.use_ground_plane = use_ground_plane
        self.disabled = False

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.
        Args:
            points (np.ndarray): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.
        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = points_in_rbbox(points[:, :3], boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after object sampling augmentation,
            'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated
            in the result dict.
        """
        gt_names_3d = input_dict['gt_names_3d']
        if self.disabled:
            return input_dict

        all_region_is_type3 = False
        if 'roi_regions' in input_dict:
            all_region_is_type3 = all(x['type'] ==3 for x in input_dict['roi_regions'])
        if (not all_region_is_type3) and 'roi_regions' in input_dict and input_dict['roi_regions'] is not None and len(input_dict['roi_regions']) > 0:
            return input_dict

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        if self.use_ground_plane:
            ground_plane = input_dict.get('plane', None)
            assert ground_plane is not None, '`use_ground_plane` is True ' \
                                             'but find plane is None'
        else:
            ground_plane = None
        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d,
                gt_names_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d,
                gt_names_3d,
                img=None,
                ground_plane=ground_plane)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']
            sampled_gt_names = sampled_dict['gt_names_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_names_3d = np.concatenate([gt_names_3d, sampled_gt_names],
                                          axis=0)
            gt_bboxes_3d = np.concatenate(
                    [gt_bboxes_3d, sampled_gt_bboxes_3d])

            # points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            points = self.remove_points_in_boxes(points, sampled_dict['collision_boxes'])
            # check the points dimension
            points = np.concatenate([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.int64)
        input_dict['gt_names_3d'] = gt_names_3d
        input_dict['points'] = points
        
        return input_dict
