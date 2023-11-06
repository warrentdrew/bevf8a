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
from paddle3d.utils_idg.box_np_ops import points_in_rbbox, rotation_points_single_angle

__all__ = [
    "RandomHorizontalFlip", "RandomVerticalFlip", "GlobalRotate", "GlobalScale",
    "GlobalTranslate", "ShufflePoint", "SamplePoint",
    "FilterPointOutsideRange", "FilterBBoxOutsideRange", "HardVoxelize",
    "RandomObjectPerturb", "ConvertBoxFormat", "ResizeShortestEdge",
    "RandomContrast", "RandomBrightness", "RandomSaturation",
    "ToVisionBasedBox", "PhotoMetricDistortionMultiViewImage",
    "RandomScaleImageMultiViewImage"
]


@manager.TRANSFORMS.add_component
class RandomHorizontalFlip(TransformABC):
    """
    Note:
        If the inputs are pixel indices, they are flipped by `(W - 1 - x, H - 1 - y)`.
        If the inputs are floating point coordinates, they are flipped by `(W - x, H - y)`.
    """

    def __init__(self, prob: float = 0.5, input_type='pixel_indices'):
        self.prob = prob
        self.input_type = input_type

    def __call__(self, sample: Sample):
        # np.random.seed(0)
        if np.random.random() < self.prob:
            if sample.modality == "image":
                sample.data = F.horizontal_flip(sample.data)
                h, w, c = sample.data.shape
            elif sample.modality == "lidar":
                sample.data.flip(axis=1)

            if self.input_type == 'pixel_indices':
                # Flip camera intrinsics
                if "camera_intrinsic" in sample.meta:
                    sample.meta.camera_intrinsic[
                        0, 2] = w - sample.meta.camera_intrinsic[0, 2] - 1

                # Flip bbox
                if sample.bboxes_3d is not None:
                    sample.bboxes_3d.horizontal_flip()
                if sample.bboxes_2d is not None and sample.modality == "image":
                    sample.bboxes_2d.horizontal_flip(image_width=w)

            elif self.input_type == 'floating_point_coordinates':
                # Flip camera intrinsics
                if "camera_intrinsic" in sample.meta:
                    sample.meta.camera_intrinsic[
                        0, 2] = w - sample.meta.camera_intrinsic[0, 2]

                # Flip bbox
                if sample.bboxes_3d is not None:
                    sample.bboxes_3d.horizontal_flip_coords()
                if sample.bboxes_2d is not None and sample.modality == "image":
                    sample.bboxes_2d.horizontal_flip_coords(image_width=w)
        return sample


@manager.TRANSFORMS.add_component
class ToVisionBasedBox(TransformABC):
    """
    as name
    """

    def __call__(self, sample: Sample):
        bboxes_3d_new = sample.bboxes_3d.to_vision_based_3d_box()
        sample.bboxes_3d = BBoxes3D(
            bboxes_3d_new,
            origin=[.5, 1, .5],
            coordmode=CoordMode.KittiCamera,
            rot_axis=1)
        return sample


@manager.TRANSFORMS.add_component
class RandomVerticalFlip(TransformABC):
    """
    as name
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, sample: Sample):
        if np.random.random() < self.prob:
            # np.random.seed(0)
            if sample.modality == "image":
                sample.data = F.vertical_flip(sample.data)
                h, w, c = sample.data.shape
            elif sample.modality == "lidar":
                sample.data.flip(axis=0)

            # Flip camera intrinsics
            if "camera_intrinsic" in sample.meta:
                sample.meta.camera_intrinsic[
                    1, 2] = h - sample.meta.camera_intrinsic[1, 2] - 1

            # Flip bbox
            if sample.bboxes_3d is not None:
                sample.bboxes_3d.vertical_flip()
            if sample.bboxes_2d is not None and sample.modality == "image":
                sample.bboxes_2d.vertical_flip(image_height=h)

        return sample


@manager.TRANSFORMS.add_component
class GlobalRotate(TransformABC):
    """
    as name
    """

    def __init__(self, min_rot: float = -np.pi / 4, max_rot: float = np.pi / 4):
        self.min_rot = min_rot
        self.max_rot = max_rot

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("GlobalRotate only supports lidar data!")
        # np.random.seed(0)
        angle = np.random.uniform(self.min_rot, self.max_rot)
        # Rotate points
        sample.data.rotate_around_z(angle)
        # Rotate bboxes_3d
        if sample.bboxes_3d is not None:
            sample.bboxes_3d.rotate_around_z(angle)
        return sample


@manager.TRANSFORMS.add_component
class GlobalScale(TransformABC):
    """
    as name
    """

    def __init__(self,
                 min_scale: float = 0.95,
                 max_scale: float = 1.05,
                 size=None):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.size = size

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("GlobalScale only supports lidar data!")
        factor = np.random.uniform(
            self.min_scale, self.max_scale, size=self.size)
        # Scale points
        sample.data.scale(factor)
        # Scale bboxes_3d
        if sample.bboxes_3d is not None:
            sample.bboxes_3d.scale(factor)
        return sample


@manager.TRANSFORMS.add_component
class GlobalTranslate(TransformABC):
    """
    Translate sample by a random offset.

    Args:
        translation_std (Union[float, List[float], Tuple[float]], optional):
            The standard deviation of the translation offset. Defaults to (.2, .2, .2).
        distribution (str):
            The random distribution. Defaults to normal.
    """

    def __init__(
            self,
            translation_std: Union[float, List[float], Tuple[float]] = (.2, .2,
                                                                        .2),
            distribution="normal"):
        if not isinstance(translation_std, (list, tuple)):
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        self.translation_std = translation_std
        self.distribution = distribution

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("GlobalScale only supports lidar data!")
        if self.distribution not in ["normal", "uniform"]:
            raise ValueError(
                "GlobalScale only supports normal and uniform random distribution!"
            )

        if self.distribution == "normal":
            translation = np.random.normal(scale=self.translation_std, size=3)
        elif self.distribution == "uniform":
            translation = np.random.uniform(
                low=-self.translation_std[0],
                high=self.translation_std[0],
                size=3)
        else:
            raise ValueError(
                "GlobalScale only supports normal and uniform random distribution!"
            )

        sample.data.translate(translation)
        if sample.bboxes_3d is not None:
            sample.bboxes_3d.translate(translation)

        return sample


@manager.TRANSFORMS.add_component
class ShufflePoint(TransformABC):
    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("ShufflePoint only supports lidar data!")
        sample.data.shuffle()
        return sample


@manager.TRANSFORMS.add_component
class ConvertBoxFormat(TransformABC):
    def __call__(self, sample: Sample):
        # convert boxes from [x,y,z,w,l,h,yaw] to [x,y,z,l,w,h,heading], bottom_center -> obj_center
        bboxes_3d = box_utils.boxes3d_kitti_lidar_to_lidar(sample.bboxes_3d)

        # limit heading
        bboxes_3d[:, -1] = box_utils.limit_period(
            bboxes_3d[:, -1], offset=0.5, period=2 * np.pi)

        # stack labels into gt_boxes, label starts from 1, instead of 0.
        labels = sample.labels + 1
        bboxes_3d = np.concatenate(
            [bboxes_3d, labels.reshape(-1, 1).astype(np.float32)], axis=-1)
        sample.bboxes_3d = bboxes_3d
        sample.pop('labels', None)

        return sample


@manager.TRANSFORMS.add_component
class SamplePoint(TransformABC):
    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, sample: Sample):
        sample = F.sample_point(sample, self.num_points)

        return sample

@manager.TRANSFORMS.add_component
class FilterBBoxOutsideRange(TransformABC):
    def __init__(self, point_cloud_range: Tuple[float]):
        self.point_cloud_range = np.asarray(point_cloud_range, dtype='float32')

    def __call__(self, sample: Sample):
        if sample.bboxes_3d.size == 0:
            return sample
        mask = sample.bboxes_3d.get_mask_of_bboxes_outside_range(
            self.point_cloud_range)
        sample.bboxes_3d = sample.bboxes_3d.masked_select(mask)
        sample.labels = sample.labels[mask]
        return sample


@manager.TRANSFORMS.add_component
class FilterPointOutsideRange(TransformABC):
    def __init__(self, point_cloud_range: Tuple[float]):
        self.point_cloud_range = np.asarray(point_cloud_range, dtype='float32')

    def __call__(self, sample: Sample):
        mask = sample.data.get_mask_of_points_outside_range(
            self.point_cloud_range)
        sample.data = sample.data[mask]
        return sample


@manager.TRANSFORMS.add_component
class HardVoxelize(TransformABC):
    def __init__(self, point_cloud_range: Tuple[float],
                 voxel_size: Tuple[float], max_points_in_voxel: int,
                 max_voxel_num: int):
        self.max_points_in_voxel = max_points_in_voxel
        self.max_voxel_num = max_voxel_num
        self.voxel_size = np.asarray(voxel_size, dtype='float32')
        self.point_cloud_range = np.asarray(point_cloud_range, dtype='float32')
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) /
            self.voxel_size).astype('int32')

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("Voxelize only supports lidar data!")

        # Voxelize
        num_points, num_point_dim = sample.data.shape[0:2]
        voxels = np.zeros(
            (self.max_voxel_num, self.max_points_in_voxel, num_point_dim),
            dtype=sample.data.dtype)
        coords = np.zeros((self.max_voxel_num, 3), dtype=np.int32)
        num_points_per_voxel = np.zeros((self.max_voxel_num, ), dtype=np.int32)
        grid_size_z, grid_size_y, grid_size_x = self.grid_size[::-1]
        grid_idx_to_voxel_idx = np.full((grid_size_z, grid_size_y, grid_size_x),
                                        -1,
                                        dtype=np.int32)

        num_voxels = points_to_voxel(
            sample.data, self.voxel_size, self.point_cloud_range,
            self.grid_size, voxels, coords, num_points_per_voxel,
            grid_idx_to_voxel_idx, self.max_points_in_voxel, self.max_voxel_num)

        voxels = voxels[:num_voxels]
        coords = coords[:num_voxels]
        num_points_per_voxel = num_points_per_voxel[:num_voxels]

        sample.voxels = voxels
        sample.coords = coords
        sample.num_points_per_voxel = num_points_per_voxel

        sample.pop('sweeps', None)
        return sample


@manager.TRANSFORMS.add_component
class RandomObjectPerturb(TransformABC):
    """
    Randomly perturb (rotate and translate) each object.

    Args:
        rotation_range (Union[float, List[float], Tuple[float]], optional):
            Range of random rotation. Defaults to pi / 4.
        translation_std (Union[float, List[float], Tuple[float]], optional):
            Standard deviation of random translation. Defaults to 1.0.
        max_num_attempts (int): Maximum number of perturbation attempts. Defaults to 100.
    """

    def __init__(
            self,
            rotation_range: Union[float, List[float], Tuple[float]] = np.pi / 4,
            translation_std: Union[float, List[float], Tuple[float]] = 1.0,
            max_num_attempts: int = 100):

        if not isinstance(rotation_range, (list, tuple)):
            rotation_range = [-rotation_range, rotation_range]
        self.rotation_range = rotation_range
        if not isinstance(translation_std, (list, tuple)):
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        self.translation_std = translation_std
        self.max_num_attempts = max_num_attempts

    def __call__(self, sample: Sample):
        num_objects = sample.bboxes_3d.shape[0]
        rotation_noises = np.random.uniform(
            self.rotation_range[0],
            self.rotation_range[1],
            size=[num_objects, self.max_num_attempts])
        translation_noises = np.random.normal(
            scale=self.translation_std,
            size=[num_objects, self.max_num_attempts, 3])
        rotation_noises, translation_noises = F.noise_per_box(
            sample.bboxes_3d[:, [0, 1, 3, 4, 6]], sample.bboxes_3d.corners_2d,
            sample.ignored_bboxes_3d.corners_2d, rotation_noises,
            translation_noises)

        # perturb points w.r.t objects' centers (inplace operation)
        normals = F.corner_to_surface_normal(sample.bboxes_3d.corners_3d)
        point_masks = points_in_convex_polygon_3d_jit(sample.data[:, :3],
                                                      normals)
        F.perturb_object_points_(sample.data, sample.bboxes_3d[:, :3],
                                 point_masks, rotation_noises,
                                 translation_noises)

        # perturb bboxes_3d w.r.t to objects' centers (inplace operation)
        F.perturb_object_bboxes_3d_(sample.bboxes_3d, rotation_noises,
                                    translation_noises)

        return sample


@manager.TRANSFORMS.add_component
class ResizeShortestEdge(TransformABC):
    """
    as name
    """

    def __init__(self,
                 short_edge_length,
                 max_size,
                 sample_style="range",
                 interp=Image.BILINEAR):
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.interp = interp
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!")

    def __call__(self, sample: Sample):
        h, w = sample.data.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0],
                                     self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        newh, neww = self.get_output_shape(h, w, size, self.max_size)
        sample.data = self.apply_image(sample.data, h, w, newh, neww)
        sample.image_sizes = np.asarray((h, w))
        if "camera_intrinsic" in sample.meta:
            sample.meta.camera_intrinsic = self.apply_intrinsics(
                sample.meta.camera_intrinsic, h, w, newh, neww)
        if sample.bboxes_2d is not None and sample.modality == "image":
            sample.bboxes_2d.resize(h, w, newh, neww)
        return sample

    def apply_image(self, img, h, w, newh, neww):
        assert len(img.shape) <= 4

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((neww, newh), self.interp)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = paddle.to_tensor(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.reshape(shape_4d).transpose([2, 3, 0, 1])  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[self.interp]
            align_corners = None if mode == "nearest" else False
            img = nn.functional.interpolate(
                img, (newh, neww), mode=mode, align_corners=align_corners)
            shape[:2] = (newh, neww)
            ret = img.transpose([2, 3, 0,
                                 1]).reshape(shape).numpy()  # nchw -> hw(c)

        return ret

    def apply_intrinsics(self, intrinsics, h, w, newh, neww):
        assert intrinsics.shape == (3, 3)
        assert intrinsics[0, 1] == 0  # undistorted
        assert np.allclose(intrinsics,
                           np.triu(intrinsics))  # check if upper triangular

        factor_x = neww / w
        factor_y = newh / h
        new_intrinsics = intrinsics * np.float32([factor_x, factor_y, 1
                                                  ]).reshape(3, 1)
        return new_intrinsics

    @staticmethod
    def get_output_shape(oldh: int, oldw: int, short_edge_length: int,
                         max_size: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.
        """
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


@manager.TRANSFORMS.add_component
class RandomContrast(TransformABC):
    """
    Randomly transforms image contrast.
    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast
    """

    def __init__(self, intensity_min: float, intensity_max: float):
        super().__init__()
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def __call__(self, sample: Sample):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        sample.data = F.blend_transform(
            sample.data,
            src_image=sample.data.mean(),
            src_weight=1 - w,
            dst_weight=w)
        return sample


@manager.TRANSFORMS.add_component
class RandomBrightness(TransformABC):
    """
    Randomly transforms image contrast.
    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast
    """

    def __init__(self, intensity_min: float, intensity_max: float):
        super().__init__()
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def __call__(self, sample: Sample):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        sample.data = F.blend_transform(
            sample.data, src_image=0, src_weight=1 - w, dst_weight=w)
        return sample


@manager.TRANSFORMS.add_component
class RandomSaturation(TransformABC):
    """
    Randomly transforms image contrast.
    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast
    """

    def __init__(self, intensity_min: float, intensity_max: float):
        super().__init__()
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def __call__(self, sample: Sample):
        assert sample.data.shape[
            -1] == 3, "RandomSaturation only works on RGB images"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = sample.data.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        sample.data = F.blend_transform(
            sample.data, src_image=grayscale, src_weight=1 - w, dst_weight=w)
        return sample


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
        gt_names_3d = np.array(gt_names_3d)[mask.astype(np.bool_)]
        
        # limit rad to [-pi, pi]
        gt_bboxes_3d = self.limit_yaw(
            gt_bboxes_3d, offset=0.5, period=2 * np.pi)
        sample['gt_bboxes_3d'] = gt_bboxes_3d
        sample['gt_labels_3d'] = gt_labels_3d
        sample['gt_names_3d'] = gt_names_3d
        
        return sample




@manager.TRANSFORMS.add_component
class ObjectRangeFilter(object):
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
        gt_names_3d = np.array(gt_names_3d)[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d = self.limit_yaw(
            gt_bboxes_3d, offset=0.5, period=2 * np.pi)
        sample['gt_bboxes_3d'] = gt_bboxes_3d
        sample['gt_labels_3d'] = gt_labels_3d
        sample['gt_names_3d'] = gt_names_3d

        return sample


@manager.TRANSFORMS.add_component
class ObjectRangeFilter_valid(object):
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
        valid_flag = sample['valid_flag']

        mask = self.in_range_bev(bev_range, gt_bboxes_3d)
        gt_bboxes_3d = gt_bboxes_3d[mask]

        gt_labels_3d = gt_labels_3d[mask.astype(np.bool_)]
        valid_flag = valid_flag[mask.astype(np.bool_)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d = self.limit_yaw(
            gt_bboxes_3d, offset=0.5, period=2 * np.pi)
        sample['gt_bboxes_3d'] = gt_bboxes_3d
        sample['gt_labels_3d'] = gt_labels_3d
        sample['valid_flag'] = valid_flag

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
class ObjectNameFilter_valid(object):
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
        sample['valid_flag'] = sample['valid_flag'][gt_bboxes_mask]

        return sample


@manager.TRANSFORMS.add_component
class ResizeCropFlipImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, sample_aug_cfg=None, training=True):
        self.sample_aug_cfg = sample_aug_cfg

        self.training = training

    def __call__(self, sample):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        imgs = sample["img"]
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()  ###different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            sample['intrinsics'][
                i][:3, :3] = ida_mat @ sample['intrinsics'][i][:3, :3]

        sample["img"] = new_imgs
        sample['lidar2img'] = [
            sample['intrinsics'][i] @ sample['extrinsics'][i].T
            for i in range(len(sample['extrinsics']))
        ]

        return sample

    def _get_rot(self, h):

        return np.array([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = np.eye(2)
        ida_tran = np.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= np.array(crop[:2])
        if flip:
            A = np.array([[-1, 0], [0, 1]])
            b = np.array([crop[2] - crop[0], 0])

            ida_rot = np.matmul(A, ida_rot)
            ida_tran = np.matmul(A, ida_tran) + b

        A = self._get_rot(rotate / 180 * np.pi)
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = np.matmul(A, -b) + b
        ida_rot = np.matmul(A, ida_rot)
        ida_tran = np.matmul(A, ida_tran) + b
        ida_mat = np.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.sample_aug_cfg["H"], self.sample_aug_cfg["W"]
        fH, fW = self.sample_aug_cfg["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.sample_aug_cfg["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.sample_aug_cfg["bot_pct_lim"])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.sample_aug_cfg["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.sample_aug_cfg["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.sample_aug_cfg["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@manager.TRANSFORMS.add_component
class MSResizeCropFlipImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self,
                 sample_aug_cfg=None,
                 training=True,
                 view_num=1,
                 center_size=2.0):
        self.sample_aug_cfg = sample_aug_cfg
        self.training = training
        self.view_num = view_num
        self.center_size = center_size

    def __call__(self, sample):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            sample (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        imgs = sample["img"]
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()

        copy_intrinsics = []
        copy_extrinsics = []
        for i in range(self.view_num):
            copy_intrinsics.append(np.copy(sample['intrinsics'][i]))
            copy_extrinsics.append(np.copy(sample['extrinsics'][i]))

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            sample['intrinsics'][
                i][:3, :3] = ida_mat @ sample['intrinsics'][i][:3, :3]

        resize, resize_dims, crop, flip, rotate = self._crop_augmentation(
            resize)
        for i in range(self.view_num):
            img = Image.fromarray(np.copy(np.uint8(imgs[i])))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            copy_intrinsics[i][:3, :3] = ida_mat @ copy_intrinsics[i][:3, :3]
            sample['intrinsics'].append(copy_intrinsics[i])
            sample['extrinsics'].append(copy_extrinsics[i])
            sample['filename'].append(sample['filename'][i].replace(
                ".jpg", "_crop.jpg"))
            sample['timestamp'].append(sample['timestamp'][i])

        sample["img"] = new_imgs
        sample['lidar2img'] = [
            sample['intrinsics'][i] @ sample['extrinsics'][i].T
            for i in range(len(sample['extrinsics']))
        ]
        return sample

    def _get_rot(self, h):

        return np.array([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = np.eye(2)
        ida_tran = np.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= np.array(crop[:2])
        if flip:
            A = np.array([[-1, 0], [0, 1]])
            b = np.array([crop[2] - crop[0], 0])

            ida_rot = np.matmul(A, ida_rot)
            ida_tran = np.matmul(A, ida_tran) + b

        A = self._get_rot(rotate / 180 * np.pi)
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = np.matmul(A, -b) + b
        ida_rot = np.matmul(A, ida_rot)
        ida_tran = np.matmul(A, ida_tran) + b
        ida_mat = np.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.sample_aug_cfg["H"], self.sample_aug_cfg["W"]
        fH, fW = self.sample_aug_cfg["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.sample_aug_cfg["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.sample_aug_cfg["bot_pct_lim"])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.sample_aug_cfg["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.sample_aug_cfg["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.sample_aug_cfg["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def _crop_augmentation(self, resize):
        H, W = self.sample_aug_cfg["H"], self.sample_aug_cfg["W"]
        fH, fW = self.sample_aug_cfg["final_dim"]
        resize = self.center_size * resize
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int(max(0, newH - fH) / 2)
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        return resize, resize_dims, crop, flip, rotate


@manager.TRANSFORMS.add_component
class GlobalRotScaleTransImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
            self,
            rot_range=[-0.3925, 0.3925],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0, 0, 0],
            reverse_angle=False,
            training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        # random rotate
        rot_angle = np.random.uniform(*self.rot_range)

        self.rotate_bev_along_z(results, rot_angle)

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio)
        # results["gt_bboxes_3d"].scale(scale_ratio)

        # TODO: support translation

        return results

    def rotate_bev_along_z(self, results, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array([[rot_cos, -rot_sin, 0, 0], [rot_sin, rot_cos, 0, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.array(
                results["lidar2img"][view]).astype('float32') @ rot_mat_inv
            results["extrinsics"][view] = np.array(
                results["extrinsics"][view]).astype('float32') @ rot_mat_inv

        if self.reverse_angle:
            rot_angle = np.array(-1 * angle)
        else:
            rot_angle = np.array(angle)

        rot_cos = np.cos(rot_angle)
        rot_sin = np.sin(rot_angle)

        rot_mat = np.array([[
            rot_cos,
            -rot_sin,
            0,
        ], [
            rot_sin,
            rot_cos,
            0,
        ], [0, 0, 1]])
        results.gt_bboxes_3d[:, :3] = results.gt_bboxes_3d[:, :3] @ rot_mat
        results.gt_bboxes_3d[:, 6] += rot_angle
        results.gt_bboxes_3d[:, 7:
                             9] = results.gt_bboxes_3d[:, 7:9] @ rot_mat[:2, :2]

    def scale_xyz(self, results, scale_ratio):
        rot_mat = np.array([
            [scale_ratio, 0, 0, 0],
            [0, scale_ratio, 0, 0],
            [0, 0, scale_ratio, 0],
            [0, 0, 0, 1],
        ])

        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.array(
                results["lidar2img"][view]).astype('float32') @ rot_mat_inv
            results["extrinsics"][view] = np.array(
                rot_mat_inv.T @ results["extrinsics"][view]).astype('float32')

        results.gt_bboxes_3d[:, :6] *= scale_ratio
        results.gt_bboxes_3d[:, 7:] *= scale_ratio
        return


@manager.TRANSFORMS.add_component
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, sample):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        sample['img'] = [
            F.normalize_use_cv2(img, self.mean, self.std, self.to_rgb)
            for img in sample['img']
        ]
        sample['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)

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


def impad(img, *, shape=None, padding=None, pad_val=0, padding_mode='constant'):
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    return img


def impad_to_multiple(img, divisor, pad_val=0):
    """Pad an image to ensure each edge to be multiple to some number.
    """
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, shape=(pad_h, pad_w), pad_val=pad_val)


@manager.TRANSFORMS.add_component
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def __call__(self, sample):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if self.size is not None:
            padded_img = [
                impad(img, shape=self.size, pad_val=self.pad_val)
                for img in sample['img']
            ]
        elif self.size_divisor is not None:
            padded_img = [
                impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in sample['img']
            ]
        sample['img_shape'] = [img.shape for img in sample['img']]
        sample['img'] = padded_img
        sample['pad_shape'] = [img.shape for img in padded_img]
        sample['pad_fixed_size'] = self.size
        sample['pad_size_divisor'] = self.size_divisor
        return sample


@manager.TRANSFORMS.add_component
class MyPad(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, sample):
        """Pad images according to ``self.size``."""
        for key in sample.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = impad(
                    sample[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                for idx in range(len(sample[key])):
                    padded_img = impad_to_multiple(
                        sample[key][idx], self.size_divisor, pad_val=self.pad_val)
                    sample[key][idx] = padded_img
        sample['pad_shape'] = padded_img.shape
        sample['pad_fixed_size'] = self.size
        sample['pad_size_divisor'] = self.size_divisor
    
    def _pad_masks(self, sample):
        """Pad masks according to ``sample['pad_shape']``."""
        pad_shape = sample['pad_shape'][:2]
        for key in sample.get('mask_fields', []):
            sample[key] = sample[key].pad(pad_shape, pad_val=self.pad_val)
    
    def _pad_seg(self, sample):
        """Pad semantic segmentation map according to
        ``sample['pad_shape']``."""
        for key in sample.get('seg_fields', []):
            sample[key] = impad(
                sample[key], shape=sample['pad_shape'][:2])

    def __call__(self, sample):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(sample)
        self._pad_masks(sample)
        self._pad_seg(sample)
        return sample


@manager.TRANSFORMS.add_component
class SampleFilerByKey(object):
    """Collect data from the loader relevant to the specific task.
    """

    def __init__(
            self,
            keys,
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                       'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                       'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                       'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                       'rect', 'Trv2c', 'P2', 'caminfo', 'lidar2cam', 'cam_intrinsic',
                       'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                       'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                       'transformation_3d_flow', 'scene_token', 'can_bus')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, sample):
        """Call function to filter sample by keys. The keys in ``meta_keys``

        Args:
            sample (dict): Result dict contains the data.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
        filtered_sample = Sample(path=sample.path, modality=sample.modality)
        filtered_sample.meta.id = sample.meta.id

        for key in self.meta_keys:
            if key in sample:
                filtered_sample.meta[key] = sample[key]

        for key in self.keys:
            filtered_sample[key] = sample[key]
        return filtered_sample


@manager.TRANSFORMS.add_component
class PhotoMetricDistortionMultiViewImage(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of np.random.contrast is in
    second or second to last.
    1. np.random.brightness
    2. np.random.contrast (mode 0)
    3. convert color from BGR to HSV
    4. np.random.saturation
    5. np.random.hue
    6. convert color from HSV to BGR
    7. np.random.contrast (mode 1)
    8. np.random.y swap channels

    This class is modified from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L99

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert_color_factory(self, src, dst):

        code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')

        def convert_color(img):
            out_img = cv2.cvtColor(img, code)
            return out_img

        convert_color.__doc__ = f"""Convert a {src.upper()} image to {dst.upper()}
            image.

        Args:
            img (ndarray or str): The input image.

        Returns:
            ndarray: The converted {dst.upper()} image.
        """

        return convert_color

    def __call__(self, sample):
        """Call function to perform photometric distortion on images.
        Args:
            sample (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = sample['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # np.random.brightness
            if np.random.randint(2):
                delta = np.random.uniform(-self.brightness_delta,
                                          self.brightness_delta)
                img += delta

            # mode == 0 --> do np.random.contrast first
            # mode == 1 --> do np.random.contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                              self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = self.convert_color_factory('bgr', 'hsv')(img)

            # np.random.saturation
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(self.saturation_lower,
                                                 self.saturation_upper)

            # np.random.hue
            if np.random.randint(2):
                img[..., 0] += np.random.uniform(-self.hue_delta,
                                                 self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = self.convert_color_factory('hsv', 'bgr')(img)

            # np.random.contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                              self.contrast_upper)
                    img *= alpha

            # np.random.y swap channels
            if np.random.randint(2):
                img = img[..., np.random.permutation(3)]
            new_imgs.append(img)
        sample['img'] = new_imgs
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
                # np.random.seed(0)
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
            #print("x_size.len: {}, sample['lidar2img'].len: {}".format(len(x_size), len(sample['lidar2img'])))
            for l2i in sample['lidar2img']:
                lidar2img.append(scale_factors[idx] @ l2i)
                #print("lidar2img[{}]: {}".format(idx, lidar2img[idx]))
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


def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None,
             backend=None):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``cv2`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]
    if backend is None:
        backend = 'cv2'
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported for resize.'
                         f"Supported backends are 'cv2', 'pillow'")

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

if Image is not None:
    pillow_interp_codes = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'box': Image.BOX,
        'lanczos': Image.LANCZOS,
        'hamming': Image.HAMMING
    }


@manager.TRANSFORMS.add_component
class DefaultFormatBundle_valid(TransformABC):
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
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = results['gt_semantic_seg'][None, ...]

        return results

    def __repr__(self):
        return self.__class__.__name__



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
            # print(isinstance(results['img'], list), 'format')
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                # print(len(results['img']), [im.shape for im in results['img']], 'list')
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = imgs
            else:
                # print(results['img'].shape, 'notlist')
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = img
        if 'img_depth' in results:
            # print(isinstance(results['img'], list), 'format')
            if isinstance(results['img_depth'], list):
                # process multiple imgs in single frame
                # print(len(results['img']), [im.shape for im in results['img']], 'list')
                imgs = np.ascontiguousarray(np.stack(results['img_depth'], axis=0))
                results['img_depth'] = imgs
            else:
                # print(results['img'].shape, 'notlist')
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
class DefaultFormatBundle3D_valid(DefaultFormatBundle_valid):
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
        super(DefaultFormatBundle3D_valid, self).__init__()
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
                    ],
                                                       dtype=np.int64)
        results = super(DefaultFormatBundle3D_valid, self).__call__(results)
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
class GlobalRotScaleTransBEV(TransformABC):
    def __init__(self, resize_lim, rot_lim, trans_lim, is_train):
        self.resize_lim = resize_lim
        self.rot_lim = rot_lim
        self.trans_lim = trans_lim
        self.is_train = is_train
    
    def rotate(self, theta, gt_boxes, points=None):
        rot_sin = np.sin(theta)
        rot_cos = np.cos(theta)
        rot_mat_T = np.array([[rot_cos, -rot_sin, 0],
                                            [rot_sin, rot_cos, 0], [0, 0, 1]])

        gt_boxes[:, :3] = np.matmul(gt_boxes[:, :3], rot_mat_T)
        gt_boxes[:, 6] += theta

        if gt_boxes.shape[1] == 9:
            # rotate velo vector
            gt_boxes[:, 7:9] = np.matmul(gt_boxes[:, 7:9], rot_mat_T[:2, :2])

        rot_mat_T_ = None
        if points is not None:
            rot_sin_ = np.sin(-theta)
            rot_cos_ = np.cos(-theta)
            rot_mat_T_ = np.array([[rot_cos_, -rot_sin_, 0],
                                                 [rot_sin_, rot_cos_, 0],
                                                 [0, 0, 1]])
            rot_mat_T_ = rot_mat_T_.T
            points[:, :3] = np.matmul(points[:, :3], rot_mat_T)                                   

        return gt_boxes, points, rot_mat_T_
    
    def boxes_scale_trans(self, boxes, scale, translation):
        boxes[:, :3] += translation
        boxes[:, :6] *= scale
        boxes[:, 7:] *= scale
        return boxes
    
    def points_scale_trans(self, points, scale, translation):
        points[:, :3] += translation
        points[:, :3] *= scale
        return points

    def __call__(self, data):
        transform = np.eye(4).astype(np.float32)

        if self.is_train:
            scale = random.uniform(*self.resize_lim)
            theta = random.uniform(*self.rot_lim)
            translation = np.array([random.normal(0, self.trans_lim) for i in range(3)])
            rotation = np.eye(3)

            if "points" in data:
                data["points"] = self.points_scale_trans(data["points"], scale, translation)

            gt_boxes, points, rotation = self.rotate(theta, data["gt_bboxes_3d"], data['points'])
            gt_boxes = self.boxes_scale_trans(gt_boxes, scale, translation)
            data["gt_bboxes_3d"] = gt_bboxes_3d = BBoxes3D(gt_boxes)
            data["points"] = points

            # =================
            # 8A
            if 'roi_regions' in data and len(data['roi_regions']) > 0:
                for region in data['roi_regions']:
                    if region['type'] == 1:
                        raise NotImplementedError
                    elif region['type'] == 2:
                        region['region'][:3] = rotation_points_single_angle(
                                (region['region'].reshape((1, -1)))[:, :3], theta, axis=2
                            )[0]
                        region['region'][:3] += translation # x, y, z, radius
                        region['region'][:] *= scale
                    elif region['type'] == 3:
                        # region['region'].rotate(theta)
                        # region['region'].translate(translation)
                        # region['region'].scale(scale)
                        region['region'] = self.rotate(theta, region['region'])
                        region['region'] = self.boxes_scale_trans(region['region'], scale, translation)

                    else:
                        raise NotImplementedError
            # ======================
            transform[:3, :3] = rotation.T * scale
            transform[:3, 3] = translation * scale

        data["lidar_aug_matrix"] = transform
        return data


@manager.TRANSFORMS.add_component
class RandomFlip3DBEVHori(TransformABC):
    def points_flip(self, points):
        points[:, 1] = -points[:, 1]
        return points
    
    def boxes_3d_flip(self, bboxes_3d):
        bboxes_3d[:, 1::7] = -bboxes_3d[:, 1::7]
        bboxes_3d[:, 6] = -bboxes_3d[:, 6] + np.pi
        return bboxes_3d

    def __call__(self, data):
        flip_horizontal = random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"] = self.points_flip(data["points"])
            if "gt_bboxes_3d" in data:  # yaw
                data["gt_bboxes_3d"] = self.boxes_3d_flip(data["gt_bboxes_3d"])
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][:, :, ::-1].copy()

        data["lidar_aug_matrix"][:3, :] = rotation @ data["lidar_aug_matrix"][:3, :]
        return data

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
                
                # 8A
                elif region['type'] == 3:
                    # region['region'].flip(direction)
                    region['region'] = self.flip(region['region'], direction)
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
            # np.random.seed(0)
            flip_horizontal = True if np.random.rand(
            ) < self.flip_ratio_bev_horizontal else False
            input_dict['pcd_horizontal_flip'] = flip_horizontal
        if 'pcd_vertical_flip' not in input_dict:
            # np.random.seed(0)
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
class ModalMask3D(TransformABC):
    def __init__(self, mode='test', mask_modal='image', **kwargs):
        super(ModalMask3D, self).__init__()
        self.mode = mode
        self.mask_modal = mask_modal
        
    def in_range_3d(self, each_point, point_range):
        in_range_flags = ((each_point[:, 0] > point_range[0])
                          & (each_point[:, 1] > point_range[1])
                          & (each_point[:, 2] > point_range[2])
                          & (each_point[:, 0] < point_range[3])
                          & (each_point[:, 1] < point_range[4])
                          & (each_point[:, 2] < point_range[5]))
        return in_range_flags
    
    def __call__(self, input_dict):
        if self.mode == 'test':
            if self.mask_modal == 'image':
                input_dict['img'] = [0. * item for item in input_dict['img']]
            elif self.mask_modal == 'points':
                input_dict['points'] = input_dict['points'] * 0.0
            elif self.mask_modal == 'points_front':
                # miss front point clouds
                miss_range = [0, -60, -5, 120, 60, 3]
                tmp = input_dict['points']
                miss_flag = self.in_range_3d(tmp, miss_range) 
                input_dict['points'] = tmp[~miss_flag]
            else:
                raise NotImplementedError
        elif self.mode == 'train':
            # np.random.seed(0)
            seed = np.random.rand()
            if seed > 0.75:
                input_dict['img'] = [0. * item for item in input_dict['img']]
            elif seed > 0.5:
                # range-mask training
                miss_range_list = [[0, -60, -5, 120, 60, 3],
                              [-120, -60, -5, 0, 60, 3],
                              [-60, 0, -5, 60, 120, 3],
                              [-60, -120, -5, 60, 0, 3]]
                # np.random.seed(0)
                miss_range = miss_range_list[np.random.randint(len(miss_range_list))]
                tmp = input_dict['points']
                miss_flag = self.in_range_3d(tmp, miss_range)
                input_dict['points'] = tmp[~miss_flag]
            elif seed > 0.25:
                import random
                # partial points mask training
                tmp = input_dict['points']
                num_points, dims = tmp.shape
                sampled_points = int(num_points * 0.25)
                index = random.sample(range(num_points), sampled_points)
                input_dict['points'] = np.take(tmp, index, axis=0)
        else:
            raise NotImplementedError
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
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
        # gt_names_3d = input_dict.pop('gt_names_3d')
        # 8A
        gt_names_3d = input_dict['gt_names_3d']
        if self.disabled:
            return input_dict

        all_region_is_type3 = False

        # if 'roi_regions' in input_dict and len(input_dict['roi_regions']) > 0:
        #     return input_dict
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
            # check the points dimension

            points = self.remove_points_in_boxes(points, sampled_dict['collision_boxes'])
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
