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
import numpy as np
import paddle
import paddle.nn as nn
from paddle3d.apis import manager
from paddle.autograd import PyLayer
from paddle3d.utils_idg.build_layer import build_norm_layer
from .utils_idg import  PFNLayer, get_paddings_indicator
from paddle3d.utils_idg.ops.scatter_point import DynamicScatter
from paddle3d.models.layers import param_init, reset_parameters, constant_init
from paddle3d.utils.checkpoint import load_pretrained_model_from_path

@manager.VOXEL_ENCODERS.add_component
class PillarFeatureNet(nn.Layer):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self, in_channels=4, 
                 feat_channels=(64,), 
                 with_distance=False, 
                 with_cluster_center=True, 
                 with_voxel_center=True,
                 remove_intensity=False,
                 voxel_size=(0.2, 0.2, 4), 
                 point_cloud_range =(0, -40, -3, 70.4, 40, 1), 
                 norm_cfg=dict(type_name='BN1d', 
                                eps=0.001,
                                momentum=0.99), 
                 mode='max',
                 legacy=False, 
                 input_norm=False):
        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        self.input_norm = input_norm
        self.remove_intensity = remove_intensity
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 2
        if with_distance:
            in_channels += 1
        if remove_intensity:
            in_channels -= 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.fp16_enabled = False
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, norm_cfg=
                norm_cfg, last_layer=last_layer, mode=mode))
        self.pfn_layers = nn.LayerList(sublayers=pfn_layers)
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range


    def forward(self, features, num_points, coors, img_feats=None,
        img_metas=None):
        """Forward function.

        Args:
            features (paddle.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (paddle.Tensor): Number of points in each pillar.
            coors (paddle.Tensor): Coordinates of each voxel.
            img_feats (list[paddle.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.
        Returns:
            paddle.Tensor: Features of pillars.
        """
        features_ls = [features]
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(axis=1, keepdim=True
                ) / num_points.astype(dtype=features.dtype).reshape((-1, 1, 1)) #.view(-1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = paddle.zeros_like(x=features[:, :, :2])
                f_center[:, :, (0)] = features[:, :, (0)] - (coors[:, (3)].cast(dtype).unsqueeze(axis=1) * self.vx + self.x_offset)
                f_center[:, :, (1)] = features[:, :, (1)] - (coors[:, (2)].cast(dtype).unsqueeze(axis=1) * self.vy + self.y_offset)
            else:
                f_center = features[:, :, :2]
                f_center[:, :, (0)] = f_center[:, :, (0)] - (coors[:, (3)].
                    astype(dtype=features.dtype).unsqueeze(axis=1) * self.
                    vx + self.x_offset)
                f_center[:, :, (1)] = f_center[:, :, (1)] - (coors[:, (2)].
                    astype(dtype=features.dtype).unsqueeze(axis=1) * self.
                    vy + self.y_offset)
            features_ls.append(f_center)
        if self._with_distance:
            points_dist = paddle.linalg.norm(x=features[:, :, :3], p=2,
                axis=2, keepdim=True)
            features_ls.append(points_dist)
        features = paddle.concat(x=features_ls, axis=-1)
        if self.input_norm:
            features[:, :, (0)] = features[:, :, (0)] / self.point_cloud_range[
                3]
            features[:, :, (1)] = features[:, :, (1)] / self.point_cloud_range[
                4]
            features[:, :, (2)] = features[:, :, (2)] / self.point_cloud_range[
                5]
            features[:, :, (3)] = features[:, :, (3)] / 255.0
            features[:, :, (7)] = features[:, :, (7)] / self.point_cloud_range[
                3]
            features[:, :, (8)] = features[:, :, (8)] / self.point_cloud_range[
                4]
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = paddle.unsqueeze(input=mask, axis=-1).astype(dtype=features.
            dtype)
        features *= mask
        for pfn in self.pfn_layers:
            if self.remove_intensity:
                features = features[..., [0, 1, 2, 4, 5, 6, 7, 8]]
            features = pfn(features, num_points)
        return features.squeeze()


@manager.VOXEL_ENCODERS.add_component
class DynamicPillarFeatureNet(PillarFeatureNet):
    """Pillar Feature Net using dynamic voxelization.

    The network prepares the pillar features and performs forward pass
    through PFNLayers. The main difference is that it is used for
    dynamic voxels, which contains different number of points inside a voxel
    without limits.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
    """

    def __init__(self, 
                in_channels=4, 
                feat_channels=(64,), 
                with_distance=False, 
                with_cluster_center=True, 
                with_voxel_center=True,
                remove_intensity=False, 
                voxel_size=(0.2, 0.2, 4.0), 
                point_cloud_range =(0, -40, -3, 70.4, 40, 1),
                norm_cfg=dict(type_name='BN1d', 
                                eps=0.001,
                                momentum=0.99), 
                mode='max',
                legacy=False, 
                input_norm=False):
        super(DynamicPillarFeatureNet, self).__init__(in_channels, 
                feat_channels, with_distance, 
                with_cluster_center= with_cluster_center, with_voxel_center=with_voxel_center,
                remove_intensity=remove_intensity, voxel_size=voxel_size,
                point_cloud_range=point_cloud_range, norm_cfg=norm_cfg, mode= mode, 
                legacy=legacy, input_norm=input_norm)
        assert len(feat_channels) > 0
        self.legacy = legacy
        self.input_norm = input_norm
        self.remove_intensity = remove_intensity
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 2
        if with_distance:
            in_channels += 1
        if remove_intensity:
            in_channels -= 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.fp16_enabled = False
        feat_channels = [self.in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            norm_layer = build_norm_layer(norm_cfg, out_filters)[1]
            pfn_layers.append(nn.Sequential(
                                        nn.Linear(in_features=in_filters, out_features=out_filters, bias_attr=False), 
                                        norm_layer,
                                        nn.ReLU()
                                    )
                            )
        self.num_pfn = len(pfn_layers)
        self.pfn_layers = nn.LayerList(sublayers=pfn_layers)
        self.pfn_scatter = DynamicScatter(voxel_size, point_cloud_range, mode=mode)
        self.cluster_scatter = DynamicScatter(voxel_size, point_cloud_range, mode='mean')
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range
        self.input_norm = input_norm
        self.pfn_layers.apply(param_init.init_weight)

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors,
        img_feats=None, img_metas=None):
        """Map the centers of voxels to its corresponding points.

        Args:
            pts_coors (paddle.Tensor): The coordinates of each points, shape
                (M, 3), where M is the number of points.
            voxel_mean (paddle.Tensor): The mean or aggreagated features of a
                voxel, shape (N, C), where N is the number of voxels.
            voxel_coors (paddle.Tensor): The coordinates of each voxel.
            img_feats (list[paddle.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            paddle.Tensor: Corresponding voxel centers of each points, shape
                (M, C), where M is the numver of points.
        """
        canvas_y = int((self.point_cloud_range[4] - self.point_cloud_range[
            1]) / self.vy)
        canvas_x = int((self.point_cloud_range[3] - self.point_cloud_range[
            0]) / self.vx)
        canvas_channel = voxel_mean.shape[1]
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_y * canvas_x * batch_size
        canvas = paddle.zeros(shape=[canvas_channel, canvas_len], dtype=
            voxel_mean.dtype)
        indices = voxel_coors[:, 0] * canvas_y * canvas_x + voxel_coors[:,
            2] * canvas_x + voxel_coors[:, 3]
        canvas_t = paddle.scatter(canvas.T, indices, voxel_mean, overwrite=True)
        voxel_index = pts_coors[:, 0] * canvas_y * canvas_x + pts_coors[:,2] * canvas_x + pts_coors[:, 3]
        voxel_index = voxel_index % canvas_t.shape[0]
        voxel_index = voxel_index.cast('int64')
        center_per_point = paddle.index_select(canvas_t, voxel_index, axis=0)
        return center_per_point


    def forward(self, features, coors):
        """Forward function.

        Args:
            features (paddle.Tensor): Point features or raw points in shape
                (N, M, C).
            coors (paddle.Tensor): Coordinates of each voxel

        Returns:
            paddle.Tensor: Features of pillars.
        """
        features_ls = [features]
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(coors, voxel_mean,
                mean_coors)
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)
        if self._with_voxel_center:
            f_center = paddle.zeros(shape=(features.shape[0], 2), dtype=features.dtype)
            f_center[:, 0] = features[:, 0] - (coors[:, 3].cast(features.dtype) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (coors[:, 2].cast(features.dtype) * self.vy + self.y_offset)
            features_ls.append(f_center)
        if self._with_distance:
            points_dist = paddle.linalg.norm(x=features[:, :3], p=2, axis=1, keepdim=True)
            features_ls.append(points_dist)
        features = paddle.concat(x=features_ls, axis=-1)
        if self.input_norm:
            features[:, 0] /= self.point_cloud_range[3]
            features[:, 1] /= self.point_cloud_range[4]
            features[:, 2] /= self.point_cloud_range[5]
            features[:, 3] = features[:, 3] / 255.0
        for i, pfn in enumerate(self.pfn_layers):
            if self.remove_intensity and i == 0:
                # pass
                features = features[..., [0, 1, 2, 4, 5, 6, 7, 8]]
            point_feats = pfn(features)
            voxel_feats, voxel_coors = self.pfn_scatter(point_feats, coors)
            if i != len(self.pfn_layers) - 1:
                feat_per_point = self.map_voxel_center_to_point(coors,
                    voxel_feats, voxel_coors)
                features = paddle.concat(x=[point_feats, feat_per_point],
                    axis=1)
        return voxel_feats, voxel_coors