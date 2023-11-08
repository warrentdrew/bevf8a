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

import random
import pickle as pkl
import os
import paddle
from paddle import nn
from paddle.nn import functional as F

from paddle3d.utils_idg.build_layer import build_norm_layer, get_activation_layer
from paddle3d.utils_idg.ops.scatter_max import scatter_max_fp32
from paddle3d.models.layers.param_init import init_weight


class MLPLayer(nn.Layer):
    """MLP layer
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        is_head=False,
        act="relu",
        bias=False,
        dropout=0.0,
    ):
        super(MLPLayer, self).__init__()
        self.is_head = is_head
        if self.is_head:
            self.linear = nn.Linear(in_channel, out_channel, bias_attr=True)
        else:
            self.linear = nn.Linear(in_channel, out_channel, bias_attr=bias)
            self.norm = build_norm_layer(norm_cfg, out_channel)[1]
            self.act = get_activation_layer(act)
            if dropout > 0:
                self.dropout = nn.Dropout(p=dropout)
            else:
                self.dropout = None

    def forward(self, inputs):
        """Forward function.
        """
        if not self.is_head:
            x = self.linear(inputs)
            x = self.norm(x)
            x = self.act(x)
            if self.dropout is not None:
                x = self.dropout(x)
        else:
            x = self.linear(inputs)
        return x

class BuildMLPLayers(nn.Layer):
    """MLP layer
    """

    def __init__(
        self,
        in_channel,
        hidden_dims,
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        is_head=False,
        act="relu",
        bias=False,
        dropout=0.0,
    ):
        super(BuildMLPLayers, self).__init__()
        layer_list = []
        last_channel = in_channel
        for i, c in enumerate(hidden_dims):
            if i == len(hidden_dims) - 1 and is_head:
                layer_list.append(
                    MLPLayer(last_channel,
                            c,
                            norm_cfg=norm_cfg,
                            is_head=True,
                            act=act,
                            bias=bias,
                            dropout=dropout)
                )
            else:
                layer_list.append(
                    MLPLayer(last_channel,
                            c,
                            norm_cfg=norm_cfg,
                            is_head=False,
                            act=act,
                            bias=bias,
                            dropout=dropout)
                )
            last_channel = c
        self.mlp = nn.LayerList(layer_list)
    def forward(self, inputs):
        """Forward function.
        """
        for i, mlp in enumerate(self.mlp):
            inputs = mlp(inputs)
        return inputs


def build_mlp(in_channel, hidden_dims, norm_cfg, is_head=False, act="relu", bias=False, dropout=0):
    layer_list = []
    last_channel = in_channel
    for i, c in enumerate(hidden_dims):
        act_layer = get_activation_layer(act)

        norm_layer = build_norm_layer(norm_cfg, c)[1]
        if i == len(hidden_dims) - 1 and is_head:
            layer_list.append(
                nn.Linear(last_channel, c, bias_attr=True),
            )
        else:
            sq = [
                nn.Linear(last_channel, c, bias_attr=bias),
                norm_layer,
                act_layer,
            ]
            if dropout > 0:
                sq.append(nn.Dropout(dropout))
            layer_list.append(nn.Sequential(*sq))

        last_channel = c
    mlp = nn.Sequential(*layer_list)
    return mlp


class DynamicVFELayer(nn.Layer):
    """Replace the Voxel Feature Encoder layer in VFE layers.
    This layer has the same utility as VFELayer above
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        act="relu",
        dropout=0.0,
    ):
        super(DynamicVFELayer, self).__init__()
        self.fp16_enabled = False
        # self.units = int(out_channels / 2)
        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.linear = nn.Linear(in_channels, out_channels, bias_attr=False)
        self.act = get_activation_layer(act)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, inputs):
        """Forward function.
        Args:
            inputs (paddle.Tensor): Voxels features of shape (M, C).
                M is the number of points, C is the number of channels of point features.
        Returns:
            paddle.Tensor: point features in shape (M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        if self.dropout is not None:
            inputs = self.dropout(inputs)
        x = self.linear(inputs)
        x = self.norm(x)
        pointwise = self.act(x)
        return pointwise


class SIRLayer(nn.Layer):
    def __init__(
        self,
        in_channels=4,
        feat_channels=[],
        with_distance=False,
        with_cluster_center=False,
        with_rel_mlp=True,
        rel_mlp_hidden_dims=[
            16,
        ],
        rel_mlp_in_channel=3,
        with_voxel_center=False,
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        mode="max",
        return_point_feats=False,
        rel_dist_scaler=10.0,
        xyz_normalizer=[20, 20, 4],
        act="relu",
        dropout=0.0,
        onnx_test=False,
        debugger=None
    ):
        super().__init__()

        self.in_channels = in_channels
        if with_distance:
            self.in_channels += 1
        self.with_distance = with_distance
        self.with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.rel_dist_scaler = rel_dist_scaler
        self.mode = mode
        self.with_rel_mlp = with_rel_mlp
        self.xyz_normalizer = xyz_normalizer
        self.fp16_enabled = False
        self.onnx_test = onnx_test
        self.debugger = debugger
        if with_rel_mlp:
            rel_mlp_hidden_dims.append(1)  # not self.in_channels
            self.rel_mlp = BuildMLPLayers(
                rel_mlp_in_channel, rel_mlp_hidden_dims, norm_cfg, act=act)


        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2

            vfe_layers.append(
                DynamicVFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg,
                    act=act,
                    dropout=dropout,
                )
            )
        self.vfe_layers = nn.LayerList(vfe_layers)
        self.num_vfe = len(vfe_layers)

    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):
        return voxel_mean[voxel2point_inds]

    def forward(
        self,
        features,
        coors,
        f_cluster,
        batch_rois,
        batch_roi_nonempty_masks=None,
    ):
        if self.with_rel_mlp:
            features = features * self.rel_mlp(f_cluster)
        voxel_feats_list = []
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)
            if self.onnx_test:
                voxel_feats = point_feats[:batch_rois.shape[0]]
            # elif self.fp16_enabled:
            #     voxel_feats = scatter_max_fp16(point_feats, coors, batch_rois)
            else:
                voxel_feats = scatter_max_fp32(point_feats, coors, batch_rois)
            if batch_roi_nonempty_masks is not None:
                voxel_feats = voxel_feats * batch_roi_nonempty_masks
            voxel_feats_list.append(voxel_feats)

            feat_per_point = self.map_voxel_center_to_point(voxel_feats, coors)
            features = paddle.concat([point_feats, feat_per_point], axis=1)

        voxel_feats = paddle.concat(voxel_feats_list, axis=1)

        return point_feats, voxel_feats


class DynamicPointNetV2(nn.Layer):
    def __init__(
        self,
        num_blocks,
        in_channels,
        feat_channels,
        rel_mlp_hidden_dims,
        rel_mlp_in_channels,
        with_rel_mlp=True,
        with_cluster_center=False,
        with_distance=False,
        rel_dist_scaler=10,
        mode="max",
        xyz_normalizer=[20, 20, 4],
        act="gelu",
        geo_input=True,
        norm_cfg=dict(type_name="LN", epsilon=1e-3),
        dropout=0,
        unique_once=True,
        onnx_test=False,
        debugger=None
    ):
        super().__init__()
        self.geo_input = geo_input
        self.num_blocks = num_blocks
        self.unique_once = unique_once
        end_channel = 0
        for c in feat_channels:
            end_channel += sum(c)
        self.end_channel = end_channel
        self.debugger = debugger

        block_list = []
        for i in range(num_blocks):
            encoder = SIRLayer(
                in_channels=in_channels[i],
                feat_channels=feat_channels[i],
                with_distance=with_distance,
                with_cluster_center=with_cluster_center,
                with_rel_mlp=with_rel_mlp,
                rel_mlp_hidden_dims=rel_mlp_hidden_dims[i],
                rel_mlp_in_channel=rel_mlp_in_channels[i],
                with_voxel_center=False,
                norm_cfg=norm_cfg,
                mode=mode,
                return_point_feats=(i != num_blocks - 1),
                rel_dist_scaler=rel_dist_scaler,
                xyz_normalizer=xyz_normalizer,
                act=act,
                dropout=dropout,
                onnx_test=onnx_test,
                debugger=debugger
            )
            block_list.append(encoder)
        self.block_list = nn.LayerList(block_list)
        self.fp16_enabled = False
        self.xyz_normalizer = paddle.to_tensor(
            xyz_normalizer).reshape([1, 3])
        self.rel_dist_scaler = rel_dist_scaler
        self.init_weights()

    def init_weights(self):
        for m in self.sublayers():
            init_weight(m)

    def forward(self, pts_xyz, pts_features, pts_info, roi_inds, rois, batch_roi_nonempty_masks=None):
        """Forward pass.

        Args:
            pts_info (paddle.Tensor): Point-wise semantic features. (num_rois, num_pts, C)
            roi_inds (paddle.Tensor): Point-wise part prediction features. (num_rois, num_pts)

        Returns:
            tuple[paddle.Tensor]: Score of class and bbox predictions.
        """

        if pts_features.shape[0] == 0:
            final_cluster_feats = paddle.zeros((0, self.end_channel), dtype=pts_features.dtype)
            out_coors = paddle.full((0,), -1, dtype='int64')
            return final_cluster_feats, out_coors
        roi_centers = rois[:, :3].clone().detach()
        roi_centers[:, 2] = roi_centers[:, 2] + rois[:, 5] * 0.5
        rel_xyz = pts_xyz - roi_centers[roi_inds]
        pts_xyz = pts_xyz / self.xyz_normalizer
        f_cluster = paddle.concat([pts_info, rel_xyz], axis=-1)
        f_cluster = f_cluster / self.rel_dist_scaler
        
        out_feats = pts_features
        # f_cluster = paddle.concat([pts_info, rel_xyz], axis=-1)
        cluster_feat_list = []
        for i, block in enumerate(self.block_list):
            in_feats = paddle.concat([pts_xyz, out_feats], 1)

            if self.geo_input:
                in_feats = paddle.concat([in_feats, f_cluster], 1)
            # return point features
            out_feats, out_cluster_feats = block(
                in_feats, roi_inds, f_cluster, rois, batch_roi_nonempty_masks
            )
            cluster_feat_list.append(out_cluster_feats)
        final_cluster_feats = paddle.concat(cluster_feat_list, axis=1)
        return final_cluster_feats, out_feats
