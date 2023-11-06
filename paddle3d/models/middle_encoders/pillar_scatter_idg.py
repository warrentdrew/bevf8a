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
from paddle3d.utils_idg.build_layer import build_norm_layer
from paddle3d.models.layers import param_init, reset_parameters, constant_init

@manager.TRANSFORMER_ENCODERS.add_component
class PointPillarsScatterExpandVisV2(nn.Layer):
    """ as name """

    def __init__(self, num_input_features=32, input_expand_filters=6,
        expand_filters=32, remove_intensity=False, norm_cfg=None, name=
        'PointPillarsScatterExpandVis', **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. 
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """
        super().__init__()
        self.name = 'PointPillarsScatterExpandVisV2'
        self.nchannels = num_input_features
        self.input_expand_filters = input_expand_filters
        self.expand_filters = expand_filters
        self.remove_intensity = remove_intensity
        input_feature_dim = input_expand_filters + num_input_features
        if self.remove_intensity:
            input_feature_dim -= 2
        if norm_cfg is None:
            self.expand_layers = paddle.nn.Sequential(paddle.nn.Conv2D(
                in_channels=input_feature_dim, out_channels=expand_filters,
                kernel_size=1, stride=1, padding=0), paddle.nn.BatchNorm2D(
                num_features=expand_filters, momentum=1 - 0.01, epsilon=
                1e-05, weight_attr=None, bias_attr=None, use_global_stats=
                True), paddle.nn.ReLU())
        else:
            self.expand_layers = paddle.nn.Sequential(nn.Conv2D(in_channels=input_feature_dim, 
                                    out_channels=expand_filters, kernel_size=1, stride=1, padding=0), 
                                                    build_norm_layer(norm_cfg, expand_filters)[1], 
                                                    nn.ReLU()
                                                    )
        
        self.expand_layers.apply(param_init.init_weight)


    def forward(self, features, coords, batch_size, input_shape,
        bev_map_features):
        """ as name """
        self.nx = int(input_shape[0])
        self.ny = int(input_shape[1])
        batch_canvas_list = []
        if getattr(self, 'export_model', False):
            canvas = paddle.zeros(shape=[self.nchannels, self.nx * self.ny],
                dtype=features.dtype)
            indices = coords[:, 2] * self.nx + coords[:, 3]
            indices = indices.cast('int64')
            voxels = features.T
            canvas = canvas.put_along_axis(indices = indices.tile((canvas.shape[0], 1)), values = voxels, axis=1)
            batch_canvas = canvas.reshape((1, self.nchannels, self.ny, self.nx))
        else:
            for batch_itt in range(batch_size):
                canvas = paddle.zeros(shape=[self.nchannels, self.nx * self.ny],
                    dtype=features.dtype)
                batch_mask = paddle.where(coords[:, 0] == batch_itt)[0].squeeze(-1)
                this_coords = coords.index_select(batch_mask, axis=0)
                indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
                indices = indices.cast('int64')
                voxels = features.index_select(batch_mask, axis=0)
                voxels = voxels.T
                canvas = canvas.put_along_axis(indices = indices.tile((canvas.shape[0], 1)), values = voxels, axis=1)
                batch_canvas_list.append(canvas)
        
            batch_canvas = paddle.stack(x=batch_canvas_list, axis=0)
            batch_canvas = batch_canvas.reshape((batch_size, self.nchannels, self.ny, self.nx))

        assert self.input_expand_filters == 0 or bev_map_features.shape[1] >= self.input_expand_filters
        if self.input_expand_filters > 0:
            bev_map_features = bev_map_features[:, :self.
                input_expand_filters, ...]
            if self.remove_intensity:
                bev_map_features = bev_map_features[:, [0, 1, 2, 5, 6, 7, 
                    8, 9]]
            batch_canvas = paddle.concat(x=[batch_canvas, bev_map_features],
                axis=1)
        batch_canvas = self.expand_layers(batch_canvas)
        return batch_canvas
