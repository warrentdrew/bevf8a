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
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from paddle3d.models.layers.param_init import (constant_init, reset_parameters,
                                               xavier_uniform_init)


def build_conv_layer(cfg, *args, **kwargs):
    """ as name """
    if cfg is None:
        cfg_ = dict(type='Conv2D')
    else:
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    conv_layer = getattr(nn, layer_type)
    layer = conv_layer(*args, **kwargs)
    return layer


def build_norm_layer(cfg, num_features, norm_decay=0.0):
    """ as name """
    if cfg is None:
        cfg_ = dict(type='BatchNorm2D')
    else:
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    norm_layer = getattr(nn, layer_type)
    layer = norm_layer(num_features,
                        weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                        bias_attr=ParamAttr(regularizer=L2Decay(norm_decay)), **cfg_)
    return layer


def build_activation_layer(cfg):
    """ as name """
    if cfg is None:
        cfg_ = dict(type='ReLU')
    else:
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    act_layer = getattr(nn, layer_type)
    return act_layer()


class ConvModule(nn.Layer):
    """ as name """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_decay=0.0,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.bn = build_norm_layer(norm_cfg, norm_channels, norm_decay=norm_decay)

        # build activation layer
        if self.with_activation:
            self.activate = build_activation_layer(act_cfg)

        self.init_weights()

    def forward(self, x):
        """ as name """
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self.with_norm:
                x = self.bn(x)
            elif layer == 'act' and self.with_activation:
                x = self.activate(x)
        return x

    def init_weights(self):
        """ as name """
        def _init_weights(m):
            if isinstance(m, (nn.Conv1D, nn.Conv2D)):
                reset_parameters(m)
            elif isinstance(m, (nn.BatchNorm1D, nn.BatchNorm2D)):
                constant_init(m.weight, value=1)
                constant_init(m.bias, value=0)

        self.apply(_init_weights)
