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

import math

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Normal, Uniform


def build_conv_layer(in_channels,
                     out_channels,
                     kernel_size,
                     stride=1,
                     padding=0,
                     dilation=1,
                     groups=1,
                     bias=True,
                     distribution="uniform"):
    """Build convolution layer."""
    if distribution == "uniform":
        bound = 1 / math.sqrt(in_channels * kernel_size ** 2)
        param_attr = ParamAttr(initializer=Uniform(-bound, bound))
        bias_attr = False
        if bias:
            bias_attr = ParamAttr(initializer=Uniform(-bound, bound))
    else:
        fan_out = out_channels * kernel_size ** 2
        std = math.sqrt(2) / math.sqrt(fan_out)
        param_attr = ParamAttr(initializer=Normal(0, std))
        bias_attr = False
        if bias:
            bias_attr = ParamAttr(initializer=Constant(0.))
    conv_layer = nn.Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        weight_attr=param_attr,
        bias_attr=bias_attr)
    return conv_layer


def build_linear_layer(in_channels, out_channels, bias=True):
    """Build linear layer."""
    bound = 1 / math.sqrt(in_channels)
    param_attr = ParamAttr(initializer=Uniform(-bound, bound))
    bias_attr = False
    if bias:
        bias_attr = ParamAttr(initializer=Uniform(-bound, bound))
    return nn.Linear(
        in_channels, out_channels, weight_attr=param_attr, bias_attr=bias_attr)


def build_norm_layer(cfg, num_features):
    """Build normalization layer."""
    norm_layer = getattr(nn, cfg['type'])(
        num_features,
        momentum=1 - cfg['momentum'],
        epsilon=cfg['eps'],
        weight_attr=ParamAttr(initializer=Constant(value=1)),
        bias_attr=ParamAttr(initializer=Constant(value=0)))

    return norm_layer


def build_upsample_layer(in_channels,
                         out_channels,
                         kernel_size,
                         stride=1,
                         padding=0,
                         bias=True,
                         distribution="uniform"):
    """Build upsample layer."""
    if distribution == "uniform":
        bound = 1 / math.sqrt(in_channels)
        param_attr = ParamAttr(initializer=Uniform(-bound, bound))
        bias_attr = False
        if bias:
            bias_attr = ParamAttr(initializer=Uniform(-bound, bound))
    else:
        fan_out = out_channels * kernel_size ** 2
        std = math.sqrt(2) / math.sqrt(fan_out)
        param_attr = ParamAttr(initializer=Normal(0, std))
        bias_attr = False
        if bias:
            bias_attr = ParamAttr(initializer=Constant(0.))
    deconv_layer = nn.Conv2DTranspose(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=param_attr,
        bias_attr=bias_attr)
    return deconv_layer
