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
import copy
import numpy as np

import paddle
from paddle import nn
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Normal

norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2D),
    "BN1d": ("bn1d", nn.BatchNorm1D),
    "GN": ("gn", nn.GroupNorm),
    "LN": ("ln", nn.LayerNorm),

}

def build_norm_layer(cfg, num_features, postfix=""):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and "type_name" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type_name")
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("epsilon", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.stop_gradient = not requires_grad

    return name, layer

def build_conv_layer(cfg, *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type_name='Conv2D')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type_name' not in cfg:
            raise KeyError('the cfg dict must contain the key "type_name"')
        cfg_ = copy.deepcopy(cfg)

    layer_type = cfg_.pop('type_name')
    out_channel = args[1]
    kernel_size = args[2]
    fan_out = out_channel * kernel_size**2
    std = math.sqrt(2) / math.sqrt(fan_out)
    param_attr = ParamAttr(initializer=Normal(0, std))
    bias_attr = kwargs.get('bias_attr', True)
    if bias_attr:
        bias_attr = ParamAttr(initializer=Constant(0.))
        kwargs['bias_attr'] = bias_attr

    conv_layer = getattr(nn, layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer

def get_activation_layer(act):
    """Return an activation function given a string"""
    act = act.lower()
    if act == "relu":
        act_layer = nn.ReLU()
    elif act == "gelu":
        act_layer = Gelu()
    elif act == "leakyrelu":
        act_layer = nn.LeakyReLU()
    else:
        raise NotImplementedError
    return act_layer

class Gelu(nn.Layer):
    def __init__(self):
        super(Gelu, self).__init__()

    def forward(self, x):
        cdf = 0.5 * (1.0 + paddle.tanh((np.sqrt(2 / np.pi)) * (x + 0.044715 * paddle.pow(x, 3))))
        return x * cdf
