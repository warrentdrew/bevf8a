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
import math
import numpy as np
from os.path import join

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import TruncatedNormal

from paddle3d.apis import manager
from paddle3d.utils import checkpoint
from paddle3d.models.layers import param_init

trunc_normal_ = TruncatedNormal(std=.02)
BN_MOMENTUM = 0.9


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)


class BasicBlock(nn.Layer):
    """ as name """
    def __init__(self, inplanes, planes, stride=1, dilation=1, norm_decay=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias_attr=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay)))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias_attr=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay)))
        self.stride = stride

    def forward(self, x, residual=None):
        """ as name """
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    """ as name """
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1, norm_decay=0.0):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2D(inplanes, bottle_planes,
                               kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(bottle_planes, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay)))
        self.conv2 = nn.Conv2D(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias_attr=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2D(bottle_planes, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay)))
        self.conv3 = nn.Conv2D(bottle_planes, planes,
                               kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay)))
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x, residual=None):
        """ as name """
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Layer):
    """ as name """
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1, norm_decay=0.0):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2D(inplanes, bottle_planes,
                               kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(bottle_planes, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay)))
        self.conv2 = nn.Conv2D(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias_attr=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2D(bottle_planes, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay)))
        self.conv3 = nn.Conv2D(bottle_planes, planes,
                               kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay)))
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x, residual=None):
        """ as name """
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Layer):
    """ as name """
    def __init__(self, in_channels, out_channels, kernel_size, residual, norm_decay=0.0):
        super(Root, self).__init__()
        self.conv = nn.Conv2D(
            in_channels, out_channels, 1,
            stride=1, bias_attr=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2D(out_channels, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay)))
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, *x):
        """ as name """
        children = x
        x = self.conv(paddle.concat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Layer):
    """ as name """
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False, norm_decay=0.0):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation, norm_decay=norm_decay)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation, norm_decay=norm_decay)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual,
                              norm_decay=norm_decay)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual,
                              norm_decay=norm_decay)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual, norm_decay=norm_decay)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2D(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2D(in_channels, out_channels,
                          kernel_size=1, stride=1, bias_attr=False),
                nn.BatchNorm2D(out_channels, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay)))
            )

    def forward(self, x, residual=None, children=None):
        """ as name """
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Layer):
    """ as name """
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False,
                 opt=None, norm_decay=0.0):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2D(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias_attr=False),
            nn.BatchNorm2D(channels[0], momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay))),
            nn.ReLU())
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0], norm_decay=norm_decay)
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2, norm_decay=norm_decay)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root, norm_decay=norm_decay)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root, norm_decay=norm_decay)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root, norm_decay=norm_decay)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root, norm_decay=norm_decay)
        if opt.pre_img:
            self.pre_img_layer = nn.Sequential(
                nn.Conv2D(3, channels[0], kernel_size=7, stride=1,
                          padding=3, bias_attr=False),
                nn.BatchNorm2D(channels[0], momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay))),
                nn.ReLU())
        if opt.pre_hm:
            self.pre_hm_layer = nn.Sequential(
                nn.Conv2D(1, channels[0], kernel_size=7, stride=1,
                          padding=3, bias_attr=False),
                nn.BatchNorm2D(channels[0], momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay))),
                nn.ReLU())
        # for m in self.sublayers():
        #     if isinstance(m, nn.Conv2D):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2D):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1, norm_decay=0.0):
        """ as name """
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2D(stride, stride=stride),
                nn.Conv2D(inplanes, planes,
                          kernel_size=1, stride=1, bias_attr=False),
                nn.BatchNorm2D(planes, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay))),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1, norm_decay=0.0):
        """ as name """
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2D(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias_attr=False, dilation=dilation),
                nn.BatchNorm2D(planes, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay))),
                nn.ReLU()])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x, pre_img=None, pre_hm=None):
        """ as name """
        y = []
        x = self.base_layer(x)
        if pre_img is not None:
            x = x + self.pre_img_layer(pre_img)
        if pre_hm is not None:
            x = x + self.pre_hm_layer(pre_hm)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)

        return y

    def load_pretrained_model(self, pretrained=None):
        """ as name """
        if pretrained is not None:
            model_weights = paddle.load(pretrained)
            num_classes = len(model_weights[list(model_weights.keys())[-1]])
            self.fc = nn.Conv2D(
                self.channels[-1], num_classes,
                kernel_size=1, stride=1, padding=0, bias_attr=True)
            checkpoint.load_pretrained_model(self, pretrained)


class DLA34(nn.Layer):
    """ as name """
    def __init__(self, levels=[1, 1, 1, 2, 2, 1], channels=[16, 32, 64, 64, 128, 256], num_classes=1000,
                 block=BasicBlock, residual_root=False, pretrained_model=None, norm_decay=0.0):
        """ as name """
        super(DLA34, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.pretrained_model = pretrained_model
        self.base_layer = nn.Sequential(
            nn.Conv2D(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias_attr=False),
            nn.BatchNorm2D(channels[0], momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay))),
            nn.ReLU())
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0], norm_decay=norm_decay)
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2, norm_decay=norm_decay)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root, norm_decay=norm_decay)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root, norm_decay=norm_decay)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root, norm_decay=norm_decay)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root, norm_decay=norm_decay)
        if pretrained_model is not None:
            checkpoint.load_pretrained_model(self, pretrained_model)
        # for m in self.sublayers():
        #     if isinstance(m, nn.Conv2D):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2D):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1, norm_decay=0.0):
        """ as name """
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2D(stride, stride=stride),
                nn.Conv2D(inplanes, planes,
                          kernel_size=1, stride=1, bias_attr=False),
                nn.BatchNorm2D(planes, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay))),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1, norm_decay=0.0):
        """ as name """
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2D(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias_attr=False, dilation=dilation),
                nn.BatchNorm2D(planes, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay))),
                nn.ReLU()])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x, pre_img=None, pre_hm=None):
        """ as name """
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)

        return y


class Conv(nn.Layer):
    """ as name """
    def __init__(self, chi, cho, norm_decay=0.0):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(chi, cho, kernel_size=1, stride=1, bias_attr=False),
            nn.BatchNorm2D(cho, momentum=BN_MOMENTUM,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay))),
            nn.ReLU())

    def forward(self, x):
        """ as name """
        return self.conv(x)


class Identity(nn.Layer):
    """ as name """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """ as name """
        return x


class IDAUp(nn.Layer):
    """ as name """
    def __init__(self, node_kernel, out_dim, channels, up_factors, norm_decay=0.0):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2D(c, out_dim,
                              kernel_size=1, stride=1, bias_attr=False),
                    nn.BatchNorm2D(out_dim,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay))),
                    nn.ReLU())
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.Conv2DTranspose(
                    out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                    output_padding=0, groups=out_dim, bias_attr=False)
                fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = nn.Sequential(
                nn.Conv2D(out_dim * 2, out_dim,
                          kernel_size=node_kernel, stride=1,
                          padding=node_kernel // 2, bias_attr=False),
                nn.BatchNorm2D(out_dim,
                                weight_attr=ParamAttr(regularizer=L2Decay(norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(norm_decay))),
                nn.ReLU())
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers):
        """ as name """
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(paddle.concat([x, layers[i]], 1))
            y.append(x)
        return x, y


class DLAUp(nn.Layer):
    """ as name """
    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None, norm_decay=0.0):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j], norm_decay=norm_decay))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def init_weights(self):
        """ as name """
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                param_init._no_grad_normal_(m.weight, 0, math.sqrt(2. / n))

    def forward(self, layers):
        """ as name """
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y
        return x

@manager.BACKBONES.add_component
class DLASeg(nn.Layer):
    """ as name """
    def __init__(self,
                 channels=[16, 32, 64, 64, 128, 256],
                 pretrained_model=None,
                 output_levels=4, use_resize=False, final_dim=None, downsample=None, norm_decay=0.0):
        super(DLASeg, self).__init__()
        down_ratio = 4
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        self.output_levels = output_levels
        self.base = DLA34(channels=channels, pretrained_model=pretrained_model, norm_decay=norm_decay)

        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales, norm_decay=norm_decay)
        self.use_resize = use_resize
        if self.use_resize:
            self.target_size = (final_dim[0] // downsample, final_dim[1] // downsample)
            self.resize = nn.AdaptiveAvgPool2D(self.target_size)
   
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight)
                if m.bias is not None:
                    param_init.constant_init(m.bias, value=0.0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1D, nn.BatchNorm2D)):
                param_init.constant_init(m.weight, value=1.0)
                param_init.constant_init(m.bias, value=0.0)
            elif isinstance(m, (nn.Conv2D)):
                param_init.reset_parameters(m)

        self.apply(_init_weights)
        
        self.dla_up.init_weights()

    def forward(self, x):
        """ as name """
        x = self.base(x)
        x = self.dla_up(x[self.first_level:])
        if self.use_resize:
            x = self.resize(x)
        return (x,)


def fill_up_weights(up):
    """ as name """
    w = up.weight
    w_numpy = w.numpy()
    f = math.ceil(w.shape[2] / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.shape[2]):
        for j in range(w.shape[3]):
            w_numpy[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.shape[0]):
        w_numpy[c, 0, :, :] = w_numpy[0, 0, :, :]

    up.weight.set_value(w_numpy)
