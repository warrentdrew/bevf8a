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
import logging
import numpy as np
import paddle
import paddle.nn as nn
from paddle3d.apis import manager
from paddle3d.utils_idg.build_layer import build_norm_layer
from paddle3d.models.layers import param_init, reset_parameters, constant_init, kaiming_normal_init

BatchNorm2d = paddle.nn.BatchNorm2D
BN_MOMENTUM = 0.99

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return paddle.nn.Conv2D(in_channels=in_planes, out_channels=out_planes,
        kernel_size=3, stride=stride, padding=1, bias_attr=False)


class Upsample(paddle.nn.Layer):
    """ as name """

    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input_tensor):
        """ as name """
        h = int(self.scale_factor * input_tensor.shape[2])
        w = int(self.scale_factor * input_tensor.shape[3])
        input_tensor = paddle.nn.functional.interpolate(x=input_tensor,
            size=(h, w), mode=self.mode, align_corners=False)
        return input_tensor


class BasicBlock(paddle.nn.Layer):
    """ as name """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        norm_cfg=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        if norm_cfg is None:
            self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        else:
            self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = paddle.nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        if norm_cfg is None:
            self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        else:
            self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """ as name """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Layer):
    """ as name """
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        norm_cfg=None):
        super(Bottleneck, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=inplanes, out_channels=
            planes, kernel_size=1, bias_attr=False)
        if norm_cfg is None:
            self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        else:
            self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.conv2 = paddle.nn.Conv2D(in_channels=planes, out_channels=
            planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        if norm_cfg is None:
            self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        else:
            self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.conv3 = paddle.nn.Conv2D(in_channels=planes, out_channels=
            planes * self.expansion, kernel_size=1, bias_attr=False)
        if norm_cfg is None:
            self.bn3 = BatchNorm2d(planes * self.expansion, momentum=
                BN_MOMENTUM)
        else:
            self.bn3 = build_norm_layer(norm_cfg, planes * self.expansion)[1]
        self.relu = paddle.nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """ as name """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class HighResolutionModule(paddle.nn.Layer):
    """ as name """

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
        num_channels, fuse_method, multi_scale_output=True, norm_cfg=None):
        super(HighResolutionModule, self).__init__()
        self.norm_cfg = norm_cfg
        self._check_branches(num_branches, blocks, num_blocks,
            num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks,
            num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = paddle.nn.ReLU()

    def _check_branches(self, num_branches, blocks, num_blocks,
        num_inchannels, num_channels):
        """ as name """
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks,
        num_channels, stride=1):
        """ as name """
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[
            branch_index] * block.expansion:
            if self.norm_cfg is None:
                downsample = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=self.num_inchannels[branch_index],
                                        out_channels=num_channels[branch_index] * block.expansion, 
                                        kernel_size=1, stride=stride, bias_attr=False), 
                                        BatchNorm2d(num_channels[branch_index] * block.expansion, 
                                        momentum=BN_MOMENTUM))
            else:
                downsample = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=self.num_inchannels[branch_index],
                                        out_channels=num_channels[branch_index] * block.expansion, 
                                        kernel_size=1, stride=stride, bias_attr=False), 
                                        build_norm_layer(self.norm_cfg, 
                                            num_channels[branch_index] * block.expansion)[1])
        layers = []
        layers.append(block(self.num_inchannels[branch_index], 
                            num_channels[branch_index], stride, 
                            downsample, norm_cfg=self.norm_cfg))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                num_channels[branch_index], norm_cfg=self.norm_cfg))
        return paddle.nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """ as name """
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks,
                num_channels))
        return paddle.nn.LayerList(sublayers=branches)

    def _make_fuse_layers(self):
        """ as name """
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    if self.norm_cfg is None:
                        fuse_layer.append(paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=num_inchannels[j],
                                        out_channels=num_inchannels[i], kernel_size=1,
                                        stride=1, padding=0, bias_attr=False),
                                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM), 
                                        Upsample(scale_factor=2 ** (j - i), mode='bilinear')))
                    else:
                        fuse_layer.append(paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=num_inchannels[j],
                            out_channels=num_inchannels[i], kernel_size=1,
                            stride=1, padding=0, bias_attr=False),
                            build_norm_layer(self.norm_cfg, num_inchannels[i])[1], 
                            Upsample(scale_factor=2 ** (j - i),
                            mode='bilinear')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            if self.norm_cfg is None:
                                conv3x3s.append(paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=num_inchannels[j], 
                                    out_channels=num_outchannels_conv3x3, kernel_size=3, stride=2, 
                                    padding=1, bias_attr=False), 
                                    BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                            else:
                                conv3x3s.append(paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=num_inchannels[j], 
                                    out_channels=num_outchannels_conv3x3, kernel_size=3, stride=2, 
                                    padding=1, bias_attr=False), 
                                    build_norm_layer(self.norm_cfg, num_outchannels_conv3x3)[1]))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            if self.norm_cfg is None:
                                conv3x3s.append(paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=num_inchannels[j], 
                                    out_channels=num_outchannels_conv3x3, kernel_size=3, stride=2, padding=1,
                                    bias_attr=False), 
                                    BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM), 
                                    paddle.nn.ReLU()))
                            else:
                                conv3x3s.append(paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=num_inchannels[j], 
                                    out_channels=num_outchannels_conv3x3, kernel_size=3, stride=2, padding=1,
                                    bias_attr=False), 
                                    build_norm_layer(self.norm_cfg, num_outchannels_conv3x3)[1],
                                    paddle.nn.ReLU()))
                    fuse_layer.append(paddle.nn.Sequential(*conv3x3s))
            fuse_layers.append(paddle.nn.LayerList(sublayers=fuse_layer))
        return paddle.nn.LayerList(sublayers=fuse_layers)

    def get_num_inchannels(self):
        """ as name """
        return self.num_inchannels

    def forward(self, x):
        """ as name """
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            if i == 0:
                y = x[0]
            else:
                y = self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


@manager.BACKBONES.add_component
class HighResolutionNet(paddle.nn.Layer):
    """ as name """

    def __init__(self, extra, ds_layer_strides=[2, 2, 2, 2],
        us_layer_strides=[1, 2, 4, 8], num_input_features=64,
        num_inner_features=32, num_output_features=256, norm_eval=True,
        zero_init_residual=False, frozen_stages=-1, pretrained=None,
        norm_cfg=None, **kwargs):
        super(HighResolutionNet, self).__init__()
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        self.zero_init_residual = zero_init_residual
        self._num_input_features = num_input_features
        self.extra = extra
        self._ds_layer_strides = ds_layer_strides
        self._us_layer_strides = us_layer_strides
        self.conv1 = paddle.nn.Conv2D(in_channels=self._num_input_features,
            out_channels=num_inner_features, kernel_size=3, stride=1,
            padding=1, bias_attr=False)
        if norm_cfg is None:
            self.bn1 = BatchNorm2d(num_inner_features, momentum=BN_MOMENTUM)
        else:
            self.bn1 = build_norm_layer(norm_cfg, num_inner_features)[1]
        self.conv2 = paddle.nn.Conv2D(in_channels=num_inner_features,
            out_channels=num_inner_features, kernel_size=3, stride=2,
            padding=1, bias_attr=False)
        if norm_cfg is None:
            self.bn2 = BatchNorm2d(num_inner_features, momentum=BN_MOMENTUM)
        else:
            self.bn2 = build_norm_layer(norm_cfg, num_inner_features)[1]
        self.relu = paddle.nn.ReLU()
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]
        block = blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, num_inner_features,
            num_channels, num_blocks)
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']
        block = blocks_dict[block_type]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channels], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg,
            num_channels)
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']
        block = blocks_dict[block_type]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
            num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg,
            num_channels)
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']
        block = blocks_dict[block_type]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
            num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg,
            num_channels)
        last_inp_channels = np.int(np.sum(pre_stage_channels))
        if norm_cfg is None:
            self.last_layer = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=last_inp_channels, 
                out_channels=num_output_features, kernel_size=1, stride=1, padding=0, bias_attr=False), 
                BatchNorm2d(num_output_features, momentum=BN_MOMENTUM), 
                paddle.nn.ReLU())
        else:
            self.last_layer = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=last_inp_channels, 
                out_channels=num_output_features, kernel_size=1, stride=1, padding=0,
                bias_attr=False), 
                build_norm_layer(norm_cfg, num_output_features)[1], 
                paddle.nn.ReLU())
        self.init_weights(pretrained)

    def _make_transition_layer(self, num_channels_pre_layer,
        num_channels_cur_layer):
        """ as name """
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    if self.norm_cfg is None:
                        transition_layers.append(paddle.nn.Sequential(
                            paddle.nn.Conv2D(in_channels=num_channels_pre_layer[i], 
                            out_channels=num_channels_cur_layer[i], 
                            kernel_size=3, stride=1, padding=1, bias_attr=False),
                            BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM), 
                            paddle.nn.ReLU()))
                    else:
                        transition_layers.append(paddle.nn.Sequential(
                            paddle.nn.Conv2D(in_channels=num_channels_pre_layer[i], 
                            out_channels=num_channels_cur_layer[i], kernel_size=3,
                            stride=1, padding=1, bias_attr=False),
                            build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])[1], 
                            paddle.nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    if self.norm_cfg is None:
                        conv3x3s.append(paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=inchannels, 
                            out_channels=outchannels, kernel_size=3, stride=2, padding=1,
                            bias_attr=False), 
                            BatchNorm2d(outchannels, momentum=BN_MOMENTUM), 
                            paddle.nn.ReLU()))
                    else:
                        conv3x3s.append(paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=inchannels, 
                            out_channels=outchannels, kernel_size=3, stride=2, padding=1,
                            bias_attr=False), 
                            build_norm_layer(self.norm_cfg, outchannels)[1], 
                            paddle.nn.ReLU()))
                transition_layers.append(paddle.nn.Sequential(*conv3x3s))
        return paddle.nn.LayerList(sublayers=transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """ as name """
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            if self.norm_cfg is None:
                downsample = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=inplanes, 
                    out_channels=planes * block.expansion, kernel_size=1, stride=stride, bias_attr=False), 
                    BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
            else:
                downsample = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=inplanes, 
                    out_channels=planes * block.expansion, kernel_size=1, stride=stride, bias_attr=False), 
                    build_norm_layer(self.norm_cfg, planes * block.expansion)[1])
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_cfg=self.norm_cfg))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_cfg=self.norm_cfg))
        return paddle.nn.Sequential(*layers)

    def _frozen_stages(self):
        """ as name """
        if self.frozen_stages >= 0:
            for m in [self.conv1, self.bn1, self.conv2, self.bn2]:
                for param in m.parameters():
                    param.requires_grad = False
        if self.frozen_stages == 1:
            for param in self.layer1.parameters():
                param.requires_grad = False

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True
        ):
        """ as name """
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = blocks_dict[layer_config['block']]
        fuse_method = layer_config['fuse_method']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block,
                num_blocks, num_inchannels, num_channels, fuse_method,
                reset_multi_scale_output, norm_cfg=self.norm_cfg))
            num_inchannels = modules[-1].get_num_inchannels()
        return paddle.nn.Sequential(*modules), num_inchannels

    def init_weights(self, pretrained=None):
        """ as name """
        if isinstance(pretrained, str):
            state_dict = paddle.load(pretrained)
            self.set_dict(state_dict)
        elif pretrained is None:
            for m in self.sublayers():
                if isinstance(m, paddle.nn.Conv2D):
                    kaiming_normal_init(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        constant_init(m.bias, value=0.0)
                elif isinstance(m, (paddle.nn.BatchNorm2D, paddle.nn.GroupNorm)):
                    constant_init(m.weight, value = 1)
                    constant_init(m.bias, value = 0)

            if self.zero_init_residual: # not applied
                for m in self.sublayers():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3.weight, value = 0)
                        constant_init(m.norm3.bias, value = 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2.weight, value = 0)
                        constant_init(m.norm2.bias, value = 0)
        else:
            raise TypeError('pretrained must be a str or None')


    @property
    def downsample_factor(self):
        """ as name """
        factor = np.prod(self._ds_layer_strides)
        if len(self.us_layer_strides) > 0:
            factor /= self.us_layer_strides[-1]
        return factor

    def forward(self, x):
        """ as name """
        feats = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        feats.append(x)
        x = self.layer1(x)
        feats.append(x)
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        y0_h, y0_w = y_list[0].shape[2], y_list[0].shape[3]
        y1 = paddle.nn.functional.interpolate(x=y_list[1], size=(y0_h, y0_w
            ), mode='bilinear', align_corners=False)
        y2 = paddle.nn.functional.interpolate(x=y_list[2], size=(y0_h, y0_w
            ), mode='bilinear', align_corners=False)
        y3 = paddle.nn.functional.interpolate(x=y_list[3], size=(y0_h, y0_w
            ), mode='bilinear', align_corners=False)
        x = paddle.concat(x=[y_list[0], y1, y2, y3], axis=1)
        if self.last_layer is not None:
            x = self.last_layer(x)
        feats.append(x)
        return [x]

    def train(self):
        """ as name """
        super(HighResolutionNet, self).train()
        if self.norm_eval:
            for m in self.sublayers():
                if isinstance(m, paddle.nn.BatchNorm2D):
                    m.eval()
