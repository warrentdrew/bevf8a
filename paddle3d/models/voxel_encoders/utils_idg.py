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
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle3d.utils_idg.build_layer import build_norm_layer

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (paddle.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        paddle.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = paddle.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = paddle.arange(
        max_num, dtype='int32').reshape(max_num_shape) #.view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.cast('int32') > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

class PFNLayer(nn.Layer):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type_name='BN1d', eps=1e-3, momentum=0.01),
                 last_layer=False,
                 mode='max'):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.linear = nn.Linear(in_channels, self.units, bias_attr=False)
        self.norm = build_norm_layer(norm_cfg, self.units)[1]

        assert mode in ['max', 'avg']
        self.mode = mode

    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (paddle.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (paddle.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (paddle.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            paddle.Tensor: Features of Pillars.
        """
        x = self.linear(inputs)
        x = self.norm(x.transpose((0, 2, 1))).transpose((0, 2, 1))
        x = F.relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = paddle.max(x, axis =1, keepdim=True)
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(
                axis=1, keepdim=True) / num_voxels.cast(inputs.dtype).reshape((-1, 1, 1))

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.tile((1, inputs.shape[1], 1))
            x_concatenated = paddle.concat([x, x_repeat], axis = 2) 
            return x_concatenated
