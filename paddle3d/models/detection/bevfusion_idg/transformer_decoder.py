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

# ------------------------------------------------------------------------
# Modified from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/transformer.py
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from paddle3d.apis import manager
from paddle3d.models.layers.param_init import (
    constant_init, normal_init, reset_parameters, xavier_uniform_init)
from paddle3d.models.transformers.utils import rotate


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x1 / x2)


@manager.TRANSFORMERS.add_component
class PerceptionTransformerDecoder(nn.Layer):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=False,
                 use_shift=False,
                 use_can_bus=False,
                 can_bus_norm=False,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 norm_decay=0.0,
                 **kwargs):
        super(PerceptionTransformerDecoder, self).__init__()
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.norm_decay = norm_decay

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center
        self.init_weights()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, 3)
        if self.use_can_bus: 
            self.can_bus_mlp = nn.Sequential(
                nn.Linear(18, self.embed_dims // 2),
                nn.ReLU(),
                nn.Linear(self.embed_dims // 2, self.embed_dims),
                nn.ReLU(),
            )
            if self.can_bus_norm:
                self.can_bus_mlp.add_sublayer('norm', nn.LayerNorm(self.embed_dims,
                                weight_attr=ParamAttr(regularizer=L2Decay(self.norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(self.norm_decay))))

    @paddle.no_grad()
    def init_weights(self):
        """Initialize the transformer weights."""
        xavier_uniform_init(self.reference_points.weight, reverse=True)
        constant_init(self.reference_points.bias, value=0)
        if self.use_can_bus:
            for layer in self.can_bus_mlp:
                if isinstance(layer, nn.Linear):
                    # reset_parameters(layer)
                    xavier_uniform_init(layer.weight, reverse=True)
                    constant_init(layer.bias, value=0)
                elif isinstance(layer, nn.LayerNorm):
                    constant_init(layer.weight, value=1)
                    constant_init(layer.bias, value=0)

    def forward(self,
                bev_embed,
                object_query_embed=None,
                bev_h=None,
                bev_w=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        bs = bev_embed.shape[0]
        query_pos, query = paddle.split(
            object_query_embed, [self.embed_dims, self.embed_dims], axis=1)
        query_pos = query_pos.unsqueeze(0).expand([bs, -1, -1])
        query = query.unsqueeze(0).expand([bs, -1, -1])
        reference_points = self.reference_points(query_pos)
        reference_points = F.sigmoid(reference_points)
        init_reference_out = reference_points

        query = query.transpose([1, 0, 2])
        query_pos = query_pos.transpose([1, 0, 2])
        bev_embed = bev_embed.transpose([1, 0, 2])

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=paddle.concat([
                paddle.full([1, 1], bev_h, dtype=paddle.int64),
                paddle.full([1, 1], bev_w, dtype=paddle.int64)
            ],
                                         axis=-1),
            level_start_index=paddle.full([1], 0, dtype=paddle.int64),
            **kwargs)

        inter_references_out = inter_references

        return {
            'bev_embed': bev_embed,
            'inter_states': inter_states,
            'init_reference_out': init_reference_out,
            'inter_references_out': inter_references_out
        }
