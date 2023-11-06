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
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import copy
import numpy as np
import cv2 as cv
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Normal

from paddle3d.apis import manager

_embed_dims = 256 # for Dual-Swin
#_embed_dims=64  # for DLA-34
#_pc_range = [-50, -50, -5, 50, 50, 3]
#_bev_h=200
#_bev_w=200


_pc_range = [-60, -60, -5, 60, 60, 3]
_bev_h=288
_bev_w=288

#_pc_range = [-120, -120, -5, 120, 120, 3]
#_bev_h=576
#_bev_w=576

@manager.TRANSFORMERS.add_component
class UniformBevGridTransformer(nn.Layer):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                lss=False,
                use_conv_bevencoder=False,
                transformer=None,
                point_cloud_range=_pc_range,
                bev_h=_bev_h,
                bev_w=_bev_w,
                positional_encoding=None,
                norm_decay=0.0,
                **kwargs):
        super(UniformBevGridTransformer, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.pc_range = point_cloud_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.norm_decay = norm_decay
        print("UniformBevGridTransformer: self.bev_h = {}, self.bev_w = {}, self.pc_range = {} ".format(self.bev_h, self.bev_w, self.pc_range))

        self.positional_encoding = positional_encoding
        self.transformer = transformer

        # add
        self.lss = lss
        self.use_conv_bevencoder = use_conv_bevencoder
        cz =256
        midC=256 # 512 used in BEVFormer
        inputC=256
        if self.use_conv_bevencoder:
            self.bevencode = nn.Sequential(
                nn.Conv2D(cz, cz, kernel_size=3, padding=1, bias_attr=False),
                nn.BatchNorm2D(cz,
                                weight_attr=ParamAttr(regularizer=L2Decay(self.norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(self.norm_decay))),
                nn.ReLU(),
                nn.Conv2D(cz, midC, kernel_size=3, padding=1, bias_attr=False),
                nn.BatchNorm2D(midC,
                                weight_attr=ParamAttr(regularizer=L2Decay(self.norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(self.norm_decay))),
                nn.ReLU(),
                nn.Conv2D(midC, midC, kernel_size=3, padding=1, bias_attr=False),
                nn.BatchNorm2D(midC,
                                weight_attr=ParamAttr(regularizer=L2Decay(self.norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(self.norm_decay))),
                nn.ReLU(),
                nn.Conv2D(midC, inputC, kernel_size=3, padding=1, bias_attr=False),
                nn.BatchNorm2D(inputC,
                                weight_attr=ParamAttr(regularizer=L2Decay(self.norm_decay)),
                                bias_attr=ParamAttr(regularizer=L2Decay(self.norm_decay))),
                nn.ReLU()
            )
        
        self.embed_dims = self.transformer.embed_dims
        num_feats = positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self._init_layers()
        
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims, weight_attr=ParamAttr(initializer=Normal()))
           
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
       
    def forward(self, mlvl_feats, img_metas, gt_bboxes_3d=None, gt_labels_3d=None, prev_bev=None,  only_bev=True, lidar_aug_matrix=None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            bev_feat (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape  
        # ([4, 6, 256, 116, 180] for AT128
        # mlvl_feats.len: 1, mlvl_feats[0].size: [Bs, 6, 256, 112, 200] when the output number of FPN is 1
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.cast(dtype)

        bev_mask = paddle.zeros((bs, self.bev_h, self.bev_w)).cast(dtype)
        bev_pos = self.positional_encoding(bev_mask).cast(dtype)
        bev_embed = self.transformer(
                mlvl_feats=mlvl_feats,
                bev_queries=bev_queries,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
                lidar_aug_matrix=lidar_aug_matrix
            )
        bs, _, _ = bev_embed.shape # (bs, bev_h*bev_w, embed_dims)
        bev_embed = bev_embed.transpose([0,2,1]).reshape([bs, -1, self.bev_h, self.bev_w])

        # (bs, embed_dims, bev_h, bev_w)
        if self.use_conv_bevencoder:
            bev_embed = self.bevencode(bev_embed)
        return bev_embed

