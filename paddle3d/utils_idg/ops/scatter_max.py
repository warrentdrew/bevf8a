# !/usr/bin/env python3
# Copyright 2023 Baidu Inc. All Rights Reserved.
# @author: Guojun Wang (wangguojun01@baidu.com)
# @file: scatter_max_op.py
# @brief: scatter_max_op

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer
from paddle3d.ops import scatter_max_op
import numpy as np


# class ScatterMaxFunction_fp16(PyLayer):

#     @staticmethod
#     # @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx, feat, unq_inv, batch_rois):
#         """RoIAwarePool3d function forward.

#         Args:
#             rois (torch.Tensor): [N, 7], in LiDAR coordinate,
#                 (x, y, z) is the bottom center of rois
#             pts (torch.Tensor): [npoints, 3]
#             pts_feature (torch.Tensor): [npoints, C]
#             out_size (int or tuple): n or [n1, n2, n3]
#             max_pts_per_voxel (int): m
#             mode (int): 0 (max pool) or 1 (average pool)

#         Returns:
#             pooled_features (torch.Tensor): [N, out_x, out_y, out_z, C]
#         """
#         feat = feat.contiguous()
#         unq_inv = unq_inv.contiguous()
#         new_feat = feat.new_zeros(
#             (batch_rois.shape[0], feat.shape[1]), requires_grad=True)  # 是否会计算梯度
#         new_index = unq_inv.new_full(
#             (batch_rois.shape[0], feat.shape[1]), feat.shape[0])
#         scatter_max_cuda.scatter_max_gpu(feat, unq_inv, new_feat, new_index)
#         ## float32 for debug
#         # new_feat_fp32 = feat.new_zeros(
#         #     (batch_rois.shape[0], feat.shape[1]), requires_grad=True, dtype=torch.float)  # 是否会计算梯度
#         # new_index_fp32 = unq_inv.new_full(
#         #     (batch_rois.shape[0], feat.shape[1]), feat.shape[0])
#         # scatter_max_cuda.scatter_max_gpu(feat.float(), unq_inv, new_feat_fp32, new_index_fp32)

#         # assert new_index.max().item() <= feat.shape[0]
#         ctx.save_for_backward(unq_inv, new_index)
#         ctx.mark_non_differentiable(new_index)
#         ctx.src_shape = [feat.shape[0], feat.shape[1]]

#         return new_feat

#     @staticmethod
#     # @once_differentiable
#     @custom_bwd
#     def backward(ctx, grad_output1):
#         (unq_inv, new_index) = ctx.saved_tensors
#         # channels_in = ctx.channels_in
#         src_shape = ctx.src_shape
#         # unq_inv = unq_inv.unsqueeze(1).expand(src_shape) #(N, C)
#         src_shape[0] += 1  # (n, c)->(n+1, c)
#         grad_input1 = grad_output1.new_zeros(src_shape)
#         grad_input1.scatter_(0, new_index, grad_output1)
#         grad_input1 = grad_input1.narrow(0, 0, src_shape[0] - 1)
#         # print("grad output max {} min {}".format(grad_output1.max(), grad_output1.min()))
#         # print("grad input max {} min {}".format(grad_input1.max(), grad_input1.min()))
#         return grad_input1, None, None

#     @staticmethod
#     def symbolic(g, feat: torch.FloatTensor, unq_inv: torch.LongTensor, batch_rois: torch.Tensor):
#         # N_out = unq_inv.max() + 1
#         # new_feat = feat.new_full((N_out, feat.shape[1]), -torch.finfo(torch.float).max) #是否会计算梯度
#         # new_index = unq_inv.new_full((N_out, feat.shape[1]), feat.shape[0])
#         return g.op("custom::ScatterMax", feat, unq_inv, batch_rois)

class ScatterMaxFunction_fp32(PyLayer):

    @staticmethod
    # @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, feat: paddle.Tensor, unq_inv: paddle.Tensor, batch_rois: paddle.Tensor):
        """RoIAwarePool3d function forward.

        Args:
            rois (torch.Tensor): [N, 7], in LiDAR coordinate,
                (x, y, z) is the bottom center of rois
            pts (torch.Tensor): [npoints, 3]
            pts_feature (torch.Tensor): [npoints, C]
            out_size (int or tuple): n or [n1, n2, n3]
            max_pts_per_voxel (int): m
            mode (int): 0 (max pool) or 1 (average pool)

        Returns:
            pooled_features (torch.Tensor): [N, out_x, out_y, out_z, C]
        """
        feat = feat#.contiguous()
        unq_inv = unq_inv#.contiguous()
        # new_feat = paddle.zeros(
        #     (batch_rois.shape[0], feat.shape[1])).astype(feat.dtype)  # 是否会计算梯度
        # new_feat.stop_gradient = False
        # # new_feat = feat.new_zeros(
        # #     (batch_rois.shape[0], feat.shape[1]), requires_grad=True)  # 是否会计算梯度
        # new_index = paddle.full(
        #     (batch_rois.shape[0], feat.shape[1]), feat.shape[0]).astype(unq_inv.dtype)
        # new_index = unq_inv.new_full(
        #     (batch_rois.shape[0], feat.shape[1]), feat.shape[0])
        result = scatter_max_op.scatter_max_gpu(feat, unq_inv, int(batch_rois.shape[0]))
        new_feat, new_index = result
        # assert new_index.max().item() <= feat.shape[0]
        ctx.save_for_backward(unq_inv, new_index)
        ctx.mark_non_differentiable(new_index)
        ctx.src_shape = [feat.shape[0], feat.shape[1]]

        return new_feat

    @staticmethod
    # @once_differentiable
    # @custom_bwd
    def backward(ctx, grad_output1):
        (unq_inv, new_index) = ctx.saved_tensor()
        # channels_in = ctx.channels_in
        src_shape = ctx.src_shape
        # unq_inv = unq_inv.unsqueeze(1).expand(src_shape) #(N, C)
        src_shape[0] += 1  # (n, c)->(n+1, c)
        grad_input1 = paddle.zeros(src_shape, dtype=grad_output1.dtype)
        # grad_input1 = grad_output1.new_zeros(src_shape)
        grad_input1.put_along_axis_(new_index, grad_output1, 0)
        # grad_input1.scatter_(0, new_index, grad_output1)
        grad_input1 = grad_input1.slice([0], 0,  0 + src_shape[0] - 1)
        # print("grad output max {} min {}".format(grad_output1.max(), grad_output1.min()))
        # print("grad input max {} min {}".format(grad_input1.max(), grad_input1.min()))
        return grad_input1, None, None

    # @staticmethod
    # def symbolic(g, feat: torch.FloatTensor, unq_inv: torch.LongTensor, batch_rois: torch.Tensor):
    #     # N_out = unq_inv.max() + 1
    #     # new_feat = feat.new_full((N_out, feat.shape[1]), -torch.finfo(torch.float).max) #是否会计算梯度
    #     # new_index = unq_inv.new_full((N_out, feat.shape[1]), feat.shape[0])
    #     return g.op("custom::ScatterMax", feat, unq_inv, batch_rois)


# scatter_max_fp16 = ScatterMaxFunction_fp16.apply
scatter_max_fp32 = ScatterMaxFunction_fp32.apply

# def scatter_max_symbolic(g, feat: torch.FloatTensor, unq_inv: torch.LongTensor, batch_rois: torch.Tensor):
#     return g.op("custom::ScatterMax", feat, unq_inv, batch_rois)


# register_custom_op_symbolic('custom::ScatterMax', scatter_max_symbolic, 16)
