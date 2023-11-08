# !/usr/bin/env python3
# Copyright 2023 Baidu Inc. All Rights Reserved.
# @author: Guojun Wang (wangguojun01@baidu.com)
# @file: roi_align_rotated.py
# @brief: roi_align_rotated
from typing import Any, Optional, Tuple, Union
from collections import Iterable

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer
from paddle3d.ops import roi_align_rotated_cuda
import numpy as np

# class RoIAlignRotatedFunction_fp16(Function):

#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx: Any,
#                 input: torch.Tensor,
#                 rois: torch.Tensor,
#                 output_size: Union[int, tuple],
#                 spatial_scale: float,
#                 pc_range_x: float,
#                 pc_range_y: float,
#                 voxel_size_x: float,
#                 voxel_size_y: float) -> torch.Tensor:
#         ctx.output_size = _pair(output_size)
#         ctx.spatial_scale = spatial_scale
#         ctx.pc_range_x = pc_range_x
#         ctx.pc_range_y = pc_range_y
#         ctx.voxel_size_x = voxel_size_x
#         ctx.voxel_size_y = voxel_size_y
#         if torch.onnx.is_in_onnx_export():
#             print("onnx export enable!!!!")
#             rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois], dim=-1)
#         ctx.save_for_backward(rois)
#         ctx.feature_size = input.size()

#         batch_size, num_channels, data_height, data_width = input.size()
#         num_rois = rois.size(0)

#         output = input.new_zeros(num_rois, num_channels, ctx.output_size[0],
#                                  ctx.output_size[1])
#         roi_align_rotated_cuda.roi_align_rotated_forward(
#             input,
#             rois,
#             output,
#             pooled_height=ctx.output_size[0],
#             pooled_width=ctx.output_size[1],
#             spatial_scale=ctx.spatial_scale,
#             pc_range_x=ctx.pc_range_x,
#             pc_range_y=ctx.pc_range_y,
#             voxel_size_x=ctx.voxel_size_x,
#             voxel_size_y=ctx.voxel_size_y
#         )
#         return output

#     @staticmethod
#     # @once_differentiable
#     @custom_bwd
#     def backward(
#         ctx: Any, grad_output: torch.Tensor
#     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, None,
#                None, None, None]:
#         feature_size = ctx.feature_size
#         rois = ctx.saved_tensors[0]
#         assert feature_size is not None
#         batch_size, num_channels, data_height, data_width = feature_size

#         out_w = grad_output.size(3)
#         out_h = grad_output.size(2)

#         grad_input = grad_rois = None

#         if ctx.needs_input_grad[0]:
#             grad_input = rois.new_zeros(batch_size, num_channels, data_height,
#                                         data_width)
#             roi_align_rotated_cuda.roi_align_rotated_backward(
#                 grad_output.contiguous(),
#                 rois,
#                 grad_input,
#                 pooled_height=out_h,
#                 pooled_width=out_w,
#                 spatial_scale=ctx.spatial_scale,
#                 pc_range_x=ctx.pc_range_x,
#                 pc_range_y=ctx.pc_range_y,
#                 voxel_size_x=ctx.voxel_size_x,
#                 voxel_size_y=ctx.voxel_size_y)
#         return grad_input, grad_rois, None, None, None, None, None

#     @staticmethod
#     def symbolic(g, input, rois, output_size, spatial_scale, pc_range_x, pc_range_y, voxel_size_x, voxel_size_y):
#         if isinstance(output_size, int):
#             out_h = output_size
#             out_w = output_size
#         elif isinstance(output_size, tuple):
#             assert len(output_size) == 2
#             assert isinstance(output_size[0], int)
#             assert isinstance(output_size[1], int)
#             out_h, out_w = output_size
#         else:
#             raise TypeError(
#                 '"output_size" must be an integer or tuple of integers')

#         return g.op(
#             'custom::RoIAlignRotated',
#             input,
#             rois,
#             pooled_height_i=output_size[0],
#             pooled_width_i=output_size[0],
#             spatial_scale_f=spatial_scale,
#             pc_range_x_f=pc_range_x,
#             pc_range_y_f=pc_range_y,
#             voxel_size_x_f=voxel_size_x,
#             voxel_size_y_f=voxel_size_y
#         )

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

class RoIAlignRotatedFunction_fp32(PyLayer):

    @staticmethod
    def forward(ctx: Any,
                input: paddle.Tensor,
                rois: paddle.Tensor,
                output_size: Union[int, tuple],
                spatial_scale: float,
                pc_range_x: float,
                pc_range_y: float,
                voxel_size_x: float,
                voxel_size_y: float) -> paddle.Tensor:
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.pc_range_x = pc_range_x
        ctx.pc_range_y = pc_range_y
        ctx.voxel_size_x = voxel_size_x
        ctx.voxel_size_y = voxel_size_y
        # if torch.onnx.is_in_onnx_export():
        #     print("onnx export enable!!!!")
        #     rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois], dim=-1)
        ctx.save_for_backward(rois)
        ctx.feature_size = input.shape

        batch_size, num_channels, data_height, data_width = input.shape
        num_rois = rois.shape[0]

        # output = paddle.zeros([num_rois, num_channels, ctx.output_size[0],
        #                          ctx.output_size[1]], dtype=input.dtype)
        # output = input.new_zeros(num_rois, num_channels, ctx.output_size[0],
        #                          ctx.output_size[1])
        if rois.shape[0] <= 0:
            output = paddle.ones([num_rois, num_channels, ctx.output_size[0], ctx.output_size[1]], dtype=input.dtype)
            return output
        (output) = roi_align_rotated_cuda.roi_align_rotated_forward(
            input,
            rois,
            # output,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            pc_range_x=ctx.pc_range_x,
            pc_range_y=ctx.pc_range_y,
            voxel_size_x=ctx.voxel_size_x,
            voxel_size_y=ctx.voxel_size_y
        )
        return output

    @staticmethod
    # @once_differentiable
    # @custom_bwd
    def backward(
        ctx: Any, grad_output: paddle.Tensor
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        feature_size = ctx.feature_size
        rois = ctx.saved_tensor()[0]
        assert feature_size is not None
        batch_size, num_channels, data_height, data_width = feature_size

        out_w = grad_output.shape[3]
        out_h = grad_output.shape[2]

        grad_input = grad_rois = None

        # if ctx.needs_input_grad[0]:
            # grad_input = paddle.zeros(batch_size, num_channels, data_height,
            #                             data_width, dtype=rois.dtype)
            # grad_input = rois.new_zeros(batch_size, num_channels, data_height,
            #                             data_width)
        (grad_input) = roi_align_rotated_cuda.roi_align_rotated_backward(
                                    grad_output,
                                    # grad_output.contiguous(),
                                    rois,
                                    # grad_input,
                                    aligned_height=out_h,
                                    aligned_width=out_w,
                                    spatial_scale=ctx.spatial_scale,
                                    pc_range_x=ctx.pc_range_x,
                                    pc_range_y=ctx.pc_range_y,
                                    voxel_size_x=ctx.voxel_size_x,
                                    voxel_size_y=ctx.voxel_size_y,
                                    batch_size=batch_size,
                                    channels=num_channels,
                                    height=data_height,
                                    width=data_width,
                                    )
        return grad_input, grad_rois

    # @staticmethod
    # def symbolic(g, input, rois, output_size, spatial_scale, pc_range_x, pc_range_y, voxel_size_x, voxel_size_y):
    #     if isinstance(output_size, int):
    #         out_h = output_size
    #         out_w = output_size
    #     elif isinstance(output_size, tuple):
    #         assert len(output_size) == 2
    #         assert isinstance(output_size[0], int)
    #         assert isinstance(output_size[1], int)
    #         out_h, out_w = output_size
    #     else:
    #         raise TypeError(
    #             '"output_size" must be an integer or tuple of integers')

    #     return g.op(
    #         'custom::RoIAlignRotated',
    #         input,
    #         rois,
    #         pooled_height_i=output_size[0],
    #         pooled_width_i=output_size[0],
    #         spatial_scale_f=spatial_scale,
    #         pc_range_x_f=pc_range_x,
    #         pc_range_y_f=pc_range_y,
    #         voxel_size_x_f=voxel_size_x,
    #         voxel_size_y_f=voxel_size_y
    #     )


# roi_align_rotated_fp16 = RoIAlignRotatedFunction_fp16.apply
roi_align_rotated_fp32 = RoIAlignRotatedFunction_fp32.apply


class RoIAlignRotated(nn.Layer):
    """RoI align pooling layer for rotated proposals.

    It accepts a feature map of shape (N, C, H, W) and rois with shape
    (n, 6) with each roi decoded as (batch_index, center_x, center_y,
    w, h, angle). The angle is in radian.

    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio(int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
            Default: True.
        clockwise (bool): If True, the angle in each proposal follows a
            clockwise fashion in image space, otherwise, the angle is
            counterclockwise. Default: False.

    Note:
        The implementation of RoIAlign when aligned=True is modified from
        https://github.com/facebookresearch/detectron2/

        The meaning of aligned=True:

        Given a continuous coordinate c, its two neighboring pixel
        indices (in our pixel model) are computed by floor(c - 0.5) and
        ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
        indices [0] and [1] (which are sampled from the underlying signal
        at continuous coordinates 0.5 and 1.5). But the original roi_align
        (aligned=False) does not subtract the 0.5 when computing
        neighboring pixel indices and therefore it uses pixels with a
        slightly incorrect alignment (relative to our pixel model) when
        performing bilinear interpolation.

        With `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5
        prior to calling roi_align. This produces the correct neighbors;

        The difference does not make a difference to the model's
        performance if ROIAlign is used together with conv layers.
    """

    def __init__(self,
                 output_size: Union[int, tuple],
                 spatial_scale: float,
                 pc_range,
                 voxel_size,
                 onnx_test: bool = False):
        super().__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.onnx_test = onnx_test
        self.fp16_enabled = False

    # @auto_fp16(apply_to=('input', 'rois'))
    def forward(self, input: paddle.Tensor, rois: paddle.Tensor) -> paddle.Tensor:
        if not self.onnx_test:
            # if input.dtype == torch.float16:
            #     roi_align_rotated_fn = RoIAlignRotatedFunction_fp16
            # else:
            roi_align_rotated_fn = RoIAlignRotatedFunction_fp32
            return roi_align_rotated_fn.apply(input, rois, self.output_size,
                                              self.spatial_scale,
                                              self.pc_range[0], self.pc_range[1], self.voxel_size[0], self.voxel_size[1])
        else:
            num_rois = rois.size(0)
            output = input[0:1, :, :self.output_size[1], :self.output_size[0]]
            output = output.repeat(num_rois, 1, 1, 1)
            return output

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale}, '
        s += f'sampling_ratio={self.sampling_ratio}, '
        s += f'aligned={self.aligned}, '
        s += f'clockwise={self.clockwise})'
        return s
