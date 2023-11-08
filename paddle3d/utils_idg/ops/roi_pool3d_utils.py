from typing import Any, Optional, Tuple, Union
from collections import Iterable
import numpy as np

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer
from paddle3d.ops import roipool3d_cuda
from paddle3d.utils_idg.box_paddle_ops import enlarge_box3d

def point_iou_gpuv2(pts, boxes3d1, boxes3d2, extra=[0, 0, 0]):
    """
    :param pts: (N, 3)
    :param boxes3d1: (M, 7)
    :param boxes3d2: (N, 7)
    :return:
        iou3d: (M, N)
    """
    # batch_size, boxes_num, feature_len = (
    #     pts.shape[0],
    #     boxes3d.shape[1],
    #     pts_feature.shape[2],
    # )
    boxes1_num = boxes3d1.shape[0]
    boxes2_num = boxes3d2.shape[0]
    expand_boxes3d1 = enlarge_box3d(boxes3d1.reshape([-1, 7]), extra)
    expand_boxes3d2 = enlarge_box3d(boxes3d2.reshape([-1, 7]), extra)

    iou3d = roipool3d_cuda.point_iou_gpuv2(
        pts,
        expand_boxes3d1,
        expand_boxes3d2,
        # pts.contiguous(),
        # expand_boxes3d1.contiguous(),
        # expand_boxes3d2.contiguous(),
    )

    return iou3d



