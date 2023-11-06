import numpy as np

import paddle
from paddle3d.geometries import BBoxes3D
from paddle3d.utils import bbox_overlaps

def limit_period(val, offset=0.5, period=2 * np.pi):
    return val - paddle.floor(val / period + offset) * period

def nearest_bev(bboxes):
    """torch.Tensor: A tensor of 2D BEV box of each box
    without rotation."""
    # Obtain BEV boxes with rotation in XYWHR format
    # bev_rotated_boxes = bboxes[:, [0, 1, 3, 4, 6]] #self.bev
    bev_rotated_boxes = paddle.index_select(bboxes, paddle.to_tensor([0, 1, 3, 4, 6]), axis=1)
    # convert the rotation to a valid range
    rotations = bev_rotated_boxes[:, -1]
    pi = paddle.to_tensor(np.pi)
    normed_rotations = paddle.abs(limit_period(rotations, 0.5, pi))

    # find the center of boxes
    conditions = (normed_rotations > pi / 4)[..., None]
    bboxes_xywh = paddle.where(conditions, 
                                paddle.index_select(bev_rotated_boxes, paddle.to_tensor([0, 1, 3, 2]), axis=1),
                                bev_rotated_boxes[:, :4])

    centers = bboxes_xywh[:, :2]
    dims = bboxes_xywh[:, 2:]
    bev_boxes = paddle.concat([centers - dims / 2, centers + dims / 2], axis=-1)
    return bev_boxes


def bbox_overlaps_nearest_3d(bboxes1,
                             bboxes2,
                             mode='iou',
                             is_aligned=False,
                             coordinate='lidar'):
    """Calculate nearest 3D IoU.

    Note:
        This function first finds the nearest 2D boxes in bird eye view
        (BEV), and then calculates the 2D IoU using :meth:`bbox_overlaps`.
        Ths IoU calculator :class:`BboxOverlapsNearest3D` uses this
        function to calculate IoUs of boxes.

        If ``is_aligned`` is ``False``, then it calculates the ious between
        each bbox of bboxes1 and bboxes2, otherwise the ious between each
        aligned pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (paddle.Tensor): shape (N, 7+C) [x, y, z, h, w, l, ry, v].
        bboxes2 (paddle.Tensor): shape (M, 7+C) [x, y, z, h, w, l, ry, v].
        mode (str): "iou" (intersection over union) or iof
            (intersection over foreground).
        is_aligned (bool): Whether the calculation is aligned

    Return:
        torch.Tensor: If ``is_aligned`` is ``True``, return ious between \
            bboxes1 and bboxes2 with shape (M, N). If ``is_aligned`` is \
            ``False``, return shape is M.
    """
    assert bboxes1.shape[-1] == bboxes2.shape[-1]
    assert bboxes2.shape[-1] >= 7

    # Change the bboxes to bev
    # box conversion and iou calculation in torch version on CUDA
    # is 10x faster than that in numpy version
    bboxes1_bev = nearest_bev(bboxes1)
    bboxes2_bev = nearest_bev(bboxes2)

    assert is_aligned == False
    
    ret = bbox_overlaps(
        bboxes1_bev, bboxes2_bev, mode=mode) #, is_aligned=is_aligned)
    return ret

