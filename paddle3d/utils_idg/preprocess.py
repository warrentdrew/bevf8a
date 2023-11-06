import numpy as np
import paddle

def noise_gt_bboxesv2_(
    gt_boxes,
    thres=0.8,
):
    """random rotate or remove each groundtrutn independently.
    use kitti viewer to test this function points_transform_

    Args:
        gt_boxes: [B, N, 7], gt box in lidar.points_transform_
    """
    # if torch.rand(size=(1,))[0] < thres:
    #     return gt_boxes
    range_config = [
        [0.1, 0.1, np.pi / 12, 0.7],
        [0.2, 0.15, np.pi / 12, 0.6],
        [0.3, 0.20, np.pi / 12, 0.5],
        [0.3, 0.25, np.pi / 9, 0.3],
        [0.4, 0.30, np.pi / 6, 0.2],
    ]
    num_gt, _ = gt_boxes.shape
    idx = paddle.randint(low=0, high=len(range_config), shape=(1,))[0]
    pos_rand = paddle.rand(size=(num_gt, 3), dtype=paddle.float32)
    pos_shift = ((pos_rand - 0.5) / 0.5) * range_config[idx][0]  # (B, N, 3)
    hwl_rand = paddle.rand(size=(num_gt, 3), dtype=paddle.float32)
    hwl_scale = ((hwl_rand - 0.5) / 0.5) * range_config[idx][1] + 1.0
    angle_rand = paddle.rand(size=(num_gt, 1), dtype=paddle.float32)
    angle_rot = ((angle_rand - 0.5) / 0.5) * range_config[idx][2]

    aug_box3d = paddle.concat(
        [gt_boxes[..., 0:3] + pos_shift, gt_boxes[..., 3:6] * hwl_scale, gt_boxes[..., 6:7] + angle_rot],
        axis=-1,
    )
    return aug_box3d