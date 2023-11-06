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
import math
import unittest

try:
    import paddle
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as paddle was not found!')

from nuscenes.prediction.models import mtp


class TestMTPLoss(unittest.TestCase):
    """
    Test each component of MTPLoss as well as the
    __call__ method.
    """

    def test_get_trajectories_and_modes(self):

        loss_n_modes_5 = mtp.MTPLoss(5, 0, 0)
        loss_n_modes_1 = mtp.MTPLoss(1, 0, 0)

        xy_pred = paddle.arange(60).reshape([1, -1]).repeat([1, 5]).reshape([-1, 60])
        mode_pred = paddle.arange(5).reshape([1, -1])

        prediction_bs_1 = paddle.concat([xy_pred.reshape([1, -1]), mode_pred], axis=1)
        prediction_bs_2 = prediction_bs_1.repeat([2, 1])

        # Testing many modes with batch size 1.
        traj, modes = loss_n_modes_5._get_trajectory_and_modes(prediction_bs_1)
        self.assertTrue(paddle.allclose(traj, xy_pred.unsqueeze(0).reshape([1, 5, 30, 2])))
        self.assertTrue(paddle.allclose(modes, mode_pred))

        # Testing many modes with batch size > 1.
        traj, modes = loss_n_modes_5._get_trajectory_and_modes(prediction_bs_2)
        self.assertTrue(paddle.allclose(traj, xy_pred.repeat([1, 2]).unsqueeze(0).reshape([2, 5, 30, 2])))
        self.assertTrue(paddle.allclose(modes, mode_pred.repeat([2, 1])))

        xy_pred = paddle.arange(60).reshape([1, -1]).repeat([1, 1]).reshape([-1, 60])
        mode_pred = paddle.arange(1).reshape([1, -1])

        prediction_bs_1 = paddle.cat([xy_pred.reshape([1, -1]), mode_pred], axis=1)
        prediction_bs_2 = prediction_bs_1.repeat([2, 1])

        # Testing one mode with batch size 1.
        traj, modes = loss_n_modes_1._get_trajectory_and_modes(prediction_bs_1)
        self.assertTrue(paddle.allclose(traj, xy_pred.unsqueeze(0).reshape([1, 1, 30, 2])))
        self.assertTrue(paddle.allclose(modes, mode_pred))

        # Testing one mode with batch size > 1.
        traj, modes = loss_n_modes_1._get_trajectory_and_modes(prediction_bs_2)
        self.assertTrue(paddle.allclose(traj, xy_pred.repeat([1, 2]).unsqueeze(0).reshape([2, 1, 30, 2])))
        self.assertTrue(paddle.allclose(modes, mode_pred.repeat([2, 1])))

    def test_angle_between_trajectories(self):

        def make_trajectory(last_point):
            traj = paddle.zeros(([12, 2]))
            traj[-1] = paddle.to_tensor(last_point)
            return traj

        loss = mtp.MTPLoss(0, 0, 0)

        # test angle is 0.
        self.assertEqual(loss._angle_between(make_trajectory([0, 0]), make_trajectory([0, 0])), 0.)
        self.assertEqual(loss._angle_between(make_trajectory([15, 15]), make_trajectory([15, 15])), 0.)

        # test angle is 15.
        self.assertAlmostEqual(loss._angle_between(make_trajectory([1, 1]),
                                                   make_trajectory([math.sqrt(3)/2, 0.5])), 15., places=4)

        # test angle is 30.
        self.assertAlmostEqual(loss._angle_between(make_trajectory([1, 0]),
                                                   make_trajectory([math.sqrt(3)/2, 0.5])), 30., places=4)

        # test angle is 45.
        self.assertAlmostEqual(loss._angle_between(make_trajectory([1, 1]),
                                                   make_trajectory([0, 1])), 45., places=4)

        # test angle is 90.
        self.assertAlmostEqual(loss._angle_between(make_trajectory([1, 1]),
                                                   make_trajectory([-1, 1])), 90., places=4)
        self.assertAlmostEqual(loss._angle_between(make_trajectory([1, 0]),
                                                   make_trajectory([0, 1])), 90., places=4)

        # test angle is 180.
        self.assertAlmostEqual(loss._angle_between(make_trajectory([1, 0]),
                               make_trajectory([-1, 0])), 180., places=4)
        self.assertAlmostEqual(loss._angle_between(make_trajectory([0, 1]),
                                                   make_trajectory([0, -1])), 180., places=4)
        self.assertAlmostEqual(loss._angle_between(make_trajectory([3, 1]),
                                                   make_trajectory([-3, -1])), 180., places=4)

    def test_compute_best_mode_nothing_below_threshold(self):
        angles = [(90, 0), (80, 1), (70, 2)]
        target = None
        traj = None

        loss = mtp.MTPLoss(3, 0, 5)
        self.assertTrue(loss._compute_best_mode(angles, target, traj) in {0, 1, 2})

        loss = mtp.MTPLoss(3, 0, 65)
        self.assertTrue(loss._compute_best_mode(angles, target, traj) in {0, 1, 2})

    def test_compute_best_mode_only_one_below_threshold(self):
        angles = [(30, 1), (3, 0), (25, 2)]

        target = paddle.ones((1, 6, 2))
        trajectory = paddle.zeros((3, 6, 2))

        loss = mtp.MTPLoss(3, 0, 5)
        self.assertEqual(loss._compute_best_mode(angles, target, trajectory), 0)

    def test_compute_best_mode_multiple_below_threshold(self):
        angles = [(2, 2), (4, 1), (10, 0)]
        target = paddle.ones((1, 6, 2))
        trajectory = paddle.zeros((3, 6, 2))
        trajectory[1] = 1

        loss = mtp.MTPLoss(3, 0, 5)
        self.assertEqual(loss._compute_best_mode(angles, target, trajectory), 1)

    def test_compute_best_mode_only_one_mode(self):
        angles = [(25, 0)]
        target = paddle.ones((1, 6, 2))
        trajectory = paddle.zeros((1, 6, 2))

        loss = mtp.MTPLoss(1, 0, 5)
        self.assertEqual(loss._compute_best_mode(angles, target, trajectory), 0)

        trajectory[0] = 1
        self.assertEqual(loss._compute_best_mode(angles, target, trajectory), 0)

    def test_loss_single_mode(self):
        targets = paddle.zeros((16, 1, 30, 2))
        targets[:, :, :, 1] = paddle.arange(start=0, end=3, step=0.1)

        predictions = paddle.ones((16, 61))
        predictions[:, :60] = targets[0, 0, :, :].reshape([-1, 60])
        predictions[:, 60] = 1/10

        loss = mtp.MTPLoss(1, 1, angle_threshold_degrees=20)

        # Only regression loss in single mode case.
        self.assertAlmostEqual(float(loss(predictions, targets).detach().numpy()),
                               0, places=4)

        # Now the best mode differs by 1 from the ground truth.
        # Smooth l1 loss subtracts 0.5 from l1 norm if diff >= 1.
        predictions[:, :60] += 1
        self.assertAlmostEqual(float(loss(predictions, targets).detach().numpy()), 0.5,
                               places=4)

        # In this case, one element has perfect regression, the others are off by 1.
        predictions[1, :60] -= 1
        self.assertAlmostEqual(float(loss(predictions, targets).detach().numpy()),
                               (15/16)*0.5,
                               places=4)

    def test_loss_many_modes(self):
        targets = paddle.zeros((16, 1, 30, 2))
        targets[:, :, :, 1] = paddle.arange(start=0, end=3, step=0.1)

        predictions = paddle.ones((16, 610))
        predictions[:, 540:600] = targets[0, 0, :, :].reshape([-1, 60])
        predictions[:, -10:] = 1/10

        loss = mtp.MTPLoss(10, 1, angle_threshold_degrees=20)

        # Since one mode exactly matches gt, loss should only be classification error.
        self.assertAlmostEqual(float(loss(predictions, targets).numpy()),
                               -math.log(1/10), places=4)

        # Now the best mode differs by 1 from the ground truth.
        # Smooth l1 loss subtracts 0.5 from l1 norm if diff >= 1.
        predictions[:, 540:600] += 1
        self.assertAlmostEqual(float(loss(predictions, targets).numpy()),
                               -math.log(1/10) + 0.5,
                               places=4)

        # In this case, one element has perfect regression, the others are off by 1.
        predictions[1, 540:600] -= 1
        self.assertAlmostEqual(float(loss(predictions, targets).numpy()),
                               -math.log(1/10) + (15/16)*0.5,
                               places=4)
