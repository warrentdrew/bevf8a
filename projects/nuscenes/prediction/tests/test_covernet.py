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
# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

import math
import unittest

try:
    import paddle
    from paddle.nn.functional import cross_entropy
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as paddle was not found!')

from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.covernet import mean_pointwise_l2_distance, ConstantLatticeLoss, CoverNet


class TestCoverNet(unittest.TestCase):

    def test_shapes_in_forward_pass_correct(self):
        resnet = ResNetBackbone('resnet50')

        covernet = CoverNet(resnet, 5, n_hidden_layers=[4096], input_shape=(3, 100, 100))

        image = paddle.zeros([4, 3, 100, 100])
        asv = paddle.randn([4, 3])

        logits = covernet(image, asv)
        self.assertTupleEqual(logits.shape, (4, 5))


class TestConstantLatticeLoss(unittest.TestCase):

    def test_l1_distance(self):

        lattice = paddle.zeros([3, 6, 2])
        lattice[0] = paddle.arange([1, 13]).reshape([6, 2])
        lattice[1] = paddle.arange([1, 13]).reshape([6, 2]) * 3
        lattice[2] = paddle.arange([1, 13]).reshape([6, 2]) * 6

        # Should select the first mode
        ground_truth = paddle.arange(1, 13, dtype='float32').reshape([6, 2]).unsqueeze(0) + 2
        self.assertEqual(mean_pointwise_l2_distance(lattice, ground_truth), 0)

        # Should select the second mode
        ground_truth = paddle.arange(1, 13, dtype='float32').reshape([6, 2]).unsqueeze(0) * 3 + 4
        self.assertEqual(mean_pointwise_l2_distance(lattice, ground_truth), 1)

        # Should select the third mode
        ground_truth = paddle.arange(1, 13, dtype='float32').reshape([6, 2]).unsqueeze(0) * 6 + 10
        self.assertEqual(mean_pointwise_l2_distance(lattice, ground_truth), 2)

    def test_constant_lattice_loss(self):


        def generate_trajectory(theta: float) -> paddle.Tensor:
            trajectory = paddle.zeros([6, 2])
            trajectory[:, 0] = paddle.arange(6, dtype='float32') * math.cos(theta)
            trajectory[:, 1] = paddle.arange(6, dtype='float32') * math.sin(theta)
            return trajectory

        lattice = paddle.zeros([3, 6, 2])
        lattice[0] = generate_trajectory(math.pi / 2)
        lattice[1] = generate_trajectory(math.pi / 4)
        lattice[2] = generate_trajectory(3 * math.pi / 4)

        ground_truth = paddle.zeros([5, 1, 6, 2])
        ground_truth[0, 0] = generate_trajectory(0.2)
        ground_truth[1, 0] = generate_trajectory(math.pi / 3)
        ground_truth[2, 0] = generate_trajectory(5 * math.pi / 6)
        ground_truth[3, 0] = generate_trajectory(6 * math.pi / 11)
        ground_truth[4, 0] = generate_trajectory(4 * math.pi / 9)

        logits = paddle.to_tensor([[2, 10, 5],
                               [-3, 4, 5],
                               [-4, 2, 7],
                               [8, -2, 3],
                               [10, 3, 6]])

        answer = cross_entropy(logits, paddle.to_tensor([1, 1, 2, 0, 0]).cast('int64'))

        loss = ConstantLatticeLoss(lattice, mean_pointwise_l2_distance)
        loss_value = loss(logits, ground_truth)

        self.assertAlmostEqual(float(loss_value.numpy()), float(answer.numpy()))
