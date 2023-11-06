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

"""
Regression test to see if CoverNet implementation can overfit on a single example.
"""

import argparse
import math

import numpy as np
import paddle
import paddle.optimizer as optim
from paddle.io import DataLoader, IterableDataset

from nuscenes.prediction.models.backbone import MobileNetBackbone
from nuscenes.prediction.models.covernet import CoverNet, ConstantLatticeLoss


def generate_trajectory(theta: float) -> paddle.Tensor:
    trajectory = paddle.zeros([6, 2])
    trajectory[:, 0] = paddle.arange(6) * math.cos(theta)
    trajectory[:, 1] = paddle.arange(6) * math.sin(theta)
    return trajectory


class Dataset(IterableDataset):
    """ Implements an infinite dataset of the same input image, agent state vector and ground truth label. """

    def __iter__(self,):

        while True:
            image = paddle.zeros((3, 100, 100))
            agent_state_vector = paddle.ones([3])
            ground_truth = generate_trajectory(math.pi / 2)

            yield image, agent_state_vector, ground_truth.unsqueeze(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run CoverNet to make sure it overfits on a single test case.')
    parser.add_argument('--use_gpu', type=int, help='Whether to use gpu', default=0)
    args = parser.parse_args()

    dataset = Dataset()
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0)

    backbone = MobileNetBackbone('mobilenet_v2')
    model = CoverNet(backbone, num_modes=3, input_shape=(3, 100, 100))

    lattice = paddle.zeros([3, 6, 2])
    lattice[0] = generate_trajectory(math.pi / 2)
    lattice[1] = generate_trajectory(math.pi / 4)
    lattice[2] = generate_trajectory(3 * math.pi / 4)

    loss_function = ConstantLatticeLoss(lattice)

    optimizer = optim.SGD(model.parameters(), learning_rate=0.1)

    n_iter = 0

    minimum_loss = 0

    for img, agent_state_vector, ground_truth in dataloader:

        optimizer.clear_grad()

        logits = model(img, agent_state_vector)
        loss = loss_function(logits, ground_truth)
        loss.backward()
        optimizer.step()

        current_loss = loss.cpu().detach().numpy()

        print(f"Current loss is {current_loss:.2f}")
        if np.allclose(current_loss, minimum_loss, atol=1e-2):
            print(f"Achieved near-zero loss after {n_iter} iterations.")
            break

        n_iter += 1

