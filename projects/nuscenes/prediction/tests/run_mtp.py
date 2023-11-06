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
Regression test to see if MTP can overfit on a single example.
"""

import argparse

import numpy as np
import paddle
import paddle.optimizer as optim
from paddle.io import DataLoader, IterableDataset

from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP, MTPLoss


class Dataset(IterableDataset):
    """
    Implements an infinite dataset where the input data
    is always the same and the target is a path going
    forward with 75% probability, and going backward
    with 25% probability.
    """

    def __init__(self, num_modes: int = 1):
        self.num_modes = num_modes

    def __iter__(self,):

        while True:
            image = paddle.zeros((3, 100, 100))
            agent_state_vector = paddle.ones([3])
            ground_truth = paddle.ones((1, 12, 2))

            if self.num_modes == 1:
                going_forward = True
            else:
                going_forward = np.random.rand() > 0.25

            if going_forward:
                ground_truth[:, :, 1] = paddle.arange(0, 6, step=0.5)
            else:
                ground_truth[:, :, 1] = -paddle.arange(0, 6, step=0.5)

            yield image, agent_state_vector, ground_truth


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run MTP to make sure it overfits on a single test case.')
    parser.add_argument('--num_modes', type=int, help='How many modes to learn.', default=1)
    parser.add_argument('--use_gpu', type=bool, help='Whether to use gpu', default=False)
    args = parser.parse_args()

    dataset = Dataset(args.num_modes)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0)

    backbone = ResNetBackbone('resnet18')
    model = MTP(backbone, args.num_modes)

    loss_function = MTPLoss(args.num_modes, 1, 5)

    current_loss = 10000

    optimizer = optim.SGD(model.parameters(), learning_rate=0.1)

    n_iter = 0

    minimum_loss = 0

    if args.num_modes == 2:

        # We expect to see 75% going_forward and
        # 25% going backward. So the minimum
        # classification loss is expected to be
        # 0.56234

        minimum_loss += 0.56234

    for img, agent_state_vector, ground_truth in dataloader:

        optimizer.clear_grad()

        prediction = model(img, agent_state_vector)
        loss = loss_function(prediction, ground_truth)
        loss.backward()
        optimizer.step()

        current_loss = loss.cpu().detach().numpy()

        print(f"Current loss is {current_loss:.4f}")
        if np.allclose(current_loss, minimum_loss, atol=1e-4):
            print(f"Achieved near-zero loss after {n_iter} iterations.")
            break

        n_iter += 1

