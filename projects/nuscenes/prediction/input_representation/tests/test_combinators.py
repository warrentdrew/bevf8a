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

import unittest

import cv2
import numpy as np

from nuscenes.prediction.input_representation.combinators import Rasterizer


class TestRasterizer(unittest.TestCase):

    def test(self):

        layer_1 = np.zeros((100, 100, 3))
        box_1 = cv2.boxPoints(((50, 50), (20, 20), 0))
        layer_1 = cv2.fillPoly(layer_1, pts=[np.int0(box_1)], color=(255, 255, 255))

        layer_2 = np.zeros((100, 100, 3))
        box_2 = cv2.boxPoints(((70, 30), (10, 10), 0))
        layer_2 = cv2.fillPoly(layer_2, pts=[np.int0(box_2)], color=(0, 0, 255))

        rasterizer = Rasterizer()
        image = rasterizer.combine([layer_1.astype('uint8'), layer_2.astype('uint8')])

        answer = np.zeros((100, 100, 3))
        answer = cv2.fillPoly(answer, pts=[np.int0(box_1)], color=(255, 255, 255))
        answer = cv2.fillPoly(answer, pts=[np.int0(box_2)], color=(0, 0, 255))
        answer = answer.astype('uint8')

        np.testing.assert_allclose(answer, image)
