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
import unittest

try:
    import paddle
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as paddle was not found!')

from nuscenes.prediction.models import backbone
from nuscenes.prediction.models import mtp


class TestMTP(unittest.TestCase):

    def setUp(self):
        self.image = paddle.ones((1, 3, 100, 100))
        self.agent_state_vector = paddle.ones((1, 3))
        self.image_5 = paddle.ones((5, 3, 100, 100))
        self.agent_state_vector_5 = paddle.ones((5, 3))

    def _run(self, model):
        pred = model(self.image, self.agent_state_vector)
        pred_5 = model(self.image_5, self.agent_state_vector_5)

        self.assertTupleEqual(pred.shape, (1, 75))
        self.assertTupleEqual(pred_5.shape, (5, 75))

        model.training = False
        pred = model(self.image, self.agent_state_vector)
        self.assertTrue(paddle.allclose(pred[:, -3:].sum(axis=1), paddle.ones([pred.shape[0]])))

    def test_works_with_resnet_18(self,):
        rn_18 = backbone.ResNetBackbone('resnet18')
        model = mtp.MTP(rn_18, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_resnet_34(self,):
        rn_34 = backbone.ResNetBackbone('resnet34')
        model = mtp.MTP(rn_34, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_resnet_50(self,):
        rn_50 = backbone.ResNetBackbone('resnet50')
        model = mtp.MTP(rn_50, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_resnet_101(self,):
        rn_101 = backbone.ResNetBackbone('resnet101')
        model = mtp.MTP(rn_101, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_resnet_152(self,):
        rn_152 = backbone.ResNetBackbone('resnet152')
        model = mtp.MTP(rn_152, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_mobilenet_v2(self,):
        mobilenet = backbone.MobileNetBackbone('mobilenet_v2')
        model = mtp.MTP(mobilenet, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)



