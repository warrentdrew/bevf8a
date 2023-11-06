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
    from paddle.vision.models.ResNet import BasicBlock, Bottleneck
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as paddle was not found!')

from nuscenes.prediction.models.backbone import ResNetBackbone, MobileNetBackbone


class TestBackBones(unittest.TestCase):

    def count_layers(self, model):
        if isinstance(model[4][0], BasicBlock):
            n_convs = 2
        elif isinstance(model[4][0], Bottleneck):
            n_convs = 3
        else:
            raise ValueError("Backbone layer block not supported!")

        return sum([len(model[i]) for i in range(4, 8)]) * n_convs + 2

    def test_resnet(self):

        rn_18 = ResNetBackbone('resnet18')
        rn_34 = ResNetBackbone('resnet34')
        rn_50 = ResNetBackbone('resnet50')
        rn_101 = ResNetBackbone('resnet101')
        rn_152 = ResNetBackbone('resnet152')

        tensor = paddle.ones((1, 3, 100, 100))

        self.assertEqual(rn_18(tensor).shape[1], 512)
        self.assertEqual(rn_34(tensor).shape[1], 512)
        self.assertEqual(rn_50(tensor).shape[1], 2048)
        self.assertEqual(rn_101(tensor).shape[1], 2048)
        self.assertAlmostEqual(rn_152(tensor).shape[1], 2048)

        self.assertEqual(self.count_layers(list(rn_18.backbone.children())), 18)
        self.assertEqual(self.count_layers(list(rn_34.backbone.children())), 34)
        self.assertEqual(self.count_layers(list(rn_50.backbone.children())), 50)
        self.assertEqual(self.count_layers(list(rn_101.backbone.children())), 101)
        self.assertEqual(self.count_layers(list(rn_152.backbone.children())), 152)

        with self.assertRaises(ValueError):
            ResNetBackbone('resnet51')

    def test_mobilenet(self):

        mobilenet = MobileNetBackbone('mobilenet_v2')

        tensor = paddle.ones((1, 3, 100, 100))

        self.assertEqual(mobilenet(tensor).shape[1], 1280)