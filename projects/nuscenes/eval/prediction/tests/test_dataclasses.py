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

import numpy as np

from nuscenes.eval.prediction.data_classes import Prediction


class TestPrediction(unittest.TestCase):

    def test(self):
        prediction = Prediction('instance', 'sample', np.ones((2, 2, 2)), np.zeros(2))

        self.assertEqual(prediction.number_of_modes, 2)
        self.assertDictEqual(prediction.serialize(), {'instance': 'instance',
                                                      'sample': 'sample',
                                                      'prediction': [[[1, 1], [1, 1]],
                                                                     [[1, 1], [1, 1]]],
                                                      'probabilities': [0, 0]})
