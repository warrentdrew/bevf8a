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
import os

from nuscenes import NuScenes


class TestNuScenesLidarseg(unittest.TestCase):
    def setUp(self):
        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'
        self.nusc = NuScenes(version='v1.0-mini', dataroot=os.environ['NUSCENES'], verbose=False)

    def test_num_classes(self) -> None:
        """
        Check that the correct number of classes (32 classes) are loaded.
        """
        self.assertEqual(len(self.nusc.lidarseg_idx2name_mapping), 32)

    def test_num_colors(self) -> None:
        """
        Check that the number of colors in the colormap matches the number of classes.
        """
        num_classes = len(self.nusc.lidarseg_idx2name_mapping)
        num_colors = len(self.nusc.colormap)
        self.assertEqual(num_colors, num_classes)

    def test_classes(self) -> None:
        """
        Check that the class names match the ones in the colormap, and are in the same order.
        """
        classes_in_colormap = list(self.nusc.colormap.keys())
        for name, idx in self.nusc.lidarseg_name2idx_mapping.items():
            self.assertEqual(name, classes_in_colormap[idx])


if __name__ == '__main__':
    # Runs the tests without throwing errors.
    test = TestNuScenesLidarseg()
    test.setUp()
    test.test_num_classes()
    test.test_num_colors()
    test.test_classes()
