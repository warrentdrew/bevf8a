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
# Code written by Oscar Beijbom, 2019.

import os
import unittest

from nuscenes import NuScenes


class TestNuScenes(unittest.TestCase):

    def test_load(self):
        """
        Loads up NuScenes.
        This is intended to simply run the NuScenes class to check for import errors, typos, etc.
        """

        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'
        nusc = NuScenes(version='v1.0-mini', dataroot=os.environ['NUSCENES'], verbose=False)

        # Trivial assert statement
        self.assertEqual(nusc.table_root, os.path.join(os.environ['NUSCENES'], 'v1.0-mini'))


if __name__ == '__main__':
    unittest.main()
