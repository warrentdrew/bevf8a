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
from collections.abc import Mapping, Sequence
from typing import List

import paddle


def dtype2float32(src_tensors):
    """ as name """
    if isinstance(src_tensors,
                  paddle.Tensor) and src_tensors.dtype != 'float32':
        return src_tensors.astype('float32')
    elif isinstance(src_tensors, Sequence):
        return type(src_tensors)([dtype2float32(x) for x in src_tensors])
    elif isinstance(src_tensors, Mapping):
        return {key: dtype2float32(x) for key, x in src_tensors.items()}
    return src_tensors
