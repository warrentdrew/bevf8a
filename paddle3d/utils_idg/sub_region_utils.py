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
def get_class2id(tasks, sub_type_postfix=None):
    """
    @param tasks:
    @param sub_type_postfix

    return dict {class_name: class_index}
    """
    tasks_class = [t["class_names"] for t in tasks]

    class_names = []
    for names in tasks_class:
        class_names.extend(names)

    class2id = {}
    for i, token in enumerate(class_names):
        class2id[token] =  i
    
    if sub_type_postfix is not None:
        for i, token in enumerate(class_names):
            if (token.endswith(sub_type_postfix)):
                class2id[token] = class2id[token[:-len(sub_type_postfix)]]

    return class2id


def cal_sub_feature_map_range(sub_region_range, full_region_range, map_size=576):
    """
    @param sub_region_range: 
    @param total_scope:
    @param map_size: feature map size

    return [grid_start, grid_end]
    """
    # x_range = [-40, 40], y_range = [-40, 40]
    total_scope = full_region_range[1] - full_region_range[0]
    scope = (sub_region_range[1] - sub_region_range[0]) / total_scope * map_size
    start = int((total_scope / 2 + sub_region_range[0]) / total_scope * map_size)
    end = int(start + scope)
    return start, end


if __name__ == "__main__":
    tasks = [
    dict(num_class=1, class_names=["TrainedOthers"]),
    dict(num_class=1, class_names=["smallMot"]),
    dict(num_class=1, class_names=["bigMot"]),
    dict(num_class=1, class_names=["nonMot"]),
    dict(num_class=1, class_names=["pedestrian"]),
    dict(num_class=1, class_names=["pedestrian_sub"]),
    dict(num_class=1, class_names=["bigMot_sub"]),
    ]
    postfix = "_sub"

    class2id = get_class2id(tasks, postfix)
    print(class2id)