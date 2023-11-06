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
from typing import Dict, Tuple


def get_colormap() -> Dict[str, Tuple[int, int, int]]:
    """
    Get the defined colormap.
    :return: A mapping from the class names to the respective RGB values.
    """

    classname_to_color = {  # RGB.
        "noise": (0, 0, 0),  # Black.
        "animal": (70, 130, 180),  # Steelblue
        "human.pedestrian.adult": (0, 0, 230),  # Blue
        "human.pedestrian.child": (135, 206, 235),  # Skyblue,
        "human.pedestrian.construction_worker": (100, 149, 237),  # Cornflowerblue
        "human.pedestrian.personal_mobility": (219, 112, 147),  # Palevioletred
        "human.pedestrian.police_officer": (0, 0, 128),  # Navy,
        "human.pedestrian.stroller": (240, 128, 128),  # Lightcoral
        "human.pedestrian.wheelchair": (138, 43, 226),  # Blueviolet
        "movable_object.barrier": (112, 128, 144),  # Slategrey
        "movable_object.debris": (210, 105, 30),  # Chocolate
        "movable_object.pushable_pullable": (105, 105, 105),  # Dimgrey
        "movable_object.trafficcone": (47, 79, 79),  # Darkslategrey
        "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
        "vehicle.bicycle": (220, 20, 60),  # Crimson
        "vehicle.bus.bendy": (255, 127, 80),  # Coral
        "vehicle.bus.rigid": (255, 69, 0),  # Orangered
        "vehicle.car": (255, 158, 0),  # Orange
        "vehicle.construction": (233, 150, 70),  # Darksalmon
        "vehicle.emergency.ambulance": (255, 83, 0),
        "vehicle.emergency.police": (255, 215, 0),  # Gold
        "vehicle.motorcycle": (255, 61, 99),  # Red
        "vehicle.trailer": (255, 140, 0),  # Darkorange
        "vehicle.truck": (255, 99, 71),  # Tomato
        "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
        "flat.other": (175, 0, 75),
        "flat.sidewalk": (75, 0, 75),
        "flat.terrain": (112, 180, 60),
        "static.manmade": (222, 184, 135),  # Burlywood
        "static.other": (255, 228, 196),  # Bisque
        "static.vegetation": (0, 175, 0),  # Green
        "vehicle.ego": (255, 240, 245)
    }

    return classname_to_color
