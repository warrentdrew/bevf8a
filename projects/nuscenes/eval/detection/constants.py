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
# Code written by Oscar Beijbom and Varun Bankiti, 2019.
DETECTION_NAMES = ['bigMot', 'smallMot', 'nonMot', 'pedestrian', 'TrainedOthers', 'Mot']

PRETTY_DETECTION_NAMES = {'bigMot': 'bigMot',
        'smallMot': 'smallMot',
        'nonMot': 'nonMot',
        'pedestrian': 'pedestrian',
        'TrainedOthers': 'TrainedOthers',
        'binary_class': 'binary_class'}

DETECTION_COLORS = {'bigMot': 'C0',
                    'smallMot': 'C1',
                    'nonMot': 'C2',
                    'pedestrian': 'C3',
                    'TrainedOthers': 'C4',
                    'binary_class': 'C5'}


ATTRIBUTE_NAMES = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing', 'cycle.with_rider',
                   'cycle.without_rider', 'vehicle.moving', 'vehicle.parked', 'vehicle.stopped']

PRETTY_ATTRIBUTE_NAMES = {'pedestrian.moving': 'Ped. Moving',
                          'pedestrian.sitting_lying_down': 'Ped. Sitting',
                          'pedestrian.standing': 'Ped. Standing',
                          'cycle.with_rider': 'Cycle w/ Rider',
                          'cycle.without_rider': 'Cycle w/o Rider',
                          'vehicle.moving': 'Veh. Moving',
                          'vehicle.parked': 'Veh. Parked',
                          'vehicle.stopped': 'Veh. Stopped'}

#TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']
TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'iou3d_err', 'ioubev_err']

PRETTY_TP_METRICS = {'trans_err': 'Trans.', 'scale_err': 'Scale', 'orient_err': 'Orient.', 'vel_err': 'Vel.',
                     'attr_err': 'Attr.'}

TP_METRICS_UNITS = {'trans_err': 'm',
                    'scale_err': '1-IOU',
                    'orient_err': 'rad.',
                    'vel_err': 'm/s',
                    'attr_err': '1-acc.'}
