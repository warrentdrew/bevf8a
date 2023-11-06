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
# Code written by Freddy Boulton, Eric Wolff, 2020.
import json
import os
from typing import List, Dict, Any

from nuscenes.eval.prediction.metrics import Metric, deserialize_metric
from nuscenes.prediction import PredictHelper


class PredictionConfig:

    def __init__(self,
                 metrics: List[Metric],
                 seconds: int = 6,
                 frequency: int = 2):
        """
        Data class that specifies the prediction evaluation settings.
        Initialized with:
        metrics: List of nuscenes.eval.prediction.metric.Metric objects.
        seconds: Number of seconds to predict for each agent.
        frequency: Rate at which prediction is made, in Hz.
        """
        self.metrics = metrics
        self.seconds = seconds
        self.frequency = frequency  # Hz

    def serialize(self) -> Dict[str, Any]:
        """ Serialize instance into json-friendly format. """

        return {'metrics': [metric.serialize() for metric in self.metrics],
                'seconds': self.seconds}

    @classmethod
    def deserialize(cls, content: Dict[str, Any], helper: PredictHelper):
        """ Initialize from serialized dictionary. """
        return cls([deserialize_metric(metric, helper) for metric in content['metrics']],
                   seconds=content['seconds'])


def load_prediction_config(helper: PredictHelper, config_name: str = 'predict_2020_icra.json') -> PredictionConfig:
    """
    Loads a PredictionConfig from json file stored in eval/prediction/configs.
    :param helper: Instance of PredictHelper. Needed for OffRoadRate metric.
    :param config_name: Name of json config file.
    :return: PredictionConfig.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(this_dir, "configs", config_name)
    assert os.path.exists(cfg_path), f'Requested unknown configuration {cfg_path}'

    # Load config file and deserialize it.
    with open(cfg_path, 'r') as f:
        config = json.load(f)

    return PredictionConfig.deserialize(config, helper)


