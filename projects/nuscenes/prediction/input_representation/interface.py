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
# Code written by Freddy Boulton 2020.
import abc
from typing import List

import numpy as np


class StaticLayerRepresentation(abc.ABC):
    """ Represents static map information as a numpy array. """

    @abc.abstractmethod
    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        raise NotImplementedError()


class AgentRepresentation(abc.ABC):
    """ Represents information of agents in scene as numpy array. """

    @abc.abstractmethod
    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        raise NotImplementedError()


class Combinator(abc.ABC):
    """ Combines the StaticLayer and Agent representations into a single one. """

    @abc.abstractmethod
    def combine(self, data: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError()


class InputRepresentation:
    """
    Specifies how to represent the input for a prediction model.
    Need to provide a StaticLayerRepresentation - how the map is represented,
    an AgentRepresentation - how agents in the scene are represented,
    and a Combinator, how the StaticLayerRepresentation and AgentRepresentation should be combined.
    """

    def __init__(self, static_layer: StaticLayerRepresentation, agent: AgentRepresentation,
                 combinator: Combinator):

        self.static_layer_rasterizer = static_layer
        self.agent_rasterizer = agent
        self.combinator = combinator

    def make_input_representation(self, instance_token: str, sample_token: str) -> np.ndarray:

        static_layers = self.static_layer_rasterizer.make_representation(instance_token, sample_token)
        agents = self.agent_rasterizer.make_representation(instance_token, sample_token)

        return self.combinator.combine([static_layers, agents])

