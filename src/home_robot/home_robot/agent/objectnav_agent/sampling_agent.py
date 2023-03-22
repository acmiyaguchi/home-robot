# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.nn import DataParallel

import home_robot.utils.pose as pu
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.navigation_planner.discrete_planner import DiscretePlanner

from .objectnav_agent_module import ObjectNavAgentModule


class ObjectNavAgent(Agent):
    """Simple object nav agent based on a 2D semantic map"""

    def __init__(self, config, device_id: int = 0):
        self.max_steps = config.AGENT.max_steps
        self.num_environments = config.NUM_ENVIRONMENTS