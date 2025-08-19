import logging
import math
import time
from enum import Enum, auto
from typing import Any, Dict, Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore

from prbench.envs.tidybot.agent.mp_policy import (
    MotionPlannerPolicy as MotionPlannerPolicyMP,)


class Policy:
    def reset(self) -> None:
        raise NotImplementedError

    def step(self, obs: Dict[str, Any]) -> Union[Dict[str, Any], str, None]:
        raise NotImplementedError


# Motion planner policy from mp_policy.py (agent)
class MotionPlannerPolicyMPWrapper(Policy):
    def __init__(self, custom_grasp: bool = False) -> None:
        self.impl: MotionPlannerPolicyMP = MotionPlannerPolicyMP(
            custom_grasp=custom_grasp
        )
        self.impl.PLACEMENT_X_OFFSET = 0.1
        self.impl.PLACEMENT_Y_OFFSET = 0.1
        self.impl.PLACEMENT_Z_OFFSET = 0.2
        # self.impl.target_location = np.array([0, 0, 0.5])
        self.episode_ended: bool = False

    def reset(self) -> None:
        self.impl.reset()
        self.episode_ended = False

    def step(self, obs: Dict[str, Any]) -> Union[Dict[str, Any], str, None]:
        action = self.impl.step(obs)
        self.episode_ended = getattr(self.impl, "episode_ended", False)
        return action
