"""Utility functions."""

from typing import Any

import numpy as np
from geom2drobotenvs.utils import CRVRobotActionSpace
from numpy.typing import NDArray


def get_geom2d_crv_robot_action_from_gui_input(
    action_space: CRVRobotActionSpace, gui_input: dict[str, Any]
) -> NDArray[np.float32]:
    """Get the mapping from human inputs to actions, derived from action
    space."""
    # Unpack the input.
    keys_pressed = gui_input["keys"]
    right_x, right_y = gui_input["right_stick"]
    left_x, _ = gui_input["left_stick"]

    # Initialize the action.
    low = action_space.low
    high = action_space.high
    action = np.zeros(action_space.shape, action_space.dtype)

    def _rescale(x: float, lb: float, ub: float) -> float:
        """Rescale from [-1, 1] to [lb, ub]."""
        return lb + (x + 1) * (ub - lb) / 2

    # The right stick controls the x, y movement of the base.
    action[0] = _rescale(right_x, low[0], high[0])
    action[1] = _rescale(right_y, low[1], high[1])

    # The left stick controls the rotation of the base. Only the x axis
    # is used right now.
    action[2] = _rescale(left_x, low[2], high[2])

    # The w/s mouse keys are used to adjust the robot arm.
    if "w" in keys_pressed:
        action[3] = low[3]
    if "s" in keys_pressed:
        action[3] = high[3]

    # The space bar is used to turn on the vacuum.
    if "space" in keys_pressed:
        action[4] = 1.0

    return action
