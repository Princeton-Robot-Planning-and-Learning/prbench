"""BaseController module for TidyBot.

This module defines the BaseController class, which provides control logic for the
mobile base using online trajectory generation (Ruckig). It is designed to be used
within the TidyBot simulation and control framework, and supports smooth, constrained
motion for the robot base.

The current controller is part of the environment.
"""

import time

import numpy as np
from ruckig import (  # pylint: disable=no-name-in-module
    InputParameter,
    OutputParameter,
    Result,
    Ruckig,
)

from prbench.envs.tidybot.motion3d import Motion3DEnvSpec


class BaseController:
    """Controller for mobile base movement using online trajectory generation.

    This class implements a controller for the mobile base using Ruckig's online
    trajectory generation to ensure smooth, constrained motion with velocity and
    acceleration limits.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    otg: "Ruckig"
    otg_inp: "InputParameter"
    otg_out: "OutputParameter"
    otg_res: int | None
    motion3d_spec: "Motion3DEnvSpec"
    last_command_time: float | None

    def __init__(
        self, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray, timestep: float
    ) -> None:
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        # OTG (online trajectory generation)
        num_dofs = 3
        self.last_command_time = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = [0.5, 0.5, 3.14]
        self.otg_inp.max_acceleration = [0.5, 0.5, 2.36]
        # self.otg_inp.max_velocity = [0.2, 0.2, 0.5]  # [x, y, theta] velocities
        # self.otg_inp.max_acceleration = [0.2, 0.2, 0.5]  # [x, y, theta] accelerations
        self.otg_res = None
        self.motion3d_spec = Motion3DEnvSpec()

    def reset(self) -> None:
        """Reset the base controller to origin position."""
        # Initialize base at origin
        self.qpos[:] = np.zeros(3)
        self.ctrl[:] = self.qpos
        # Initialize OTG
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command: dict) -> None:
        """Process control commands and update base trajectory."""
        if command is not None:
            self.last_command_time = time.time()
            if "base_pose" in command:
                # Set target base qpos
                self.otg_inp.target_position = command["base_pose"]
                self.otg_res = Result.Working
        # Maintain current pose if command stream is disrupted
        if (
            time.time() - self.last_command_time
            > 2.5 * self.motion3d_spec.policy_control_period
        ):
            self.otg_inp.target_position = self.qpos
            self.otg_res = Result.Working
        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position
