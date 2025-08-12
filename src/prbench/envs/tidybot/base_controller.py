"""BaseController module for TidyBot.

This module defines the BaseController class, which provides control logic for the
mobile base using online trajectory generation (Ruckig). It is designed to be used
within the TidyBot simulation and control framework, and supports smooth, constrained
motion for the robot base.

The current controller is part of the environment.
"""

import time
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
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

    Attributes:
        qpos: Current base pose/state vector, typically ``[x, y, theta]``.
        qvel: Current base velocity vector, typically ``[vx, vy, omega]``.
        ctrl: Actuator target for the base state (same shape as ``qpos``).
        last_command_time: Timestamp (seconds since epoch) of the last received
            command.
        otg: Ruckig trajectory generator instance for the base.
        otg_inp: Ruckig input parameters buffer (contains target/current states and
            motion limits such as ``max_velocity`` and ``max_acceleration``).
        otg_out: Ruckig output parameters buffer (provides new positions for control).
        otg_res: Latest Ruckig result status (e.g., Working/Finished).
        motion3d_spec: Environment timing/specs (e.g., ``policy_control_period``).
        command_timeout_factor: Multiplier applied to ``policy_control_period`` for
            detecting command timeouts.
        reset_qpos: Default pose used by ``reset()`` when initializing the base.
    """

    def __init__(
        self,
        qpos: NDArray[np.float64],
        qvel: NDArray[np.float64],
        ctrl: NDArray[np.float64],
        timestep: float,
        num_dofs: int = 3,
        max_velocity: Optional[Sequence[float]] = None,
        max_acceleration: Optional[Sequence[float]] = None,
        command_timeout_factor: float = 2.5,
        reset_qpos: Optional[NDArray[np.float64]] = None,
    ) -> None:
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.last_command_time: Optional[float] = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        if max_velocity is None:
            max_velocity = [0.5, 0.5, 3.14]
        if max_acceleration is None:
            max_acceleration = [0.5, 0.5, 2.36]
        self.otg_inp.max_velocity = list(max_velocity)
        self.otg_inp.max_acceleration = list(max_acceleration)
        self.otg_res = None
        self.motion3d_spec = Motion3DEnvSpec()
        self.command_timeout_factor = command_timeout_factor
        self.reset_qpos = (
            reset_qpos
            if reset_qpos is not None
            else np.zeros(num_dofs, dtype=np.float64)
        )

    def reset(self) -> None:
        """Reset the base controller to origin position."""
        # Initialize base at origin
        self.qpos[:] = self.reset_qpos
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
            > self.command_timeout_factor * self.motion3d_spec.policy_control_period
        ):
            self.otg_inp.target_position = self.qpos
            self.otg_res = Result.Working
        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position
