"""ArmController module for TidyBot.

This module defines the ArmController class, which provides control
logic for the robotic arm using inverse kinematics and online trajectory
generation (Ruckig). It is designed to be used within the TidyBot
simulation and control framework, and supports smooth, constrained
motion for the arm and gripper. 

The current controller is part of the environment. 
"""

# pylint: disable=no-member
# pylint: disable=no-name-in-module

import math
import time

import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from prbench.envs.tidybot.ik_solver import IKSolver
from prbench.envs.tidybot.motion3d import Motion3DEnvSpec


class ArmController:
    """Controller for robotic arm movement using inverse kinematics and
    trajectory generation.

    This class implements a controller for the robotic arm using inverse
    kinematics to convert end-effector poses to joint configurations,
    and Ruckig's online trajectory generation for smooth motion control.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    qpos_gripper: np.ndarray
    ctrl_gripper: np.ndarray
    ik_solver: "IKSolver"
    otg: "Ruckig"
    otg_inp: "InputParameter"
    otg_out: "OutputParameter"
    otg_res: int | None
    motion3d_spec: "Motion3DEnvSpec"
    last_command_time: float | None

    def __init__(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        qpos_gripper: np.ndarray,
        ctrl_gripper: np.ndarray,
        timestep: float,
    ) -> None:
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.qpos_gripper = qpos_gripper
        self.ctrl_gripper = ctrl_gripper
        # IK solver
        self.ik_solver = IKSolver(ee_offset=0.12)
        # OTG (online trajectory generation)
        num_dofs = 7
        self.last_command_time = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
        self.otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [
            math.radians(450)
        ]
        self.otg_res = None
        self.motion3d_spec = Motion3DEnvSpec()

    def reset(self) -> None:
        """Reset the arm controller to retract configuration."""
        # Initialize arm in "retract" configuration
        self.qpos[:] = np.array(
            [0.0, -0.34906585, 3.14159265, -2.54818071, 0.0, -0.87266463, 1.57079633]
        )
        self.ctrl[:] = self.qpos
        self.ctrl_gripper[:] = 0.0
        # Initialize OTG
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command: dict) -> None:
        """Process control commands and update arm trajectory."""
        if command is not None:
            self.last_command_time = time.time()
            if "arm_pos" in command:
                # Run inverse kinematics on new target pose
                qpos = self.ik_solver.solve(
                    command["arm_pos"], command["arm_quat"], self.qpos
                )
                qpos = (
                    self.qpos + np.mod((qpos - self.qpos) + np.pi, 2 * np.pi) - np.pi
                )  # Unwrapped joint angles
                # Set target arm qpos
                self.otg_inp.target_position = qpos
                self.otg_res = Result.Working
            if "gripper_pos" in command:
                # Set target gripper pos
                self.ctrl_gripper[:] = (
                    255.0 * command["gripper_pos"]
                )  # fingers_actuator, ctrlrange [0, 255]
        # Maintain current pose if command stream is disrupted
        if (
            time.time() - self.last_command_time
            > 2.5 * self.motion3d_spec.policy_control_period
        ):
            self.otg_inp.target_position = self.otg_out.new_position
            self.otg_res = Result.Working
        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position
