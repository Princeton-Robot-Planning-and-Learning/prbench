"""Motion planner policy for TidyBot.

This module defines state machines and utilities to move the mobile base and
manipulator to pick and place objects using ground-truth observations from the
simulated environment. It provides a high-level policy that produces actions
compatible with the `MujocoEnv` interface.
"""

import math
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from prbench.envs.tidybot.agent.base_agent import BaseAgent


class PickState(Enum):
    """States of the pick subroutine."""

    APPROACH = "approach"
    LOWER = "lower"
    GRASP = "grasp"
    LIFT = "lift"
    RETURN_HOME = "return_home"


class PlaceState(Enum):
    """States of the place subroutine."""

    APPROACH = "approach"
    LOWER = "lower"
    RELEASE = "release"
    HOME = "home"


# Motion Planner generated plan.
class MotionPlannerPolicy(BaseAgent):
    """High-level mobile manipulation policy for pick-and-place.

    The policy uses a waypoint-based base controller and a small state machine for
    grasping and placement. It consumes observation dictionaries and produces low-level
    actions for base, arm, and gripper.
    """

    # Base following parameters (from BaseController)
    LOOKAHEAD_DISTANCE = 0.3
    POSITION_TOLERANCE = 0.005
    GRASP_BASE_TOLERANCE = 0.002
    PLACE_BASE_TOLERANCE = 0.02

    # Object and target locations
    PLACEMENT_X_OFFSET = 1.0
    PLACEMENT_Y_OFFSET = 0.3
    PLACEMENT_Z_OFFSET = 0.0

    # Manipulation parameters
    ROBOT_BASE_HEIGHT = 0.48
    PICK_APPROACH_HEIGHT_OFFSET = 0.25
    PICK_LOWER_DIST = 0.08
    PICK_LIFT_DIST = 0.28
    PLACE_APPROACH_HEIGHT_OFFSET = 0.10
    SIDE_PLACE_OFFSET = 0.1

    # Grasping parameters
    GRASP_SUCCESS_THRESHOLD = 0.7
    GRASP_PROGRESS_THRESHOLD = 0.3
    GRASP_TIMEOUT_S = 3.0
    PLACE_SUCCESS_THRESHOLD = 0.2

    def __init__(
        self,
        cupboard_mode: bool = False,
        custom_grasp: bool = False,
        low_grasp: bool = False,
    ) -> None:
        """Initialize MotionPlannerPolicy.

        Args:
            cupboard_mode (bool): Enable cupboard-specific placement behavior
            custom_grasp (bool): Enable experimental grasping parameters for testing
        """
        super().__init__()
        # Motion planning state - following controller.py pattern
        self.state: str = "idle"  # States: idle, moving, manipulating, grasping
        self.current_command: Optional[Dict[str, Any]] = None
        self.base_waypoints: List[List[float]] = []
        self.current_waypoint_idx: int = 0
        self.target_ee_pos: Optional[List[float]] = None
        self.grasp_state: Optional[Union[PickState, PlaceState]] = (
            None  # Replaces grasp_step
        )

        # Base following parameters
        self.lookahead_position: Optional[List[float]] = None

        # Object and target locations (using ground truth from MuJoCo)
        self.object_location: Optional[np.ndarray] = None
        self.target_location: Optional[np.ndarray] = None

        # Enable policy execution immediately (no web interface required)
        self.enabled: bool = True
        self.episode_ended: bool = False

        self.cupboard_mode: bool = cupboard_mode
        self.custom_grasp: bool = custom_grasp
        self.low_grasp: bool = low_grasp

    def reset(self) -> None:
        """Reset internal state and prepare for a new episode."""
        # Reset motion planning state
        self.state = "idle"
        self.current_command = None
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.lookahead_position = None
        self.episode_ended = False
        self.grasp_state = None
        # Clean up any grasp tracking variables
        if hasattr(self, "grasp_start_time"):
            delattr(self, "grasp_start_time")
        if hasattr(self, "initial_gripper_pos"):
            delattr(self, "initial_gripper_pos")
        # Enable policy execution immediately
        self.enabled = True
        print("Motion planner reset - starting episode automatically")

    def step(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Advance the policy by one step and return an action.

        Returns None when the policy is disabled or the episode has ended.
        """
        # Return no action if episode has ended
        if self.episode_ended:
            return None

        # Return no action if robot is not enabled
        if not self.enabled:
            return None

        return self._step(obs)

    def _step(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Internal step implementation operating the state machine."""
        # Extract current state
        base_pose = obs["base_pose"]
        arm_pos = obs["arm_pos"]
        arm_quat = obs["arm_quat"]
        gripper_pos = obs["gripper_pos"]

        # State machine following controller.py pattern
        if self.state == "idle":
            # Detect objects and plan new command
            detected_objects = self.detect_objects_from_ground_truth(obs)

            if detected_objects:
                # Create pick command
                self.object_location = detected_objects[0]

                # Safety check: ensure object_location is a proper 3D array
                if (
                    not hasattr(self.object_location, "shape")
                    or len(self.object_location.shape) != 1
                    or self.object_location.shape[0] != 3
                ):
                    print(
                        f"""ERROR: object_location is not a 3D position array.
                        Shape: {self.object_location.shape
                        if hasattr(self.object_location, 'shape') else 'no shape'}"""
                    )
                    print(f"ERROR: object_location value: {self.object_location}")
                    return None

                # Set placement location relative to detected object (e.g., 50cm away)
                if self.target_location is None:
                    self.target_location = np.array(
                        [
                            self.object_location[0]
                            + self.PLACEMENT_X_OFFSET,  # 50cm in X direction
                            self.object_location[1]
                            + self.PLACEMENT_Y_OFFSET,  # Same Y as object
                            self.object_location[2]
                            + self.PLACEMENT_Z_OFFSET,  # Same Z as object (table height)
                        ]
                    )
                pick_command: Dict[str, Any] = {
                    "primitive_name": "pick",
                    "waypoints": [
                        base_pose[:2].tolist(),
                        self.object_location[:2].tolist(),
                    ],
                    "object_3d_pos": self.object_location.copy(),
                }

                print(
                    f"""Object detected at: [{self.object_location[0]:.3f},
                    {self.object_location[1]:.3f}, {self.object_location[2]:.3f}]"""
                )
                print(
                    f"""Target placement location: [{self.target_location[0]:.3f},
                    {self.target_location[1]:.3f}, {self.target_location[2]:.3f}]"""
                )
                print(
                    f"Creating pick command with waypoints: {pick_command['waypoints']}"  # pylint: disable=line-too-long
                )

                # Build base command and start moving
                base_command = self.build_base_command(pick_command)
                if base_command:
                    self.current_command = pick_command
                    self.base_waypoints = base_command["waypoints"]
                    self.target_ee_pos = base_command["target_ee_pos"]
                    self.current_waypoint_idx = 1
                    self.lookahead_position = None
                    self.state = "moving"
                    print(f"Starting base movement to object at {self.object_location}")
                    print(f"Base waypoints: {self.base_waypoints}")
                    print(f"Target EE position: {self.target_ee_pos}")
                else:
                    print("Failed to build base command")
            else:
                # No objects found, do nothing
                return None

        elif self.state == "moving":
            assert self.current_command is not None
            # Execute base movement following waypoints (like BaseController)
            action = self.execute_base_movement(obs)
            if action is None:  # Base movement complete
                print("Base movement complete!")
                # Check if we're close enough for arm manipulation
                if self.target_ee_pos is not None:
                    distance_to_target = self.distance(
                        base_pose[:2], self.target_ee_pos
                    )
                    end_effector_offset = self.get_end_effector_offset()
                    diff = abs(end_effector_offset - distance_to_target)
                    print(
                        f"Distance to target EE: {distance_to_target:.3f}, EE offset: {end_effector_offset:.3f}, diff: {diff:.3f}"  # pylint: disable=line-too-long
                    )
                    if self.current_command["primitive_name"] == "pick":
                        base_tolerance = self.GRASP_BASE_TOLERANCE
                    else:
                        base_tolerance = self.PLACE_BASE_TOLERANCE
                    if diff < base_tolerance:
                        # if diff < 0.002:  # 0.2 cm tolerance (reduced from 10 cm)
                        self.state = "manipulating"
                        print("Base reached target, starting arm manipulation")
                    else:
                        print(
                            f"Too far from target end effector position ({(100 * diff):.1f} cm)"  # pylint: disable=line-too-long
                        )
                        self.state = "idle"
                else:
                    self.state = "idle"
            else:
                # Print target base pose from action
                if action and "base_pose" in action:
                    target_pose = action["base_pose"]
                    print(
                        f"Target base pose: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]"  # pylint: disable=line-too-long
                    )
            return action

        elif self.state == "manipulating":
            assert self.current_command is not None
            # Execute arm manipulation
            if self.current_command["primitive_name"] == "pick":
                if self.grasp_state is None:
                    self.grasp_state = PickState.APPROACH

                # Define default targets to hold current pose
                target_arm_pos = arm_pos.copy()
                target_arm_quat = arm_quat.copy()
                target_gripper_pos = gripper_pos.copy()

                # Position arm above object and close gripper to grasp
                object_3d_pos = self.current_command["object_3d_pos"]
                # Calculate global position difference with better approach height
                global_diff = np.array(
                    [
                        object_3d_pos[0] - base_pose[0],
                        object_3d_pos[1] - base_pose[1],
                        object_3d_pos[2]
                        + self.PICK_APPROACH_HEIGHT_OFFSET
                        - self.ROBOT_BASE_HEIGHT,
                    ]
                )

                # Transform to base's local coordinate frame (account for base rotation)
                base_angle = base_pose[2]
                cos_angle = math.cos(-base_angle)  # Negative for inverse rotation
                sin_angle = math.sin(-base_angle)

                object_relative_pos = np.array(
                    [
                        cos_angle * global_diff[0] - sin_angle * global_diff[1],
                        sin_angle * global_diff[0] + cos_angle * global_diff[1],
                        global_diff[2],  # Z component unchanged
                    ]
                )
                print(f"Primary path: global_diff = {global_diff}")

                print(
                    f"Base angle: {base_pose[2]:.3f} rad ({math.degrees(base_pose[2]):.1f} deg)"  # pylint: disable=line-too-long
                )
                print(
                    f"Object global pos: {self.current_command.get('object_3d_pos', 'N/A')}"  # pylint: disable=line-too-long
                )

                print(f"Current arm pos: {arm_pos}")
                print(
                    f"Position error: {np.linalg.norm(arm_pos - object_relative_pos):.4f}m"  # pylint: disable=line-too-long
                )
                print(f"Grasp state: {self.grasp_state}")

                if self.grasp_state == PickState.APPROACH:
                    # Step 1: Position arm above object with open gripper (safe approach)
                    target_arm_pos = object_relative_pos
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([0.0])  # Gripper open

                    print("Step 1: Positioning arm above object with open gripper")
                    if np.allclose(
                        arm_pos, target_arm_pos, atol=0.03
                    ):  # Tighter tolerance: 3cm
                        self.grasp_state = PickState.LOWER
                        print("Arm positioned above object, moving to lower approach")

                elif self.grasp_state == PickState.LOWER:
                    # Step 2: Lower gripper closer to object for precise grasping
                    target_arm_pos = object_relative_pos.copy()
                    target_arm_pos[
                        2
                    ] -= self.PICK_LOWER_DIST  # Lower by 8cm for closer approach
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([0.0])  # Gripper open

                    print(
                        f"Step 2: Lowering gripper for precise approach... target: {target_arm_pos[2]:.3f}, current: {arm_pos[2]:.3f}"  # pylint: disable=line-too-long
                    )
                    if np.allclose(
                        arm_pos, target_arm_pos, atol=0.02
                    ):  # Very tight tolerance: 2cm
                        self.grasp_state = PickState.GRASP
                        print("Gripper lowered to grasping position, closing gripper")

                elif self.grasp_state == PickState.GRASP:
                    # Step 3: Close gripper to grasp
                    target_arm_pos = object_relative_pos.copy()
                    target_arm_pos[
                        2
                    ] -= self.PICK_LOWER_DIST  # Maintain lowered position
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([1.0])  # Close gripper

                    print(
                        f"Step 3: Closing gripper... current position: {gripper_pos[0]:.3f}"  # pylint: disable=line-too-long
                    )

                    # Initialize grasp attempt tracking
                    if not hasattr(self, "grasp_start_time"):
                        self.grasp_start_time = (  # pylint: disable=attribute-defined-outside-init
                            time.time()
                        )
                        self.initial_gripper_pos = gripper_pos[  # pylint: disable=attribute-defined-outside-init
                            0
                        ]
                        print(
                            f"Started grasp attempt, initial gripper pos: {self.initial_gripper_pos:.3f}"  # pylint: disable=line-too-long
                        )

                    # Check for successful grasp (multiple criteria)
                    gripper_closed_enough = (
                        gripper_pos[0] > self.GRASP_SUCCESS_THRESHOLD
                    )
                    gripper_progress = (
                        gripper_pos[0] - self.initial_gripper_pos
                    ) > self.GRASP_PROGRESS_THRESHOLD
                    grasp_timeout = (
                        time.time() - self.grasp_start_time
                    ) > self.GRASP_TIMEOUT_S

                    if gripper_closed_enough or gripper_progress or grasp_timeout:
                        if gripper_closed_enough or gripper_progress:
                            print(
                                f"""Grasp successful! Gripper pos: {gripper_pos[0]:.3f},
                                progress: {gripper_pos[0] - self.initial_gripper_pos:.3f}"""  # pylint: disable=line-too-long
                            )
                        else:
                            print(
                                f"Grasp timeout reached, proceeding with current grip: {gripper_pos[0]:.3f}"  # pylint: disable=line-too-long
                            )

                        self.grasp_state = PickState.LIFT
                        # Clean up tracking variables
                        delattr(self, "grasp_start_time")
                        delattr(self, "initial_gripper_pos")
                        print("Moving to lift phase!")

                elif self.grasp_state == PickState.LIFT:
                    # Step 4: Lift object from the grasping position
                    lifted_pos = object_relative_pos.copy()
                    lifted_pos[2] += (
                        self.PICK_LIFT_DIST - self.PICK_LOWER_DIST
                    )  # Net lift
                    target_arm_pos = lifted_pos
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([1.0])  # Gripper closed

                    print(
                        f"""Step 4: Lifting object... target height:
                        {target_arm_pos[2]:.3f}, current: {arm_pos[2]:.3f}"""
                    )
                    if np.allclose(arm_pos, target_arm_pos, atol=0.05):  # 5cm tolerance
                        print(
                            "Object lifted successfully! Now moving to placement location."  # pylint: disable=line-too-long
                        )
                        # Create place command
                        assert self.target_location is not None
                        place_command: Dict[str, Any] = {
                            "primitive_name": "place",
                            "waypoints": [
                                base_pose[:2].tolist(),
                                self.target_location[:2].tolist(),
                            ],
                            "target_3d_pos": self.target_location.copy(),
                        }

                        base_command = self.build_base_command(place_command)
                        if base_command:
                            self.current_command = place_command
                            self.base_waypoints = base_command["waypoints"]
                            self.target_ee_pos = base_command["target_ee_pos"]
                            self.current_waypoint_idx = 1
                            self.lookahead_position = None
                            self.state = "moving"
                            self.grasp_state = None  # Reset for next manipulation
                            print(
                                f"Starting base movement to placement location at {self.target_location}"  # pylint: disable=line-too-long
                            )
                        else:
                            print("Failed to build place command")
                            self.episode_ended = True
                            self.state = "idle"

                # Create action from targets
                out_action: Dict[str, Any] = {
                    "base_pose": base_pose.copy(),
                    "arm_pos": target_arm_pos,
                    "arm_quat": target_arm_quat,
                    "gripper_pos": target_gripper_pos,
                }
                return out_action

            if self.current_command["primitive_name"] == "place":
                if self.grasp_state is None:
                    self.grasp_state = PlaceState.APPROACH

                # Define default targets to hold current pose
                target_arm_pos = arm_pos.copy()
                target_arm_quat = arm_quat.copy()
                target_gripper_pos = gripper_pos.copy()

                # Position arm above placement location
                target_3d_pos = self.current_command["target_3d_pos"]
                # Calculate global position difference for approach (above target)
                global_diff = np.array(
                    [
                        target_3d_pos[0] - base_pose[0],
                        target_3d_pos[1] - base_pose[1],
                        target_3d_pos[2]
                        + self.PLACE_APPROACH_HEIGHT_OFFSET
                        - self.ROBOT_BASE_HEIGHT,
                    ]
                )
                # For lowering, use no offset
                global_diff_lower = np.array(
                    [
                        target_3d_pos[0] - base_pose[0],
                        target_3d_pos[1] - base_pose[1],
                        target_3d_pos[2] - self.ROBOT_BASE_HEIGHT,
                    ]
                )
                # Transform to base's local coordinate frame (account for base rotation)
                base_angle = base_pose[2]
                cos_angle = math.cos(-base_angle)
                sin_angle = math.sin(-base_angle)
                target_relative_pos = np.array(
                    [
                        cos_angle * global_diff[0] - sin_angle * global_diff[1],
                        sin_angle * global_diff[0] + cos_angle * global_diff[1],
                        global_diff[2],
                    ]
                )
                target_relative_pos_lower = np.array(
                    [
                        cos_angle * global_diff_lower[0]
                        - sin_angle * global_diff_lower[1],
                        sin_angle * global_diff_lower[0]
                        + cos_angle * global_diff_lower[1],
                        global_diff_lower[2],
                    ]
                )
                # Home position (in base frame, e.g., [0.4, 0, 0.4])
                arm_home_pos = np.array([[0.14322269, 0.0, 0.20784938]])
                # arm_home_pos = np.array([[0.1, 0.0, 0.2]])

                arm_home_quat = np.array(
                    [0.707, 0.707, 0, 0]
                )  # np.array([1.0, 0.0, 0.0, 0.0])

                print(f"Placing: target_relative_pos = {target_relative_pos}")
                print(f"Target EE pos: {self.target_ee_pos}, Base pose: {base_pose}")
                print(f"Grasp state: {self.grasp_state}")

                if self.grasp_state == PlaceState.APPROACH:
                    # Step 1: Position arm above placement location with closed gripper
                    target_arm_pos = target_relative_pos
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
                    # target_arm_quat = np.array([ 0.707, 0, 0.707, 0 ]) # point forward
                    target_gripper_pos = np.array([1.0])
                    print(
                        "Step 1: Positioning arm above placement location with closed gripper"  # pylint: disable=line-too-long
                    )
                    if np.allclose(arm_pos, target_arm_pos, atol=0.03):
                        self.grasp_state = PlaceState.LOWER
                        print("Arm above placement, lowering...")
                elif self.grasp_state == PlaceState.LOWER:
                    # Step 2: Lower arm to placement height
                    target_arm_pos = target_relative_pos_lower
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
                    target_gripper_pos = np.array([1.0])
                    print("Step 2: Lowering arm to placement height")
                    if np.allclose(arm_pos, target_arm_pos, atol=0.02):
                        self.grasp_state = PlaceState.RELEASE
                        print("Arm at placement height, opening gripper...")
                elif self.grasp_state == PlaceState.RELEASE:
                    # Step 3: Open gripper to place object
                    target_arm_pos = target_relative_pos_lower
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
                    target_gripper_pos = np.array([0.0])
                    print("Step 3: Opening gripper to release object")
                    if gripper_pos[0] < self.PLACE_SUCCESS_THRESHOLD:
                        self.grasp_state = PlaceState.HOME
                        print("Object placed, moving to home position...")
                elif self.grasp_state == PlaceState.HOME:
                    # Step 4: Move arm to home position
                    target_arm_pos = arm_home_pos
                    target_arm_quat = arm_home_quat
                    target_gripper_pos = np.array([0.0])
                    print("Step 4: Moving arm to home position")
                    if np.allclose(arm_pos, target_arm_pos, atol=0.03):
                        print("Arm at home position. Task complete.")
                        self.episode_ended = True
                        self.state = "idle"
                # Create action from targets
                out_action = {
                    "base_pose": base_pose.copy(),
                    "arm_pos": target_arm_pos,
                    "arm_quat": target_arm_quat,
                    "gripper_pos": target_gripper_pos,
                }
                return out_action

        # Default: hold current pose
        out_action = {
            "base_pose": base_pose.copy(),
            "arm_pos": arm_pos.copy(),
            "arm_quat": arm_quat.copy(),
            "gripper_pos": gripper_pos.copy(),
        }
        print("Default action - holding current pose")
        return out_action

    def execute_base_movement(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute base movement following waypoints like BaseController."""
        base_pose = obs["base_pose"]

        # Check if we've reached the final waypoint
        if self.current_waypoint_idx >= len(self.base_waypoints):
            print("All waypoints completed")
            return None  # Movement complete

        print(
            f"Current waypoint index: {self.current_waypoint_idx}/{len(self.base_waypoints)}"  # pylint: disable=line-too-long
        )
        print(f"Base waypoints: {self.base_waypoints}")

        # Compute lookahead position (simplified version of BaseController logic)
        while True:
            if self.current_waypoint_idx >= len(self.base_waypoints):
                self.lookahead_position = None
                print("Reached end of waypoints, no lookahead")
                break

            start = self.base_waypoints[self.current_waypoint_idx - 1]
            end = self.base_waypoints[self.current_waypoint_idx]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - base_pose[0], start[1] - base_pose[1])
            t2 = self.intersect(d, f, self.LOOKAHEAD_DISTANCE)

            print(f"Waypoint {self.current_waypoint_idx}: start={start}, end={end}")
            print(f"d={d}, f={f}, t2={t2}")

            if t2 is not None:
                self.lookahead_position = [start[0] + t2 * d[0], start[1] + t2 * d[1]]
                print(f"Lookahead position: {self.lookahead_position}")
                break
            if self.current_waypoint_idx == len(self.base_waypoints) - 1:
                self.lookahead_position = None
                print("Last waypoint, no lookahead")
                break
            print(f"Moving to next waypoint: {self.current_waypoint_idx + 1}")
            self.current_waypoint_idx += 1

        # Determine target position
        if self.lookahead_position is None:
            target_position = self.base_waypoints[-1]
            print(f"Using final waypoint as target: {target_position}")
            # Check if we've reached the final position
            position_error = self.distance(base_pose[:2], target_position)
            print(
                f"""Position error to final target: {position_error:.3f}
                (tolerance: {self.POSITION_TOLERANCE})"""
            )
            if position_error < self.POSITION_TOLERANCE:
                print("Reached final position within tolerance")
                return None  # Movement complete
        else:
            target_position = self.lookahead_position
            print(f"Using lookahead as target: {target_position}")

        # Compute target heading
        target_heading = base_pose[2]
        if self.target_ee_pos is not None:
            # Turn to face target end effector position
            dx = self.target_ee_pos[0] - base_pose[0]
            dy = self.target_ee_pos[1] - base_pose[1]
            desired_heading = math.atan2(
                dy, dx
            )  # Removed + math.pi to point towards target, not away

            print(
                f"""Target EE: {self.target_ee_pos}, dx={dx:.3f},
                dy={dy:.3f}, desired_heading={desired_heading:.3f}"""
            )

            frac = 1.0
            if self.lookahead_position is not None:
                # Turn slowly at first, more quickly as we approach
                remaining_path_length = self.LOOKAHEAD_DISTANCE
                curr_waypoint = self.lookahead_position
                for idx in range(self.current_waypoint_idx, len(self.base_waypoints)):
                    next_waypoint = self.base_waypoints[idx]
                    remaining_path_length += self.distance(curr_waypoint, next_waypoint)
                    curr_waypoint = next_waypoint
                frac = math.sqrt(
                    self.LOOKAHEAD_DISTANCE
                    / max(remaining_path_length, self.LOOKAHEAD_DISTANCE)
                )

            heading_diff = self.restrict_heading_range(desired_heading - base_pose[2])
            target_heading += frac * heading_diff
            print(
                f"""Heading: current={base_pose[2]:.3f}, desired={desired_heading:.3f},
                diff={heading_diff:.3f}, frac={frac:.3f}, target={target_heading:.3f}"""
            )

        # Create action to move towards target
        action = {
            "base_pose": np.array(
                [target_position[0], target_position[1], target_heading]
            ),
            "arm_pos": obs["arm_pos"].copy(),
            "arm_quat": obs["arm_quat"].copy(),
            "gripper_pos": obs["gripper_pos"].copy(),
        }

        return action

    def dot(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Dot product helper function from controller.py."""
        return a[0] * b[0] + a[1] * b[1]

    def intersect(
        self,
        d: Tuple[float, float],
        f: Tuple[float, float],
        r: float,
        use_t1: bool = False,
    ) -> Optional[float]:
        """Line-circle intersection from controller.py."""
        # https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm/1084899%231084899
        a = self.dot(d, d)
        b = 2 * self.dot(f, d)
        c = self.dot(f, f) - r * r
        discriminant = (b * b) - (4 * a * c)
        if discriminant >= 0:
            if use_t1:
                t1 = (-b - math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t1 <= 1:
                    return t1
            else:
                t2 = (-b + math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t2 <= 1:
                    return t2
        return None

    def detect_objects_from_ground_truth(self, obs: Dict[str, Any]) -> List[np.ndarray]:
        """Detect objects using ground truth from MuJoCo simulation and find the one
        with smallest x value."""
        detected_objects: List[np.ndarray] = []

        # Get all object positions from MuJoCo environment dynamically
        cubes: List[Tuple[np.ndarray, str]] = []
        for key in obs.keys():
            if (
                key.endswith("_pos")
                and not key.startswith("arm_")
                and not key.startswith("base_")
                and not key.startswith("left_")
                and not key.startswith("right_")
                and not key.startswith("gripper_")
            ):
                cube_pos = obs[key].copy()
                cube_name = key.replace("_pos", "")
                cubes.append((cube_pos, cube_name))
                print(f"Detected {cube_name} at position: {cube_pos}")

        if cubes:
            # Sort cubes by x position and select the one with smallest x value
            if self.custom_grasp or self.low_grasp:
                cubes.sort(key=lambda x: x[0][1])  # sort by y
                target_cube_pos, target_cube_name = cubes[0]
            else:
                cubes.sort(
                    key=lambda x: x[0][0]
                )  # Sort by x coordinate (first element of position)
                target_cube_pos, target_cube_name = cubes[0]
            detected_objects.append(target_cube_pos)
            print(
                f"Selected {target_cube_name} with smallest x value: {target_cube_pos[0]:.3f}"  # pylint: disable=line-too-long
            )

        return detected_objects

    def distance(
        self,
        pt1: Union[Tuple[float, float], List[float], np.ndarray],
        pt2: Union[Tuple[float, float], List[float], np.ndarray],
    ) -> float:
        """Calculate distance between two points from controller.py."""
        return math.sqrt(
            (float(pt2[0]) - float(pt1[0])) ** 2 + (float(pt2[1]) - float(pt1[1])) ** 2
        )

    def restrict_heading_range(self, h: float) -> float:
        """Normalize heading to [-π, π] range from controller.py."""
        return (h + math.pi) % (2 * math.pi) - math.pi

    def get_end_effector_offset(self) -> float:
        """Calculate end-effector offset based on task and gripper state from
        controller.py."""
        return 0.55

    def build_base_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build base command using exact logic from controller.py."""
        assert command["primitive_name"] in {
            "move",
            "pick",
            "place",
            "toss",
            "shelf",
            "drawer",
        }

        # Base movement only
        if command["primitive_name"] == "move":
            return {
                "waypoints": command["waypoints"],
                "target_ee_pos": None,
                "position_tolerance": 0.1,
            }
        target_ee_pos = command["waypoints"][-1]
        end_effector_offset = self.get_end_effector_offset()
        new_waypoint = None
        reversed_waypoints = command["waypoints"][::-1]

        for idx in range(1, len(reversed_waypoints)):
            start = reversed_waypoints[idx - 1]
            end = reversed_waypoints[idx]
            print("start, end", [start, end])
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - target_ee_pos[0], start[1] - target_ee_pos[1])
            t2 = self.intersect(d, f, end_effector_offset)
            if t2 is not None:
                new_waypoint = (start[0] + t2 * d[0], start[1] + t2 * d[1])
                break

        if new_waypoint is not None and not self.cupboard_mode and not self.low_grasp:
            # Discard all waypoints that are too close to target_ee_pos
            waypoints = reversed_waypoints[idx:][::-1] + [new_waypoint]
        else:
            # Base is too close to target end effector position and needs to back up
            print(
                """Warning: Base needs to deviate from commanded
                path to reach target position, watch out for potential collisions"""
            )
            curr_position = command["waypoints"][0]
            signed_dist = (
                self.distance(curr_position, target_ee_pos) - end_effector_offset
            )
            dx = target_ee_pos[0] - curr_position[0]
            dy = target_ee_pos[1] - curr_position[1]
            target_heading = self.restrict_heading_range(math.atan2(dy, dx))
            if self.cupboard_mode and command["primitive_name"] == "place":
                target_position = (
                    target_ee_pos[0] - end_effector_offset,
                    target_ee_pos[1],
                )
                middle_position = (target_ee_pos[0] - 1.0, target_ee_pos[1])
                waypoints = [curr_position, middle_position, target_position]
            elif (self.custom_grasp or self.low_grasp) and self.cupboard_mode:
                if command["primitive_name"] == "pick":
                    target_position = (
                        target_ee_pos[0] - end_effector_offset,
                        target_ee_pos[1],
                    )
                    middle_position = (target_ee_pos[0] - 1.0, target_ee_pos[1])
                    waypoints = [curr_position, middle_position, target_position]
                elif command["primitive_name"] == "place":
                    target_position = (
                        target_ee_pos[0] - end_effector_offset,
                        target_ee_pos[1],
                    )
                    middle_position = (target_ee_pos[0] - 1.0, target_ee_pos[1])
                    middle_position_1 = (curr_position[0] - 0.5, curr_position[1])
                    waypoints = [
                        curr_position,
                        middle_position_1,
                        middle_position,
                        target_position,
                    ]
            else:
                target_position = (
                    curr_position[0] + signed_dist * math.cos(target_heading),
                    curr_position[1] + signed_dist * math.sin(target_heading),
                )
                waypoints = [curr_position, target_position]

        return {
            "waypoints": waypoints,  # pylint: disable=possibly-used-before-assignment
            "target_ee_pos": target_ee_pos,
        }
