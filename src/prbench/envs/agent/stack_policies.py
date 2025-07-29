import math
import time
from enum import Enum

import numpy as np

# Import BaseAgent base class
from .base_agent import BaseAgent


class PickState(Enum):
    APPROACH = "approach"
    LOWER = "lower"
    GRASP = "grasp"
    LIFT = "lift"


class PlaceState(Enum):
    APPROACH = "approach"
    LOWER_TO_PLACE = "lower_to_place"
    RELEASE = "release"
    LIFT_AWAY = "lift_away"


# Motion Planner generated plan for stacking cubes
class MotionPlannerPolicyStack(BaseAgent):
    # Base following parameters (from BaseController)
    LOOKAHEAD_DISTANCE = 0.3  # 30 cm
    POSITION_TOLERANCE = 0.005  # 0.5 cm (reduced from 1.5 cm)
    HEADING_TOLERANCE = math.radians(2.1)  # 2.1 degrees
    GRASP_BASE_TOLERANCE = 0.002  # 0.2 cm for grasp
    PLACE_BASE_TOLERANCE = 0.005  # 0.5 cm for placement (example, adjust as needed)

    # Stacking parameters
    STACK_HEIGHT_OFFSET = 0.13  # 10cm above the target cube for stacking

    # Manipulation parameters
    ROBOT_BASE_HEIGHT = 0.48
    PICK_APPROACH_HEIGHT_OFFSET = 0.25
    PICK_LOWER_DIST = 0.07
    PICK_LIFT_DIST = 0.28  # Net lift is (PICK_LIFT_DIST - PICK_LOWER_DIST)
    PLACE_APPROACH_HEIGHT_OFFSET = 0.10

    # Grasping parameters
    GRASP_SUCCESS_THRESHOLD = 0.7
    GRASP_PROGRESS_THRESHOLD = 0.3
    GRASP_TIMEOUT_S = 3.0
    PLACE_SUCCESS_THRESHOLD = 0.05
    GRASP_FAILURE_THRESHOLD = (
        0.95  # Gripper position above which grasp is considered failed after lift
    )
    MAX_GRASP_RETRIES = 3

    def __init__(self):
        # Motion planning state - following controller.py pattern
        self.state = "idle"  # States: idle, moving, manipulating, grasping
        self.current_command = None
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.grasp_state = None  # Replaces grasp_step
        self.base_movement_retries = 0
        self.grasp_retries = 0

        # Base following parameters
        self.lookahead_position = None

        # Object and target locations (using ground truth from MuJoCo)
        self.source_cube_location = None  # Cube with smallest x value
        self.target_cube_location = None  # Cube with largest x value
        self.stack_location = None  # Position on top of target cube

        # Enable policy execution immediately (no web interface required)
        self.enabled = True
        self.episode_ended = False

        print(f"Motion planner stack policy initialized - ready to start automatically")

    def reset(self):
        # Reset motion planning state
        self.state = "idle"
        self.current_command = None
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.lookahead_position = None
        self.episode_ended = False
        self.grasp_state = None
        self.base_movement_retries = 0
        self.grasp_retries = 0

        # Reset cube locations
        self.source_cube_location = None
        self.target_cube_location = None
        self.stack_location = None

        # Clean up any grasp tracking variables
        if hasattr(self, "grasp_start_time"):
            delattr(self, "grasp_start_time")
        if hasattr(self, "initial_gripper_pos"):
            delattr(self, "initial_gripper_pos")

        # Enable policy execution immediately
        self.enabled = True

        print("Motion planner stack policy reset - starting episode automatically")

    def step(self, obs):
        # Return no action if episode has ended
        if self.episode_ended:
            return None

        # Return no action if robot is not enabled
        if not self.enabled:
            return None

        return self._step(obs)

    def _step(self, obs):
        # Extract current state
        base_pose = obs["base_pose"]
        arm_pos = obs["arm_pos"]
        arm_quat = obs["arm_quat"]
        gripper_pos = obs["gripper_pos"]

        # Debug: Print current base pose
        print(
            f"Current base pose: [{base_pose[0]:.3f}, {base_pose[1]:.3f}, {base_pose[2]:.3f}]"
        )

        # State machine following controller.py pattern
        if self.state == "idle":
            # Detect objects and plan new command
            cube_info = self.detect_cubes_for_stacking(obs)
            if cube_info:
                self.source_cube_location, self.target_cube_location = cube_info
                # Set stack location on top of target cube
                self.stack_location = np.array(
                    [
                        self.target_cube_location[0],  # Same X as target cube
                        self.target_cube_location[1],  # Same Y as target cube
                        self.target_cube_location[2]
                        + self.STACK_HEIGHT_OFFSET,  # Stack on top
                    ]
                )

                pick_command = {
                    "primitive_name": "pick",
                    "waypoints": [
                        base_pose[:2].tolist(),
                        self.source_cube_location[:2].tolist(),
                    ],
                    "object_3d_pos": self.source_cube_location.copy(),  # Store full 3D position
                }

                print(f"Source cube (smallest x) at: {self.source_cube_location}")
                print(f"Target cube (largest x) at: {self.target_cube_location}")
                print(f"Stack location: {self.stack_location}")
                print(
                    f"Creating pick command with waypoints: {pick_command['waypoints']}"
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
                    self.base_movement_retries = 0
                    self.grasp_retries = 0
                    print(
                        f"Starting base movement to source cube at {self.source_cube_location}"
                    )
                    print(f"Base waypoints: {self.base_waypoints}")
                    print(f"Target EE position: {self.target_ee_pos}")
                else:
                    print("Failed to build base command")
            else:
                # No objects found, do nothing
                return None

        elif self.state == "moving":
            # Execute base movement following waypoints (like BaseController)
            action = self.execute_base_movement(obs)
            if action is None:  # Base movement complete
                print("Base movement complete!")
                # Check if we're close enough for arm manipulation
                if self.target_ee_pos is not None:
                    distance_to_target = self.distance(
                        base_pose[:2], self.target_ee_pos
                    )
                    end_effector_offset = self.get_end_effector_offset(
                        self.current_command["primitive_name"]
                    )
                    diff = abs(end_effector_offset - distance_to_target)
                    print(
                        f"Distance to target EE: {distance_to_target:.3f}, EE offset: {end_effector_offset:.3f}, diff: {diff:.3f}"
                    )
                    if self.current_command["primitive_name"] == "pick":
                        base_tolerance = self.GRASP_BASE_TOLERANCE
                    else:
                        base_tolerance = self.PLACE_BASE_TOLERANCE
                    if diff < base_tolerance:
                        self.state = "manipulating"
                        print("Base reached target, starting arm manipulation")
                    else:
                        if self.base_movement_retries < 3:
                            self.base_movement_retries += 1
                            print(
                                f"Too far from target end effector position ({(100 * diff):.1f} cm). Retrying, attempt {self.base_movement_retries}/3."
                            )

                            # Rebuild base command from current pose
                            self.current_command["waypoints"][0] = base_pose[
                                :2
                            ].tolist()
                            base_command = self.build_base_command(self.current_command)

                            if base_command:
                                self.base_waypoints = base_command["waypoints"]
                                self.target_ee_pos = base_command["target_ee_pos"]
                                self.current_waypoint_idx = 1
                                self.lookahead_position = None
                                # Stay in 'moving' state to execute new path
                                print(
                                    f"Restarting base movement. New waypoints: {self.base_waypoints}"
                                )
                            else:
                                print("Failed to build base command for retry.")
                                self.state = "idle"
                        else:
                            print(
                                f"Too far from target end effector position ({(100 * diff):.1f} cm) after 3 retries."
                            )
                            self.state = "idle"
                else:
                    self.state = "idle"
            else:
                # Print target base pose from action
                if action and "base_pose" in action:
                    target_pose = action["base_pose"]
                    print(
                        f"Target base pose: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]"
                    )
            return action

        elif self.state == "manipulating":
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
                        - self.ROBOT_BASE_HEIGHT,  # Object height + higher offset - base height (was 0.15, now 0.25)
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
                    f"Base angle: {base_pose[2]:.3f} rad ({math.degrees(base_pose[2]):.1f} deg)"
                )
                print(
                    f"Object global pos: {self.current_command.get('object_3d_pos', 'N/A')}"
                )

                print(f"Current arm pos: {arm_pos}")
                print(
                    f"Position error: {np.linalg.norm(arm_pos - object_relative_pos):.4f}m"
                )
                print(f"Grasp state: {self.grasp_state}")

                if self.grasp_state == PickState.APPROACH:
                    # Step 1: Position arm well above object with open gripper (safe approach)
                    target_arm_pos = object_relative_pos
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([0.0])  # Gripper open

                    print(f"Step 1: Positioning arm above object with open gripper")
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
                        f"Step 2: Lowering gripper for precise approach... target: {target_arm_pos[2]:.3f}, current: {arm_pos[2]:.3f}"
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
                        f"Step 3: Closing gripper... current position: {gripper_pos[0]:.3f}"
                    )

                    # Initialize grasp attempt tracking
                    if not hasattr(self, "grasp_start_time"):
                        self.grasp_start_time = time.time()
                        self.initial_gripper_pos = gripper_pos[0]
                        print(
                            f"Started grasp attempt, initial gripper pos: {self.initial_gripper_pos:.3f}"
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
                                f"Grasp successful! Gripper pos: {gripper_pos[0]:.3f}, progress: {gripper_pos[0] - self.initial_gripper_pos:.3f}"
                            )
                        else:
                            print(
                                f"Grasp timeout reached, proceeding with current grip: {gripper_pos[0]:.3f}"
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
                        f"Step 4: Lifting object... target height: {target_arm_pos[2]:.3f}, current: {arm_pos[2]:.3f}"
                    )
                    if np.allclose(arm_pos, target_arm_pos, atol=0.05):  # 5cm tolerance
                        # Check for grasp failure after lifting
                        if gripper_pos[0] > self.GRASP_FAILURE_THRESHOLD:
                            if self.grasp_retries < self.MAX_GRASP_RETRIES:
                                self.grasp_retries += 1
                                print(
                                    f"Grasp failed after lift (gripper pos: {gripper_pos[0]:.3f}). Retrying grasp, attempt {self.grasp_retries}/{self.MAX_GRASP_RETRIES}."
                                )

                                # Update cube positions after grasp failure since they may have moved
                                cube_info = self.detect_cubes_for_stacking(obs)
                                if cube_info:
                                    (
                                        self.source_cube_location,
                                        self.target_cube_location,
                                    ) = cube_info
                                    # Update stack location with new target cube position
                                    self.stack_location = np.array(
                                        [
                                            self.target_cube_location[
                                                0
                                            ],  # Same X as target cube
                                            self.target_cube_location[
                                                1
                                            ],  # Same Y as target cube
                                            self.target_cube_location[2]
                                            + self.STACK_HEIGHT_OFFSET,  # Stack on top
                                        ]
                                    )
                                    # Update the current command with new object position
                                    self.current_command["object_3d_pos"] = (
                                        self.source_cube_location.copy()
                                    )
                                    print(
                                        f"Updated cube positions after grasp failure:"
                                    )
                                    print(f"  Source cube: {self.source_cube_location}")
                                    print(f"  Target cube: {self.target_cube_location}")
                                    print(f"  Stack location: {self.stack_location}")
                                else:
                                    print(
                                        "Warning: Could not detect cubes after grasp failure"
                                    )

                                self.grasp_state = PickState.APPROACH
                            else:
                                print(
                                    f"Grasp failed after {self.MAX_GRASP_RETRIES} retries. Aborting."
                                )
                                self.episode_ended = True
                                self.state = "idle"
                        else:
                            print(
                                "Object lifted successfully! Now moving to stack location on target cube."
                            )
                            self.grasp_retries = 0  # Reset for next pick
                            # Create place command for stacking
                            place_command = {
                                "primitive_name": "place",
                                "waypoints": [
                                    base_pose[:2].tolist(),
                                    self.stack_location[:2].tolist(),
                                ],
                                "target_3d_pos": self.stack_location.copy(),
                            }

                            base_command = self.build_base_command(place_command)
                            if base_command:
                                self.current_command = place_command
                                self.base_waypoints = base_command["waypoints"]
                                self.target_ee_pos = base_command["target_ee_pos"]
                                self.current_waypoint_idx = 1
                                self.lookahead_position = None
                                self.state = "moving"
                                self.base_movement_retries = 0
                                self.grasp_state = None  # Reset for next manipulation
                                print(
                                    f"Starting base movement to stack location at {self.stack_location}"
                                )
                            else:
                                print("Failed to build place command")
                                self.episode_ended = True
                                self.state = "idle"

                # Create action from targets
                action = {
                    "base_pose": base_pose.copy(),
                    "arm_pos": target_arm_pos,
                    "arm_quat": target_arm_quat,
                    "gripper_pos": target_gripper_pos,
                }
                return action

            elif self.current_command["primitive_name"] == "place":
                if self.grasp_state is None:
                    self.grasp_state = PlaceState.APPROACH

                # Define default targets to hold current pose
                target_arm_pos = arm_pos.copy()
                target_arm_quat = arm_quat.copy()
                target_gripper_pos = gripper_pos.copy()

                # Recalculate precise stack location based on current target cube position
                # This accounts for any base positioning errors and ensures accurate placement
                cube_info = self.detect_cubes_for_stacking(obs)
                if cube_info:
                    _, current_target_cube_location = cube_info
                    # Update stack location with current target cube position
                    precise_stack_location = np.array(
                        [
                            current_target_cube_location[
                                0
                            ],  # Same X as current target cube position
                            current_target_cube_location[
                                1
                            ],  # Same Y as current target cube position
                            current_target_cube_location[2]
                            + self.STACK_HEIGHT_OFFSET,  # Stack on top
                        ]
                    )
                    target_3d_pos = precise_stack_location
                    print(f"Updated precise stack location: {precise_stack_location}")
                else:
                    # Fallback to original target if detection fails
                    target_3d_pos = self.current_command["target_3d_pos"]
                    print("Using fallback stack location")

                # Calculate positions for two-stage placement
                # Stage 1: Safe approach position (20cm above target cube)
                safe_approach_height = 0.20  # 20cm above target cube
                global_diff_safe = np.array(
                    [
                        target_3d_pos[0] - base_pose[0],
                        target_3d_pos[1] - base_pose[1],
                        target_3d_pos[2]
                        + safe_approach_height
                        - self.ROBOT_BASE_HEIGHT,  # Target height + 20cm - base height
                    ]
                )

                # Stage 2: Final placement position (10cm above target cube - already in target_3d_pos)
                global_diff_place = np.array(
                    [
                        target_3d_pos[0] - base_pose[0],
                        target_3d_pos[1] - base_pose[1],
                        target_3d_pos[2]
                        + self.PLACE_APPROACH_HEIGHT_OFFSET
                        - self.ROBOT_BASE_HEIGHT,  # Target height + offset - base height
                    ]
                )

                # Transform to base's local coordinate frame (account for base rotation)
                base_angle = base_pose[2]
                cos_angle = math.cos(-base_angle)  # Negative for inverse rotation
                sin_angle = math.sin(-base_angle)

                safe_approach_pos = np.array(
                    [
                        cos_angle * global_diff_safe[0]
                        - sin_angle * global_diff_safe[1],
                        sin_angle * global_diff_safe[0]
                        + cos_angle * global_diff_safe[1],
                        global_diff_safe[2],  # Z component unchanged
                    ]
                )

                final_place_pos = np.array(
                    [
                        cos_angle * global_diff_place[0]
                        - sin_angle * global_diff_place[1],
                        sin_angle * global_diff_place[0]
                        + cos_angle * global_diff_place[1],
                        global_diff_place[2],  # Z component unchanged
                    ]
                )

                print(f"Stacking: safe_approach_pos = {safe_approach_pos}")
                print(f"Stacking: final_place_pos = {final_place_pos}")
                print(f"Target EE pos: {self.target_ee_pos}, Base pose: {base_pose}")
                print(f"Grasp state: {self.grasp_state}")

                if self.grasp_state == PlaceState.APPROACH:
                    # Step 1: Position arm at safe approach height (20cm above target cube)
                    target_arm_pos = safe_approach_pos
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([1.0])  # Gripper closed

                    print(
                        f"Step 1: Moving to safe approach position (20cm above target cube)"
                    )
                    print(
                        f"Target arm pos: {target_arm_pos}, Current arm pos: {arm_pos}"
                    )
                    print(
                        f"Position error: {np.linalg.norm(arm_pos - target_arm_pos):.4f}m"
                    )
                    if np.allclose(
                        arm_pos[:2], target_arm_pos[:2], atol=0.02
                    ):  # 2cm tolerance for approach
                        self.grasp_state = PlaceState.LOWER_TO_PLACE
                        print(
                            "Reached safe approach position, now lowering to placement height"
                        )

                elif self.grasp_state == PlaceState.LOWER_TO_PLACE:
                    # Step 2: Lower to final placement position (10cm above target cube)
                    target_arm_pos = final_place_pos
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([1.0])  # Gripper closed

                    print(
                        f"Step 2: Lowering to final placement position (10cm above target cube)"
                    )
                    print(
                        f"Target arm pos: {target_arm_pos}, Current arm pos: {arm_pos}"
                    )
                    print(
                        f"Position error: {np.linalg.norm(arm_pos - target_arm_pos):.4f}m"
                    )
                    if np.allclose(
                        arm_pos, target_arm_pos, atol=0.01
                    ):  # 1cm tolerance for precise placement
                        self.grasp_state = PlaceState.RELEASE
                        print(
                            "Arm positioned at final placement height, opening gripper"
                        )

                elif self.grasp_state == PlaceState.RELEASE:
                    # Step 3: Open gripper to place object on top of target cube
                    target_arm_pos = (
                        final_place_pos  # Maintain arm position at placement height
                    )
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([0.0])  # Gripper open

                    print(
                        f"Step 3: Opening gripper to stack cube... current position: {gripper_pos[0]:.3f}"
                    )
                    if gripper_pos[0] < self.PLACE_SUCCESS_THRESHOLD:
                        print("Cube stacked successfully! Now lifting arm away.")
                        self.grasp_state = PlaceState.LIFT_AWAY

                elif self.grasp_state == PlaceState.LIFT_AWAY:
                    # Step 4: Lift arm away from stacked cubes with open gripper
                    lifted_away_pos = final_place_pos.copy()
                    lifted_away_pos[
                        2
                    ] += 0.15  # Lift arm 15cm above the placement position
                    target_arm_pos = lifted_away_pos
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([0.0])  # Gripper open

                    print(
                        f"Step 4: Lifting arm away from stacked cubes... target height: {target_arm_pos[2]:.3f}, current: {arm_pos[2]:.3f}"
                    )
                    print(
                        f"Position error: {np.linalg.norm(arm_pos - target_arm_pos):.4f}m"
                    )
                    if np.allclose(
                        arm_pos[:3], target_arm_pos[:3], atol=0.03
                    ):  # 3cm tolerance (reduced from 5cm)
                        print("Arm lifted away successfully! Task complete.")
                        self.episode_ended = True  # End the episode
                        self.state = "idle"

                # Create action from targets
                action = {
                    "base_pose": base_pose.copy(),
                    "arm_pos": target_arm_pos,
                    "arm_quat": target_arm_quat,
                    "gripper_pos": target_gripper_pos,
                }
                return action

        # Default: hold current pose
        action = {
            "base_pose": base_pose.copy(),
            "arm_pos": arm_pos.copy(),
            "arm_quat": arm_quat.copy(),
            "gripper_pos": gripper_pos.copy(),
        }
        print(f"Default action - holding current pose")
        return action

    def execute_base_movement(self, obs):
        """Execute base movement following waypoints like BaseController."""
        base_pose = obs["base_pose"]

        # Check if we've reached the final waypoint
        if self.current_waypoint_idx >= len(self.base_waypoints):
            print("All waypoints completed")
            return None  # Movement complete

        print(
            f"Current waypoint index: {self.current_waypoint_idx}/{len(self.base_waypoints)}"
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
                f"Position error to final target: {position_error:.3f} (tolerance: {self.POSITION_TOLERANCE})"
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
                f"Target EE: {self.target_ee_pos}, dx={dx:.3f}, dy={dy:.3f}, desired_heading={desired_heading:.3f}"
            )

            frac = 1
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
                f"Heading: current={base_pose[2]:.3f}, desired={desired_heading:.3f}, diff={heading_diff:.3f}, frac={frac:.3f}, target={target_heading:.3f}"
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

    def dot(self, a, b):
        """Dot product helper function from controller.py."""
        return a[0] * b[0] + a[1] * b[1]

    def intersect(self, d, f, r, use_t1=False):
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

    def detect_cubes_for_stacking(self, obs):
        """Detect cubes and return the source cube (smallest x) and target cube
        (largest x)"""
        # Get all three cube positions from MuJoCo environment
        cubes = []
        for i in range(1, 4):
            cube_key = f"cube{i}_pos"
            if cube_key in obs:
                cube_pos = obs[cube_key].copy()
                cubes.append((cube_pos, i))
                print(f"Detected cube {i} at position: {cube_pos}")
            else:
                print(f"Warning: {cube_key} not found in observation")

        if len(cubes) < 2:
            print("Warning: Need at least 2 cubes for stacking")
            return None

        # Sort cubes by x position
        cubes.sort(key=lambda x: x[0][0])  # Sort by x coordinate

        # Source cube: smallest x value
        source_cube_pos, source_cube_id = cubes[0]
        # Target cube: largest x value
        target_cube_pos, target_cube_id = cubes[-1]

        print(
            f"Source cube (smallest x): cube {source_cube_id} at x={source_cube_pos[0]:.3f}"
        )
        print(
            f"Target cube (largest x): cube {target_cube_id} at x={target_cube_pos[0]:.3f}"
        )

        return source_cube_pos, target_cube_pos

    def distance(self, pt1, pt2):
        """Calculate distance between two points from controller.py."""
        return math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    def restrict_heading_range(self, h):
        """Normalize heading to [-π, π] range from controller.py."""
        return (h + math.pi) % (2 * math.pi) - math.pi

    def get_end_effector_offset(self, primitive_name):
        """Calculate end-effector offset based on task and gripper state from
        controller.py."""
        # Simplified version - assume gripper starts open
        gripper_open = True
        if gripper_open:
            return 0.55
        return {"toss": 1.30, "shelf": 0.75, "drawer": 0.80}.get(primitive_name, 0.55)

    def build_base_command(self, command):
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

        # Modify waypoints so that the end effector is placed at the target end effector position
        target_ee_pos = command["waypoints"][-1]
        end_effector_offset = self.get_end_effector_offset(command["primitive_name"])
        new_waypoint = None  # Find new_waypoint such that distance(new_waypoint, target_ee_pos) == end_effector_offset
        reversed_waypoints = command["waypoints"][::-1]

        for idx in range(1, len(reversed_waypoints)):
            start = reversed_waypoints[idx - 1]
            end = reversed_waypoints[idx]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - target_ee_pos[0], start[1] - target_ee_pos[1])
            t2 = self.intersect(d, f, end_effector_offset)
            if t2 is not None:
                new_waypoint = (start[0] + t2 * d[0], start[1] + t2 * d[1])
                break

        if new_waypoint is not None:
            # Discard all waypoints that are too close to target_ee_pos
            waypoints = reversed_waypoints[idx:][::-1] + [new_waypoint]
        else:
            # Base is too close to target end effector position and needs to back up
            print(
                "Warning: Base needs to deviate from commanded path to reach target position, watch out for potential collisions"
            )
            curr_position = command["waypoints"][0]
            signed_dist = (
                self.distance(curr_position, target_ee_pos) - end_effector_offset
            )
            dx = target_ee_pos[0] - curr_position[0]
            dy = target_ee_pos[1] - curr_position[1]
            target_heading = self.restrict_heading_range(math.atan2(dy, dx))
            target_position = (
                curr_position[0] + signed_dist * math.cos(target_heading),
                curr_position[1] + signed_dist * math.sin(target_heading),
            )
            waypoints = [curr_position, target_position]

        return {"waypoints": waypoints, "target_ee_pos": target_ee_pos}


# Table stacking policy - inherits from MotionPlannerPolicyStack but adjusts placement height
class MotionPlannerPolicyStackTable(MotionPlannerPolicyStack):
    def __init__(self):
        super().__init__()
        # Adjust stacking parameters for table scene
        # The table height is 0.4m, and cubes are placed at 0.44m (table + 0.04m)
        # For stacking on table, we need to account for the table height
        self.STACK_HEIGHT_OFFSET = (
            0.04  # 4cm above the target cube for stacking on table
        )
        self.PLACE_APPROACH_HEIGHT_OFFSET = (
            0.20  # 14cm above target for safer approach on table
        )

    def get_end_effector_offset(self, primitive_name):
        """Calculate end-effector offset for table environment - smaller offset for larger reachable space"""
        # Simplified version - assume gripper starts open
        gripper_open = True
        if gripper_open:
            return 0.45  # Reduced from 0.55 to 0.45 for larger reachable space on table
        return {"toss": 1.20, "shelf": 0.65, "drawer": 0.70}.get(primitive_name, 0.45)


# Drawer stacking policy - inherits from MotionPlannerPolicyStack but adjusts placement height for the drawer environment
class MotionPlannerPolicyStackDrawer(MotionPlannerPolicyStack):
    """Stacking policy for the drawer environment.

    Adjusts stacking height and approach for cubes on the ground or in
    cubbies.
    """

    def __init__(self):
        super().__init__()
        # For the drawer scene, cubes are on the ground (z ~ 0.02), so stack height is just above a cube
        self.STACK_HEIGHT_OFFSET = (
            0.04  # 4cm above the target cube for stacking in cubby/ground
        )
        self.PLACE_APPROACH_HEIGHT_OFFSET = (
            0.15  # 15cm above target for safe approach in drawer scene
        )

    def get_end_effector_offset(self, primitive_name):
        """Calculate end-effector offset for drawer environment - smaller offset for reachable space"""
        gripper_open = True
        if gripper_open:
            return 0.45  # Reduced from 0.55 to 0.45 for drawer scene
        return {"toss": 1.10, "shelf": 0.60, "drawer": 0.45}.get(primitive_name, 0.45)


# Cupboard stacking policy - inherits from MotionPlannerPolicyStack but adjusts placement height for the cupboard environment
class MotionPlannerPolicyStackCupboard(MotionPlannerPolicyStack):
    """Stacking policy for the cupboard environment.

    Adjusts stacking height and approach for cubes on the ground in
    front of a cupboard.
    """

    def __init__(self):
        super().__init__()
        # For the cupboard scene, cubes are on the ground (z ~ 0.05), so stack height is just above a cube
