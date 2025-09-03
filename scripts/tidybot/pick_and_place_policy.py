#!/usr/bin/env python3
"""Pick-and-place policy runner for TidyBot3D.

This script creates a TidyBot3D environment and executes a pick-and-place
episode using a self-contained motion planner policy based on mp_policy.py.
"""

import math
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio.v2 as iio
import numpy as np

import prbench

DEBUG_DIR = Path(__file__).parent / "debug"


class PickState(Enum):
    """States of the pick subroutine."""
    APPROACH = "approach"
    LOWER = "lower"
    GRASP = "grasp"
    LIFT = "lift"


class PlaceState(Enum):
    """States of the place subroutine."""
    APPROACH = "approach"
    LOWER = "lower"
    RELEASE = "release"
    HOME = "home"


class MotionPlannerPolicy:
    """High-level mobile manipulation policy for pick-and-place."""

    # Base following parameters
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

    # Grasping parameters
    GRASP_SUCCESS_THRESHOLD = 0.7
    GRASP_PROGRESS_THRESHOLD = 0.3
    GRASP_TIMEOUT_S = 3.0
    PLACE_SUCCESS_THRESHOLD = 0.2

    def __init__(self) -> None:
        self.state: str = "idle"
        self.current_command: Optional[Dict[str, Any]] = None
        self.base_waypoints: List[List[float]] = []
        self.target_ee_pos: Optional[List[float]] = None
        self.grasp_state: Optional[Union[PickState, PlaceState]] = None
        self.object_location: Optional[np.ndarray] = None
        self.target_location: Optional[np.ndarray] = None
        self.enabled: bool = True
        self.episode_ended: bool = False

    def reset(self) -> None:
        """Reset internal state and prepare for a new episode."""
        self.state = "idle"
        self.current_command = None
        self.base_waypoints = []
        self.target_ee_pos = None
        self.episode_ended = False
        self.grasp_state = None
        self.object_location = None
        self.target_location = None
        if hasattr(self, "grasp_start_time"):
            delattr(self, "grasp_start_time")
        if hasattr(self, "initial_gripper_pos"):
            delattr(self, "initial_gripper_pos")
        self.enabled = True
        print("Motion planner reset - starting episode automatically")

    def step(self, obs: Dict[str, Any], robot_env: Any) -> Optional[Dict[str, Any]]:
        """Advance the policy by one step and return an action."""
        if self.episode_ended or not self.enabled:
            return None
        return self._step(obs, robot_env)

    def _step(self, obs: Dict[str, Any], robot_env: Any) -> Optional[Dict[str, Any]]:
        """Internal step implementation operating the state machine."""
        # Extract robot state from robot environment directly
        base_pose = robot_env.qpos_base.copy()
        arm_pos = robot_env.get_end_effector_position() if hasattr(robot_env, 'get_end_effector_position') else np.array([0.0, 0.0, 0.0])
        arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Default quaternion
        gripper_pos = np.array([0.0])  # Default gripper position

        if self.state == "idle":
            detected_objects = self.detect_objects_from_ground_truth(robot_env)
            if detected_objects:
                self.object_location = detected_objects[0]
                if self.target_location is None:
                    self.target_location = np.array([
                        self.object_location[0] + self.PLACEMENT_X_OFFSET,
                        self.object_location[1] + self.PLACEMENT_Y_OFFSET,
                        self.object_location[2] + self.PLACEMENT_Z_OFFSET,
                    ])

                pick_command = {
                    "primitive_name": "pick",
                    "waypoints": [base_pose[:2].tolist(), self.object_location[:2].tolist()],
                    "object_3d_pos": self.object_location.copy(),
                }

                print(f"Object detected at: {self.object_location}")
                print(f"Target placement location: {self.target_location}")

                base_command = self.build_base_command(pick_command)
                if base_command:
                    self.current_command = pick_command
                    self.base_waypoints = base_command["waypoints"]
                    self.target_ee_pos = base_command["target_ee_pos"]
                    self.state = "moving"
                    print("Starting base movement to object")
                else:
                    print("Failed to build base command")
            else:
                return None

        elif self.state == "moving":
            assert self.current_command is not None
            action = self.execute_base_movement(base_pose, arm_pos, arm_quat, gripper_pos)
            if action is None:
                print("Base movement complete!")
                if self.target_ee_pos is not None:
                    distance_to_target = self.distance(base_pose[:2], self.target_ee_pos)
                    end_effector_offset = self.get_end_effector_offset()
                    diff = abs(end_effector_offset - distance_to_target)
                    
                    if self.current_command["primitive_name"] == "pick":
                        base_tolerance = self.GRASP_BASE_TOLERANCE
                    else:
                        base_tolerance = self.PLACE_BASE_TOLERANCE
                        
                    if diff < base_tolerance:
                        self.state = "manipulating"
                        print("Base reached target, starting arm manipulation")
                    else:
                        print(f"Too far from target end effector position ({100 * diff:.1f} cm)")
                        self.state = "idle"
                else:
                    self.state = "idle"
            return action

        elif self.state == "manipulating":
            assert self.current_command is not None
            if self.current_command["primitive_name"] == "pick":
                return self._execute_pick(base_pose, arm_pos, arm_quat, gripper_pos)
            elif self.current_command["primitive_name"] == "place":
                return self._execute_place(base_pose, arm_pos, arm_quat, gripper_pos)

        return {
            "base_pose": base_pose.copy(),
            "arm_pos": arm_pos.copy(),
            "arm_quat": arm_quat.copy(),
            "gripper_pos": gripper_pos.copy(),
        }

    def _execute_pick(self, base_pose: np.ndarray, arm_pos: np.ndarray, 
                     arm_quat: np.ndarray, gripper_pos: np.ndarray) -> Dict[str, Any]:
        """Execute the pick manipulation sequence."""
        if self.grasp_state is None:
            self.grasp_state = PickState.APPROACH

        object_3d_pos = self.current_command["object_3d_pos"]
        global_diff = np.array([
            object_3d_pos[0] - base_pose[0],
            object_3d_pos[1] - base_pose[1],
            object_3d_pos[2] + self.PICK_APPROACH_HEIGHT_OFFSET - self.ROBOT_BASE_HEIGHT,
        ])

        base_angle = base_pose[2]
        cos_angle = math.cos(-base_angle)
        sin_angle = math.sin(-base_angle)

        object_relative_pos = np.array([
            cos_angle * global_diff[0] - sin_angle * global_diff[1],
            sin_angle * global_diff[0] + cos_angle * global_diff[1],
            global_diff[2],
        ])

        print(f"Grasp state: {self.grasp_state}")

        if self.grasp_state == PickState.APPROACH:
            target_arm_pos = object_relative_pos
            target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
            target_gripper_pos = np.array([0.0])

            print("Step 1: Positioning arm above object with open gripper")
            if np.allclose(arm_pos, target_arm_pos, atol=0.03):
                self.grasp_state = PickState.LOWER
                print("Arm positioned above object, moving to lower approach")

        elif self.grasp_state == PickState.LOWER:
            target_arm_pos = object_relative_pos.copy()
            target_arm_pos[2] -= self.PICK_LOWER_DIST
            target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
            target_gripper_pos = np.array([0.0])

            print(f"Step 2: Lowering gripper... target: {target_arm_pos[2]:.3f}")
            if np.allclose(arm_pos, target_arm_pos, atol=0.02):
                self.grasp_state = PickState.GRASP
                print("Gripper lowered to grasping position, closing gripper")

        elif self.grasp_state == PickState.GRASP:
            target_arm_pos = object_relative_pos.copy()
            target_arm_pos[2] -= self.PICK_LOWER_DIST
            target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
            target_gripper_pos = np.array([1.0])

            print(f"Step 3: Closing gripper... current position: {gripper_pos[0]:.3f}")

            if not hasattr(self, "grasp_start_time"):
                self.grasp_start_time = time.time()  # type: ignore[attr-defined]
                self.initial_gripper_pos = gripper_pos[0]  # type: ignore[attr-defined]

            gripper_closed_enough = gripper_pos[0] > self.GRASP_SUCCESS_THRESHOLD
            gripper_progress = (gripper_pos[0] - self.initial_gripper_pos) > self.GRASP_PROGRESS_THRESHOLD
            grasp_timeout = (time.time() - self.grasp_start_time) > self.GRASP_TIMEOUT_S

            if gripper_closed_enough or gripper_progress or grasp_timeout:
                if gripper_closed_enough or gripper_progress:
                    print(f"Grasp successful! Gripper pos: {gripper_pos[0]:.3f}")
                else:
                    print(f"Grasp timeout, proceeding: {gripper_pos[0]:.3f}")

                self.grasp_state = PickState.LIFT
                delattr(self, "grasp_start_time")
                delattr(self, "initial_gripper_pos")
                print("Moving to lift phase!")

        elif self.grasp_state == PickState.LIFT:
            lifted_pos = object_relative_pos.copy()
            lifted_pos[2] += self.PICK_LIFT_DIST - self.PICK_LOWER_DIST
            target_arm_pos = lifted_pos
            target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
            target_gripper_pos = np.array([1.0])

            print(f"Step 4: Lifting object... target height: {target_arm_pos[2]:.3f}")
            if np.allclose(arm_pos, target_arm_pos, atol=0.05):
                print("Object lifted successfully! Moving to placement location.")
                
                assert self.target_location is not None
                place_command = {
                    "primitive_name": "place",
                    "waypoints": [base_pose[:2].tolist(), self.target_location[:2].tolist()],
                    "target_3d_pos": self.target_location.copy(),
                }

                base_command = self.build_base_command(place_command)
                if base_command:
                    self.current_command = place_command
                    self.base_waypoints = base_command["waypoints"]
                    self.target_ee_pos = base_command["target_ee_pos"]
                    self.state = "moving"
                    self.grasp_state = None
                    print("Starting base movement to placement location")
                else:
                    print("Failed to build place command")
                    self.episode_ended = True
                    self.state = "idle"

        else:
            target_arm_pos = arm_pos.copy()
            target_arm_quat = arm_quat.copy()
            target_gripper_pos = gripper_pos.copy()

        return {
            "base_pose": base_pose.copy(),
            "arm_pos": target_arm_pos,
            "arm_quat": target_arm_quat,
            "gripper_pos": target_gripper_pos,
        }

    def _execute_place(self, base_pose: np.ndarray, arm_pos: np.ndarray,
                      arm_quat: np.ndarray, gripper_pos: np.ndarray) -> Dict[str, Any]:
        """Execute the place manipulation sequence."""
        if self.grasp_state is None:
            self.grasp_state = PlaceState.APPROACH

        target_3d_pos = self.current_command["target_3d_pos"]
        
        global_diff = np.array([
            target_3d_pos[0] - base_pose[0],
            target_3d_pos[1] - base_pose[1],
            target_3d_pos[2] + self.PLACE_APPROACH_HEIGHT_OFFSET - self.ROBOT_BASE_HEIGHT,
        ])
        
        global_diff_lower = np.array([
            target_3d_pos[0] - base_pose[0],
            target_3d_pos[1] - base_pose[1],
            target_3d_pos[2] - self.ROBOT_BASE_HEIGHT,
        ])

        base_angle = base_pose[2]
        cos_angle = math.cos(-base_angle)
        sin_angle = math.sin(-base_angle)
        
        target_relative_pos = np.array([
            cos_angle * global_diff[0] - sin_angle * global_diff[1],
            sin_angle * global_diff[0] + cos_angle * global_diff[1],
            global_diff[2],
        ])
        
        target_relative_pos_lower = np.array([
            cos_angle * global_diff_lower[0] - sin_angle * global_diff_lower[1],
            sin_angle * global_diff_lower[0] + cos_angle * global_diff_lower[1],
            global_diff_lower[2],
        ])

        arm_home_pos = np.array([0.14322269, 0.0, 0.20784938])
        arm_home_quat = np.array([0.707, 0.707, 0, 0])

        print(f"Grasp state: {self.grasp_state}")

        if self.grasp_state == PlaceState.APPROACH:
            target_arm_pos = target_relative_pos
            target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
            target_gripper_pos = np.array([1.0])
            
            print("Step 1: Positioning arm above placement location")
            if np.allclose(arm_pos, target_arm_pos, atol=0.03):
                self.grasp_state = PlaceState.LOWER
                print("Arm above placement, lowering...")

        elif self.grasp_state == PlaceState.LOWER:
            target_arm_pos = target_relative_pos_lower
            target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
            target_gripper_pos = np.array([1.0])
            
            print("Step 2: Lowering arm to placement height")
            if np.allclose(arm_pos, target_arm_pos, atol=0.02):
                self.grasp_state = PlaceState.RELEASE
                print("Arm at placement height, opening gripper...")

        elif self.grasp_state == PlaceState.RELEASE:
            target_arm_pos = target_relative_pos_lower
            target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
            target_gripper_pos = np.array([0.0])
            
            print("Step 3: Opening gripper to release object")
            if gripper_pos[0] < self.PLACE_SUCCESS_THRESHOLD:
                self.grasp_state = PlaceState.HOME
                print("Object placed, moving to home position...")

        elif self.grasp_state == PlaceState.HOME:
            target_arm_pos = arm_home_pos
            target_arm_quat = arm_home_quat
            target_gripper_pos = np.array([0.0])
            
            print("Step 4: Moving arm to home position")
            if np.allclose(arm_pos, target_arm_pos, atol=0.03):
                print("Arm at home position. Task complete.")
                self.episode_ended = True
                self.state = "idle"

        else:
            target_arm_pos = arm_pos.copy()
            target_arm_quat = arm_quat.copy()
            target_gripper_pos = gripper_pos.copy()

        return {
            "base_pose": base_pose.copy(),
            "arm_pos": target_arm_pos,
            "arm_quat": target_arm_quat,
            "gripper_pos": target_gripper_pos,
        }

    def execute_base_movement(self, base_pose: np.ndarray, arm_pos: np.ndarray, 
                             arm_quat: np.ndarray, gripper_pos: np.ndarray) -> Optional[Dict[str, Any]]:
        """Execute base movement following waypoints."""
        if not self.base_waypoints:
            return None

        target_position = self.base_waypoints[-1]
        position_error = self.distance(base_pose[:2], target_position)
        
        if position_error < self.POSITION_TOLERANCE:
            return None

        target_heading = base_pose[2]
        if self.target_ee_pos is not None:
            dx = self.target_ee_pos[0] - base_pose[0]
            dy = self.target_ee_pos[1] - base_pose[1]
            desired_heading = math.atan2(dy, dx)
            heading_diff = self.restrict_heading_range(desired_heading - base_pose[2])
            target_heading += 0.5 * heading_diff

        return {
            "base_pose": np.array([target_position[0], target_position[1], target_heading]),
            "arm_pos": arm_pos.copy(),
            "arm_quat": arm_quat.copy(),
            "gripper_pos": gripper_pos.copy(),
        }

    def detect_objects_from_ground_truth(self, robot_env: Any) -> List[np.ndarray]:
        """Detect objects using ground truth from MuJoCo simulation."""
        detected_objects: List[np.ndarray] = []

        try:
            # Get object poses from robot environment using the new function
            object_poses = robot_env.get_object_poses()
            print(f"Object poses: {object_poses}")
            
            if object_poses:
                cubes: List[Tuple[np.ndarray, str]] = []
                for obj_name, pose_dict in object_poses.items():
                    position = pose_dict['position']
                    orientation = pose_dict['orientation']
                    
                    # Skip objects at origin (0,0,0) as they may be uninitialized
                    if np.allclose(position, [0.0, 0.0, 0.0], atol=1e-6):
                        print(f"Skipping {obj_name} at origin: {position}")
                        continue
                    
                    cubes.append((position, obj_name))
                    print(f"Detected {obj_name} at position: {position}, orientation: {orientation}")

                if cubes:
                    # Sort by x coordinate and select the leftmost one
                    cubes.sort(key=lambda x: x[0][0])
                    target_cube_pos, target_cube_name = cubes[0]
                    detected_objects.append(target_cube_pos)
                    print(f"Selected {target_cube_name} with x value: {target_cube_pos[0]:.3f}")
                else:
                    print("No valid objects found (all at origin)")
        except Exception as e:
            print(f"Error in object detection: {e}")

        return detected_objects

    def distance(self, pt1: Union[Tuple[float, float], List[float], np.ndarray],
                pt2: Union[Tuple[float, float], List[float], np.ndarray]) -> float:
        """Calculate distance between two points."""
        return math.sqrt((float(pt2[0]) - float(pt1[0])) ** 2 + 
                        (float(pt2[1]) - float(pt1[1])) ** 2)

    def restrict_heading_range(self, h: float) -> float:
        """Normalize heading to [-π, π] range."""
        return (h + math.pi) % (2 * math.pi) - math.pi

    def get_end_effector_offset(self) -> float:
        """Return constant end-effector offset."""
        return 0.55

    def build_base_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build base command for movement."""
        if command["primitive_name"] == "move":
            return {"waypoints": command["waypoints"], "target_ee_pos": None}
        
        target_ee_pos = command["waypoints"][-1]
        end_effector_offset = self.get_end_effector_offset()
        
        curr_position = command["waypoints"][0]
        dx = target_ee_pos[0] - curr_position[0]
        dy = target_ee_pos[1] - curr_position[1]
        target_heading = self.restrict_heading_range(math.atan2(dy, dx))
        
        dist_to_target = self.distance(curr_position, target_ee_pos)
        target_position = (
            curr_position[0] + (dist_to_target - end_effector_offset) * math.cos(target_heading),
            curr_position[1] + (dist_to_target - end_effector_offset) * math.sin(target_heading),
        )
        
        waypoints = [curr_position, target_position]
        return {"waypoints": waypoints, "target_ee_pos": target_ee_pos}


def dict_action_to_vector(action: Dict[str, Any]) -> np.ndarray:
    """Convert policy dict action to TidyBot3DEnv action vector."""
    base = np.asarray(action.get("base_pose", np.zeros(3)), dtype=float).reshape(3)
    arm_pos = np.asarray(action.get("arm_pos", np.zeros(3)), dtype=float).reshape(3)
    arm_quat = np.asarray(action.get("arm_quat", np.array([1.0, 0.0, 0.0, 0.0])), dtype=float).reshape(4)
    gripper = np.asarray(action.get("gripper_pos", np.zeros(1)), dtype=float).reshape(1)
    return np.concatenate([base, arm_pos, arm_quat, gripper]).astype(np.float32)


def create_hold_action(robot_env: Any) -> Dict[str, Any]:
    """Create an action that holds the current pose from robot environment."""
    try:
        arm_pos = robot_env.get_end_effector_position() if hasattr(robot_env, 'get_end_effector_position') else np.array([0.0, 0.0, 0.0])
    except:
        arm_pos = np.array([0.0, 0.0, 0.0])
    
    return {
        "base_pose": robot_env.qpos_base.copy(),
        "arm_pos": arm_pos,
        "arm_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "gripper_pos": np.array([0.0]),
    }


def run_episode() -> None:
    """Run a single pick-and-place episode with the motion planner policy."""
    prbench.register_all_environments()

    env_id = "prbench/TidyBot3D-ground-o3-v0"

    env = prbench.make(
        env_id,
        render_images=True,
        show_images=True,
        show_viewer=False,
        render_mode="rgb_array",
    )

    _, _ = env.reset()
    # pylint: disable=protected-access
    robot_env = env.env.env._tidybot_robot_env  # type: ignore[attr-defined]

    policy = MotionPlannerPolicy()
    policy.reset()

    max_steps = 2000
    imgs: List[np.ndarray] = []

    # Capture initial frame
    imgs.append(env.render())

    for step in range(max_steps):
        obs_dict = robot_env.get_obs()

        action_dict = policy.step(obs_dict, robot_env)
        if action_dict is None:
            action_dict = create_hold_action(robot_env)

        action_vec = dict_action_to_vector(action_dict)
        step_result = env.step(action_vec)

        # Capture frame every few steps to keep GIF manageable
        if step % 5 == 0:
            imgs.append(env.render())

        if len(step_result) >= 3:
            done = bool(step_result[2])
        else:
            done = False

        if step % 10 == 0:
            print(f"Step {step}: executing policy action")

        if done or policy.episode_ended:
            print("Episode completed.")
            # Capture final frame
            imgs.append(env.render())
            break

        time.sleep(0.01)

    env.close()  # type: ignore[attr-defined]

    # Save GIF
    DEBUG_DIR.mkdir(exist_ok=True)
    gif_filename = DEBUG_DIR / "pick_and_place_policy.gif"
    fps = env.metadata.get("render_fps", 10)
    print(f"Saving GIF with {len(imgs)} frames to {gif_filename}")
    iio.mimsave(gif_filename, imgs, fps=fps, loop=0)
    print(f"GIF saved successfully to {gif_filename}")


def main() -> None:
    run_episode()


if __name__ == "__main__":
    main()
