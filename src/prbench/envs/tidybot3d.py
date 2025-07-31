"""TidyBot 3D environment wrapper for PRBench."""

import inspect
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

import tempfile
import xml.etree.ElementTree as ET
import random

# Import local constants
from . import constants
from .agent.mp_policy import MotionPlannerPolicy

# Import TidyBot components from local files
from .mujoco_env import MujocoEnv
from .policies import (
    MotionPlannerPolicyCustomGraspThreeWrapper,
    MotionPlannerPolicyCustomGraspWrapper,
    MotionPlannerPolicyMPCabinetTwoPhaseWrapper,
    MotionPlannerPolicyMPCabinetWrapper,
    MotionPlannerPolicyMPCupboardWrapper,
    MotionPlannerPolicyMPNCupboardWrapper,
    MotionPlannerPolicyMPThreeWrapper,
    MotionPlannerPolicyMPWrapper,
    MotionPlannerPolicyStackCupboardThreeWrapper,
    MotionPlannerPolicyStackCupboardWrapper,
    MotionPlannerPolicyStackDrawerThreeWrapper,
    MotionPlannerPolicyStackDrawerWrapper,
    MotionPlannerPolicyStackTableThreeWrapper,
    MotionPlannerPolicyStackTableWrapper,
    MotionPlannerPolicyStackThreeWrapper,
    MotionPlannerPolicyStackWrapper,
)


class TidyBot3DEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
    """TidyBot 3D environment with mobile manipulation tasks."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        scene_type: str = "table",  # "table", "drawer", "cupboard", "cabinet"
        num_objects: int = 3,
        policy_type: str = "mp",  # "motion_planning", "stack", "mp"
        render_mode: str | None = None,
        custom_grasp: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.scene_type = scene_type
        self.num_objects = num_objects
        self.policy_type = policy_type
        self.render_mode = render_mode
        self.custom_grasp = custom_grasp
        # Allow show_viewer/show_images to be set via kwargs
        self.show_viewer = kwargs.pop("show_viewer", False)
        self.show_images = kwargs.pop("show_images", False)
        # Store any other kwargs for future use
        self._extra_kwargs = kwargs

        # Initialize TidyBot environment
        self._tidybot_env = self._create_tidybot_env()

        # Initialize policy
        self._policy = self._create_policy()

        # Initialize reward calculator
        from prbench.envs.tidybot_rewards import create_reward_calculator

        self._reward_calculator = create_reward_calculator(
            self.scene_type, self.num_objects, self.policy_type
        )

        # Define observation and action spaces
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

        # Add metadata for documentation
        self.metadata.update(
            {
                "description": self._create_env_markdown_description(),
                "observation_space_description": self._create_obs_markdown_description(),
                "action_space_description": self._create_action_markdown_description(),
                "reward_description": self._create_reward_markdown_description(),
                "references": self._create_references_markdown_description(),
                "render_fps": 20,
            }
        )

    def _create_tidybot_env(self) -> MujocoEnv:
        """Create the underlying TidyBot MuJoCo environment."""
        # Set model path to local models directory
        model_base_path = os.path.join(
            os.path.dirname(__file__), "models", "stanford_tidybot"
        )

        # Determine the correct model file based on scene type
        if self.scene_type == "table":
            model_file = "blocks_table_scene.xml"
        elif self.scene_type == "drawer":
            model_file = "drawer_scene.xml"
        elif self.scene_type == "cupboard":
            if self.custom_grasp:
                model_file = "cupboard_scene_objects_inside.xml"
            else:
                model_file = "cupboard_scene.xml"
        elif self.scene_type == "cabinet":
            model_file = "cabinet.xml"
        else:
            model_file = "scene.xml"

        # Construct absolute path to model file
        absolute_model_path = os.path.join(model_base_path, model_file)

        # --- Dynamic object insertion logic ---
        needs_dynamic_objects = self.scene_type in ["ground", "table"]
        if needs_dynamic_objects:
            tree = ET.parse(absolute_model_path)
            root = tree.getroot()
            worldbody = root.find("worldbody")
            # Remove all existing cube bodies
            for body in list(worldbody):
                if body.tag == "body" and body.attrib.get("name", "").startswith("cube"):
                    worldbody.remove(body)
            # Insert new cubes
            for i in range(self.num_objects):
                name = f"cube{i+1}"
                if self.scene_type == "ground":
                    # Randomize positions for ground
                    x = round(random.uniform(0.4, 0.8), 3)
                    y = round(random.uniform(-0.3, 0.3), 3)
                    z = 0.02
                    pos = f"{x} {y} {z}"
                elif self.scene_type == "table":
                    x = 0.5 - 0.05 * i
                    y = 0.1 * ((i % 3) - 1)
                    z = 0.44
                    pos = f"{x} {y} {z}"
                elif self.scene_type == "drawer":
                    x = 0.7 - 0.05 * i
                    y = -0.1 + 0.05 * (i % 3)
                    z = 0.12
                    pos = f"{x} {y} {z}"
                elif self.scene_type == "cupboard":
                    x = 1.0
                    y = -0.4 + 0.1 * (i % 3)
                    z = 0.33
                    pos = f"{x} {y} {z}"
                elif self.scene_type == "cabinet":
                    x = 0.75
                    y = -0.1 - 0.1 * (i % 3)
                    z = 0.12
                    pos = f"{x} {y} {z}"
                else:
                    pos = "0.6 0 0.02"
                body = ET.Element("body", name=name, pos=pos)
                ET.SubElement(body, "freejoint")
                ET.SubElement(body, "geom", type="box", size="0.02 0.02 0.02", rgba=".5 .7 .5 1", mass="0.1")
                worldbody.append(body)
            # Write to a file in the models directory
            dynamic_model_filename = f"auto_{self.scene_type}_{self.num_objects}_objs.xml"
            dynamic_model_path = os.path.join(model_base_path, dynamic_model_filename)
            tree.write(dynamic_model_path)
        else:
            dynamic_model_path = absolute_model_path

        kwargs = {
            "render_images": True,
            "show_viewer": self.show_viewer,
            "show_images": self.show_images,
            "custom_grasp": self.custom_grasp,
            "mjcf_path": dynamic_model_path,
        }
        # Allow any extra kwargs to override
        kwargs.update(self._extra_kwargs)

        if self.scene_type == "table":
            kwargs["table_scene"] = True
        elif self.scene_type == "drawer":
            kwargs["drawer_scene"] = True
        elif self.scene_type == "cupboard":
            kwargs["cupboard_scene"] = True
        elif self.scene_type == "cabinet":
            kwargs["cabinet_scene"] = True

        return MujocoEnv(**kwargs)

    def _create_policy(self):
        """Create appropriate policy based on policy_type."""
        if self.policy_type == "stack":
            if self.scene_type == "table":
                return MotionPlannerPolicyStackTableWrapper()
            elif self.scene_type == "drawer":
                return MotionPlannerPolicyStackDrawerWrapper()
            elif self.scene_type == "cupboard":
                return MotionPlannerPolicyStackCupboardWrapper()
            else:
                return MotionPlannerPolicyStackWrapper()
        elif self.policy_type == "stack_three":
            if self.scene_type == "table":
                return MotionPlannerPolicyStackTableThreeWrapper()
            elif self.scene_type == "drawer":
                return MotionPlannerPolicyStackDrawerThreeWrapper()
            elif self.scene_type == "cupboard":
                return MotionPlannerPolicyStackCupboardThreeWrapper()
            else:
                return MotionPlannerPolicyStackThreeWrapper()
        elif self.policy_type == "mp":
            if self.scene_type == "cupboard":
                return MotionPlannerPolicyMPCupboardWrapper(
                    custom_grasp=self.custom_grasp
                )
            elif self.scene_type == "cabinet":
                return MotionPlannerPolicyMPCabinetWrapper(
                    custom_grasp=self.custom_grasp
                )
            else:
                return MotionPlannerPolicyMPWrapper(custom_grasp=self.custom_grasp)
        elif self.policy_type == "mp_three":
            return MotionPlannerPolicyMPThreeWrapper(custom_grasp=self.custom_grasp)
        elif self.policy_type == "mp_cabinet_two_phase":
            return MotionPlannerPolicyMPCabinetTwoPhaseWrapper(
                custom_grasp=self.custom_grasp
            )
        elif self.policy_type == "custom_grasp":
            return MotionPlannerPolicyCustomGraspWrapper()
        elif self.policy_type == "custom_grasp_three":
            return MotionPlannerPolicyCustomGraspThreeWrapper()
        elif self.policy_type == "mp_n_cupboard":
            # Default target locations for N objects
            target_locations = [
                np.array([0.9, 0.08, 0.38]),  # Center position
                # np.array([0.9, -0.08, 0.38]),  # Left position
                # np.array([0.83, 0, 0.38]),  # Right position
            ]
            return MotionPlannerPolicyMPNCupboardWrapper(
                target_locations=target_locations[: self.num_objects],
                custom_grasp=self.custom_grasp,
            )
        else:
            # Default to motion planning
            return MotionPlannerPolicyMPWrapper(custom_grasp=self.custom_grasp)

    def _create_observation_space(self) -> spaces.Box:
        """Create observation space based on TidyBot's observation
        structure."""
        # Get example observation to determine dimensions
        self._tidybot_env.reset()
        example_obs = self._tidybot_env.get_obs()

        # Calculate total observation dimension
        obs_dim = 0
        for key, value in example_obs.items():
            if isinstance(value, np.ndarray):
                obs_dim += value.size
            else:
                obs_dim += 1

        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _create_action_space(self) -> spaces.Box:
        """Create action space for TidyBot's control interface."""
        # TidyBot actions: base_pose (3), arm_pos (3), arm_quat (4), gripper_pos (1)
        return spaces.Box(
            low=np.array(
                [-1.0, -1.0, -np.pi, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
            ),
            high=np.array([1.0, 1.0, np.pi, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

    def _vectorize_observation(self, obs: Dict[str, Any]) -> NDArray[np.float32]:
        """Convert TidyBot observation dict to vector."""
        obs_vector = []
        for key in sorted(obs.keys()):  # Sort for consistency
            value = obs[key]
            if isinstance(value, np.ndarray):
                obs_vector.extend(value.flatten())
            else:
                obs_vector.append(float(value))
        return np.array(obs_vector, dtype=np.float32)

    def _dict_to_action(self, action_vector: NDArray[np.float32]) -> Dict[str, Any]:
        """Convert action vector to TidyBot action dict."""
        return {
            "base_pose": action_vector[:3],
            "arm_pos": action_vector[3:6],
            "arm_quat": action_vector[6:10],
            "gripper_pos": action_vector[10:11],
        }

    def reset(self, *args, **kwargs) -> Tuple[NDArray[np.float32], dict]:
        """Reset the environment."""
        super().reset(*args, **kwargs)
        self._tidybot_env.reset()
        self._policy.reset()
        # Reset reward calculator
        from prbench.envs.tidybot_rewards import create_reward_calculator

        self._reward_calculator = create_reward_calculator(
            self.scene_type, self.num_objects, self.policy_type
        )
        obs = self._tidybot_env.get_obs()
        vec_obs = self._vectorize_observation(obs)
        return vec_obs, {}

    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], float, bool, bool, dict]:
        """Execute action and return next observation."""
        action_dict = self._dict_to_action(action)
        self._tidybot_env.step(action_dict)

        # Get observation
        obs = self._tidybot_env.get_obs()
        vec_obs = self._vectorize_observation(obs)

        # Calculate reward and termination
        reward = self._calculate_reward(obs)
        terminated = self._is_terminated(obs)
        truncated = False

        return vec_obs, reward, terminated, truncated, {}

    def step_with_policy(
        self, obs: Optional[Dict[str, Any]] = None
    ) -> Tuple[NDArray[np.float32], float, bool, bool, dict]:
        """Execute step using the internal policy."""
        if obs is None:
            obs = self._tidybot_env.get_obs()

        # Get action from policy
        action = self._policy.step(obs)

        

        if self._policy.episode_ended:
            # Policy signaled episode end
            return self._vectorize_observation(obs), 0.0, True, False, {}

        if action is None:
            # Policy returned no action
            return self._vectorize_observation(obs), -0.01, False, False, {}

        if action == "reset_env":
            # Policy signaled reset
            return self.reset()[0], 0.0, False, True, {}

        # Execute action
        self._tidybot_env.step(action)

        # Get new observation
        new_obs = self._tidybot_env.get_obs()
        vec_obs = self._vectorize_observation(new_obs)

        # Calculate reward and termination
        reward = self._calculate_reward(new_obs)
        terminated = self._is_terminated(new_obs)
        truncated = False

        return vec_obs, reward, terminated, truncated, {}

    def _calculate_reward(self, obs: Dict[str, Any]) -> float:
        """Calculate reward based on task completion."""
        return self._reward_calculator.calculate_reward(obs)

    def _is_terminated(self, obs: Dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        return self._reward_calculator.is_terminated(obs)
        
        

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # Return camera images if available
            obs = self._tidybot_env.get_obs()
            for key, value in obs.items():
                if key.endswith("_image") and isinstance(value, np.ndarray):
                    return value
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None

    def close(self):
        """Close the environment."""
        self._tidybot_env.close()

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        return f"""A 3D mobile manipulation environment using the TidyBot platform.
        
The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: {self.scene_type} with {self.num_objects} objects.
Policy type: {self.policy_type}

Available scenes:
- table: Object stacking and manipulation on a table
- drawer: Opening/closing drawers and placing objects inside
- cupboard: Opening cupboards and organizing objects
- cabinet: Cabinet manipulation tasks

Available policy types:
- stack: Object stacking policies
- stack_three: Three-object stacking policies
- mp: Motion planning policies
- mp_three: Three-sequential motion planning
- mp_cabinet_two_phase: Two-phase cabinet manipulation
- custom_grasp: Custom grasping policies
- custom_grasp_three: Three-sequential custom grasping
- mp_n_cupboard: N-object cupboard manipulation

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)
"""

    def _create_obs_markdown_description(self) -> str:
        """Create observation space description."""
        return """Observation includes:
- Robot state: base pose, arm position/orientation, gripper state
- Object states: positions and orientations of all objects
- Camera images: RGB images from base and wrist cameras
- Scene-specific features: handle positions for cabinets/drawers
"""

    def _create_action_markdown_description(self) -> str:
        """Create action space description."""
        return """Actions control:
- base_pose: [x, y, theta] - Mobile base position and orientation
- arm_pos: [x, y, z] - End effector position in world coordinates
- arm_quat: [x, y, z, w] - End effector orientation as quaternion
- gripper_pos: [pos] - Gripper open/close position (0=closed, 1=open)
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        return """Reward function depends on the specific task:
- Object stacking: Reward for successfully stacking objects
- Drawer/cabinet tasks: Reward for opening/closing and placing objects
- General manipulation: Reward for successful pick-and-place operations

Currently returns a small negative reward (-0.01) per timestep to encourage exploration.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        return """TidyBot++: An Open-Source Holonomic Mobile Manipulator for Robot Learning
Jimmy Wu, William Chong, Robert Holmberg, Aaditya Prasad, Yihuai Gao, Oussama Khatib, Shuran Song, Szymon Rusinkiewicz, Jeannette Bohg
Conference on Robot Learning (CoRL), 2024

https://github.com/tidybot2/tidybot2
"""

    @classmethod
    def get_available_environments(cls) -> List[str]:
        """Get list of available TidyBot environment IDs."""
        scene_configs = [
            ("table", [3, 5, 7]),
            ("drawer", [2, 4, 6]),
            ("cupboard", [3, 5, 8]),
            ("cabinet", [2, 4, 6]),
        ]

        policy_types = [
            "stack",
            "stack_three",
            "mp",
            "mp_three",
            "mp_cabinet_two_phase",
            "custom_grasp",
            "custom_grasp_three",
            "mp_n_cupboard",
        ]

        env_ids = []
        for scene_type, object_counts in scene_configs:
            for num_objects in object_counts:
                for policy_type in policy_types:
                    # Skip incompatible combinations
                    if (
                        (
                            scene_type == "cabinet"
                            and policy_type in ["stack", "stack_three"]
                        )
                        or (
                            scene_type == "table"
                            and policy_type in ["mp_cabinet_two_phase"]
                        )
                        or (
                            scene_type == "drawer"
                            and policy_type in ["mp_cabinet_two_phase", "mp_n_cupboard"]
                        )
                        or (
                            scene_type == "cupboard"
                            and policy_type in ["mp_cabinet_two_phase"]
                        )
                    ):
                        continue

                    env_ids.append(
                        f"prbench/TidyBot3D-{scene_type}-o{num_objects}-{policy_type}-v0"
                    )

        return env_ids
