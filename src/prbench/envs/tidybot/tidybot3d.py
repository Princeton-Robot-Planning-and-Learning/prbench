"""TidyBot 3D environment wrapper for PRBench."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from prbench.envs.tidybot.tidybot_mujoco_env import MujocoEnv
from prbench.envs.tidybot.tidybot_rewards import create_reward_calculator


class TidyBot3DEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
    """TidyBot 3D environment with mobile manipulation tasks."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        scene_type: str = "ground",
        num_objects: int = 3,
        render_mode: str | None = None,
        custom_grasp: bool = False,
        render_images: bool = True,
        seed: int | None = None,
        show_viewer: bool = False,
        show_images: bool = False,
    ) -> None:
        super().__init__()

        self.scene_type = scene_type
        self.num_objects = num_objects
        self.render_mode = render_mode
        self.custom_grasp = custom_grasp
        self.show_viewer = show_viewer
        self.show_images = show_images
        self.render_images = render_images
        self._render_camera_name: str | None = None

        # Initialize random number generator
        if seed is not None:
            self.np_random, _ = gymnasium.utils.seeding.np_random(seed)
        else:
            self.np_random = np.random.default_rng()

        # Initialize TidyBot environment
        self._tidybot_env = self._create_tidybot_env()

        self._reward_calculator = create_reward_calculator(
            self.scene_type, self.num_objects
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
        model_base_path = Path(__file__).parent / "models" / "stanford_tidybot"
        if self.scene_type == "cupboard":
            model_file = "cupboard_scene.xml"
        elif self.scene_type == "table":
            model_file = "table_scene.xml"
        else:
            model_file = "ground_scene.xml"
        # Construct absolute path to model file
        absolute_model_path = model_base_path / model_file

        # --- Dynamic object insertion logic ---
        needs_dynamic_objects = self.scene_type in ["ground", "table"]
        if needs_dynamic_objects:
            tree = ET.parse(str(absolute_model_path))
            root = tree.getroot()
            worldbody = root.find("worldbody")
            if worldbody is not None:
                # Remove all existing cube bodies
                for body in list(worldbody):
                    if body.tag == "body" and body.attrib.get("name", "").startswith(
                        "cube"
                    ):
                        worldbody.remove(body)
                # Insert new cubes
                for i in range(self.num_objects):
                    name = f"cube{i+1}"
                    pos = f"{0} {0} {0}"
                    body = ET.Element("body", name=name, pos=pos)
                    ET.SubElement(body, "freejoint")
                    ET.SubElement(
                        body,
                        "geom",
                        type="box",
                        size="0.02 0.02 0.02",
                        rgba=".5 .7 .5 1",
                        mass="0.1",
                    )
                    worldbody.append(body)
                # Write to a file in the models directory
                dynamic_model_filename = (
                    f"auto_{self.scene_type}_{self.num_objects}_objs.xml"
                )
                dynamic_model_path = model_base_path / dynamic_model_filename
                tree.write(str(dynamic_model_path))
            else:
                dynamic_model_path = absolute_model_path
        else:
            dynamic_model_path = absolute_model_path

        kwargs = {
            "render_images": self.render_images,
            "show_viewer": self.show_viewer,
            "show_images": self.show_images,
            "mjcf_path": str(dynamic_model_path),
        }

        if self.scene_type == "cupboard":
            kwargs["cupboard_scene"] = True
        elif self.scene_type == "table":
            kwargs["table_scene"] = True

        return MujocoEnv(**kwargs)  # type: ignore

    def _create_observation_space(self) -> spaces.Box:
        """Create observation space based on TidyBot's observation structure."""
        # Get example observation to determine dimensions
        self._tidybot_env.reset()
        example_obs = self._tidybot_env.get_obs()

        # Calculate total observation dimension (all values are ndarrays)
        obs_dim = sum(value.size for value in example_obs.values())

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

    def _vectorize_observation(self, obs: dict[str, Any]) -> NDArray[np.float32]:
        """Convert TidyBot observation dict to vector."""
        obs_vector: list[float] = []
        for key in sorted(obs.keys()):  # Sort for consistency
            value = obs[key]
            obs_vector.extend(value.flatten())
        return np.array(obs_vector, dtype=np.float32)

    def _dict_to_action(self, action_vector: NDArray[np.float32]) -> dict[str, Any]:
        """Convert action vector to TidyBot action dict."""
        return {
            "base_pose": action_vector[:3],
            "arm_pos": action_vector[3:6],
            "arm_quat": action_vector[6:10],
            "gripper_pos": action_vector[10:11],
        }

    def reset(self, *args, **kwargs) -> tuple[NDArray[np.float32], dict]:
        """Reset the environment."""
        # Capture seed from kwargs if provided
        seed = kwargs.get("seed", None)

        super().reset(*args, **kwargs)

        # Pass the seed to the TidyBot environment
        self._tidybot_env.reset(seed=seed)

        obs = self._tidybot_env.get_obs()
        vec_obs = self._vectorize_observation(obs)
        return vec_obs, {}

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
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

    def _calculate_reward(self, obs: dict[str, Any]) -> float:
        """Calculate reward based on task completion."""
        return self._reward_calculator.calculate_reward(obs)

    def _is_terminated(self, obs: dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        return self._reward_calculator.is_terminated(obs)

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            obs = self._tidybot_env.get_obs()
            # If a specific camera is requested, use it.
            if self._render_camera_name:
                key = f"{self._render_camera_name}_image"
                if key in obs:
                    return obs[key]
            # Otherwise, fall back to the first available image.
            for key, value in obs.items():
                if key.endswith("_image"):
                    return value
            raise RuntimeError("No camera image available in observation.")
        return None

    def close(self) -> None:
        """Close the environment."""
        self._tidybot_env.close()

    def set_render_camera(self, camera_name: str | None) -> None:
        """Set the camera to use for rendering."""
        self._render_camera_name = camera_name

    def _create_env_markdown_description(self) -> str:
        """Create environment description (policy-agnostic)."""
        scene_description = ""
        if self.scene_type == "ground":
            scene_description = """ In the 'ground' scene, objects are placed randomly on a flat ground plane."""  # pylint: disable=line-too-long

        return f"""A 3D mobile manipulation environment using the TidyBot platform.
        
The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: {self.scene_type} with {self.num_objects} objects.{scene_description}

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
        if self.scene_type == "ground":
            return """The primary reward is for successfully placing objects at their target locations. # pylint: disable=line-too-long
- A reward of +1.0 is given for each object placed within a 5cm tolerance of its target.
- A smaller positive reward is given for objects within a 10cm tolerance to guide the robot.
- A small negative reward (-0.01) is applied at each timestep to encourage efficiency.
The episode terminates when all objects are placed at their respective targets.
"""
        return """Reward function depends on the specific task:
- Object stacking: Reward for successfully stacking objects
- Drawer/cabinet tasks: Reward for opening/closing and placing objects
- General manipulation: Reward for successful pick-and-place operations

Currently returns a small negative reward (-0.01) per timestep to encourage exploration.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        return """TidyBot++: An Open-Source Holonomic Mobile Manipulator
for Robot Learning
- Jimmy Wu, William Chong, Robert Holmberg, Aaditya Prasad, Yihuai Gao, 
  Oussama Khatib, Shuran Song, Szymon Rusinkiewicz, Jeannette Bohg
- Conference on Robot Learning (CoRL), 2024

https://github.com/tidybot2/tidybot2
"""

    @classmethod
    def get_available_environments(cls) -> list[str]:
        """Get list of available TidyBot environment IDs (policy-agnostic)."""
        scene_configs = [
            ("ground", [3, 5, 7]),
        ]
        env_ids = []
        for scene_type, object_counts in scene_configs:
            for num_objects in object_counts:
                env_ids.append(f"prbench/TidyBot3D-{scene_type}-o{num_objects}-v0")
        return env_ids
