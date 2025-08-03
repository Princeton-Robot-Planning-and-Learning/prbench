"""TidyBot 3D environment wrapper for PRBench."""

import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from prbench.envs.tidybot_rewards import create_reward_calculator

# Import TidyBot components from local files
from .mujoco_env import MujocoEnv


class TidyBot3DEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
    """TidyBot 3D environment with mobile manipulation tasks.

    (Policy-agnostic, random actions only)
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        scene_type: str = "ground",  
        num_objects: int = 3,
        render_mode: str | None = None,
        custom_grasp: bool = False,
        render_images: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.scene_type = scene_type
        self.num_objects = num_objects
        self.render_mode = render_mode
        self.custom_grasp = custom_grasp
        # Allow show_viewer/show_images/render_images to be set via kwargs or directly
        self.show_viewer = kwargs.pop("show_viewer", False)
        self.show_images = kwargs.pop("show_images", False)
        self.render_images = kwargs.pop("render_images", render_images)
        # Store any other kwargs for future use
        self._extra_kwargs = kwargs

        # Initialize TidyBot environment
        self._tidybot_env = self._create_tidybot_env()

        # Remove policy initialization

        self._reward_calculator = create_reward_calculator(
            self.scene_type, self.num_objects
        )

        # Define observation and action spaces
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

        # Add metadata for documentation (remove policy references)
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

    def _create_tidybot_env(self) -> "MujocoEnv":
        """Create the underlying TidyBot MuJoCo environment."""
        # Set model path to local models directory
        model_base_path = os.path.join(
            os.path.dirname(__file__), "models", "stanford_tidybot"
        )

        # Remove table, cabinet, and cupboard from scene_type options
        # Only keep 'ground' (if present)
        if self.scene_type == "ground":
            model_file = "scene.xml"

        # Construct absolute path to model file
        absolute_model_path = os.path.join(model_base_path, model_file)

        # --- Dynamic object insertion logic ---
        needs_dynamic_objects = self.scene_type in ["ground"]
        if needs_dynamic_objects:
            tree = ET.parse(absolute_model_path)
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
                    # Only support ground scene
                    x = round(np.random.uniform(0.4, 0.8), 3)
                    y = round(np.random.uniform(-0.3, 0.3), 3)
                    z = 0.02
                    pos = f"{x} {y} {z}"
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
                dynamic_model_path = os.path.join(
                    model_base_path, dynamic_model_filename
                )
                tree.write(dynamic_model_path)
            else:
                dynamic_model_path = absolute_model_path
        else:
            dynamic_model_path = absolute_model_path

        kwargs = {
            "render_images": self.render_images,
            "show_viewer": self.show_viewer,
            "show_images": self.show_images,
            "mjcf_path": dynamic_model_path,
        }
        # Allow any extra kwargs to override
        kwargs.update(self._extra_kwargs)

        return MujocoEnv(**kwargs)  # type: ignore

    # Remove _create_policy method

    def _create_observation_space(self) -> spaces.Box:
        """Create observation space based on TidyBot's observation
        structure."""
        # Get example observation to determine dimensions
        self._tidybot_env.reset()  # type: ignore[no-untyped-call]
        example_obs = self._tidybot_env.get_obs()  # type: ignore[no-untyped-call]

        # Calculate total observation dimension
        obs_dim = 0
        for _, value in example_obs.items():
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
        obs_vector: list[float] = []
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
        # Capture seed from kwargs if provided
        seed = kwargs.get("seed", None)

        super().reset(*args, **kwargs)

        # Pass the seed to the TidyBot environment
        self._tidybot_env.reset(seed=seed)

        # Remove policy reset
        self._reward_calculator = create_reward_calculator(
            self.scene_type, self.num_objects
        )
        obs = self._tidybot_env.get_obs()  # type: ignore[no-untyped-call]
        vec_obs = self._vectorize_observation(obs)
        return vec_obs, {}

    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], float, bool, bool, dict]:
        """Execute action and return next observation."""
        action_dict = self._dict_to_action(action)
        self._tidybot_env.step(action_dict)  # type: ignore[no-untyped-call]

        # Get observation
        obs = self._tidybot_env.get_obs()  # type: ignore[no-untyped-call]
        vec_obs = self._vectorize_observation(obs)

        # Calculate reward and termination
        reward = self._calculate_reward(obs)
        terminated = self._is_terminated(obs)
        truncated = False

        return vec_obs, reward, terminated, truncated, {}

    # Remove step_with_policy method

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
        """Create environment description (policy-agnostic)."""
        return f"""A 3D mobile manipulation environment using the TidyBot platform.
        
The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: {self.scene_type} with {self.num_objects} objects.

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
        return """TidyBot++: An Open-Source Holonomic Mobile Manipulator
- for Robot Learning
- Jimmy Wu, William Chong, Robert Holmberg, Aaditya Prasad, Yihuai Gao, 
  Oussama Khatib, Shuran Song, Szymon Rusinkiewicz, Jeannette Bohg
- Conference on Robot Learning (CoRL), 2024

https://github.com/tidybot2/tidybot2
"""

    @classmethod
    def get_available_environments(cls) -> List[str]:
        """Get list of available TidyBot environment IDs (policy-agnostic)."""
        scene_configs = [
            ("ground", [3, 5, 7]),
        ]
        env_ids = []
        for scene_type, object_counts in scene_configs:
            for num_objects in object_counts:
                env_ids.append(f"prbench/TidyBot3D-{scene_type}-o{num_objects}-v0")
        return env_ids
