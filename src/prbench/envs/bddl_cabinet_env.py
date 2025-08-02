"""BDDL Cabinet Environment for TidyBot based on example_1_parsed.json."""

import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from prbench.envs.tidybot_rewards import create_reward_calculator
from prbench.envs.mujoco_env import MujocoEnv


class BDDLCabinetRewardCalculator:
    """Reward calculator for the BDDL cabinet task."""
    
    def __init__(self):
        self.task_completed = False
        self.bowl_in_cabinet = False
        
    def calculate_reward(self, obs: Dict[str, Any]) -> float:
        """Calculate reward based on bowl placement in cabinet."""
        reward = -0.01  # Small negative reward per timestep
        
        # Check if bowl (object_1) is in the cabinet top region
        if "object_1_pos" in obs:
            bowl_pos = obs["object_1_pos"]
            
            # Define the cabinet top region bounds (approximate)
            # Based on the wooden_cabinet.xml, the top_region site is around pos="0.00328 0.01128 0.18563"
            cabinet_x_range = (-0.03, 0.04)  # Approximate cabinet width
            cabinet_y_range = (-0.06, 0.08)  # Approximate cabinet depth  
            cabinet_z_range = (0.15, 0.22)   # Top drawer height range
            
            if (cabinet_x_range[0] <= bowl_pos[0] <= cabinet_x_range[1] and
                cabinet_y_range[0] <= bowl_pos[1] <= cabinet_y_range[1] and
                cabinet_z_range[0] <= bowl_pos[2] <= cabinet_z_range[1]):
                
                if not self.bowl_in_cabinet:
                    reward += 10.0  # Large reward for completing the task
                    self.bowl_in_cabinet = True
                    self.task_completed = True
                else:
                    reward += 1.0  # Smaller reward for maintaining position
        
        return reward
    
    def is_terminated(self, obs: Dict[str, Any]) -> bool:
        """Check if the task is completed."""
        return self.task_completed


class BDDLCabinetEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
    """BDDL Cabinet manipulation environment with TidyBot.
    
    Task: Put the white bowl in the top drawer of the wooden cabinet.
    Based on example_1_parsed.json BDDL specification.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: str | None = None,
        show_viewer: bool = False,
        show_images: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.show_viewer = show_viewer
        self.show_images = show_images
        self._extra_kwargs = kwargs

        # Create the scene XML file
        self.scene_path = self._create_scene_xml()
        
        # Initialize TidyBot environment
        self._tidybot_env = self._create_tidybot_env()

        # Initialize reward calculator
        self._reward_calculator = BDDLCabinetRewardCalculator()

        # Define observation and action spaces
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

        # Add metadata
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

    def _create_scene_xml(self) -> str:
        """Create the MuJoCo scene XML file based on BDDL specification."""
        # Get paths to assets
        assets_dir = os.path.join(os.path.dirname(__file__), "models", "libero_assets")
        cabinet_path = os.path.join(assets_dir, "articulated_objects", "wooden_cabinet")
        bowl_path = os.path.join(assets_dir, "stable_scanned_objects", "white_bowl")

        # Make assets_dir relative to the stanford_tidybot directory
        relative_assets_dir = "../libero_assets"
        scene_xml = f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="bddl_cabinet_scene">
    <include file="tidybot.xml"/>
    <compiler angle="radian" meshdir="{relative_assets_dir}" />
    <size njmax="500" nconmax="100" />
    
    <statistic center="0.25 0 0.6" extent="1.0" meansize="0.05"/>
    
    <visual>
        <quality offsamples="0"/>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-20"/>
    </visual>
    
    <asset>
        <!-- Skybox and ground -->
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
          markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
        
        <!-- Table -->
        <texture name="tex-table" file="textures/wood.png" type="2d"/>
        <material name="table_mat" reflectance="0.3" texrepeat="1 1" texture="tex-table"/>
       
       <!-- Cabinet assets -->
       <texture file="articulated_objects/wooden_cabinet/dark_fine_wood.png" name="tex-wooden_cabinet" type="2d"/>
       <material name="wooden_cabinet_top" reflectance="0.5" texrepeat="1 1" texture="tex-wooden_cabinet" texuniform="false"/>      
       <mesh file="articulated_objects/wooden_cabinet/wooden_cabinet_top/visual/wooden_cabinet_top_vis.msh" name="wooden_cabinet_top_vis" scale="0.05 0.05 0.05"/>
       <texture file="articulated_objects/wooden_cabinet/dark_fine_wood.png" name="tex-wooden_cabinet_base" type="2d"/>
       <material name="wooden_cabinet_base" reflectance="0.5" texrepeat="1 1" texture="tex-wooden_cabinet_base" texuniform="false"/>
       <mesh file="articulated_objects/wooden_cabinet/wooden_cabinet_base/visual/wooden_cabinet_base_vis.msh" name="wooden_cabinet_base_vis" scale="0.05 0.05 0.05"/>
       <texture file="articulated_objects/wooden_cabinet/metal.png" name="tex-wooden_cabinet_handle" type="2d"/>
       <material name="wooden_cabinet_top_handle" reflectance="0.5" texrepeat="1 1" texture="tex-wooden_cabinet_handle" texuniform="false"/>
       <mesh file="articulated_objects/wooden_cabinet/wooden_cabinet_top_handle/visual/wooden_cabinet_top_handle_vis.msh" name="wooden_cabinet_top_handle_vis" scale="0.05 0.05 0.05"/>
       
       <!-- Bowl assets -->
       <texture file="stable_scanned_objects/white_bowl/texture.png" name="tex-bowl" type="2d" />
       <material name="bowl_mat" reflectance="0.5" texrepeat="1 1" texture="tex-bowl" texuniform="false" />
       <mesh file="stable_scanned_objects/white_bowl/visual/model_vis.msh" name="bowl_vis" scale="0.5 0.5 0.5" />
   </asset>

    <worldbody>
        <!-- Lighting and ground -->
        <light pos="0 0 1.5" directional="true"/>
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
       
       <!-- Table (based on BDDL regions) -->
       <body name="table" pos="0.2 0.0 0.4">
           <geom name="table_top" type="box" size="0.4 0.3 0.02" rgba="0.8 0.6 0.4 1" material="table_mat"/>
           <geom name="table_leg1" type="box" pos="0.35 0.25 -0.2" size="0.02 0.02 0.2" rgba="0.8 0.6 0.4 1"/>
           <geom name="table_leg2" type="box" pos="0.35 -0.25 -0.2" size="0.02 0.02 0.2" rgba="0.8 0.6 0.4 1"/>
           <geom name="table_leg3" type="box" pos="-0.35 0.25 -0.2" size="0.02 0.02 0.2" rgba="0.8 0.6 0.4 1"/>
           <geom name="table_leg4" type="box" pos="-0.35 -0.25 -0.2" size="0.02 0.02 0.2" rgba="0.8 0.6 0.4 1"/>
       </body>
       
       <!-- Wooden Cabinet (positioned based on BDDL wooden_cabinet_init_region) -->
       <body name="wooden_cabinet_1" pos="0.4 0.0 0.0">
           <body name="base" pos="0 0 0.0">
               <geom solimp="0.998 0.998 0.001" solref="0.001 1" density="100" friction="0.95 0.3 0.1" 
                     type="mesh" mesh="wooden_cabinet_base_vis" conaffinity="0" contype="0" group="1" 
                     material="wooden_cabinet_base"/>
               <!-- Simplified collision geometry for cabinet base -->
               <geom type="box" pos="0.0 0.0 0.11" size="0.12 0.09 0.11" rgba="0.8 0.8 0.8 0.3"/>
               
               <!-- Top drawer (openable) -->
               <body name="cabinet_top" pos="0 0.0 0">
                   <inertial pos="0 0 0" mass="3" diaginertia="1 1 1" />
                   <joint name="top_level" type="slide" pos="0 0 0" axis="0 1 0"
                          limited="true" range="-0.16 0.01" damping="50"/>
                   
                   <!-- Top region site for goal detection -->
                   <site type="box" pos="0.00328 0.01128 0.18563"
                         quat="0.70711 0.00000 0.70711 0.00000"
                         size="0.02993 0.07561 0.10224" group="0"
                         rgba="0.8 0.8 0.8 0" name="top_region"/>
                   
                   <geom solimp="0.998 0.998 0.001" solref="0.001 1" density="100" friction="0.95 0.3 0.1" 
                         type="mesh" mesh="wooden_cabinet_top_vis" conaffinity="0" contype="0" group="1" 
                         material="wooden_cabinet_top"/>
                   <!-- Simplified collision for drawer -->
                   <geom type="box" pos="0.0 -0.07 0.18" size="0.11 0.03 0.10" rgba="0.8 0.8 0.8 0.3"/>
               </body>
           </body>
       </body>
       
       <!-- White Bowl (object_1, initially on table in object_init_region) -->
       <body name="object_1" pos="0.1 0.05 0.44">
           <freejoint/>
           <geom solimp="0.998 0.998 0.001" solref="0.001 1" density="100" friction="0.95 0.3 0.1" 
                 type="mesh" mesh="bowl_vis" conaffinity="0" contype="0" group="1" material="bowl_mat"/>
           <!-- Simplified collision for bowl -->
           <geom type="sphere" radius="0.03" rgba="0.8 0.8 0.8 0.3"/>
           
           <site rgba="0 0 0 0" size="0.005" pos="0 0 -0.03" name="bowl_bottom_site" />
           <site rgba="0 0 0 0" size="0.005" pos="0 0 0.02" name="bowl_top_site" />
       </body>
   </worldbody>
</mujoco>
'''

        # Save to file in the stanford_tidybot directory so it can find tidybot.xml
        tidybot_models_dir = os.path.join(os.path.dirname(__file__), "models", "stanford_tidybot")
        scene_file = os.path.join(tidybot_models_dir, "bddl_cabinet_scene.xml")
        with open(scene_file, 'w') as f:
            f.write(scene_xml)
        return scene_file

    def _create_tidybot_env(self) -> "MujocoEnv":
        """Create the underlying TidyBot MuJoCo environment."""
        kwargs = {
            "render_images": True,
            "show_viewer": self.show_viewer,
            "show_images": self.show_images,
            "mjcf_path": self.scene_path,
            "cabinet_scene": True,  # Enable cabinet-specific features
        }
        kwargs.update(self._extra_kwargs)
        
        return MujocoEnv(**kwargs)

    def _create_observation_space(self) -> spaces.Box:
        """Create observation space based on TidyBot's observation structure."""
        self._tidybot_env.reset()
        example_obs = self._tidybot_env.get_obs()

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
        for key in sorted(obs.keys()):
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
        self._reward_calculator = BDDLCabinetRewardCalculator()
        
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
        reward = self._reward_calculator.calculate_reward(obs)
        terminated = self._reward_calculator.is_terminated(obs)
        truncated = False

        return vec_obs, reward, terminated, truncated, {}

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
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
        return """BDDL Cabinet Manipulation Environment using TidyBot platform.

Task: Put the white bowl in the top drawer of the wooden cabinet.

The scene contains:
- A table with a white bowl initially placed on it
- A wooden cabinet with an openable top drawer
- The robot must pick up the bowl and place it inside the cabinet's top drawer

This environment is based on the BDDL specification from example_1_parsed.json.
"""

    def _create_obs_markdown_description(self) -> str:
        """Create observation space description."""
        return """Observation includes:
- Robot state: base pose, arm position/orientation, gripper state
- Object states: position and orientation of the white bowl (object_1)
- Cabinet state: position of wooden cabinet and handle positions
- Camera images: RGB images from base and wrist cameras
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
        return """Reward function for cabinet manipulation task:
- Small negative reward (-0.01) per timestep to encourage efficiency
- Large positive reward (+10.0) for successfully placing bowl in cabinet top drawer
- Smaller positive reward (+1.0) for maintaining bowl in correct position
- Task terminates when bowl is successfully placed in cabinet
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        return """Based on BDDL (Behavior Domain Definition Language) specification.

BDDL Paper:
- BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments
- Sanjana Srivastava, Chengshu Li, Michael Lingelbach, Roberto Martín-Martín, Fei Xia, Kent Elliott Vainio, Zheng Lian, Cem Gokmen, Shyamal Buch, Karen Liu, Silvio Savarese, Hyowon Gweon, Jiajun Wu, Li Fei-Fei
- Conference on Robot Learning (CoRL), 2021

TidyBot++:
- Jimmy Wu, William Chong, Robert Holmberg, Aaditya Prasad, Yihuai Gao, 
  Oussama Khatib, Shuran Song, Szymon Rusinkiewicz, Jeannette Bohg
- Conference on Robot Learning (CoRL), 2024
""" 