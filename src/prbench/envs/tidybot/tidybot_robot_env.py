"""This module defines the TidyBotRobotEnv class, which is the base class for the
TidyBot robot in simulation."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from prbench.envs.tidybot.arm_controller import ArmController
from prbench.envs.tidybot.base_controller import BaseController
from prbench.envs.tidybot.mujoco_utils import MujocoEnv


class TidyBotRobotEnv(MujocoEnv):
    """This is the base class for TidyBot environments that use MuJoCo for
    simulation."""

    def __init__(
        self,
        control_frequency: float,
        horizon: int = 1000,
        camera_names: Optional[list[str]] = None,
        camera_width: int = 640,
        camera_height: int = 480,
        seed: Optional[int] = None,
        show_viewer: bool = False,
    ) -> None:
        """
        Args:
            xml_string: A string containing the MuJoCo XML model.
            control_frequency: Frequency at which control actions are applied (in Hz).
            horizon: Maximum number of steps per episode.
        """

        super().__init__(
            control_frequency,
            horizon=horizon,
            camera_names=camera_names,
            camera_width=camera_width,
            camera_height=camera_height,
            seed=seed,
            show_viewer=show_viewer,
        )

        self.base_controller: Optional[BaseController] = None
        self.arm_controller: Optional[ArmController] = None

    def reset(self, xml_string: str) -> Tuple[Dict[str, np.ndarray], None, None, None]:
        """Reset the environment using xml string.

        Args:
            xml_string: A string containing the MuJoCo XML model.
        Returns:
            observation: The observation from the environment.
            reward: None (placeholder for compatibility).
            done: None (placeholder for compatibility).
            info: None (placeholder for compatibility).
        """
        # Insert robot in the xml_string
        xml_string = self._insert_robot_into_xml(xml_string)
        super().reset(xml_string)

        # Randomize the base pose of the robot in the sim
        self._randomize_base_pose()

        # Setup controllers after resetting the environment
        self._setup_controllers()

        return self.get_obs(), None, None, None  # reward, done, info

    def _randomize_base_pose(self) -> None:
        """Randomize the base pose of the robot within defined limits."""
        # Define limits for x, y, and theta
        x_limit = (-1.0, 1.0)
        y_limit = (-1.0, 1.0)
        theta_limit = (-np.pi, np.pi)
        # Sample random values within the limits
        x = self.np_random.uniform(*x_limit)
        y = self.np_random.uniform(*y_limit)
        theta = self.np_random.uniform(*theta_limit)
        # Set the base position and orientation in the simulation
        self.sim.data.qpos[0] = x  # x position
        self.sim.data.qpos[1] = y  # y position
        self.sim.data.qpos[2] = theta  # orientation (theta)
        self.sim.forward()  # Update the simulation state

    def _insert_robot_into_xml(self, xml_string: str) -> str:
        """Insert the robot model into the provided XML string."""
        xml_string = self._merge_additional_model_into_scene(
            xml_string,
            additional_model_xml_path=Path(__file__).parent
            / "models"
            / "stanford_tidybot"
            / "tidybot.xml",
            output_filename="tidybot_robot_env.xml",
            body_pos=(0.0, 0.0, 0.0),  # robot base frame is always at the origin
            name_prefix="robot_1",
        )
        return xml_string

    def _update_element_references(
        self,
        element: ET.Element,
        rename_map: dict[str, str],
    ) -> None:
        """Recursively update asset references in an element and its children."""
        for attr_ref in ["material", "mesh", "texture"]:
            original_ref = element.get(attr_ref)
            if original_ref and original_ref in rename_map:
                element.set(attr_ref, rename_map[original_ref])

        for child in element:
            self._update_element_references(child, rename_map)

    def _merge_additional_model_into_scene(
        self,
        xml_string: str,
        additional_model_xml_path: Union[str, Path],
        output_filename: str,
        body_pos: Optional[Tuple[float, float, float]] = None,
        name_prefix: Optional[str] = None,
    ) -> str:
        """Merge another MuJoCo model XML (e.g., tidybot) into an existing scene XML by:

        - Merging <default> classes
        - Merging <asset> entries while prefixing names to avoid collisions and
        rewriting file paths relative to the scene
        - Appending the top-level body from the additional model's <worldbody>
        into the scene's <worldbody>
        - Merging <contact>, <tendon>, <equality>, and <actuator> sections

        Returns absolute path to the newly written merged XML.
        """
        scene_tree = ET.ElementTree(ET.fromstring(xml_string))
        scene_root = scene_tree.getroot()

        # scene_output_dir = Path(scene_xml_path).parent

        # Ensure required roots exist in the scene
        scene_default = scene_root.find("default")
        if scene_default is None:
            scene_default = ET.SubElement(scene_root, "default")

        scene_asset = scene_root.find("asset")
        if scene_asset is None:
            scene_asset = ET.SubElement(scene_root, "asset")

        scene_worldbody = scene_root.find("worldbody")
        if scene_worldbody is None:
            scene_worldbody = ET.SubElement(scene_root, "worldbody")

        # Parse the additional model
        add_tree: ET.ElementTree = ET.parse(additional_model_xml_path)
        add_root: ET.Element = add_tree.getroot()

        # Determine name prefix for all elements from the additional model
        model_name: str = Path(additional_model_xml_path).stem
        prefix: str = name_prefix if name_prefix is not None else model_name

        # 1) Merge defaults (append all children of <default>)
        add_default = add_root.find("default")
        if add_default is not None:
            for child in list(add_default):
                scene_default.append(child)

        # 2) Merge assets with name prefixing and file path rewriting
        rename_map: Dict[str, str] = {}

        add_asset: Optional[ET.Element] = add_root.find("asset")
        if add_asset is not None:
            add_xml_dir: Path = Path(additional_model_xml_path).parent
            # Resolve meshdir if specified in additional model compiler
            add_compiler: Optional[ET.Element] = add_root.find("compiler")
            meshdir: Optional[str] = (
                add_compiler.get("meshdir") if add_compiler is not None else None
            )
            base_asset_dir: Path = (
                (add_xml_dir / meshdir).resolve() if meshdir else add_xml_dir.resolve()
            )
            for asset_elem in list(add_asset):
                # Determine original asset name
                original_name: Optional[str] = asset_elem.get("name")
                if original_name is None:
                    # Infer from file basename if available
                    file_attr: Optional[str] = asset_elem.get("file")
                    if file_attr:
                        original_name = Path(file_attr).stem
                # If we still don't have a name, skip renaming for this asset
                if original_name:
                    new_name: str = f"{prefix}_{original_name}"
                    rename_map[original_name] = new_name
                    asset_elem.set("name", new_name)

                # Rewrite file path to be the absolute path
                if "file" in asset_elem.attrib:
                    original_file: str = asset_elem.attrib["file"]
                    abs_asset_file_path: Path = (
                        base_asset_dir / original_file
                    ).resolve()
                    asset_elem.set("file", str(abs_asset_file_path))

                # Update intra-asset references (e.g., texture/material/mesh)
                # within the asset element itself
                for attr_ref in ["texture", "material", "mesh"]:
                    ref = asset_elem.get(attr_ref)
                    if ref and ref in rename_map:
                        asset_elem.set(attr_ref, rename_map[ref])

                scene_asset.append(asset_elem)

        # 3) Merge worldbody: copy the first top-level body from additional model,
        # update references, set pos
        add_worldbody: Optional[ET.Element] = add_root.find("worldbody")
        if add_worldbody is not None:
            add_top_body: Optional[ET.Element] = add_worldbody.find("body")
            if add_top_body is not None:
                # Deep-copy the body
                new_body: ET.Element = ET.fromstring(ET.tostring(add_top_body))
                # Update material/mesh/texture references according to rename_map
                self._update_element_references(new_body, rename_map)
                # Set position if provided
                if body_pos is not None:
                    new_body.set("pos", f"{body_pos[0]} {body_pos[1]} {body_pos[2]}")
                scene_worldbody.append(new_body)

        # 4) Merge contact / tendon / equality / actuator sections
        def _merge_section(tag_name: str) -> None:
            scene_sec = scene_root.find(tag_name)
            add_sec = add_root.find(tag_name)
            if add_sec is None:
                return
            if scene_sec is None:
                scene_sec = ET.SubElement(scene_root, tag_name)
            for child in list(add_sec):
                # These sections typically reference bodies/joints by name;
                # no renaming needed
                scene_sec.append(child)

        for tag in ["contact", "tendon", "equality", "actuator"]:
            _merge_section(tag)

        # Write merged model
        merged_output_path: str = os.path.join(
            os.path.dirname(__file__), output_filename
        )
        ET.indent(scene_tree, space="  ")
        scene_tree.write(
            str(merged_output_path), encoding="utf-8", xml_declaration=True
        )

        # return the modified XML string
        return ET.tostring(scene_root, encoding="unicode")

    def _pre_action(self, action: np.ndarray | None) -> None:
        """Do any preprocessing before taking an action.

        Args:
            action: Action to execute within the environment.
        """
        if self.base_controller is not None and action is not None:
            self.base_controller.run_controller(action)
        if self.arm_controller is not None and action is not None:
            self.arm_controller.run_controller(action)

    def step(
        self, action: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step the environment.

        Args:
            action: Optional action to apply before stepping.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        return super().step(action)

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Get the current observation.

        Returns:
            Dictionary containing the current observation.
        """
        return super().get_obs()

    def reward(self, **kwargs) -> float:
        """Compute the reward for the current state and action."""
        return 0.0  # Placeholder reward

    def _setup_controllers(self) -> None:
        """Setup the controllers for the robot."""

        # Cache references to array slices
        base_dofs: int = self.sim.model._model.body(  # pylint: disable=protected-access
            "base_link"
        ).jntnum.item()
        # VS: maybe "base_link" should include a prefix, such as "robot_1_base_link"
        arm_dofs: int = 7
        # buffers for base
        qpos_base: np.ndarray = self.sim.data.qpos[:base_dofs]
        qvel_base: np.ndarray = self.sim.data.qvel[:base_dofs]
        ctrl_base: np.ndarray = self.sim.data.ctrl[:base_dofs]
        # buffers for arm
        qpos_arm: np.ndarray = self.sim.data.qpos[base_dofs : (base_dofs + arm_dofs)]
        qvel_arm: np.ndarray = self.sim.data.qvel[base_dofs : (base_dofs + arm_dofs)]
        ctrl_arm: np.ndarray = self.sim.data.ctrl[base_dofs : (base_dofs + arm_dofs)]
        # buffers for gripper
        qpos_gripper: np.ndarray = self.sim.data.qpos[
            (base_dofs + arm_dofs) : (base_dofs + arm_dofs + 1)
        ]
        ctrl_gripper: np.ndarray = self.sim.data.ctrl[
            (base_dofs + arm_dofs) : (base_dofs + arm_dofs + 1)
        ]

        # Initialize controllers
        self.base_controller = BaseController(
            qpos_base,
            qvel_base,
            ctrl_base,
            self.sim.model._model.opt.timestep,  # pylint: disable=protected-access
        )
        self.arm_controller = ArmController(
            qpos_arm,
            qvel_arm,
            ctrl_arm,
            qpos_gripper,
            ctrl_gripper,
            self.sim.model._model.opt.timestep,  # pylint: disable=protected-access
        )

        # Reset controllers
        self.base_controller.reset()
        self.arm_controller.reset()  # also resets arm to retract position
