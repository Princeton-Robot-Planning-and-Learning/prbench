"""This module defines the TidyBotRobotEnv class, which is the base class for the
TidyBot robot in simulation."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

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
        show_images: bool = False,
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

        # Visualization flag
        self.show_images = show_images

        # Robot state/actuator references (initialized in _setup_robot_references)
        self.qpos_base: Optional[NDArray[np.float64]] = None
        self.qvel_base: Optional[NDArray[np.float64]] = None
        self.ctrl_base: Optional[NDArray[np.float64]] = None
        self.qpos_arm: Optional[NDArray[np.float64]] = None
        self.qvel_arm: Optional[NDArray[np.float64]] = None
        self.ctrl_arm: Optional[NDArray[np.float64]] = None
        self.qpos_gripper: Optional[NDArray[np.float64]] = None
        self.ctrl_gripper: Optional[NDArray[np.float64]] = None

        # Cached identifiers for commonly used sites/bodies
        self.ee_site_id: Optional[int] = None
        self.ee_body_id: Optional[int] = None

        # Cached environment IDs grouped by category (objects, tables, grounds, etc.)
        self.environment_ids: dict[str, list[int]] = {}

    def _visualize_image_in_window(self, image: NDArray[np.uint8], window_name: str) -> None:
        """Visualize an image in an OpenCV window."""
        if image.dtype == np.uint8 and len(image.shape) == 3:
            display_image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # pylint: disable=no-member
            cv.imshow(window_name, display_image)  # pylint: disable=no-member
            cv.waitKey(1)  # pylint: disable=no-member

    def _setup_robot_references(self) -> None:
        """Setup references to robot state/actuator buffers in the simulation data."""
        assert self.sim is not None, "Simulation must be initialized."

        # Joint names for the base and arm
        base_joint_names: list[str] = ["joint_x", "joint_y", "joint_th"]
        arm_joint_names: list[str] = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]

        # Joint positions: joint_id corresponds to qpos index
        base_qpos_indices = [
            self.sim.model.get_joint_qpos_addr(name) for name in base_joint_names
        ]
        arm_qpos_indices = [
            self.sim.model.get_joint_qpos_addr(name) for name in arm_joint_names
        ]

        # Joint velocities: joint_id corresponds to qvel index
        base_qvel_indices = [
            self.sim.model.get_joint_qvel_addr(name) for name in base_joint_names
        ]
        arm_qvel_indices = [
            self.sim.model.get_joint_qvel_addr(name) for name in arm_joint_names
        ]

        # Actuators: actuator_id corresponds to ctrl index
        base_ctrl_indices = [
            self.sim.model._actuator_name2id[name]  # pylint: disable=protected-access
            for name in base_joint_names
        ]
        arm_ctrl_indices = [
            self.sim.model._actuator_name2id[name]  # pylint: disable=protected-access
            for name in arm_joint_names
        ]

        # Verify indices are contiguous for slicing
        assert base_qpos_indices == list(
            range(min(base_qpos_indices), max(base_qpos_indices) + 1)
        ), "Base qpos indices not contiguous"
        assert arm_qpos_indices == list(
            range(min(arm_qpos_indices), max(arm_qpos_indices) + 1)
        ), "Arm qpos indices not contiguous"
        assert base_qvel_indices == list(
            range(min(base_qvel_indices), max(base_qvel_indices) + 1)
        ), "Base qvel indices not contiguous"
        assert arm_qvel_indices == list(
            range(min(arm_qvel_indices), max(arm_qvel_indices) + 1)
        ), "Arm qvel indices not contiguous"
        assert base_ctrl_indices == list(
            range(min(base_ctrl_indices), max(base_ctrl_indices) + 1)
        ), "Base ctrl indices not contiguous"
        assert arm_ctrl_indices == list(
            range(min(arm_ctrl_indices), max(arm_ctrl_indices) + 1)
        ), "Arm ctrl indices not contiguous"

        # Create views using correct slice ranges
        base_qpos_start, base_qpos_end = (
            min(base_qpos_indices),
            max(base_qpos_indices) + 1,
        )
        base_qvel_start, base_qvel_end = (
            min(base_qvel_indices),
            max(base_qvel_indices) + 1,
        )
        arm_qpos_start, arm_qpos_end = min(arm_qpos_indices), max(arm_qpos_indices) + 1
        arm_qvel_start, arm_qvel_end = (
            min(arm_qvel_indices),
            max(arm_qvel_indices) + 1,
        )
        base_ctrl_start, base_ctrl_end = (
            min(base_ctrl_indices),
            max(base_ctrl_indices) + 1,
        )
        arm_ctrl_start, arm_ctrl_end = min(arm_ctrl_indices), max(arm_ctrl_indices) + 1

        self.qpos_base = self.sim.data.qpos[base_qpos_start:base_qpos_end]
        self.qvel_base = self.sim.data.qvel[base_qvel_start:base_qvel_end]
        self.ctrl_base = self.sim.data.ctrl[base_ctrl_start:base_ctrl_end]

        self.qpos_arm = self.sim.data.qpos[arm_qpos_start:arm_qpos_end]
        self.qvel_arm = self.sim.data.qvel[arm_qvel_start:arm_qvel_end]
        self.ctrl_arm = self.sim.data.ctrl[arm_ctrl_start:arm_ctrl_end]

        # Buffers for gripper
        gripper_ctrl_id = (
            self.sim.model._actuator_name2id[  # pylint: disable=protected-access
                "fingers_actuator"
            ]
        )
        self.qpos_gripper = None
        self.ctrl_gripper = self.sim.data.ctrl[gripper_ctrl_id : gripper_ctrl_id + 1]

        # Cache end-effector identifiers if available
        try:
            self.ee_site_id = self.sim.model._site_name2id.get("pinch_site")  # type: ignore[attr-defined]
        except Exception:  # pylint: disable=broad-exception-caught
            self.ee_site_id = None
        try:
            self.ee_body_id = self.sim.model._body_name2id.get("tool_frame")  # type: ignore[attr-defined]
        except Exception:  # pylint: disable=broad-exception-caught
            self.ee_body_id = None

        # Cache environment IDs for objects, tables, and other fixtures
        self._cache_environment_ids()

    def _cache_environment_ids(self) -> None:
        """Populate environment_ids mapping by scanning model body names."""
        assert self.sim is not None, "Simulation must be initialized."

        objects: list[int] = []
        tables: list[int] = []
        grounds: list[int] = []
        cabinets: list[int] = []
        drawers: list[int] = []

        try:
            for name, body_id in self.sim.model._body_name2id.items():  # type: ignore[attr-defined]
                if name is None:
                    continue
                lname = name.lower()
                if lname.startswith("cube") or "object" in lname:
                    objects.append(body_id)
                if "table" in lname:
                    tables.append(body_id)
                if "ground" in lname or "floor" in lname:
                    grounds.append(body_id)
                if "cabinet" in lname or "cupboard" in lname:
                    cabinets.append(body_id)
                if "drawer" in lname:
                    drawers.append(body_id)
        except Exception:  # pylint: disable=broad-exception-caught
            # If mappings unavailable, leave empty
            pass

        self.environment_ids = {
            "objects": objects,
            "tables": tables,
            "grounds": grounds,
            "cabinets": cabinets,
            "drawers": drawers,
        }

    def get_environment_ids(self) -> dict[str, list[int]]:
        """Return cached environment IDs grouped by category."""
        return self.environment_ids

    def reset(
        self, xml_string: str
    ) -> tuple[dict[str, NDArray[Any]], None, None, None]:
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

        # Setup references to robot state/actuator buffers
        self._setup_robot_references()

        # Randomize the base pose of the robot in the sim
        # self._randomize_base_pose()

        # Setup controllers after resetting the environment
        self._setup_controllers()

        return self.get_obs(), None, None, None  # reward, done, info

    def _randomize_base_pose(self) -> None:
        """Randomize the base pose of the robot within defined limits."""
        assert (
            self.sim is not None
        ), "Simulation must be initialized before randomizing base pose."
        assert self.qpos_base is not None, "Base qpos must be initialized first"
        assert self.ctrl_base is not None, "Base ctrl must be initialized first"

        # Define limits for x, y, and theta
        x_limit = (-1.0, 1.0)
        y_limit = (-1.0, 1.0)
        theta_limit = (-np.pi, np.pi)
        # Sample random values within the limits
        x = self.np_random.uniform(*x_limit)
        y = self.np_random.uniform(*y_limit)
        theta = self.np_random.uniform(*theta_limit)
        # Set the base position and orientation in the simulation
        self.qpos_base[:] = [x, y, theta]
        self.ctrl_base[:] = [x, y, theta]
        self.sim.forward()  # Update the simulation state

    def _insert_robot_into_xml(self, xml_string: str) -> str:
        """Insert the robot model into the provided XML string."""
        # Parse the provided XML string
        input_tree = ET.ElementTree(ET.fromstring(xml_string))
        input_root = input_tree.getroot()

        # Read the scene XML content
        models_dir = Path(__file__).parent / "models" / "stanford_tidybot"
        tidybot_path = models_dir / "tidybot.xml"
        assets_dir = Path(__file__).parent / "models" / "assets"

        # Check if the input XML has an include directive for tidybot.xml
        include_elem = input_root.find("include")  # type: ignore[union-attr]
        if include_elem is not None and include_elem.get("file") == "tidybot.xml":
            # Remove the include directive since we'll merge the content directly
            input_root.remove(include_elem)  # type: ignore[union-attr]

        with open(tidybot_path, "r", encoding="utf-8") as f:
            tidybot_content = f.read()

        # Parse tidybot XML
        tidybot_tree = ET.ElementTree(ET.fromstring(tidybot_content))
        tidybot_root = tidybot_tree.getroot()
        if tidybot_root is None:
            raise ValueError("Missing <tidybot> element")

        # Update compiler meshdir to absolute path in tidybot content
        tidybot_compiler = tidybot_root.find("compiler")  # type: ignore[union-attr]
        if tidybot_compiler is not None:
            tidybot_compiler.set("meshdir", str(assets_dir.resolve()))

        # Merge the tidybot content into the input XML
        # Copy all children from tidybot root to input root (except mujoco tag itself)
        for child in list(tidybot_root):
            if child.tag == "worldbody":
                # Merge worldbody content
                input_worldbody = input_root.find(  # type:ignore[union-attr]
                    "worldbody"
                )
                if input_worldbody is not None:
                    for tidybot_body in list(child):
                        input_worldbody.append(tidybot_body)
                else:
                    input_root.append(child)  # type: ignore[union-attr]
            elif child.tag in ["asset", "default"]:
                # Merge or append asset and default sections
                input_section = input_root.find(child.tag)  # type: ignore[union-attr]
                if input_section is not None:
                    for sub_child in list(child):
                        input_section.append(sub_child)
                else:
                    input_root.append(child)  # type: ignore[union-attr]
            else:
                # For other sections (compiler, actuator, contact, etc.), just append
                input_root.append(child)  # type: ignore[union-attr]

        if input_root is None:
            raise ValueError("input_root is None, cannot serialize to string")

        # Return the merged XML as string
        return ET.tostring(input_root, encoding="unicode")

    def _pre_action(self, action: NDArray[Any] | dict[str, Any]) -> None:
        """Do any preprocessing before taking an action.

        Args:
            action: Action to execute within the environment.
        """
        if self.base_controller is not None and action is not None:
            self.base_controller.run_controller(action)
        if self.arm_controller is not None and action is not None:
            self.arm_controller.run_controller(action)

    def step(
        self, action: NDArray[Any] | dict[str, Any]
    ) -> tuple[dict[str, NDArray[Any]], float, bool, dict[str, Any]]:
        """Step the environment.

        Args:
            action: Optional action to apply before stepping.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        assert isinstance(action, dict), "Action must be a dictionary."
        obs, reward, done, info = super().step(action)

        # If requested, visualize available camera images
        if self.show_images and hasattr(self, "camera_names") and self.camera_names is not None:
            for camera_name in self.camera_names:
                key = f"{camera_name}_image"
                if key in obs:
                    self._visualize_image_in_window(obs[key], f"TidyBot {camera_name} camera")

        return obs, reward, done, info

    def reward(self, **kwargs: Any) -> float:
        """Compute the reward for the current state and action."""
        return 0.0  # Placeholder reward

    def _setup_controllers(self) -> None:
        """Setup the controllers for the robot."""

        assert (
            self.sim is not None
        ), "Simulation must be initialized before setting up controllers."

        # Ensure robot references are properly initialized
        assert self.qpos_base is not None, "Robot references must be set up first"
        assert self.qvel_base is not None, "Robot references must be set up first"
        assert self.ctrl_base is not None, "Robot references must be set up first"
        assert self.qpos_arm is not None, "Robot references must be set up first"
        assert self.qvel_arm is not None, "Robot references must be set up first"
        assert self.ctrl_arm is not None, "Robot references must be set up first"
        assert self.ctrl_gripper is not None, "Robot references must be set up first"

        # Initialize controllers
        self.base_controller = BaseController(
            self.qpos_base,
            self.qvel_base,
            self.ctrl_base,
            self.sim.model._model.opt.timestep,  # pylint: disable=protected-access
        )
        self.arm_controller = ArmController(
            self.qpos_arm,
            self.qvel_arm,
            self.ctrl_arm,
            self.qpos_gripper,
            self.ctrl_gripper,
            self.sim.model._model.opt.timestep,  # pylint: disable=protected-access
        )

        # Reset controllers
        self.base_controller.reset()
        self.arm_controller.reset()  # also resets arm to retract position

        self.sim.forward()

    def close(self) -> None:
        """Close environment and any visualization windows."""
        if self.show_images:
            cv.destroyAllWindows()  # pylint: disable=no-member
        super().close()
