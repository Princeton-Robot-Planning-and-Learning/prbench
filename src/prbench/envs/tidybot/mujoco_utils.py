"""This module provides utilities for working with MuJoCo environments.

It includes the `MujocoEnv` class, which serves as a base class for environments
that use MuJoCo for simulation, and the `MjSim` class, which encapsulates the
MuJoCo simulation logic.
"""

import abc
import ctypes
import ctypes.util
import gc
import os
import platform
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from threading import Lock
from typing import Any, TypeAlias

import gymnasium
import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

# This value is then used by the physics engine to determine how much time
# to simulate for each step.
SIMULATION_TIMESTEP = 0.002  # (in seconds)

# Set macros needed for MuJoCo rendering
_SYSTEM = platform.system()
if _SYSTEM == "Windows":
    ctypes.WinDLL(  # type: ignore[attr-defined]
        os.path.join(os.path.dirname(__file__), "mujoco.dll")
    )
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if CUDA_VISIBLE_DEVICES != "":
    MUJOCO_EGL_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
    if MUJOCO_EGL_DEVICE_ID is not None:
        assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
            MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
        ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device \
            id specified in CUDA_VISIBLE_DEVICES"
if os.environ.get("MUJOCO_GL", None) not in [
    "osmesa",
    "glx",
]:
    # VS: maybe this should be put behind a macro that toggles GPU rendering
    if _SYSTEM == "Darwin":
        os.environ["MUJOCO_GL"] = "cgl"
    else:
        os.environ["MUJOCO_GL"] = "egl"
_MUJOCO_GL = os.environ.get("MUJOCO_GL", "").lower().strip()

_MjSim_render_lock = Lock()


MjObs: TypeAlias = dict[str, NDArray[Any]]


@dataclass(frozen=True)
class MjAct:
    """An action in a MuJoCo environment.

    The position_ctrl field is used to set sim.data.ctrl in the MuJoCo environment. For
    now, we assume that all actuators (as defined in the MuJoCo xml) use position
    control, hence the variable name.
    """

    position_ctrl: NDArray[np.float64]


class MujocoEnv(gymnasium.Env[MjObs, MjAct]):
    """This is the base class for environments that use MuJoCo for simulation."""

    def __init__(
        self,
        control_frequency: float,
        horizon: int = 1000,
        camera_names: list[str] | None = None,
        camera_width: int = 640,
        camera_height: int = 480,
        seed: int | None = None,
        show_viewer: bool = False,
    ) -> None:
        """
        Args:
            control_frequency: Frequency at which control actions are applied (in Hz).
            horizon: Maximum number of steps per episode.
            camera_names: List of camera names to use for rendering.
            camera_width: Width of camera images.
            camera_height: Height of camera images.
            seed: Random seed for reproducibility.
            show_viewer: Whether to show the MuJoCo viewer.
        """
        # Simulation-related attributes, change with creating/closing env
        self.sim: MjSim | None = None
        self.timestep: int | None = None

        self.show_viewer: bool = show_viewer
        self.control_frequency: float = control_frequency
        self.horizon: int = horizon
        self.camera_names: list[str] = camera_names if camera_names is not None else []
        self.camera_width: int = camera_width
        self.camera_height: int = camera_height

        # Initialize random number generator
        self.np_random = np.random.default_rng(seed)
        super().__init__()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[MjObs, dict[str, Any]]:
        # Reset the random seed.
        super().reset(seed=seed, options=options)

        # Access the xml.
        assert options is not None and "xml" in options, "XML required to reset env"
        xml_string = options["xml"]

        # Destroy the current simulation if it exists.
        self._close_sim()

        # Initialize the simulation with the provided XML string.
        self._create_sim(xml_string)

        assert self.sim is not None, "Simulation must be initialized after _create_sim"
        self.sim.reset()
        self.sim.forward()
        return self.get_obs(), {}

    def render(self):
        # Subclasses should override.
        pass

    @abc.abstractmethod
    def reward(self, obs: MjObs) -> float:
        """Compute the reward from an observation."""

    def step(self, action: MjAct) -> tuple[MjObs, float, bool, bool, dict[str, Any]]:
        assert self.sim is not None, "Simulation must be initialized before stepping."

        assert self.timestep is not None, "Timestep must be initialized."
        self.timestep += 1

        # Step the simulation with the same action until the control frequency
        # is reached
        control_timestep = 1.0 / self.control_frequency
        for _ in range(int(control_timestep / SIMULATION_TIMESTEP)):
            self.sim.data.ctrl[:] = action.position_ctrl
            self.sim.forward()
            self.sim.step()

        # Post-action processing
        obs = self.get_obs()
        reward = self.reward(obs)
        terminated = self.timestep >= self.horizon
        truncated = False
        info: dict[str, object] = {}

        return obs, reward, terminated, truncated, info

    def get_obs(self) -> MjObs:
        """Get the current observation."""
        assert self.sim is not None, "Simulation must be initialized."

        # Add a copy of qpos and qvel to observation
        obs_dict: dict[str, NDArray[Any]] = {
            "qpos": np.copy(self.sim.data.qpos),
            "qvel": np.copy(self.sim.data.qvel),
        }

        # Render images and update obs_dict
        images: dict[str, NDArray[Any]] | None = self._get_camera_images()
        if images is not None:
            obs_dict.update(images)

        return obs_dict

    def _get_camera_images(self) -> dict[str, NDArray[np.uint8]] | None:
        """Get images from cameras in simulation."""
        if not self.camera_names or self.sim is None:
            return None

        images: dict[str, NDArray[np.uint8]] = {}
        for camera_name in self.camera_names:
            rendered_image = self.sim.render(
                width=self.camera_width,
                height=self.camera_height,
                camera_name=camera_name,
                depth=False,
                mode="offscreen",
            )
            # Handle both single image and tuple return types
            if isinstance(rendered_image, tuple):
                images[f"{camera_name}_image"] = rendered_image[0]
            else:
                images[f"{camera_name}_image"] = rendered_image
        return images

    def _close_sim(self) -> None:
        """Destroys the current MjSim instance.

        Sets timestep counter to None.
        """
        if self.sim is not None:
            self.sim.free()
        self.sim = None
        self.timestep = None

    def _create_sim(self, xml_string: str) -> None:
        """Initialize the MuJoCo simulation with the provided XML string. Also resets
        the timestep counter.

        Args:
            xml_string: A string containing the MuJoCo XML model.
        """
        self.sim: MjSim = MjSim(  # type: ignore[no-redef]
            xml_string,
            self.camera_width,
            self.camera_height,
        )
        self.timestep: int = 0  # type: ignore[no-redef]

        if self.show_viewer:
            mujoco.viewer.launch(
                self.sim.model._model,  # pylint: disable=protected-access
                self.sim.data,
                show_left_ui=False,
                show_right_ui=False,
            )

    def close(self) -> None:
        """Close the environment and free resources."""
        self._close_sim()


class MjModel:
    """A simplified MjModel class for MuJoCo model."""

    # pylint: disable=no-member

    def __init__(self, xml_string: str) -> None:
        """
        Args:
            xml_string: A string containing the MuJoCo XML model.
        """

        self._model = mujoco.MjModel.from_xml_string(xml_string)
        self._make_mappings()

    def get_joint_qpos_addr(self, name: str) -> int:
        """
        See
        https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1178

        Returns the qpos address for given joint.

        Args:
            name (str): name of the joint

        Returns:
            address (int): returns int address in qpos array
        """
        if name not in self._joint_name2id:
            # Filter out None names for display
            available_names = [n for n in self.joint_names if n is not None]
            raise ValueError(
                f'No "joint" with name {name} exists. '
                f'Available "joint" names = {available_names}.'
            )
        joint_id = self._joint_name2id[name]
        assert joint_id is not None, "Joint ID should not be None here."
        return self._model.jnt_qposadr[joint_id]

    def get_joint_qvel_addr(self, name: str) -> int:
        """
        See
        https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1202

        Returns the qvel address for given joint.

        Args:
            name (str): name of the joint

        Returns:
            address (int): returns int address in qvel array
        """
        if name not in self._joint_name2id:
            # Filter out None names for display
            available_names = [n for n in self.joint_names if n is not None]
            raise ValueError(
                f'No "joint" with name {name} exists. '
                f'Available "joint" names = {available_names}.'
            )
        joint_id = self._joint_name2id[name]
        assert joint_id is not None, "Joint ID should not be None here."
        return self._model.jnt_dofadr[joint_id]

    def _make_mappings(self) -> None:
        """Make some useful internal mappings that mujoco-py supported."""
        self.body_names: tuple[str | None, ...]
        self._body_name2id: dict[str | None, int]
        self._body_id2name: dict[int, str | None]
        (
            self.body_names,
            self._body_name2id,
            self._body_id2name,
        ) = self._extract_mj_names(
            self._model.nbody,
            mujoco.mjtObj.mjOBJ_BODY,
        )
        self.joint_names: tuple[str | None, ...]
        self._joint_name2id: dict[str | None, int]
        self._joint_id2name: dict[int, str | None]
        (
            self.joint_names,
            self._joint_name2id,
            self._joint_id2name,
        ) = self._extract_mj_names(
            self._model.njnt,
            mujoco.mjtObj.mjOBJ_JOINT,
        )
        self.geom_names: tuple[str | None, ...]
        self._geom_name2id: dict[str | None, int]
        self._geom_id2name: dict[int, str | None]
        (
            self.geom_names,
            self._geom_name2id,
            self._geom_id2name,
        ) = self._extract_mj_names(
            self._model.ngeom,
            mujoco.mjtObj.mjOBJ_GEOM,
        )
        self.site_names: tuple[str | None, ...]
        self._site_name2id: dict[str | None, int]
        self._site_id2name: dict[int, str | None]
        (
            self.site_names,
            self._site_name2id,
            self._site_id2name,
        ) = self._extract_mj_names(
            self._model.nsite,
            mujoco.mjtObj.mjOBJ_SITE,
        )
        self.light_names: tuple[str | None, ...]
        self._light_name2id: dict[str | None, int]
        self._light_id2name: dict[int, str | None]
        (
            self.light_names,
            self._light_name2id,
            self._light_id2name,
        ) = self._extract_mj_names(
            self._model.nlight,
            mujoco.mjtObj.mjOBJ_LIGHT,
        )
        self.camera_names: tuple[str | None, ...]
        self._camera_name2id: dict[str | None, int]
        self._camera_id2name: dict[int, str | None]
        (
            self.camera_names,
            self._camera_name2id,
            self._camera_id2name,
        ) = self._extract_mj_names(
            self._model.ncam,
            mujoco.mjtObj.mjOBJ_CAMERA,
        )
        self.actuator_names: tuple[str | None, ...]
        self._actuator_name2id: dict[str | None, int]
        self._actuator_id2name: dict[int, str | None]
        (
            self.actuator_names,
            self._actuator_name2id,
            self._actuator_id2name,
        ) = self._extract_mj_names(
            self._model.nu,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
        )
        self.sensor_names: tuple[str | None, ...]
        self._sensor_name2id: dict[str | None, int]
        self._sensor_id2name: dict[int, str | None]
        (
            self.sensor_names,
            self._sensor_name2id,
            self._sensor_id2name,
        ) = self._extract_mj_names(
            self._model.nsensor,
            mujoco.mjtObj.mjOBJ_SENSOR,
        )
        self.tendon_names: tuple[str | None, ...]
        self._tendon_name2id: dict[str | None, int]
        self._tendon_id2name: dict[int, str | None]
        (
            self.tendon_names,
            self._tendon_name2id,
            self._tendon_id2name,
        ) = self._extract_mj_names(
            self._model.ntendon,
            mujoco.mjtObj.mjOBJ_TENDON,
        )
        self.mesh_names: tuple[str | None, ...]
        self._mesh_name2id: dict[str | None, int]
        self._mesh_id2name: dict[int, str | None]
        (
            self.mesh_names,
            self._mesh_name2id,
            self._mesh_id2name,
        ) = self._extract_mj_names(
            self._model.nmesh,
            mujoco.mjtObj.mjOBJ_MESH,
        )

    def camera_name2id(self, name: str) -> int:
        """Get camera id from camera name."""
        if name == "free":
            return -1
        if name not in self._camera_name2id:
            # Filter out None names for display
            available_names = [n for n in self.camera_names if n is not None]
            raise ValueError(
                f'No "camera" with name {name} exists. '
                f'Available "camera" names = {available_names}.'
            )
        return self._camera_name2id[name]

    def _extract_mj_names(
        self,
        num_obj: int,
        obj_type: int,
    ) -> tuple[tuple[str | None, ...], dict[str | None, int], dict[int, str | None]]:
        """Extract MuJoCo object names and create mappings.

        See:
        https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1127
        """

        # Objects don't need to be named in the XML, so name might be None
        id2name: dict[int, str | None] = {i: None for i in range(num_obj)}
        name2id: dict[str | None, int] = {}
        for i in range(num_obj):
            name: str | None = mujoco.mj_id2name(
                self._model, obj_type, i
            )  # pylint: disable=no-member
            name2id[name] = i
            id2name[i] = name

        # Sort names by increasing id to keep order deterministic
        return tuple(id2name[nid] for nid in sorted(name2id.values())), name2id, id2name


class MjSim:
    """A simplified MjSim class for MuJoCo simulation."""

    def __init__(self, xml_string: str, camera_width: int, camera_height: int) -> None:
        """
        Args:
            xml_string: A string containing the MuJoCo XML model.
            camera_width: Width of camera images.
            camera_height: Height of camera images.
        """

        xml_string = self._set_simulation_timestep(xml_string)

        self.model: MjModel = MjModel(xml_string)
        self.data: mujoco.MjData = mujoco.MjData(  # pylint: disable=no-member
            self.model._model  # pylint: disable=protected-access
        )

        # Offscreen render context object
        self._render_context_offscreen: MjRenderContextOffscreen = (
            MjRenderContextOffscreen(
                self, device_id=-1, max_width=camera_width, max_height=camera_height
            )
        )

    def _set_simulation_timestep(self, xml_string: str) -> str:
        """Set the simulation timestep in the XML string.

        Args:
            xml_string: A string containing the MuJoCo XML model.

        Returns:
            Modified XML string with updated timestep.
        """
        # Parse the XML string
        root: ET.Element = ET.fromstring(xml_string)

        # Find the <option> tag and set its timestep attribute
        option: ET.Element | None = root.find("option")
        if option is not None:
            option.set("timestep", str(SIMULATION_TIMESTEP))
        else:
            # If <option> tag does not exist, create it and insert as first child
            option = ET.Element("option", {"timestep": str(SIMULATION_TIMESTEP)})
            root.insert(0, option)

        # Convert the modified XML tree back to a string
        return ET.tostring(root, encoding="unicode")

    def reset(self) -> None:
        """Reset the simulation."""
        mujoco.mj_resetData(  # pylint: disable=no-member
            self.model._model, self.data  # pylint: disable=protected-access
        )

    def forward(self) -> None:
        """Synchronize derived quantities."""
        mujoco.mj_forward(  # pylint: disable=no-member
            self.model._model, self.data  # pylint: disable=protected-access
        )

    def step(self) -> None:
        """Step the simulation."""
        mujoco.mj_step(  # pylint: disable=no-member
            self.model._model, self.data  # pylint: disable=protected-access
        )

    def render(
        self,
        width: int | None = None,
        height: int | None = None,
        camera_name: str | None = None,
        depth: bool = False,
        mode: str = "offscreen",
    ) -> NDArray[np.uint8] | tuple[NDArray[np.uint8], NDArray[np.float32] | None]:
        """Renders view from a camera and returns image as an `numpy.ndarray`.

        Args:
        - width (int): desired image width.
        - height (int): desired image height.
        - camera_name (str): name of camera in model. If None, the free
            camera will be used.
        - depth (bool): if True, also return depth buffer
        - device (int): device to use for rendering (only for GPU-backed
            rendering).
        Returns:
        - rgb (uint8 array): image buffer from camera
        - depth (float array): depth buffer from camera (only returned
            if depth=True)
        """
        if camera_name is None:
            camera_id = None
        else:
            camera_id = self.model.camera_name2id(camera_name)

        # Use default dimensions if not provided
        render_width = width or self._render_context_offscreen.con.offWidth
        render_height = height or self._render_context_offscreen.con.offHeight

        assert mode == "offscreen", "only offscreen supported for now"
        assert self._render_context_offscreen is not None
        with _MjSim_render_lock:
            self._render_context_offscreen.render(
                width=render_width,
                height=render_height,
                camera_id=camera_id,
            )
            return self._render_context_offscreen.read_pixels(
                render_width, render_height, depth=depth
            )

    def free(self) -> None:
        """Free the simulation resources."""
        del self._render_context_offscreen
        del self.data
        del self.model
        del self
        gc.collect()


class MjRenderContext:
    """Class that encapsulates rendering functionality for a MuJoCo simulation.

    See
    https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/mjrendercontext.pyx
    """

    # pylint: disable=no-member

    def __init__(
        self,
        sim: MjSim,
        offscreen: bool = True,
        device_id: int = -1,
        max_width: int = 640,
        max_height: int = 480,
    ) -> None:
        if _MUJOCO_GL not in ("disable", "disabled", "off", "false", "0"):
            _VALID_MUJOCO_GL = ("enable", "enabled", "on", "true", "1", "glfw", "")
            if _SYSTEM == "Linux":
                _VALID_MUJOCO_GL += ("glx", "egl", "osmesa")  # type: ignore[assignment]
            elif _SYSTEM == "Windows":
                _VALID_MUJOCO_GL += ("wgl",)  # type: ignore[assignment]
            elif _SYSTEM == "Darwin":
                _VALID_MUJOCO_GL += ("cgl",)  # type: ignore[assignment]
            if _MUJOCO_GL not in _VALID_MUJOCO_GL:
                raise RuntimeError(
                    f"invalid value for environment variable MUJOCO_GL: {_MUJOCO_GL}"
                )
            # fmt: off
            # pylint: disable=import-outside-toplevel
            # isort: off
            if _SYSTEM == "Linux" and _MUJOCO_GL == "osmesa":
                from prbench.envs.tidybot.renderers.context.osmesa_context import (
                    OSMesaGLContext as GLContext,)

                # TODO this needs testing on a Linux machine  # pylint: disable=fixme
            elif _SYSTEM == "Linux" and _MUJOCO_GL == "egl":
                from prbench.envs.tidybot.renderers.context.egl_context import (  # type: ignore[assignment] # pylint: disable=line-too-long
                    EGLGLContext as GLContext,)

                # TODO this needs testing on a Linux machine  # pylint: disable=fixme
            else:
                from prbench.envs.tidybot.renderers.context.glfw_context import (  # type: ignore[assignment] # pylint: disable=line-too-long
                    GLFWGLContext as GLContext,)
            # isort: on
            # fmt: on

        assert offscreen, "only offscreen supported for now"
        self.sim: MjSim = sim
        self.offscreen: bool = offscreen
        self.device_id: int = device_id

        # Setup GL context with defaults for now
        self.gl_ctx = GLContext(  # type: ignore[no-untyped-call] # pylint: disable=possibly-used-before-assignment
            max_width=max_width,
            max_height=max_height,
            device_id=self.device_id,
        )
        self.gl_ctx.make_current()

        # Ensure the model data has been updated so that there
        # is something to render
        sim.forward()

        self.model: mujoco.MjModel = (
            sim.model._model
        )  # pylint: disable=protected-access
        self.data: mujoco.MjData = sim.data

        # Create default scene
        # Set maxgeom to 10k to support large-scale scenes
        self.scn: mujoco.MjvScene = mujoco.MjvScene(self.model, maxgeom=10000)

        # Camera
        self.cam: mujoco.MjvCamera = mujoco.MjvCamera()
        self.cam.fixedcamid = 0
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # Options for visual / collision mesh can be set externally,
        # e.g. vopt.geomgroup[0], vopt.geomgroup[1]
        self.vopt: mujoco.MjvOption = mujoco.MjvOption()

        self.pert: mujoco.MjvPerturb = mujoco.MjvPerturb()
        self.pert.active = 0
        self.pert.select = 0
        self.pert.skinselect = -1

        self._set_mujoco_context_and_buffers()

    def _set_mujoco_context_and_buffers(self) -> None:
        """Set up the mujoco rendering context and buffers."""
        self.con: mujoco.MjrContext = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150
        )
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)

    def update_offscreen_size(self, width: int, height: int) -> None:
        """Update the offscreen rendering context size if necessary."""
        if (width != self.con.offWidth) or (height != self.con.offHeight):
            self.model.vis.global_.offwidth = width
            self.model.vis.global_.offheight = height
            self.con.free()
            del self.con
            self._set_mujoco_context_and_buffers()

    def render(
        self,
        width: int,
        height: int,
        camera_id: int | None = None,
    ) -> None:
        """Render the scene."""
        viewport = mujoco.MjrRect(0, 0, width, height)

        # update width and height of rendering context if necessary
        if width > self.con.offWidth or height > self.con.offHeight:
            new_width = max(width, self.model.vis.global_.offwidth)
            new_height = max(height, self.model.vis.global_.offheight)
            self.update_offscreen_size(new_width, new_height)

        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        mujoco.mjr_render(viewport=viewport, scn=self.scn, con=self.con)

    def read_pixels(
        self,
        width: int,
        height: int,
        depth: bool = False,
    ) -> NDArray[np.uint8] | tuple[NDArray[np.uint8], NDArray[np.float32] | None]:
        """Read the pixels from the current rendering context.

        Returns:
            NDArray[np.uint8] if depth is False,
            tuple of (NDArray[np.uint8], NDArray[np.float32] or None) if depth is True.
        """
        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb_img: NDArray[np.uint8] = np.empty((height, width, 3), dtype=np.uint8)
        depth_img: NDArray[np.float32] | None = (
            np.empty((height, width), dtype=np.float32) if depth else None
        )

        mujoco.mjr_readPixels(
            rgb=rgb_img, depth=depth_img, viewport=viewport, con=self.con
        )

        rgb_img = np.flipud(rgb_img)
        if depth_img is not None:
            depth_img = np.flipud(depth_img)

        if depth:
            return (rgb_img, depth_img)
        return rgb_img

    def upload_texture(self, tex_id: int) -> None:
        """Uploads given texture to the GPU."""
        self.gl_ctx.make_current()
        mujoco.mjr_uploadTexture(self.model, self.con, tex_id)

    def __del__(self) -> None:
        """Free mujoco rendering context and GL rendering context."""
        self.con.free()
        try:
            self.gl_ctx.free()
        except Exception:  # pylint: disable=broad-exception-caught
            # avoid getting OpenGL.error.GLError
            pass
        del self.con
        del self.gl_ctx
        del self.scn
        del self.cam
        del self.vopt
        del self.pert


class MjRenderContextOffscreen(MjRenderContext):
    """Class that encapsulates offscreen rendering functionality for a MuJoCo
    simulation."""

    def __init__(
        self,
        sim: MjSim,
        device_id: int,
        max_width: int = 640,
        max_height: int = 480,
    ) -> None:
        super().__init__(
            sim,
            offscreen=True,
            device_id=device_id,
            max_width=max_width,
            max_height=max_height,
        )
