"""MuJoCo environment for tidybot simulation and control.

This module provides a comprehensive MuJoCo-based environment for simulating robotic
systems with shared memory communication, real-time control, and multi-process rendering
capabilities. It includes classes for state management, image handling, rendering, and
various controllers for base and arm control.

Adapted from
https://github.com/jimmyyhwu/tidybot2
"""

import math
import multiprocessing as mp
import os
import re
import time
import traceback
from multiprocessing import shared_memory
from threading import Thread
from typing import Any, Optional

import cv2 as cv
import gymnasium
import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from prbench.envs.tidybot.arm_controller import ArmController
from prbench.envs.tidybot.base_controller import BaseController


class ShmState:
    """Shared memory state manager for robotic environment data.

    This class manages shared memory for storing and accessing robot state information
    including base pose, arm position/orientation, gripper state, and object
    positions/orientations across multiple processes.
    """

    def __init__(
        self,
        existing_instance: Optional["ShmState"] = None,
        num_objects: int = 3,
        object_names: Optional[list[str]] = None,
    ) -> None:
        # Calculate array size: 3 (base_pose) + 3 (arm_pos) + 4 (arm_quat) +
        # 1 (gripper_pos) + 1 (initialized) + num_objects * 7 (pos + quat for
        # each object)
        # initialized Meaning: It’s a shared-memory readiness flag (float 0.0/1.0)
        # indicating whether the producer process has populated the state
        # since the last reset.
        # It’s a synchronization barrier between processes to ensure
        # observations are ready before continuing.
        arr_size = 3 + 3 + 4 + 1 + 1 + num_objects * 7 + 6
        arr = np.empty(arr_size)
        if existing_instance is None:
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        self.data: NDArray = np.ndarray(arr.shape, buffer=self.shm.buf)

        # Fixed indices
        self.base_pose = self.data[:3]
        self.arm_pos = self.data[3:6]
        self.arm_quat = self.data[6:10]
        self.gripper_pos = self.data[10:11]
        self.initialized = self.data[11:12]

        # Dynamic object tracking
        self.num_objects = num_objects
        self.object_names = (
            object_names
            if object_names is not None
            else [f"cube{i+1}" for i in range(num_objects)]
        )
        self.object_positions = []
        self.object_quaternions = []

        # Calculate starting index for objects (after fixed fields)
        obj_start_idx = 12
        for i in range(num_objects):
            pos_start = obj_start_idx + i * 7
            quat_start = pos_start + 3
            self.object_positions.append(self.data[pos_start : pos_start + 3])
            self.object_quaternions.append(self.data[quat_start : quat_start + 4])

        self.initialized[:] = 0.0

    def close(self) -> None:
        """Close the shared memory segment."""
        self.shm.close()


class ShmImage:
    """Shared memory image manager for camera data.

    This class manages shared memory for storing and accessing camera image data across
    multiple processes, supporting both creation of new shared memory segments and
    attachment to existing ones.
    """

    def __init__(
        self,
        camera_name: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        existing_instance: Optional["ShmImage"] = None,
    ) -> None:
        if existing_instance is None:
            if camera_name is None or width is None or height is None:
                raise ValueError(
                    "ShmImage requires camera_name, width, and height when "
                    "creating new shared memory"
                )
            self.camera_name: str = camera_name
            arr = np.empty((height, width, 3), dtype=np.uint8)
            self.shm = shared_memory.SharedMemory(
                create=True,
                size=arr.nbytes,
            )
        else:
            self.camera_name = existing_instance.camera_name
            arr = existing_instance.data
            self.shm = shared_memory.SharedMemory(
                name=existing_instance.shm.name,
            )
        self.data: NDArray = np.ndarray(arr.shape, dtype=np.uint8, buffer=self.shm.buf)
        self.data.fill(0)

    def close(self) -> None:
        """Close the shared memory segment."""
        self.shm.close()


# Adapted from https://github.com/google-deepmind/mujoco/
# blob/main/python/mujoco/renderer.py
class Renderer:
    """MuJoCo renderer for offscreen image generation.

    This class provides offscreen rendering capabilities for MuJoCo simulations,
    generating camera images and storing them in shared memory for access by other
    processes.
    """

    def __init__(
        self,
        model: mujoco.MjModel,  # pylint: disable=no-member
        data: mujoco.MjData,  # pylint: disable=no-member
        shm_image: ShmImage,
    ) -> None:
        self.model = model
        self.data = data
        self.image = np.empty_like(shm_image.data)

        # Attach to existing shared memory image
        self.shm_image = ShmImage(existing_instance=shm_image)

        # Set up camera
        camera_id = mujoco.mj_name2id(  # pylint: disable=no-member
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA.value,  # pylint: disable=no-member
            shm_image.camera_name,  # pylint: disable=no-member
        )
        width, height = model.cam_resolution[camera_id]
        self.camera = mujoco.MjvCamera()  # pylint: disable=no-member
        self.camera.fixedcamid = camera_id
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED  # pylint: disable=no-member

        # Set up context depending on backend
        self.rect = mujoco.MjrRect(0, 0, width, height)  # pylint: disable=no-member

        if os.environ.get("MUJOCO_GL", "") == "osmesa":
            from mujoco.osmesa import (  # pylint: disable=relative-beyond-top-level, import-outside-toplevel
                GLContext,
            )
        else:
            from mujoco.gl_context import (  # pylint: disable=relative-beyond-top-level, import-outside-toplevel
                GLContext,
            )
        self.gl_context = GLContext(width, height)
        self.gl_context.make_current()

        self.mjr_context = mujoco.MjrContext(  # pylint: disable=no-member
            model,
            mujoco.mjtFontScale.mjFONTSCALE_150.value,  # pylint: disable=no-member
        )
        mujoco.mjr_setBuffer(  # pylint: disable=no-member
            mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value,  # pylint: disable=no-member
            self.mjr_context,  #  pylint: disable=no-member
        )

        # Set up scene
        self.scene_option = mujoco.MjvOption()  # pylint: disable=no-member
        self.scene = mujoco.MjvScene(model, 10000)  # pylint: disable=no-member

    def render(self) -> None:
        """Render the current scene to shared memory image."""
        self.gl_context.make_current()
        mujoco.mjv_updateScene(  # pylint: disable=no-member
            self.model,
            self.data,
            self.scene_option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,  # pylint: disable=no-member
            self.scene,
        )
        mujoco.mjr_render(  # pylint: disable=no-member
            self.rect, self.scene, self.mjr_context
        )
        mujoco.mjr_readPixels(  # pylint: disable=no-member
            self.image, None, self.rect, self.mjr_context
        )
        self.shm_image.data[:] = np.flipud(self.image)

    def close(self) -> None:
        """Free OpenGL and MuJoCo rendering resources."""
        if self.gl_context is not None:
            self.gl_context.free()
            self.gl_context = None
        if self.mjr_context is not None:
            self.mjr_context.free()
            self.mjr_context = None


class TidybotMujocoSim:
    """MuJoCo simulation environment with shared memory communication.

    This class manages a MuJoCo simulation with real-time control, object detection,
    state management, and multi-process rendering capabilities. It provides the core
    simulation loop and handles communication between different processes via shared
    memory.
    """

    def __init__(
        self,
        mjcf_path: str,
        command_queue: mp.Queue,
        shm_state: ShmState,
        show_viewer: bool = True,
        seed=None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(  # pylint: disable=no-member
            mjcf_path
        )
        self.data = mujoco.MjData(self.model)  # pylint: disable=no-member
        self.command_queue = command_queue
        self.show_viewer = show_viewer

        # Initialize random number generator
        if seed is not None:
            self.np_random, _ = gymnasium.utils.seeding.np_random(seed)
        else:
            self.np_random = np.random.default_rng()

        # Dynamically extract objects from the model
        self.extract_objects()

        # Enable gravity compensation for everything except objects
        self.model.body_gravcomp[:] = 1.0
        for object_name in self.object_names:
            if object_name in self.body_names:
                self.model.body_gravcomp[self.model.body(object_name).id] = 0.0

        # Cache references to array slices
        base_dofs = self.model.body("base_link").jntnum.item()
        arm_dofs = 7
        self.qpos_base = self.data.qpos[:base_dofs]
        qvel_base = self.data.qvel[:base_dofs]
        ctrl_base = self.data.ctrl[:base_dofs]
        qpos_arm = self.data.qpos[base_dofs : (base_dofs + arm_dofs)]
        qvel_arm = self.data.qvel[base_dofs : (base_dofs + arm_dofs)]
        ctrl_arm = self.data.ctrl[base_dofs : (base_dofs + arm_dofs)]
        self.qpos_gripper = self.data.qpos[
            (base_dofs + arm_dofs) : (base_dofs + arm_dofs + 1)
        ]
        ctrl_gripper = self.data.ctrl[
            (base_dofs + arm_dofs) : (base_dofs + arm_dofs + 1)
        ]

        # Dynamically track object qpos arrays
        self.qpos_objects = []
        current_idx = base_dofs + arm_dofs + 8  # After gripper
        for _ in range(self.num_objects):
            self.qpos_objects.append(self.data.qpos[current_idx : current_idx + 7])
            current_idx += 7

        # Controllers
        self.base_controller = BaseController(
            self.qpos_base, qvel_base, ctrl_base, self.model.opt.timestep
        )
        self.arm_controller = ArmController(
            qpos_arm,
            qvel_arm,
            ctrl_arm,
            self.qpos_gripper,
            ctrl_gripper,
            self.model.opt.timestep,
        )

        # Shared memory state for observations
        self.shm_state = ShmState(
            existing_instance=shm_state,
            num_objects=self.num_objects,
            object_names=self.object_names,
        )

        # Variables for calculating arm pos and quat
        site_id = self.model.site("pinch_site").id
        self.site_xpos = self.data.site(site_id).xpos
        self.site_xmat = self.data.site(site_id).xmat
        self.site_quat = np.empty(4)
        self.base_height = self.model.body("gen3/base_link").pos[2]
        self.base_rot_axis = np.array([0.0, 0.0, 1.0])
        self.base_quat_inv = np.empty(4)

        # Reset the environment
        self.reset()

        # Set control callback
        mujoco.set_mjcb_control(self.control_callback)  # pylint: disable=no-member

    def extract_objects(self) -> None:
        """Detects all object bodies in the MuJoCo model whose names match the pattern
        'cube+'.

        Populates self.object_names with the sorted list of detected object names and
        sets self.num_objects accordingly. Prints the number and names of detected
        objects for debugging purposes.
        """
        self.body_names = {self.model.body(i).name for i in range(self.model.nbody)}
        self.object_names = []
        for body_name in self.body_names:
            if re.match(r"cube\d+", body_name):
                self.object_names.append(body_name)
        self.object_names.sort()
        self.num_objects = len(self.object_names)

    def reset(self, seed=None) -> None:
        """Reset the simulation and randomize object positions."""
        # Set the random seed if provided
        if seed is not None:
            self.np_random, _ = gymnasium.utils.seeding.np_random(seed)
        else:
            self.np_random = np.random.default_rng()

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)  # pylint: disable=no-member

        # Randomize positions and orientations for all detected objects
        for cube_qpos in self.qpos_objects:

            # Randomize position within a reasonable range for the ground environment
            cube_qpos[0] = round(self.np_random.uniform(0.4, 0.8), 3)
            cube_qpos[1] = round(self.np_random.uniform(-0.3, 0.3), 3)
            cube_qpos[2] = 0.02

            # Randomize orientation around Z-axis (yaw)
            theta = self.np_random.uniform(-math.pi, math.pi)
            cube_quat = np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])
            cube_qpos[3:7] = cube_quat

        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

        # Reset controllers
        self.base_controller.reset()
        self.arm_controller.reset()

    def control_callback(
        self,
        _model: mujoco.MjModel,  # pylint: disable=no-member
        _data: mujoco.MjData,  # pylint: disable=no-member
    ) -> None:
        """Process control commands and update simulation state."""
        # Check for new command
        command = None if self.command_queue.empty() else self.command_queue.get()

        # Handle different command types
        if command is not None:
            if isinstance(command, dict) and command.get("action") == "reset":
                # Handle reset command with seed
                seed = command.get("seed")
                self.reset(seed=seed)
                command = None  # Clear command after processing reset
            elif command == "reset":
                # Handle legacy reset command
                self.reset()
                command = None  # Clear command after processing reset

        # Control callbacks
        cmd: dict[Any, Any] = (
            {}
            if command is None
            else command if isinstance(command, dict) else {"action": command}
        )
        self.base_controller.control_callback(cmd)
        self.arm_controller.control_callback(cmd)

        # Update base pose
        self.shm_state.base_pose[:] = self.qpos_base

        # Update arm pos
        site_xpos = self.site_xpos.copy()
        site_xpos[2] -= self.base_height  # Base height offset
        site_xpos[:2] -= self.qpos_base[:2]  # Base position inverse
        mujoco.mju_axisAngle2Quat(  # pylint: disable=no-member
            self.base_quat_inv, self.base_rot_axis, -self.qpos_base[2]
        )  # Base orientation inverse # pylint: disable=no-member
        mujoco.mju_rotVecQuat(  # pylint: disable=no-member
            self.shm_state.arm_pos, site_xpos, self.base_quat_inv
        )  # Arm pos in local frame # pylint: disable=no-member

        # Update arm quat
        mujoco.mju_mat2Quat(self.site_quat, self.site_xmat)  # pylint: disable=no-member
        mujoco.mju_mulQuat(  # pylint: disable=no-member
            self.shm_state.arm_quat, self.base_quat_inv, self.site_quat
        )  # Arm quat in local frame

        # Update gripper pos
        self.shm_state.gripper_pos[:] = (
            self.qpos_gripper / 0.8
        )  # right_driver_joint, joint range [0, 0.8]

        # Update all object positions and quaternions
        for i, qpos_obj in enumerate(self.qpos_objects):
            self.shm_state.object_positions[i][:] = qpos_obj[
                :3
            ]  # First 3 elements are position
            self.shm_state.object_quaternions[i][:] = qpos_obj[
                3:7
            ]  # Next 4 elements are quaternion

        # Notify reset() function that state has been initialized
        self.shm_state.initialized[:] = 1.0

    def launch(self) -> None:
        """Launch the MuJoCo viewer or run headless simulation."""
        if self.show_viewer:
            mujoco.viewer.launch(  # pylint: disable=no-member
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )

        else:
            # Run headless simulation at real-time speed
            last_step_time: float = 0.0
            while True:
                while time.time() - last_step_time < self.model.opt.timestep:
                    time.sleep(0.0001)
                last_step_time = time.time()
                mujoco.mj_step(self.model, self.data)  # pylint: disable=no-member


class MujocoEnv:
    """Main MuJoCo environment interface for robotic control.

    This class provides the main interface for interacting with the MuJoCo simulation
    environment, including observation retrieval, action execution, and multi-process
    management for rendering and physics simulation.
    """

    suppress_render_warning: bool = True  # Class-level default

    def __init__(
        self,
        render_images: bool = True,
        show_viewer: bool = True,
        show_images: bool = False,
        mjcf_path: Optional[str] = None,
    ) -> None:
        if mjcf_path is not None:
            self.mjcf_path = mjcf_path
        else:
            self.mjcf_path = "models/stanford_tidybot/scene.xml"
        self.render_images = render_images
        self.show_viewer = show_viewer
        self.show_images = show_images
        self.command_queue: mp.Queue = mp.Queue(1)
        self.suppress_render_warning = True  # Instance-level default

        # Detect objects from the model to determine shared memory size
        model = mujoco.MjModel.from_xml_path(  # pylint: disable=no-member
            self.mjcf_path
        )
        body_names = {model.body(i).name for i in range(model.nbody)}

        object_names = []
        for body_name in body_names:
            if re.match(r"cube\d+", body_name):
                object_names.append(body_name)
        object_names.sort()
        num_objects = len(object_names)

        # Shared memory for state observations
        self.shm_state = ShmState(num_objects=num_objects, object_names=object_names)

        # Shared memory for image observations
        if self.render_images:
            self.shm_images = []
            for camera_id in range(model.ncam):
                camera_name = model.camera(camera_id).name
                width, height = model.cam_resolution[camera_id]
                self.shm_images.append(ShmImage(camera_name, width, height))

        # Start physics loop
        mp.Process(target=self.physics_loop, daemon=True).start()

        if self.render_images and self.show_images:
            # Start visualizer loop
            mp.Process(target=self.visualizer_loop, daemon=True).start()

    def physics_loop(self) -> None:
        """Run the physics simulation loop in a separate process."""
        try:
            # Create sim
            sim = TidybotMujocoSim(
                self.mjcf_path,
                self.command_queue,
                self.shm_state,
                show_viewer=self.show_viewer,
            )

            # Start render loop
            if self.render_images:
                Thread(
                    target=self.render_loop, args=(sim.model, sim.data), daemon=True
                ).start()

            # Launch sim
            sim.launch()  # Launch in same thread as creation to avoid segfault
        except Exception as e:

            print("Physics process crashed:", e)
            traceback.print_exc()

    def render_loop(
        self,
        model: mujoco.MjModel,  # pylint: disable=no-member
        data: mujoco.MjData,  # pylint: disable=no-member
    ) -> None:
        """Run the rendering loop for camera images in a separate process."""
        # Set up renderers
        renderers = []
        for shm_image in self.shm_images:
            try:
                renderers.append(Renderer(model, data, shm_image))
            except Exception as e:
                print(
                    f"Error creating Renderer for camera '{shm_image.camera_name}': {e}"
                )
                traceback.print_exc()

        # Render camera images continuously
        while True:
            start_time = time.time()
            for renderer in renderers:
                try:
                    renderer.render()
                except Exception as e:
                    cam_name = getattr(
                        getattr(renderer, "shm_image", None), "camera_name", "<unknown>"
                    )
                    print(f"Error during rendering for camera '{cam_name}': {e}")
                    traceback.print_exc()
            render_time = time.time() - start_time
            if render_time > 0.1 and not self.suppress_render_warning:
                print(
                    f"Warning: Offscreen rendering took {1000 * render_time:.1f} ms, "
                    f"try making the Mujoco viewer window smaller to speed up "
                    f"offscreen rendering"
                )

    def visualizer_loop(self) -> None:
        """Run the visualizer loop for displaying camera images."""
        shm_images = [
            ShmImage(existing_instance=shm_image) for shm_image in self.shm_images
        ]
        last_imshow_time = time.time()
        while True:
            while time.time() - last_imshow_time < 0.1:  # 10 fps
                time.sleep(0.01)
            last_imshow_time = time.time()
            for i, shm_image in enumerate(shm_images):
                cv.imshow(  # pylint: disable=no-member
                    shm_image.camera_name,
                    cv.cvtColor(  # pylint: disable=no-member
                        shm_image.data, cv.COLOR_RGB2BGR  # pylint: disable=no-member
                    ),
                )
                cv.moveWindow(  # pylint: disable=no-member
                    shm_image.camera_name, 640 * i, -100
                )
            cv.waitKey(1)  # pylint: disable=no-member

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the environment and wait for initialization."""
        self.shm_state.initialized[:] = 0.0
        # Pass seed along with reset command
        reset_command = {"action": "reset", "seed": seed}
        self.command_queue.put(reset_command)

        # Wait for state publishing to initialize with timeout
        state_timeout_s = 10.0
        start_time = time.time()
        while self.shm_state.initialized[0] == 0.0:
            if time.time() - start_time > state_timeout_s:
                raise RuntimeError(
                    f"State initialization timed out after {state_timeout_s} seconds"
                )
            time.sleep(0.01)

        # Wait for image rendering to initialize
        # (Note: Assumes all zeros is not a valid image)
        if self.render_images:
            image_timeout_s = 15.0
            start_time = time.time()
            while any(np.all(shm_image.data == 0) for shm_image in self.shm_images):
                if time.time() - start_time > image_timeout_s:
                    not_ready = [
                        shm_image.camera_name
                        for shm_image in self.shm_images
                        if np.all(shm_image.data == 0)
                    ]
                    raise RuntimeError(
                        (
                            f"Image initialization timed out after {image_timeout_s} "
                            f"seconds; cameras not ready: {not_ready}"
                        )
                    )
                time.sleep(0.01)

    def get_obs(self) -> dict[str, NDArray]:
        """Get the current observation from the environment."""
        arm_quat = self.shm_state.arm_quat[[1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)

        # Process all object quaternions
        obs = {
            "base_pose": self.shm_state.base_pose.copy(),
            "arm_pos": self.shm_state.arm_pos.copy(),
            "arm_quat": arm_quat,
            "gripper_pos": self.shm_state.gripper_pos.copy(),
        }

        for i, object_name in enumerate(self.shm_state.object_names):
            if i < len(self.shm_state.object_positions):
                obj_pos = self.shm_state.object_positions[i].copy()
                obj_quat = self.shm_state.object_quaternions[i].copy()

                # Convert quaternion format (w, x, y, z) -> (x, y, z, w)
                obj_quat_converted = obj_quat[[1, 2, 3, 0]]
                if obj_quat_converted[3] < 0.0:  # Enforce quaternion uniqueness
                    np.negative(obj_quat_converted, out=obj_quat_converted)

                obs[f"{object_name}_pos"] = obj_pos
                obs[f"{object_name}_quat"] = obj_quat_converted

        if self.render_images:
            for shm_image in self.shm_images:
                obs[f"{shm_image.camera_name}_image"] = shm_image.data.copy()
        return obs

    def step(self, action) -> None:
        """Execute an action in the environment.

        Args:
            action: Dictionary containing control commands for base, arm, and gripper.
        """
        # Note: We intentionally do not return obs here to prevent the policy
        # from using outdated data
        self.command_queue.put(action)

    def close(self) -> None:
        """Close the environment and clean up shared memory resources."""
        self.shm_state.close()
        self.shm_state.shm.unlink()
        if self.render_images:
            for shm_image in self.shm_images:
                shm_image.close()
                shm_image.shm.unlink()
