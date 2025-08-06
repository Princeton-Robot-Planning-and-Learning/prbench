"""MuJoCo environment for robotic simulation and control.

This module provides a comprehensive MuJoCo-based environment for
simulating robotic systems with shared memory communication, real-time
control, and multi-process rendering capabilities. It includes classes
for state management, image handling, rendering, and various controllers
for base and arm control.
"""

# pylint: disable=no-member
# pylint: disable=no-name-in-module

import math
import multiprocessing as mp
import re
import time
import traceback
from multiprocessing import shared_memory
from threading import Thread

import cv2 as cv
import mujoco
import mujoco.viewer
import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from .constants import POLICY_CONTROL_PERIOD
from .ik_solver import IKSolver


class ShmState:
    """Shared memory state manager for robotic environment data.

    This class manages shared memory for storing and accessing robot
    state information including base pose, arm position/orientation,
    gripper state, and object positions/orientations across multiple
    processes.
    """

    def __init__(self, existing_instance=None, num_objects=3, object_names=None):
        # Calculate array size: 3 (base_pose) + 3 (arm_pos) + 4 (arm_quat) +
        # 1 (gripper_pos) + 1 (initialized) + num_objects * 7 (pos + quat for
        # each object) + 6 (2 handle positions)
        arr_size = 3 + 3 + 4 + 1 + 1 + num_objects * 7 + 6
        arr = np.empty(arr_size)
        if existing_instance is None:
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        self.data = np.ndarray(arr.shape, buffer=self.shm.buf)

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

        # Handle positions (after all objects)
        handle_start_idx = obj_start_idx + num_objects * 7
        self.left_handle_pos = self.data[handle_start_idx : handle_start_idx + 3]
        self.right_handle_pos = self.data[handle_start_idx + 3 : handle_start_idx + 6]

        self.initialized[:] = 0.0

    def close(self) -> None:
        """Close the shared memory segment."""
        self.shm.close()


class ShmImage:
    """Shared memory image manager for camera data.

    This class manages shared memory for storing and accessing camera
    image data across multiple processes, supporting both creation of
    new shared memory segments and attachment to existing ones.
    """

    def __init__(
        self, camera_name=None, width=None, height=None, existing_instance=None
    ):
        if existing_instance is None:
            self.camera_name = camera_name
            arr = np.empty((height, width, 3), dtype=np.uint8)
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.camera_name = existing_instance.camera_name
            arr = existing_instance.data
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        self.data = np.ndarray(arr.shape, dtype=np.uint8, buffer=self.shm.buf)
        self.data.fill(0)

    def close(self) -> None:
        """Close the shared memory segment."""
        self.shm.close()


# Adapted from https://github.com/google-deepmind/mujoco/
# blob/main/python/mujoco/renderer.py
class Renderer:
    """MuJoCo renderer for offscreen image generation.

    This class provides offscreen rendering capabilities for MuJoCo
    simulations, generating camera images and storing them in shared
    memory for access by other processes.
    """

    def __init__(self, model, data, shm_image):
        self.model = model
        self.data = data
        self.image = np.empty_like(shm_image.data)

        # Attach to existing shared memory image
        self.shm_image = ShmImage(existing_instance=shm_image)

        # Set up camera
        camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA.value, shm_image.camera_name
        )
        width, height = model.cam_resolution[camera_id]
        self.camera = mujoco.MjvCamera()
        self.camera.fixedcamid = camera_id
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # Set up context
        self.rect = mujoco.MjrRect(0, 0, width, height)
        self.gl_context = mujoco.gl_context.GLContext(width, height)
        self.gl_context.make_current()
        self.mjr_context = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        mujoco.mjr_setBuffer(
            mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.mjr_context
        )

        # Set up scene
        self.scene_option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, 10000)

    def render(self):
        """Render the current scene to shared memory image."""
        self.gl_context.make_current()
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.scene_option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mujoco.mjr_render(self.rect, self.scene, self.mjr_context)
        mujoco.mjr_readPixels(self.image, None, self.rect, self.mjr_context)
        self.shm_image.data[:] = np.flipud(self.image)

    def close(self) -> None:
        """Free OpenGL and MuJoCo rendering resources."""
        self.gl_context.free()
        self.gl_context = None
        self.mjr_context.free()
        self.mjr_context = None


class BaseController:
    """Controller for mobile base movement using online trajectory generation.

    This class implements a controller for the mobile base using
    Ruckig's online trajectory generation to ensure smooth, constrained
    motion with velocity and acceleration limits.
    """

    def __init__(self, qpos, qvel, ctrl, timestep):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl

        # OTG (online trajectory generation)
        num_dofs = 3
        self.last_command_time = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = [0.5, 0.5, 3.14]
        self.otg_inp.max_acceleration = [0.5, 0.5, 2.36]
        # self.otg_inp.max_velocity = [0.2, 0.2, 0.5]  # [x, y, theta] velocities
        # self.otg_inp.max_acceleration = [0.2, 0.2, 0.5]  # [x, y, theta] accelerations
        self.otg_res = None

    def reset(self):
        """Reset the base controller to origin position."""
        # Initialize base at origin
        self.qpos[:] = np.zeros(3)
        self.ctrl[:] = self.qpos

        # Initialize OTG
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command):
        """Process control commands and update base trajectory."""
        if command is not None:
            self.last_command_time = time.time()
            if "base_pose" in command:
                # Set target base qpos
                self.otg_inp.target_position = command["base_pose"]
                self.otg_res = Result.Working

        # Maintain current pose if command stream is disrupted
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.qpos
            self.otg_res = Result.Working

        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position


class ArmController:
    """Controller for robotic arm movement using inverse kinematics and
    trajectory generation.

    This class implements a controller for the robotic arm using inverse
    kinematics to convert end-effector poses to joint configurations,
    and Ruckig's online trajectory generation for smooth motion control.
    """

    def __init__(self, qpos, qvel, ctrl, qpos_gripper, ctrl_gripper, timestep):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.qpos_gripper = qpos_gripper
        self.ctrl_gripper = ctrl_gripper

        # IK solver
        self.ik_solver = IKSolver(ee_offset=0.12)

        # OTG (online trajectory generation)
        num_dofs = 7
        self.last_command_time = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
        self.otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [
            math.radians(450)
        ]
        self.otg_res = None

    def reset(self):
        """Reset the arm controller to retract configuration."""
        # Initialize arm in "retract" configuration
        self.qpos[:] = np.array(
            [0.0, -0.34906585, 3.14159265, -2.54818071, 0.0, -0.87266463, 1.57079633]
        )
        self.ctrl[:] = self.qpos
        self.ctrl_gripper[:] = 0.0

        # Initialize OTG
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command):
        """Process control commands and update arm trajectory."""
        if command is not None:
            self.last_command_time = time.time()

            if "arm_pos" in command:
                # Run inverse kinematics on new target pose
                qpos = self.ik_solver.solve(
                    command["arm_pos"], command["arm_quat"], self.qpos
                )
                qpos = (
                    self.qpos + np.mod((qpos - self.qpos) + np.pi, 2 * np.pi) - np.pi
                )  # Unwrapped joint angles

                # Set target arm qpos
                self.otg_inp.target_position = qpos
                self.otg_res = Result.Working

            if "gripper_pos" in command:
                # Set target gripper pos
                self.ctrl_gripper[:] = (
                    255.0 * command["gripper_pos"]
                )  # fingers_actuator, ctrlrange [0, 255]

        # Maintain current pose if command stream is disrupted
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.otg_out.new_position
            self.otg_res = Result.Working

        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position


class MujocoSim:
    """MuJoCo simulation environment with shared memory communication.

    This class manages a MuJoCo simulation with real-time control,
    object detection, state management, and multi-process rendering
    capabilities. It provides the core simulation loop and handles
    communication between different processes via shared memory.
    """

    def __init__(
        self,
        mjcf_path,
        command_queue,
        shm_state,
        show_viewer=True,
    ):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.command_queue = command_queue
        self.show_viewer = show_viewer

        # Dynamically detect objects from the model
        self.detect_objects()

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
        mujoco.set_mjcb_control(self.control_callback)

    def detect_objects(self):
        """Detects all object bodies in the MuJoCo model whose names match the
        pattern 'cube+'.

        Populates self.object_names with the sorted list of detected
        object names and sets self.num_objects accordingly. Prints the
        number and names of detected objects for debugging purposes.
        """
        self.body_names = {self.model.body(i).name for i in range(self.model.nbody)}
        self.object_names = []
        for body_name in self.body_names:
            if re.match(r"cube\d+", body_name):
                self.object_names.append(body_name)
        self.object_names.sort()
        self.num_objects = len(self.object_names)
        print(f"Detected {self.num_objects} objects: {self.object_names}", flush=True)

    def reset(self, seed=None):
        """Reset the simulation and randomize object positions."""

        print("RESET CALLED IN MUJOCO SIM", flush=True)

        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Randomize positions and orientations for all detected objects
        for _, (object_name, cube_qpos) in enumerate(
            zip(self.object_names, self.qpos_objects)
        ):

            # Randomize position within a reasonable range around the table
            cube_qpos[:2] += np.random.uniform(-0.3, 0.3, 2)
            # Keep Z position at table height (don't randomize vertical position)

            # Randomize orientation around Z-axis (yaw)
            theta = np.random.uniform(-math.pi, math.pi)
            cube_quat = np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])
            cube_qpos[3:7] = cube_quat

            print(
                f"{object_name} reset to position: "
                f"[{cube_qpos[0]:.3f}, {cube_qpos[1]:.3f}, {cube_qpos[2]:.3f}], "
                f"theta: {theta:.3f}",
                flush=True
            )

        mujoco.mj_forward(self.model, self.data)

        # Reset controllers
        self.base_controller.reset()
        self.arm_controller.reset()

    def control_callback(self, *_):
        """Process control commands and update simulation state."""
        # Check for new command
        command = None if self.command_queue.empty() else self.command_queue.get()

        # Handle different command types
        if command is not None:
            if isinstance(command, dict) and command.get("action") == "reset":
                # Handle reset command with seed
                seed = command.get("seed")
                print("RESET CALLED IN CONTROL CALLBACK (v1)", flush=True)
                self.reset(seed=seed)
                command = None  # Clear command after processing reset
            elif command == "reset":
                # Handle legacy reset command
                print("RESET CALLED IN CONTROL CALLBACK (v2)", flush=True)
                self.reset()
                command = None  # Clear command after processing reset

        # Control callbacks
        self.base_controller.control_callback(command)
        self.arm_controller.control_callback(command)

        # Update base pose
        self.shm_state.base_pose[:] = self.qpos_base

        # Update arm pos
        # self.shm_state.arm_pos[:] = self.site_xpos
        site_xpos = self.site_xpos.copy()
        site_xpos[2] -= self.base_height  # Base height offset
        site_xpos[:2] -= self.qpos_base[:2]  # Base position inverse
        mujoco.mju_axisAngle2Quat(
            self.base_quat_inv, self.base_rot_axis, -self.qpos_base[2]
        )  # Base orientation inverse
        mujoco.mju_rotVecQuat(
            self.shm_state.arm_pos, site_xpos, self.base_quat_inv
        )  # Arm pos in local frame

        # Update arm quat
        mujoco.mju_mat2Quat(self.site_quat, self.site_xmat)
        # self.shm_state.arm_quat[:] = self.site_quat
        mujoco.mju_mulQuat(
            self.shm_state.arm_quat, self.base_quat_inv, self.site_quat
        )  # Arm quat in local frame

        # Update gripper pos
        self.shm_state.gripper_pos[:] = (
            self.qpos_gripper / 0.8
        )  # right_driver_joint, joint range [0, 0.8]

        # Update all object positions and quaternions
        for i, (_, qpos_obj) in enumerate(zip(self.object_names, self.qpos_objects)):
            self.shm_state.object_positions[i][:] = qpos_obj[
                :3
            ]  # First 3 elements are position
            self.shm_state.object_quaternions[i][:] = qpos_obj[
                3:7
            ]  # Next 4 elements are quaternion

        # Update handle positions if cabinet_scene
        # if self.cabinet_scene:
        #     try:
        #         left_id = self.model.site("leftdoor_site").id
        #         right_id = self.model.site("rightdoor_site").id
        #         self.shm_state.left_handle_pos[:] = self.data.site(left_id).xpos
        #         self.shm_state.right_handle_pos[:] = self.data.site(right_id).xpos
        #     except Exception as e:
        #         print(f"Warning: Could not update handle positions: {e}")

        # Notify reset() function that state has been initialized
        self.shm_state.initialized[:] = 1.0

    def launch(self):
        """Launch the MuJoCo viewer or run headless simulation."""
        if self.show_viewer:
            mujoco.viewer.launch(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )

        else:
            # Run headless simulation at real-time speed
            last_step_time = 0
            while True:
                while time.time() - last_step_time < self.model.opt.timestep:
                    time.sleep(0.0001)
                last_step_time = time.time()
                mujoco.mj_step(self.model, self.data)


class MujocoEnv:
    """Main MuJoCo environment interface for robotic control.

    This class provides the main interface for interacting with the
    MuJoCo simulation environment, including observation retrieval,
    action execution, and multi-process management for rendering and
    physics simulation.
    """

    def __init__(
        self,
        render_images=True,
        show_viewer=True,
        show_images=False,
        mjcf_path=None,
    ):
        if mjcf_path is not None:
            self.mjcf_path = mjcf_path
        else:
            self.mjcf_path = "models/stanford_tidybot/scene.xml"
        self.render_images = render_images
        self.show_viewer = show_viewer
        self.show_images = show_images
        self.command_queue = mp.Queue(1)

        # Detect objects from the model to determine shared memory size
        model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        body_names = {model.body(i).name for i in range(model.nbody)}

        object_names = []
        for body_name in body_names:
            if re.match(r"cube\d+", body_name):
                object_names.append(body_name)
        object_names.sort()
        num_objects = len(object_names)
        print(f"Detected {num_objects} objects in scene: {object_names}", flush=True)

        # Shared memory for state observations
        self.shm_state = ShmState(num_objects=num_objects, object_names=object_names)

        # Shared memory for image observations
        if self.render_images:
            self.shm_images = []
            print("model.ncam:", model.ncam, flush=True)
            for camera_id in range(model.ncam):
                camera_name = model.camera(camera_id).name
                width, height = model.cam_resolution[camera_id]
                self.shm_images.append(ShmImage(camera_name, width, height))

        # Start physics loop
        print("Starting physics loop", flush=True)
        mp.Process(target=self.physics_loop, daemon=True).start()

        if self.render_images and self.show_images:
            print("Starting visualizer loop", flush=True)
            # Start visualizer loop
            mp.Process(target=self.visualizer_loop, daemon=True).start()
        
        print("Finished __init__ for mujoco env", flush=True)

    def physics_loop(self):
        """Run the physics simulation loop in a separate process."""
        try:
            # Create sim
            print("Creating mujoco sim", flush=True)
            sim = MujocoSim(
                self.mjcf_path,
                self.command_queue,
                self.shm_state,
                show_viewer=self.show_viewer,
            )

            # Start render loop
            if self.render_images:
                print("Starting thread with render loop", flush=True)
                Thread(
                    target=self.render_loop, args=(sim.model, sim.data), daemon=True
                ).start()

            # Launch sim
            print("Launching mujoco sim", flush=True)
            sim.launch()  # Launch in same thread as creation to avoid segfault
            print("Done launching mujoco sim", flush=True)
        except Exception as e:

            print("Physics process crashed:", e, flush=True)
            traceback.print_exc()

    def render_loop(self, model, data):
        """Run the rendering loop for camera images in a separate process."""
        # Set up renderers
        print("Attempting to create renderers, I wonder if this will crash silently...", flush=True)
        try:
            renderers = [Renderer(model, data, shm_image) for shm_image in self.shm_images]
            print("Nope it didn't crash", flush=True)
        except Exception as e:
            print("YES IT CRASHED :SCREAMFACE:", flush=True)
            print(str(e), flush=True)

        # Render camera images continuously
        while True:
            print("In rendering loop", flush=True)
            start_time = time.time()
            for renderer in renderers:
                renderer.render()
            render_time = time.time() - start_time
            if render_time > 0.1:  # 10 fps
                print(
                    f"Warning: Offscreen rendering took {1000 * render_time:.1f} ms, "
                    f"try making the Mujoco viewer window smaller to speed up "
                    f"offscreen rendering",
                    flush=True,
                )

    def visualizer_loop(self):
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
                cv.imshow(
                    shm_image.camera_name, cv.cvtColor(shm_image.data, cv.COLOR_RGB2BGR)
                )
                cv.moveWindow(shm_image.camera_name, 640 * i, -100)
            cv.waitKey(1)

    def reset(self, seed: int | None = None) -> None:
        """Reset the environment and wait for initialization."""
        print("RESET CALLED IN MUJOCO ENV", flush=True)
        self.shm_state.initialized[:] = 0.0
        # Pass seed along with reset command
        reset_command = {"action": "reset", "seed": seed}
        self.command_queue.put(reset_command)

        print("L687", flush=True)

        # Wait for state publishing to initialize
        while self.shm_state.initialized == 0.0:
            time.sleep(0.01)

        print("L693", flush=True)

        # Wait for image rendering to initialize
        # (Note: Assumes all zeros is not a valid image)
        if self.render_images:
            print("LEN(self.shm_images):", self.shm_images, flush=True)
            while any(np.all(shm_image.data == 0) for shm_image in self.shm_images):
                time.sleep(0.01)

        print("L701", flush=True)

    def get_obs(self):
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

        # if self.cabinet_scene:
        #     obs["left_handle_pos"] = self.shm_state.left_handle_pos.copy()
        #     obs["right_handle_pos"] = self.shm_state.right_handle_pos.copy()
        if self.render_images:
            for shm_image in self.shm_images:
                obs[f"{shm_image.camera_name}_image"] = shm_image.data.copy()
        return obs

    def step(self, action):
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
