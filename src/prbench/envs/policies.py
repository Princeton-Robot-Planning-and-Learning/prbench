# Author: Jimmy Wu
# Date: October 2024

import logging
import math
import socket
import threading
import time
from enum import Enum, auto
from queue import Queue

import cv2 as cv
import numpy as np
import zmq
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from scipy.spatial.transform import Rotation as R

from .agent.mp_policy import (
    MotionPlannerPolicy as MotionPlannerPolicyMP,)  # Import new MP policy
from .agent.open_policy import (
    CloseCabinetPolicy as CloseCabinetPolicy,)  # Import new MP policy for second phase
from .agent.open_policy import (
    MotionPlannerPolicyCabinetMP as MotionPlannerPolicyCabinetMP,)  # Import new MP policy
from .agent.open_policy import (
    MotionPlannerPolicyCabinetMP_1 as MotionPlannerPolicyCabinetMP_1,)  # Import new MP policy for second phase
from .agent.stack_policies import MotionPlannerPolicyStack  # <-- Add this import
from .agent.stack_policies import (  # <-- Add this import for table stacking; <-- Add this import for drawer stacking; <-- Add this import for cupboard stacking
    MotionPlannerPolicyStackCupboard,
    MotionPlannerPolicyStackDrawer,
    MotionPlannerPolicyStackTable,
)
from .constants import (
    POLICY_IMAGE_HEIGHT,
    POLICY_IMAGE_WIDTH,
    POLICY_SERVER_HOST,
    POLICY_SERVER_PORT,
)


class Policy:
    def reset(self):
        raise NotImplementedError

    def step(self, obs):
        raise NotImplementedError


class WebServer:
    def __init__(self, queue):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.queue = queue

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.socketio.on("message")
        def handle_message(data):
            # Send the timestamp back for RTT calculation (expected RTT on 5 GHz Wi-Fi is 7 ms)
            emit("echo", data["timestamp"])

            # Add data to queue for processing
            self.queue.put(data)

        # Reduce verbose Flask log output
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

    def run(self):
        # Get IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(("8.8.8.8", 1))
            address = s.getsockname()[0]
        except Exception:
            address = "127.0.0.1"
        finally:
            s.close()
        print(f"Starting server at {address}:5000")
        self.socketio.run(self.app, host="0.0.0.0")


DEVICE_CAMERA_OFFSET = np.array([0.0, 0.02, -0.04])  # iPhone 14 Pro


# Convert coordinate system from WebXR to robot
def convert_webxr_pose(pos, quat):
    # WebXR: +x right, +y up, +z back; Robot: +x forward, +y left, +z up
    pos = np.array([-pos["z"], -pos["x"], pos["y"]], dtype=np.float64)
    rot = R.from_quat([-quat["z"], -quat["x"], quat["y"], quat["w"]])

    # Apply offset so that rotations are around device center instead of device camera
    pos = pos + rot.apply(DEVICE_CAMERA_OFFSET)

    return pos, rot


TWO_PI = 2 * math.pi


class TeleopController:
    def __init__(self):
        # Teleop device IDs
        self.primary_device_id = (
            None  # Primary device controls either the arm or the base
        )
        self.secondary_device_id = None  # Optional secondary device controls the base
        self.enabled_counts = {}

        # Mobile base pose
        self.base_pose = None

        # Teleop targets
        self.targets_initialized = False
        self.base_target_pose = None
        self.arm_target_pos = None
        self.arm_target_rot = None
        self.gripper_target_pos = None

        # WebXR reference poses
        self.base_xr_ref_pos = None
        self.base_xr_ref_rot_inv = None
        self.arm_xr_ref_pos = None
        self.arm_xr_ref_rot_inv = None

        # Robot reference poses
        self.base_ref_pose = None
        self.arm_ref_pos = None
        self.arm_ref_rot = None
        self.arm_ref_base_pose = None  # For optional secondary control of base
        self.gripper_ref_pos = None

    def process_message(self, data):
        if not self.targets_initialized:
            return

        # Use device ID to disambiguate between primary and secondary devices
        device_id = data["device_id"]

        # Update enabled count for the device that sent this message
        self.enabled_counts[device_id] = (
            self.enabled_counts.get(device_id, 0) + 1 if "teleop_mode" in data else 0
        )

        # Assign primary and secondary devices
        if self.enabled_counts[device_id] > 2:
            if self.primary_device_id is None and device_id != self.secondary_device_id:
                # Note: We skip the first 2 steps because WebXR pose updates have higher latency than touch events
                self.primary_device_id = device_id
            elif (
                self.secondary_device_id is None and device_id != self.primary_device_id
            ):
                self.secondary_device_id = device_id
        elif self.enabled_counts[device_id] == 0:
            if device_id == self.primary_device_id:
                self.primary_device_id = None  # Primary device no longer enabled
                self.base_xr_ref_pos = None
                self.arm_xr_ref_pos = None
            elif device_id == self.secondary_device_id:
                self.secondary_device_id = None
                self.base_xr_ref_pos = None

        # Teleop is enabled
        if self.primary_device_id is not None and "teleop_mode" in data:
            pos, rot = convert_webxr_pose(data["position"], data["orientation"])

            # Base movement
            if (
                data["teleop_mode"] == "base" or device_id == self.secondary_device_id
            ):  # Note: Secondary device can only control base
                # Store reference poses
                if self.base_xr_ref_pos is None:
                    self.base_ref_pose = self.base_pose.copy()
                    self.base_xr_ref_pos = pos[:2]
                    self.base_xr_ref_rot_inv = rot.inv()

                # Position
                self.base_target_pose[:2] = self.base_ref_pose[:2] + (
                    pos[:2] - self.base_xr_ref_pos
                )

                # Orientation
                base_fwd_vec_rotated = (rot * self.base_xr_ref_rot_inv).apply(
                    [1.0, 0.0, 0.0]
                )
                base_target_theta = self.base_ref_pose[2] + math.atan2(
                    base_fwd_vec_rotated[1], base_fwd_vec_rotated[0]
                )
                self.base_target_pose[2] += (
                    base_target_theta - self.base_target_pose[2] + math.pi
                ) % TWO_PI - math.pi  # Unwrapped

            # Arm movement
            elif data["teleop_mode"] == "arm":
                # Store reference poses
                if self.arm_xr_ref_pos is None:
                    self.arm_xr_ref_pos = pos
                    self.arm_xr_ref_rot_inv = rot.inv()
                    self.arm_ref_pos = self.arm_target_pos.copy()
                    self.arm_ref_rot = self.arm_target_rot
                    self.arm_ref_base_pose = self.base_pose.copy()
                    self.gripper_ref_pos = self.gripper_target_pos

                # Rotations around z-axis to go between global frame (base) and local frame (arm)
                z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.base_pose[2])
                z_rot_inv = z_rot.inv()
                ref_z_rot = R.from_rotvec(
                    np.array([0.0, 0.0, 1.0]) * self.arm_ref_base_pose[2]
                )

                # Position
                pos_diff = pos - self.arm_xr_ref_pos  # WebXR
                pos_diff += ref_z_rot.apply(self.arm_ref_pos) - z_rot.apply(
                    self.arm_ref_pos
                )  # Secondary base control: Compensate for base rotation
                pos_diff[:2] += (
                    self.arm_ref_base_pose[:2] - self.base_pose[:2]
                )  # Secondary base control: Compensate for base translation
                self.arm_target_pos = self.arm_ref_pos + z_rot_inv.apply(pos_diff)

                # Orientation
                self.arm_target_rot = (
                    z_rot_inv * (rot * self.arm_xr_ref_rot_inv) * ref_z_rot
                ) * self.arm_ref_rot

                # Gripper position
                self.gripper_target_pos = np.clip(
                    self.gripper_ref_pos + data["gripper_delta"], 0.0, 1.0
                )

        # Teleop is disabled
        elif self.primary_device_id is None:
            # Update target pose in case base is pushed while teleop is disabled
            self.base_target_pose = self.base_pose

    def step(self, obs):
        # Update robot state
        self.base_pose = obs["base_pose"]

        # Initialize targets
        if not self.targets_initialized:
            self.base_target_pose = obs["base_pose"]
            self.arm_target_pos = obs["arm_pos"]
            self.arm_target_rot = R.from_quat(obs["arm_quat"])
            self.gripper_target_pos = obs["gripper_pos"]
            self.targets_initialized = True

        # Return no action if teleop is not enabled
        if self.primary_device_id is None:
            return None

        # Get most recent teleop command
        arm_quat = self.arm_target_rot.as_quat()
        if (
            arm_quat[3] < 0.0
        ):  # Enforce quaternion uniqueness (Note: Not strictly necessary since policy training uses 6D rotation representation)
            np.negative(arm_quat, out=arm_quat)
        action = {
            "base_pose": self.base_target_pose.copy(),
            "arm_pos": self.arm_target_pos.copy(),
            "arm_quat": arm_quat,
            "gripper_pos": self.gripper_target_pos.copy(),
        }

        return action


# Teleop using WebXR phone web app
class TeleopPolicy(Policy):
    def __init__(self):
        self.web_server_queue = Queue()
        self.teleop_controller = None
        self.teleop_state = (
            None  # States: episode_started -> episode_ended -> reset_env
        )
        self.episode_ended = False

        # Web server for serving the WebXR phone web app
        server = WebServer(self.web_server_queue)
        threading.Thread(target=server.run, daemon=True).start()

        # Listener thread to process messages from WebXR client
        threading.Thread(target=self.listener_loop, daemon=True).start()

    def reset(self):
        self.teleop_controller = TeleopController()
        self.episode_ended = False

        # Wait for user to signal that episode has started
        self.teleop_state = None
        while self.teleop_state != "episode_started":
            time.sleep(0.01)

    def step(self, obs):
        # Signal that user has ended episode
        if not self.episode_ended and self.teleop_state == "episode_ended":
            self.episode_ended = True
            return "end_episode"

        # Signal that user is ready for env reset (after ending the episode)
        if self.teleop_state == "reset_env":
            return "reset_env"

        return self._step(obs)

    def _step(self, obs):
        return self.teleop_controller.step(obs)

    def listener_loop(self):
        while True:
            if not self.web_server_queue.empty():
                data = self.web_server_queue.get()

                # Update state
                if "state_update" in data:
                    self.teleop_state = data["state_update"]

                # Process message if not stale
                elif 1000 * time.time() - data["timestamp"] < 250:  # 250 ms
                    self._process_message(data)

            time.sleep(0.001)

    def _process_message(self, data):
        self.teleop_controller.process_message(data)


# Execute policy running on remote server
class RemotePolicy(TeleopPolicy):
    def __init__(self):
        super().__init__()

        # Use phone as enabling device during policy rollout
        self.enabled = False

        # Connection to policy server
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}")
        print(
            f"Connected to policy server at {POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}"
        )

    def reset(self):
        # Wait for user to signal that episode has started
        super().reset()  # Note: Comment out to run without phone

        # Check connection to policy server and reset policy
        default_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # Temporarily set 1000 ms timeout
        self.socket.send_pyobj({"reset": True})
        try:
            self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
        except zmq.error.Again as e:
            raise Exception("Could not communicate with policy server") from e
        self.socket.setsockopt(
            zmq.RCVTIMEO, default_timeout
        )  # Put default timeout back

        # Disable policy execution until user presses on screen
        self.enabled = False  # Note: Set to True to run without phone

    def _step(self, obs):
        # Return teleop command if episode has ended
        if self.episode_ended:
            return self.teleop_controller.step(obs)

        # Return no action if robot is not enabled
        if not self.enabled:
            return None

        # Encode images
        encoded_obs = {}
        for k, v in obs.items():
            if v.ndim == 3:
                # Resize image to resolution expected by policy server
                v = cv.resize(v, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT))

                # Encode image as JPEG
                _, v = cv.imencode(".jpg", v)  # Note: Interprets RGB as BGR
                encoded_obs[k] = v
            else:
                encoded_obs[k] = v

        # Send obs to policy server
        req = {"obs": encoded_obs}
        self.socket.send_pyobj(req)

        # Get action from policy server
        rep = (
            self.socket.recv_pyobj()
        )  # Note: Not secure. Only unpickle data you trust.
        action = rep["action"]

        return action

    def _process_message(self, data):
        if self.episode_ended:
            # Run teleop controller if episode has ended
            self.teleop_controller.process_message(data)
        else:
            # Enable policy execution if user is pressing on screen
            self.enabled = "teleop_mode" in data


# Stacking motion planner policy (stacking three objects)
class MotionPlannerPolicyStackWrapper(Policy):
    def __init__(self):
        self.impl = MotionPlannerPolicyStack()

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Stacking three cubes using two sequential stack policies
class MotionPlannerPolicyStackThreeWrapper(Policy):
    def __init__(self):
        self.stack1 = MotionPlannerPolicyStack()
        self.stack2 = MotionPlannerPolicyStack()
        self.phase = 0  # 0: first stack, 1: second stack, 2: done
        self.episode_ended = False

    def reset(self):
        self.stack1.reset()
        self.stack2.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None

        # Phase 0: stack first two cubes
        if self.phase == 0:
            action = self.stack1.step(obs)
            if self.stack1.episode_ended:
                # Prepare for second stacking: adjust stack2's placement height
                # Find the current top cube's position and set stack2's STACK_HEIGHT_OFFSET
                # We'll use the same logic as stack1, but increase the offset
                # Get the last stack location from stack1
                if (
                    hasattr(self.stack1, "stack_location")
                    and self.stack1.stack_location is not None
                ):
                    # The new stack height should be one cube height above the previous stack
                    # Assume cube height is the same as stack1.STACK_HEIGHT_OFFSET
                    cube_height = self.stack1.STACK_HEIGHT_OFFSET
                    self.stack2.STACK_HEIGHT_OFFSET = 1.5 * cube_height
                self.phase = 1
                self.stack2.reset()  # Ensure stack2 is ready
            return action
        # Phase 1: stack third cube on top
        elif self.phase == 1:
            action = self.stack2.step(obs)
            if self.stack2.episode_ended:
                self.phase = 2
                self.episode_ended = True
            return action
        # Phase 2: done
        else:
            return None


# Table stacking policy wrapper
class MotionPlannerPolicyStackTableWrapper(Policy):
    def __init__(self):
        self.impl = MotionPlannerPolicyStackTable()

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Table stacking three cubes using two sequential table stack policies
class MotionPlannerPolicyStackTableThreeWrapper(Policy):
    def __init__(self):
        self.stack1 = MotionPlannerPolicyStackTable()
        self.stack2 = MotionPlannerPolicyStackTable()
        self.phase = 0  # 0: first stack, 1: second stack, 2: done
        self.episode_ended = False

    def reset(self):
        self.stack1.reset()
        self.stack2.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None

        # Phase 0: stack first two cubes
        if self.phase == 0:
            action = self.stack1.step(obs)
            if self.stack1.episode_ended:
                # Prepare for second stacking: adjust stack2's placement height
                # Find the current top cube's position and set stack2's STACK_HEIGHT_OFFSET
                # We'll use the same logic as stack1, but increase the offset
                # Get the last stack location from stack1
                if (
                    hasattr(self.stack1, "stack_location")
                    and self.stack1.stack_location is not None
                ):
                    # The new stack height should be one cube height above the previous stack
                    # Assume cube height is the same as stack1.STACK_HEIGHT_OFFSET
                    cube_height = self.stack1.STACK_HEIGHT_OFFSET
                    self.stack2.STACK_HEIGHT_OFFSET = (
                        1.8 * cube_height
                    )  # Stack third cube on top of second
                self.phase = 1
                self.stack2.reset()  # Ensure stack2 is ready
            return action
        # Phase 1: stack third cube on top
        elif self.phase == 1:
            action = self.stack2.step(obs)
            if self.stack2.episode_ended:
                self.phase = 2
                self.episode_ended = True
            return action
        # Phase 2: done
        else:
            return None


# Drawer stacking policy wrapper
class MotionPlannerPolicyStackDrawerWrapper(Policy):
    def __init__(self):
        self.impl = MotionPlannerPolicyStackDrawer()

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Drawer stacking three cubes using two sequential drawer stack policies
class MotionPlannerPolicyStackDrawerThreeWrapper(Policy):
    def __init__(self):
        self.stack1 = MotionPlannerPolicyStackDrawer()
        self.stack2 = MotionPlannerPolicyStackDrawer()
        self.phase = 0  # 0: first stack, 1: second stack, 2: done
        self.episode_ended = False

    def reset(self):
        self.stack1.reset()
        self.stack2.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None

        # Phase 0: stack first two cubes
        if self.phase == 0:
            action = self.stack1.step(obs)
            if self.stack1.episode_ended:
                # Prepare for second stacking: adjust stack2's placement height
                if (
                    hasattr(self.stack1, "stack_location")
                    and self.stack1.stack_location is not None
                ):
                    cube_height = self.stack1.STACK_HEIGHT_OFFSET
                    self.stack2.STACK_HEIGHT_OFFSET = (
                        1.8 * cube_height
                    )  # Stack third cube on top of second
                self.phase = 1
                self.stack2.reset()  # Ensure stack2 is ready
            return action
        # Phase 1: stack third cube on top
        elif self.phase == 1:
            action = self.stack2.step(obs)
            if self.stack2.episode_ended:
                self.phase = 2
                self.episode_ended = True
            return action
        # Phase 2: done
        else:
            return None


# Cupboard stacking policy wrapper
class MotionPlannerPolicyStackCupboardWrapper(Policy):
    def __init__(self):
        self.impl = MotionPlannerPolicyStackCupboard()

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Cupboard stacking three cubes using two sequential cupboard stack policies
class MotionPlannerPolicyStackCupboardThreeWrapper(Policy):
    def __init__(self):
        self.stack1 = MotionPlannerPolicyStackCupboard()
        self.stack2 = MotionPlannerPolicyStackCupboard()
        self.phase = 0  # 0: first stack, 1: second stack, 2: done
        self.episode_ended = False

    def reset(self):
        self.stack1.reset()
        self.stack2.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None

        # Phase 0: stack first two cubes
        if self.phase == 0:
            action = self.stack1.step(obs)
            if self.stack1.episode_ended:
                # Prepare for second stacking: adjust stack2's placement height
                if (
                    hasattr(self.stack1, "stack_location")
                    and self.stack1.stack_location is not None
                ):
                    cube_height = self.stack1.STACK_HEIGHT_OFFSET
                    self.stack2.STACK_HEIGHT_OFFSET = (
                        1.8 * cube_height
                    )  # Stack third cube on top of second
                self.phase = 1
                self.stack2.reset()  # Ensure stack2 is ready
            return action
        # Phase 1: stack third cube on top
        elif self.phase == 1:
            action = self.stack2.step(obs)
            if self.stack2.episode_ended:
                self.phase = 2
                self.episode_ended = True
            return action
        # Phase 2: done
        else:
            return None


# Motion planner policy from mp_policy.py (agent)
class MotionPlannerPolicyMPWrapper(Policy):
    def __init__(self, custom_grasp=False):
        self.impl = MotionPlannerPolicyMP(custom_grasp=custom_grasp)
        self.impl.PLACEMENT_X_OFFSET = 0.1
        self.impl.PLACEMENT_Y_OFFSET = 0.1
        self.impl.PLACEMENT_Z_OFFSET = 0.2
        self.impl.target_location = np.array([0, 0, 0.5])

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Motion planner policy for cupboard environment
class MotionPlannerPolicyMPCupboardWrapper(Policy):
    def __init__(self, custom_grasp=False):
        self.impl = MotionPlannerPolicyMP(cupboard_mode=True, custom_grasp=custom_grasp)
        self.impl.PLACEMENT_X_OFFSET = 0.1
        self.impl.PLACEMENT_Y_OFFSET = 0.1
        self.impl.PLACEMENT_Z_OFFSET = 0.5
        self.impl.target_location = np.array([0.8, 0, 0.5])

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Motion planner policy for cabinet environment
class MotionPlannerPolicyMPCabinetWrapper(Policy):
    def __init__(self, custom_grasp=False):
        # Use cupboard_mode=True since cabinet environment is similar to cupboard
        self.impl = MotionPlannerPolicyCabinetMP(custom_grasp=custom_grasp)
        self.impl.PLACEMENT_X_OFFSET = 0.6  # Distance to cabinet
        self.impl.PLACEMENT_Y_OFFSET = 0.0  # Center alignment
        self.impl.PLACEMENT_Z_OFFSET = 0.25  # Cabinet shelf height
        self.impl.target_location = np.array([0, -0.1, 0.25])  # Cabinet position

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Motion planner policy for cabinet environment with two-phase execution
class MotionPlannerPolicyMPCabinetTwoPhaseWrapper(Policy):
    def __init__(self, custom_grasp=False, mp_configs=None):
        """
        Args:
            custom_grasp (bool): Whether to use custom grasp
            mp_configs (list): List of dicts, each specifying the class and kwargs for each phase, e.g.
                [
                    {'class': MotionPlannerPolicyCabinetMP, 'kwargs': {'custom_grasp': custom_grasp, 'open_left_cabinet': True}},
                    {'class': MotionPlannerPolicyCabinetMP_1, 'kwargs': {'custom_grasp': False, 'open_left_cabinet': True}},
                    ...
                ]
        """

        # First phase: MotionPlannerPolicyCabinetMP
        self.mp1 = MotionPlannerPolicyCabinetMP(
            custom_grasp=custom_grasp, open_left_cabinet=True
        )

        # Second phase: MotionPlannerPolicyCabinetMP_1
        self.mp2 = MotionPlannerPolicyCabinetMP_1(
            custom_grasp=False, open_left_cabinet=True
        )

        # First phase: MotionPlannerPolicyCabinetMP
        # self.mp3 = MotionPlannerPolicyCabinetMP(custom_grasp=custom_grasp, open_left_cabinet=False)

        # # Second phase: MotionPlannerPolicyCabinetMP_1
        # self.mp4 = MotionPlannerPolicyCabinetMP_1(custom_grasp=False, open_left_cabinet=False)

        # # First phase: MotionPlannerPolicyCabinetMP
        # self.mp5 = CloseCabinetPolicy(custom_grasp=False, close_left_cabinet=True)

        # # # Second phase: MotionPlannerPolicyCabinetMP_1
        # self.mp6 = CloseCabinetPolicy(custom_grasp=False, close_left_cabinet=False)

        self.mp3 = MotionPlannerPolicyMP(cupboard_mode=True, low_grasp=True)
        self.mp3.target_location = np.array([0.75, -0.15, 0.12])  # Center position
        self.mp3.PICK_APPROACH_HEIGHT_OFFSET = 0.05
        self.mp3.PLACE_APPROACH_HEIGHT_OFFSET = 0.05

        self.mp4 = MotionPlannerPolicyMP(cupboard_mode=True, low_grasp=True)
        self.mp4.target_location = np.array([0.75, -0.3, 0.12])  # Left position
        self.mp4.PICK_APPROACH_HEIGHT_OFFSET = 0.05
        self.mp4.PLACE_APPROACH_HEIGHT_OFFSET = 0.05

        self.mps = [self.mp1, self.mp2, self.mp3, self.mp4]

        # self.mps = [self.mp1, self.mp2, self.mp3, self.mp4, self.mp5, self.mp6]
        self.phase = 0
        self.episode_ended = False

    def reset(self):
        for mp in self.mps:
            mp.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None
        if self.phase < len(self.mps):
            action = self.mps[self.phase].step(obs)
            if getattr(self.mps[self.phase], "episode_ended", False):
                self.phase += 1
                if self.phase < len(self.mps):
                    self.mps[self.phase].reset()
                    print(f"Phase {self.phase} completed, starting next phase")
                else:
                    self.episode_ended = True
                    print(f"All phases completed, episode ended")
            return action
        else:
            return None


# Motion planner policy for three sequential placements
class MotionPlannerPolicyMPThreeWrapper(Policy):
    def __init__(self, custom_grasp=False):
        self.mp1 = MotionPlannerPolicyMP(custom_grasp=custom_grasp)
        self.mp1.target_location = np.array([0.8, 0, 0.2])
        self.mp2 = MotionPlannerPolicyMP(custom_grasp=custom_grasp)
        self.mp2.target_location = np.array([0.8, -0.1, 0.2])
        self.mp3 = MotionPlannerPolicyMP(custom_grasp=custom_grasp)
        self.mp3.target_location = np.array([0.8, 0.1, 0.2])
        self.phase = 0
        self.episode_ended = False

    def reset(self):
        self.mp1.reset()
        self.mp2.reset()
        self.mp3.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None
        if self.phase == 0:
            action = self.mp1.step(obs)
            if getattr(self.mp1, "episode_ended", False):
                self.phase = 1
                self.mp2.reset()
            return action
        elif self.phase == 1:
            action = self.mp2.step(obs)
            if getattr(self.mp2, "episode_ended", False):
                self.phase = 2
                self.mp3.reset()
            return action
        elif self.phase == 2:
            action = self.mp3.step(obs)
            if getattr(self.mp3, "episode_ended", False):
                self.phase = 3
                self.episode_ended = True
            return action
        else:
            return None


# Custom grasp policy wrapper for experimentation
class MotionPlannerPolicyCustomGraspWrapper(Policy):
    def __init__(self):
        self.impl = MotionPlannerPolicyMP(cupboard_mode=True, custom_grasp=True)
        # This wrapper is designed to work with cupboard_scene_objects_inside.xml
        # where objects are already placed inside the cupboard
        # Custom grasping parameters are now set automatically in the MP policy
        # Placement parameters
        self.impl.PLACEMENT_X_OFFSET = 0.1
        self.impl.PLACEMENT_Y_OFFSET = 0.1
        self.impl.PLACEMENT_Z_OFFSET = 0.5
        self.impl.GRASP_SUCCESS_THRESHOLD = 0.75
        self.impl.PICK_LOWER_DIST = 0.09
        self.impl.PICK_LIFT_DIST = 0.18
        self.impl.target_location = np.array([0.8, 0, 0.5])
        print("Custom grasp policy initialized with experimental parameters")
        print(
            "Designed for cupboard_scene_objects_inside.xml (objects already in cupboard)"
        )

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Custom grasp policy wrapper for three sequential pick-place actions in cupboard environment
class MotionPlannerPolicyCustomGraspThreeWrapper(Policy):
    def __init__(self):
        # Create three separate motion planner instances for sequential actions
        self.mp1 = MotionPlannerPolicyMP(cupboard_mode=True, custom_grasp=True)
        self.mp1.GRASP_SUCCESS_THRESHOLD = 0.75
        self.mp1.PICK_LOWER_DIST = 0.09
        self.mp1.PICK_LIFT_DIST = 0.18
        self.mp1.target_location = np.array([0.8, 0.08, 0.38])  # Center position

        self.mp2 = MotionPlannerPolicyMP(cupboard_mode=True, custom_grasp=True)
        self.mp2.GRASP_SUCCESS_THRESHOLD = 0.75
        self.mp2.PICK_LOWER_DIST = 0.09
        self.mp2.PICK_LIFT_DIST = 0.18
        self.mp2.target_location = np.array([0.8, -0.08, 0.38])  # Left position

        self.mp3 = MotionPlannerPolicyMP(cupboard_mode=True, custom_grasp=True)
        self.mp3.GRASP_SUCCESS_THRESHOLD = 0.75
        self.mp3.PICK_LOWER_DIST = 0.09
        self.mp3.PICK_LIFT_DIST = 0.18
        self.mp3.target_location = np.array([0.73, 0, 0.38])  # Right position

        self.phase = (
            0  # 0: first pick-place, 1: second pick-place, 2: third pick-place, 3: done
        )
        self.episode_ended = False

        print("Custom grasp three policy initialized with experimental parameters")
        print(
            "Will execute three sequential pick-place actions in cupboard environment"
        )
        print("Target locations: center, left, right")

    def reset(self):
        self.mp1.reset()
        self.mp2.reset()
        self.mp3.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None

        # Phase 0: First pick-place action (center)
        if self.phase == 0:
            action = self.mp1.step(obs)
            if getattr(self.mp1, "episode_ended", False):
                print("First pick-place action completed, moving to second")
                self.phase = 1
                self.mp2.reset()
            return action

        # Phase 1: Second pick-place action (left)
        elif self.phase == 1:
            action = self.mp2.step(obs)
            if getattr(self.mp2, "episode_ended", False):
                print("Second pick-place action completed, moving to third")
                self.phase = 2
                self.mp3.reset()
            return action

        # Phase 2: Third pick-place action (right)
        elif self.phase == 2:
            action = self.mp3.step(obs)
            if getattr(self.mp3, "episode_ended", False):
                print("Third pick-place action completed, all tasks done")
                self.phase = 3
                self.episode_ended = True
            return action

        # Phase 3: done
        else:
            return None


# Motion planner policy for N sequential pick-place actions in cupboard environment
class MotionPlannerPolicyMPNCupboardWrapper(Policy):
    def __init__(self, target_locations=None, custom_grasp=False, grasp_params=None):
        """Initialize MotionPlannerPolicyMPNCupboardWrapper for N objects.

        Args:
            target_locations (list): List of target locations as numpy arrays [x, y, z]
                                   If None, uses default 3 locations
            custom_grasp (bool): Enable custom grasping parameters
            grasp_params (dict): Optional custom grasping parameters
        """
        # Default target locations if none provided
        if target_locations is None:
            target_locations = [
                np.array([0.8, 0.08, 0.38]),  # Center position
                np.array([0.8, -0.08, 0.38]),  # Left position
                np.array([0.73, 0, 0.38]),  # Right position
            ]

        self.target_locations = target_locations
        self.num_objects = len(target_locations)

        # Default grasping parameters
        default_grasp_params = {
            "GRASP_SUCCESS_THRESHOLD": 0.7,
            "PICK_LOWER_DIST": 0.09,
            "PICK_LIFT_DIST": 0.18,
        }

        # Override with custom parameters if provided
        if grasp_params:
            default_grasp_params.update(grasp_params)

        # Create motion planner instances for each object
        self.motion_planners = []
        for i, target_loc in enumerate(target_locations):
            mp = MotionPlannerPolicyMP(cupboard_mode=True, custom_grasp=True)
            mp.GRASP_SUCCESS_THRESHOLD = default_grasp_params["GRASP_SUCCESS_THRESHOLD"]
            mp.PICK_LOWER_DIST = default_grasp_params["PICK_LOWER_DIST"]
            mp.PICK_LIFT_DIST = default_grasp_params["PICK_LIFT_DIST"]
            mp.target_location = target_loc
            self.motion_planners.append(mp)

        self.current_phase = 0  # Current object being processed
        self.episode_ended = False

        print(
            f"MotionPlannerPolicyMPNCupboardWrapper initialized for {self.num_objects} objects"
        )
        print(
            f"Target locations: {[f'[{loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f}]' for loc in target_locations]}"
        )
        print(f"Custom grasp: {custom_grasp}")
        if grasp_params:
            print(f"Custom grasp parameters: {grasp_params}")

    def reset(self):
        """Reset all motion planners and episode state."""
        for mp in self.motion_planners:
            mp.reset()
        self.current_phase = 0
        self.episode_ended = False
        print(
            f"Reset MotionPlannerPolicyMPNCupboardWrapper - ready to process {self.num_objects} objects"
        )

    def step(self, obs):
        """Execute the current phase of the sequential pick-place task."""
        if self.episode_ended:
            return None

        # Check if we've completed all phases
        if self.current_phase >= self.num_objects:
            self.episode_ended = True
            print(f"All {self.num_objects} pick-place actions completed!")
            return None

        # Execute current motion planner
        current_mp = self.motion_planners[self.current_phase]
        action = current_mp.step(obs)

        # Check if current phase is complete
        if getattr(current_mp, "episode_ended", False):
            print(f"Phase {self.current_phase + 1}/{self.num_objects} completed")
            self.current_phase += 1

            # Reset next motion planner if there are more phases
            if self.current_phase < self.num_objects:
                print(f"Moving to phase {self.current_phase + 1}/{self.num_objects}")
                self.motion_planners[self.current_phase].reset()
            else:
                print("All phases completed!")

        return action


if __name__ == "__main__":
    # WebServer(Queue()).run(); time.sleep(1000)
    # WebXRListener(); time.sleep(1000)
    from constants import POLICY_CONTROL_PERIOD

    obs = {
        "base_pose": np.zeros(3),
        "arm_pos": np.zeros(3),
        "arm_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "gripper_pos": np.zeros(1),
        "base_image": np.zeros((640, 360, 3)),
        "wrist_image": np.zeros((640, 480, 3)),
    }
    policy = TeleopPolicy()
    # policy = RemotePolicy()
    while True:
        policy.reset()
        for _ in range(100):
            print(policy.step(obs))
            time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
