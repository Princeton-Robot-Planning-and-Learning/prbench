"""Utilities for Dynamic2D PyMunk-based environments."""

from typing import Any

import math
import numpy as np
import pymunk
from gymnasium.spaces import Box
import matplotlib.pyplot as plt
from relational_structs import (
    Object,
    ObjectCentricState
)
from tomsgeoms2d.structs import Geom2D, Rectangle, Circle
from prpl_utils.utils import fig2data

from numpy.typing import NDArray
from pymunk.vec2d import Vec2d
from prbench.envs.geom2d.structs import SE2Pose, MultiBody2D, Body2D, ZOrder, z_orders_may_collide
from prbench.envs.geom2d.utils import geom2ds_intersect
from prbench.envs.dynamic2d.object_types import (
    RectangleType,
    KinRobotType,
    Dynamic2DType
)

# Collision types from the basic_pymunk.py script
STATIC_COLLISION_TYPE = 0
DYNAMIC_COLLISION_TYPE = 1
ROBOT_COLLISION_TYPE = 2
HELD_OBJECT_COLLISION_TYPE = 3

PURPLE: tuple[float, float, float] = (128 / 255, 0 / 255, 128 / 255)
BLACK: tuple[float, float, float] = (0.1, 0.1, 0.1)

class KinRobotActionSpace(Box):
    """An action space for a fingered robot with gripper control.

    Actions are bounded relative movements of the base, arm extension, and gripper opening/closing.
    """

    def __init__(
        self,
        min_dx: float = -5e-1,
        max_dx: float = 5e-1,
        min_dy: float = -5e-1,
        max_dy: float = 5e-1,
        min_dtheta: float = -np.pi / 16,
        max_dtheta: float = np.pi / 16,
        min_darm: float = -1e-1,
        max_darm: float = 1e-1,
        min_dgripper: float = -0.02,
        max_dgripper: float = 0.02,
    ) -> None:
        low = np.array([min_dx, min_dy, min_dtheta, min_darm, min_dgripper])
        high = np.array([max_dx, max_dy, max_dtheta, max_darm, max_dgripper])
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        features = [
            ("dx", "Change in robot x position (positive is right)"),
            ("dy", "Change in robot y position (positive is up)"),
            ("dtheta", "Change in robot angle in radians (positive is ccw)"),
            ("darm", "Change in robot arm length (positive is out)"),
            ("dgripper", "Change in gripper gap (positive is open)"),
        ]
        md_table_str = (
            "| **Index** | **Feature** | **Description** | **Min** | **Max** |"
        )
        md_table_str += "\n| --- | --- | --- | --- | --- |"
        for idx, (feature, description) in enumerate(features):
            lb = self.low[idx]
            ub = self.high[idx]
            md_table_str += (
                f"\n| {idx} | {feature} | {description} | {lb:.3f} | {ub:.3f} |"
            )
        return f"The entries of an array in this Box space correspond to the following action features:\n{md_table_str}\n"


class KinRobot:
    """Robot implementation using PyMunk physics engine with PD control."""

    def __init__(
        self,
        init_pos: Vec2d = Vec2d(5.0, 5.0),
        base_radius: float = 0.4,
        arm_length_max: float = 0.8,
        gripper_base_width: float = 0.01,
        gripper_base_height: float = 0.1,
        gripper_finger_width: float = 0.1,
        gripper_finger_height: float = 0.01,
        kp_pos: float = 100.0,
        kv_pos: float = 20.0,
        kp_rot: float = 80.0,
        kv_rot: float = 10.0,
    ) -> None:
        # Robot parameters
        self.base_radius = base_radius
        self.gripper_base_width = gripper_base_width
        self.gripper_base_height = gripper_base_height
        self.gripper_finger_width = gripper_finger_width
        self.gripper_finger_height = gripper_finger_height
        self.arm_length_max = arm_length_max
        self.gripper_gap_max = gripper_base_height

        # PD Control parameters
        self.kp_pos = kp_pos
        self.kv_pos = kv_pos
        self.kp_rot = kp_rot
        self.kv_rot = kv_rot

        # Track last robot state
        self._base_position = init_pos
        self._base_angle = 0.0
        self._arm_length = base_radius
        self._gripper_gap = gripper_base_height
        self.held_objects: list[tuple[pymunk.Body, SE2Pose]] = []
        self.is_opening_finger = False
        self.is_closing_finger = False

        # Body and shape references (Protected)
        self.create_base()
        self.create_gripper_base()
        self._left_finger_body, self._left_finger_shape = self.create_finger()
        self._right_finger_body, self._right_finger_shape = self.create_finger(left=False)

    def add_to_space(self, space: pymunk.Space) -> None:
        """Add robot components to the PyMunk space."""
        space.add(self._base_body, self._base_shape)
        space.add(self._gripper_base_body, self._gripper_base_shape)
        space.add(self._left_finger_body, self._left_finger_shape)
        space.add(self._right_finger_body, self._right_finger_shape)

    def create_base(self) -> None:
        """Create the robot base."""
        self._base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._base_shape = pymunk.Circle(self._base_body, self.base_radius)
        self._base_shape.color = (255, 50, 50, 255)
        self._base_shape.friction = 1
        self._base_shape.collision_type = ROBOT_COLLISION_TYPE
        self._base_shape.density = 1.0
        self._base_body.position = self._base_position
        self._base_body.angle = self._base_angle

    @property
    def base_pose(self) -> SE2Pose:
        """Get the base pose as SE2Pose."""
        return SE2Pose(
            x=self._base_body.position.x,
            y=self._base_body.position.y,
            theta=self._base_body.angle,
        )
    
    @property
    def base_vel(self) -> tuple[Vec2d, float]:
        """Get the base linear and angular velocity."""
        return self._base_body.velocity, self._base_body.angular_velocity

    def create_gripper_base(self) -> None:
        """Create the gripper base."""
        half_w = self.gripper_base_width / 2
        half_h = self.gripper_base_height / 2
        vs = [(-half_w, half_h), (-half_w, -half_h), (half_w, -half_h), (half_w, half_h)]
        self._gripper_base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._gripper_base_shape = pymunk.Poly(self._gripper_base_body, vs)
        self._gripper_base_shape.friction = 1
        self._gripper_base_shape.collision_type = ROBOT_COLLISION_TYPE
        self._gripper_base_shape.density = 1.0

        init_rel_pos = SE2Pose(x=self._arm_length, y=0.0, theta=0.0)
        init_pose = self.base_pose * init_rel_pos
        self._gripper_base_body.position = (init_pose.x, init_pose.y)
        self._gripper_base_body.angle = init_pose.theta

    @property
    def gripper_base_pose(self) -> SE2Pose:
        """Get the gripper base pose as SE2Pose."""
        return SE2Pose(
            x=self._gripper_base_body.position.x,
            y=self._gripper_base_body.position.y,
            theta=self._gripper_base_body.angle,
        )

    @property
    def gripper_base_vel(self) -> tuple[Vec2d, float]:
        """Get the gripper base linear and angular velocity."""
        return self._gripper_base_body.velocity, self._gripper_base_body.angular_velocity

    def create_finger(self, left: bool = True) -> tuple[pymunk.Body, pymunk.Shape]:
        """Create a gripper finger."""
        half_w = self.gripper_finger_width / 2
        half_h = self.gripper_finger_height / 2
        vs = [(-half_w, half_h), (-half_w, -half_h), (half_w, -half_h), (half_w, half_h)]
        finger_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        finger_shape = pymunk.Poly(finger_body, vs)
        finger_shape.friction = 1
        finger_shape.density = 1.0
        finger_shape.collision_type = ROBOT_COLLISION_TYPE

        if left:
            init_rel_pos = SE2Pose(
                x=half_w, y=self._gripper_gap -self.gripper_base_height / 2, theta=0.0
            )
        else:
            init_rel_pos = SE2Pose(
                x=half_w, y=-self.gripper_base_height / 2, theta=0.0
            )
        init_pose = self.gripper_base_pose * init_rel_pos
        finger_body.position = (init_pose.x, init_pose.y)
        finger_body.angle = init_pose.theta
        return finger_body, finger_shape

    @property
    def finger_poses(self) -> dict[str, SE2Pose]:
        """Get the left finger pose as SE2Pose."""
        return {
            "left": SE2Pose(
                x=self._left_finger_body.position.x,
                y=self._left_finger_body.position.y,
                theta=self._left_finger_body.angle,
            ),
            "right": SE2Pose(
                x=self._right_finger_body.position.x,
                y=self._right_finger_body.position.y,
                theta=self._right_finger_body.angle,
            ),
        }
    
    @property
    def finger_vels(self) -> dict[str, tuple[Vec2d, float]]:
        """Get the left finger linear and angular velocity."""
        return {
            "left": (self._left_finger_body.velocity, self._left_finger_body.angular_velocity),
            "right": (self._right_finger_body.velocity, self._right_finger_body.angular_velocity),
        }
    @property
    def curr_gripper(self) -> float:
        """Get the current gripper opening."""
        relative_finger_pose = self.gripper_base_pose.inverse * self.finger_poses['left']
        return relative_finger_pose.y + self.gripper_base_height / 2

    @property
    def curr_arm_length(self) -> float:
        """Get the current arm length."""
        relative_pose = self.base_pose.inverse * self.gripper_base_pose
        return relative_pose.x

    @property
    def body_id(self) -> dict[str, int]:
        """Get the base id in pymunk space."""
        return self._base_body.id

    def reset_positions(self,
                        base_x: float,
                        base_y: float,
                        base_theta: float,
                        arm_length: float,
                        gripper_gap: float) -> None:
        """Reset robot to specified positions."""
        self._base_body.position = (base_x, base_y)
        self._base_body.angle = base_theta
        
        base_to_gripper = SE2Pose(x=arm_length, y=0.0, theta=0.0)
        gripper_pose = self.base_pose * base_to_gripper
        self._gripper_base_body.position = (gripper_pose.x, gripper_pose.y)
        self._gripper_base_body.angle = gripper_pose.theta

        gripper_to_left_finger = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=gripper_gap - self.gripper_base_height / 2,
            theta=0.0,
        )
        left_finger_pose = gripper_pose * gripper_to_left_finger
        self._left_finger_body.position = (left_finger_pose.x, left_finger_pose.y)
        self._left_finger_body.angle = left_finger_pose.theta
        gripper_to_right_finger = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=-self.gripper_base_height / 2,
            theta=0.0,
        )
        right_finger_pose = gripper_pose * gripper_to_right_finger
        self._right_finger_body.position = (right_finger_pose.x, right_finger_pose.y)
        self._right_finger_body.angle = right_finger_pose.theta

        # Update last state
        self.update_last_state()

    def reset_last_state(self) -> None:
        """Reset to last state when collide with static objects."""
        self._base_body.position = self._base_position
        self._base_body.angle = self._base_angle
        
        gripper_base_rel_pos = SE2Pose(x=self._arm_length, y=0.0, theta=0.0)
        gripper_base_pose = self.base_pose * gripper_base_rel_pos
        self._gripper_base_body.position = (gripper_base_pose.x, gripper_base_pose.y)
        self._gripper_base_body.angle = gripper_base_pose.theta
        
        left_finger_rel_pos = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=self._gripper_gap - self.gripper_base_height / 2,
            theta=0.0,
        )
        left_finger_pose = gripper_base_pose * left_finger_rel_pos
        self._left_finger_body.position = (left_finger_pose.x, left_finger_pose.y)
        self._left_finger_body.angle = left_finger_pose.theta
        
        right_finger_rel_pos = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=-self.gripper_base_height / 2,
            theta=0.0,
        )
        right_finger_pose = gripper_base_pose * right_finger_rel_pos
        if self._right_finger_body:
            self._right_finger_body.position = (right_finger_pose.x, right_finger_pose.y)
            self._right_finger_body.angle = right_finger_pose.theta

        # Update held objects
        for obj, relative_pose in self.held_objects:
            new_obj_pose = gripper_base_pose * relative_pose
            obj.position = (new_obj_pose.x, new_obj_pose.y)
            obj.angle = new_obj_pose.theta

    def update_last_state(self) -> None:
        """Update the last state tracking variables."""
        self._base_position = Vec2d(self._base_body.position.x, self._base_body.position.y)
        self._base_angle = self._base_body.angle
        
        relative_pose = self.base_pose.inverse * self.gripper_base_pose
        self._arm_length = relative_pose.x
        relative_finger_pose = self.gripper_base_pose.inverse * self.finger_poses['left']
        self._gripper_gap = relative_finger_pose.y + self.gripper_base_height / 2

    def update(
        self,
        base_vel: Vec2d,
        base_ang_vel: float,
        gripper_base_vel: Vec2d,
        finger_vel_l: Vec2d
    ) -> None:
        """Update the body velocities using PD control."""
        # Update robot last state
        self.update_last_state()
        # Update velocities
        self._base_body.velocity = base_vel
        self._base_body.angular_velocity = base_ang_vel
        # Calculate target gripper base
        # It only has relative translational velocity
        self._gripper_base_body.velocity = gripper_base_vel
        self._gripper_base_body.angular_velocity = base_ang_vel
        # Fingers
        self._left_finger_body.velocity = finger_vel_l
        self._left_finger_body.angular_velocity = base_ang_vel
        # Right finger has the same vel as gripper base always
        self._right_finger_body.velocity = self._gripper_base_body.velocity
        self._right_finger_body.angular_velocity = base_ang_vel

        # Update held objects - they have the same velocity as gripper base
        for obj, _ in self.held_objects:
            obj.velocity = self._gripper_base_body.velocity
            obj.angular_velocity = self._gripper_base_body.angular_velocity

        # Update finger opening/closing status
        rel_vel = finger_vel_l - gripper_base_vel
        rel_vel.rotated(-self._gripper_base_body.angle)
        if rel_vel.y > 0.01:
            self.is_opening_finger = True
            self.is_closing_finger = False
        elif rel_vel.y < -0.01:
            self.is_closing_finger = True
            self.is_opening_finger = False
        else:
            self.is_closing_finger = False
            self.is_opening_finger = False

    def drop_held_objects(self, space: pymunk.Space) -> None:
        """Drop held objects if gripper is opening."""
        if self.is_opening_finger and len(self.held_objects) > 0:
            for obj, _ in self.held_objects:
                self.del_from_hand_space((obj, obj.shapes[0]), space)
            self.held_objects = []

    def is_grasping(self, contact_point_set: pymunk.ContactPointSet, tgt_body: pymunk.Body) -> bool:
        """Check if robot is grasping a target body."""
        # Checker 0: If robot is closing gripper
        if not self.is_closing_finger:
            return False
        # Checker 1: If contact normal is roughly perpendicular 
        # to gripper_base_body base
        normal = contact_point_set.normal
        if not self._gripper_base_body:
            return False
        dtheta = abs(self._gripper_base_body.angle - normal.angle)
        dtheta = min(dtheta, 2 * np.pi - dtheta)
        theta_ok = abs(dtheta - np.pi / 2) < 0.1
        if not theta_ok:
            return False
        # Checker 2: If the body is within the grasping area
        p_a = SE2Pose(x=tgt_body.position.x, y=tgt_body.position.y, theta=0.0)
        rel_a = self.gripper_base_pose.inverse * p_a
        if (abs(rel_a.y) < self.gripper_base_height / 4) and (
            abs(rel_a.x) < self.gripper_finger_width
        ):
            print("Grasped!")
            return True
        return False

    def add_to_hand(self, obj: tuple[pymunk.Body, pymunk.Shape]) -> None:
        """Add an object to the robot's hand."""
        obj_body, _ = obj
        obj_pose = SE2Pose(x=obj_body.position.x, y=obj_body.position.y, theta=obj_body.angle)
        gripper_base_pose = SE2Pose(
            x=self._gripper_base_body.position.x,
            y=self._gripper_base_body.position.y,
            theta=self._gripper_base_body.angle,
        )
        relative_obj_pose = gripper_base_pose.inverse * obj_pose
        self.held_objects.append((obj_body, relative_obj_pose))

    def del_from_hand_space(self, obj: tuple[pymunk.Body, pymunk.Shape], space: pymunk.Space) -> None:
        """Remove an object from hand and make it dynamic."""
        mass = 1.0
        kinematic_body, kinematic_shape = obj
        points = kinematic_shape.get_vertices()
        moment = pymunk.moment_for_poly(mass, points, (0, 0))
        dynamic_body = pymunk.Body(mass, moment)
        dynamic_body.position = kinematic_body.position
        dynamic_body.velocity = kinematic_body.velocity  # Preserve velocity
        dynamic_body.angular_velocity = kinematic_body.angular_velocity
        dynamic_body.angle = kinematic_body.angle
        shape = pymunk.Poly(dynamic_body, points)
        shape.friction = 1
        shape.density = 1.0
        shape.collision_type = DYNAMIC_COLLISION_TYPE
        space.add(dynamic_body, shape)
        space.remove(kinematic_body, kinematic_shape)


class PDController:
    """A simple PD controller for the robot."""

    def __init__(
        self,
        kp_pos: float = 100.0,
        kv_pos: float = 20.0,
        kp_rot: float = 500.0,
        kv_rot: float = 50.0,
    ) -> None:
        self.kp_pos = kp_pos
        self.kv_pos = kv_pos
        self.kp_rot = kp_rot
        self.kv_rot = kv_rot

    def _R(self, theta: float) -> tuple[Vec2d, Vec2d]:
        """Return world-frame unit basis of the base frame: (ex_w, ey_w)."""
        c, s = math.cos(theta), math.sin(theta)
        ex_w = Vec2d(c, s)     # base x-axis in world
        ey_w = Vec2d(-s, c)    # base y-axis in world
        return ex_w, ey_w

    def _cross2d(self, omega: float, v: Vec2d) -> Vec2d:
        """2D angular × vector = perpendicular vector scaled by omega."""
        # [0,0,omega] x [vx,vy,0] = [0,0, omega*vx*k?] -> in 2D gives (-omega*vy, omega*vx)
        return Vec2d(-omega * v.y, omega * v.x)

    def compute_control(
        self,
        robot: KinRobot,
        tgt_x: float,
        tgt_y: float,
        tgt_theta: float,
        tgt_arm: float,       # target arm length L*
        tgt_gripper: float,   # target finger opening g*
        dt: float,
    ) -> tuple[Vec2d, float, Vec2d, Vec2d]:
        """Compute base vel, base ang vel, gripper-base vel (world), finger vel (world)."""
        # === 0) Read current state ===
        base_pos_curr = Vec2d(robot.base_pose.x, robot.base_pose.y)
        base_vel_curr = robot.base_vel[0]          # Vec2d(vx, vy) in world
        base_ang_curr = robot.base_pose.theta
        base_ang_vel_curr = robot.base_vel[1]      # scalar omega
        base_rot_omega_vec = Vec2d(math.cos(base_ang_curr + math.pi / 2),
                               math.sin(base_ang_curr + math.pi / 2))

        L_curr = robot.curr_arm_length             # current arm length (scalar)
        Ldot_curr = robot.gripper_base_vel[0]

        # If available (recommended), provide current gripper opening and its rate:
        g_curr = robot.curr_gripper                 # opening distance
        finger_vel_abs_w = robot.finger_vels['left'][0]  # Vec2d

        # === 1) Base PD (same as yours, but keep structure tidy) ===
        base_pos_tgt = Vec2d(tgt_x, tgt_y)
        a_base_lin = self.kp_pos * (base_pos_tgt - base_pos_curr) \
                + self.kv_pos * (Vec2d(0, 0) - base_vel_curr)
        base_vel = base_vel_curr + a_base_lin * dt

        a_base_ang = self.kp_rot * (tgt_theta - base_ang_curr) \
                + self.kv_rot * (0.0 - base_ang_vel_curr)
        base_ang_vel = base_ang_vel_curr + a_base_ang * dt

        # === 2) Arm prismatic rate via PD on length in the base frame ===
        # PD on arm length
        kp_arm = getattr(self, "kp_arm", self.kp_pos)
        kv_arm = getattr(self, "kv_arm", self.kv_pos)
        arm_center_omega_vec = base_rot_omega_vec * L_curr * base_ang_vel_curr  # omega x r
        # Extract prismatic vel from a moving base
        rel_Ldot_curr = (Ldot_curr - base_vel_curr - \
                         arm_center_omega_vec).rotated(-base_ang_curr).x  # R^T * v_gripper_base
        a_L = kp_arm * (tgt_arm - L_curr) + kv_arm * (0.0 - rel_Ldot_curr)

        # Integrate prismatic rate
        rel_Ldot_next = rel_Ldot_curr + a_L * dt
        # Note: We need to use the *next base_ang_vel* to compute the 
        # world-frame gripper-base velocity
        v_gripper_base = base_vel + \
            base_rot_omega_vec * L_curr * base_ang_vel + \
            Vec2d(rel_Ldot_next, 0.0).rotated(base_ang_curr)

        # === 3) Gripper-base world velocity = rigid motion + prismatic contribution ===
        # Use *updated* base_vel & base_ang_vel for consistency in this control step
        kp_finger = getattr(self, "kp_finger", self.kp_pos)
        kv_finger = getattr(self, "kv_finger", self.kv_pos)
        gripper_centr = Vec2d(robot.finger_poses['left'].x, robot.finger_poses['left'].y)
        # Extract the rotate omega x r contribution
        relative_pos = gripper_centr - base_pos_curr
        finger_rot_omega_vec_base = relative_pos.normalized().rotated(math.pi / 2)
        finger_rot_omega_vec = finger_rot_omega_vec_base * relative_pos.length \
            * base_ang_vel_curr
        # We only care about y-dir as finger only moves in y in the base frame
        rel_gdot_curr = (finger_vel_abs_w - base_vel_curr - \
                            finger_rot_omega_vec).rotated(-base_ang_curr).y
        a_g = kp_finger * (tgt_gripper - g_curr) + kv_finger * (0.0 - rel_gdot_curr)
        rel_gdot_next = rel_gdot_curr + a_g * dt
        # Finger world velocity, similar to gripper-base vel but with the additional 
        # primistamic part in y-dir
        finger_vel_l = base_vel + \
            finger_rot_omega_vec_base * relative_pos.length * base_ang_vel + \
            Vec2d(rel_Ldot_next, rel_gdot_next).rotated(base_ang_curr)

        return base_vel, base_ang_vel, v_gripper_base, finger_vel_l

def rectangle_object_to_geom(
    state: ObjectCentricState,
    rect_obj: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> Rectangle:
    """Helper to extract a rectangle for an object."""
    assert rect_obj.is_instance(RectangleType)
    multibody = object_to_multibody2d(rect_obj, state, static_object_cache)
    assert len(multibody.bodies) == 1
    geom = multibody.bodies[0].geom
    assert isinstance(geom, Rectangle)
    return geom

def object_to_multibody2d(
    obj: Object,
    state: ObjectCentricState,
    static_object_cache: dict[Object, MultiBody2D],
) -> MultiBody2D:
    """Create a Body2D instance for objects of standard geom types.
        This is borrowed from Geom2D
    """
    if obj.is_instance(KinRobotType):
        return _robot_to_multibody2d(obj, state)
    assert obj.is_instance(Dynamic2DType)
    is_static = state.get(obj, "static") > 0.5
    if is_static and obj in static_object_cache:
        return static_object_cache[obj]
    geom: Geom2D  # rectangle or circle
    if obj.is_instance(RectangleType):
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        width = state.get(obj, "width")
        height = state.get(obj, "height")
        theta = state.get(obj, "theta")
        geom = Rectangle.from_center(x, y, width, height, theta)
        z_order = ZOrder(int(state.get(obj, "z_order")))
        rendering_kwargs = {
            "facecolor": (
                state.get(obj, "color_r"),
                state.get(obj, "color_g"),
                state.get(obj, "color_b"),
            ),
            "edgecolor": BLACK,
        }
        body = Body2D(geom, z_order, rendering_kwargs)
        multibody = MultiBody2D(obj.name, [body])
    return multibody

def _robot_to_multibody2d(obj: Object, state: ObjectCentricState) -> MultiBody2D:
    """Helper for object_to_multibody2d()."""
    assert obj.is_instance(KinRobotType)
    bodies: list[Body2D] = []

    # Base.
    base_x = state.get(obj, "x")
    base_y = state.get(obj, "y")
    base_radius = state.get(obj, "base_radius")
    circ = Circle(
        x=base_x,
        y=base_y,
        radius=base_radius,
    )
    z_order = ZOrder.ALL
    rendering_kwargs = {"facecolor": PURPLE, "edgecolor": BLACK}
    base = Body2D(circ, z_order, rendering_kwargs, name="base")
    bodies.append(base)

    # Gripper Base
    theta = state.get(obj, "theta")
    arm_joint = state.get(obj, "arm_joint")
    gripper_base_cx = base_x + np.cos(theta) * arm_joint
    gripper_base_cy = base_y + np.sin(theta) * arm_joint
    gripper_base_height = state.get(obj, "gripper_base_height")
    gripper_base_width = state.get(obj, "gripper_base_width")
    rect = Rectangle.from_center(
        center_x=gripper_base_cx,
        center_y=gripper_base_cy,
        height=gripper_base_height,
        width=gripper_base_width,
        rotation_about_center=theta,
    )
    z_order = ZOrder.SURFACE
    rendering_kwargs = {"facecolor": PURPLE, "edgecolor": BLACK}
    gripper_base = Body2D(rect, z_order, rendering_kwargs, name="gripper_base")
    gripper_base_pose = SE2Pose(
        x=gripper_base_cx,
        y=gripper_base_cy,
        theta=theta,
    )
    bodies.append(gripper_base)

    # Fingers
    relative_dx = state.get(obj, "finger_width") / 2
    relative_dy_r = -gripper_base_height / 2
    relative_dy_l = state.get(obj, "finger_gap") - gripper_base_height / 2
    finger_r_pose = gripper_base_pose * SE2Pose(
        x=relative_dx,
        y=relative_dy_r,
        theta=0.0,
    )
    finger_l_pose = gripper_base_pose * SE2Pose(
        x=relative_dx,
        y=relative_dy_l,
        theta=0.0,
    )
    finger_r = Rectangle.from_center(
        center_x=finger_r_pose.x,
        center_y=finger_r_pose.y,
        height=state.get(obj, "finger_height"),
        width=state.get(obj, "finger_width"),
        rotation_about_center=finger_r_pose.theta,
    )
    finger_l = Rectangle.from_center(
        center_x=finger_l_pose.x,
        center_y=finger_l_pose.y,
        height=state.get(obj, "finger_height"),
        width=state.get(obj, "finger_width"),
        rotation_about_center=finger_l_pose.theta,
    )
    z_order = ZOrder.SURFACE
    rendering_kwargs = {"facecolor": PURPLE, "edgecolor": BLACK}
    finger_l_body = Body2D(finger_r, z_order, rendering_kwargs, name="arm")
    bodies.append(finger_l_body)
    finger_r_body = Body2D(finger_l, z_order, rendering_kwargs, name="arm")
    bodies.append(finger_r_body)

    multibody = MultiBody2D(obj.name, bodies)
    return multibody

def on_gripper_grasp(arbiter: pymunk.Arbiter, 
                     space: pymunk.Space, 
                     data: dict[str, Any]) -> bool:
    """Collision callback for gripper grasping objects."""
    robot = data["robot"]
    print("Gripper Collision detected!")
    dynamic_body = arbiter.bodies[0]
    if robot.is_grasping(arbiter.contact_point_set, dynamic_body):
        # Create a new kinematic object
        kinematic_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        kinematic_body.position = dynamic_body.position
        kinematic_body.angle = dynamic_body.angle
        points = arbiter.shapes[0].get_vertices()
        shape = pymunk.Poly(kinematic_body, points)
        shape.friction = 1
        shape.density = 1.0
        shape.collision_type = HELD_OBJECT_COLLISION_TYPE
        space.add(kinematic_body, shape)
        robot.add_to_hand((kinematic_body, shape))
        # Remove the dynamic body from the space
        space.remove(dynamic_body, arbiter.shapes[0])
        return True
    return False


def on_collision_w_static(arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict[str, Any]) -> bool:
    """Collision callback for robot colliding with static objects."""
    robot = data["robot"]
    del arbiter
    del space
    print("Static Collision detected!")
    robot.reset_last_state()
    return True

def create_walls_from_world_boundaries(
    world_min_x: float,
    world_max_x: float,
    world_min_y: float,
    world_max_y: float,
    min_dx: float,
    max_dx: float,
    min_dy: float,
    max_dy: float,
) -> dict[Object, dict[str, float]]:
    """Create wall objects and feature dicts based on world boundaries.

    Velocities are used to determine how large the walls need to be to avoid the
    possibility that the robot will transport over the wall.
    """
    state_dict: dict[Object, dict[str, float]] = {}
    # Right wall.
    right_wall = Object("right_wall", RectangleType)
    side_wall_height = world_max_y - world_min_y
    state_dict[right_wall] = {
        "x": world_max_x + max_dx,
        "vx": 0.0,
        "y": (world_min_y + world_max_y) / 2,
        "vy": 0.0,
        "width": 2 * max_dx,  # 2x just for safety
        "height": side_wall_height,
        "theta": 0.0,
        "omega": 0.0,
        "static": True,
        "kinematic": False,
        "dynamic": False,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Left wall.
    left_wall = Object("left_wall", RectangleType)
    state_dict[left_wall] = {
        "x": world_min_x + min_dx,
        "vx": 0.0,
        "y": (world_min_y + world_max_y) / 2,
        "vy": 0.0,
        "width": 2 * abs(min_dx),  # 2x just for safety
        "height": side_wall_height,
        "theta": 0.0,
        "omega": 0.0,
        "static": True,
        "kinematic": False,
        "dynamic": False,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Top wall.
    top_wall = Object("top_wall", RectangleType)
    horiz_wall_width = 2 * 2 * abs(min_dx) + world_max_x - world_min_x
    state_dict[top_wall] = {
        "x": (world_min_x + world_max_x) / 2,
        "vx": 0.0,
        "y": world_max_y + max_dy,
        "vy": 0.0,
        "width": horiz_wall_width,
        "height": 2 * max_dy,
        "theta": 0.0,
        "omega": 0.0,
        "static": True,
        "kinematic": False,
        "dynamic": False,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Bottom wall.
    bottom_wall = Object("bottom_wall", RectangleType)
    state_dict[bottom_wall] = {
        "x": (world_min_x + world_max_x) / 2,
        "vx": 0.0,
        "y": world_min_y + min_dy,
        "vy": 0.0,
        "width": horiz_wall_width,
        "height": 2 * abs(min_dy),
        "theta": 0.0,
        "omega": 0.0,
        "static": True,
        "kinematic": False,
        "dynamic": False,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    return state_dict

def state_has_collision(
    state: ObjectCentricState,
    group1: set[Object],
    group2: set[Object],
    static_object_cache: dict[Object, MultiBody2D],
    ignore_z_orders: bool = False,
) -> bool:
    """Check for collisions between any objects in two groups."""
    # Create multibodies once.
    obj_to_multibody = {
        o: object_to_multibody2d(o, state, static_object_cache) for o in state
    }
    # Check pairwise collisions.
    for obj1 in group1:
        for obj2 in group2:
            if obj1 == obj2:
                continue
            multibody1 = obj_to_multibody[obj1]
            multibody2 = obj_to_multibody[obj2]
            for body1 in multibody1.bodies:
                for body2 in multibody2.bodies:
                    if not (
                        ignore_z_orders
                        or z_orders_may_collide(body1.z_order, body2.z_order)
                    ):
                        continue
                    if geom2ds_intersect(body1.geom, body2.geom):
                        return True
    return False

def is_on(
    state: ObjectCentricState,
    top: Object,
    bottom: Object,
    static_object_cache: dict[Object, MultiBody2D],
    tol: float = 0.025,
) -> bool:
    """Checks top object is completely on the bottom one.

    Only rectangles are currently supported.

    Assumes that "up" is positive y.
    """
    top_geom = rectangle_object_to_geom(state, top, static_object_cache)
    bottom_geom = rectangle_object_to_geom(state, bottom, static_object_cache)
    # The bottom-most vertices of top_geom should be contained within the bottom
    # geom when those vertices are offset by tol.
    sorted_vertices = sorted(top_geom.vertices, key=lambda v: v[1])
    for x, y in sorted_vertices[:2]:
        offset_y = y - tol
        if not bottom_geom.contains_point(x, offset_y):
            return False
    return True

def get_fingered_robot_action_from_gui_input(
    action_space: KinRobotActionSpace, gui_input: dict[str, Any]
) -> NDArray[np.float32]:
    """Get the mapping from human inputs to actions, derived from action space."""
    # This will be implemented later - placeholder for now
    del gui_input  # Unused for now
    action = np.zeros(action_space.shape, action_space.dtype)
    return action

def render_state_on_ax(
    state: ObjectCentricState,
    ax: plt.Axes,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
) -> None:
    """Render a state on an existing plt.Axes."""
    if static_object_body_cache is None:
        static_object_body_cache = {}

    # Sort objects by ascending z order, with the robot first.
    def _render_order(obj: Object) -> int:
        if obj.is_instance(KinRobotType):
            return -1
        return int(state.get(obj, "z_order"))

    for obj in sorted(state, key=_render_order):
        body = object_to_multibody2d(obj, state, static_object_body_cache)
        body.plot(ax)


def render_state(
    state: ObjectCentricState,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
    world_min_x: float = 0.0,
    world_max_x: float = 10.0,
    world_min_y: float = 0.0,
    world_max_y: float = 10.0,
    render_dpi: int = 150,
) -> NDArray[np.uint8]:
    """Render a state.

    Useful for viz and debugging.
    """
    if static_object_body_cache is None:
        static_object_body_cache = {}

    figsize = (
        world_max_x - world_min_x,
        world_max_y - world_min_y,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=render_dpi)

    render_state_on_ax(state, ax, static_object_body_cache)

    pad_x = (world_max_x - world_min_x) / 25
    pad_y = (world_max_y - world_min_y) / 25
    ax.set_xlim(world_min_x - pad_x, world_max_x + pad_x)
    ax.set_ylim(world_min_y - pad_y, world_max_y + pad_y)
    ax.axis("off")
    plt.tight_layout()
    img = fig2data(fig)
    plt.close()
    return img
