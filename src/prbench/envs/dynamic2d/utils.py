"""Utilities for Dynamic2D PyMunk-based environments."""

from typing import Any

import math
import numpy as np
import pymunk
from gymnasium.spaces import Box
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from pymunk.vec2d import Vec2d
from pymunk.examples.shapes_for_draw_demos import fill_space
from prpl_utils.utils import fig2data
import pymunk.matplotlib_util

from prbench.envs.geom2d.structs import SE2Pose

# Collision types from the basic_pymunk.py script
STATIC_COLLISION_TYPE = 0
DYNAMIC_COLLISION_TYPE = 1
ROBOT_COLLISION_TYPE = 2
HELD_OBJECT_COLLISION_TYPE = 3


class FingeredRobotActionSpace(Box):
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
        if self._base_body is None:
            return SE2Pose(x=0.0, y=0.0, theta=0.0)
        return SE2Pose(
            x=self._base_body.position.x,
            y=self._base_body.position.y,
            theta=self._base_body.angle,
        )
    
    @property
    def base_vel(self) -> tuple[Vec2d, float]:
        """Get the base linear and angular velocity."""
        if self._base_body is None:
            return Vec2d(0.0, 0.0), 0.0
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
        return self._gripper_gap

    @property
    def curr_arm_length(self) -> float:
        """Get the current arm length."""
        return self._arm_length

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
        rel_Ldot_curr = (Ldot_curr - base_vel_curr).rotated(-base_ang_curr).x  # R^T * v_gripper_base
        a_L = kp_arm * (tgt_arm - L_curr) + kv_arm * (0.0 - rel_Ldot_curr)

        # Integrate prismatic rate
        rel_Ldot_next = rel_Ldot_curr + a_L * dt
        v_gripper_base = base_vel + Vec2d(rel_Ldot_next, 0.0).rotated(base_ang_curr)

        # === 3) Gripper-base world velocity = rigid motion + prismatic contribution ===
        # Use *updated* base_vel & base_ang_vel for consistency in this control step
        # v_rigid = base_vel + self._cross2d(base_ang_vel, r_w)
        # v_prismatic = Vec2d(Ldot * ex_w.x, Ldot * ex_w.y)             # R * [Ldot, 0]
        # v_gripper_base = v_rigid + v_prismatic

        # # === 4) Finger PD along gripper-base y, then map to world ===
        # # Finger moves along +y of gripper-base (same orientation as base here).
        # kp_f = getattr(self, "kp_finger", self.kp_pos)
        # kv_f = getattr(self, "kv_finger", self.kv_pos)
        # gdot_curr = finger_vel_abs_w.dot(ey_w)
        # a_g = kp_f * (tgt_gripper - g_curr) + kv_f * (0.0 - gdot_curr)
        # gdot = gdot_curr + a_g * dt

        # v_finger_world = v_gripper_base + gdot * ey_w       # add relative [0, gdot] in world

        return base_vel, base_ang_vel, v_gripper_base, Vec2d(0.0, 0.0)  # Finger vel not implemented yet

def on_gripper_grasp(arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict[str, Any]) -> bool:
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

def render_state(
    space: pymunk.Space,
    world_min_x: float = 0.0,
    world_max_x: float = 10.0,
    world_min_y: float = 0.0,
    world_max_y: float = 10.0,
    render_dpi: int = 150,
) -> NDArray[np.uint8]:
    """Render a state from the physics space to an image.
    """

    figsize = (
        world_max_x - world_min_x,
        world_max_y - world_min_y,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=render_dpi)
    pad_x = (world_max_x - world_min_x) / 25
    pad_y = (world_max_y - world_min_y) / 25
    ax.set_xlim(world_min_x - pad_x, world_max_x + pad_x)
    ax.set_ylim(world_min_y - pad_y, world_max_y + pad_y)
    ax.axis("off")
    plt.tight_layout()

    captions = fill_space(space, (1,1,0,1))
    for caption in captions:
        x, y = caption[0]
        y = y - 15
        ax.text(x, y, caption[1], fontsize=12)
    o = pymunk.matplotlib_util.DrawOptions(ax)
    space.debug_draw(o)
    img = fig2data(fig)
    plt.close()
    return img

def get_fingered_robot_action_from_gui_input(
    action_space: FingeredRobotActionSpace, gui_input: dict[str, Any]
) -> NDArray[np.float32]:
    """Get the mapping from human inputs to actions, derived from action space."""
    # This will be implemented later - placeholder for now
    del gui_input  # Unused for now
    action = np.zeros(action_space.shape, action_space.dtype)
    return action