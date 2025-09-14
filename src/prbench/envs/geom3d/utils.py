"""Utilities."""

import numpy as np
from gymnasium.spaces import Box
from relational_structs import ObjectCentricState, Object
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.geometry import Pose
from prbench.envs.geom3d.object_types import Geom3DCuboidType


class Geom3DObjectCentricState(ObjectCentricState):
    """A state in the Geom3D environment.
    
    Inherits from ObjectCentricState but adds some conveninent look ups.
    """

    def get_object_from_name(self, name: str) -> Object:
        """Look up an object from its name."""
        matches = [o for o in self if o.name == name]
        assert len(matches) == 1, f"Object '{name}' not found in state"
        return matches[0]

    @property
    def robot(self) -> Object:
        """Assumes there is a unique robot object named "robot"."""
        return self.get_object_from_name("robot")

    @property
    def joint_positions(self) -> JointPositions:
        """The robot joint positions."""
        joint_names = [f"joint_{i}" for i in range(1, 8)]
        return [self.get(self.robot, n) for n in joint_names]
    
    @property
    def grasped_object(self) -> str | None:
        """The name of the currently grasped object, or None if there is none."""
        grasped_objs: list[Object] = []
        for obj in self.get_objects(Geom3DCuboidType):
            if self.get(obj, "grasp_active") > 0.5:
                grasped_objs.append(obj)
        if not grasped_objs:
            return None
        assert len(grasped_objs) == 1, "Multiple objects should not be grasped"
        grasped_obj = grasped_objs[0]
        return grasped_obj.name
    
    @property
    def grasped_object_transform(self) -> Pose | None:
        """The grasped object transform, or None if there is no grasped object."""
        grasped_object = self.get_object_from_name(self.grasped_object)
        if grasped_object is None:
            return None
        x = self.get(grasped_object, "grasp_tf_x")
        y = self.get(grasped_object, "grasp_tf_y")
        z = self.get(grasped_object, "grasp_tf_z")
        qx = self.get(grasped_object, "grasp_tf_qx")
        qy = self.get(grasped_object, "grasp_tf_qy")
        qz = self.get(grasped_object, "grasp_tf_qz")
        qw = self.get(grasped_object, "grasp_tf_qw")
        grasp_tf = Pose((x, y, z), (qx, qy, qz, qw))
        return grasp_tf


class Geom3DRobotActionSpace(Box):
    """An action space for a 7 DOF robot that can open and close its gripper.

    Actions are bounded relative joint positions and open / close.

    The open / close logic is: <-0.5 is close, >0.5 is open, and otherwise no change.
    """

    def __init__(
        self,
        max_magnitude: float = 0.05,
    ) -> None:
        low = np.array([-max_magnitude] * 7 + [-1])
        high = np.array([max_magnitude] * 7 + [-1])
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        return """An action space for a 7 DOF robot that can open and close its gripper.

    Actions are bounded relative joint positions and open / close.

    The open / close logic is: <-0.5 is close, >0.5 is open, and otherwise no change.
"""