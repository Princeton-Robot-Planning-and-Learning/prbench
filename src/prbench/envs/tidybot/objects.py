"""Object definitions for TidyBot environments."""

import xml.etree.ElementTree as ET
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


class MujocoObjectState:
    """Class to represent the state of a MujocoObject."""

    def __init__(
        self,
        position: NDArray[np.float64],
        orientation: NDArray[np.float64],
    ) -> None:
        """Initialize a MujocoObjectState.

        Args:
            position: Position as [x, y, z] array
            orientation: Orientation as quaternion [x, y, z, w] array
        """
        self.position = position
        self.orientation = orientation

    def __repr__(self) -> str:
        """Detailed string representation of the object state."""
        return (
            f"MujocoObjectState(position={self.position.tolist()}, "
            f"orientation={self.orientation.tolist()})"
        )


class MujocoObject:
    """Base class for MuJoCo objects with position and orientation control."""

    def __init__(
        self,
        name: str,
        env: Optional[object] = None,
    ) -> None:
        """Initialize a MujocoObject.

        Args:
            name: Name of the object body in the XML
            env: Reference to the environment (needed for position get/set operations)
        """
        self.name = name
        self.joint_name = f"{name}_joint"
        self.env = env

    def get_position(self) -> NDArray[np.float64]:
        """Get the object's current position.

        Returns:
            Position as [x, y, z] array

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to get position")

        pos, _ = self.env.get_joint_pos_quat(self.joint_name)
        return pos

    def get_orientation(self) -> NDArray[np.float64]:
        """Get the object's current orientation.

        Returns:
            Orientation as quaternion [x, y, z, w] array

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to get orientation")

        _, quat = self.env.get_joint_pos_quat(self.joint_name)
        return quat

    def get_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the object's current position and orientation.

        Returns:
            Tuple of (position, quaternion)

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to get pose")

        return self.env.get_joint_pos_quat(self.joint_name)

    def set_position(self, position: Union[list[float], NDArray[np.float64]]) -> None:
        """Set the object's position.

        Args:
            position: New position as [x, y, z]

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to set position")

        # Get current orientation to preserve it
        _, current_quat = self.env.get_joint_pos_quat(self.joint_name)

        # Set new position with current orientation
        self.env.set_joint_pos_quat(self.joint_name, np.array(position), current_quat)

    def set_orientation(
        self, quaternion: Union[list[float], NDArray[np.float64]]
    ) -> None:
        """Set the object's orientation.

        Args:
            quaternion: New orientation as quaternion [x, y, z, w]

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to set orientation")

        # Get current position to preserve it
        current_pos, _ = self.env.get_joint_pos_quat(self.joint_name)

        # Set new orientation with current position
        self.env.set_joint_pos_quat(self.joint_name, current_pos, np.array(quaternion))

    def set_pose(
        self,
        position: Union[list[float], NDArray[np.float64]],
        quaternion: Union[list[float], NDArray[np.float64]],
    ) -> None:
        """Set the object's position and orientation.

        Args:
            position: New position as [x, y, z]
            quaternion: New orientation as quaternion [x, y, z, w]

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to set pose")

        self.env.set_joint_pos_quat(
            self.joint_name, np.array(position), np.array(quaternion)
        )

    def get_state(self) -> MujocoObjectState:
        """Get the object's current state as a MujocoObjectState.

        Returns:
            MujocoObjectState with current position and orientation

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to get state")

        pos, quat = self.env.get_joint_pos_quat(self.joint_name)
        return MujocoObjectState(np.array(pos), np.array(quat))


class Cube(MujocoObject):
    """A cube object for TidyBot environments."""

    def __init__(
        self,
        name: str,
        size: Union[float, list[float]] = 0.02,
        rgba: Union[str, list[float]] = ".5 .7 .5 1",
        mass: float = 0.1,
        env: Optional[object] = None,
    ) -> None:
        """Initialize a Cube object.

        Args:
            name: Name of the cube body in the XML
            size: Size of the cube (either scalar or [x, y, z] dimensions)
            rgba: Color of the cube (either string or [r, g, b, a] values)
            mass: Mass of the cube
            env: Reference to the environment (needed for position get/set operations)
        """
        # Initialize base class
        super().__init__(name, env)

        # Handle size parameter
        if isinstance(size, (int, float)):
            self.size = [size, size, size]
        else:
            self.size = list(size)

        # Handle rgba parameter
        if isinstance(rgba, str):
            self.rgba = rgba
        else:
            self.rgba = " ".join(str(x) for x in rgba)

        self.mass = mass

        # Create the XML element
        self.xml_element = self._create_xml_element()

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this cube.

        Returns:
            ET.Element representing the cube body
        """
        # Create body element
        body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(body, "freejoint", name=self.joint_name)

        # Add geom element with cube properties
        size_str = " ".join(str(x) for x in self.size)
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=size_str,
            rgba=self.rgba,
            mass=str(self.mass),
        )

        return body

    def __str__(self) -> str:
        """String representation of the cube."""
        return (
            f"Cube(name='{self.name}', size={self.size}, "
            f"rgba='{self.rgba}', mass={self.mass})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the cube."""
        return (
            f"Cube(name='{self.name}', joint_name='{self.joint_name}', "
            f"size={self.size}, rgba='{self.rgba}', mass={self.mass})"
        )
