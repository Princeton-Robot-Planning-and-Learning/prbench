"""This code adds additional object types for 2D environments."""

from geom2drobotenvs.object_types import Geom2DRobotEnvTypeFeatures, Geom2DType
from relational_structs import Type

LObjectType = Type("hook", parent=Geom2DType)
Geom2DRobotEnvTypeFeatures[LObjectType] = Geom2DRobotEnvTypeFeatures[Geom2DType] + [
    "width",
    "length_side1",
    "length_side2",
]
