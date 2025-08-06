"""Spec for 3D motion planning and manipulation environments (TidyBot3D,
etc)."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Motion3DEnvSpec:
    """Spec for 3D motion planning/manipulation environments."""

    # Mobile base parameters (from constants.py)
    h_x: np.ndarray = 0.190150 * np.array(
        [1.0, 1.0, -1.0, -1.0]
    )  # Vehicle center to steer axis (x)
    h_y: np.ndarray = 0.170150 * np.array(
        [-1.0, 1.0, 1.0, -1.0]
    )  # Vehicle center to steer axis (y)
    # Encoder magnet offsets
    encoder_magnet_offsets: list[float] = field(
        default_factory=lambda: [0.0 / 4096, 0.0 / 4096, 0.0 / 4096, 0.0 / 4096]
    )

    # Policy server settings
    policy_server_host: str = "localhost"
    policy_server_port: int = 5555
    policy_control_freq: int = 10
    policy_control_period: float = 1.0 / 10
    policy_image_width: int = 84
    policy_image_height: int = 84
