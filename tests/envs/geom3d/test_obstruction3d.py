"""Tests for motion3d.py."""

import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from pybullet_helpers.geometry import Pose

from prbench.envs.geom3d.obstruction3d import (
    Obstruction3DAction,
    Obstruction3DEnv,
    Obstruction3DState,
)


def test_motion3d_env():
    """Tests for basic methods in motion3D env."""

    env = Obstruction3DEnv(use_gui=False)  # set use_gui=True to debug
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, Obstruction3DState)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, Obstruction3DAction)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env.physics_client_id)
