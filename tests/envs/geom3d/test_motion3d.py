"""Tests for motion3d.py."""

from prbench.envs.geom3d.motion3d import Motion3DAction, Motion3DEnv, Motion3DState


def test_motion3d_env():
    """Tests for basic methods in motion3D env."""

    env = Motion3DEnv(use_gui=False)  # set use_gui=True to debug
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, Motion3DState)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, Motion3DAction)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env.physics_client_id)
