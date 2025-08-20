"""Tests for motion3d.py."""

from prbench.envs.geom3d.motion3d import Motion3DEnv

def test_motion3d_env():
    """Tests for basic methods in motion3D env."""
    
    env = Motion3DEnv(use_gui=True)  # set use_gui=True to debug
    obs, _ = env.reset()

    # Uncomment to debug.
    import pybullet as p
    while True:
        p.getMouseEvents(env.physics_client_id)
