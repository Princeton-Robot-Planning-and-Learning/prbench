"""Tests for prbench.envs.tidybot.policies MotionPlannerPolicyMPWrapper.

Covers initialization hooks and wrapper behavior mirroring the wrapped policy's episode
end state.
"""

import numpy as np

from prbench.envs.tidybot.policies import MotionPlannerPolicyMPWrapper


def make_min_obs():
    """Create a minimal valid observation dictionary for wrapper tests."""
    return {
        "base_pose": np.array([0.0, 0.0, 0.0], dtype=float),
        "arm_pos": np.array([0.0, 0.0, 0.0], dtype=float),
        "arm_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        "gripper_pos": np.array([0.0], dtype=float),
    }


def test_wrapper_reset_and_episode_flag():
    """Reset should clear wrapper flags; step mirrors wrapped policy end state."""
    wrapper = MotionPlannerPolicyMPWrapper()
    wrapper.reset()
    assert wrapper.episode_ended is False
    # Force episode end on impl and verify wrapper mirrors it after step
    wrapper.impl.enabled = False
    wrapper.impl.episode_ended = True
    obs = make_min_obs()
    result = wrapper.step(obs)
    assert result is None
    assert wrapper.episode_ended is True
