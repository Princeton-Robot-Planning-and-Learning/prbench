"""Basic tests for the TidyBot3D environment: observation and action space validity."""

from prbench.envs.tidybot.tidybot3d import TidyBot3DEnv


def test_tidybot3d_observation_space():
    """Test that the observation returned by TidyBot3DEnv.reset() is within the
    observation space."""
    env = TidyBot3DEnv(num_objects=3, render_images=False)
    obs = env.reset()[0]
    assert env.observation_space.contains(obs), "Observation not in observation space"
    env.close()


def test_tidybot3d_action_space():
    """Test that a sampled action is within the TidyBot3DEnv action space."""
    env = TidyBot3DEnv(num_objects=3, render_images=False)
    action = env.action_space.sample()
    assert env.action_space.contains(action), "Action not in action space"
    env.close()
