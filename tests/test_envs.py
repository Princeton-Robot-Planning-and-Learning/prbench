"""Common tests for all environments."""

import gymnasium
from gymnasium.utils.env_checker import check_env

import prbench


def test_env_make_and_check_env():
    """Tests that all registered environments can be created with make.

    Also calls gymnasium.utils.env_checker.check_env() to test API
    functions.
    """
    print("Running test_env_make_and_check_env", flush=True)
    prbench.register_all_environments()
    env_ids = prbench.get_all_env_ids()
    assert len(env_ids) > 0
    for env_id in env_ids:
        print("Starting env_id", env_id, flush=True)
        # We currently require all environments to have RGB rendering.
        env = prbench.make(env_id, render_mode="rgb_array")
        assert env.render_mode == "rgb_array"
        assert isinstance(env, gymnasium.Env)
        # TODO remove skip_render_check, just adding for testing
        check_env(env.unwrapped, skip_render_check=True)
        env.close()
