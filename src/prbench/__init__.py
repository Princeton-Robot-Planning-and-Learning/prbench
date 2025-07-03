"""Register environments and expose them through make()."""

import gymnasium
from gymnasium.envs.registration import register


def register_all_environments() -> None:
    """Add all benchmark environments to the gymnasium registry."""
    # NOTE: ids must start with "prbench/" to be properly registered.
    register(
        id="prbench/Obstruction2D-v0",
        entry_point="prbench.envs.obstruction2d:Obstruction2DEnv",
    )


def make(*args, **kwargs) -> gymnasium.Env:
    """Create a registered environment from its name."""
    return gymnasium.make(*args, **kwargs)


def get_all_env_ids() -> set[str]:
    """Get all known benchmark environments."""
    return {env for env in gymnasium.registry if env.startswith("prbench/")}
