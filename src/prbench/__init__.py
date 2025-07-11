"""Register environments and expose them through make()."""

import gymnasium
from gymnasium.envs.registration import register


def register_all_environments() -> None:
    """Add all benchmark environments to the gymnasium registry."""
    # NOTE: ids must start with "prbench/" to be properly registered.

    # Obstructions2D environment with different numbers of obstructions.
    num_obstructions = [0, 1, 2, 3, 4]
    for num_obstruction in num_obstructions:
        register(
            id=f"prbench/Obstruction2D-o{num_obstruction}-v0",
            entry_point="prbench.envs.obstruction2d:Obstruction2DEnv",
            kwargs={"num_obstructions": num_obstruction},
        )

    # Clutter2D environment with different numbers of obstructions.
    num_obstructions = [1, 10, 25]
    for num_obstruction in num_obstructions:
        register(
            id=f"prbench/Clutter2D-o{num_obstruction}-v0",
            entry_point="prbench.envs.clutter2d:Clutter2DEnv",
            kwargs={"num_obstructions": num_obstruction},
        )

    # Motion2D environment with different numbers of passages.
    num_passages = [1, 2, 3, 4, 5]
    for num_passage in num_passages:
        register(
            id=f"prbench/Motion2D-p{num_passage}-v0",
            entry_point="prbench.envs.motion2d:Motion2DEnv",
            kwargs={"num_passages": num_passage},
        )

    # StickButton2D environment with different numbers of buttons.
    num_buttons = [1, 5, 10]
    for num_button in num_buttons:
        register(
            id=f"prbench/StickButton2D-b{num_button}-v0",
            entry_point="prbench.envs.stickbutton2d:StickButton2DEnv",
            kwargs={"num_buttons": num_button},
        )


def make(*args, **kwargs) -> gymnasium.Env:
    """Create a registered environment from its name."""
    return gymnasium.make(*args, **kwargs)


def get_all_env_ids() -> set[str]:
    """Get all known benchmark environments."""
    return {env for env in gymnasium.registry if env.startswith("prbench/")}
