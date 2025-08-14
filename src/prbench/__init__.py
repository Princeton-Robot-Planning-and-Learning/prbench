"""Register environments and expose them through make()."""

import gymnasium
from gymnasium.envs.registration import register


def register_all_environments() -> None:
    """Add all benchmark environments to the gymnasium registry."""
    # NOTE: ids must start with "prbench/" to be properly registered.

    # Obstructions2D environment with different numbers of obstructions.
    num_obstructions = [0, 1, 2, 3, 4]
    for num_obstruction in num_obstructions:
        _register(
            id=f"prbench/Obstruction2D-o{num_obstruction}-v0",
            entry_point="prbench.envs.geom2d.obstruction2d:Obstruction2DEnv",
            kwargs={"num_obstructions": num_obstruction},
        )

    # ClutteredRetrieval2D environment with different numbers of obstructions.
    num_obstructions = [1, 10, 25]
    for num_obstruction in num_obstructions:
        _register(
            id=f"prbench/ClutteredRetrieval2D-o{num_obstruction}-v0",
            entry_point="prbench.envs.geom2d.clutteredretrieval2d:ClutteredRetrieval2DEnv",  # pylint: disable=line-too-long
            kwargs={"num_obstructions": num_obstruction},
        )

    # ClutteredStorage2D environment with different numbers of blocks.
    num_blocks = [1, 7, 15]
    for num_block in num_blocks:
        _register(
            id=f"prbench/ClutteredStorage2D-b{num_block}-v0",
            entry_point="prbench.envs.geom2d.clutteredstorage2d:ClutteredStorage2DEnv",
            kwargs={"num_blocks": num_block},
        )

    # Motion2D environment with different numbers of passages.
    num_passages = [1, 2, 3, 4, 5]
    for num_passage in num_passages:
        _register(
            id=f"prbench/Motion2D-p{num_passage}-v0",
            entry_point="prbench.envs.geom2d.motion2d:Motion2DEnv",
            kwargs={"num_passages": num_passage},
        )

    # StickButton2D environment with different numbers of buttons.
    num_buttons = [1, 2, 3, 5, 10]
    for num_button in num_buttons:
        _register(
            id=f"prbench/StickButton2D-b{num_button}-v0",
            entry_point="prbench.envs.geom2d.stickbutton2d:StickButton2DEnv",
            kwargs={"num_buttons": num_button},
        )


def _register(id: str, *args, **kwargs) -> None:  # pylint: disable=redefined-builtin
    """Call register(), but only if the environment id is not already registered.

    This is to avoid noisy logging.warnings in register(). We are assuming that envs
    with the same id are equivalent, so this is safe.
    """
    if id not in gymnasium.registry:
        register(id, *args, **kwargs)


def make(*args, **kwargs) -> gymnasium.Env:
    """Create a registered environment from its name."""
    return gymnasium.make(*args, **kwargs)


def get_all_env_ids() -> set[str]:
    """Get all known benchmark environments."""
    return {env for env in gymnasium.registry if env.startswith("prbench/")}
