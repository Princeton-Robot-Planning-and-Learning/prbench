"""Register environments and expose them through make()."""

import os

import gymnasium
from gymnasium.envs.registration import register


def register_all_environments() -> None:
    """Add all benchmark environments to the gymnasium registry."""
    # NOTE: ids must start with "prbench/" to be properly registered.

    # Detect headless mode (no DISPLAY) and set OSMesa if needed
    if not os.environ.get("DISPLAY"):
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    # Obstructions2D environment with different numbers of obstructions.
    num_obstructions = [0, 1, 2, 3, 4]
    for num_obstruction in num_obstructions:
        register(
            id=f"prbench/Obstruction2D-o{num_obstruction}-v0",
            entry_point="prbench.envs.obstruction2d:Obstruction2DEnv",
            kwargs={"num_obstructions": num_obstruction},
        )

    # ClutteredRetrieval2D environment with different numbers of obstructions.
    num_obstructions = [1, 10, 25]
    for num_obstruction in num_obstructions:
        register(
            id=f"prbench/ClutteredRetrieval2D-o{num_obstruction}-v0",
            entry_point="prbench.envs.clutteredretrieval2d:ClutteredRetrieval2DEnv",
            kwargs={"num_obstructions": num_obstruction},
        )

    # ClutteredStorage2D environment with different numbers of blocks.
    num_blocks = [1, 7, 15]
    for num_block in num_blocks:
        register(
            id=f"prbench/ClutteredStorage2D-b{num_block}-v0",
            entry_point="prbench.envs.clutteredstorage2d:ClutteredStorage2DEnv",
            kwargs={"num_blocks": num_block},
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

    # TidyBot3D environments with different scenes and object counts (no policy_type)
    scene_configs = [
        ("ground", [3, 5, 7]),  # Ground/scene.xml with different object counts
    ]

    for scene_type, object_counts in scene_configs:
        for num_objects in object_counts:
            register(
                id=f"prbench/TidyBot3D-{scene_type}-o{num_objects}-v0",
                entry_point="prbench.envs.tidybot3d:TidyBot3DEnv",
                nondeterministic=True,
                kwargs={
                    "scene_type": scene_type,
                    "num_objects": num_objects,
                },
            )


def make(*args, **kwargs) -> gymnasium.Env:
    """Create a registered environment from its name."""
    return gymnasium.make(*args, **kwargs)


def make_unwrapped(*args, **kwargs) -> gymnasium.Env:
    """Create a registered environment and return the unwrapped version.

    This allows access to custom methods like step_with_policy.
    """
    env = gymnasium.make(*args, **kwargs)
    while hasattr(env, "env"):
        env = env.env
    return env


def get_all_env_ids() -> set[str]:
    """Get all known benchmark environments."""
    return {env for env in gymnasium.registry if env.startswith("prbench/")}
