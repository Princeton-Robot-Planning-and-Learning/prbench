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

    # TidyBot3D environments with different scene types, object counts, and policy types
    scene_configs = [
        ("table", [3, 5, 7]),  # Table stacking with different object counts
        ("cupboard", [3, 5, 8]),  # Cupboard organization with different object counts
        ("cabinet", [2, 4, 6]),  # Cabinet manipulation with different object counts
        ("ground", [3, 5, 7]),  # Ground/scene.xml with different object counts
    ]

    policy_types = [
        "mp",  # Motion planning policies
        "stack",  # Object stacking policies
        "stack_three",  # Three-object stacking policies
        "mp_three",  # Three-sequential motion planning
        "mp_cabinet_two_phase",  # Two-phase cabinet manipulation
        "custom_grasp",  # Custom grasping policies
        "custom_grasp_three",  # Three-sequential custom grasping
        "mp_n_cupboard",  # N-object cupboard manipulation
    ]

    for scene_type, object_counts in scene_configs:
        for num_objects in object_counts:
            for policy_type in policy_types:
                # Skip incompatible combinations
                if (
                    (
                        scene_type == "cabinet"
                        and policy_type in ["stack", "stack_three"]
                    )
                    or (
                        scene_type == "table"
                        and policy_type in ["mp_cabinet_two_phase"]
                    )
                    or (
                        scene_type == "drawer"
                    )
                    or (
                        scene_type == "cupboard"
                        and policy_type in ["mp_cabinet_two_phase"]
                    )
                ):
                    continue

                register(
                    id=f"prbench/TidyBot3D-{scene_type}-o{num_objects}-{policy_type}-v0",
                    entry_point="prbench.envs.tidybot3d:TidyBot3DEnv",
                    kwargs={
                        "scene_type": scene_type,
                        "num_objects": num_objects,
                        "policy_type": policy_type,
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
