"""Generate TidyBot random action GIFs for debugging."""

import argparse
from pathlib import Path

import gymnasium
import imageio.v2 as iio

import prbench

DEBUG_DIR = Path(__file__).parent / "debug"


def sanitize_env_id(env_id: str) -> str:
    """Remove unnecessary stuff from the env ID."""
    assert env_id.startswith("prbench/")
    env_id = env_id[len("prbench/") :]
    env_id = env_id.replace("/", "_")
    assert env_id[-3:-1] == "-v"
    return env_id[:-3]


def create_random_action_gif(
    env_id: str,
    env: gymnasium.Env,
    num_actions: int = 25,
    seed: int = 0,
    default_fps: int = 10,
) -> None:
    """Create a GIF of taking random actions in the environment."""
    imgs: list = []
    env.reset(seed=seed)
    env.action_space.seed(seed)
    imgs.append(env.render())
    for _ in range(num_actions):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        imgs.append(env.render())
        if terminated or truncated:
            break
    env_filename = sanitize_env_id(env_id)
    outfile = DEBUG_DIR / f"{env_filename}.gif"
    fps = env.metadata.get("render_fps", default_fps)
    iio.mimsave(outfile, imgs, fps=fps, loop=0)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate TidyBot random action GIFs")
    parser.add_argument(
        "--num-actions", type=int, default=25, help="Number of random actions to take"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    print("Generating TidyBot random action GIFs...")

    DEBUG_DIR.mkdir(exist_ok=True)

    prbench.register_all_environments()

    tidybot_envs = [
        env_id for env_id in prbench.get_all_env_ids() if "TidyBot" in env_id
    ]

    for env_id in tidybot_envs:
        print(f"  Generating {env_id}...")
        env = prbench.make(env_id, render_mode="rgb_array")
        create_random_action_gif(
            env_id, env, num_actions=args.num_actions, seed=args.seed
        )
        env.close()  # type: ignore[no-untyped-call]

    print(f"Finished generating {len(tidybot_envs)} TidyBot GIFs in {DEBUG_DIR}")


if __name__ == "__main__":
    _main()
