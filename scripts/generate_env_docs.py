"""Automatically create markdown documents for every registered environment."""

from pathlib import Path

import gymnasium
import imageio.v2 as iio

import prbench

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "envs"


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
    outfile = OUTPUT_DIR / "assets" / "random_action_gifs" / f"{env_filename}.gif"
    fps = env.metadata.get("render_fps", default_fps)
    iio.mimsave(outfile, imgs, fps=fps)


def generate_markdown(env_id: str, env: gymnasium.Env) -> str:
    """Generate markdown for a given env."""
    md = f"# {env_id}\n"
    env_filename = sanitize_env_id(env_id)
    md += f"![random action GIF](assets/random_action_gifs/{env_filename}.gif)\n\n"
    description = env.metadata.get("description", "No description defined.")
    md += f"### Description\n{description}"
    return md


def _main() -> None:
    print("Regenerating environment docs...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    prbench.register_all_environments()
    for env_id in prbench.get_all_env_ids():
        env = prbench.make(env_id, render_mode="rgb_array")
        create_random_action_gif(env_id, env)
        md = generate_markdown(env_id, env)
        assert env_id.startswith("prbench/")
        env_filename = sanitize_env_id(env_id)
        filename = OUTPUT_DIR / f"{env_filename}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(md)
    print("Finished generating environment docs.")


if __name__ == "__main__":
    _main()
