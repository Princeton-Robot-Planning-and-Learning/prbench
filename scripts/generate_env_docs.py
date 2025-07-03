"""Automatically create markdown documents for every registered environment."""

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import gymnasium
import imageio.v2 as iio

import prbench

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "envs"
CACHE_DIR = Path(__file__).parent.parent / ".cache" / "env_docs"


def get_env_source_hash(env_id: str, env: gymnasium.Env) -> str:
    """Get a hash of the environment's source code and metadata."""
    env_class = env.__class__
    source_code = inspect.getsource(env_class)
    module_path = inspect.getfile(env_class)
    # Read the entire module file to catch changes in imports, helper functions, etc.
    with open(module_path, "r", encoding="utf-8") as f:
        module_source = f.read()
    # Include metadata in the hash.
    metadata_str = json.dumps(env.metadata, sort_keys=True)
    # Create hash from source code, module source, and metadata.
    hash_input = f"{env_id}\n{source_code}\n{module_source}\n{metadata_str}"
    return hashlib.sha256(hash_input.encode()).hexdigest()


def get_cache_file(env_id: str) -> Path:
    """Get the cache file path for a specific environment."""
    env_filename = sanitize_env_id(env_id)
    return CACHE_DIR / f"{env_filename}.json"


def is_env_changed(env_id: str, env: gymnasium.Env) -> bool:
    """Check if the environment has changed since last generation."""
    cache_file = get_cache_file(env_id)

    if not cache_file.exists():
        return True

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        current_hash = get_env_source_hash(env_id, env)
        return cache_data.get("source_hash") != current_hash
    except (json.JSONDecodeError, KeyError):
        return True


def update_cache(env_id: str, env: gymnasium.Env) -> None:
    """Update the cache for a specific environment."""
    cache_file = get_cache_file(env_id)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Get the module file path for timestamp.
    module_path = inspect.getfile(env.__class__)

    cache_data = {
        "source_hash": get_env_source_hash(env_id, env),
        "env_id": env_id,
        "timestamp": os.path.getmtime(module_path),
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)


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
    iio.mimsave(outfile, imgs, fps=fps, loop=0)


def create_initial_state_gif(
    env_id: str,
    env: gymnasium.Env,
    num_resets: int = 25,
    seed: int = 0,
    default_fps: int = 10,
) -> None:
    """Create a GIF of different initial states by calling reset()."""
    imgs: list = []
    for i in range(num_resets):
        env.reset(seed=seed + i)
        imgs.append(env.render())
    env_filename = sanitize_env_id(env_id)
    outfile = OUTPUT_DIR / "assets" / "initial_state_gifs" / f"{env_filename}.gif"
    fps = env.metadata.get("render_fps", default_fps)
    iio.mimsave(outfile, imgs, fps=fps, loop=0)


def generate_markdown(env_id: str, env: gymnasium.Env) -> str:
    """Generate markdown for a given env."""
    md = f"# {env_id}\n"
    env_filename = sanitize_env_id(env_id)
    md += f"![random action GIF](assets/random_action_gifs/{env_filename}.gif)\n\n"
    description = env.metadata.get("description", "No description defined.")
    md += f"### Description\n{description}\n"
    md += "### Initial State Distribution\n"
    md += f"![initial state GIF](assets/initial_state_gifs/{env_filename}.gif)\n\n"
    md += "### Observation Space\n"
    md += env.metadata["observation_space_description"] + "\n\n"
    md += "### Action Space\n"
    md += env.metadata["action_space_description"] + "\n\n"
    md += "### Rewards\n"
    md += env.metadata["reward_description"] + "\n\n"
    if "references" in env.metadata:
        md += "### References\n"
        md += env.metadata["references"] + "\n\n"
    return md.rstrip() + "\n"


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate environment documentation")
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration of all environments"
    )
    args = parser.parse_args()

    print("Regenerating environment docs...")
    if args.force:
        print("Force flag detected - regenerating all environments")
    else:
        print("NOTE: the first time you commit locally, this may take some time.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "assets" / "random_action_gifs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "assets" / "initial_state_gifs").mkdir(parents=True, exist_ok=True)

    prbench.register_all_environments()

    total_envs = 0
    regenerated_envs = 0

    for env_id in prbench.get_all_env_ids():
        total_envs += 1
        env = prbench.make(env_id, render_mode="rgb_array")

        if args.force or is_env_changed(env_id, env):
            print(f"  Regenerating {env_id}...")
            create_random_action_gif(env_id, env)
            create_initial_state_gif(env_id, env)
            md = generate_markdown(env_id, env)
            assert env_id.startswith("prbench/")
            env_filename = sanitize_env_id(env_id)
            filename = OUTPUT_DIR / f"{env_filename}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md)
            update_cache(env_id, env)
            regenerated_envs += 1
        else:
            print(f"  Skipping {env_id} (no changes detected)")

    print("Finished generating environment docs.")

    # Add the results.
    subprocess.run(["git", "add", OUTPUT_DIR], check=True)


if __name__ == "__main__":
    _main()
