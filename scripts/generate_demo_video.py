"""Generate videos from pickled demonstrations."""

import argparse
import sys
from pathlib import Path

import dill as pkl
import imageio.v2 as iio
import numpy as np
from generate_env_docs import sanitize_env_id

import prbench


def load_demo(demo_path: Path) -> dict:
    """Load a demonstration from a pickle file."""
    try:
        with open(demo_path, "rb") as f:
            demo_data = pkl.load(f)

        # Validate demo data structure
        required_keys = ["env_id", "observations", "actions"]
        for key in required_keys:
            if key not in demo_data:
                raise ValueError(f"Demo data missing required key: {key}")

        if not demo_data["actions"]:
            raise ValueError("Demo contains no actions")

        if len(demo_data["observations"]) != len(demo_data["actions"]) + 1:
            print(
                f"Warning: Expected {len(demo_data['actions']) + 1} observations, got {len(demo_data['observations'])}"
            )

        # Check if seed is available (for backward compatibility)
        if "seed" not in demo_data:
            print("Warning: Demo does not contain seed information. Using seed=0.")
            demo_data["seed"] = 0

        return demo_data
    except Exception as e:
        print(f"Error loading demo from {demo_path}: {e}")
        sys.exit(1)


def generate_demo_video(
    demo_path: Path,
    output_path: Path = None,
    fps: int = None,
    loop: int = 0,
) -> None:
    """Generate a video from a pickled demonstration.

    Args:
        demo_path: Path to the pickled demonstration file
        output_path: Path to save the video (default: auto-generated)
        fps: Frames per second for the video (default: from env metadata)
        loop: Number of loops for GIF (0 = infinite)
    """
    # Load the demonstration
    demo_data = load_demo(demo_path)

    # Extract demo information
    env_id = demo_data["env_id"]
    observations = demo_data["observations"]
    actions = demo_data["actions"]
    seed = demo_data["seed"]

    print(f"Loaded demo for environment: {env_id}")
    print(f"Demo length: {len(actions)} actions")
    print(f"Demo seed: {seed}")

    # Create the environment
    prbench.register_all_environments()
    env = prbench.make(env_id, render_mode="rgb_array")

    # Get FPS from environment metadata if not specified
    if fps is None:
        fps = env.metadata.get("render_fps", 10)

    # Generate output path if not specified
    if output_path is None:
        env_filename = sanitize_env_id(env_id)
        output_dir = Path("./docs/envs/assets/demo_gifs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{env_filename}.gif"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reset environment to initial state with the correct seed
    env.reset(seed=seed)

    # Collect frames by replaying the demonstration
    frames = []

    # Add initial frame
    initial_frame = env.render()
    frames.append(initial_frame)

    # Replay each action and capture frames
    for i, action in enumerate(actions):
        try:
            obs, reward, terminated, truncated, _ = env.step(action)
            frame = env.render()
            frames.append(frame)

            if terminated or truncated:
                print(f"Episode ended after {i+1} actions")
                break
        except Exception as e:
            print(f"Error during action {i}: {e}")
            print(f"Continuing with {len(frames)} frames collected so far")
            break

    # Check if we have enough frames
    if len(frames) < 2:
        print("Error: Not enough frames collected to create a video")
        sys.exit(1)

    # Save the video
    print(f"Saving video to {output_path}")
    print(f"Video specs: {len(frames)} frames, {fps} fps")

    try:
        iio.mimsave(output_path, frames, fps=fps, loop=loop)
        print(f"Video saved successfully!")
    except Exception as e:
        print(f"Error saving video: {e}")
        sys.exit(1)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate videos from pickled demonstrations",
        epilog="""
Examples:
  # Generate video with default settings
  python scripts/generate_demo_video.py demos/Obstruction2D-o0/1/1751653925.p
  
  # Generate video with custom output path and FPS
  python scripts/generate_demo_video.py demos/Obstruction2D-o1/1/1751653958.p --output my_demo.gif --fps 15
  
  # Generate video that loops 3 times
  python scripts/generate_demo_video.py demos/Obstruction2D-o0/1/1751653925.p --loop 3
        """,
    )
    parser.add_argument(
        "demo_path", type=Path, help="Path to the pickled demonstration file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for the video (default: auto-generated in docs/envs/assets/demo_gifs/)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="Frames per second for the video (default: from environment metadata)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Number of loops for GIF (0 = infinite, default: 0)",
    )

    args = parser.parse_args()

    # Check if demo file exists
    if not args.demo_path.exists():
        print(f"Error: Demo file {args.demo_path} does not exist")
        sys.exit(1)

    generate_demo_video(
        demo_path=args.demo_path,
        output_path=args.output,
        fps=args.fps,
        loop=args.loop,
    )


if __name__ == "__main__":
    _main()
