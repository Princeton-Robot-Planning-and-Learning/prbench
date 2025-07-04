"""Generate videos from pickled demonstrations.

NOTE: this currently assumes that environments are deterministic. If that is
not the case, we will need to be able to render observations (which are being
saving in the demonstrations also).
"""

import argparse
import sys
from pathlib import Path

import dill as pkl
import imageio.v2 as iio
from generate_env_docs import sanitize_env_id

import prbench


def load_demo(demo_path: Path) -> dict:
    """Load a demonstration from a pickle file."""
    try:
        with open(demo_path, "rb") as f:
            demo_data = pkl.load(f)

        # Validate demo data structure.
        required_keys = ["env_id", "observations", "actions"]
        for key in required_keys:
            if key not in demo_data:
                raise ValueError(f"Demo data missing required key: {key}")

        if not demo_data["actions"]:
            raise ValueError("Demo contains no actions")

        if len(demo_data["observations"]) != len(demo_data["actions"]) + 1:
            print(
                f"Warning: Expected {len(demo_data['actions']) + 1} observations, "
                f"got {len(demo_data['observations'])}"
            )

        if "seed" not in demo_data:
            raise ValueError(" Demo does not contain seed information.")

        return demo_data
    except Exception as e:
        print(f"Error loading demo from {demo_path}: {e}")
        sys.exit(1)


def generate_demo_video(
    demo_path: Path,
    output_path: Path | None = None,
    fps: int | None = None,
    loop: int = 0,
) -> None:
    """Generate a video from a pickled demonstration."""
    # Load the demonstration.
    demo_data = load_demo(demo_path)

    # Extract demo information.
    env_id = demo_data["env_id"]
    actions = demo_data["actions"]
    seed = demo_data["seed"]

    print(f"Loaded demo for environment: {env_id}")
    print(f"Demo length: {len(actions)} actions")
    print(f"Demo seed: {seed}")

    # Create the environment.
    prbench.register_all_environments()
    env = prbench.make(env_id, render_mode="rgb_array")

    # Get FPS from environment metadata if not specified.
    if fps is None:
        fps = env.metadata.get("render_fps", 10)

    # Generate output path if not specified.
    if output_path is None:
        env_filename = sanitize_env_id(env_id)
        output_dir = Path("./docs/envs/assets/demo_gifs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{env_filename}.gif"

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reset environment to initial state with the correct seed.
    env.reset(seed=seed)

    # Collect frames by replaying the demonstration.
    frames = []

    # Add initial frame.
    initial_frame = env.render()  # type: ignore
    frames.append(initial_frame)

    # Replay each action and capture frames.
    for i, action in enumerate(actions):
        try:
            _, _, terminated, truncated, _ = env.step(action)
            frame = env.render()  # type: ignore
            frames.append(frame)

            if terminated or truncated:
                print(f"Episode ended after {i+1} actions")
                break
        except Exception as e:
            print(f"Error during action {i}: {e}")
            print(f"Continuing with {len(frames)} frames collected so far")
            break

    # Check if we have enough frames.
    if len(frames) < 2:
        print("Error: Not enough frames collected to create a video")
        sys.exit(1)

    # Save the video.
    print(f"Saving video to {output_path}")
    print(f"Video specs: {len(frames)} frames, {fps} fps")

    try:
        iio.mimsave(output_path, frames, fps=fps, loop=loop)  # type: ignore
        print("Video saved successfully!")
    except Exception as e:
        print(f"Error saving video: {e}")
        sys.exit(1)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate videos from pickled demonstrations"
    )
    parser.add_argument(
        "demo_path", type=Path, help="Path to the pickled demonstration file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path (default: auto-generated in docs/envs/assets/demo_gifs/)",
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
