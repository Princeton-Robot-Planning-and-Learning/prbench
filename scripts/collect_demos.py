"""Collect human demonstrations in prbench environments."""

import argparse
import dill as pkl
import time
from pathlib import Path

import numpy as np

try:
    import pygame
except ImportError:
    print("Error: pygame is required for demo collection.")
    print("Install it with: pip install pygame")
    exit(1)

import prbench


class DemoCollector:
    """Collect human demonstrations in prbench environments using pygame."""

    def __init__(self, env_id: str, demo_dir: Path, screen_width: int = 800, screen_height: int = 600,
                 render_fps: int = 20, font_size: int = 24, start_seed: int = 0) -> None:
        # Initialize the demo directory.
        self.env_id = env_id
        self.demo_dir = demo_dir
        self.demo_dir.mkdir(parents=True, exist_ok=True)

        # Create the environment.
        prbench.register_all_environments()
        self.env = prbench.make(env_id, render_mode="rgb_array")
        self.unwrapped_env = self.env.unwrapped
        if not hasattr(self.unwrapped_env, 'get_action_from_gui_input'):
            raise RuntimeError(f"Environment {env_id} must implement get_action_from_gui_input.")
        
        # Initialize the data (for a single demo).
        self.observations: list = []
        self.actions: list = []
        self.rewards: list[float] = []
        self.terminated: bool = False
        self.truncated: bool = False

        # The user may be pressing multiple keys at once, and pygame only gets
        # up/down events, rather than accessing all keys currently pressed.
        self.keys_pressed = set()
        
        # Initialize pygame.
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"Demo Collection - {env_id}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, font_size)
        self.render_fps = render_fps        
        
        # Reset the environment.
        self.seed = start_seed
        self.reset_env()

    def reset_env(self) -> None:
        obs, _ = self.env.reset(seed=self.seed)
        self.seed += 1
        self.observations = [obs]
        self.actions = []
        self.rewards = []
        self.terminated = False
        self.truncated = False
        self.keys_pressed.clear()

    def save_demo(self) -> None:
        if not self.observations or not self.actions:
            print("Warning: No demo data to save!")
            return
        timestamp = int(time.time())
        demo_filename = f"{self.env_id.replace('/', '_')}_{timestamp}.p"
        demo_path = self.demo_dir / demo_filename
        demo_data = {
            "env_id": self.env_id,
            "timestamp": timestamp,
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        with open(demo_path, "wb") as f:
            pkl.dump(demo_data, f)
        print(f"Demo saved to {demo_path}")

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset_env()
                elif event.key == pygame.K_g:
                    self.save_demo()
                elif event.key == pygame.K_q:
                    return False
                elif not self.terminated and not self.truncated:
                    key_name = pygame.key.name(event.key)
                    if key_name not in {"r", "g", "q"}:
                        self.keys_pressed.add(key_name)
                        self.step_env()
            elif event.type == pygame.KEYUP:
                if not self.terminated and not self.truncated:
                    key_name = pygame.key.name(event.key)
                    if key_name in self.keys_pressed:
                        self.keys_pressed.discard(key_name)
                        self.step_env()
        return not (self.terminated or self.truncated)

    def render(self) -> None:
        img = self.env.render()
        # For now, assume a certain image format.
        assert len(img.shape) == 3 and img.shape[2] in (3, 4) and img.dtype == np.uint8
        img_surface = pygame.surfarray.make_surface(img[:, :, :3].swapaxes(0, 1))
        
        # Scale image to fit screen.
        img_rect = img_surface.get_rect()
        scale = min(self.screen_width / img_rect.width, self.screen_height / img_rect.height)
        new_width = int(img_rect.width * scale)
        new_height = int(img_rect.height * scale)
        img_surface = pygame.transform.scale(img_surface, (new_width, new_height))
        
        # Center image on screen.
        img_rect = img_surface.get_rect()
        img_rect.center = (self.screen_width // 2, self.screen_height // 2)
        
        # Clear screen.
        self.screen.fill((0, 0, 0))
        
        # Draw image.
        self.screen.blit(img_surface, img_rect)
        
        # Draw status text.
        status_text = f"{self.env_id} - Demo Length: {len(self.actions)}"
        text_surface = self.font.render(status_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        text_rect.topleft = (10, 10)
        self.screen.blit(text_surface, text_rect)
        
        # Draw instructions
        instructions = [
            "Press 'r' to start/reset demo",
            "Press 'g' to save demo", 
            "Press 'q' to quit"
        ]
        for i, instruction in enumerate(instructions):
            text_surface = self.font.render(instruction, True, (200, 200, 200))
            text_rect = text_surface.get_rect()
            text_rect.bottomleft = (10, self.screen_height - 10 - (len(instructions) - i - 1) * 25)
            self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()

    def step_env(self) -> None:
        action = self.unwrapped_env.get_action_from_gui_input(self.keys_pressed)
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminated = terminated
        self.truncated = truncated

    def run(self) -> None:
        """Run the demo collection GUI."""
        running = True
        while running:
            running = self.handle_events()
            self.render()
            self.clock.tick(self.render_fps)
        
        pygame.quit()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Collect human demonstrations in prbench environments")
    parser.add_argument(
        "env_id", 
        help="Environment ID (e.g., prbench/Obstruction2D-o2-v0)"
    )
    parser.add_argument(
        "--demo-dir", 
        type=Path, 
        default=Path("demos"),
        help="Directory to save demonstrations (default: demos)"
    )
    args = parser.parse_args()
    if not args.env_id.startswith("prbench/"):
        print("Error: Environment ID must start with 'prbench/'")
        return
    collector = DemoCollector(args.env_id, args.demo_dir)
    collector.run()


if __name__ == "__main__":
    _main() 