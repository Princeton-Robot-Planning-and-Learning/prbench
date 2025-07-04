"""Collect human demonstrations in prbench environments."""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

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

    def __init__(self, env_id: str, demo_dir: Path) -> None:
        self.env_id = env_id
        self.demo_dir = demo_dir
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        prbench.register_all_environments()
        self.env = prbench.make(env_id, render_mode="rgb_array")
        self.unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        if not hasattr(self.unwrapped_env, 'map_human_input_to_action'):
            raise RuntimeError(f"Environment {env_id} must implement map_human_input_to_action.")
        if not hasattr(self.unwrapped_env, 'get_human_input_mapping'):
            raise RuntimeError(f"Environment {env_id} must implement get_human_input_mapping.")
        
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.terminated: bool = False
        self.truncated: bool = False
        self.keys_pressed = set()
        self.current_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        self.demo_active = False
        self.mouse_click_pending = False  # Track if we have a pending mouse click
        
        # Initialize pygame
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"Demo Collection - {env_id}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        self.reset_env()
        self.print_instructions()

    def print_instructions(self) -> None:
        mapping = self.unwrapped_env.get_human_input_mapping()
        print("\n" + "="*60)
        print("DEMO COLLECTION INSTRUCTIONS")
        print("="*60)
        print("Controls:")
        print(mapping.get('description', 'See environment docs.'))
        print("\nDemo Management:")
        print("  Press 'r' to start/reset demo")
        print("  Press 's' to save demo")
        print("  Press 'q' to quit")
        print("="*60)

    def reset_env(self) -> None:
        obs, _ = self.env.reset()
        self.observations = [obs]
        self.actions = []
        self.rewards = []
        self.terminated = False
        self.truncated = False
        self.current_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        self.keys_pressed.clear()

    def start_demo(self) -> None:
        self.reset_env()
        self.demo_active = True
        print("Demo started!")

    def save_demo(self) -> None:
        if not self.observations or not self.actions:
            print("Warning: No demo data to save!")
            return
        timestamp = int(time.time())
        demo_filename = f"{self.env_id.replace('/', '_')}_{timestamp}.json"
        demo_path = self.demo_dir / demo_filename
        demo_data = {
            "env_id": self.env_id,
            "timestamp": timestamp,
            "observations": [obs.tolist() for obs in self.observations],
            "actions": [action.tolist() for action in self.actions],
            "rewards": self.rewards,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        with open(demo_path, "w", encoding="utf-8") as f:
            json.dump(demo_data, f, indent=2)
        print(f"Demo saved to {demo_path}")

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.start_demo()
                elif event.key == pygame.K_s:
                    self.save_demo()
                elif event.key == pygame.K_q:
                    return False
                elif self.demo_active and not self.terminated and not self.truncated:
                    key_name = pygame.key.name(event.key)
                    # Only add keys for non-base controls (rotation, arm, vacuum)
                    if key_name not in {"w", "a", "s", "d"}:
                        self.keys_pressed.add(key_name)
                        self.update_action_from_keys()
            elif event.type == pygame.KEYUP:
                if self.demo_active and not self.terminated and not self.truncated:
                    key_name = pygame.key.name(event.key)
                    if key_name in self.keys_pressed:
                        self.keys_pressed.discard(key_name)
                        self.update_action_from_keys()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.demo_active and not self.terminated and not self.truncated:
                    if event.button == 1:  # Left click
                        self.set_base_action_from_mouse(event.pos)
        return True

    def set_base_action_from_mouse(self, mouse_pos):
        """Set dx/dy in the action vector to move toward the mouse click position."""
        # Get environment render and scaling
        img = self.env.render()
        img_h, img_w = img.shape[:2]
        scale = min(self.screen_width / img_w, self.screen_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        offset_x = (self.screen_width - new_w) // 2
        offset_y = (self.screen_height - new_h) // 2
        # Convert mouse position to image coordinates
        mx, my = mouse_pos
        ix = int((mx - offset_x) / scale)
        iy = int((my - offset_y) / scale)
        ix = np.clip(ix, 0, img_w - 1)
        iy = np.clip(iy, 0, img_h - 1)
        # Get robot base position from observation (assume first two obs are x, y)
        obs = self.observations[-1]
        robot_x, robot_y = obs[0], obs[1]
        # Map image coordinates to environment coordinates
        # Assume the environment's render covers [0, 1] x [0, 1] (customize if needed)
        env_x = ix / img_w
        env_y = 1.0 - (iy / img_h)  # Flip y-axis so up is up
        # Compute direction
        dx = env_x - robot_x
        dy = env_y - robot_y
        # Normalize to max step size
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        max_dx = float(max(abs(action_low[0]), abs(action_high[0])))
        max_dy = float(max(abs(action_low[1]), abs(action_high[1])))
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            norm = np.hypot(dx, dy)
            dx_step = max_dx * dx / norm if norm > 0 else 0.0
            dy_step = max_dy * dy / norm if norm > 0 else 0.0
        else:
            dx_step = 0.0
            dy_step = 0.0
        # Set dx/dy in the current action
        self.current_action[0] = np.clip(dx_step, action_low[0], action_high[0])
        self.current_action[1] = np.clip(dy_step, action_low[1], action_high[1])
        self.mouse_click_pending = True  # Mark that we have a pending mouse movement

    def update_action_from_keys(self) -> None:
        # Only update non-base controls (rotation, arm, vacuum, etc.)
        base_dx = self.current_action[0]
        base_dy = self.current_action[1]
        action = self.unwrapped_env.map_human_input_to_action(self.keys_pressed)
        action[0] = base_dx
        action[1] = base_dy
        self.current_action = action

    def render(self) -> None:
        # Get environment render
        img = self.env.render()
        
        # Convert numpy array to pygame surface
        if len(img.shape) == 3:
            # RGB image - ensure it's in the right format
            if img.shape[2] == 3:  # RGB
                # Convert to uint8 if needed
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            else:
                # Handle other 3D formats
                img = img[:, :, 0]  # Take first channel
                img = (img * 255).astype(np.uint8)
                img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        else:
            # Grayscale image
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        
        # Scale image to fit screen
        img_rect = img_surface.get_rect()
        scale = min(self.screen_width / img_rect.width, self.screen_height / img_rect.height)
        new_width = int(img_rect.width * scale)
        new_height = int(img_rect.height * scale)
        img_surface = pygame.transform.scale(img_surface, (new_width, new_height))
        
        # Center image on screen
        img_rect = img_surface.get_rect()
        img_rect.center = (self.screen_width // 2, self.screen_height // 2)
        
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw image
        self.screen.blit(img_surface, img_rect)
        
        # Draw status text
        status = "Demo Active" if self.demo_active else "Ready"
        status_text = f"{self.env_id} - {status} - Demo Length: {len(self.actions)}"
        text_surface = self.font.render(status_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        text_rect.topleft = (10, 10)
        self.screen.blit(text_surface, text_rect)
        
        # Draw instructions
        instructions = [
            "Press 'r' to start/reset demo",
            "Press 's' to save demo", 
            "Press 'q' to quit"
        ]
        for i, instruction in enumerate(instructions):
            text_surface = self.font.render(instruction, True, (200, 200, 200))
            text_rect = text_surface.get_rect()
            text_rect.bottomleft = (10, self.screen_height - 10 - (len(instructions) - i - 1) * 25)
            self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()

    def step_env(self) -> None:
        obs, reward, terminated, truncated, _ = self.env.step(self.current_action)
        self.observations.append(obs)
        self.actions.append(self.current_action.copy())
        self.rewards.append(reward)
        self.terminated = terminated
        self.truncated = truncated
        
        # Reset base movement after one step if mouse click was pending
        if self.mouse_click_pending:
            self.current_action[0] = 0.0  # Reset dx
            self.current_action[1] = 0.0  # Reset dy
            self.mouse_click_pending = False
        
        if terminated or truncated:
            print("Demo completed! Press 's' to save or 'r' to start a new demo.")
            self.demo_active = False

    def run(self) -> None:
        """Run the demo collection GUI."""
        running = True
        while running:
            running = self.handle_events()
            
            # Always update action from current keys (for held keys)
            self.update_action_from_keys()
            
            # Step environment if demo is active
            if (self.demo_active and not self.terminated and not self.truncated and 
                len(self.observations) > 0):
                self.step_env()
            
            self.render()
            self.clock.tick(20)  # 20 FPS
        
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