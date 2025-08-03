"""Reward functions for TidyBot tasks."""

from typing import Any, Dict, List

import numpy as np


class TidyBotRewardCalculator:
    """Base class for TidyBot task rewards."""

    completed_objects: set[int]
    episode_step: int

    def __init__(self, scene_type: str, num_objects: int):
        self.scene_type = scene_type
        self.num_objects = num_objects
        self.completed_objects = set()
        self.episode_step = 0

    def calculate_reward(self, obs: Dict[str, Any]) -> float:
        """Calculate reward based on current observation."""
        self.episode_step += 1
        base_reward = -0.01  # Small negative reward per timestep

        # Add task-specific rewards
        task_reward = self._calculate_task_reward(obs)

        return base_reward + task_reward

    def _calculate_task_reward(self, obs: Dict[str, Any]) -> float:
        """Calculate task-specific reward.

        Override in subclasses.
        """
        _ = obs  # Unused in base class, overridden in subclasses
        return 0

    def is_terminated(self, obs: Dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        return self._is_task_completed(obs)

    def _is_task_completed(self, obs: Dict[str, Any]) -> bool:
        """Check if task is completed.

        Override in subclasses.
        """
        _ = obs  # Unused in base class, overridden in subclasses
        return False


class MotionPlanningReward(TidyBotRewardCalculator):
    """Reward for motion planning tasks."""

    def __init__(self, scene_type: str, num_objects: int):
        super().__init__(scene_type, num_objects)
        self.target_locations = self._get_target_locations()
        self.objects_placed = 0

    def _get_target_locations(self) -> List[np.ndarray]:
        """Get target locations based on scene type."""
        return [np.array([0.5, 0.0, 0.2]) for _ in range(self.num_objects)]

    def _calculate_task_reward(self, obs: Dict[str, Any]) -> float:
        reward = 0.0

        # Check object placement at target locations
        for i, target_loc in enumerate(self.target_locations):
            if i >= self.num_objects:
                break

            obj_pos_key = f"cube{i+1}_pos"
            if obj_pos_key in obs:
                pos = obs[obj_pos_key]
                distance = np.linalg.norm(pos - target_loc)

                # Reward for being close to target
                if distance < 0.05:  # 5cm tolerance
                    if i not in self.completed_objects:
                        reward += 1.0  # Reward for reaching target
                        self.completed_objects.add(i)
                        self.objects_placed += 1
                elif distance < 0.1:  # 10cm tolerance
                    reward += float(
                        0.1 * (0.1 - distance)
                    )  # Small reward for being close

        return float(reward)

    def _is_task_completed(self, obs: Dict[str, Any]) -> bool:
        return len(self.completed_objects) == self.num_objects


def create_reward_calculator(
    scene_type: str, num_objects: int
) -> TidyBotRewardCalculator:
    """Factory function to create appropriate reward calculator."""
    return MotionPlanningReward(scene_type, num_objects)
