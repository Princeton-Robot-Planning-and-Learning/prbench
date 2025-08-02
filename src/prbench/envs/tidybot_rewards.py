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


class TableStackingReward(TidyBotRewardCalculator):
    """Reward for table stacking tasks."""

    def __init__(self, num_objects: int):
        super().__init__("table", num_objects)
        self.target_height = 0.1  # Target stacking height
        self.target_area = np.array([0.6, 0.4])  # Target stacking area
        self.stack_height = 0.0
        self.stacked_objects = 0

    def _calculate_task_reward(self, obs: Dict[str, Any]) -> float:
        reward = 0.0

        # Check object stacking
        current_stacked = 0
        current_height = 0.0

        for i in range(self.num_objects):
            obj_pos_key = f"cube{i+1}_pos"
            if obj_pos_key in obs:
                pos = obs[obj_pos_key]
                # Check if object is in target area and stacked
                if (
                    self.target_area[0] - 0.1 <= pos[0] <= self.target_area[0] + 0.1
                    and self.target_area[1] - 0.1 <= pos[1] <= self.target_area[1] + 0.1
                ):
                    current_height = max(current_height, pos[2])
                    current_stacked += 1

        # Reward for new objects being stacked
        if current_stacked > self.stacked_objects:
            reward += (current_stacked - self.stacked_objects) * 0.5
            self.stacked_objects = current_stacked

        # Reward for reaching new heights
        if current_height > self.stack_height:
            reward += (current_height - self.stack_height) * 2.0
            self.stack_height = current_height

        # Bonus for reaching target height
        if current_height >= self.target_height and current_stacked == self.num_objects:
            reward += 5.0

        return float(reward)

    def _is_task_completed(self, obs: Dict[str, Any]) -> bool:
        # Terminate when all objects are stacked to target height
        stacked_objects = 0
        max_height = 0.0

        for i in range(self.num_objects):
            obj_pos_key = f"cube{i+1}_pos"
            if obj_pos_key in obs:
                pos = obs[obj_pos_key]
                if (
                    self.target_area[0] - 0.1 <= pos[0] <= self.target_area[0] + 0.1
                    and self.target_area[1] - 0.1 <= pos[1] <= self.target_area[1] + 0.1
                ):
                    max_height = max(max_height, pos[2])
                    stacked_objects += 1

        return stacked_objects == self.num_objects and max_height >= self.target_height


class DrawerReward(TidyBotRewardCalculator):
    """Reward for drawer opening/closing tasks."""

    def __init__(self, num_objects: int):
        super().__init__("drawer", num_objects)
        self.drawer_opened = False
        self.objects_placed = 0
        self.drawer_area = np.array(
            [0.4, 0.8, -0.2, 0.2, 0.0, 0.1]
        )  # [x_min, x_max, y_min, y_max, z_min, z_max]

    def _calculate_task_reward(self, obs: Dict[str, Any]) -> float:
        reward = 0.0

        # Check if drawer is opened (handle position indicates drawer state)
        if "left_handle_pos" in obs and "right_handle_pos" in obs:
            handle_pos = obs["left_handle_pos"]
            # Simplified check - in reality would need more sophisticated detection
            if handle_pos[0] > 0.5:  # Drawer is opened
                if not self.drawer_opened:
                    reward += 1.0  # Reward for opening drawer
                    self.drawer_opened = True

        # Check object placement in drawer
        if self.drawer_opened:
            for i in range(self.num_objects):
                obj_pos_key = f"cube{i+1}_pos"
                if obj_pos_key in obs:
                    pos = obs[obj_pos_key]
                    # Check if object is inside drawer area
                    if (
                        self.drawer_area[0] <= pos[0] <= self.drawer_area[1]
                        and self.drawer_area[2] <= pos[1] <= self.drawer_area[3]
                        and self.drawer_area[4] <= pos[2] <= self.drawer_area[5]
                    ):
                        if i not in self.completed_objects:
                            reward += 0.5  # Reward for placing object
                            self.completed_objects.add(i)
                            self.objects_placed += 1

        return float(reward)

    def _is_task_completed(self, obs: Dict[str, Any]) -> bool:
        return len(self.completed_objects) == self.num_objects


class CupboardReward(TidyBotRewardCalculator):
    """Reward for cupboard organization tasks."""

    def __init__(self, num_objects: int):
        super().__init__("cupboard", num_objects)
        self.cupboard_opened = False
        self.objects_placed = 0
        self.cupboard_area = np.array(
            [0.7, 0.9, -0.2, 0.2, 0.3, 0.5]
        )  # [x_min, x_max, y_min, y_max, z_min, z_max]

    def _calculate_task_reward(self, obs: Dict[str, Any]) -> float:
        reward = 0.0

        # Check if cupboard is opened (simplified check)
        # In reality, this would check cupboard door state
        if self.episode_step > 50:  # Assume cupboard is opened after some steps
            if not self.cupboard_opened:
                reward += 1.0  # Reward for opening cupboard
                self.cupboard_opened = True

        # Check object placement in cupboard
        if self.cupboard_opened:
            for i in range(self.num_objects):
                obj_pos_key = f"cube{i+1}_pos"
                if obj_pos_key in obs:
                    pos = obs[obj_pos_key]
                    # Check if object is inside cupboard area
                    if (
                        self.cupboard_area[0] <= pos[0] <= self.cupboard_area[1]
                        and self.cupboard_area[2] <= pos[1] <= self.cupboard_area[3]
                        and self.cupboard_area[4] <= pos[2] <= self.cupboard_area[5]
                    ):
                        if i not in self.completed_objects:
                            reward += 0.5  # Reward for placing object
                            self.completed_objects.add(i)
                            self.objects_placed += 1

        return float(reward)

    def _is_task_completed(self, obs: Dict[str, Any]) -> bool:
        return len(self.completed_objects) == self.num_objects


class CabinetReward(TidyBotRewardCalculator):
    """Reward for cabinet manipulation tasks."""

    def __init__(self, num_objects: int):
        super().__init__("cabinet", num_objects)
        self.cabinet_opened = False
        self.objects_placed = 0
        self.cabinet_area = np.array(
            [0.0, 0.3, -0.2, 0.2, 0.2, 0.4]
        )  # [x_min, x_max, y_min, y_max, z_min, z_max]

    def _calculate_task_reward(self, obs: Dict[str, Any]) -> float:
        reward = 0.0

        # Check if cabinet is opened (handle position indicates cabinet state)
        if "left_handle_pos" in obs and "right_handle_pos" in obs:
            handle_pos = obs["left_handle_pos"]
            # Simplified check - in reality would need more sophisticated detection
            if handle_pos[0] < 0.3:  # Cabinet is opened
                if not self.cabinet_opened:
                    reward += 1.0  # Reward for opening cabinet
                    self.cabinet_opened = True

        # Check object placement in cabinet
        if self.cabinet_opened:
            for i in range(self.num_objects):
                obj_pos_key = f"cube{i+1}_pos"
                if obj_pos_key in obs:
                    pos = obs[obj_pos_key]
                    # Check if object is inside cabinet area
                    if (
                        self.cabinet_area[0] <= pos[0] <= self.cabinet_area[1]
                        and self.cabinet_area[2] <= pos[1] <= self.cabinet_area[3]
                        and self.cabinet_area[4] <= pos[2] <= self.cabinet_area[5]
                    ):
                        if i not in self.completed_objects:
                            reward += 0.5  # Reward for placing object
                            self.completed_objects.add(i)
                            self.objects_placed += 1

        return float(reward)

    def _is_task_completed(self, obs: Dict[str, Any]) -> bool:
        return len(self.completed_objects) == self.num_objects


class MotionPlanningReward(TidyBotRewardCalculator):
    """Reward for motion planning tasks."""

    def __init__(self, scene_type: str, num_objects: int):
        super().__init__(scene_type, num_objects)
        self.target_locations = self._get_target_locations()
        self.objects_placed = 0

    def _get_target_locations(self) -> List[np.ndarray]:
        """Get target locations based on scene type."""
        if self.scene_type == "table":
            return [np.array([0.6, 0.4, 0.1]) for _ in range(self.num_objects)]
        if self.scene_type == "cupboard":
            return [
                np.array([0.8, 0.08, 0.38]),  # Center position
                np.array([0.8, -0.08, 0.38]),  # Left position
                np.array([0.73, 0, 0.38]),  # Right position
            ][: self.num_objects]
        if self.scene_type == "cabinet":
            return [np.array([0.1, 0.0, 0.25]) for _ in range(self.num_objects)]
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
    if scene_type == "table":
        return TableStackingReward(num_objects)
    if scene_type == "drawer":
        return DrawerReward(num_objects)
    if scene_type == "cupboard":
        return CupboardReward(num_objects)
    if scene_type == "cabinet":
        return CabinetReward(num_objects)
    # Default to motion planning reward
    return MotionPlanningReward(scene_type, num_objects)
