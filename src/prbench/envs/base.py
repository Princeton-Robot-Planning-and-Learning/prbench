"""Base class for prbench environments with human input support."""

from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


class HumanInputEnv(ABC):
    """Base class for environments that support human input for demo collection."""

    @abstractmethod
    def get_human_input_mapping(self) -> Dict[str, Any]:
        """Get the mapping from human inputs to actions.
        
        Returns:
            Dictionary containing:
            - 'key_mappings': Dict mapping key names to action indices and values
            - 'description': String describing the controls
            - 'action_bounds': List of (min, max) tuples for each action dimension
        """
        pass

    def map_human_input_to_action(self, keys_pressed: set[str]) -> np.ndarray:
        """Map pressed keys to an action vector.
        
        Args:
            keys_pressed: Set of currently pressed key names
            
        Returns:
            Action array with the same shape as self.action_space
        """
        mapping = self.get_human_input_mapping()
        key_mappings = mapping['key_mappings']
        action_bounds = mapping['action_bounds']
        
        # Initialize action with zeros
        action = np.zeros(len(action_bounds), dtype=np.float32)
        
        # Apply key mappings
        for key, (action_idx, value) in key_mappings.items():
            if key in keys_pressed:
                action[action_idx] = value
                
        # Clip to bounds
        for i, (min_val, max_val) in enumerate(action_bounds):
            action[i] = np.clip(action[i], min_val, max_val)
            
        return action 