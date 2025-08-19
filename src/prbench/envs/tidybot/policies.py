"""Policy wrappers and interfaces for TidyBot.

This module defines a light-weight `Policy` interface and a concrete wrapper
`MotionPlannerPolicyMPWrapper` around the motion-planner policy implemented in
`agent/mp_policy.py`. The wrapper exposes a simple reset/step API compatible
with the environment and allows small configuration tweaks.
"""

from typing import Any, Dict, Union

from prbench.envs.tidybot.agent.mp_policy import MotionPlannerPolicy as MotionPlannerPolicyMP


class Policy:
    """Abstract policy interface.

    Concrete policies should implement `reset` and `step`. The `step` method
    returns an action dictionary compatible with the TidyBot environment, a string
    signal, or `None` when no action should be taken.
    """

    def reset(self) -> None:
        """Reset internal policy state (start of an episode)."""
        raise NotImplementedError

    def step(self, obs: Dict[str, Any]) -> Union[Dict[str, Any], str, None]:
        """Compute the next action given an observation.

        Returns a dict action, a control string (e.g., to request a reset), or
        `None` if no action should be executed this step.
        """
        raise NotImplementedError


# Motion planner policy from mp_policy.py (agent)
class MotionPlannerPolicyMPWrapper(Policy):
    """Wrapper around the motion-planner policy.

    Bridges the agent implementation in `agent/mp_policy.py` to the generic
    `Policy` interface, and exposes a small set of configuration hooks.
    """

    def __init__(self, custom_grasp: bool = False) -> None:
        """Initialize the wrapper.

        Args:
            custom_grasp: Enable experimental grasping behavior in the wrapped policy.
        """
        self.impl: MotionPlannerPolicyMP = MotionPlannerPolicyMP(
            custom_grasp=custom_grasp
        )
        self.impl.PLACEMENT_X_OFFSET = 0.1
        self.impl.PLACEMENT_Y_OFFSET = 0.1
        self.impl.PLACEMENT_Z_OFFSET = 0.2
        self.episode_ended: bool = False

    def reset(self) -> None:
        """Reset the wrapped motion-planner policy and episode flags."""
        self.impl.reset()
        self.episode_ended = False

    def step(self, obs: Dict[str, Any]) -> Union[Dict[str, Any], str, None]:
        """Delegate to the wrapped policy and mirror its episode status."""
        action = self.impl.step(obs)
        self.episode_ended = getattr(self.impl, "episode_ended", False)
        return action
