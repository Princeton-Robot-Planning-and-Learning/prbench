# Author: Jimmy Wu
# Date: October 2024

import logging
import math
import time
from enum import Enum, auto

import numpy as np
from scipy.spatial.transform import Rotation as R

from .agent.mp_policy import (
    MotionPlannerPolicy as MotionPlannerPolicyMP,)  # Import new MP policy
from .agent.open_policy import (
    CloseCabinetPolicy as CloseCabinetPolicy,)  # Import new MP policy for second phase
from .agent.open_policy import (
    MotionPlannerPolicyCabinetMP as MotionPlannerPolicyCabinetMP,)  # Import new MP policy
from .agent.open_policy import (
    MotionPlannerPolicyCabinetMP_1 as MotionPlannerPolicyCabinetMP_1,)  # Import new MP policy for second phase
from .agent.stack_policies import MotionPlannerPolicyStack  # <-- Add this import
from .agent.stack_policies import (  # <-- Add this import for table stacking; <-- Add this import for drawer stacking; <-- Add this import for cupboard stacking
    MotionPlannerPolicyStackCupboard,
    MotionPlannerPolicyStackDrawer,
    MotionPlannerPolicyStackTable,
)
from .constants import (
    POLICY_IMAGE_HEIGHT,
    POLICY_IMAGE_WIDTH,
)


class Policy:
    def reset(self):
        raise NotImplementedError

    def step(self, obs):
        raise NotImplementedError














# Stacking motion planner policy (stacking three objects)
class MotionPlannerPolicyStackWrapper(Policy):
    def __init__(self):
        self.impl = MotionPlannerPolicyStack()

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Stacking three cubes using two sequential stack policies
class MotionPlannerPolicyStackThreeWrapper(Policy):
    def __init__(self):
        self.stack1 = MotionPlannerPolicyStack()
        self.stack2 = MotionPlannerPolicyStack()
        self.phase = 0  # 0: first stack, 1: second stack, 2: done
        self.episode_ended = False

    def reset(self):
        self.stack1.reset()
        self.stack2.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None

        # Phase 0: stack first two cubes
        if self.phase == 0:
            action = self.stack1.step(obs)
            if self.stack1.episode_ended:
                # Prepare for second stacking: adjust stack2's placement height
                # Find the current top cube's position and set stack2's STACK_HEIGHT_OFFSET
                # We'll use the same logic as stack1, but increase the offset
                # Get the last stack location from stack1
                if (
                    hasattr(self.stack1, "stack_location")
                    and self.stack1.stack_location is not None
                ):
                    # The new stack height should be one cube height above the previous stack
                    # Assume cube height is the same as stack1.STACK_HEIGHT_OFFSET
                    cube_height = self.stack1.STACK_HEIGHT_OFFSET
                    self.stack2.STACK_HEIGHT_OFFSET = 1.5 * cube_height
                self.phase = 1
                self.stack2.reset()  # Ensure stack2 is ready
            return action
        # Phase 1: stack third cube on top
        elif self.phase == 1:
            action = self.stack2.step(obs)
            if self.stack2.episode_ended:
                self.phase = 2
                self.episode_ended = True
            return action
        # Phase 2: done
        else:
            return None


# Table stacking policy wrapper
class MotionPlannerPolicyStackTableWrapper(Policy):
    def __init__(self):
        self.impl = MotionPlannerPolicyStackTable()

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Table stacking three cubes using two sequential table stack policies
class MotionPlannerPolicyStackTableThreeWrapper(Policy):
    def __init__(self):
        self.stack1 = MotionPlannerPolicyStackTable()
        self.stack2 = MotionPlannerPolicyStackTable()
        self.phase = 0  # 0: first stack, 1: second stack, 2: done
        self.episode_ended = False

    def reset(self):
        self.stack1.reset()
        self.stack2.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None

        # Phase 0: stack first two cubes
        if self.phase == 0:
            action = self.stack1.step(obs)
            if self.stack1.episode_ended:
                # Prepare for second stacking: adjust stack2's placement height
                # Find the current top cube's position and set stack2's STACK_HEIGHT_OFFSET
                # We'll use the same logic as stack1, but increase the offset
                # Get the last stack location from stack1
                if (
                    hasattr(self.stack1, "stack_location")
                    and self.stack1.stack_location is not None
                ):
                    # The new stack height should be one cube height above the previous stack
                    # Assume cube height is the same as stack1.STACK_HEIGHT_OFFSET
                    cube_height = self.stack1.STACK_HEIGHT_OFFSET
                    self.stack2.STACK_HEIGHT_OFFSET = (
                        1.8 * cube_height
                    )  # Stack third cube on top of second
                self.phase = 1
                self.stack2.reset()  # Ensure stack2 is ready
            return action
        # Phase 1: stack third cube on top
        elif self.phase == 1:
            action = self.stack2.step(obs)
            if self.stack2.episode_ended:
                self.phase = 2
                self.episode_ended = True
            return action
        # Phase 2: done
        else:
            return None


# Drawer stacking policy wrapper
class MotionPlannerPolicyStackDrawerWrapper(Policy):
    def __init__(self):
        self.impl = MotionPlannerPolicyStackDrawer()

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Drawer stacking three cubes using two sequential drawer stack policies
class MotionPlannerPolicyStackDrawerThreeWrapper(Policy):
    def __init__(self):
        self.stack1 = MotionPlannerPolicyStackDrawer()
        self.stack2 = MotionPlannerPolicyStackDrawer()
        self.phase = 0  # 0: first stack, 1: second stack, 2: done
        self.episode_ended = False

    def reset(self):
        self.stack1.reset()
        self.stack2.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None

        # Phase 0: stack first two cubes
        if self.phase == 0:
            action = self.stack1.step(obs)
            if self.stack1.episode_ended:
                # Prepare for second stacking: adjust stack2's placement height
                if (
                    hasattr(self.stack1, "stack_location")
                    and self.stack1.stack_location is not None
                ):
                    cube_height = self.stack1.STACK_HEIGHT_OFFSET
                    self.stack2.STACK_HEIGHT_OFFSET = (
                        1.8 * cube_height
                    )  # Stack third cube on top of second
                self.phase = 1
                self.stack2.reset()  # Ensure stack2 is ready
            return action
        # Phase 1: stack third cube on top
        elif self.phase == 1:
            action = self.stack2.step(obs)
            if self.stack2.episode_ended:
                self.phase = 2
                self.episode_ended = True
            return action
        # Phase 2: done
        else:
            return None


# Cupboard stacking policy wrapper
class MotionPlannerPolicyStackCupboardWrapper(Policy):
    def __init__(self):
        self.impl = MotionPlannerPolicyStackCupboard()

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Cupboard stacking three cubes using two sequential cupboard stack policies
class MotionPlannerPolicyStackCupboardThreeWrapper(Policy):
    def __init__(self):
        self.stack1 = MotionPlannerPolicyStackCupboard()
        self.stack2 = MotionPlannerPolicyStackCupboard()
        self.phase = 0  # 0: first stack, 1: second stack, 2: done
        self.episode_ended = False

    def reset(self):
        self.stack1.reset()
        self.stack2.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None

        # Phase 0: stack first two cubes
        if self.phase == 0:
            action = self.stack1.step(obs)
            if self.stack1.episode_ended:
                # Prepare for second stacking: adjust stack2's placement height
                if (
                    hasattr(self.stack1, "stack_location")
                    and self.stack1.stack_location is not None
                ):
                    cube_height = self.stack1.STACK_HEIGHT_OFFSET
                    self.stack2.STACK_HEIGHT_OFFSET = (
                        1.8 * cube_height
                    )  # Stack third cube on top of second
                self.phase = 1
                self.stack2.reset()  # Ensure stack2 is ready
            return action
        # Phase 1: stack third cube on top
        elif self.phase == 1:
            action = self.stack2.step(obs)
            if self.stack2.episode_ended:
                self.phase = 2
                self.episode_ended = True
            return action
        # Phase 2: done
        else:
            return None


# Motion planner policy from mp_policy.py (agent)
class MotionPlannerPolicyMPWrapper(Policy):
    def __init__(self, custom_grasp=False):
        self.impl = MotionPlannerPolicyMP(custom_grasp=custom_grasp)
        self.impl.PLACEMENT_X_OFFSET = 0.1
        self.impl.PLACEMENT_Y_OFFSET = 0.1
        self.impl.PLACEMENT_Z_OFFSET = 0.2
        self.impl.target_location = np.array([0, 0, 0.5])

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Motion planner policy for cupboard environment
class MotionPlannerPolicyMPCupboardWrapper(Policy):
    def __init__(self, custom_grasp=False):
        self.impl = MotionPlannerPolicyMP(cupboard_mode=True, custom_grasp=custom_grasp)
        self.impl.PLACEMENT_X_OFFSET = 0.1
        self.impl.PLACEMENT_Y_OFFSET = 0.1
        self.impl.PLACEMENT_Z_OFFSET = 0.5
        self.impl.target_location = np.array([0.8, 0, 0.5])

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Motion planner policy for cabinet environment
class MotionPlannerPolicyMPCabinetWrapper(Policy):
    def __init__(self, custom_grasp=False):
        # Use cupboard_mode=True since cabinet environment is similar to cupboard
        self.impl = MotionPlannerPolicyCabinetMP(custom_grasp=custom_grasp)
        self.impl.PLACEMENT_X_OFFSET = 0.6  # Distance to cabinet
        self.impl.PLACEMENT_Y_OFFSET = 0.0  # Center alignment
        self.impl.PLACEMENT_Z_OFFSET = 0.25  # Cabinet shelf height
        self.impl.target_location = np.array([0, -0.1, 0.25])  # Cabinet position

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Motion planner policy for cabinet environment with two-phase execution
class MotionPlannerPolicyMPCabinetTwoPhaseWrapper(Policy):
    def __init__(self, custom_grasp=False, mp_configs=None):
        """
        Args:
            custom_grasp (bool): Whether to use custom grasp
            mp_configs (list): List of dicts, each specifying the class and kwargs for each phase, e.g.
                [
                    {'class': MotionPlannerPolicyCabinetMP, 'kwargs': {'custom_grasp': custom_grasp, 'open_left_cabinet': True}},
                    {'class': MotionPlannerPolicyCabinetMP_1, 'kwargs': {'custom_grasp': False, 'open_left_cabinet': True}},
                    ...
                ]
        """

        # First phase: MotionPlannerPolicyCabinetMP
        self.mp1 = MotionPlannerPolicyCabinetMP(
            custom_grasp=custom_grasp, open_left_cabinet=True
        )

        # Second phase: MotionPlannerPolicyCabinetMP_1
        self.mp2 = MotionPlannerPolicyCabinetMP_1(
            custom_grasp=False, open_left_cabinet=True
        )

        # First phase: MotionPlannerPolicyCabinetMP
        # self.mp3 = MotionPlannerPolicyCabinetMP(custom_grasp=custom_grasp, open_left_cabinet=False)

        # # Second phase: MotionPlannerPolicyCabinetMP_1
        # self.mp4 = MotionPlannerPolicyCabinetMP_1(custom_grasp=False, open_left_cabinet=False)

        # # First phase: MotionPlannerPolicyCabinetMP
        # self.mp5 = CloseCabinetPolicy(custom_grasp=False, close_left_cabinet=True)

        # # # Second phase: MotionPlannerPolicyCabinetMP_1
        # self.mp6 = CloseCabinetPolicy(custom_grasp=False, close_left_cabinet=False)

        self.mp3 = MotionPlannerPolicyMP(cupboard_mode=True, low_grasp=True)
        self.mp3.target_location = np.array([0.75, -0.15, 0.12])  # Center position
        self.mp3.PICK_APPROACH_HEIGHT_OFFSET = 0.05
        self.mp3.PLACE_APPROACH_HEIGHT_OFFSET = 0.05

        self.mp4 = MotionPlannerPolicyMP(cupboard_mode=True, low_grasp=True)
        self.mp4.target_location = np.array([0.75, -0.3, 0.12])  # Left position
        self.mp4.PICK_APPROACH_HEIGHT_OFFSET = 0.05
        self.mp4.PLACE_APPROACH_HEIGHT_OFFSET = 0.05

        self.mps = [self.mp1, self.mp2, self.mp3, self.mp4]

        # self.mps = [self.mp1, self.mp2, self.mp3, self.mp4, self.mp5, self.mp6]
        self.phase = 0
        self.episode_ended = False

    def reset(self):
        for mp in self.mps:
            mp.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None
        if self.phase < len(self.mps):
            action = self.mps[self.phase].step(obs)
            if getattr(self.mps[self.phase], "episode_ended", False):
                self.phase += 1
                if self.phase < len(self.mps):
                    self.mps[self.phase].reset()
                    print(f"Phase {self.phase} completed, starting next phase")
                else:
                    self.episode_ended = True
                    print(f"All phases completed, episode ended")
            return action
        else:
            return None


# Motion planner policy for three sequential placements
class MotionPlannerPolicyMPThreeWrapper(Policy):
    def __init__(self, custom_grasp=False):
        self.mp1 = MotionPlannerPolicyMP(custom_grasp=custom_grasp)
        self.mp1.target_location = np.array([0.8, 0, 0.2])
        self.mp2 = MotionPlannerPolicyMP(custom_grasp=custom_grasp)
        self.mp2.target_location = np.array([0.8, -0.1, 0.2])
        self.mp3 = MotionPlannerPolicyMP(custom_grasp=custom_grasp)
        self.mp3.target_location = np.array([0.8, 0.1, 0.2])
        self.phase = 0
        self.episode_ended = False

    def reset(self):
        self.mp1.reset()
        self.mp2.reset()
        self.mp3.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None
        if self.phase == 0:
            action = self.mp1.step(obs)
            if getattr(self.mp1, "episode_ended", False):
                self.phase = 1
                self.mp2.reset()
            return action
        elif self.phase == 1:
            action = self.mp2.step(obs)
            if getattr(self.mp2, "episode_ended", False):
                self.phase = 2
                self.mp3.reset()
            return action
        elif self.phase == 2:
            action = self.mp3.step(obs)
            if getattr(self.mp3, "episode_ended", False):
                self.phase = 3
                self.episode_ended = True
            return action
        else:
            return None


# Custom grasp policy wrapper for experimentation
class MotionPlannerPolicyCustomGraspWrapper(Policy):
    def __init__(self):
        self.impl = MotionPlannerPolicyMP(cupboard_mode=True, custom_grasp=True)
        # This wrapper is designed to work with cupboard_scene_objects_inside.xml
        # where objects are already placed inside the cupboard
        # Custom grasping parameters are now set automatically in the MP policy
        # Placement parameters
        self.impl.PLACEMENT_X_OFFSET = 0.1
        self.impl.PLACEMENT_Y_OFFSET = 0.1
        self.impl.PLACEMENT_Z_OFFSET = 0.5
        self.impl.GRASP_SUCCESS_THRESHOLD = 0.75
        self.impl.PICK_LOWER_DIST = 0.09
        self.impl.PICK_LIFT_DIST = 0.18
        self.impl.target_location = np.array([0.8, 0, 0.5])
        print("Custom grasp policy initialized with experimental parameters")
        print(
            "Designed for cupboard_scene_objects_inside.xml (objects already in cupboard)"
        )

    def reset(self):
        self.impl.reset()

    def step(self, obs):
        return self.impl.step(obs)


# Custom grasp policy wrapper for three sequential pick-place actions in cupboard environment
class MotionPlannerPolicyCustomGraspThreeWrapper(Policy):
    def __init__(self):
        # Create three separate motion planner instances for sequential actions
        self.mp1 = MotionPlannerPolicyMP(cupboard_mode=True, custom_grasp=True)
        self.mp1.GRASP_SUCCESS_THRESHOLD = 0.75
        self.mp1.PICK_LOWER_DIST = 0.09
        self.mp1.PICK_LIFT_DIST = 0.18
        self.mp1.target_location = np.array([0.8, 0.08, 0.38])  # Center position

        self.mp2 = MotionPlannerPolicyMP(cupboard_mode=True, custom_grasp=True)
        self.mp2.GRASP_SUCCESS_THRESHOLD = 0.75
        self.mp2.PICK_LOWER_DIST = 0.09
        self.mp2.PICK_LIFT_DIST = 0.18
        self.mp2.target_location = np.array([0.8, -0.08, 0.38])  # Left position

        self.mp3 = MotionPlannerPolicyMP(cupboard_mode=True, custom_grasp=True)
        self.mp3.GRASP_SUCCESS_THRESHOLD = 0.75
        self.mp3.PICK_LOWER_DIST = 0.09
        self.mp3.PICK_LIFT_DIST = 0.18
        self.mp3.target_location = np.array([0.73, 0, 0.38])  # Right position

        self.phase = (
            0  # 0: first pick-place, 1: second pick-place, 2: third pick-place, 3: done
        )
        self.episode_ended = False

        print("Custom grasp three policy initialized with experimental parameters")
        print(
            "Will execute three sequential pick-place actions in cupboard environment"
        )
        print("Target locations: center, left, right")

    def reset(self):
        self.mp1.reset()
        self.mp2.reset()
        self.mp3.reset()
        self.phase = 0
        self.episode_ended = False

    def step(self, obs):
        if self.episode_ended:
            return None

        # Phase 0: First pick-place action (center)
        if self.phase == 0:
            action = self.mp1.step(obs)
            if getattr(self.mp1, "episode_ended", False):
                print("First pick-place action completed, moving to second")
                self.phase = 1
                self.mp2.reset()
            return action

        # Phase 1: Second pick-place action (left)
        elif self.phase == 1:
            action = self.mp2.step(obs)
            if getattr(self.mp2, "episode_ended", False):
                print("Second pick-place action completed, moving to third")
                self.phase = 2
                self.mp3.reset()
            return action

        # Phase 2: Third pick-place action (right)
        elif self.phase == 2:
            action = self.mp3.step(obs)
            if getattr(self.mp3, "episode_ended", False):
                print("Third pick-place action completed, all tasks done")
                self.phase = 3
                self.episode_ended = True
            return action

        # Phase 3: done
        else:
            return None


# Motion planner policy for N sequential pick-place actions in cupboard environment
class MotionPlannerPolicyMPNCupboardWrapper(Policy):
    def __init__(self, target_locations=None, custom_grasp=False, grasp_params=None):
        """Initialize MotionPlannerPolicyMPNCupboardWrapper for N objects.

        Args:
            target_locations (list): List of target locations as numpy arrays [x, y, z]
                                   If None, uses default 3 locations
            custom_grasp (bool): Enable custom grasping parameters
            grasp_params (dict): Optional custom grasping parameters
        """
        # Default target locations if none provided
        if target_locations is None:
            target_locations = [
                np.array([0.8, 0.08, 0.38]),  # Center position
                np.array([0.8, -0.08, 0.38]),  # Left position
                np.array([0.73, 0, 0.38]),  # Right position
            ]

        self.target_locations = target_locations
        self.num_objects = len(target_locations)

        # Default grasping parameters
        default_grasp_params = {
            "GRASP_SUCCESS_THRESHOLD": 0.7,
            "PICK_LOWER_DIST": 0.09,
            "PICK_LIFT_DIST": 0.18,
        }

        # Override with custom parameters if provided
        if grasp_params:
            default_grasp_params.update(grasp_params)

        # Create motion planner instances for each object
        self.motion_planners = []
        for i, target_loc in enumerate(target_locations):
            mp = MotionPlannerPolicyMP(cupboard_mode=True, custom_grasp=True)
            mp.GRASP_SUCCESS_THRESHOLD = default_grasp_params["GRASP_SUCCESS_THRESHOLD"]
            mp.PICK_LOWER_DIST = default_grasp_params["PICK_LOWER_DIST"]
            mp.PICK_LIFT_DIST = default_grasp_params["PICK_LIFT_DIST"]
            mp.target_location = target_loc
            self.motion_planners.append(mp)

        self.current_phase = 0  # Current object being processed
        self.episode_ended = False

        print(
            f"MotionPlannerPolicyMPNCupboardWrapper initialized for {self.num_objects} objects"
        )
        print(
            f"Target locations: {[f'[{loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f}]' for loc in target_locations]}"
        )
        print(f"Custom grasp: {custom_grasp}")
        if grasp_params:
            print(f"Custom grasp parameters: {grasp_params}")

    def reset(self):
        """Reset all motion planners and episode state."""
        for mp in self.motion_planners:
            mp.reset()
        self.current_phase = 0
        self.episode_ended = False
        print(
            f"Reset MotionPlannerPolicyMPNCupboardWrapper - ready to process {self.num_objects} objects"
        )

    def step(self, obs):
        """Execute the current phase of the sequential pick-place task."""
        if self.episode_ended:
            return None

        # Check if we've completed all phases
        if self.current_phase >= self.num_objects:
            self.episode_ended = True
            print(f"All {self.num_objects} pick-place actions completed!")
            return None

        # Execute current motion planner
        current_mp = self.motion_planners[self.current_phase]
        action = current_mp.step(obs)

        # Check if current phase is complete
        if getattr(current_mp, "episode_ended", False):
            print(f"Phase {self.current_phase + 1}/{self.num_objects} completed")
            self.current_phase += 1

            # Reset next motion planner if there are more phases
            if self.current_phase < self.num_objects:
                print(f"Moving to phase {self.current_phase + 1}/{self.num_objects}")
                self.motion_planners[self.current_phase].reset()
            else:
                print("All phases completed!")

        return action



