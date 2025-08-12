import xml.etree.ElementTree as ET

import mujoco
import numpy as np

# This value is then used by the physics engine to determine how much time
# to simulate for each step.
SIMULATION_TIMESTEP = 0.002  # (in seconds)


class MujocoEnv:
    """This is the base class for environments that use MuJoCo for
    simulation."""

    def __init__(self, xml_string, control_frequency, horizon=1000):
        """
        Args:
            xml_string: A string containing the MuJoCo XML model.
            control_frequency: Frequency at which control actions are applied (in Hz).
            horizon: Maximum number of steps per episode.
        """
        self.sim = MjSim(xml_string)

        self.control_frequency = control_frequency
        self.horizon = horizon
        self.timestep = 0
        self.done = False

    def reset(self):
        """Reset the environment."""
        self.sim.reset()
        self.sim.forward()
        return self._get_obs()

    def _pre_action(self, action):
        """Do any preprocessing before taking an action.

        Args:
            action (np.array): Action to execute within the environment.
        """
        if action is not None:
            self.sim.data.ctrl[:] = action

    def _post_action(self, action):
        """Do any housekeeping after taking an action.

        Args:
            action (np.array): Action to execute within the environment.

        Returns:
            reward (float): Reward from the environment.
            done (bool): Whether the episode is completed.
            info (dict): Additional information.
        """
        reward = self.reward(action)
        done = False  # Default to not done unless overridden
        info = {}
        return reward, done, info

    def reward(self):
        """Compute the reward for the current state and action.

        Returns:
            reward (float): Computed reward.
        """
        raise NotImplementedError

    def step(self, action=None):
        """Step the environment.

        Args:
            action: Optional action to apply before stepping.

        Returns:
            obs: Observation after step.
            reward: Reward from the environment.
            done: Whether the episode is completed.
            info: Additional information.
        """
        if self.done:
            raise ValueError("Executing action in a terminated episode.")

        self.timestep += 1

        # Step the simulation with the same action until the control frequency is reached
        for _ in range(int(SIMULATION_TIMESTEP / self.control_frequency)):
            self._pre_action(action)
            self.sim.forward()
            self.sim.step()

        # Post-action processing
        reward, self.done, info = self._post_action(action)

        # Check if the episode is done due to horizon
        self.done = self.done or (self.timestep >= self.horizon)

        return self._get_obs(), reward, self.done, info

    def _get_obs(self):
        """Get the current observation."""
        # return a copy of qpos and qvel as observation
        return {
            "qpos": np.copy(self.sim.data.qpos),
            "qvel": np.copy(self.sim.data.qvel),
        }


class MjSim:
    """A simplified MjSim class for MuJoCo simulation."""

    def __init__(self, xml_string):
        """
        Args:
            xml_string: A string containing the MuJoCo XML model.
        """

        xml_string = self._set_simulation_timestep(xml_string)

        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        # Offscreen render context object
        self._render_context_offscreen = None

    def _set_simulation_timestep(self, xml_string):
        """Set the simulation timestep in the XML string.

        Args:
            xml_string: A string containing the MuJoCo XML model.

        Returns:
            Modified XML string with updated timestep.
        """
        # Parse the XML string
        root = ET.fromstring(xml_string)

        # Find the <option> tag and set its timestep attribute
        option = root.find("option")
        if option is not None:
            option.set("timestep", str(SIMULATION_TIMESTEP))
        else:
            # If <option> tag does not exist, create it and insert as first child
            option = ET.Element("option", {"timestep": str(SIMULATION_TIMESTEP)})
            root.insert(0, option)

        # Convert the modified XML tree back to a string
        return ET.tostring(root, encoding="unicode")

    def reset(self):
        """Reset the simulation."""
        mujoco.mj_resetData(self.model, self.data)

    def forward(self):
        """Synchronize derived quantities."""
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        """Step the simulation."""
        mujoco.mj_step(self.model, self.data)
