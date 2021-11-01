import gym
from gym.spaces import Box
import numpy as np


class GentlyTerminating(gym.Wrapper):
    """
    This env wrapper sends zero command to the robot when an episode is done.
    """
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            self.env.step(np.zeros(self.env.action_space.shape))
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()


class HistoryWrapper(gym.Wrapper):
    """
    Track history of observations for given amount of steps
    Initial steps are zero-filled
    """
    def __init__(self, env, steps):
        super(HistoryWrapper, self).__init__(env)
        self.steps = steps

        # concat obs with action
        self.step_low = np.concatenate([self.observation_space.low,
                                        self.action_space.low])
        self.step_high = np.concatenate([self.observation_space.high,
                                         self.action_space.high])

        # stack for each step
        obs_low = np.tile(self.step_low, (self.steps, 1))
        obs_high = np.tile(self.step_high, (self.steps, 1))

        self.observation_space = Box(low=obs_low, high=obs_high)

        self.history = self._make_history()

    def _make_history(self):
        return [np.zeros_like(self.step_low) for _ in range(self.steps)]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.history.pop(0)

        obs = np.concatenate([obs, action])
        self.history.append(obs)

        return np.array(self.history), reward, done, info

    def reset(self):
        self.history = self._make_history()
        print("history before: ", self.history)
        self.history.pop(0)
        obs = np.concatenate([self.env.reset(),
                              np.zeros_like(self.env.action_space.low)])
        self.history.append(obs)
        print("obs: ", obs)
        print("history: ", self.history)
        return np.array(self.history)
