import time

import gym
from gym.spaces import Box
import numpy as np

import wandb


class GentlyTerminating(gym.Wrapper):
    """
    This env wrapper sends zero command to the robot when an episode is done.
    """
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            # TODO: find how to bypass sb3 monitor: tried to step env that needs reset
            # self.env.step(np.zeros(self.env.action_space.shape))
            # maybe by changing the wrappers order
            print("episode done, killing motor.")
            self.env.motor.set_speed(0)
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()


class ControlFrequency(gym.Wrapper):
    """
    Enforce a sleeping time (dt) between each step.
    """
    def __init__(self, env, dt):
        super(ControlFrequency, self).__init__(env)
        self.dt = dt

    def step(self, action):
        current = time.time()
        loop_time = 0
        if self.last is not None:
            loop_time = current - self.last
            sleeping_time = self.dt - loop_time

            if sleeping_time > 0:
                time.sleep(sleeping_time)
            else:
                print("warning, loop time > dt")

        self.last = time.time()
        obs, reward, done, info = self.env.step(action)
        # TODO: log this only if debug enabled
        # wandb.log({**info, **{"loop time": loop_time}})

        return obs, reward, done, info

    def reset(self):
        self.last = None
        return self.env.reset()


class HistoryWrapper(gym.Wrapper):
    """
    Track history of observations for given amount of steps
    Initial steps are zero-filled
    """
    def __init__(self, env, steps, use_continuity_cost):
        super(HistoryWrapper, self).__init__(env)
        self.steps = steps
        self.use_continuity_cost = use_continuity_cost

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

    def _continuity_cost(self, obs):
        action = obs[-1][-1]
        last_action = obs[-2][-1]
        continuity_cost = np.power((action-last_action), 2).sum()

        return continuity_cost

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.history.pop(0)

        obs = np.concatenate([obs, action])
        self.history.append(obs)
        obs = np.array(self.history)

        if self.use_continuity_cost:
            reward -= self._continuity_cost(obs)

        return obs, reward, done, info

    def reset(self):
        self.history = self._make_history()
        self.history.pop(0)
        obs = np.concatenate([self.env.reset(),
                              np.zeros_like(self.env.action_space.low)])
        self.history.append(obs)
        return np.array(self.history)
