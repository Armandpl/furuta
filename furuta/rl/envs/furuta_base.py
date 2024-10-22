from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from furuta.utils import ALPHA, ALPHA_DOT, THETA, THETA_DOT, Timing
from furuta.viewer import Viewer2D


def exp_alpha_theta_reward(state, exp=2):
    al_rew = exp_alpha_reward(state, exp)
    th_rew = theta_reward(state)
    return al_rew * th_rew


def exp_alpha_reward(state, exp=2):
    al = np.mod((state[ALPHA] + np.pi), 2 * np.pi) - np.pi  # between -pi and pi
    al_rew = np.abs(al) / np.pi  # 0 at 0, 1 at pi
    al_rew = (np.exp(al_rew * exp) - np.exp(0)) / np.exp(exp)
    return al_rew


def alpha_theta_reward(state):
    return alpha_reward(state) * theta_reward(state)


def alpha_reward(state):
    return (1 + -np.cos(state[ALPHA])) / 2


def theta_reward(state):
    # return (1 + np.cos(state[THETA])) / 2
    theta_rew = (np.cos(state[THETA] + np.pi) + 1) / 2
    return 1 - theta_rew**2


REWARDS = {
    "cos_alpha": alpha_theta_reward,
    "exp_alpha_2": lambda x: exp_alpha_theta_reward(x, exp=2),
    "exp_alpha_3": lambda x: exp_alpha_theta_reward(x, exp=3),
    "exp_alpha_4": lambda x: exp_alpha_theta_reward(x, exp=4),
    "exp_alpha_6": lambda x: exp_alpha_theta_reward(x, exp=6),
}


class FurutaBase(gym.Env):
    def __init__(
        self,
        control_freq,
        reward,
        angle_limits=[np.pi, np.pi],  # used to help convergence?
        speed_limits=[60, 400],  # used to avoid damaging the real robot or diverging sim
        render_mode="rgb_array",
    ):
        self.viewer = Viewer2D(control_freq, render_mode)

        self.timing = Timing(control_freq)
        self._state = None
        self.reward = reward

        self._reward_func = REWARDS[self.reward]

        act_max = np.array([1.0], dtype=np.float32)

        angle_limits = np.array(angle_limits, dtype=np.float32)
        speed_limits = np.array(speed_limits, dtype=np.float32)

        # replace none values with inf
        angle_limits = np.where(np.isnan(angle_limits), np.inf, angle_limits)  # noqa
        speed_limits = np.where(np.isnan(speed_limits), np.inf, speed_limits)  # noqa

        self.state_max = np.concatenate([angle_limits, speed_limits])

        # max obs based on max speeds measured on the robot
        # in sim the speeds spike at 30 rad/s when trained
        # selected 50 rad/s to be safe bc its probably higher during training
        # it's also ok if the speeds exceed theses values as we only use them for rescaling
        # and it's okay if the nn sees values a little bit above 1
        # obs is [cos(th), sin(th), cos(al), sin(al), th_d, al_d)]
        obs_max = np.array([1.0, 1.0, 1.0, 1.0, 30, 30], dtype=np.float32)

        # if limit on angles, add them to the obs
        if not np.isinf(self.state_max[ALPHA]):
            obs_max = np.concatenate([np.array([self.state_max[ALPHA]]), obs_max])
        if not np.isinf(self.state_max[THETA]):
            obs_max = np.concatenate([np.array([self.state_max[THETA]]), obs_max])

        # Spaces
        self.state_space = Box(
            # ('theta', 'alpha', 'theta_dot', 'alpha_dot'),
            low=-self.state_max,
            high=self.state_max,
            dtype=np.float32,
        )

        self.observation_space = Box(
            # ('cos_th', 'sin_th', 'cos_al', 'sin_al', 'th_d', 'al_d'),
            low=-obs_max,
            high=obs_max,
            dtype=np.float32,
        )

        self.action_space = Box(
            # ('action',),
            low=-act_max,
            high=act_max,
            dtype=np.float32,
        )

    def step(self, action):
        # first read the robot/sim state
        rwd = self._reward_func(self._state)
        obs = self.get_obs()

        # then take action/step the sim
        self._update_state(action[0])

        terminated = not self.state_space.contains(self._state)
        truncated = False

        return obs, rwd, terminated, truncated, {}

    def get_obs(self):
        obs = np.float32(
            [
                np.cos(self._state[THETA]),
                np.sin(self._state[THETA]),
                np.cos(self._state[ALPHA]),
                np.sin(self._state[ALPHA]),
                self._state[THETA_DOT],
                self._state[ALPHA_DOT],
            ]
        )
        if not np.isinf(self.state_max[ALPHA]):
            obs = np.concatenate([np.array([self._state[ALPHA]]), obs])
        if not np.isinf(self.state_max[THETA]):
            obs = np.concatenate([np.array([self._state[THETA]]), obs])

        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)

    def _update_state(self, a):
        raise NotImplementedError

    def render(self):
        return self.viewer.display(self._state)

    def close(self):
        self.viewer.close()
