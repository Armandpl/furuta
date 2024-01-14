from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from furuta.utils import ALPHA, ALPHA_DOT, THETA, THETA_DOT, Timing


def alpha_reward(state):
    try:
        rwd = (1 + -np.cos(state[ALPHA])) / 2
    except Exception as e:  # TODO why would this fail?
        print(e)
        print(state)

    return rwd


def alpha_theta_reward(state):
    return alpha_reward(state) + (1 + np.cos(state[THETA])) / 2


REWARDS = {
    "alpha": alpha_reward,
    "alpha_theta": alpha_theta_reward,
}


class FurutaBase(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 50,  # TODO should this be the same as the control freq?
    }  # TODO add headless mode?

    def __init__(self, control_freq, reward, state_limits=None, render_mode="rgb_array"):
        self.render_mode = render_mode

        self.timing = Timing(control_freq)
        self._state = None
        self.reward = reward

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self._reward_func = REWARDS[self.reward]

        act_max = np.array([1.0], dtype=np.float32)

        if state_limits:
            self.state_max = np.array(state_limits, dtype=np.float32)
        else:
            self.state_max = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)

        obs_max = np.array(
            [1.0, 1.0, 1.0, 1.0, self.state_max[2], self.state_max[3]], dtype=np.float32
        )

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
        # TODO this is slow, do we even need it?
        # sb3 knows the env action space, probably it won't pass invalid actions

        # assert a is not None, "Action should be not None"
        # assert isinstance(a, np.ndarray), "The action should be a ndarray"
        # assert np.all(not np.isnan(a)), "Action NaN is not a valid action"
        # assert a.ndim == 1, "The action = {a} must be 1d but the input is {a.ndim}d"
        # err_msg = f"{action!r} ({type(action)}) invalid"

        # assert self.action_space.contains(action), "Action is not in action space"
        # assert self._state is not None, "Call reset before using step method."

        # first read the robot/sim state
        rwd = self._reward_func(self._state)
        obs = self.get_obs()

        # we use this info dict to log the state in mcap files
        info = {
            "motor_angle": float(self._state[THETA]),
            "pendulum_angle": float(self._state[ALPHA]),
            "motor_angle_velocity": float(self._state[THETA_DOT]),
            "pendulum_angle_velocity": float(self._state[ALPHA_DOT]),
            "reward": float(rwd),
            "action": float(action),
        }

        # then take action/step the sim
        self._update_state(action[0])

        terminated = not self.state_space.contains(self._state)

        # if terminated:
        #     rwd -= self.reward.oob_penalty

        info["terminated"] = bool(terminated)
        truncated = False

        return obs, rwd, terminated, truncated, info

    def get_obs(self):
        return np.float32(
            [
                # TODO maybe call cos, sin at once? save a bit of time
                np.cos(self._state[THETA]),
                np.sin(self._state[THETA]),
                np.cos(self._state[ALPHA]),
                np.sin(self._state[ALPHA]),
                self._state[THETA_DOT],
                self._state[ALPHA_DOT],
            ]
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        raise NotImplementedError

    def _update_state(self, a):
        raise NotImplementedError

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        import pygame
        from pygame import gfxdraw

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = 2 * np.pi
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * 0.5)  # TODO use the right pole len?
        cartwidth = 50.0
        cartheight = 30.0

        if self._state is None:
            return None

        x = self._state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[THETA] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(x[ALPHA] + np.pi)
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
