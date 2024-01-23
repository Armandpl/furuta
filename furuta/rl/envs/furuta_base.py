from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from furuta.utils import ALPHA, ALPHA_DOT, THETA, THETA_DOT, Timing


def alpha_reward(state):
    return (1 + -np.cos(state[ALPHA])) / 2


def alpha_theta_reward(state):
    return alpha_reward(state) + (1 + np.cos(state[THETA])) / 2


REWARDS = {
    "alpha": alpha_reward,
    "alpha_theta": alpha_theta_reward,
}


class FurutaBase(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 50,  # TODO should this be the same as the control freq/sim dt?
    }

    def __init__(
        self,
        control_freq,
        reward,
        angle_limits=[np.pi, np.pi],  # used to help convergence?
        speed_limits=[60, 400],  # used to avoid damaging the real robot or diverging sim
        render_mode="rgb_array",
    ):
        self.render_mode = render_mode

        self.timing = Timing(control_freq)
        self._state = None
        self.reward = reward

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None

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

        # if terminated:
        #     rwd -= self.reward.oob_penalty

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
        super().reset(seed=seed)

    def _update_state(self, a):
        raise NotImplementedError

    def render(self):
        # https://github.com/Farama-Foundation/Gymnasium/blob/6baf8708bfb08e37ce3027b529193169eaa230fd/gymnasium/envs/classic_control/cartpole.py#L229
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

        # make sure theta stays between -pi and pi
        theta = (x[THETA] % (2 * np.pi)) - np.pi
        cartx = theta * scale + self.screen_width / 2.0  # MIDDLE OF CART
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
