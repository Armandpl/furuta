from collections import namedtuple
from dataclasses import dataclass
from math import cos, sin

import gym
import numpy as np
from gym.spaces import Box

from furuta_gym.utils import ALPHA, ALPHA_DOT, THETA, THETA_DOT, Timing


def alpha_reward(state):
    try:
        rwd = (1 + -cos(state[ALPHA])) / 2
    except Exception as e:
        print(e)
        print(state)

    return rwd


def alpha_theta_reward(state):
    return alpha_reward(state) + (1 + cos(state[THETA])) / 2


REWARDS = {
    "alpha": alpha_reward,
    "alpha_theta": alpha_theta_reward,
}


class FurutaBase(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}  # TODO add headless mode

    def __init__(self, fs, reward, state_limits=None):

        self.timing = Timing(fs)
        self.viewer = None
        self._state = None
        self.reward = reward

        self._reward_func = REWARDS[self.reward.name]

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
            # labels=('theta', 'alpha', 'theta_dot', 'alpha_dot'),
            low=-self.state_max,
            high=self.state_max,
            dtype=np.float32,
        )

        self.observation_space = Box(
            # labels=('cos_th', 'sin_th', 'cos_al', 'sin_al', 'th_d', 'al_d'),
            low=-obs_max,
            high=obs_max,
            dtype=np.float32,
        )

        self.action_space = Box(
            # labels=('action',),
            low=-act_max,
            high=act_max,
            dtype=np.float32,
        )

    @profile
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

        info = {
            "motor_angle": float(self._state[THETA]),
            "pendulum_angle": float(self._state[ALPHA]),
            "motor_angle_velocity": float(self._state[THETA_DOT]),
            "pendulum_angle_velocity": float(self._state[ALPHA_DOT]),
            "reward": float(rwd),
            "action": float(action),
        }

        # then take action
        self._update_state(action[0])

        done = not self.state_space.contains(self._state)
        if done:
            rwd -= self.reward.oob_penalty

        info["done"] = bool(done)

        return obs, rwd, done, info

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

    def reset(self):
        raise NotImplementedError

    def _update_state(self, a):
        raise NotImplementedError

    def render(self, mode="rgb_array"):
        if self.viewer is None:
            self.viewer = CartPoleSwingUpViewer(world_width=(2 * np.pi))

        if self._state is None:
            return None

        self.viewer.update(self._state)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")


@dataclass(frozen=True)
class CartParams:
    """Parameters defining the Cart."""

    width: float = 1 / 3
    height: float = 1 / 6
    mass: float = 0.5


@dataclass(frozen=True)
class PoleParams:
    """Parameters defining the Pole."""

    width: float = 0.05
    length: float = 0.6
    mass: float = 0.5


Screen = namedtuple("Screen", "width height")


class CartPoleSwingUpViewer:
    """Class that encapsulates all the variables and objectecs needed to render a
    CartPoleSwingUpEnv.

    It handles all the initialization and updating of each object on screen and handles calls to
    the underlying gym.envs.classic_control.rendering.Viewer instance.
    """

    screen = Screen(width=600, height=400)

    def __init__(self, world_width):
        # TODO: make sure that's not redundant
        import pyglet
        from gym.envs.classic_control import rendering

        pyglet.options["headless"] = True  # noqa: F821

        cart = CartParams()
        pole = PoleParams()

        self.cart = cart
        self.pole = pole

        self.world_width = world_width
        screen = self.screen
        scale = screen.width / self.world_width
        cartwidth, cartheight = scale * cart.width, scale * cart.height
        polewidth, polelength = scale * pole.width, scale * pole.length
        self.viewer = rendering.Viewer(screen.width, screen.height)
        self.transforms = {
            "cart": rendering.Transform(),
            "pole": rendering.Transform(translation=(0, 0)),
            "pole_bob": rendering.Transform(),
            "wheel_l": rendering.Transform(translation=(-cartwidth / 2, -cartheight / 2)),
            "wheel_r": rendering.Transform(translation=(cartwidth / 2, -cartheight / 2)),
        }

        self._init_track(rendering, cartheight)
        self._init_cart(rendering, cartwidth, cartheight)
        self._init_wheels(rendering, cartheight)
        self._init_pole(rendering, polewidth, polelength)
        self._init_axle(rendering, polewidth)
        # Make another circle on the top of the pole
        self._init_pole_bob(rendering, polewidth)

    def _init_track(self, rendering, cartheight):
        screen = self.screen
        carty = screen.height / 2
        track_height = carty - cartheight / 2 - cartheight / 4
        track = rendering.Line((0, track_height), (screen.width, track_height))
        track.set_color(0, 0, 0)
        self.viewer.add_geom(track)

    def _init_cart(self, rendering, cartwidth, cartheight):
        lef, rig, top, bot = (
            -cartwidth / 2,
            cartwidth / 2,
            cartheight / 2,
            -cartheight / 2,
        )
        cart = rendering.FilledPolygon([(lef, bot), (lef, top), (rig, top), (rig, bot)])
        cart.add_attr(self.transforms["cart"])
        cart.set_color(1, 0, 0)
        self.viewer.add_geom(cart)

    def _init_pole(self, rendering, polewidth, polelength):
        lef, rig, top, bot = (
            -polewidth / 2,
            polewidth / 2,
            polelength - polewidth / 2,
            -polewidth / 2,
        )
        pole = rendering.FilledPolygon([(lef, bot), (lef, top), (rig, top), (rig, bot)])
        pole.set_color(0, 0, 1)
        pole.add_attr(self.transforms["pole"])
        pole.add_attr(self.transforms["cart"])
        self.viewer.add_geom(pole)

    def _init_axle(self, rendering, polewidth):
        axle = rendering.make_circle(polewidth / 2)
        axle.add_attr(self.transforms["pole"])
        axle.add_attr(self.transforms["cart"])
        axle.set_color(0.1, 1, 1)
        self.viewer.add_geom(axle)

    def _init_pole_bob(self, rendering, polewidth):
        pole_bob = rendering.make_circle(polewidth / 2)
        pole_bob.add_attr(self.transforms["pole_bob"])
        pole_bob.add_attr(self.transforms["pole"])
        pole_bob.add_attr(self.transforms["cart"])
        pole_bob.set_color(0, 0, 0)
        self.viewer.add_geom(pole_bob)

    def _init_wheels(self, rendering, cartheight):
        wheel_l = rendering.make_circle(cartheight / 4)
        wheel_r = rendering.make_circle(cartheight / 4)
        wheel_l.add_attr(self.transforms["wheel_l"])
        wheel_l.add_attr(self.transforms["cart"])
        wheel_r.add_attr(self.transforms["wheel_r"])
        wheel_r.add_attr(self.transforms["cart"])
        wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
        wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
        self.viewer.add_geom(wheel_l)
        self.viewer.add_geom(wheel_r)

    def update(self, state):
        """Updates the positions of the objects on screen."""
        screen = self.screen
        scale = screen.width / self.world_width

        th, al, _, _ = state
        al = (al - np.pi) % (2 * np.pi)  # change angle origin

        # keep th between -pi and pi
        th = (th + np.pi) % (2 * np.pi) - np.pi

        # use motor angle (theta) as cart x position
        cartx = th * scale + screen.width / 2.0  # MIDDLE OF CART
        carty = screen.height / 2

        self.transforms["cart"].set_translation(cartx, carty)
        self.transforms["pole"].set_rotation(al)
        self.transforms["pole_bob"].set_translation(
            -self.pole.length * np.sin(al), self.pole.length * np.cos(al)
        )

    def render(self, *args, **kwargs):
        """Forwards the call to the underlying Viewer instance."""
        return self.viewer.render(*args, **kwargs)

    def close(self):
        """Closes the underlying Viewer instance."""
        self.viewer.close()
