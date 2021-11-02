from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import gym
from gym.utils import seeding

from .common import LabeledBox, Timing


class FurutaBase(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, fs, fs_ctrl, action_limiter, safety_th_lim, reward):

        self._state = None
        self.timing = Timing(fs, fs_ctrl)
        self.rwd_name = reward
        self.viewer = None

        act_max = np.array([1.0])
        state_max = np.array([2.0, 4.0 * np.pi, 30.0, 40.0])
        obs_max = np.array([1.0, 1.0, 1.0, 1.0, state_max[2], state_max[3]])

        # Spaces
        self.state_space = LabeledBox(
            labels=('theta', 'alpha', 'theta_dot', 'alpha_dot'),
            low=-state_max, high=state_max, dtype=np.float32)
        self.observation_space = LabeledBox(
            labels=('cos_th', 'sin_th', 'cos_al', 'sin_al', 'th_d', 'al_d'),
            low=-obs_max, high=obs_max, dtype=np.float32)
        self.action_space = LabeledBox(
            labels=('action',),
            low=-act_max, high=act_max, dtype=np.float32)

        # Function to ensure that state and action constraints are satisfied
        if action_limiter:
            self._lim_act = ActionLimiter(self.state_space,
                                          self.action_space,
                                          safety_th_lim)
        else:
            self._lim_act = None

        # Initialize random number generator
        self._np_random = None
        self.seed()

    def _ctrl_step(self, a):
        x = self._state
        a_cmd = None

        for _ in range(self.timing.n_sim_per_ctrl):
            if self.action_limiter:
                a_cmd = self._lim_act(x, a)
            else:
                a_cmd = a
            x = self._update_state(a_cmd)

        return x, a_cmd  # return the last applied (clipped) command

    def _rwd(self, state, a):
        th, al, thd, ald = state

        al_mod = (al % 2 * np.pi) - np.pi
        if self.rwd_name == "quanser":
            cost = al_mod**2 + 5e-3*ald**2 + 1e-1*th**2
            + 2e-2*thd**2 + 12*3e-3*a[0]**2

            rwd = np.exp(-cost) * self.timing.dt_ctrl
        elif self.rwd_name == "simple":
            rwd = (1 + -np.cos(al, dtype=np.float32)) / 2

        return rwd

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        assert a is not None, "Action should be not None"
        assert isinstance(a, np.ndarray), "The action should be a ndarray"
        assert np.all(not np.isnan(a)), "Action NaN is not a valid action"
        assert a.ndim == 1, \
            "The action = {a} must be 1d but the input is {a.ndim}d"

        self._state, act = self._ctrl_step(a)

        rwd = self._rwd(self._state, a)
        done = not self.state_space.contains(self._state)

        obs = np.float32([np.cos(self._state[0]), np.sin(self._state[0]),
                          np.cos(self._state[1]), np.sin(self._state[1]),
                          self._state[2], self._state[3]])

        info = {"env/motor_angle": self._state[0],
                "env/pendulum_angle": self._state[1],
                "env/motor_angle_velocity": self._state[2],
                "env/pendulum_angle_velocity": self._state[3],
                "env/action": act}

        return obs, rwd, done, info

    def reset(self):
        raise NotImplementedError

    def render(self, mode="rgb_array"):
        if self.viewer is None:
            self.viewer = CartPoleSwingUpViewer(world_width=5)

        if self._state is None:
            return None

        self.viewer.update(self._state)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")


class ActionLimiter:
    def __init__(self, state_space, action_space, th_lim_min):
        self._th_lim_min = th_lim_min
        self._th_lim_max = (state_space.high[0] + self._th_lim_min) / 2.0
        self._th_lim_stiffness = \
            action_space.high[0] / (self._th_lim_max - self._th_lim_min)
        self._clip = lambda a: np.clip(a, action_space.low, action_space.high)
        self._relu = lambda x: x * (x > 0.0)

    def _joint_lim_violation_force(self, x):
        th, _, thd, _ = x
        up = self._relu(th-self._th_lim_max) - self._relu(th-self._th_lim_min)
        dn = -self._relu(-th-self._th_lim_max)+self._relu(-th-self._th_lim_min)
        if (th > self._th_lim_min and thd > 0.0 or
                th < -self._th_lim_min and thd < 0.0):
            force = self._th_lim_stiffness * (up + dn)
        else:
            force = 0.0
        return force

    def __call__(self, x, a):
        force = self._joint_lim_violation_force(x)
        return self._clip(force if force else a)


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
    """Class that encapsulates all the variables and objectecs needed
       to render a CartPoleSwingUpEnv. It handles all the initialization
       and updating of each object on screen and handles calls to the
       underlying gym.envs.classic_control.rendering.Viewer instance.
    """

    screen = Screen(width=600, height=400)

    def __init__(self, world_width):
        # TODO: make sure that's not redundant
        from gym.envs.classic_control import rendering
        import pyglet
        pyglet.options['headless'] = True  # noqa: F821

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
            "wheel_l": rendering.Transform(
                translation=(-cartwidth / 2, -cartheight / 2)
            ),
            "wheel_r": rendering.Transform(
                translation=(cartwidth / 2, -cartheight / 2)
            ),
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
        cart = rendering.FilledPolygon([(lef, bot), (lef, top),
                                        (rig, top), (rig, bot)])
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
        pole = rendering.FilledPolygon([(lef, bot), (lef, top),
                                        (rig, top), (rig, bot)])
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
        """Updates the positions of the objects on screen"""
        screen = self.screen
        scale = screen.width / self.world_width

        th, al, _, _ = state
        al = (al - np.pi) % (2*np.pi)  # change angle origin

        # use motor angle (theta) as cart x position
        cartx = th * scale + screen.width / 2.0  # MIDDLE OF CART
        carty = screen.height / 2

        self.transforms["cart"].set_translation(cartx, carty)
        self.transforms["pole"].set_rotation(al)
        self.transforms["pole_bob"].set_translation(
            -self.pole.length * np.sin(al), self.pole.length * np.cos(al)
        )

    def render(self, *args, **kwargs):
        """Forwards the call to the underlying Viewer instance"""
        return self.viewer.render(*args, **kwargs)

    def close(self):
        """Closes the underlying Viewer instance"""
        self.viewer.close()

