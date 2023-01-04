import gym
import numpy as np
from furuta_gym.utils import ALPHA, ALPHA_DOT, THETA, THETA_DOT

from .furuta_base import FurutaBase


class FurutaSim(FurutaBase):
    def __init__(
        self, fs=50, action_limiter=True, safety_th_lim=1.5, state_limits="low", sim_params=None
    ):

        super().__init__(fs, action_limiter, safety_th_lim, state_limits)
        self.dyn = QubeDynamics()
        if sim_params:
            self.dyn.params = sim_params

    def _init_state(self):
        # TODO could also sample from state space
        # though we also use it as upper speed limit
        # the two use case are kind of conflicting
        # e.g we don't want to sample 400 rad/s for the init state
        # but we do want the max speed to be 400 rad/s for any state
        # and we dont want this:
        # self._state = np.zeros(4)
        # bc then gSDE doesn't work? if state = 0 no action is taken?
        # or maybe it's too slow to move even the simulated pendulum?
        # and maybe it should have a min voltage as well?
        # self._state = np.random.rand(4)  # self.state_space.sample()
        self._state = 0.01 * np.float32(np.random.randn(self.state_space.shape[0]))
        # print(self._state)
        self._state = self._update_state(0)
        # print(self._state)

    def _update_state(self, a):
        thdd, aldd = self.dyn(self._state, a)
        self._state[ALPHA_DOT] += self.timing.dt * aldd
        self._state[THETA_DOT] += self.timing.dt * thdd
        self._state[ALPHA] += self.timing.dt * self._state[ALPHA_DOT]
        self._state[THETA] += self.timing.dt * self._state[THETA_DOT]
        return np.copy(self._state)

    def reset(self):
        self._init_state()
        return self.step(np.array([0.0]))[0]


class Parameterized(gym.Wrapper):
    """Allow passing new dynamics parameters upon environment reset."""

    def params(self):
        return self.unwrapped.dyn.params

    def reset(self, params):
        self.unwrapped.dyn.params = params
        return self.env.reset()


class QubeDynamics:
    """Solve equation M qdd + C(q, qd) = tau for qdd."""

    def __init__(self):
        # Gravity
        self.g = 9.81

        # Motor
        self.Rm = 8.4  # resistance (rated voltage/stall current)
        self.V = 12.0  # nominal voltage
        self.min_V = 0.2 * self.V  # minimum voltage to move the pendulum

        # back-emf constant (V-s/rad)
        self.km = 0.042  # (rated voltage / no load speed)

        # Rotary arm
        self.Mr = 0.095  # mass (kg)
        self.Lr = 0.085  # length (m)
        self.Dr = 5e-6  # viscous damping (N-m-s/rad), original: 0.0015

        # Pendulum link
        self.Mp = 0.024  # mass (kg)
        self.Lp = 0.129  # length (m)
        self.Dp = 1e-6  # viscous damping (N-m-s/rad), original: 0.0005

        # Init constants
        self._init_const()

    def _init_const(self):
        # Moments of inertia
        Jr = self.Mr * self.Lr**2 / 12  # inertia about COM (kg-m^2)
        Jp = self.Mp * self.Lp**2 / 12  # inertia about COM (kg-m^2)

        # Constants for equations of motion
        self._c = np.zeros(5)
        self._c[0] = Jr + self.Mp * self.Lr**2
        self._c[1] = 0.25 * self.Mp * self.Lp**2
        self._c[2] = 0.5 * self.Mp * self.Lp * self.Lr
        self._c[3] = Jp + self._c[1]
        self._c[4] = 0.5 * self.Mp * self.Lp * self.g

    @property
    def params(self):
        params = self.__dict__.copy()
        params.pop("_c")
        return params

    @params.setter
    def params(self, params):
        self.__dict__.update(params)
        self._init_const()

    def __call__(self, s, a):
        th, al, thd, ald = s
        voltage = a * self.V

        # Define mass matrix M = [[a, b], [b, c]]
        a = self._c[0] + self._c[1] * np.sin(al) ** 2
        b = self._c[2] * np.cos(al)
        c = self._c[3]
        d = a * c - b * b

        # Calculate vector [x, y] = tau - C(q, qd)
        trq = self.km * (voltage - self.km * thd) / self.Rm
        c0 = self._c[1] * np.sin(2 * al) * thd * ald - self._c[2] * np.sin(al) * ald * ald
        c1 = -0.5 * self._c[1] * np.sin(2 * al) * thd * thd + self._c[4] * np.sin(al)
        x = trq - self.Dr * thd - c0
        y = -self.Dp * ald - c1

        # Compute M^{-1} @ [x, y]
        thdd = (c * x - b * y) / d
        aldd = (a * y - b * x) / d

        return thdd, aldd
