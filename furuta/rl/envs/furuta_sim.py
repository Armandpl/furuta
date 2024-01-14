from math import cos, sin

import gym
import numpy as np
from numpy.linalg import inv

from furuta.utils import ALPHA, ALPHA_DOT, THETA, THETA_DOT, VelocityFilter

from .furuta_base import FurutaBase


class FurutaSim(FurutaBase):
    def __init__(
        self,
        fs=50,
        reward="alpha",
        state_limits=None,
        sim_params=None,
        encoders_CPRs=None,
        velocity_filter: int = None,
    ):

        super().__init__(fs, reward, state_limits)
        self.dyn = QubeDynamics()
        if sim_params:
            self.dyn.params = sim_params

        self.encoders_CPRs = encoders_CPRs

        if velocity_filter:
            self.vel_filt = VelocityFilter(velocity_filter, dt=self.timing.dt)
        else:
            self.vel_filt = None

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
        self._simulation_state = 0.01 * np.float32(np.random.randn(self.state_space.shape[0]))
        self._state = np.zeros(self.state_space.shape[0])

        self._update_state(0)

    @profile
    def _update_state(self, a):
        # ok so we simulate two things: the systems's state
        # and the way we would measure it

        # update the simulation state
        thdd, aldd = self.dyn(self._simulation_state, a)

        # integrate
        self._simulation_state[ALPHA_DOT] += self.timing.dt * aldd
        self._simulation_state[THETA_DOT] += self.timing.dt * thdd
        self._simulation_state[ALPHA] += self.timing.dt * self._simulation_state[ALPHA_DOT]
        self._simulation_state[THETA] += self.timing.dt * self._simulation_state[THETA_DOT]

        # simulate measurements
        # 1. Reduce the resolution of THETA and ALPHA based on encoders's CPRS
        # do this by rounding _simulation_state[THETA/ALPHA] to the nearest multiple of 2pi/CPRs
        if self.encoders_CPRs:
            # TODO dedupe code here
            theta_increment = 2 * np.pi / self.encoders_CPRs["motor_encoder_CPRs"]
            self._state[THETA] = (
                np.round(self._simulation_state[THETA] / theta_increment) * theta_increment
            )

            alpha_increment = 2 * np.pi / self.encoders_CPRs["pendulum_encoder_CPRs"]
            self._state[ALPHA] = (
                np.round(self._simulation_state[ALPHA] / alpha_increment) * alpha_increment
            )
        else:
            self._state[THETA] = self._simulation_state[THETA]
            self._state[ALPHA] = self._simulation_state[ALPHA]

        # 2. Compute the velocities using the velocity filter
        if self.vel_filt:
            self._state[2:4] = self.vel_filt(self._state[0:2])
        else:
            self._state[THETA_DOT] = self._simulation_state[THETA_DOT]
            self._state[ALPHA_DOT] = self._simulation_state[ALPHA_DOT]

    def reset(self):
        self._init_state()
        obs, _, _, _ = self.step(np.array([0.0]))
        return obs


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

    @profile
    def __call__(self, state, action):
        # """
        # action between 0 and 1, maps to +V and -V
        # """
        th, al, thd, ald = state
        voltage = action * self.V

        # Precompute some values
        sin_al = sin(al)
        sin_2al = sin(2 * al)
        cos_al = cos(al)

        # Define mass matrix M = [[a, b], [b, c]]
        a = self._c[0] + self._c[1] * sin_al**2
        b = self._c[2] * cos_al
        c = self._c[3]
        d = a * c - b * b

        # Calculate vector [x, y] = tau - C(q, qd)
        trq = self.km * (voltage - self.km * thd) / self.Rm
        c0 = self._c[1] * sin_2al * thd * ald - self._c[2] * sin_al * ald * ald
        c1 = -0.5 * self._c[1] * sin_2al * thd * thd + self._c[4] * sin_al
        x = trq - self.Dr * thd - c0
        y = -self.Dp * ald - c1

        # Compute M^{-1} @ [x, y]
        thdd = (c * x - b * y) / d
        aldd = (a * y - b * x) / d

        # chat gpt optimized code lol
        # # Define mass matrix M = [[a, b], [b, c]]
        # a = self._c[0] + self._c[1] * np.sin(al) ** 2
        # b = self._c[2] * np.cos(al)
        # c = self._c[3]
        # M = np.array([[a, b], [b, c]])
        # Minv = inv(M)

        # # Calculate vector [x, y] = tau - C(q, qd)
        # trq = self.km * (voltage - self.km * thd) / self.Rm
        # c0 = self._c[1] * np.sin(2 * al) * thd * ald - self._c[2] * np.sin(al) * ald * ald
        # c1 = -0.5 * self._c[1] * np.sin(2 * al) * thd * thd + self._c[4] * np.sin(al)
        # x = trq - self.Dr * thd - c0
        # y = -self.Dp * ald - c1
        # v = np.array([x, y])

        # # Compute M^{-1} @ v
        # acc = np.dot(Minv, v)
        # thdd, aldd = acc

        return thdd, aldd
