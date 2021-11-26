import gym
import numpy as np

from .furuta_base import FurutaBase


class FurutaSim(FurutaBase):

    def __init__(self, fs=200, fs_ctrl=100, action_limiter=True,
                 safety_th_lim=1.5, reward="simple", state_limits='low'):

        super().__init__(fs, fs_ctrl, action_limiter, safety_th_lim,
                         reward, state_limits)
        self.dyn = QubeDynamics()

    def _calibrate(self):
        self._state = \
            0.01 * np.float32(self._np_random.randn(self.state_space.shape[0]))
        self._state = self._update_state([0])

    def _update_state(self, a):
        # for sim a is motor input voltage ([-12, 12])
        # but robot input and model output belong to [-1, 1]
        a = a * 12

        thdd, aldd = self.dyn(self._state, a)
        self._state[3] += self.timing.dt * aldd
        self._state[2] += self.timing.dt * thdd
        self._state[1] += self.timing.dt * self._state[3]
        self._state[0] += self.timing.dt * self._state[2]
        return np.copy(self._state)

    def reset(self):
        self._calibrate()
        return self.step(np.array([0.0]))[0]


class Parameterized(gym.Wrapper):
    """
    Allow passing new dynamics parameters upon environment reset.
    """
    def params(self):
        return self.unwrapped.dyn.params

    def step(self, action):
        return self.env.step(action)

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

        # back-emf constant (V-s/rad)
        self.km = 0.042  # (rated voltage / no load speed)

        # Rotary arm
        self.Mr = 0.095  # mass (kg)
        self.Lr = 0.085  # length (m)
        self.Dr = 5e-6   # viscous damping (N-m-s/rad), original: 0.0015

        # Pendulum link
        self.Mp = 0.024  # mass (kg)
        self.Lp = 0.129  # length (m)
        self.Dp = 1e-6   # viscous damping (N-m-s/rad), original: 0.0005

        # Init constants
        self._init_const()

    def _init_const(self):
        # Moments of inertia
        Jr = self.Mr * self.Lr ** 2 / 12  # inertia about COM (kg-m^2)
        Jp = self.Mp * self.Lp ** 2 / 12  # inertia about COM (kg-m^2)

        # Constants for equations of motion
        self._c = np.zeros(5)
        self._c[0] = Jr + self.Mp * self.Lr ** 2
        self._c[1] = 0.25 * self.Mp * self.Lp ** 2
        self._c[2] = 0.5 * self.Mp * self.Lp * self.Lr
        self._c[3] = Jp + self._c[1]
        self._c[4] = 0.5 * self.Mp * self.Lp * self.g

    @property
    def params(self):
        params = self.__dict__.copy()
        params.pop('_c')
        return params

    @params.setter
    def params(self, params):
        self.__dict__.update(params)
        self._init_const()

    def __call__(self, s, u):
        th, al, thd, ald = s
        voltage = u[0]

        # Define mass matrix M = [[a, b], [b, c]]
        a = self._c[0] + self._c[1] * np.sin(al) ** 2
        b = self._c[2] * np.cos(al)
        c = self._c[3]
        d = a * c - b * b

        # Calculate vector [x, y] = tau - C(q, qd)
        trq = self.km * (voltage - self.km * thd) / self.Rm
        c0 = self._c[1] * np.sin(2 * al) * thd * ald \
            - self._c[2] * np.sin(al) * ald * ald
        c1 = -0.5 * self._c[1] * np.sin(2 * al) * thd * thd \
            + self._c[4] * np.sin(al)
        x = trq - self.Dr * thd - c0
        y = -self.Dp * ald - c1

        # Compute M^{-1} @ [x, y]
        thdd = (c * x - b * y) / d
        aldd = (a * y - b * x) / d

        return thdd, aldd
