from typing import Optional

import gymnasium as gym
import numpy as np

from furuta.robot import QubeDynamics
from furuta.utils import ALPHA, ALPHA_DOT, THETA, THETA_DOT, VelocityFilter

from .furuta_base import FurutaBase


class FurutaSim(FurutaBase):
    def __init__(
        self,
        dyn: QubeDynamics = QubeDynamics(),
        control_freq=50,
        reward="alpha",
        state_limits=[np.pi, 2 * np.pi, 20, 20],
        encoders_CPRs=None,
        velocity_filter: int = None,
        render_mode="rgb_array",
    ):

        super().__init__(control_freq, reward, state_limits, render_mode)
        self.dyn = dyn

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

        # TODO actually sample from system bounds?
        # start at zero for now
        self._simulation_state = np.zeros(
            4, dtype=np.float32
        )  # 0.01 * np.float32(np.random.randn(self.state_space.shape[0]))
        self._state = np.zeros(self.state_space.shape[0], dtype=np.float32)

        self._update_state(0)

    def _update_state(self, a):
        # ok so we simulate two things: the systems's state
        # and the way we would measure it

        # update the simulation state
        thdd, aldd = self.dyn(self._simulation_state, a)

        # TODO integrate
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

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self._init_state()
        obs, _, _, _, _ = self.step(np.array([0.0]))
        return obs, {}


class Parameterized(gym.Wrapper):
    """Allow passing new dynamics parameters upon environment reset."""

    def params(self):
        return self.unwrapped.dyn.params

    def reset(self, params):
        self.unwrapped.dyn.params = params
        return self.env.reset()