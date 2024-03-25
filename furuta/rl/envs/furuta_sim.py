from typing import List, Optional, Union

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
        reward="cos_alpha",
        angle_limits=[None, None],
        speed_limits=[None, None],
        encoders_CPRs: Optional[List[float]] = None,
        velocity_filter: int = None,
        render_mode="rgb_array",
        dt_std: float = 0.0,
        integration_dt: float = 1 / 500,
    ):

        super().__init__(control_freq, reward, angle_limits, speed_limits, render_mode)
        self.dyn = dyn
        self.dt_std = dt_std
        self.integration_dt = integration_dt

        self.encoders_CPRs = encoders_CPRs

        self.velocity_filter = velocity_filter
        self._init_vel_filt()

    def _init_vel_filt(self):
        if self.velocity_filter:
            self.vel_filt = VelocityFilter(self.velocity_filter, dt=self.timing.dt)
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
        self._state = self._simulation_state.copy()

    def _update_state(self, a):
        # ok so we simulate two things: the systems's state
        # and the way we would measure it

        # TODO integrate
        # dt = np.random.normal(self.timing.dt, self.dt_std)

        integration_steps = int(self.timing.dt / self.integration_dt)
        for _ in range(integration_steps):
            # update the simulation state
            thdd, aldd = self.dyn(self._simulation_state, a)

            self._simulation_state[ALPHA_DOT] += self.integration_dt * aldd
            self._simulation_state[THETA_DOT] += self.integration_dt * thdd
            self._simulation_state[ALPHA] += (
                self.integration_dt * self._simulation_state[ALPHA_DOT]
            )
            self._simulation_state[THETA] += (
                self.integration_dt * self._simulation_state[THETA_DOT]
            )

        # simulate measurements
        # 1. Reduce the resolution of THETA and ALPHA based on encoders's CPRS
        # do this by rounding _simulation_state[THETA/ALPHA] to the nearest multiple of 2pi/CPRs
        if self.encoders_CPRs:
            # TODO dedupe code here
            theta_increment = 2 * np.pi / self.encoders_CPRs[THETA]
            self._state[THETA] = (
                np.round(self._simulation_state[THETA] / theta_increment) * theta_increment
            )

            alpha_increment = 2 * np.pi / self.encoders_CPRs[ALPHA]
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
        super().reset(seed=seed, options=options)
        self.dyn.randomize()
        self._init_state()
        obs = self.get_obs()
        self._init_vel_filt()
        return obs, {}


class Parameterized(gym.Wrapper):
    """Allow passing new dynamics parameters upon environment reset."""

    def params(self):
        return self.unwrapped.dyn.params

    def reset(self, params):
        self.unwrapped.dyn.params = params
        return self.env.reset()
