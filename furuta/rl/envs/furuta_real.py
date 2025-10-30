from typing import Optional

import numpy as np

from furuta.rl.envs.furuta_base import FurutaBase
from furuta.robot import Robot
from furuta.utils import VelocityFilter


class FurutaReal(FurutaBase):
    def __init__(
        self,
        robot: Robot,
        control_freq=100,
        reward="cos_alpha",
        angle_limits=None,
        speed_limits=None,
        motor_stop_pid=[0.04, 0.0, 0.001],
    ):
        super().__init__(control_freq, reward, angle_limits, speed_limits)
        self.motor_stop_pid = motor_stop_pid

        self.robot = robot
        self._state = None

    def _init_vel_filt(self):
        self.vel_filt = VelocityFilter(2, dt=self.timing.dt)

    def _update_state(self, action):
        motor_angle, pendulum_angle, _ = self.robot.step(action)

        # motor_angle: theta, pendulum angle: alpha
        pos = np.array([motor_angle, pendulum_angle], dtype=np.float32)
        vel = self.vel_filt(pos)
        state = np.concatenate([pos, vel])
        self._state = state

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # else the first computed velocity will take into account previous episode
        # and it'll be huge and wrong and will terminate the episode
        self._init_vel_filt()
        self._update_state(0.0)  # initial state
        return self.get_obs(), {}

    # TODO: override parent render function
    # replace by taking webcam snapshot
    # and outputing rgb array?
    # but webcam can't record at 100hz
    # or could just use the same render function!!!
    # def render(self, mode='human'):
    #     raise NotImplementedError

    def close(self):
        self.robot.close()
