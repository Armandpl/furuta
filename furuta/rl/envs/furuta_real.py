import logging
from time import sleep
from typing import Optional

import numpy as np

from furuta.rl.envs.furuta_base import FurutaBase
from furuta.robot import Robot
from furuta.utils import ALPHA, ALPHA_DOT, THETA, THETA_DOT, VelocityFilter

MAX_RESET_TIME = 5  # seconds


class FurutaReal(FurutaBase):
    def __init__(
        self,
        control_freq=100,
        reward="alpha_theta",
        state_limits=None,
        usb_device="/dev/ttyACM0",
    ):
        super().__init__(control_freq, reward, state_limits)

        self.robot = Robot(usb_device)

        self.vel_filt = VelocityFilter(2, dt=self.timing.dt)

        self._update_state(0.0)

    def _update_state(self, action):
        motor_angle, pendulum_angle = self.robot.step(action)

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
        logging.info("Reset env...")

        # wait for pendulum to fall back to start position
        reset_time = 0
        while np.abs(self._state[ALPHA_DOT]) > 0.1 and reset_time < MAX_RESET_TIME:
            sleep(0.01)
            self._update_state(0.0)
            reset_time += 0.01

        if reset_time >= MAX_RESET_TIME:
            logging.error("Reset timeout")

        # reset both encoder, motor back to pos=0
        self.robot.reset_encoders()

        logging.info("Reset done")
        self._update_state(0.0)
        return self.get_obs(), {}

    # TODO: override parent render function
    # replace by taking webcam snapshot
    # and outputing rgb array?
    # but webcam can't record at 100hz
    # def render(self, mode='human'):
    #     raise NotImplementedError

    def close(self):
        self.robot.close()
