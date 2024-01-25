import logging
from time import sleep
from typing import Optional

import numpy as np
from simple_pid import PID

from furuta.rl.envs.furuta_base import FurutaBase
from furuta.robot import Robot
from furuta.utils import ALPHA, ALPHA_DOT, THETA, THETA_DOT, VelocityFilter

MAX_RESET_TIME = 7  # seconds
MAX_MOTOR_RESET_TIME = 0.2  # seconds
RESET_TIME = 0.5
ALPHA_THRESH = np.cos(
    np.deg2rad(2)
)  # alpha should stay between -2 and 2 deg for 0.5 sec for us to consider the env reset


class FurutaReal(FurutaBase):
    def __init__(
        self,
        control_freq=100,
        reward="alpha_theta",
        angle_limits=None,
        speed_limits=None,
        usb_device="/dev/ttyACM0",
        motor_stop_pid=[0.04, 0.0, 0.001],
    ):
        super().__init__(control_freq, reward, angle_limits, speed_limits)
        self.motor_stop_pid = motor_stop_pid

        self.robot = Robot(usb_device)
        self._state = None

    def _init_vel_filt(self):
        self.vel_filt = VelocityFilter(2, dt=self.timing.dt)

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
        super().reset(seed=seed)
        logging.info("Reset env...")

        if self._state is not None:  # if not first reset
            logging.debug("Stopping motor")
            motor_pid = PID(
                self.motor_stop_pid[0],
                self.motor_stop_pid[1],
                self.motor_stop_pid[2],
                setpoint=0.0,
                output_limits=(-1, 1),
            )

            reset_time = 0
            while abs(self._state[THETA_DOT]) > 0.5 and reset_time < MAX_MOTOR_RESET_TIME:
                act = motor_pid(self._state[THETA_DOT])
                self._update_state(act)
                reset_time += self.timing.dt
                sleep(self.timing.dt)

            logging.debug("Waiting for pendulum to fall back down")
            time_under_thresh = 0
            reset_time = 0
            while time_under_thresh < RESET_TIME and reset_time < MAX_RESET_TIME:
                if np.cos(self._state[ALPHA]) > ALPHA_THRESH:
                    time_under_thresh += self.timing.dt
                else:
                    time_under_thresh = 0
                self._update_state(0.0)
                reset_time += self.timing.dt
                sleep(self.timing.dt)

            if reset_time >= MAX_RESET_TIME:
                logging.info(f"Reset timeout, alpha: {np.rad2deg(self._state[ALPHA])}")

        # reset both encoder, motor back to pos=0
        self.robot.reset_encoders()

        logging.info("Reset done")
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
