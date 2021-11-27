import configparser
import logging
from time import sleep

import numpy as np

from .common import VelocityFilter
from .furuta_base import FurutaBase
from .hardware.motor import Motor
from .hardware.LS7366R import LS7366R


class FurutaReal(FurutaBase):

    def __init__(self, fs=100, fs_ctrl=100,
                 action_limiter=False, safety_th_lim=1.5,
                 reward="simple", state_limits='low',
                 config_file="furuta.ini"):
        super().__init__(fs, fs_ctrl, action_limiter, safety_th_lim,
                         reward, state_limits)

        self.vel_filt = VelocityFilter(2, dt=self.timing.dt)

        # hardware setup
        config = configparser.ConfigParser()
        config.read(config_file)

        D2 = int(config["Motor"]["D2"])
        IN1 = int(config["Motor"]["IN1"])
        IN2 = int(config["Motor"]["IN2"])

        self.motor = Motor(D2, IN1, IN2)

        BUS_motor = int(config["Motor Encoder"]["BUS"])
        CS_motor = int(config["Motor Encoder"]["CS"])
        BUS_pendulum = int(config["Pendulum Encoder"]["BUS"])
        CS_pendulum = int(config["Pendulum Encoder"]["CS"])

        self.motor_enc = LS7366R(CS_motor, 1000000, 4, BUS_motor)
        self.pendulum_enc = LS7366R(CS_pendulum, 1000000, 4, BUS_pendulum)

        self.motor_CPR = float(config["Motor Encoder"]["CPR"])
        self.pendulum_CPR = float(config["Pendulum Encoder"]["CPR"])

        self._state = self._read_state()

    def _update_state(self, action):
        self.motor.set_speed(action[0])

        state = self._read_state()

        return state

    def _read_state(self):
        pendulum_deg_per_count = 360/self.pendulum_CPR
        p_count = self.pendulum_enc.readCounter()
        # p_count_modulo = p_count % self.pendulum_CPR
        pendulum_angle = pendulum_deg_per_count * p_count
        # pendulum_angle = (pendulum_angle + 180) % 360
        pendulum_angle = pendulum_angle * np.pi / 180

        motor_deg_per_count = 360/self.motor_CPR
        m_count = self.motor_enc.readCounter()
        m_count_modulo = m_count % self.motor_CPR
        motor_angle = m_count_modulo * motor_deg_per_count
        motor_angle = (motor_angle + 180) % 360 - 180
        motor_angle = motor_angle * np.pi / 180

        # motor_angle: theta, pendulum angle: alpha
        pos = np.array([motor_angle, pendulum_angle], dtype=np.float32)
        vel = self.vel_filt(pos)  # TODO understand unit: rad/s ?
        state = np.concatenate([pos, vel])

        return state

    def get_state(self):
        return self._state

    def _reset_pendulum(self, tolerance=10, still_time=1, clear=True):
        """
        This function resets the pendulum.

        :param tolerance: this is the degree windows
                          in which we want the pendulum to be
        :param still_time: number of seconds we want the pendulum to be
                           in the window before considering the pendulum reset
        :param clear: do we want to clear the encoder count once
                      pendulum is reset
        """
        logging.debug("reset pendulum")
        tolerance = tolerance/2
        count = 0
        debug_count = 0
        while True:
            pendulum_angle = self._read_state()[1]*180/np.pi
            pendulum_angle = pendulum_angle % 360

            if 360-tolerance < pendulum_angle or pendulum_angle < tolerance:
                count += 1
                debug_count = 0
            else:
                count = 0
                debug_count += 1

            if count >= int(still_time/self.timing.dt_ctrl):
                if clear:
                    self.pendulum_enc.clearCounter()
                break

            if debug_count > 700:
                enc_count = self.pendulum_enc.readCounter()
                logging.debug(f"{pendulum_angle} pendulum angle, \
                              {enc_count} enc count")
                self.pendulum_enc.clearCounter()

            sleep(self.timing.dt_ctrl)

    def reset(self):
        logging.info("Reset env...")
        # reset pendulum
        self._reset_pendulum(40, 0.5, False)
        # reset motor
        logging.debug("Reset motor")
        while True:
            state = self._read_state()*180/np.pi
            motor_angle = state[0]
            motor_speed = state[2]

            speed = 0.3
            if abs(motor_angle) < 10:

                if abs(motor_speed) > 10:
                    # braking
                    if motor_angle <= 0:
                        self.motor.set_speed(-0.3)
                    elif motor_angle > 0:
                        self.motor.set_speed(0.3)

                    sleep(10/100)

                    self.motor.set_speed(0)

                self.motor.set_speed(0)

                break
            elif motor_angle >= 0:
                self.motor.set_speed(-speed)
            elif motor_angle < 0:
                self.motor.set_speed(speed)

        # wait for pendulum to reset to start position
        self._reset_pendulum(10, 1, True)

        logging.info("Reset done")
        self._state = self._read_state()
        return self.get_obs()

    # TODO: override parent render function
    # replace by taking webcam snapshot
    # and outputing rgb array?
    # but webcam can't record at 100hz
    # def render(self, mode='human'):
    #     raise NotImplementedError

    def close(self):
        self.motor.close()
