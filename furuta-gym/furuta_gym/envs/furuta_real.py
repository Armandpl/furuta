import configparser
import logging
from time import sleep

import numpy as np

from .common import VelocityFilter
from .furuta_base import FurutaBase
from .hardware.motor import Motor
from .hardware.LS7366R import LS7366R


class FurutaReal(FurutaBase):

    def __init__(self, fs=200, fs_ctrl=100, max_steps=300, reward="quanser",
                 action_limiter=True, safety_th_lim=1.5,
                 config_file="furuta.ini"):
        super().__init__(fs, fs_ctrl, max_steps, reward,
                         action_limiter, safety_th_lim)

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

    def _update_state(self, action):
        self.motor.set_speed(action[0])

        pendulum_deg_per_count = 360/self.pendulum_CPR
        p_count = self.pendulum_enc.readCounter()
        p_count_modulo = p_count % self.pendulum_CPR
        pendulum_angle = pendulum_deg_per_count * p_count_modulo
        # pendulum_angle = (pendulum_angle + 180) % 360
        pendulum_angle = pendulum_angle * np.pi / 180

        motor_deg_per_count = 360/self.motor_CPR
        m_count = self.motor_enc.readCounter()
        m_count_modulo = m_count % self.motor_CPR
        motor_angle = m_count_modulo * motor_deg_per_count
        motor_angle = motor_angle % 360
        motor_angle = motor_angle * np.pi / 180

        # motor_angle: theta, pendulum angle: alpha
        pos = np.array([motor_angle, pendulum_angle])
        vel = self.vel_filt(pos)  # TODO understand unit: rad/s ?
        state = np.concatenate([pos, vel])

        return state

    def reset(self):
        super().reset()

        # reset motor
        logging.debug("Reset motor")
        while True:
            motor_angle = self.update_state()[0]*180/np.pi

            speed = abs(motor_angle) / self.motor_angle_limit * 0.1 + 0.2
            if motor_angle < 10 or motor_angle > 350:

                # braking
                if motor_angle > 350:
                    self.motor.set_speed(-0.4)
                elif motor_angle < 10:
                    self.motor.set_speed(0.4)

                sleep(10/100)

                self.motor.set_speed(0)
                break
            elif motor_angle >= 180:
                self.motor.set_speed(speed)
            elif motor_angle < 180:
                self.motor.set_speed(-speed)

        # wait for pendulum to reset to start position
        print("reset pendulum")
        count = 0
        debug_count = 0
        while True:
            pendulum_angle = self.update_state()[1]*180/np.pi

            if 175 < pendulum_angle < 185:  # TODO use cos/sin instead
                count += 1
                debug_count = 0
            else:
                count = 0
                debug_count += 1

            if count >= int(1/self.timing.dt_ctrl):
                break

            if debug_count > 700:
                enc_count = self.pendulum_enc.readCounter()
                logging.debug(f"{pendulum_angle} pendulum angle, \
                              {enc_count} enc count")
                self.pendulum_enc.clearCounter()

            sleep(self.timing.dt_ctrl)

        logging.debug("reset done")
        return self.update_state([0])

    # TODO: override parent render function
    # replace by taking webcam snapshot
    # and outputing rgb array?
    # but webcam can't record at 100hz
    # def render(self, mode='human'):
    #     raise NotImplementedError

    def close(self):
        self.motor.close()

