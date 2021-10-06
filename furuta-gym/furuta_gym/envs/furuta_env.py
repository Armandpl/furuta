from time import sleep
import configparser

import numpy as np
import gym
from gym import spaces

from .hardware.motor import Motor
from .hardware.LS7366R import LS7366R


class FurutaEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self, dt=0.01, horizon=2000,
                 motor_speed=60, motor_angle_limit=120,
                 config_file="furuta.ini"):
        self.dt = dt
        self.horizon = horizon
        self.motor_speed = motor_speed
        self.motor_angle_limit = motor_angle_limit

        config = configparser.ConfigParser()
        config.read(config_file)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0,
                                            high=360,
                                            shape=(2,))

        D2 = int(config["Motor"]["D2"])
        IN1 = int(config["Motor"]["IN1"])
        IN2 = int(config["Motor"]["IN2"])

        self.motor = Motor(D2, IN1, IN2)

        CS_motor = int(config["Motor Encoder"]["CS"])
        CS_pendulum = int(config["Pendulum Encoder"]["CS"])

        self.motor_enc = LS7366R(CS_motor, 1000000, 4)
        self.pendulum_enc = LS7366R(CS_pendulum, 1000000, 4)

        self.motor_CPR = float(config["Motor Encoder"]["CPR"])
        self.pendulum_CPR = float(config["Pendulum Encoder"]["CPR"])

    def step(self, action):
        self.steps_taken += 1
        observation = self.get_observation()
        reward = self.get_reward(observation)

        # action = action - 1  # to get -1, 0, 1
        # action = action * self.motor_speed
        self.motor.set_speed(action)
        # sleep(self.dt)

        # check si il est alle trop loin
        motor_angle = observation[1]
        motor_out_of_bounds = abs(motor_angle) > self.motor_angle_limit
        # done = self.steps_taken > self.horizon or motor_out_of_bounds
        done = motor_out_of_bounds

        info = {}

        return observation, reward, done, info

    def get_reward(self, observation):
        # todo: consider using a named tuple for the observation
        pendulum_angle = observation[0]
        return 180 - abs(pendulum_angle)

    def get_observation(self):
        pendulum_deg_per_count = 360/self.pendulum_CPR
        p_count = self.pendulum_enc.readCounter()
        p_count_modulo = p_count % self.pendulum_CPR
        pendulum_angle = pendulum_deg_per_count * p_count_modulo
        pendulum_angle = (pendulum_angle) % 360 - 180
        pendulum_angle = -pendulum_angle

        motor_deg_per_count = 360/self.motor_CPR
        m_count = self.motor_enc.readCounter()
        m_count_modulo = m_count % self.motor_CPR
        motor_angle = m_count_modulo * motor_deg_per_count
        motor_angle = (motor_angle + 180) % 360 - 180

        return np.array([pendulum_angle, motor_angle])

    def reset(self):
        self.steps_taken = 0

        # reset motor
        while True:
            motor_angle = self.get_observation()[1]

            speed = abs(motor_angle)/self.motor_angle_limit * 10 + 15
            if abs(motor_angle) < 10:

                # braking
                if motor_angle > 0:
                    self.motor.set_speed(-50)
                elif motor_angle < 0:
                    self.motor.set_speed(50)

                sleep(10/100)

                self.motor.set_speed(0)
                break
            elif motor_angle > 90:
                self.motor.set_speed(speed)
            elif motor_angle < -90:
                self.motor.set_speed(-speed)

        # wait for pendulum to reset to start position
        count = 0
        while True:
            pendulum_angle = self.get_observation()[0]

            if abs(pendulum_angle) > 175:
                count += 1
            else:
                count = 0

            if count >= 100:
                break

            sleep(self.dt)

        return self.get_observation()

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        self.reset()
        self.motor.close()
