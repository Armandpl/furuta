import configparser
import math
from time import sleep

import numpy as np
import gym
from gym import spaces

from .common import VelocityFilter
from .hardware.motor import Motor
from .hardware.LS7366R import LS7366R


class FurutaEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self, dt=0.01, horizon=3000,
                 motor_speed=0.6, motor_angle_limit=120,
                 config_file="furuta.ini"):
        self.dt = dt
        self.horizon = horizon # not in use rn
        self.motor_speed = motor_speed # only for categorical
        self.motor_angle_limit = motor_angle_limit

        self.vel_filt = VelocityFilter(2, dt=self.dt)

        config = configparser.ConfigParser()
        config.read(config_file)

        # limits
        act_max = np.array([1])
        obs_max = np.array([1.0, 1.0, 1.0, 1.0, np.pi, np.pi])

        self.action_space = spaces.Box(low=-act_max,
                                       high=act_max,
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-obs_max,
                                            high=obs_max,
                                            dtype=np.float32)

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

    def step(self, action):
        self.steps_taken += 1
        self.update_state()
        observation = self.get_observation()

        # clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = float(action[0])

        # todo: interestingly we compute the reward based on state not obs?
        reward = self.get_reward(self.state, action)

        self.motor.set_speed(action)

        # check si il est alle trop loin
        motor_angle = self.state[0]*180/np.pi
        motor_out_of_bounds = 240 > motor_angle > 120
        # done = self.steps_taken > self.horizon or motor_out_of_bounds
        done = motor_out_of_bounds or steps_taken >= self.horizon

        info = {}

        return observation, reward, done, info

    def get_reward(self, state, action):
        # todo: consider using a named tuple for the observation
        # pendulum_angle = observation[0]
        # return 180 - abs(pendulum_angle)
        
        # -- quanser reward
        # th, al, thd, ald = state
        # al_mod = al % (2 * np.pi) - np.pi
        # cost = al_mod**2 + 5e-3*ald**2 + 1e-1*th**2 + 2e-2*thd**2 + 3e-3*action**2

        # rwd = np.exp(-cost) * self.dt
        # return np.float32(rwd)

        ## -- CartPoleSwingUp-v1 reward
        th, al, thd, ald = state
        reward = (1 + np.cos(al, dtype=np.float32)) / 2
        return reward
        
    def get_observation(self):
        obs = np.float32([np.cos(self.state[0]), np.sin(self.state[0]),
                          np.cos(self.state[1]), np.sin(self.state[1]),
                          self.state[2], self.state[3]])

        return obs


    def update_state(self):
        pendulum_deg_per_count = 360/self.pendulum_CPR
        p_count = self.pendulum_enc.readCounter()
        p_count_modulo = p_count % self.pendulum_CPR
        pendulum_angle = pendulum_deg_per_count * p_count_modulo
        pendulum_angle = (pendulum_angle + 180) % 360
        pendulum_angle = pendulum_angle * np.pi / 180

        motor_deg_per_count = 360/self.motor_CPR
        m_count = self.motor_enc.readCounter()
        m_count_modulo = m_count % self.motor_CPR
        motor_angle = m_count_modulo * motor_deg_per_count
        motor_angle = motor_angle % 360
        motor_angle = motor_angle * np.pi / 180

        # motor_angle: theta, pendulum angle: alpha
        pos =  np.array([motor_angle, pendulum_angle])
        vel = self.vel_filt(pos) # todo understand unit: rad/s ?
        self.state = np.concatenate([pos, vel])

        return self.state

    def reset(self):
        self.steps_taken = 0

        # reset motor
        print("reset motor")
        while True:
            motor_angle = self.update_state()[0]*180/np.pi

            speed = abs(motor_angle)/self.motor_angle_limit * 0.1 + 0.2
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

            if 175 < pendulum_angle < 185:
                count += 1
                debug_count = 0
            else:
                count = 0
                debug_count += 1

            if count >= 100:
                break

            if debug_count > 700:
                enc_count = self.pendulum_enc.readCounter()
                print(f"{pendulum_angle} pendulum angle, {enc_count} enc count")
                self.pendulum_enc.clearCounter()

            sleep(self.dt)

        print("reset done")
        return self.get_observation()

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        self.motor.close()

