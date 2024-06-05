import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from panda3d_viewer import Viewer
from pinocchio.visualize.panda3d_visualizer import Panda3dVisualizer

from furuta.robot import PendulumDynamics

ROOT_DIR = "/home/pierfabre/pendulum_workspace/src/furuta/"
STATE = ["phi", "theta", "phi_dot", "theta_dot"]


@dataclass
class RobotData:
    time = 0.0
    state = [0.0, 0.0, 0.0, 0.0]
    control = 0.0
    elapsed_time = 0.0

    def update(self, time=None, state=None, control=None, elapsed_time=None):
        if time is not None:
            self.time = time
        if state is not None:
            self.state = state
        if control is not None:
            self.control = control
        if elapsed_time is not None:
            self.elapsed_time = elapsed_time


class Logger:
    def __init__(self):
        self.times = []
        self.states = []
        self.controls = []
        self.elapsed_times = []

    def save(self, directory: str):
        np.save(directory + "times.npy", self.times)
        np.save(directory + "states.npy", self.states)
        np.save(directory + "controls.npy", self.controls)
        np.save(directory + "elapsed_times.npy", self.elapsed_times)

    def update(self, data: RobotData):
        self.times.append(data.time)
        self.states.append(data.state)
        self.controls.append(data.control)
        self.elapsed_times.append(data.elapsed_time)

    def load(self, directory: str):
        self.states = np.load(directory + "states.npy")
        self.times = np.load(directory + "times.npy")
        self.control = np.load(directory + "controls.npy")
        self.elapsed_times = np.load(directory + "elapsed_times.npy")

    def plot(self):
        plt.figure(1)
        for i in range(len(STATE)):
            plt.subplot(2, 2, i + 1)
            plt.plot(self.times, np.array(self.states)[:, i])
            plt.title(STATE[i])

        plt.figure(2)
        plt.plot(self.times, self.controls)
        plt.title("Control")

        plt.figure(3)
        plt.plot(self.times, self.elapsed_times)
        plt.title("Elapsed Time")

        plt.show()


class Integrator:
    def __init__(self):
        self.integrator = "euler"

    # TODO : add rungekutta dopri integrator if needed

    def euler(self, phi_ddot, th_ddot, state, dt):
        phi, th, phi_dot, th_dot = state
        phi_dot += phi_ddot * dt
        th_dot += th_ddot * dt
        phi += phi_dot * dt
        th += th_dot * dt
        return [phi, th, phi_dot, th_dot]

    def __call__(self, phi_ddot, th_ddot, state, dt):
        if self.integrator == "euler":
            return self.euler(phi_ddot, th_ddot, state, dt)


class SimulatedRobot:
    def __init__(self, robot):
        self.state = np.array([0.0, np.pi, 0.0, 0.0])
        self.dt = 1e-6
        self.dyn = PendulumDynamics()
        self.integrator = Integrator()
        self.robot = robot
        self.model = self.robot.model
        self.data = self.model.createData()

    def step(self, u: float, dt: float):
        """Simulate the robot for a given control input for a given duration :param u: control
        input :param dt: duration of the control input."""
        t = 0.0
        q = np.array(self.state[:2])
        v = np.array(self.state[2:])
        while t < dt:
            tau = np.array([u, 0.0])
            tau[1] = -self.dyn.f_p * v[1]
            tau[0] -= self.dyn.motor.f_v * v[0] + self.dyn.motor.f_c * np.sign(v[0])
            a = pin.aba(self.model, self.data, q, v, tau)
            v += a * self.dt
            q = pin.integrate(self.model, q, v * self.dt)
            t += self.dt
        self.state = np.concatenate((q, v))
        return self.state


class RobotViewer:
    def __init__(self, robot: pin.RobotWrapper = None):
        self.robot = robot
        viewer = Viewer()
        # Attach the robot to the viewer scene
        self.robot.setVisualizer(Panda3dVisualizer())
        self.robot.initViewer(viewer=viewer)
        self.robot.loadViewerModel(group_name=self.robot.model.name)

    def display(self, q):
        self.robot.display(q)

    def animate(self, times, states):
        # Initial state
        q = np.array(states)[:, :2]
        self.display(q[0])
        time.sleep(1.0)
        tic = time.time()
        for i in range(1, len(times)):
            toc = time.time()
            time.sleep(max(0, times[i] - times[i - 1] - (toc - tic)))
            tic = time.time()
            if i % 10 == 0:
                self.display(q[i])
