from dataclasses import dataclass
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from panda3d_viewer import Viewer
from pinocchio.visualize.panda3d_visualizer import Panda3dVisualizer

from furuta.robot import PendulumDynamics

ROOT_DIR = "/home/pierfabre/pendulum_workspace/src/furuta/"
STATE = ["phi", "theta", "phi_dot", "theta_dot"]


class Log:
    def __init__(self, times=[], state=[]):
        self.times = times
        self.state = state
        self.directory = ROOT_DIR + "logs/sim/"

    def save(self):
        np.save(self.directory + "times.npy", self.times)
        np.save(self.directory + "state.npy", self.state)

    def set_log_directory(self, directory: str):
        self.directory = directory

    def load(self, directory: str):
        self.times = np.load(directory + "time.npy")
        self.state = np.load(directory + "traj.npy")

    def plot(self):
        plt.figure(1)
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.plot(self.times, self.state[:, i])
            plt.title(STATE[i])
        plt.show()


class Integrator:
    def __init__(self):
        self.integrator = "euler"

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
    def __init__(self, simulator: str = "pinocchio"):
        self.simulator = simulator
        self.dyn = PendulumDynamics()
        self.state = [0, np.pi, 0, 0]
        self.dt = 1e-5
        self.integrator = Integrator()
        robot = pin.RobotWrapper.BuildFromURDF(
            ROOT_DIR + "robot/hardware/furuta.urdf",
            package_dirs=[ROOT_DIR + "robot/hardware/CAD/stl/"],
        )
        self.model = robot.model
        self.data = self.model.createData()

    def step(self, u: float, dt: float):
        """Simulate the robot for a given control input for a given duration :param u: control
        input :param dt: duration of the control input."""
        if self.simulator == "dynamic":
            t = 0.0
            while t < dt:
                phi_ddot, th_ddot = self.dyn(self.state, u)
                self.state = self.integrator(phi_ddot, th_ddot, self.state, self.dt)
                t += self.dt
            return self.state
        if self.simulator == "pinocchio":
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
    def __init__(self):
        self.robot = pin.RobotWrapper.BuildFromURDF(
            ROOT_DIR + "robot/hardware/furuta.urdf",
            package_dirs=[ROOT_DIR + "robot/hardware/CAD/stl/"],
        )
        viewer = Viewer()
        # Attach the robot to the viewer scene
        self.robot.setVisualizer(Panda3dVisualizer())
        self.robot.initViewer(viewer=viewer)
        self.robot.loadViewerModel(group_name=self.robot.model.name)

    def display(self, q):
        self.robot.display(q)

    def animate(self, log: Log):
        # Initial state
        q = log.state[:, :2]
        self.display(q[0])
        sleep(1.0)
        for i in range(1, len(log.times)):
            sleep(log.times[i] - log.times[i - 1])
            self.display(q[i])


if __name__ == "__main__":
    robot = SimulatedRobot()
    robot.state = [0.0, 0.01, 0.0, 0.0]
    control_freq = 100  # Hz
    time_step = 1 / control_freq
    times = np.arange(0, 3.0, time_step)
    x = np.zeros((len(times), 4))
    for i, t in enumerate(times):
        robot.step(u=0.0, dt=time_step)
        x[i] = robot.state
        print(f"Time: {t:.2f} s, State: {robot.state}")
    log = Log(times=times, state=x)
    log.save()
    robot_viewer = RobotViewer()
    robot_viewer.animate(log)
    log.plot()
