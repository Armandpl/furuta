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


@dataclass
class Log:
    times = []
    states = []

    def save(self, directory):
        np.save(directory + "times.npy", self.times)
        np.save(directory + "state.npy", self.states)

    def plot(self):
        plt.figure(1)
        for i in range(len(STATE)):
            plt.subplot(2, 2, i + 1)
            plt.plot(self.times, np.array(self.states)[:, i])
            plt.title(STATE[i])
        plt.show()

    def load(self, directory: str):
        self.states = np.load(directory + "traj.npy")
        self.times = np.load(directory + "time.npy")

    def update(self, data: RobotData):
        self.times.append(data.time)
        self.states.append(data.state)


class Logger:
    def __init__(self):
        self.log = Log()

    def save_log(self, directory: str = ROOT_DIR + "../../logs/sim/"):
        self.log.save(directory)

    def update_log(self, data: RobotData):
        self.log.update(data)


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
    def __init__(self):
        self.state = np.array([0.0, np.pi, 0.0, 0.0])
        self.dt = 1e-6
        self.dyn = PendulumDynamics()
        self.integrator = Integrator()
        self.robot = pin.RobotWrapper.BuildFromURDF(
            ROOT_DIR + "robot/hardware/furuta.urdf",
            package_dirs=[ROOT_DIR + "robot/hardware/CAD/stl/"],
        )
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

    def animate(self, log: Log):
        # Initial state
        q = np.array(log.states)[:, :2]
        self.display(q[0])
        time.sleep(1.0)
        tic = time.time()
        for i in range(1, len(log.times)):
            toc = time.time()
            time.sleep(max(0, log.times[i] - log.times[i - 1] - (toc - tic)))
            tic = time.time()
            if i % 10 == 0:
                self.display(q[i])


if __name__ == "__main__":
    sim = SimulatedRobot()
    data = RobotData()
    state = np.array([0.0, 0.01, 0.0, 0.0])
    sim.state = state
    t_final = 5.0
    control_freq = 50  # Hz
    time_step = 1 / control_freq
    times = np.arange(0, 3.0, time_step)
    logger = Logger()
    for t in times:
        data.time = t
        data.state = state.tolist()
        logger.update_log(data)
        u = 0.0
        state = sim.step(u, time_step)
    log = logger.log
    robot_viewer = RobotViewer(sim.robot)
    robot_viewer.animate(log)
    log.plot()
