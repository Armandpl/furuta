import numpy as np
import pinocchio as pin

from furuta.robot import Encoders, PendulumDynamics


class SimulatedRobot:
    def __init__(self, robot: pin.RobotWrapper, init_state: np.ndarray, dt: float):
        self.robot = robot
        self.state = init_state
        self.dt = dt
        self.dyn = PendulumDynamics(robot)
        self.encoders = Encoders()

    def step(self, u: float, dt: float) -> np.ndarray:
        """Simulate the robot for a given control input for a given duration :param u: control
        input :param dt: duration of the control input."""
        t = 0.0
        q = np.array(self.state[:2])
        v = np.array(self.state[2:])
        while t < dt:
            # Compute dynamics
            a = self.dyn(q, v, u)
            # Integrate
            v += a * self.dt
            q = pin.integrate(self.robot.model, q, v * self.dt)
            t += self.dt
        self.state = np.concatenate((q, v))
        return self.encoders.measure(self.state[:2])
