from pathlib import Path

import numpy as np

from furuta.logger import SimpleLogger
from furuta.robot import RobotModel
from furuta.sim import SimulatedRobot
from furuta.viewer import Viewer3D

LOG_DIR = Path("../logs/rollout/")

# Time constants
t_final = 3.0
control_freq = 100  # Hz
time_step = 1 / control_freq

times = np.arange(0, t_final, time_step)

# Set the initial state
init_state = np.array([0.0, 3.14, 0.0, 0.0])

# Robot
robot = RobotModel.robot

# Create the simulation
sim = SimulatedRobot(robot, init_state, dt=1e-5)

# Create the data logger
logger = SimpleLogger()
logger.update(time=0.0, state=init_state)

robot_viewer = Viewer3D(robot)

# Rollout
u = 0.0
for i, t in enumerate(times[1:]):
    motor_angle, pendulum_angle = sim.step(u, time_step)
    motor_speed = (motor_angle - logger.states[-1][0]) / time_step
    pendulum_speed = (pendulum_angle - logger.states[-1][1]) / time_step
    logger.update(t, np.array([motor_angle, pendulum_angle, motor_speed, pendulum_speed]))

# Save log and plot
logger.plot()
logger.show()
logger.save(LOG_DIR)

# Animate
robot_viewer = Viewer3D(robot)
robot_viewer.animate_log(logger)
