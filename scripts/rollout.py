import numpy as np

from furuta.logger import Logger
from furuta.robot import RobotModel
from furuta.sim import SimulatedRobot
from furuta.utils import ROOT_DIR
from furuta.viewer import RobotViewer

LOG_DIR = ROOT_DIR / "logs/rollout/"

# Time constants
t_final = 3.0
control_freq = 50  # Hz
time_step = 1 / control_freq

times = np.arange(0, t_final, time_step)

# Set the initial state
init_state = np.array([0.0, 1e-2, 0.0, 0.0])

# Robot
robot = RobotModel.robot

# Create the simulation
sim = SimulatedRobot(robot, init_state, dt=1e-6)

# Create the data logger
logger = Logger()
logger.update(time=0.0, state=sim.state)

# Rollout
u = 0.0
for t in times[1:]:
    sim.step(u, time_step)
    logger.update(t, sim.state)

# Save log and plot
logger.plot()
logger.save(LOG_DIR)

# Animate
robot_viewer = RobotViewer(robot)
robot_viewer.animate_log(logger)
