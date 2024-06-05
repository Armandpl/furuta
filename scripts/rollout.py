import numpy as np
import pinocchio as pin

from furuta.sim import Logger, RobotData, RobotViewer, SimulatedRobot

ROOT_DIR = "/home/pierfabre/pendulum_workspace/src/furuta/"

# Time constants
t_final = 3.0
control_freq = 50  # Hz
time_step = 1 / control_freq

times = np.arange(0, t_final, time_step)

# Create robot from URDF
robot = pin.RobotWrapper.BuildFromURDF(
    ROOT_DIR + "robot/hardware/furuta.urdf",
    package_dirs=[ROOT_DIR + "robot/hardware/CAD/stl/"],
)

# Create the simulation
sim = SimulatedRobot(robot)

# Create the data logger
data = RobotData()
logger = Logger()

# Set the initial state
state = np.array([0.0, 0.01, 0.0, 0.0])
sim.state = state

# Rollout
u = 0.0
for t in times:
    data.update(time=t, state=sim.state.tolist())
    logger.update(data)
    state = sim.step(u, time_step)

# Animate and plot
robot_viewer = RobotViewer(robot)
robot_viewer.animate(logger.times, logger.states)
logger.plot()
