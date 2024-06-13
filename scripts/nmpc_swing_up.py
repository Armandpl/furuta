import time

import numpy as np
import pinocchio as pin

from furuta.controls.controllers import SwingUpController
from furuta.controls.utils import read_parameters_file
from furuta.sim import Logger, RobotData, RobotViewer, SimulatedRobot

ROOT_DIR = "/home/pierfabre/pendulum_workspace/src/furuta/"
LOGS_DIR = "/home/pierfabre/pendulum_workspace/logs/closed_loop/"

T_SIM = 3.0

# Read parameters
parameters = read_parameters_file(ROOT_DIR + "scripts/configs/parameters.json")[
    "swing_up_controller"
]

# Time constants
t_final = parameters["t_final"]  # MPC Time Horizon (s)
control_freq = parameters["control_frequency"]  # Hz
time_step = 1 / control_freq  # s

times = np.arange(0, T_SIM, time_step)

# Initial state
init_state = np.array([0.0, np.pi, 0.0, 0.0])

# Create robot from URDF
robot = pin.RobotWrapper.BuildFromURDF(
    ROOT_DIR + "robot/hardware/furuta.urdf",
    package_dirs=[ROOT_DIR + "robot/hardware/CAD/stl/"],
)

# Desired State
x_ref = np.zeros((robot.model.nq + robot.model.nv,))

# Create the controller
controller = SwingUpController(robot, parameters, x_ref)

# Create the data logger
data = RobotData()
logger = Logger()

# Solve the OCP a first time to get the warm start
controller.compute_command(state=init_state, max_iter=500, callback=True)

# Create the robot viewer
robot_viewer = RobotViewer(robot)

# Display the solution
robot_viewer.animate(np.arange(0, t_final, time_step), controller.solver.xs.tolist())

# Create the simulated robot
sim = SimulatedRobot(robot)

# Set the initial state
sim.state = init_state

# Warm start
x_ws = controller.solver.xs
u_ws = controller.solver.us

# Run the simulation
for t in times:
    tic = time.time()
    # Solve the OCP
    u = controller.compute_command(sim.state, 3, x_ws, u_ws)
    toc = time.time()
    # Simulate the robot
    sim.step(u, time_step)
    # Get the warm start from the controller
    x_ws, u_ws = controller.get_warm_start()
    # Log data
    data.update(time=t, state=sim.state.tolist(), control=u, elapsed_time=toc - tic)
    logger.update(data)

# Save and visualize
logger.save(LOGS_DIR)
# Create the robot viewer
robot_viewer = RobotViewer(robot)
robot_viewer.animate(logger.times, logger.states)
logger.plot()
