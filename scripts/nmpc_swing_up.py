import time

import crocoddyl
import numpy as np

from furuta.controls.controllers import SwingUpController
from furuta.controls.utils import read_parameters_file
from furuta.logger import Logger
from furuta.robot import RobotModel
from furuta.sim import SimulatedRobot
from furuta.utils import ROOT_DIR
from furuta.viewer import RobotViewer

LOG_DIR = ROOT_DIR / "logs/closed_loop/"

T_SIM = 3.0

# Read parameters
parameters = read_parameters_file(ROOT_DIR / "src/scripts/configs/parameters.json")[
    "swing_up_controller"
]

# Time constants
t_final = parameters["t_final"]  # MPC Time Horizon (s)
control_freq = parameters["control_frequency"]  # Hz
time_step = 1 / control_freq  # s

times = np.arange(0, T_SIM, time_step)

# Initial state
init_state = np.array([0.0, np.pi, 0.0, 0.0])

# Robot
robot = RobotModel.robot

# Desired State
x_ref = np.zeros((robot.model.nq + robot.model.nv,))

# Create the controller
controller = SwingUpController(robot, parameters, x_ref)

# Create the data logger
logger = Logger()

# Solve the OCP a first time to get the warm start
controller.compute_command(init_state)

# Create the robot viewer
robot_viewer = RobotViewer(robot)

# Display the solution
robot_viewer.animate(np.arange(0, t_final, time_step), controller.solver.xs.tolist())
crocoddyl.plotOCSolution(controller.solver.xs, controller.solver.us)

# Create the simulated robot
sim = SimulatedRobot(robot, init_state, dt=1e-6)

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
    logger.update(time=t, state=sim.state, control=u, elapsed_time=toc - tic)

# Save log and plot
logger.plot()
logger.save(LOG_DIR)

# Animate
robot_viewer.animate_log(logger)
