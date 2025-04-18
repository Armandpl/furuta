import time
from pathlib import Path

import crocoddyl
import numpy as np

from furuta.controls.controllers import SwingUpController
from furuta.controls.filters import VelocityFilter
from furuta.controls.utils import read_parameters_file
from furuta.logger import Logger
from furuta.robot import Robot, RobotModel
from furuta.viewer import RobotViewer

PARAMETERS_PATH = "scripts/configs/control/parameters.json"
LOG_DIR = Path("../logs/XP/swing_up")
DEVICE = "/dev/ttyACM0"

# Init robot
robot = Robot(DEVICE)

# Read parameters
parameters = read_parameters_file(PARAMETERS_PATH)["swing_up_controller"]

# Time constants
t_final = parameters["t_final"]  # MPC Time Horizon (s)
control_freq = parameters["control_frequency"]  # Hz
time_step = 1 / control_freq  # s

# Robot
robot_model = RobotModel.robot

# Initial state
init_state = np.zeros((robot_model.model.nq + robot_model.model.nv,))  # Down

# Desired State
x_ref = np.array([0.0, np.pi, 0.0, 0.0])  # Up

# Create the controller
controller = SwingUpController(robot_model, parameters, x_ref)

# Create the data logger
logger = Logger()

# Solve the OCP a first time to get the warm start
controller.compute_command(init_state)

# # Create the robot viewer
# robot_viewer = RobotViewer(robot_model)

# # Display the solution
# robot_viewer.animate(np.arange(0, t_final, time_step), controller.solver.xs.tolist())
crocoddyl.plotOCSolution(controller.solver.xs, controller.solver.us)

motor_speed_filter = VelocityFilter(2, 30.0, control_freq, 0.0)
pendulum_speed_filter = VelocityFilter(2, 49.0, control_freq, 0.0)

# Warm start
x_ws = controller.solver.xs
u_ws = controller.solver.us

# Wait for user input to start the control loop
input("Go?")

# Reset encoders
robot.reset_encoders()

# Get the initial motor and pendulum angles
motor_angle, pendulum_angle, init_timestamp, action = robot.step(0.0, 0.0)

logger.update(
    time=0.0,
    control=0.0,
    state=np.array([motor_angle, pendulum_angle, 0.0, 0.0]),
    unfiltered_speed=np.array([0.0, 0.0]),
    desired_state=init_state,
    elapsed_time=0.0,
)

timestamp = 0.0
tic = time.time()
# Control loop
# f = 1.5
while timestamp < 3.0:
    controller.compute_command(logger.states[-1], 3, x_ws, u_ws)
    desired_state = controller.get_next_desired_state()
    # desired_state = x_ws[i]
    desired_motor_position = desired_state[0]
    desired_motor_velocity = desired_state[2]
    # desired_motor_position = 2 * np.sin(2 * np.pi * timestamp * f)
    # desired_motor_velocity = 2 * np.cos(2 * np.pi * timestamp * f) * 2 * np.pi * f
    # desired_motor_position = np.pi
    # desired_motor_velocity = 0.0
    # desired_state = np.array([desired_motor_position, 0.0, desired_motor_velocity, 0.0])

    toc = time.time()
    time.sleep(max(0.0, time_step - (toc - tic)))
    tic = time.time()

    # Call the step function and get the motor and pendulum angles
    try:
        motor_angle, pendulum_angle, timestamp, action = robot.step(
            desired_motor_position, desired_motor_velocity
        )
        timestamp = timestamp - init_timestamp
    except ValueError as error_message:
        print(error_message)
        break
    dt = timestamp - logger.times[-1]
    unfiltered_motor_speed = (motor_angle - logger.states[-1][0]) / dt
    unfiltered_pendulum_speed = (pendulum_angle - logger.states[-1][1]) / dt
    motor_speed = motor_speed_filter(unfiltered_motor_speed)
    pendulum_speed = pendulum_speed_filter(unfiltered_pendulum_speed)

    x_ws, u_ws = controller.get_warm_start()

    # Log data
    logger.update(
        time=timestamp,
        control=action,
        state=np.array([motor_angle, pendulum_angle, motor_speed, pendulum_speed]),
        unfiltered_speed=np.array([unfiltered_motor_speed, unfiltered_pendulum_speed]),
        desired_state=desired_state,
        elapsed_time=dt,
    )

# Close the serial connection
robot.close()

# Save log and plot
logger.save(LOG_DIR)
logger.plot()
