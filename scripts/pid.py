import argparse
import time

import numpy as np

from furuta.controls.controllers import Controller
from furuta.controls.filters import VelocityFilter
from furuta.controls.utils import read_parameters_file
from furuta.logger import Loader, SimpleLogger
from furuta.plotter import Plotter
from furuta.robot import Robot
from furuta.state import Signal, State

PARAMETERS_PATH = "scripts/configs/control/parameters.json"
DEVICE = "/dev/ttyACM0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        default="../logs/XP/pid/",
        required=False,
        help="Log destination directory",
    )
    args = parser.parse_args()

    # Constants
    control_freq = 100.0
    dt = 1.0 / control_freq

    # Init robot
    robot = Robot(DEVICE)

    # Init controllers
    parameters = read_parameters_file()
    pendulum_controller = Controller.build_controller(parameters["pendulum_controller"])
    motor_controller = Controller.build_controller(parameters["motor_controller"])

    # Low pass velocity filter
    motor_velocity_filter = VelocityFilter(2, 20.0, control_frequency=control_freq, init_vel=0.0)
    pendulum_velocity_filter = VelocityFilter(
        2, 49.0, control_frequency=control_freq, init_vel=0.0
    )

    # Create the logger
    fname = f"{time.strftime('%Y%m%d-%H%M%S')}.mcap"
    log_path = args.dir + fname
    logger = SimpleLogger(log_path)

    # Reset encoders
    robot.reset_encoders()

    # Wait for user input to start the control loop
    input("Go?")

    # Set the initial state
    measured_motor_position, measured_pendulum_position, timestamp = robot.step(0.0)
    state = State(
        motor_position=Signal(measured=measured_motor_position),
        pendulum_position=Signal(measured=measured_pendulum_position),
        motor_velocity=Signal(measured=0.0, filtered=0.0),
        pendulum_velocity=Signal(measured=0.0, filtered=0.0),
        action=0.0,
        timing=dt,
    )
    logger.update(int(0.0 * 1e9), state)

    t = 0.0

    tic = time.time()
    while t < 3.0:
        toc = time.time()
        if toc - tic > dt:
            t += dt
            # Compute action
            action = 0
            action -= pendulum_controller.compute_command(measured_pendulum_position)
            action -= motor_controller.compute_command(measured_motor_position)
            action = np.clip(action, -1, 1)

            # Send action
            measured_motor_position, measured_pendulum_position, _ = robot.step(action)

            # Compute velocity via finite differences
            measured_motor_velocity = (
                measured_motor_position - state.motor_position.measured
            ) / dt
            measured_pendulum_velocity = (
                measured_pendulum_position - state.pendulum_position.measured
            ) / dt

            # Filter velocity
            filtered_motor_velocity = motor_velocity_filter(measured_motor_velocity)
            filtered_pendulum_velocity = pendulum_velocity_filter(measured_pendulum_velocity)

            # Log
            state = State(
                motor_position=Signal(measured=measured_motor_position),
                pendulum_position=Signal(measured=measured_pendulum_position),
                motor_velocity=Signal(
                    measured=measured_motor_velocity, filtered=filtered_motor_velocity
                ),
                pendulum_velocity=Signal(
                    measured=measured_pendulum_velocity, filtered=filtered_pendulum_velocity
                ),
                action=action,
                timing=toc - tic,
            )

            logger.update(int(t * 1e9), state)

            tic = time.time()

    # Close logger
    logger.stop()

    # Load log
    loader = Loader()
    times, states = loader.load(log_path)

    # Plot
    plotter = Plotter(times, states)
    plotter.plot()
