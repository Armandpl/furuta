import time
from pathlib import Path

import numpy as np

import furuta
from furuta.logger import Loader, SimpleLogger
from furuta.plotter import Plotter
from furuta.robot import Robot
from furuta.state import Signal, State

DEVICE = "/dev/ttyACM0"

if __name__ == "__main__":
    # Init robot
    robot = Robot(DEVICE)

    # Constants
    control_freq = 100.0
    dt = 1.0 / control_freq
    f = 1  # Hz
    omega = 2 * np.pi * f
    A = np.pi / 2

    desired_motor_position = 0.0
    desired_motor_velocity = 0.0

    # Create the logger
    file_name = f"{time.strftime('%Y%m%d-%H%M%S')}.mcap"
    log_dir = Path(furuta.__path__[0]).parent / "logs" / "xp" / "control_motor"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / file_name
    logger = SimpleLogger(log_path)

    # Wait for user input to start the control loop
    input("Go?")

    # Reset encoders
    robot.reset_encoders()

    t = 0.0

    tic = time.time()
    while t < 3.0:
        toc = time.time()
        if toc - tic > dt:
            t += dt
            desired_motor_position = A * (1 - np.cos(omega * t))
            desired_motor_velocity = A * omega * np.sin(omega * t)

            (
                motor_position,
                _,
                motor_velocity,
                _,
                timestamp,
                motor_command,
            ) = robot.step_PID(desired_motor_position, desired_motor_velocity)

            state = State(
                motor_position=Signal(measured=motor_position, desired=desired_motor_position),
                motor_velocity=Signal(measured=motor_velocity, desired=desired_motor_velocity),
                action=motor_command,
                timing=timestamp,
            )

            logger.update(int(timestamp * 1e9), state)
            tic = time.time()

    # Stop robot
    robot.reset_encoders()

    # Close logger
    logger.stop()

    # Read log
    loader = Loader()
    times, states_dict = loader.load(log_path)

    # Plot
    plotter = Plotter(times, states_dict)
    plotter.plot()
