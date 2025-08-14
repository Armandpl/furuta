import time
from pathlib import Path

import numpy as np

import furuta
from furuta.controls.controllers import PIDController
from furuta.controls.filters import VelocityFilter
from furuta.logger import Loader, SimpleLogger
from furuta.plotter import Plotter
from furuta.robot import Robot
from furuta.state import Signal, State

DEVICE = "/dev/ttyACM0"

if __name__ == "__main__":
    # Constants
    control_freq = 100.0
    dt = 1.0 / control_freq

    # Init robot
    robot = Robot(DEVICE)

    # Init controllers
    pendulum_controller = PIDController(dt=1.0 / 2500.0, Kp=3.5, Ki=20.0, Kd=0.1, setpoint=np.pi)
    motor_controller = PIDController(Kp=0.2, Ki=0.001)

    # Low pass velocity filter
    motor_velocity_filter = VelocityFilter(2, 20.0, control_frequency=control_freq, init_vel=0.0)
    pendulum_velocity_filter = VelocityFilter(
        2, 49.0, control_frequency=control_freq, init_vel=0.0
    )

    # Create the logger
    file_name = f"{time.strftime('%Y%m%d-%H%M%S')}.mcap"
    log_dir = Path(furuta.__path__[0]).parent / "logs" / "pid"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / file_name
    logger = SimpleLogger(log_path)

    # Reset encoders
    robot.reset()

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
