import time
from pathlib import Path

import numpy as np

import furuta
from furuta.controls.controllers import PIDController
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

    # Initial state
    motor_position = 0.0
    pendulum_position = 0.0

    # Init controllers
    pendulum_controller = PIDController(dt=dt, Kp=3.5, Ki=20.0, Kd=0.1, setpoint=pendulum_position)
    motor_controller = PIDController(dt=dt, Kp=0.2, Ki=0.001, Kd=0.0, setpoint=motor_position)

    # Create the logger
    file_name = f"{time.strftime('%Y%m%d-%H%M%S')}.mcap"
    log_dir = Path(furuta.__path__[0]).parent / "logs" / "xp" / "pid"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / file_name
    logger = SimpleLogger(log_path)

    # Wait for user input to start the control loop
    input("Lift the pendulum upward and press Enter")

    # Reset encoders. The upward position is now 0.0
    robot.reset()

    t = 0.0

    tic = time.time()
    while t < 3.0:
        toc = time.time()
        if toc - tic > dt:
            t += dt
            # Compute motorn command
            motor_command = 0
            motor_command -= pendulum_controller.compute_command(pendulum_position)
            motor_command -= motor_controller.compute_command(motor_position)
            motor_command = np.clip(motor_command, -1, 1)

            (
                motor_position,
                pendulum_position,
                motor_velocity,
                pendulum_velocity,
                timestamp,
                action,
            ) = robot.step(motor_command)

            # Basic safety
            if abs(pendulum_position) > np.pi / 2:
                break
            if abs(motor_position) > 2 * np.pi:
                break

            # Log
            state = State(
                motor_position=Signal(measured=motor_position),
                pendulum_position=Signal(measured=pendulum_position),
                motor_velocity=Signal(measured=motor_velocity),
                pendulum_velocity=Signal(measured=pendulum_velocity, filtered=pendulum_velocity),
                action=action,
                timing=toc - tic,
            )

            logger.update(int(timestamp * 1e9), state)
            tic = time.time()

    # Stop robot
    robot.close()

    # Close logger
    logger.stop()

    # Read log
    loader = Loader()
    times, states_dict = loader.load(log_path)

    # Plot
    plotter = Plotter(times, states_dict)
    plotter.plot()
