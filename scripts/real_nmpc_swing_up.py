import time
from pathlib import Path

import crocoddyl
import numpy as np

import furuta
from furuta.controls.controllers import SwingUpController
from furuta.logger import Loader, SimpleLogger
from furuta.plotter import Plotter
from furuta.robot import Robot, RobotModel
from furuta.state import Signal, State
from furuta.viewer import Viewer3D

DEVICE = "/dev/ttyACM0"

if __name__ == "__main__":
    # Init robot
    robot = Robot(DEVICE)

    # Constants
    control_freq = 100.0
    dt = 1.0 / control_freq
    t_MPC = 0.5  # MPC Time Horizon (s)
    t_XP = 3.0  # Sim duration (s)

    # Initial state
    init_state = np.zeros(4)
    # Desired State
    x_ref = np.array([0.0, np.pi, 0.0, 0.0])

    # Create the controller
    model = RobotModel().robot
    controller = SwingUpController(model, x_ref, control_freq, t_MPC)

    # Create the data logger
    file_name = f"{time.strftime('%Y%m%d-%H%M%S')}.mcap"
    log_dir = Path(furuta.__path__[0]).parent / "logs" / "xp" / "nmpc_swing_up"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / file_name
    logger = SimpleLogger(log_path)

    # Solve the OCP a first time to get the warm start
    controller.compute_command(init_state)
    xs = controller.get_trajectoy()
    us = controller.get_command()
    # Create the robot viewer
    viewer = Viewer3D(model)

    # Display the solution
    viewer.animate(np.arange(0, t_MPC, dt), xs)
    crocoddyl.plotOCSolution(xs, us)

    # Warm start
    x_ws = xs
    u_ws = us

    # Wait for user input to start the control loop
    input("Go?")

    # Reset encoders
    robot.reset()

    t = 0.0
    u = 0.0

    (
        motor_position,
        pendulum_position,
        motor_velocity,
        pendulum_velocity,
        timestamp,
        motor_command,
    ) = robot.step(0.0)

    t0 = timestamp
    x = np.array(
        [
            motor_position,
            pendulum_position,
            motor_velocity,
            pendulum_velocity,
        ]
    )

    while timestamp - t0 < t_XP:
        # Update residual ref
        controller.control_rate_residual.reference = np.array([u])

        # Solve the OCP
        start = time.time()
        u = controller.compute_command(x, 20, x_ws, u_ws)
        stop = time.time()
        compute_time = stop - start

        (
            desired_motor_position,
            desired_pendulum_position,
            desired_motor_velocity,
            desired_pendulum_velocity,
        ) = controller.get_trajectoy()[1]

        # Basic safety
        if abs(desired_pendulum_position) > 2 * np.pi:
            break
        if abs(desired_motor_position) > np.pi:
            break
        if abs(desired_motor_velocity) > 50.0:
            break

        (
            motor_position,
            pendulum_position,
            motor_velocity,
            pendulum_velocity,
            timestamp,
            motor_command,
        ) = robot.step_PID(desired_motor_position, desired_motor_velocity)

        # Basic safety
        if abs(pendulum_position) > 2 * np.pi:
            break
        if abs(motor_position) > np.pi:
            break
        if abs(motor_velocity) > 50.0:
            break

        state = State(
            motor_position=Signal(measured=motor_position, desired=desired_motor_position),
            motor_velocity=Signal(measured=motor_velocity, desired=desired_motor_velocity),
            pendulum_position=Signal(
                measured=pendulum_position, desired=desired_pendulum_position
            ),
            pendulum_velocity=Signal(
                measured=pendulum_velocity, desired=desired_pendulum_velocity
            ),
            action=motor_command,
            timing=compute_time,
        )
        logger.update(int((timestamp - t0) * 1e9), state)

        # Get the warm start from the controller
        x_ws, u_ws = controller.get_warm_start()

        # Update state
        x = np.array(
            [
                motor_position,
                pendulum_position,
                motor_velocity,
                pendulum_velocity,
            ]
        )

    # Close logger
    logger.stop()

    # Read log
    loader = Loader()
    times, states_dict = loader.load(log_path)

    # Plot
    plotter = Plotter(times, states_dict)
    plotter.plot()

    # Animate
    states = loader.get_state("measured")
    viewer.animate(times, states)
    viewer.close()
