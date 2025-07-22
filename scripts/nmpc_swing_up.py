import time
from pathlib import Path
from time import strftime

import crocoddyl
import numpy as np

import furuta
from furuta.controls.controllers import SwingUpController
from furuta.controls.filters import VelocityFilter
from furuta.logger import Loader, SimpleLogger
from furuta.plotter import Plotter
from furuta.robot import RobotModel
from furuta.sim import SimulatedRobot
from furuta.state import Signal, State
from furuta.viewer import Viewer3D

if __name__ == "__main__":
    # Time constants
    t_final = 0.5  # MPC Time Horizon (s)
    t_sim = 3.0  # Sim duration (s)
    time_step = 0.01  # s
    control_freq = 1 / time_step  # Hz

    times = np.arange(0, t_sim, time_step)

    # Robot
    robot = RobotModel().robot
    # Initial state
    state = State(0.0, 0.0, 0.0, 0.0)
    init_state = np.zeros(4)
    # Desired State
    x_ref = np.array([0.0, np.pi, 0.0, 0.0])

    # Create the controller
    controller = SwingUpController(robot, x_ref, control_freq, t_final)

    # Create the data logger
    file_name = f"{strftime('%Y%m%d-%H%M%S')}.mcap"
    log_dir = Path(furuta.__path__[0]).parent / "logs" / "nmpc_swing_up"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / file_name
    logger = SimpleLogger(log_path)

    # Solve the OCP a first time to get the warm start
    controller.compute_command(init_state)
    xs = controller.get_trajectoy()
    us = controller.get_command()
    # Create the robot viewer
    viewer = Viewer3D(robot)

    # Display the solution
    viewer.animate(np.arange(0, t_final, time_step), xs)
    crocoddyl.plotOCSolution(xs, us)

    # Create the simulated robot
    sim = SimulatedRobot(robot, init_state, dt=1e-5)

    # Low pass velocity filter
    motor_velocity_filter = VelocityFilter(2, 20.0, control_freq, init_state[2])
    pendulum_velocity_filter = VelocityFilter(2, 20.0, control_freq, init_state[3])

    # Warm start
    x_ws = xs
    u_ws = us

    state = State()
    logger.update(int(0.0 * 1e9), state)

    # Run the simulation
    for t in times:
        # Update state
        x = np.array(
            [
                state.motor_position.measured,
                state.pendulum_position.measured,
                state.motor_velocity.measured,
                state.pendulum_velocity.measured,
            ]
        )

        # Solve the OCP
        tic = time.time()
        u = controller.compute_command(x, 10, x_ws, u_ws)
        toc = time.time()
        compute_time = toc - tic

        # Update residual ref
        controller.control_rate_residual.reference = np.array([u])

        # Simulate the robot
        measured_motor_position, measured_pendulum_position = sim.step(u, time_step)

        # Compute velocity via finite differences
        measured_motor_velocity = (
            measured_motor_position - state.motor_position.measured
        ) / time_step
        measured_pendulum_velocity = (
            measured_pendulum_position - state.pendulum_position.measured
        ) / time_step

        # Filter velocity
        filtered_motor_velocity = motor_velocity_filter(measured_motor_velocity)
        filtered_pendulum_velocity = pendulum_velocity_filter(measured_pendulum_velocity)

        # Log data
        state = State(
            motor_position=Signal(
                measured=measured_motor_position,
                simulated=sim.state[0],
            ),
            pendulum_position=Signal(
                measured=measured_pendulum_position,
                simulated=sim.state[1],
            ),
            motor_velocity=Signal(
                measured=measured_motor_velocity,
                simulated=sim.state[2],
                filtered=filtered_motor_velocity,
            ),
            pendulum_velocity=Signal(
                measured=measured_pendulum_velocity,
                simulated=sim.state[3],
                filtered=filtered_pendulum_velocity,
            ),
            action=u,
            timing=compute_time,
        )
        logger.update(int(t * 1e9), state)

        # Get the warm start from the controller
        x_ws, u_ws = controller.get_warm_start()

    # Close logger
    logger.stop()

    # Read log
    loader = Loader()
    times, states_dict = loader.load(log_path)

    # Plot
    plotter = Plotter(times, states_dict)
    plotter.plot()

    # Animate
    states = loader.get_state("simulated")
    viewer.animate(times, states)
    viewer.close()
