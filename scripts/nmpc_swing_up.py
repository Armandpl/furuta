import argparse
import time
from time import strftime

import crocoddyl
import numpy as np

from furuta.controls.controllers import SwingUpController
from furuta.controls.filters import VelocityFilter
from furuta.controls.utils import read_parameters_file
from furuta.logger import SimpleLogger
from furuta.robot import RobotModel
from furuta.sim import SimulatedRobot
from furuta.utils import State
from furuta.viewer import Viewer3D

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        default="../logs/closed_loop/",
        required=False,
        help="Log destination directory",
    )
    parser.add_argument(
        "-t", "--duration", default=3.0, required=False, help="Duration of the simulation"
    )
    args = parser.parse_args()

    # Read parameters
    parameters = read_parameters_file("scripts/configs/parameters.json")["swing_up_controller"]

    # Time constants
    t_final = parameters["t_final"]  # MPC Time Horizon (s)
    control_freq = parameters["control_frequency"]  # Hz
    time_step = 1 / control_freq  # s

    times = np.arange(0, args.duration, time_step)

    # Robot
    robot = RobotModel.robot
    # Initial state
    state = State(0.0, 0.0, 0.0, 0.0)
    init_state = np.zeros(4)
    # Desired State
    x_ref = np.array([0.0, np.pi, 0.0, 0.0])

    # Create the controller
    controller = SwingUpController(robot, parameters, x_ref)

    # Create the data logger
    fname = f"{strftime('%Y%m%d-%H%M%S')}.mcap"
    logger = SimpleLogger(args.dir + fname)

    # Solve the OCP a first time to get the warm start
    controller.compute_command(init_state)

    # Create the robot viewer
    robot_viewer = Viewer3D(robot)

    # Display the solution
    robot_viewer.animate(np.arange(0, t_final, time_step), np.array(controller.solver.xs.tolist()))
    crocoddyl.plotOCSolution(controller.solver.xs, controller.solver.us)

    # Create the simulated robot
    sim = SimulatedRobot(robot, init_state, dt=1e-5)

    # Low pass velocity filter
    motor_speed_filter = VelocityFilter(5, 20.0, control_freq, init_state[2])
    pendulum_speed_filter = VelocityFilter(2, 49.0, control_freq, init_state[3])

    # Warm start
    x_ws = controller.solver.xs
    u_ws = controller.solver.us

    # Start logging
    logger.start()

    # Run the simulation
    for t in times:
        tic = time.time()
        # Solve the OCP
        x = np.array(
            [
                state.motor_angle,
                state.pendulum_angle,
                state.motor_angle_velocity,
                state.pendulum_angle_velocity,
            ]
        )
        u = controller.compute_command(x, 1, x_ws, u_ws)
        toc = time.time()
        compute_time = toc - tic
        # Simulate the robot
        desired_state = controller.get_trajectoy()
        motor_angle, pendulum_angle = sim.step(u, time_step)
        # Compute speed via finite differences
        unfiltered_motor_speed = (motor_angle - state.motor_angle) / time_step
        unfiltered_pendulum_speed = (pendulum_angle - state.pendulum_angle) / time_step
        # Filter speed
        motor_speed = motor_speed_filter(unfiltered_motor_speed)
        pendulum_speed = pendulum_speed_filter(unfiltered_pendulum_speed)
        # Get the warm start from the controller
        x_ws, u_ws = controller.get_warm_start()
        # Log data
        # FIXME: Use the measured state
        # state = State(motor_angle, pendulum_angle, motor_speed, pendulum_speed, action=u)
        state = State(sim.state[0], sim.state[1], sim.state[2], sim.state[3], action=u)
        logger.update(int(t * 1e9), state)

    # Close logger
    logger.stop()

    # Read log
    times, states = logger.load()

    # Animate
    robot_viewer.animate(times, states)
    robot_viewer.close()

    # Plot
    logger.plot(times, states)
