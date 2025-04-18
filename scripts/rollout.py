import argparse
from time import strftime

import numpy as np

from furuta.logger import SimpleLogger
from furuta.robot import RobotModel
from furuta.sim import SimulatedRobot
from furuta.utils import State
from furuta.viewer import Viewer3D

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir", default="../logs/rollout/", required=False, help="Log destination directory"
    )
    args = parser.parse_args()

    # Time constants
    t_final = 3.0
    control_freq = 100  # Hz
    time_step = 1 / control_freq

    times = np.arange(0.0, t_final, time_step)

    # Set the initial state
    state = State(0.0, 3.14, 0.0, 0.0)

    # Robot
    robot = RobotModel().robot

    # Create the simulation
    sim_state = np.array(
        [
            state.motor_angle,
            state.pendulum_angle,
            state.motor_angle_velocity,
            state.pendulum_angle_velocity,
        ]
    )
    sim = SimulatedRobot(robot, sim_state, dt=1e-5)

    # Create the logger
    fname = f"{strftime('%Y%m%d-%H%M%S')}.mcap"
    logger = SimpleLogger(log_path=args.dir + fname)
    logger.start()
    logger.update(int(times[0] * 1e9), state)

    # Rollout
    u = 0.0
    for i, t in enumerate(times[1:]):
        motor_angle, pendulum_angle = sim.step(u, time_step)
        motor_speed = (motor_angle - state.motor_angle) / time_step
        pendulum_speed = (pendulum_angle - state.pendulum_angle) / time_step
        state = State(motor_angle, pendulum_angle, motor_speed, pendulum_speed, action=u)
        logger.update(int(t * 1e9), state)

    # Close logger
    logger.stop()

    # Read log
    times, states = logger.load()

    # Animate
    robot_viewer = Viewer3D(robot)
    robot_viewer.animate(times, states)
    robot_viewer.close()

    # Plot
    logger.plot(times, states)
