import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from furuta.logger import SimpleLogger, State
from furuta.robot import Robot

DEVICE = "/dev/ttyACM0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        default="../logs/XP/low_level_control/",
        required=False,
        help="Log destination directory",
    )
    args = parser.parse_args()

    # Init robot
    robot = Robot(DEVICE)

    # Constants
    control_freq = 100.0
    dt = 1.0 / control_freq
    f = 2
    A = np.pi / 2

    desired_motor_position = 0.0
    desired_motor_velocity = 0.0

    # Set the initial state
    state = State(motor_angle=0.0, motor_angle_velocity=0.0, action=0.0)

    # Create the logger
    fname = f"{time.strftime('%Y%m%d-%H%M%S')}.mcap"
    logger = SimpleLogger(log_path=args.dir + fname)
    logger.start()
    logger.update(int(0.0 * 1e9), state)

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
            desired_motor_position = A * np.sin(2 * np.pi * f * t)
            desired_motor_velocity = 2 * np.pi * f * A * np.cos(2 * np.pi * f * t)

            (
                motor_angle,
                pendulum_angle,
                motor_velocity,
                pendulum_velocity,
                timestamp,
                action,
            ) = robot.step(desired_motor_position, desired_motor_velocity)
            state = State(
                motor_angle=motor_angle,
                pendulum_angle=desired_motor_position,
                motor_angle_velocity=motor_velocity,
                pendulum_angle_velocity=desired_motor_velocity,
                reward=pendulum_angle,
                action=action,
            )

            logger.update(int(t * 1e9), state)
            tic = time.time()

    # Close logger
    logger.stop()

    # Read log
    times, states = logger.load()

    # Plot
    # logger.plot(times, states)

    plt.figure(1)
    plt.title("Motor Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Motor angle (rad)")
    plt.plot(times, states[:, 1])
    plt.plot(times, states[:, 0])
    plt.legend(["desired", "measured"])

    plt.figure(2)
    plt.title("Motor Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Motor velocity (rad/s)")
    plt.plot(times, states[:, 3])
    plt.plot(times, states[:, 2])
    plt.legend(["desired", "measured"])

    plt.figure(3)
    plt.title("Action")
    plt.xlabel("Time (s)")
    plt.plot(times, states[:, 5])

    plt.figure(4)
    plt.title("Elapsed time")
    plt.plot(times, states[:, 4])

    plt.show()
