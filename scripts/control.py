import json

import matplotlib.pyplot as plt
import numpy as np

from furuta.controls.controllers import Controller
from furuta.robot import Robot


def read_parameters_file():
    with open("scripts/configs/parameters.json") as f:
        parameters = json.load(f)
    return parameters


def has_pendulum_fallen(pendulum_angle: float, parameters: dict):
    setpoint = parameters["pendulum_controller"]["setpoint"]
    angle_threshold = parameters["angle_threshold"]
    return np.abs(pendulum_angle - setpoint) > angle_threshold


def plot_data(actions, motor_angles, pendulum_angles):
    # plot actions
    plt.figure(1)
    plt.plot(actions)
    plt.title("Actions Over Time")
    plt.xlabel("Time")
    plt.ylabel("Action Value")
    plt.grid(True)

    # plot motor angles
    plt.figure(2)
    plt.plot(motor_angles)
    plt.title("Motor Angles Over Time")
    plt.xlabel("Time")
    plt.ylabel("Motor Angle (in radians)")
    plt.grid(True)

    # plot pendulum angles
    plt.figure(3)
    plt.plot(pendulum_angles)
    plt.title("Pendulum Angles Over Time")
    plt.xlabel("Time")
    plt.ylabel("Pendulum Angle (in radians)")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    # Read parameters from the .json file, angles are in degrees
    parameters = read_parameters_file()

    # Convert the setpoint and the angle threshold to radians
    parameters["pendulum_controller"]["setpoint"] = np.deg2rad(
        parameters["pendulum_controller"]["setpoint"]
    )
    parameters["angle_threshold"] = np.deg2rad(parameters["angle_threshold"])

    robot = Robot(parameters["device"])

    # Init data lists
    actions = []
    motor_angles = []
    pendulum_angles = []

    # Init pendulum controller
    pendulum_controller = Controller.build_controller(parameters["pendulum_controller"])

    # Init motor controller
    motor_controller = Controller.build_controller(parameters["motor_controller"])

    # Reset encoders
    robot.reset_encoders()

    # Wait for user input to start the control loop
    input("Encoders reset, lift the pendulum and press enter to start the control loop.")

    # Get the initial motor and pendulum angles
    motor_angle, pendulum_angle = robot.step(0)

    # Control loop
    while True:
        # Init action
        action = 0

        # Add the motor command from pendulum controller
        action -= pendulum_controller.compute_command(pendulum_angle)

        # Add the motor command from motor controller
        action -= motor_controller.compute_command(motor_angle)

        # Clip the command between -1 and 1
        action = np.clip(action, -1, 1)

        # Call the step function and get the motor and pendulum angles
        motor_angle, pendulum_angle = robot.step(action)

        # Take the modulus of the pendulum angle between 0 and 2pi
        pendulum_angle = np.mod(pendulum_angle, 2 * np.pi)

        # Append the data to the lists
        actions.append(action)
        motor_angles.append(motor_angle)
        pendulum_angles.append(pendulum_angle)

        # Check if pendulum fell
        if has_pendulum_fallen(pendulum_angle, parameters):
            print("Pendulum fell!")
            break

    # Close the serial connection
    robot.close()

    # Plot data
    plot_data(actions, motor_angles, pendulum_angles)
