import matplotlib.pyplot as plt
import numpy as np

from furuta.controls.controllers import Controller
from furuta.controls.utils import read_parameters_file
from furuta.robot import Robot

PARAMETERS_PATH = "scripts/configs/parameters.json"

DEVICE = "/dev/ttyACM0"


class Data:
    def __init__(self):
        self.actions = []
        self.motor_angles = []
        self.pendulum_angles = []

    def log_action(self, action):
        self.actions.append(action)

    def log_motor_angle(self, angle):
        self.motor_angles.append(angle)

    def log_pendulum_angle(self, angle):
        self.pendulum_angles.append(angle)

    def plot(self):
        # plot actions
        plt.figure(1)
        plt.plot(self.actions)
        plt.title("Actions Over Time")
        plt.xlabel("Time")
        plt.ylabel("Action Value")
        plt.grid(True)

        # plot motor angles
        plt.figure(2)
        plt.plot(np.rad2deg(self.motor_angles))
        plt.title("Motor Angles Over Time")
        plt.xlabel("Time")
        plt.ylabel("Motor Angle (in deg)")
        plt.grid(True)

        # plot pendulum angles
        plt.figure(3)
        plt.plot(np.rad2deg(self.pendulum_angles))
        plt.title("Pendulum Angles Over Time")
        plt.xlabel("Time")
        plt.ylabel("Pendulum Angle (in deg)")
        plt.grid(True)

        plt.show()


class SafeRobotWrapper:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.__motor_max_number_rev = 0.5
        self.__pendulum_angle_threshold = np.deg2rad(60)
        self.__pendulum_setpoint = np.pi

    def reset_encoders(self):
        self.robot.reset_encoders()

    def close(self):
        self.robot.close()

    def step(self, action):
        # Perform the action
        motor_angle, pendulum_angle = self.robot.step(action)

        # Run safety checks
        self.__run_checks(motor_angle, np.mod(pendulum_angle, 2 * np.pi))

        return motor_angle, pendulum_angle

    def __run_checks(self, motor_angle: float, pendulum_angle: float):
        if self.__is_motor_out_of_bounds(motor_angle):
            raise ValueError("Motor is out of bounds")

        if self.__has_pendulum_fallen(pendulum_angle):
            raise ValueError("Pendulum has fallen")

    def __has_pendulum_fallen(self, pendulum_angle):
        return np.abs(pendulum_angle - self.__pendulum_setpoint) > self.__pendulum_angle_threshold

    def __is_motor_out_of_bounds(self, motor_angle):
        return np.abs(motor_angle) > self.__motor_max_number_rev * 2 * np.pi


def main():
    # Read parameters from the .json file, angles are in degrees
    parameters = read_parameters_file(PARAMETERS_PATH)

    # Init safe robot
    robot = SafeRobotWrapper(Robot(DEVICE))

    # Init data logger
    data = Data()

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
        try:
            motor_angle, pendulum_angle = robot.step(action)
        except ValueError as error_message:
            print(error_message)
            break

        # Take the modulus of the pendulum angle between 0 and 2pi
        pendulum_angle = np.mod(pendulum_angle, 2 * np.pi)

        # Log data
        data.log_action(action)
        data.log_motor_angle(motor_angle)
        data.log_pendulum_angle(pendulum_angle)

    # Close the serial connection
    robot.close()

    # Plot data
    data.plot()


if __name__ == "__main__":
    main()
