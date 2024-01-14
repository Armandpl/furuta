import time

import numpy as np

from furuta.robot import Robot
from scripts.control import has_pendulum_fallen

# not using pytest here bc couldn't get user input during test session


def test_control_freq(device):
    robot = Robot(device)
    nb_iterations = 10_000
    start_time = time.time()

    for _ in range(nb_iterations):
        robot.step(0)

    elapsed_time = time.time() - start_time
    mean_loop_time = elapsed_time / nb_iterations
    max_control_freq = round(1 / mean_loop_time)
    robot.close()
    assert max_control_freq > 100, "Control frequency weirdly low."
    return f"max_control_freq: {max_control_freq}"


def test_motor_stop(device):
    robot = Robot(device)  # replace with your actual constructor if needed
    robot.step(0.35)

    time.sleep(0.5)  # wait for 500ms

    # Ask the user if the motor has stopped
    user_response = input("Has the motor stopped? (yes/no): ")
    robot.close()
    assert user_response.lower() == "yes", "The motor did not stop after 500ms"
    return ""


def test_motor_direction(device):
    robot = Robot(device)
    robot.step(0.35)
    time.sleep(0.5)  # wait for 500ms

    user_response = input("Was the motor spinning clockwise? (yes/no): ")
    robot.close()
    assert user_response.lower() == "yes", "Motor spinning backwards, swap motor wires."
    return ""


def test_has_pendulum_fallen(device):
    robot = Robot(device)
    robot.reset_encoders()

    input("Lift the pendulum and press enter, then let the pendulum fall")
    time.sleep(3.0)

    _, pendulum_angle = robot.step(0.0)
    robot.close()
    assert has_pendulum_fallen(
        pendulum_angle, setpoint=np.pi, angle_threshold=np.deg2rad(60.0)
    ), "The fall was not detcected"
    return ""


def print_report(results):
    for result in results:
        print(result)


def run_tests(tests, device):
    results = []

    for test in tests:
        try:
            result = test(device)
            results.append(f"{test.__name__}: OK\n{str(result)}")
        except Exception as e:
            results.append(f"{test.__name__}: FAIL\n{str(e)}")

    return results


if __name__ == "__main__":
    device = input("Enter robot device (press enter for default: /dev/ttyACM0): ")
    if device == "":
        device = "/dev/ttyACM0"

    tests = [test_control_freq, test_motor_stop, test_motor_direction]

    results = run_tests(tests, device)
    print()
    print_report(results)
