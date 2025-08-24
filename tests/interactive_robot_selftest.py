import sys
import time

import numpy as np

from furuta.robot import Robot

# not using pytest here bc couldn't get user input during test session


def test_control_freq(device):
    print("Running test_control_freq...")
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
    robot = Robot(device)
    robot.step(1.0)

    time.sleep(.5)

    user_response = input("Has the motor started then stopped? (yes/no): ")
    robot.close()
    assert user_response.lower() == "yes", "The motor did not stop after 500ms"
    return ""


def test_motor_direction(device):
    robot = Robot(device)
    robot.step(-1.0)
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

    _, pendulum_angle, _ = robot.step(0.0)
    robot.close()
    assert np.cos(pendulum_angle) < np.cos(np.deg2rad(15.0)), f"Pendulum angle too high: {pendulum_angle:.2f} rad"
    return f"pendulum angle OK: {pendulum_angle:.2f} rad"


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
    if len(sys.argv) > 1: device = sys.argv[1]
    else: device = "/dev/ttyACM0"

    tests = [test_control_freq, test_motor_stop, test_motor_direction, test_has_pendulum_fallen]

    results = run_tests(tests, device)
    print()
    print_report(results)
