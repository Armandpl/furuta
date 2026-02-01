import time

from furuta.robot import Robot

if __name__ == "__main__":
    robot = Robot("/dev/tty.usbmodem2101", motor_encoder_cpr=1024, pendulum_encoder_cpr=4096)
    robot.reset_encoders()

    while True:
        motor_angle, pendulum_angle, timestamp = robot.step(0.0)
        print(f"motor_angle: {motor_angle:.2f}, pendulum_angle: {pendulum_angle:.2f}")
        time.sleep(0.01)
