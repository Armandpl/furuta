import struct

import matplotlib.pyplot as plt
import numpy as np
import serial
import simple_pid


class Robot:
    def __init__(self, device="dev/ttyACM0", baudrate=921600):
        self.ser = serial.Serial(device, baudrate)

    def step(self, motor_command: float):
        """motor command is a float between -1 and 1."""
        direction = motor_command < 0
        # convert motor command to 16 bit unsigned int
        int_motor_command = int(np.abs(motor_command) * (2**16 - 1))
        tx = b"\x10\x02"  # start sequence
        tx += b"\x01"  # command type = STEP = 0x01
        tx += struct.pack("<?H", direction, int_motor_command)
        self.ser.write(tx)
        resp = self.ser.read(8)
        motor_angle, pendulum_angle = struct.unpack("<ff", resp)
        return motor_angle, pendulum_angle

    def reset_encoders(self):
        tx = b"\x10\x02"  # start sequence
        tx += b"\x00"  # command type = RESET = 0x00
        tx += b"\x00\x00\x00"  # three empty bytes to have fixed len packets
        self.ser.write(tx)

    def close(self):
        self.step(0)
        self.ser.close()
