import struct

import numpy as np
import serial


class Robot:
    def __init__(self, device="dev/ttyACM0", baudrate=921600):
        self.ser = serial.Serial(device, baudrate)

    def step(self, motor_command: float):
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


class QubeDynamics:
    """Solve equation M qdd + C(q, qd) = tau for qdd."""

    def __init__(
        self,
        g=9.81,
        Rm=8.4,
        V=12.0,
        km=0.042,
        Mr=0.095,
        Lr=0.085,
        Dr=5e-6,
        Mp=0.024,
        Lp=0.129,
        Dp=1e-6,
    ):
        # Gravity
        self.g = g

        # Motor
        self.Rm = Rm
        self.V = V

        # back-emf constant (V-s/rad)
        self.km = km

        # Rotary arm
        self.Mr = Mr
        self.Lr = Lr
        self.Dr = Dr

        # Pendulum link
        self.Mp = Mp
        self.Lp = Lp
        self.Dp = Dp

        # Init constants
        self._init_const()

    def _init_const(self):
        # Moments of inertia
        Jr = self.Mr * self.Lr**2 / 12  # inertia about COM (kg-m^2)
        Jp = self.Mp * self.Lp**2 / 12  # inertia about COM (kg-m^2)

        # Constants for equations of motion
        self._c = np.zeros(5)
        self._c[0] = Jr + self.Mp * self.Lr**2
        self._c[1] = 0.25 * self.Mp * self.Lp**2
        self._c[2] = 0.5 * self.Mp * self.Lp * self.Lr
        self._c[3] = Jp + self._c[1]
        self._c[4] = 0.5 * self.Mp * self.Lp * self.g

    @property
    def params(self):
        params = self.__dict__.copy()
        params.pop("_c")
        return params

    @params.setter
    def params(self, params):
        self.__dict__.update(params)
        self._init_const()

    def __call__(self, state, action):
        # """
        # action between 0 and 1, maps to +V and -V
        # """
        th, al, thd, ald = state
        voltage = action * self.V

        # Precompute some values
        sin_al = np.sin(al)
        sin_2al = np.sin(2 * al)
        cos_al = np.cos(al)

        # Define mass matrix M = [[a, b], [b, c]]
        a = self._c[0] + self._c[1] * sin_al**2
        b = self._c[2] * cos_al
        c = self._c[3]
        d = a * c - b * b

        # Calculate vector [x, y] = tau - C(q, qd)
        trq = self.km * (voltage - self.km * thd) / self.Rm
        c0 = self._c[1] * sin_2al * thd * ald - self._c[2] * sin_al * ald * ald
        c1 = -0.5 * self._c[1] * sin_2al * thd * thd + self._c[4] * sin_al
        x = trq - self.Dr * thd - c0
        y = -self.Dp * ald - c1

        # Compute M^{-1} @ [x, y]
        thdd = (c * x - b * y) / d
        aldd = (a * y - b * x) / d

        return thdd, aldd
