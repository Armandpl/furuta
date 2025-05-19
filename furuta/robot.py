import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
import serial


class RobotModel:
    def __init__(self):
        self.robot = None
        base_path = Path("robot/hardware/v2/")
        if base_path.exists():
            urdf_path = str(base_path / "robot.urdf")
            stls_path = str(base_path / "stl")
            self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, [stls_path])


class Robot:
    def __init__(
        self,
        device="/dev/ttyACM0",
        baudrate=921600,
        motor_encoder_cpr=400,
        pendulum_encoder_cpr=5120 * 4,
    ):
        self.ser = serial.Serial(device, baudrate)
        self.motor_encoder_cpr = motor_encoder_cpr
        self.pendulum_encoder_cpr = pendulum_encoder_cpr

    def step(self, motor_command: float):
        direction = motor_command < 0
        # convert motor command to 16 bit unsigned int
        int_motor_command = int(np.abs(motor_command) * (2**16 - 1))
        tx = b"\x10\x02"  # start sequence
        tx += b"\x01"  # command type = STEP = 0x01
        tx += struct.pack("<?H", direction, int_motor_command)
        self.ser.write(tx)
        resp = self.ser.read(12)
        raw_motor_angle, raw_pendulum_angle, raw_timestamp = struct.unpack("<iiL", resp)
        motor_angle = 2 * np.pi * raw_motor_angle / self.motor_encoder_cpr
        pendulum_angle = 2 * np.pi * raw_pendulum_angle / self.pendulum_encoder_cpr
        timestamp = raw_timestamp / 1e6
        return motor_angle, pendulum_angle, timestamp

    def reset_encoders(self):
        tx = b"\x10\x02"  # start sequence
        tx += b"\x00"  # command type = RESET = 0x00
        tx += b"\x00\x00\x00"  # three empty bytes to have fixed len packets
        self.ser.write(tx)

    def close(self):
        self.step(0)
        self.ser.close()


@dataclass
class MotorDynamics:
    V = 24.0  # voltage (V) only used to store the value to rescale RL actions from [-1, 1] to [-V, V]
    R = 8.0  # resistance (Ohm)
    r = 6.25  # gear ratio
    J_mot = 1.0 * 1e-8  # motor inertia (kg m^2)
    Kc = 0.235  # torque constant (N-m/A)
    Ke = 0.235  # back-emf constant (V-s/rad)
    f_v = 3e-4  # viscous damping (N-m-s/rad)
    f_s = 1e-1  # Coulomb
    f_c = 8e-3  # striction
    phi_dot_eps = 0.01  # limit stiction speed (rad/s)
    L = 0.24  # inductance (H)

    def compute_motor_torque(self, U: float, phi_dot: float) -> float:
        """
        This method computes the steady state torque in output of the gear box
        U: is the voltage applied to the motor
        phi_dot: is the angular velocity in output of the gear box (as measured by encoders)
        return: the torque in output of the gear box
        """
        return self.r * self.Kc * (U - self.Ke * self.r * phi_dot) / self.R


class PendulumDynamics:
    def __init__(self, robot: pin.RobotWrapper):
        self.f_p = 2e-4  # N-m-s/rad
        self.motor = MotorDynamics()
        self.model = robot.model
        self.data = self.model.createData()

    def __call__(self, q: np.ndarray, v: np.ndarray, u: float) -> np.ndarray:
        tau = np.array([u, 0.0])
        tau[1] = -self.f_p * v[1]
        tau[0] -= self.motor.f_v * v[0] + self.motor.f_c * np.sign(v[0])
        a = pin.aba(self.model, self.data, q, v, tau)
        return a


class Encoders:
    def __init__(self):
        self.motor_encoder_cpr = 400.0
        self.pendulum_encoder_cpr = 5120.0

    def measure(self, angles):
        motor_angle = (
            int(angles[0] / (2 * np.pi) * self.motor_encoder_cpr)
            * 2
            * np.pi
            / self.motor_encoder_cpr
        )
        pendulum_angle = (
            int(angles[1] / (2 * np.pi) * self.pendulum_encoder_cpr)
            * 2
            * np.pi
            / self.pendulum_encoder_cpr
        )
        return motor_angle, pendulum_angle


class QubeDynamics:
    """Solve equation M qdd + C(q, qd) = tau for qdd."""

    def __init__(
        self,
        g=9.81,
        # Motor
        Rm=8.4,  # reskstance (rated voltage/stall current)
        Rm_std=0.0,
        V=12.0,  # nominal voltage
        reduction_ratio=1.0,
        stall_torque=0.16,  # N/m
        km=0.042,  # back-emf constant (V-s/rad) = (rated voltage / no load speed)
        km_std=0.0,
        # Rotary Arm
        Mr=0.095,  # mass (kg)
        Mr_std=0.0,
        Lr=0.085,  # length (m)
        Lr_std=0.0,
        Dr=5e-6,  # viscous damping (N-m-s/rad), original: 0.0015
        Dr_std=0.0,
        # Pendulum Link
        Mp=0.024,  # mass (kg)
        Mp_std=0.0,
        Lp=0.129,  # length (m)
        Lp_std=0.0,
        Dp=1e-6,  # viscous damping (N-m-s/rad), original: 0.0005
        Dp_std=0.0,
    ):
        # Gravity
        self.g = g

        # Motor
        self.Rm_mean = Rm
        self.Rm_std = Rm_std
        self.V = V
        self.reduction_ratio = reduction_ratio
        self.stall_torque = stall_torque

        self.km_mean = km
        self.km_std = km_std

        # Rotary arm
        self.Mr_mean = Mr
        self.Mr_std = Mr_std
        self.Lr_mean = Lr
        self.Lr_std = Lr_std
        self.Dr_mean = Dr
        self.Dr_std = Dr_std

        # Pendulum link
        self.Mp_mean = Mp
        self.Mp_std = Mp_std
        self.Lp_mean = Lp
        self.Lp_std = Lp_std
        self.Dp_mean = Dp
        self.Dp_std = Dp_std

        self.randomize()

        # Init constants
        self._init_const()

    def randomize(self):
        self.Rm = np.random.normal(self.Rm_mean, self.Rm_std)
        self.km = np.random.normal(self.km_mean, self.km_std)
        self.Mr = np.random.normal(self.Mr_mean, self.Mr_std)
        self.Lr = np.random.normal(self.Lr_mean, self.Lr_std)
        self.Dr = np.random.normal(self.Dr_mean, self.Dr_std)
        self.Mp = np.random.normal(self.Mp_mean, self.Mp_std)
        self.Lp = np.random.normal(self.Lp_mean, self.Lp_std)
        self.Dp = np.random.normal(self.Dp_mean, self.Dp_std)

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
        trq = (
            self.reduction_ratio
            * self.km
            * (voltage - self.km * thd * self.reduction_ratio)
            / self.Rm
        )
        trq = np.clip(trq, -self.stall_torque, self.stall_torque)
        c0 = self._c[1] * sin_2al * thd * ald - self._c[2] * sin_al * ald * ald
        c1 = -0.5 * self._c[1] * sin_2al * thd * thd + self._c[4] * sin_al
        x = trq - self.Dr * thd - c0
        y = -self.Dp * ald - c1

        # Compute M^{-1} @ [x, y]
        thdd = (c * x - b * y) / d
        aldd = (a * y - b * x) / d

        return thdd, aldd
