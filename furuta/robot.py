import struct
from dataclasses import dataclass

import numpy as np
import serial


class Robot:
    def __init__(
        self,
        device="dev/ttyACM0",
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


class MotorDynamics:
    def __init__(
        self,
        V=24.0,  # voltage (V) only used to store the value to rescale RL actions from [-1, 1] to [-V, V]
        R=8.0,  # resistance (Ohm)
        r=6.25,  # gear ratio
        J_mot=1.0 * 1e-8,  # motor inertia (kg m^2)
        Kc=0.235,  # torque constant (N-m/A)
        Ke=0.235,  # back-emf constant (V-s/rad)
        f_v=1e-3,  # viscous damping (N-m-s/rad)
        f_s=1e-1,  # Coulomb
        f_c=5e-2,  # striction
        phi_dot_eps=0.01,  # limit stiction speed (rad/s)
        L=0.24,  # inductance (H)
    ):
        self.V = V
        self.R = R
        self.r = r
        self.J_mot = J_mot
        self.Kc = Kc
        self.Ke = Ke
        self.f_v = f_v
        self.f_s = f_s
        self.f_c = f_c
        self.phi_dot_eps = phi_dot_eps
        self.L = L

    def compute_motor_torque(self, U, phi_dot):
        """
        This method computes the steady state torque in output of the gear box
        U: is the voltage applied to the motor
        phi_dot: is the angular velocity in output of the gear box (as measured by encoders)
        return: the torque in output of the gear box
        """
        return self.r * self.Kc * (U - self.Ke * self.r * phi_dot) / self.R


class PendulumDynamics:
    def __init__(
        self,
        g=9.81,  # m/s^2
        m_bearing=11.5 * 1e-3,  # kg
        l_bearing=25.0 * 1e-3,  # m
        m_enc=17.0 * 1e-3,  # kg
        l_enc=35.0 * 1e-3,  # m
        m_shaft=2.8 * 1e-3,  # kg
        l_shaft=15.0 * 1e-3,  # m
        m_mount=5.25 * 1e-3,  # kg
        l_p=74.0 * 1e-3,  # m
        M=28.0 * 1e-3,  # kg
        L=86.0 * 1e-3,  # m
        f_p=4e-3,  # N-m-s/rad
        motor: MotorDynamics = MotorDynamics(),
    ):
        # Gravity
        self.g = g  # m/s^2
        # Bearings
        self.m_bearing = m_bearing  # kg
        self.l_bearing = l_bearing  # m
        # Encoder
        self.m_enc = m_enc  # kg
        self.l_enc = l_enc  # m
        # Shaft Collars
        self.m_shaft = m_shaft  # kg
        self.l_shaft = l_shaft  # m
        # Mount
        self.m_mount = m_mount  # kg
        self.l_mount = l_bearing  # m
        # Pendulum
        self.l_p = l_p  # m
        self.M = M  # kg
        self.L = L  # m
        self.f_p = f_p  # viscous damping (N-m-s/rad)
        # Motor
        self.motor = MotorDynamics()
        # Inertia
        self.J = (
            self.motor.r**2 * self.motor.J_mot
            + 2 * self.m_bearing * self.l_bearing**2
            + self.m_enc * self.l_enc**2
            + 2 * self.m_shaft * self.l_shaft**2
            + 2 * self.m_mount * self.l_mount**2
        )
        # Init dynamics
        self.phi_ddot = 0.0
        self.th_ddot = 0.0
        self.mot = motor

    def __call__(self, x, u):
        _, th, phi_dot, th_dot = x[0], x[1], x[2], x[3]
        # Shorter names
        J = self.J
        M = self.M
        L = self.L
        l_p = self.l_p
        g = self.g
        mot = self.motor
        f_p = self.f_p
        s, c = np.sin(th), np.cos(th)
        J_th = J + M * (L**2 + l_p**2) * c**2
        # Compute motor friction
        tau_f = 0.0
        # F_n = np.abs(self.M * self.th_ddot * self.L / s)
        tau = (
            M * g * l_p * s**2
            - M * L * l_p * s**2 * c * phi_dot**2
            + M * L * l_p * c * th_dot**2
            + 2 * M * L**2 * s * c * th_dot * phi_dot
            + u
        )
        # if abs(phi_dot) < mot.phi_dot_eps:
        #     # Stiction
        #     tau_s = mot.f_s * F_n
        #     if abs(tau) < tau_s:
        #         tau_f = tau
        #     else:
        #         tau_f = np.sign(tau) * mot.f_s * F_n
        # else:
        # Coulomb
        # tau_c = mot.f_c * F_n
        tau_c = 0.01
        tau_f = np.sign(phi_dot) * tau_c + mot.f_v * phi_dot

        # Compute dynamics
        phi_ddot = (tau - tau_f) / J_th
        th_ddot = (g * s - L * s * c * phi_dot**2 + l_p * s * phi_ddot - f_p * th_dot / M) / L
        self.phi_ddot = phi_ddot
        self.th_ddot = th_ddot
        # return np.array([phi_dot, th_dot, phi_ddot, th_ddot])
        return phi_ddot, th_ddot
