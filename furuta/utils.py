# https://git.ias.informatik.tu-darmstadt.de/quanser/clients/-/blob/v0.1.1/quanser_robots/common.py
import numpy as np
from gymnasium import spaces
from scipy import signal

THETA = 0
ALPHA = 1
THETA_DOT = 2
ALPHA_DOT = 3


class VelocityFilter:
    """Discrete velocity filter derived from a continuous one."""

    def __init__(self, x_len, dt, num=(50, 0), den=(1, 50), x_init=None):
        """Initialize discrete filter coefficients.

        :param x_len: number of measured state variables to receive
        :param num: continuous-time filter numerator
        :param den: continuous-time filter denominator
        :param dt: sampling time interval
        :param x_init: initial observation of the signal to filter
        """
        derivative_filter = signal.cont2discrete((num, den), dt)
        self.b = derivative_filter[0].ravel().astype(np.float32)
        self.a = derivative_filter[1].astype(np.float32)
        if x_init is None:
            self.z = np.zeros((max(len(self.a), len(self.b)) - 1, x_len), dtype=np.float32)
        else:
            self.set_initial_state(x_init)

    def set_initial_state(self, x_init):
        """This method can be used to set the initial state of the velocity filter This is useful
        when the initial (position) observation has been retrieved and it is non-zero. Otherwise
        the filter would assume a very high velocity.

        :param x_init: initial observation
        """
        assert isinstance(x_init, np.ndarray)
        # Get the initial condition of the filter
        zi = signal.lfilter_zi(self.b, self.a)  # dim = order of the filter = 1
        # Set the filter state
        self.z = np.outer(zi, x_init)

    def __call__(self, x):
        xd, self.z = signal.lfilter(self.b, self.a, x[None, :], 0, self.z)
        return xd.ravel()


class Timing:
    def __init__(self, f):
        self.f = f
        self.dt = 1.0 / f
