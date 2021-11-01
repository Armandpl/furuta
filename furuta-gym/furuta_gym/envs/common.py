# https://git.ias.informatik.tu-darmstadt.de/quanser/clients/-/blob/v0.1.1/quanser_robots/common.py
import gym
from gym import spaces
import numpy as np
from scipy import signal


def deg2rad(deg):
    return deg * np.pi / 180


def rad2deg(rad):
    return rad * 180 / np.pi


class VelocityFilter:
    """
    Discrete velocity filter derived from a continuous one.
    """
    def __init__(self, x_len, dt, num=(50, 0), den=(1, 50), x_init=None):
        """
        Initialize discrete filter coefficients.
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
            self.z = np.zeros((max(len(self.a), len(self.b)) - 1, x_len),
                              dtype=np.float32)
        else:
            self.set_initial_state(x_init)

    def set_initial_state(self, x_init):
        """
        This method can be used to set the initial state of the velocity filter
        This is useful when the initial (position) observation
        has been retrieved and it is non-zero.
        Otherwise the filter would assume a very high velocity.
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


class LabeledBox(spaces.Box):
    """
    Adds `labels` field to gym.spaces.Box to keep track of variable names.
    """
    def __init__(self, labels, **kwargs):
        super(LabeledBox, self).__init__(**kwargs)
        assert len(labels) == self.high.size
        self.labels = labels


class Timing:
    def __init__(self, fs, fs_ctrl):
        fs_ctrl_min = 50.0  # minimal control rate
        assert fs_ctrl >= fs_ctrl_min, \
            "control frequency must be at least {}".format(fs_ctrl_min)
        self.n_sim_per_ctrl = int(fs / fs_ctrl)
        assert fs == fs_ctrl * self.n_sim_per_ctrl, \
            "sampling frequency must be a multiple of the control frequency"
        self.dt = 1.0 / fs
        self.dt_ctrl = 1.0 / fs_ctrl
        self.render_rate = int(fs_ctrl)


class PhysicSystem:

    def __init__(self, dt, **kwargs):
        self.dt = dt
        for k in kwargs:
            setattr(self, k, kwargs[k])
            setattr(self, k + "_dot", 0.)

    def add_acceleration(self, **kwargs):
        for k in kwargs:
            setattr(self, k + "_dot", getattr(self, k + "_dot") +
                    self.dt * kwargs[k])
            setattr(self, k, getattr(self, k) +
                    self.dt * getattr(self, k + "_dot"))

    def get_state(self, entities_list):
        ret = []
        for k in entities_list:
            ret.append(getattr(self, k))
        return np.array(ret)
