import typing as tp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STATE = ["phi", "theta", "phi_dot", "theta_dot"]


class SimpleLogger:
    def __init__(self):
        self.times: tp.List[float] = []
        self.states: tp.List[np.ndarray] = []

    def update(self, time: float, state: np.ndarray):
        self.times.append(time)
        self.states.append(state)


class Logger(SimpleLogger):
    def __init__(self):
        super().__init__()
        self.controls: tp.List[float] = []
        self.elapsed_times: tp.List[float] = []

    def update(
        self, time: float, state: np.ndarray, control: float = 0.0, elapsed_time: float = 0.0
    ):
        super().update(time, state)
        self.controls.append(control)
        self.elapsed_times.append(elapsed_time)

    def save(self, directory: Path):
        np.save(directory / "times.npy", self.times)
        np.save(directory / "states.npy", np.array(self.states))
        np.save(directory / "controls.npy", self.controls)
        np.save(directory / "elapsed_times.npy", self.elapsed_times)

    def load(self, directory: Path):
        self.states = np.load(directory / "states.npy")
        self.times = np.load(directory / "times.npy")
        self.controls = np.load(directory / "controls.npy")
        self.elapsed_times = np.load(directory / "elapsed_times.npy")

    def plot(self):
        plt.figure(1)
        for i in range(len(STATE)):
            plt.subplot(2, 2, i + 1)
            plt.plot(self.times, np.array(self.states)[:, i])
            plt.title(STATE[i])

        plt.figure(2)
        plt.plot(self.times, self.controls)
        plt.title("Control")

        plt.figure(3)
        plt.plot(self.times, self.elapsed_times)
        plt.title("Elapsed Time")

        plt.show()