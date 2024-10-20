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

    def save(self, directory: Path):
        np.save(directory / "times.npy", self.times)
        np.save(directory / "states.npy", np.array(self.states))

    def load(self, directory: Path):
        self.states = np.load(directory / "states.npy")
        self.times = np.load(directory / "times.npy")

    def plot(self):
        plt.figure(1)
        for i in range(len(STATE)):
            plt.subplot(2, 2, i + 1)
            plt.plot(self.times, np.array(self.states)[:, i])
            plt.title(STATE[i])

    def show():
        plt.show()
