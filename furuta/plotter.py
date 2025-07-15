from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from furuta.logger import Loader


@dataclass
class Plot:
    name: str
    unit: str
    variables: list[str]


@dataclass
class Tab:
    name: str
    shape: tuple[int, int]
    plots: list[Plot]


@dataclass
class Layout:
    name: str
    tabs: list[Tab]


def load_layout(path: str | Path) -> Layout:
    with open(path) as f:
        data = yaml.safe_load(f)

    tabs = []
    for tab in data["tabs"]:
        plots = []

        for plot in tab["plots"]:
            plots.append(Plot(name=plot["name"], unit=plot["unit"], variables=plot["variables"]))

        tabs.append(Tab(name=tab["name"], shape=tuple(tab["shape"]), plots=plots))

    return Layout(name=data["name"], tabs=tabs)


class Plotter:
    def __init__(self, times: np.ndarray, states: dict[str, np.ndarray]):
        self.times = times
        self.states = states

    @classmethod
    def from_log_path(cls, log_path: str | Path) -> "Plotter":
        loader = Loader()
        times, states = loader.load(log_path)
        return cls(times, states)

    def plot(self):
        layout_path = (
            Path(__file__).resolve().parent.parent
            / "scripts"
            / "configs"
            / "logging"
            / "layout.yaml"
        )
        layout = load_layout(layout_path)
        for k, tab in enumerate(layout.tabs):
            plt.figure(k + 1)
            plt.suptitle(tab.name)

            for idx, plot in enumerate(tab.plots):
                ax = plt.subplot(*tab.shape, idx + 1)
                ax.set_title(plot.name)

                for var in plot.variables:
                    if var not in self.states:
                        print(f"Warning: Variable '{var}' not found in states.")
                        continue
                    if np.all(self.states[var] == 0.0):
                        continue
                    ax.plot(self.times, self.states[var], label=var)

                ax.set_ylabel(plot.unit)
                ax.set_xlabel("Time (s)")
                ax.grid(True, linestyle="--", alpha=0.7)
                ax.legend(loc="best")

        plt.show()
