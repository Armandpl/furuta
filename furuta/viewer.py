import time

import numpy as np
import pinocchio as pin
from panda3d_viewer import Viewer
from pinocchio.visualize.panda3d_visualizer import Panda3dVisualizer

from furuta.logger import SimpleLogger


class RobotViewer:
    def __init__(self, robot: pin.RobotWrapper = None):
        self.robot = robot
        viewer = Viewer()
        # Attach the robot to the viewer scene
        self.robot.setVisualizer(Panda3dVisualizer())
        self.robot.initViewer(viewer=viewer)
        self.robot.loadViewerModel(group_name=self.robot.model.name)

    def display(self, q: np.ndarray):
        self.robot.display(q)

    def animate(self, times: np.ndarray, states: np.ndarray):
        # Initial state
        q = np.array(states)[:, :2]
        self.display(q[0])
        time.sleep(1.0)
        tic = time.time()
        for i in range(1, len(times)):
            toc = time.time()
            time.sleep(max(0, times[i] - times[i - 1] - (toc - tic)))
            tic = time.time()
            if i % 10 == 0:
                self.display(q[i])

    def animate_log(self, log: SimpleLogger):
        self.animate(log.times, log.states)
