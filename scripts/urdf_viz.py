from time import sleep

import numpy as np
import pinocchio as pin
from panda3d_viewer import Viewer
from pinocchio.visualize.panda3d_visualizer import Panda3dVisualizer

ROOT_DIR = "/home/pierfabre/pendulum_workspace/src/furuta/"

times = np.load(ROOT_DIR + "logs/rollout/times.npy")
motor_angles = np.load(ROOT_DIR + "logs/rollout/motor_angles.npy")
pendulum_angles = np.load(ROOT_DIR + "logs/rollout/pendulum_angles.npy")

viewer = Viewer(window_title="python-pinocchio")

# The robot is loaded as a RobotWrapper object
robot = pin.RobotWrapper.BuildFromURDF(
    ROOT_DIR + "robot/hardware/furuta.urdf", package_dirs=[ROOT_DIR + "robot/hardware/CAD/stl/"]
)
# Attach the robot to the viewer scene
robot.setVisualizer(Panda3dVisualizer())
robot.initViewer(viewer=viewer)
robot.loadViewerModel(group_name=robot.model.name)
model = robot.model
q = robot.q0[:]
q[0] = motor_angles[0]
q[1] = pendulum_angles[0]
robot.display(q)
sleep(5.0)
for i in range(1, len(times)):
    sleep(times[i] - times[i - 1])
    q[0] = motor_angles[i]
    q[1] = pendulum_angles[i]
    robot.display(q)
