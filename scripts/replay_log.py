from pathlib import Path

from furuta.logger import Logger
from furuta.robot import RobotModel
from furuta.viewer import RobotViewer

logger = Logger()
logger.load(Path("../logs/rollout"))
logger.plot()

robot_viewer = RobotViewer(RobotModel.robot)
robot_viewer.animate_log(logger)
