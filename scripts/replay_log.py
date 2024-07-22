from furuta.logger import Logger
from furuta.robot import RobotModel
from furuta.utils import ROOT_DIR
from furuta.viewer import RobotViewer

logger = Logger()
logger.load(ROOT_DIR / "logs/rollout")
logger.plot()

robot_viewer = RobotViewer(RobotModel.robot)
robot_viewer.animate_log(logger)
