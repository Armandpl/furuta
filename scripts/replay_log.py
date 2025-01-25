import argparse

from furuta.logger import SimpleLogger

# from furuta.robot import RobotModel
from furuta.viewer import Viewer2D  # , Viewer3D

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=False, default="2D", choices=("2D", "3D"))
    parser.add_argument("-f", "--file_path", required=True)
    args = parser.parse_args()

    logger = SimpleLogger(args.file_path)
    times, states = logger.load()

    if args.type == "2D":
        viewer = Viewer2D()
    else:
        print("3D viewer is not supported yet")
        assert False
        # viewer = Viewer3D(RobotModel.robot)

    viewer.animate(times, states)
    viewer.close()

    logger.plot(times, states)
