import argparse

from furuta.logger import Loader
from furuta.plotter import Plotter
from furuta.robot import RobotModel
from furuta.viewer import Viewer2D, Viewer3D

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=False, default="2D", choices=("2D", "3D"))
    parser.add_argument("-f", "--file_path", required=True)
    args = parser.parse_args()

    # Read log
    loader = Loader()
    times, states_dict = loader.load(args.file_path)

    # Plot
    plotter = Plotter(times, states_dict)
    plotter.plot()

    if args.type == "2D":
        viewer = Viewer2D()
    else:
        viewer = Viewer3D(RobotModel().robot)

    # Animate
    states = loader.get_state("measured")
    viewer.animate(times, states)
    viewer.close()
