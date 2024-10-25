import argparse
import time

import numpy as np
from mcap_protobuf.reader import read_protobuf_messages

from furuta.robot import RobotModel
from furuta.viewer import Viewer2D, Viewer3D

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=False, default="2D", choices=("2D", "3D"))
    parser.add_argument("-f", "--file_path", required=True)
    args = parser.parse_args()
    if args.type == "2D":
        viewer = Viewer2D(render_fps=30, render_mode="rgb_array")
    else:
        viewer = Viewer3D(RobotModel.robot)

    # TODO get dt from the mcap file
    dt = 0.02

    for msg in read_protobuf_messages(args.file_path, log_time_order=True):
        p = msg.proto_msg
        state = np.array(
            [p.motor_angle, p.pendulum_angle, p.motor_angle_velocity, p.pendulum_angle_velocity]
        )
        viewer.display(state)
        time.sleep(dt)

    viewer.close()
