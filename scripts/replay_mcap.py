from mcap_protobuf.reader import read_protobuf_messages
from furuta_gym.envs.furuta_base import CartPoleSwingUpViewer
import numpy as np
import time

if __name__ == "__main__":
    # TODO use smth like argparse to choose the file
    # or a text interface
    mcap_path = "../data/3qhetqlp/ep266_20221201-012458.mcap"
    viewer = CartPoleSwingUpViewer(world_width=5)
    # TODO get dt from the mcap file
    dt = 0.02

    for msg in read_protobuf_messages(mcap_path, log_time_order=True):
        p = msg.proto_msg
        state = np.array([p.motor_angle, p.pendulum_angle, p.motor_angle_velocity, p.pendulum_angle_velocity])
        viewer.update(state)
        viewer.render(return_rgb_array=False)
        time.sleep(dt)