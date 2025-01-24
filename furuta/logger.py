from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mcap_protobuf.reader import read_protobuf_messages
from mcap_protobuf.writer import Writer

from furuta.logging.protobuf.pendulum_state_pb2 import PendulumState
from furuta.utils import STATE, State


class SimpleLogger:
    def __init__(self, log_path: (str | Path)):
        self.log_path = log_path

    def start(self):
        self.output_file = open(self.log_path, "wb")
        self.mcap_writer = Writer(self.output_file)

    def stop(self):
        self.mcap_writer.finish()
        self.output_file.close()

    def update(self, time_ns: int, state: State):
        self.mcap_writer.write_message(
            topic="/pendulum_state",
            message=PendulumState(
                motor_angle=state.motor_angle,
                pendulum_angle=state.pendulum_angle,
                motor_angle_velocity=state.motor_angle_velocity,
                pendulum_angle_velocity=state.pendulum_angle_velocity,
                reward=state.reward,
                action=state.action,
            ),
            log_time=time_ns,
            publish_time=time_ns,
        )

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        times: List[float] = list()
        states: List[np.ndarray] = list()
        for msg in read_protobuf_messages(self.log_path, log_time_order=True):
            p = msg.proto_msg
            state = np.array(
                [
                    p.motor_angle,
                    p.pendulum_angle,
                    p.motor_angle_velocity,
                    p.pendulum_angle_velocity,
                    p.reward,
                    p.action,
                ]
            )
            times.append(float(msg.log_time_ns * 1e-9))
            states.append(state)
        return np.array(times), np.array(states)

    def plot(self, times: List[float], states: List[np.ndarray]):
        for title, idx in STATE.items():
            plt.figure(idx + 1)
            plt.plot(times, states[:, idx])
            plt.title(title)
        plt.show()
