from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from mcap_protobuf.reader import read_protobuf_messages
from mcap_protobuf.writer import Writer

from furuta.state import State


class SimpleLogger:
    def __init__(self, log_path: (str | Path)):
        self.output_file = open(log_path, "wb")
        self.mcap_writer = Writer(self.output_file)

    def stop(self):
        self.mcap_writer.finish()
        self.output_file.close()

    def update(self, time_ns: int, state: State):
        self.mcap_writer.write_message(
            topic="/pendulum_state",
            message=state.to_protobuf(),
            log_time=time_ns,
            publish_time=time_ns,
        )


class Loader:
    def __init__(self):
        self.states_dict: Dict[str, List] = {}
        self.times = []

    def load(self, log_path: Union[str, Path]) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        for msg in read_protobuf_messages(log_path, log_time_order=True):
            proto = msg.proto_msg
            self.times.append(msg.log_time_ns * 1e-9)
            state = State.from_protobuf(proto).to_dict()
            for key, value in state.items():
                if key not in self.states_dict:
                    self.states_dict[key] = []
                self.states_dict[key].append(value)

        times_array = np.array(self.times)
        states_array_dict = {k: np.array(v) for k, v in self.states_dict.items()}
        return times_array, states_array_dict

    def get_state(self, signal: str) -> np.ndarray:
        return np.array(
            [
                self.states_dict["motor_position." + signal],
                self.states_dict["pendulum_position." + signal],
                self.states_dict["motor_velocity." + signal],
                self.states_dict["pendulum_velocity." + signal],
            ]
        ).T
