from dataclasses import dataclass, field

from furuta.logging.protobuf.pendulum_state_pb2 import PendulumState as PBState
from furuta.logging.protobuf.pendulum_state_pb2 import Signal as PBSignal


@dataclass
class Signal:
    desired: float = 0.0
    measured: float = 0.0
    filtered: float = 0.0
    simulated: float = 0.0

    def to_dict(self) -> dict:
        return {
            "desired": self.desired,
            "measured": self.measured,
            "filtered": self.filtered,
            "simulated": self.simulated,
        }

    def to_protobuf(self) -> PBSignal:
        return PBSignal(
            desired=self.desired,
            measured=self.measured,
            filtered=self.filtered,
            simulated=self.simulated,
        )

    @classmethod
    def from_protobuf(cls, pb_signal: PBSignal) -> "Signal":
        return cls(
            desired=pb_signal.desired,
            measured=pb_signal.measured,
            filtered=pb_signal.filtered,
            simulated=pb_signal.simulated,
        )


@dataclass
class State:
    motor_position: Signal = field(default_factory=Signal)
    pendulum_position: Signal = field(default_factory=Signal)
    motor_velocity: Signal = field(default_factory=Signal)
    pendulum_velocity: Signal = field(default_factory=Signal)
    reward: float = 0.0
    action: float = 0.0
    timing: float = 0.0

    def to_dict(self) -> dict:
        nested_dict = {
            "motor_position": self.motor_position.to_dict(),
            "pendulum_position": self.pendulum_position.to_dict(),
            "motor_velocity": self.motor_velocity.to_dict(),
            "pendulum_velocity": self.pendulum_velocity.to_dict(),
            "reward": self.reward,
            "action": self.action,
            "timing": self.timing,
        }
        flat_dict = {}
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_dict[f"{key}.{subkey}"] = subvalue
            else:
                flat_dict[key] = value
        return flat_dict

    def to_protobuf(self) -> PBState:
        return PBState(
            motor_position=self.motor_position.to_protobuf(),
            pendulum_position=self.pendulum_position.to_protobuf(),
            motor_velocity=self.motor_velocity.to_protobuf(),
            pendulum_velocity=self.pendulum_velocity.to_protobuf(),
            reward=self.reward,
            action=self.action,
            timing=self.timing,
        )

    @classmethod
    def from_protobuf(cls, pb_state: PBState) -> "State":
        return cls(
            motor_position=Signal.from_protobuf(pb_state.motor_position),
            pendulum_position=Signal.from_protobuf(pb_state.pendulum_position),
            motor_velocity=Signal.from_protobuf(pb_state.motor_velocity),
            pendulum_velocity=Signal.from_protobuf(pb_state.pendulum_velocity),
            reward=pb_state.reward,
            action=pb_state.action,
            timing=pb_state.timing,
        )
