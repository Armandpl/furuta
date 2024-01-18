from abc import ABC, abstractmethod

from simple_pid import PID


class Controller(ABC):
    @abstractmethod
    def compute_command(self, position: float):
        pass

    @staticmethod
    def build_controller(parameters: dict):
        controller_type = parameters["controller_type"]
        # match controller_type:
        #     case "PIDController":
        #         return PIDController(parameters)
        #     case _:
        #         raise ValueError(f"Invalid controller type: {controller_type}")
        if controller_type == "PIDController":
            return PIDController(parameters)
        else:
            raise ValueError(f"Invalid controller type: {controller_type}")


class PIDController(Controller):
    def __init__(self, parameters):
        try:
            sample_time = 1 / parameters["control_frequency"]
        except KeyError:
            sample_time = None

        try:
            self.pid = PID(
                Kp=parameters["Kp"],
                Ki=parameters["Ki"],
                Kd=parameters["Kd"],
                setpoint=parameters["setpoint"],
                sample_time=sample_time,
            )
        except KeyError:
            raise ValueError("Invalid PID parameters")

    def compute_command(self, position: float):
        return self.pid(position)
