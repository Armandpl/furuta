import typing as tp
from abc import ABC, abstractmethod

import crocoddyl
import numpy as np
import pinocchio as pin
from simple_pid import PID


class Controller(ABC):
    @abstractmethod
    def compute_command(self, state):
        pass


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
                setpoint=np.deg2rad(parameters["setpoint"]),
                sample_time=sample_time,
            )
        except KeyError:
            raise ValueError("Invalid PID parameters")

    def compute_command(self, position: float):
        return self.pid(position)


class SwingUpController(Controller):
    class FurutaActuationModel(crocoddyl.ActuationModelAbstract):
        def __init__(self, state, nu):
            crocoddyl.ActuationModelAbstract.__init__(self, state, nu=nu)

        def calc(self, data, x, u):
            assert len(data.tau) == 2
            # Map the control dimensions to the joint torque
            data.tau[0] = u
            data.tau[1] = 0  # Underactuated joint

        def calcDiff(self, data, x, u):
            # Specify the actuation jacobian
            data.dtau_du[0] = 1
            data.dtau_du[1] = 0

    def __init__(
        self,
        robot: pin.RobotWrapper,
        x_target: np.ndarray,
        control_freq: float = 100.0,
        t_final: float = 0.5,
        u_lim: float = 0.1,
        Q: np.ndarray = np.array([10, 50, 1, 1]),
        R: np.ndarray = np.array([0.1]),
        S: np.ndarray = np.array([1.0]),
        M: int = 10,
    ):
        self.u_lim = u_lim

        # Time variables
        dt = 1 / control_freq  # Time step
        self.N = int(t_final * control_freq)

        # Instantiate robot as a pinocchio RobotWrapper
        self.robot = robot

        # State
        state = crocoddyl.StateMultibody(robot.model)

        # Actuation model
        nu = 1
        actuation = self.FurutaActuationModel(state, nu)

        # State Cost
        state_residual = crocoddyl.ResidualModelState(state, xref=x_target, nu=nu)
        state_residual_activation = crocoddyl.ActivationModelWeightedQuad(Q)
        state_cost = crocoddyl.CostModelResidual(state, state_residual_activation, state_residual)

        # Control Cost
        control_residual = crocoddyl.ResidualModelControl(state, nu=nu)
        control_residual_activation = crocoddyl.ActivationModelWeightedQuad(R)
        control_cost = crocoddyl.CostModelResidual(
            state, control_residual_activation, control_residual
        )

        # Control rate cost
        self.control_rate_residual = crocoddyl.ResidualModelControl(state, uref=np.array([0.0]))
        control_rate_residual_activation = crocoddyl.ActivationModelWeightedQuad(S)
        control_rate_cost = crocoddyl.CostModelResidual(
            state, control_rate_residual_activation, self.control_rate_residual
        )

        self.running_models = []
        for k in range(self.N):
            running_cost = crocoddyl.CostModelSum(state, nu=nu)
            running_cost.addCost("state_cost", cost=state_cost, weight=1.0)
            running_cost.addCost("control_cost", cost=control_cost, weight=1.0)
            running_cost.addCost("control_rate_cost", cost=control_rate_cost, weight=np.exp(M - k))

            running_model = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, running_cost),
                dt,
            )
            self.running_models.append(running_model)

        # Terminal cost
        terminal_cost = crocoddyl.CostModelSum(state, nu=nu)
        terminal_cost.addCost("state_cost", cost=state_cost, weight=self.N)

        self.terminal_model = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminal_cost),
            0.0,
        )

    def create_problem(self, state: np.ndarray):
        self.problem = crocoddyl.ShootingProblem(state, self.running_models, self.terminal_model)

    def init_solver(self):
        self.solver = crocoddyl.SolverFDDP(self.problem)
        callbacks = []
        callbacks.append(crocoddyl.CallbackVerbose())
        self.solver.setCallbacks(callbacks)

    def compute_command(self, state: np.ndarray, max_iter: int = 500, x_ws=[], u_ws=[]) -> float:
        self.create_problem(state)
        self.init_solver()
        self.solver.solve(x_ws, u_ws, max_iter, False, 1e-5)
        u = np.clip(self.solver.us[0][0], -self.u_lim, self.u_lim)
        return u

    def get_trajectoy(self) -> np.ndarray:
        return self.solver.xs.tolist()

    def get_command(self) -> np.ndarray:
        return self.solver.us.tolist()

    def get_warm_start(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        x_ws = self.solver.xs.tolist()[1:] + [self.solver.xs[-1]]
        u_ws = self.solver.us.tolist()[1:] + [self.solver.us[-1]]
        return x_ws, u_ws
