from abc import ABC, abstractmethod
from typing import Dict

import crocoddyl
import mim_solvers
import numpy as np
from simple_pid import PID

ROOT_DIR = "/home/pierfabre/pendulum_workspace/src/furuta/"


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
                setpoint=np.deg2rad(parameters["setpoint"]),
                sample_time=sample_time,
            )
        except KeyError:
            raise ValueError("Invalid PID parameters")

    def compute_command(self, position: float):
        return self.pid(position)


class SwingUpController:
    class FurutaActuationModel(crocoddyl.ActuationModelAbstract):
        # TODO : Implement in c++ if too slow
        def __init__(self, state):
            nu = 1  # Control dimension
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

    def __init__(self, robot, parameters: Dict, xref: np.ndarray):

        # Read parameters
        control_freq = parameters["control_frequency"]
        t_final = parameters["t_final"]
        self.u_lim = parameters["u_lim"]
        constraints_type = parameters["constraints_type"]
        self.solver_type = parameters["solver_type"]

        # Time variables
        dt = 1 / control_freq  # Time step
        self.T = int(t_final * control_freq)

        # Instantiate robot as a pinocchio RobotWrapper
        self.robot = robot

        # State
        state = crocoddyl.StateMultibody(robot.model)

        # Actuation model
        actuationModel = self.FurutaActuationModel(state)

        # State Residual
        stateResidual = crocoddyl.ResidualModelState(state, xref=xref, nu=actuationModel.nu)
        stateCostModel = crocoddyl.CostModelResidual(state, stateResidual)

        # Control Residual
        controlResidual = crocoddyl.ResidualModelControl(state, nu=actuationModel.nu)
        if constraints_type == "SOFT":
            bounds = crocoddyl.ActivationBounds(np.array([self.u_lim]), np.array([self.u_lim]))
            activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
            controlCost = crocoddyl.CostModelResidual(
                state, residual=controlResidual, activation=activation
            )
        else:
            controlCost = crocoddyl.CostModelResidual(state, residual=controlResidual)

        # Constraints
        constraintManager = crocoddyl.ConstraintModelManager(state, nu=actuationModel.nu)
        if constraints_type == "HARD":
            constraint = crocoddyl.ConstraintModelResidual(
                state,
                controlResidual,
                np.array([-self.u_lim, -self.u_lim]),
                np.array([self.u_lim, self.u_lim]),
            )
            constraintManager.addConstraint("control_constraint", constraint)

        # Running cost
        runningCostModel = crocoddyl.CostModelSum(state, nu=actuationModel.nu)
        runningCostModel.addCost("state_cost", cost=stateCostModel, weight=1)
        runningCostModel.addCost("control_cost", cost=controlCost, weight=0.1)

        # Terminal cost
        terminalCostModel = crocoddyl.CostModelSum(state, nu=actuationModel.nu)
        terminalCostModel.addCost("state_cost", cost=stateCostModel, weight=10 * self.T)

        # IAM
        self.runningModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                state, actuationModel, runningCostModel, constraintManager
            ),
            dt,
        )
        self.terminalModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                state, actuationModel, terminalCostModel
            ),
            0.0,
        )

        # Initial state
        q0 = np.zeros((state.nq + state.nv,))
        q0[1] = np.pi  # Down

        # Create the shooting problem
        self.create_problem(q0)

    def create_problem(self, state):
        self.problem = crocoddyl.ShootingProblem(
            state, [self.runningModel] * self.T, self.terminalModel
        )

    def compute_command(self, state, max_iter, x_ws=[], u_ws=[], callback=False):
        self.create_problem(state)
        if self.solver_type == "CSQP":
            self.solver = mim_solvers.SolverCSQP(self.problem)
        elif self.solver_type == "FDDP":
            self.solver = crocoddyl.SolverFDDP(self.problem)
        else:
            raise ValueError(f"Invalid solver type: {self.solver_type}")
        if callback:
            callbacks = []
            callbacks.append(crocoddyl.CallbackVerbose())
            self.solver.setCallbacks(callbacks)
        self.solver.solve(x_ws, u_ws, max_iter, False, 1e-5)
        # Clamp the control signal
        u = np.clip(self.solver.us[0][0], -self.u_lim, self.u_lim)
        return u

    def get_warm_start(self):
        xs = self.solver.xs
        us = self.solver.us
        xs[:-1] = xs[1:]
        xs[-1] = self.solver.xs[-1]
        us[:-1] = us[1:]
        us[-1] = self.solver.us[-1]
        return xs, us
