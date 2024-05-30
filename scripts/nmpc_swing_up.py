import time
from dataclasses import dataclass

import crocoddyl
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from panda3d_viewer import Viewer
from pinocchio.visualize.panda3d_visualizer import Panda3dVisualizer
from sim import Integrator, Log, Logger, RobotData, RobotViewer, SimulatedRobot

from furuta.robot import PendulumDynamics

ROOT_DIR = "/home/pierfabre/pendulum_workspace/src/furuta/"
STATE = ["phi", "theta", "phi_dot", "theta_dot"]


# Define the control signal to actuated joint mapping
class FurutaActuationModel(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        nu = 1  # Control dimension
        crocoddyl.ActuationModelAbstract.__init__(self, state, nu=nu)

    def calc(self, data, x, u):
        assert len(data.tau) == 2
        # Map the control dimensions to the joint torque
        data.tau[0] = u
        data.tau[1] = 0

    def calcDiff(self, data, x, u):
        # Specify the actuation jacobian
        data.dtau_du[0] = 1
        data.dtau_du[1] = 0


class SwingUpController:
    def __init__(self, robot, t_final, control_freq):
        self.robot = robot
        dt = 1 / control_freq  # Time step
        self.T = int(t_final * control_freq)
        state = crocoddyl.StateMultibody(robot.model)
        # Actuation model
        actuationModel = FurutaActuationModel(state)
        # Cost models
        runningCostModel = crocoddyl.CostModelSum(state, nu=actuationModel.nu)
        terminalCostModel = crocoddyl.CostModelSum(state, nu=actuationModel.nu)
        # Add a cost for the configuration positions and velocities
        xref = np.array([0, 0, 0, 0])  # Desired state
        stateResidual = crocoddyl.ResidualModelState(state, xref=xref, nu=actuationModel.nu)
        stateCostModel = crocoddyl.CostModelResidual(state, stateResidual)
        runningCostModel.addCost("state_cost", cost=stateCostModel, weight=0.1)
        terminalCostModel.addCost("state_cost", cost=stateCostModel, weight=1000)
        # Add a cost on control
        controlResidual = crocoddyl.ResidualModelControl(state, nu=actuationModel.nu)
        bounds = crocoddyl.ActivationBounds(np.array([-100.0]), np.array([100.0]))
        activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
        controlCost = crocoddyl.CostModelResidual(
            state, activation=activation, residual=controlResidual
        )
        # controlCost = crocoddyl.CostModelResidual(state, residual=controlResidual)
        runningCostModel.addCost("control_cost", cost=controlCost, weight=1e5 / dt)
        # Create the action models for the state
        self.runningModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                state, actuationModel, runningCostModel
            ),
            dt,
        )
        self.terminalModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                state, actuationModel, terminalCostModel
            ),
            0.0,
        )
        # Define a shooting problem
        q0 = np.zeros((state.nq,))  # Inital joint configurations
        q0[1] = np.pi  # Down
        v0 = np.zeros((state.nv,))  # Initial joint velocities
        x0 = np.concatenate((q0, v0))  # Inital robot state

        self.problem = crocoddyl.ShootingProblem(
            x0, [self.runningModel] * self.T, self.terminalModel
        )

    def update_problem(self, state):
        self.problem = crocoddyl.ShootingProblem(
            state, [self.runningModel] * self.T, self.terminalModel
        )

    def compute_command(self, state, max_iter, x_ws=[], u_ws=[], callback=False, solver="FDDP"):
        self.update_problem(state)
        if solver == "FDDP":
            self.solver = crocoddyl.SolverFDDP(self.problem)
        elif solver == "BoxDDP":
            self.solver = crocoddyl.SolverBoxDDP(self.problem)
        if callback:
            callbacks = []
            callbacks.append(crocoddyl.CallbackVerbose())
            self.solver.setCallbacks(callbacks)
        self.solver.solve(x_ws, u_ws, max_iter, False, 1e-5)
        return self.solver.us[0][0]

    def get_warm_start(self):
        xs = self.solver.xs
        us = self.solver.us
        xs[:-1] = xs[1:]
        xs[-1] = self.solver.xs[-1]
        us[:-1] = us[1:]
        us[-1] = self.solver.us[-1]
        return xs, us


if __name__ == "__main__":

    robot = pin.RobotWrapper.BuildFromURDF(
        ROOT_DIR + "robot/hardware/furuta.urdf",
        package_dirs=[ROOT_DIR + "robot/hardware/CAD/stl/"],
    )

    t_final = 1.0
    control_freq = 100  # Hz
    time_step = 1 / control_freq
    dt = time_step  # Time step

    controller = SwingUpController(robot, t_final, control_freq)
    init_state = np.array([0.0, np.pi, 0.0, 0.0])
    controller.compute_command(state=init_state, max_iter=500, callback=True)

    sim = SimulatedRobot()
    data = RobotData()
    sim.state = init_state
    state = init_state
    times = np.arange(0, t_final, time_step)
    logger = Logger()
    elapsed_time = []
    for t in times:
        data.time = t
        data.state = state.tolist()
        logger.update_log(data)
        tic = time.time()
        x_ws, u_ws = controller.get_warm_start()
        u = controller.compute_command(state, 3, x_ws, u_ws, solver="BoxDDP")
        toc = time.time()
        elapsed_time.append(toc - tic)
        state = sim.step(u, time_step)
    plt.show()
    plt.figure(10)
    plt.plot(times, elapsed_time)
    logger.save_log()
    log = logger.log
    robot_viewer = RobotViewer(sim.robot)
    robot_viewer.animate(log)
    log.plot()
