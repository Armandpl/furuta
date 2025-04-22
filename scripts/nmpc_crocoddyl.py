"""
Hints:

residual = crocoddyl.ResidualModelState(state, xref: np.array, nu)
residual = crocoddyl.ResidualModelControl(state, uref: np.array)

activation = crocoddyl.ActivationModelWeightedQuad(weights: np.ndarray)

cost = crocoddyl.CostModelResidual(state, residual)

costs = crocoddyl.CostModelSum(state, nu)

cost.addCost("cost", cost, weight)

constraint = crocoddyl.ConstraintModelResidual(state, residual, lb: np.array, ub:np.array)

constraints = crocoddyl.ConstraintModelManager(state, nu)

constraints.addConstraint("constraint", constraint)

dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, costs, constraints)

iam = crocoddyl.IntegratedActionModelEuler(dam, dt)
iam = crocoddyl.IntegratedActionModelRk(dam, crocoddyl.RKType.four, dt)

problem = crocoddyl.ShootingProblem(x0: np.ndarray, iams: tp.List[iam], terminal_iam (dt=0.0))

solver = mim_solvers.SolverCSQP(problem) <- with constraints
solver = mim_solvers.SolverSQP(problem)
solver = crocoddyl.SolverFDDP(problem)

solver.solve(x_ws=[], u_ws=[], max_iter)

xs = solver.xs.tolist()
us = solver.us.tolist()

Useful links:
https://github.com/loco-3d/crocoddyl
https://github.com/loco-3d/crocoddyl/blob/master/examples/notebooks/acrobot_urdf.ipynb
"""

import time

import crocoddyl
import mim_solvers
import numpy as np

from furuta.controls.filters import VelocityFilter
from furuta.logger import Loader, SimpleLogger
from furuta.plotter import Plotter
from furuta.robot import RobotModel
from furuta.sim import SimulatedRobot
from furuta.state import Signal, State
from furuta.viewer import Viewer3D

#################################
# PART 1 : Optimal Control
#################################

# Parameters
dt = None  # TODO

# Simulation setup
x0 = np.array([0.0, 0.0, 0.0, 0.0])
sim_time = 2  # Total simulation time in seconds

# Robot
robot = RobotModel().robot

# State
nu = 1
state = crocoddyl.StateMultibody(robot.model)


# Actuation model
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


actuation = FurutaActuationModel(state, nu)

# State Cost
# TODO

# Control Cost
# TODO

# Control rate cost (optional)
# TODO

# Control Constraints (optional)
# TODO

# State Constraints (optional)
# TODO

# Running cost
# TODO

# Terminal cost
# TODO

# DAMs
# TODO

# IAMs
# TODO

# Problem
problem = None  # TODO

# Solver
solver = None  # TODO

# Solve offline once without warm start
# TODO

# Plot solution
times = np.arange(len(solver.us)) * dt
x_sol = np.array(solver.xs)
states_dict = {
    "motor_position.desired": x_sol[1:, 0],
    "pendulum_position.desired": x_sol[1:, 1],
    "motor_velocity.desired": x_sol[1:, 2],
    "pendulum_velocity.desired": x_sol[1:, 3],
    "action": np.array(solver.us),
}
plotter = Plotter(times, states_dict)
plotter.plot()

# Visualize
viewer = Viewer3D(robot)
viewer.animate(times, x_sol)

#################################
# PART 2 : NMPC
#################################

times = np.arange(0.0, sim_time, dt)

# Create the data logger
fname = f"{time.strftime('%Y%m%d-%H%M%S')}.mcap"
log_path = "logs/" + fname
logger = SimpleLogger(log_path)

# Create the simulated robot
sim = SimulatedRobot(robot, x0, dt=1e-5)

# Warm start
x_ws = solver.xs.tolist()
u_ws = solver.us.tolist()

# Initial state
u = 0.0
motor_position = x0[0]
pendulum_position = x0[1]
measured_motor_velocity = x0[2]
measured_pendulum_velocity = x0[3]
state = State(
    motor_position=Signal(
        measured=motor_position,
        simulated=sim.state[0],
    ),
    pendulum_position=Signal(
        measured=pendulum_position,
        simulated=sim.state[1],
    ),
    motor_velocity=Signal(
        measured=measured_motor_velocity,
        simulated=sim.state[2],
    ),
    pendulum_velocity=Signal(
        measured=measured_pendulum_velocity,
        simulated=sim.state[3],
    ),
    action=u,
)
logger.update(int(0.0 * 1e9), state)

# Control loop
for t in times[1:]:
    x = np.array(
        [motor_position, pendulum_position, measured_motor_velocity, measured_pendulum_velocity]
    )
    # Problem
    problem = None  # TODO

    # Solver
    solver = None  # TODO

    start_solve = time.time()

    # Solve
    # TODO

    compute_time = time.time() - start_solve

    # Get solution
    xs = None  # TODO
    us = None  # TODO

    # Clamp command to bounds (optional)
    # TODO

    # Update refs in residuals if needed (optional)
    # Hint: residual.uref = ...
    # TODO

    # Send action to robot
    u = None  # TODO
    motor_position, pendulum_position = sim.step(u, dt)

    # Compute velocity via finite differences
    measured_motor_velocity = None  # TODO
    measured_pendulum_velocity = None  # TODO

    # Log data
    state = State(
        motor_position=Signal(
            measured=motor_position,
            simulated=sim.state[0],
        ),
        pendulum_position=Signal(
            measured=pendulum_position,
            simulated=sim.state[1],
        ),
        motor_velocity=Signal(
            measured=measured_motor_velocity,
            simulated=sim.state[2],
        ),
        pendulum_velocity=Signal(
            measured=measured_pendulum_velocity,
            simulated=sim.state[3],
        ),
        action=u,
        timing=compute_time,
    )
    logger.update(int(t * 1e9), state)

    # Warm Start (optional)
    # Hint: you may need to modify the solution...
    x_ws = None  # TODO
    u_ws = None  # TODO

# Close logger
logger.stop()

# Read log
loader = Loader()
times, states_dict = loader.load(log_path)

# Plot
plotter = Plotter(times, states_dict)
plotter.plot()

# Animate
states = loader.get_state("simulated")
viewer.animate(times, states)
viewer.close()
