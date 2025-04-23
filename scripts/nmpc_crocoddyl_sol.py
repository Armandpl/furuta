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

# Parameters
dt = 0.01
N = 100
M = 10

# Simulation setup
x0 = np.array([0.0, 0.0, 0.0, 0.0])
x_target = np.array([0.0, np.pi, 0.0, 0.0])
sim_time = 2  # Total simulation time in seconds

# Cost weights
Q = np.array([10, 50, 1, 1])
R = 0.1
r = 4

# Control bounds
u_lim = 0.1
u_ref = 0.0

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
state_residual = crocoddyl.ResidualModelState(state, xref=x_target, nu=nu)
state_residual_activation = crocoddyl.ActivationModelWeightedQuad(Q)
state_cost = crocoddyl.CostModelResidual(state, state_residual_activation, state_residual)

# Control Cost
control_residual = crocoddyl.ResidualModelControl(state, nu=nu)
control_cost = crocoddyl.CostModelResidual(state, residual=control_residual)

# Control rate cost
control_rate_residual = crocoddyl.ResidualModelControl(state, uref=np.array([0.0]))
control_rate_cost = crocoddyl.CostModelResidual(state, residual=control_rate_residual)

# Constraints
constraint_manager = crocoddyl.ConstraintModelManager(state, nu=nu)
control_constraint = crocoddyl.ConstraintModelResidual(
    state, control_residual, np.array([-u_lim, 0]), np.array([u_lim, 0])
)
constraint_manager.addConstraint("control_constraint", control_constraint)

# Running cost
running_cost = crocoddyl.CostModelSum(state, nu=nu)
running_cost.addCost("state_cost", cost=state_cost, weight=1.0)
running_cost.addCost("control_cost", cost=control_cost, weight=R)

# Terminal cost
terminal_cost = crocoddyl.CostModelSum(state, nu=nu)
terminal_cost.addCost("state_cost", cost=state_cost, weight=1.0)

starting_models = []
for k in range(M):
    # Starting cost
    starting_cost = crocoddyl.CostModelSum(state, nu=nu)
    starting_cost.addCost("state_cost", cost=state_cost, weight=1.0)
    starting_cost.addCost("control_cost", cost=control_cost, weight=R)
    starting_cost.addCost("control_rate_cost", cost=control_rate_cost, weight=10 ** (r - k))

    # IAM
    starting_model = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, starting_cost, constraint_manager
        ),
        dt,
    )

    starting_models.append(starting_model)

running_model = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, running_cost, constraint_manager
    ),
    dt,
)
terminal_model = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, terminal_cost, constraint_manager
    ),
    0.0,
)

# Problem
problem = crocoddyl.ShootingProblem(
    x0, starting_models + [running_model] * (N - M), terminal_model
)

# Solver (FDDP, SQP, CSQP)
solver = mim_solvers.SolverCSQP(problem)

# Solve offline once
solver.solve([], [], 500, False, 1e-5)

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

times = np.arange(0.0, sim_time, dt)

# Create the data logger
fname = f"{time.strftime('%Y%m%d-%H%M%S')}.mcap"
log_path = "logs/" + fname
logger = SimpleLogger(log_path)

# Create the simulated robot
sim = SimulatedRobot(robot, x0, dt=1e-5)

# Low pass velocity filter
motor_velocity_filter = VelocityFilter(2, 20.0, control_frequency=1 / dt, init_vel=x0[2])
pendulum_velocity_filter = VelocityFilter(2, 20.0, control_frequency=1 / dt, init_vel=x0[3])

# Warm start
x_ws = solver.xs.tolist()
u_ws = solver.us.tolist()

# Initial state
u = 0.0
motor_position, pendulum_position = sim.step(u, dt)
# Compute velocity via finite differences
measured_motor_velocity = (motor_position - x0[0]) / dt
measured_pendulum_velocity = (pendulum_position - x0[1]) / dt
# Filter velocity
filtered_motor_velocity = motor_velocity_filter(measured_motor_velocity)
filtered_pendulum_velocity = pendulum_velocity_filter(measured_pendulum_velocity)
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
        filtered=filtered_motor_velocity,
    ),
    pendulum_velocity=Signal(
        measured=measured_pendulum_velocity,
        simulated=sim.state[3],
        filtered=filtered_pendulum_velocity,
    ),
    action=u,
)
logger.update(int(0.0 * 1e9), state)

# Run the simulation
for t in times[1:]:
    # Problem
    x = np.array(
        [motor_position, pendulum_position, measured_motor_velocity, measured_pendulum_velocity]
    )
    problem = crocoddyl.ShootingProblem(
        x, starting_models + [running_model] * (N - M), terminal_model
    )

    # Solver (FDDP, SQP, CSQP)
    # solver = crocoddyl.SolverFDDP(problem)
    solver = mim_solvers.SolverSQP(problem)

    start_solve = time.time()

    # Solve
    solver.solve(x_ws, u_ws, 1, False, 1e-5)

    compute_time = time.time() - start_solve

    # Clamp command
    u = np.clip(solver.us[0][0], -u_lim, u_lim)
    control_rate_residual.uref = u

    # Simulate the robot
    motor_position, pendulum_position = sim.step(u, dt)

    # Compute velocity via finite differences
    measured_motor_velocity = (motor_position - state.motor_position.measured) / dt
    measured_pendulum_velocity = (pendulum_position - state.pendulum_position.measured) / dt

    # Filter velocity
    filtered_motor_velocity = motor_velocity_filter(measured_motor_velocity)
    filtered_pendulum_velocity = pendulum_velocity_filter(measured_pendulum_velocity)

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
            filtered=filtered_motor_velocity,
        ),
        pendulum_velocity=Signal(
            measured=measured_pendulum_velocity,
            simulated=sim.state[3],
            filtered=filtered_pendulum_velocity,
        ),
        action=u,
        timing=compute_time,
    )
    logger.update(int(t * 1e9), state)

    # Get the warm start by shifting the solution
    x_ws = solver.xs.tolist()[1:] + [solver.xs[-1]]
    u_ws = solver.us.tolist()[1:] + [solver.us[-1]]

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
