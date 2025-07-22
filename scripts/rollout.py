import furuta

from time import strftime
from pathlib import Path

import numpy as np

from furuta.controls.filters import VelocityFilter
from furuta.logger import Loader, SimpleLogger
from furuta.plotter import Plotter
from furuta.robot import RobotModel
from furuta.sim import SimulatedRobot
from furuta.state import Signal, State
from furuta.viewer import Viewer3D

if __name__ == "__main__":
    # Time constants
    t_final = 3.0
    control_freq = 100  # Hz
    dt = 1 / control_freq

    init_state = np.array([0.0, 3.14, 0.0, 0.0])
    times = np.arange(0.0, t_final, dt)

    # Robot
    robot = RobotModel().robot

    # Create the simulation
    sim = SimulatedRobot(robot, init_state, dt=1e-5)

    # Low pass velocity filter
    motor_velocity_filter = VelocityFilter(2, 20.0, control_frequency=1 / dt, init_vel=0.0)
    pendulum_velocity_filter = VelocityFilter(2, 49.0, control_frequency=1 / dt, init_vel=0.0)

    # Create the logger
    file_name = f"{strftime('%Y%m%d-%H%M%S')}.mcap"
    log_dir = Path(furuta.__path__[0]).parent / "logs" / "rollout"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / file_name
    logger = SimpleLogger(log_path)

    # Set the initial state
    state = State(
        motor_position=Signal(measured=init_state[0], simulated=init_state[0]),
        pendulum_position=Signal(measured=init_state[1], simulated=init_state[1]),
        motor_velocity=Signal(
            measured=init_state[2], filtered=init_state[2], simulated=init_state[2]
        ),
        pendulum_velocity=Signal(
            measured=init_state[3], filtered=init_state[3], simulated=init_state[3]
        ),
        action=0.0,
    )
    logger.update(int(times[0] * 1e9), state)

    # Rollout
    u = 0.0
    for i, t in enumerate(times[1:]):
        motor_position, pendulum_position = sim.step(u, dt)

        # Compute velocity via finite differences
        measured_motor_velocity = (motor_position - state.motor_position.measured) / dt
        measured_pendulum_velocity = (pendulum_position - state.pendulum_position.measured) / dt

        # Filter velocity
        motor_velocity = motor_velocity_filter(measured_motor_velocity)
        pendulum_velocity = pendulum_velocity_filter(measured_pendulum_velocity)

        state = State(
            motor_position=Signal(measured=motor_position, simulated=sim.state[0]),
            pendulum_position=Signal(measured=pendulum_position, simulated=sim.state[1]),
            motor_velocity=Signal(
                measured=measured_motor_velocity, filtered=motor_velocity, simulated=sim.state[2]
            ),
            pendulum_velocity=Signal(
                measured=measured_pendulum_velocity,
                filtered=pendulum_velocity,
                simulated=sim.state[3],
            ),
            action=0.0,
        )
        logger.update(int(t * 1e9), state)

    # Close logger
    logger.stop()

    # Read log
    loader = Loader()
    times, states_dict = loader.load(log_path)

    # Plot
    plotter = Plotter(times, states_dict)
    plotter.plot()

    # Animate
    robot_viewer = Viewer3D(robot)
    states = loader.get_state("simulated")
    robot_viewer.animate(times, states)
    robot_viewer.close()
