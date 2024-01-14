import logging
import time
from pathlib import Path
from typing import Union

import gym
import numpy as np
import wandb
from gym.spaces import Box
from mcap_protobuf.writer import Writer

from furuta.logging.protobuf.pendulum_state_pb2 import PendulumState


class GentlyTerminating(gym.Wrapper):
    """This env wrapper sends zero command to the robot when an episode is done."""

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            logging.debug("episode done, killing motor.")
            self.env.motor.set_speed(0)
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()


class MCAPLogger(gym.Wrapper):
    def __init__(self, env: gym.Env, log_dir: Union[str, Path], use_sim_time: bool):
        super().__init__(env)
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        self.log_dir = log_dir
        self.use_sim_time = use_sim_time

        self.episodes = 0
        self.mcap_writer = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.use_sim_time:
            self.sim_time += self.unwrapped.timing.dt

        if self.use_sim_time:
            time_to_log = self.sim_time
        else:
            time_to_log = time.time_ns()  # TODO check that's the right clock

        # convert to milliseconds
        time_to_log = round(time_to_log * 1e9)

        self.mcap_writer.write_message(
            topic="/pendulum_state",
            message=PendulumState(**info),
            log_time=time_to_log,
            publish_time=time_to_log,
        )

        return observation, reward, done, info

    def reset(self):
        # create log dir if doesn't exist
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        # close previous log file
        self.close_mcap_writer()

        # instantiate a new MCAP writer
        fname = f"ep{self.episodes}_{time.strftime('%Y%m%d-%H%M%S')}.mcap"
        self.output_file = open(self.log_dir / fname, "wb")
        self.mcap_writer = Writer(self.output_file)

        # TODO add metadata
        # date, control frequency, wandb run id, sim parameters, robot parameters, etc.

        self.episodes += 1

        # reset sim time
        self.sim_time = 0
        return self.env.reset()

    def close(self):
        self.close_mcap_writer()
        return self.env.close()

    def close_mcap_writer(self):
        if self.mcap_writer is not None:
            self.mcap_writer.finish()
            self.output_file.close()


class ControlFrequency(gym.Wrapper):
    """Enforce a sleeping time (dt) between each step."""

    def __init__(self, env, dt):
        super().__init__(env)
        self.dt = dt

    def step(self, action):
        current = time.time()
        _, _, _, _ = self.env.step(action)
        loop_time = 0
        if self.last is not None:
            loop_time = current - self.last
            sleeping_time = self.dt - loop_time

            if sleeping_time > 0:
                time.sleep(sleeping_time)
            else:
                print("warning, loop time > dt")

        obs, reward, done, info = self.env.step(np.array([0.0]))
        self.last = time.time()

        if logging.root.level == logging.DEBUG:
            wandb.log({**info, **{"loop time": loop_time}})

        return obs, reward, done, info

    def reset(self):
        self.last = None
        return self.env.reset()


class HistoryWrapper(gym.Wrapper):
    """Track history of observations for given amount of steps Initial steps are zero-filled."""

    def __init__(self, env: gym.Env, steps: int, use_continuity_cost: bool):
        super().__init__(env)
        assert steps > 1, "steps must be > 1"
        self.steps = steps
        self.use_continuity_cost = use_continuity_cost

        # concat obs with action
        self.step_low = np.concatenate([self.observation_space.low, self.action_space.low])
        self.step_high = np.concatenate([self.observation_space.high, self.action_space.high])

        # stack for each step
        obs_low = np.tile(self.step_low, (self.steps, 1))
        obs_high = np.tile(self.step_high, (self.steps, 1))

        self.observation_space = Box(low=obs_low, high=obs_high)

        self.history = self._make_history()

    def _make_history(self):
        return [np.zeros_like(self.step_low) for _ in range(self.steps)]

    def _continuity_cost(self, obs):
        # TODO compute continuity cost for all steps and average?
        # and compare smoothness between training run, and viz smoothness over time
        action = obs[-1][-1]
        last_action = obs[-2][-1]
        continuity_cost = np.power((action - last_action), 2).sum()

        return continuity_cost

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.history.pop(0)

        obs = np.concatenate([obs, action])
        self.history.append(obs)
        obs = np.array(self.history)

        if self.use_continuity_cost:
            continuity_cost = self._continuity_cost(obs)
            reward -= continuity_cost
            info["continuity_cost"] = continuity_cost

        return obs, reward, done, info

    def reset(self):
        self.history = self._make_history()
        self.history.pop(0)
        obs = np.concatenate([self.env.reset(), np.zeros_like(self.env.action_space.low)])
        self.history.append(obs)
        return np.array(self.history)
