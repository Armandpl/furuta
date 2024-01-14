import argparse
import logging
import os
from collections import namedtuple
from distutils.util import strtobool

import gym
import wandb
from gym.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

import furuta  # noqa F420
from furuta.envs.wrappers import ControlFrequency, GentlyTerminating, HistoryWrapper


class Robot:
    def __init__(self, args):
        self.args = args
        self.load_model()
        self.create_env()

    def load_model(self):
        print(f"loading model from Artifacts, version {args.model_artifact}")
        artifact = wandb.use_artifact(f"sac_model:{args.model_artifact}")
        artifact_dir = artifact.download()
        self.model = SAC.load(os.path.join(artifact_dir, "sac.zip"))
        self.model_producer_run = artifact.logged_by()

    def create_env(self):
        # update producer run config with CLI args if they are not None
        for attr, value in self.args.items():
            if value:
                self.model_producer_run.config[attr] = value

        # convert dict into object
        config = dict2obj(self.model_producer_run.config)
        wandb.summary["env_config"] = config
        self.env = setup_env(config)

    def run_episode(self):
        print("Episode starts")
        total_reward = 0
        obs = self.env.reset()
        while True:
            action, _states = self.model.predict(obs, deterministic=self.args.deterministic)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                obs = self.env.reset()
                break
        return total_reward


def dict2obj(d):
    return namedtuple("d", d.keys())(*d.values())


def main(args):
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        sync_tensorboard=True,
        save_code=True,
        job_type="inference",
    )

    robot = Robot(run.config)

    try:
        while True:
            rwd = robot.run_episode()
            print(f"Episode Reward: {rwd}")
            input("Press enter to run episode\n")
    except KeyboardInterrupt:
        robot.env.close()
        run.finish()


def setup_env(args):
    # base env
    env = gym.make(
        args.gym_id,
        fs=args.fs,
        fs_ctrl=args.fs_ctrl,
        action_limiter=args.action_limiter,
        safety_th_lim=args.safety_th_lim,
        state_limits=args.state_limits,
    )

    wandb.run.summary["state_max"] = env.state_max

    if args.episode_length != -1:
        env = TimeLimit(env, args.episode_length)

    if args.history > 1:
        env = HistoryWrapper(env, args.history, args.continuity_cost)

    env = Monitor(env)

    # if robot
    if args.gym_id == "FurutaReal-v0":
        env = GentlyTerminating(env)
        env = ControlFrequency(env, env.timing.dt_ctrl)

    return env


def parse_args():
    parser = argparse.ArgumentParser(description="TD3 agent")
    # Common arguments
    parser.add_argument(
        "model_artifact", type=str, help="the artifact version of the model to load"
    )

    parser.add_argument(
        "--deterministic",
        default=True,
        type=lambda x: bool(strtobool(x)),
        help="Whether to use a deterministic policy or not",
    )
    parser.add_argument(
        "--gym_id", type=str, default="FurutaReal-v0", help="the id of the gym environment"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="furuta", help="the wandb's project name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="the entity (team) of wandb's project"
    )
    parser.add_argument("-d", "--debug", action="store_true")

    # env params
    parser.add_argument("--fs", type=int, help="Sampling frequency")
    parser.add_argument("--fs_ctrl", type=int, help="control frequency")
    parser.add_argument(
        "--episode_length",
        type=int,
        help="the maximum length of each episode. \
                        -1 = infinite",
    )
    parser.add_argument("--safety_th_lim", type=float, help="Max motor (theta) angle in rad.")
    parser.add_argument(
        "--action_limiter", type=lambda x: bool(strtobool(x)), help="Restrict actions"
    )
    parser.add_argument(
        "--state_limits", type=str, help="Wether to use high or low limits. See code."
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging_level)
    main(args)
