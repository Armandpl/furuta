import argparse
import configparser
from distutils.util import strtobool
import logging
from pathlib import Path
import os

import gym
from gym.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb

import furuta_gym  # noqa F420
from furuta_gym.envs.wrappers import GentlyTerminating, \
                                     HistoryWrapper, \
                                     ControlFrequency, \
                                     MCAPWriter


def main(args):
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=args,
        sync_tensorboard=True,
        monitor_gym=args.capture_video,
        save_code=True
    )
    args = run.config

    env = setup_env(args)

    verbose = 2 if args.debug else 0

    model = SAC(
                "MlpPolicy", env, verbose=verbose,
                learning_rate=args.learning_rate, seed=args.seed,
                buffer_size=args.buffer_size, tau=args.tau,
                gamma=args.gamma, batch_size=args.batch_size,
                target_update_interval=args.target_update_interval,
                learning_starts=args.learning_starts,
                use_sde=args.use_sde, use_sde_at_warmup=args.use_sde_at_warmup,
                sde_sample_freq=args.sde_sample_freq,
                train_freq=(args.train_freq, args.train_freq_unit),
                gradient_steps=args.gradient_steps,
                tensorboard_log=f"runs/{run.id}",
            )

    if args.model_artifact:
        model.load(download_artifact_file(f"sac_model:{args.model_artifact}",
                                          "sac.zip"))

    if args.rb_artifact:
        rb_path = download_artifact_file(
                    f"sac_replay_buffer:{args.model_artifact}",
                    "buffer.pkl")
        model.load_replay_buffer(rb_path)

    try:
        logging.info("Starting to train")
        model.learn(total_timesteps=args.total_timesteps)
    except KeyboardInterrupt:
        logging.info("Interupting training")

    model_path = f"runs/{run.id}/models/sac.zip"
    model.save(model_path)
    upload_file_to_artifacts(model_path, "sac_model", "model")

    buffer_path = f"runs/{run.id}/buffers/buffer.pkl"
    model.save_replay_buffer(buffer_path)
    upload_file_to_artifacts(buffer_path, "sac_replay_buffer", "replay buffer")

    env.close()
    run.finish()


def setup_env(args):
    # base env
    env = gym.make(args.gym_id, fs=args.fs, fs_ctrl=args.fs_ctrl,
                   action_limiter=args.action_limiter,
                   safety_th_lim=args.safety_th_lim,
                   state_limits=args.state_limits)

    wandb.run.summary["state_max"] = env.state_max

    # load custom sim params
    if args.gym_id == "FurutaSim-v0" and args.custom_sim:
        load_sim_params(env, args.custom_sim)
        logging.info(f"Loaded sim params: \n{env.dyn.params}")
        wandb.run.summary["sim_params"] = env.dyn.params

    if args.log_mcap:
        env = MCAPWriter(env, f"./data/{wandb.run.id}")

    if args.episode_length != -1:
        env = TimeLimit(env, args.episode_length)

    if args.history > 1:
        env = HistoryWrapper(env, args.history, args.continuity_cost)

    env = Monitor(env)

    if args.gym_id == "FurutaReal-v0":  # if robot
        env = GentlyTerminating(env)
        env = ControlFrequency(env, env.timing.dt_ctrl)

    if args.capture_video:
        # TODO add headleas arg, depends on the machine
        # import pyvirtualdisplay
        # pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
        env = DummyVecEnv([lambda: env])
        env = VecVideoRecorder(env, f"videos/{wandb.run.id}",
                               record_video_trigger=lambda x: x % 30000 == 0,
                               video_length=300)

    return env


def download_artifact_file(artifact_alias, filename):
    """
    Download artifact and returns path to filename.

    :param artifact_name: wandb artifact alias
    :param filename: filename in the artifact
    """
    logging.info(f"loading {filename} from {artifact_alias}")

    artifact = wandb.use_artifact(artifact_alias)
    artifact_dir = Path(artifact.download())
    filepath = artifact_dir / filename

    assert filepath.is_file(), f"{artifact_alias} doesn't contain {filename}"

    return filepath


def upload_file_to_artifacts(pth, artifact_name, artifact_type):
    logging.info(f"Saving {pth} to {artifact_name}")
    if not isinstance(pth, Path):
        pth = Path(pth)

    assert os.path.isfile(pth), f"{pth} is not a file"

    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(pth)
    wandb.log_artifact(artifact)


def load_sim_params(env, param_pth):
    config = configparser.ConfigParser()
    config = config["DEFAULT"]

    # convert from str to float
    params = {k: float(v) for k, v in config.items()}
    env.dyn.params = params


def parse_args():
    parser = argparse.ArgumentParser(description='TD3 agent')
    # Common arguments
    parser.add_argument('--gym_id', type=str, default="FurutaSim-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='the learning rate for the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--capture_video',
                        type=lambda x: bool(strtobool(x)), default=False,
                        help='capture videos of the agent\
                            (check out `videos` folder)')
    parser.add_argument('--log_mcap',
                        type=lambda x: bool(strtobool(x)), default=False,
                        help='log mcap data')
    parser.add_argument('--wandb_project', type=str, default="furuta",
                        help="the wandb's project name")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("-d", "--debug", action="store_true")

    # Algorithm specific arguments
    parser.add_argument('--buffer_size', type=int, default=int(1e6),
                        help='the replay memory buffer size')
    parser.add_argument('--tau', type=float, default=0.005,
                        help="target smoothing coefficient.")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="target smoothing coefficient.")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="the batch size of sample from the replay memory")
    parser.add_argument('--target_update_interval',
                        type=int, default=1,
                        help="the frequency of training policy (delayed)")
    parser.add_argument('--learning_starts',
                        type=int, default=25e3,
                        help="when to start learning")
    parser.add_argument('--sde_sample_freq',
                        type=int, default=-1,
                        help="Sample a new noise matrix every n steps when using gSDE \
                              Default: -1 \
                              (only sample at the beginning of the rollout)")
    parser.add_argument('--use_sde',
                        type=lambda x: bool(strtobool(x)), default=True,
                        help="Whether to use generalized State Dependent Exploration (gSDE) \
                        instead of action noise exploration")
    parser.add_argument('--use_sde_at_warmup',
                        type=lambda x: bool(strtobool(x)), default=True,
                        help="Whether to use gSDE instead of uniform sampling during the warm up \
                        phase (before learning starts)")

    # params to accomodate embedded system
    parser.add_argument('--model_artifact', type=str, default=None,
                        help="the artifact version of the model to load")
    parser.add_argument('--rb_artifact', type=str, default=None,
                        help="Artifact version of the replay buffer to load")
    parser.add_argument('--train_freq',
                        type=int, default=1,
                        help="The frequency of training critics/q functions")
    parser.add_argument('--train_freq_unit',
                        type=str, default="episode",
                        help="The frequency unit")
    parser.add_argument('--gradient_steps',
                        type=int, default=-1,
                        help="How many training iterations.")

    # env params
    parser.add_argument('--fs', type=int, default=100,
                        help='Sampling frequency')
    parser.add_argument('--fs_ctrl', type=int, default=100,
                        help='control frequency')
    parser.add_argument('--episode_length', type=int, default=3000,
                        help='the maximum length of each episode. \
                        -1 = infinite')
    parser.add_argument('--safety_th_lim', type=float, default=1.5,
                        help='Max motor (theta) angle in rad.')
    parser.add_argument('--action_limiter',
                        type=lambda x: bool(strtobool(x)), default=False,
                        help='Restrict actions')
    parser.add_argument('--state_limits', type=str, default="low",
                        help='Wether to use high or low limits. See code.')
    parser.add_argument('--reward', type=str, default="simple",
                        help='Which reward to use? See env code.')
    parser.add_argument('--continuity_cost',
                        type=lambda x: bool(strtobool(x)), default=False,
                        help='If true use continuity cost from HistoryWrapper')
    parser.add_argument('--history', type=int, default=1,
                        help='If >1 use HistoryWrapper')
    parser.add_argument('--custom_sim',
                        type=str, default=None,
                        help='Use params from the provided file.')

    args = parser.parse_args()

    if args.history < 2 and args.continuity_cost:
        logging.error("Can't use continuity cost if history < 2")
        quit()

    return args


if __name__ == "__main__":
    args = parse_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging_level
    )
    main(args)
