import argparse
import logging
import os
import pickle

import gym
from gym.wrappers import TimeLimit
import pyvirtualdisplay
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb

import furuta_gym  # noqa F420
from furuta_gym.envs.wrappers import GentlyTerminating, HistoryWrapper


def main(args):
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
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
                train_freq=args.train_freq, gradient_steps=args.gradient_steps,
                tensorboard_log=f"runs/{run.id}"
            )

    try:
        model.learn(total_timesteps=args.total_timesteps)
    except KeyboardInterrupt:
        logging.info("Interupting training")

    logging.info("Saving model")
    model_path = f"runs/{run.id}/models/sac.zip"
    model.save(model_path)
    artifact = wandb.Artifact("sac_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    logging.info("Saving replay buffer")
    buffer_path = f"runs/{run.id}/buffers/buffer.pkl"
    model.save_replay_buffer(buffer_path)
    artifact = wandb.Artifact("sac_replay_buffer", type="replay buffer")
    artifact.add_file(buffer_path)
    wandb.log_artifact(artifact)

    env.close()
    run.finish()


def setup_env(args):
    # base env
    env = gym.make(args.gym_id, fs=args.fs, fs_ctrl=args.fs_ctrl,
                   action_limiter=args.action_limiter,
                   safety_th_lim=args.safety_th_lim)

    # load custom sim params
    if args.gym_id == "FurutaSim-v0" and args.custom_sim:
        from furuta_params import params
        env.dyn.params = params
        logging.info(f"Loaded sim params: \n{env.dyn.params}")
        wandb.run.summary["sim_params"] = params

    if args.episode_length != -1:
        env = TimeLimit(env, args.episode_length)

    if args.history > 1:
        env = HistoryWrapper(env, args.history, args.continuity_cost)

    env = Monitor(env)

    # if robot
    if args.gym_id == "FurutaReal-v0":
        env = GentlyTerminating(env)
        # TODO: add wrapper to enforce control freq

    if args.capture_video:
        env = DummyVecEnv([lambda: env])
        pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
        env = VecVideoRecorder(env, f"videos/{wandb.run.id}",
                               record_video_trigger=lambda x: x % 30e3 == 0,
                               video_length=200)

    return env


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
    parser.add_argument('--capture_video', action="store_true",
                        help='weather to capture videos of the agent\
                              performances (check out `videos` folder)')
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
    parser.add_argument('--use_sde',
                        type=bool, default=True,
                        help="Whether to use generalized State Dependent Exploration (gSDE) \
                        instead of action noise exploration")
    parser.add_argument('--use_sde_at_warmup',
                        type=bool, default=True,
                        help="Whether to use gSDE instead of uniform sampling during the warm up \
                        phase (before learning starts)")

    # params to accomodate embedded system
    # parser.add_argument('--model-artifact', type=str, default=None,
    #                     help="the artifact version of the model to load")
    # parser.add_argument('--rb-artifact', type=str, default=None,
    #                     help="Artifact version of the replay buffer to load")
    parser.add_argument('--train_freq',
                        type=int, default=3000,  # = 30 sec at 100hz
                        help="The frequency of training critics/q functions")
    parser.add_argument('--gradient_steps',
                        type=int, default=400,
                        help="How many training iterations")

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
    parser.add_argument('--action_limiter', type=bool, default=True,
                        help='Restrict actions')
    parser.add_argument('--reward', type=str, default="simple",
                        help='Which reward to use? See env code.')
    parser.add_argument('--continuity_cost', type=bool, default=True,
                        help='If true use continuity cost from HistoryWrapper')
    parser.add_argument('--history', type=int, default=1,
                        help='If >1 use HistoryWrapper')
    parser.add_argument('--custom_sim', action="store_true",
                        help='If specified, use params from furuta_params.py')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
