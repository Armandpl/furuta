import argparse

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback


def main(config):
    run = wandb.init(
        project="furuta",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    config = run.config

    # conditionnal imports because different dependencies on the robot
    if config.env_name == "CartPoleSwingUp-v1":
        import gym_cartpole_swingup
    elif config.env_name == "Furuta-v0":
        import furuta_gym

    def make_env():
        env = gym.make(config.env_name)
        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % config["video_frequency"] == 0, video_length=500)
    model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=10000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    run.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an agent to swing up a pendulum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--policy_type",
        type=str,
        default="MlpPolicy",
        help="Which type of policy to train."
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=5e5,
        help="How long should we train for."
    )
    parser.add_argument(
        "--video_frequency",
        type=int,
        default=5e4,
        help="A video is recorded every {video_frequency} steps."
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPoleSwingUp-v1",
        help="Which env to train in. Either sim or real."
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="If specified loads policy from Artifacts."
    )

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
