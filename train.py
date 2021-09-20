import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback

# hack because I didn't manage to install CartPoleSwingUp in editable mode
import sys
sys.path.append("gym-cartpole-swingup")
import gym_cartpole_swingup

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 5e5,
    "env_name": "CartPoleSwingUp-v0",
}
run = wandb.init(
    project="furuta",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # record stats such as returns
    return env


env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 1e5 == 0, video_length=500)
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
