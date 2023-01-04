import importlib
import logging
import os
import random
from pathlib import Path

import furuta_gym  # noqa F420
import gym
import hydra
import numpy as np
import stable_baselines3
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import wandb

# TODO
# - save model/replay buffer X
# - finish setup wrappers: think about how to setup; maybe have some hardcoded wrappers?
# - load custom sim parameters X


def download_artifact_file(artifact_alias, filename):
    """Download artifact and returns path to filename.

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


def instantiate_gym_wrapper(wrapper_name: str, wrapper_args: DictConfig, env: gym.Env) -> gym.Env:
    """Instantiate a gym wrapper from a config."""
    module = importlib.import_module(wrapper_args.module)
    wrapper_class = getattr(module, wrapper_name)

    # pop module from DictConfig wrappers args
    with open_dict(wrapper_args):
        wrapper_args.pop("module")

    return wrapper_class(env, **wrapper_args)


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    print(cfg)

    logging_level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging_level)

    # wandb expect a primitive dict
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # setup wandb run
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        sync_tensorboard=True,
        monitor_gym=cfg.capture_video,
        save_code=True,
    )

    # setup env
    env = gym.make(cfg.gym_id, **cfg.env)

    # seed everything
    env.seed(cfg.seed)
    env.action_space.seed(cfg.seed)
    env.observation_space.seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic

    # setup wrappers
    for wrapper_name, wrapper_args in cfg.wrappers.items():
        try:
            env = instantiate_gym_wrapper(wrapper_name, wrapper_args, env)
        except Exception as e:
            # print stack trace
            logging.warning(e, exc_info=True)

    if cfg.capture_video:
        # TODO add headleas arg, depends on the machine
        # import pyvirtualdisplay
        # pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
        env = DummyVecEnv([lambda: env])
        env = VecVideoRecorder(
            env,
            f"videos/{wandb.run.id}",
            record_video_trigger=lambda x: x % 3000 == 0,
            video_length=300,
        )

    # setup algo/model
    verbose = 2 if cfg.debug else 0
    model = hydra.utils.instantiate(
        cfg.algo, env=env, tensorboard_log=f"runs/{run.id}", verbose=verbose
    )

    # load model/replay buffer
    if cfg.model_artifact:
        model.load(download_artifact_file(f"model:{cfg.model_artifact}", "model.zip"))

    if cfg.replay_buffer_artifact:
        rb_path = download_artifact_file(
            f"replay_buffer:{cfg.replay_buffer_artifact}", "buffer.pkl"
        )
        model.load_replay_buffer(rb_path)

    # Stop training when the model reaches the reward threshold
    eval_callback = None
    if cfg.early_stopping_reward_threshold:
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=cfg.early_stopping_reward_threshold, verbose=1
        )
        # use same env for eval
        eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

    try:
        logging.info("Starting to train")
        model.learn(total_timesteps=cfg.total_timesteps, callback=eval_callback)
    except KeyboardInterrupt:
        logging.info("Interupting training")

    if cfg.save_model:
        logging.info("Saving model to artifacts")
        model_path = f"runs/{run.id}/models/sac.zip"
        model.save(model_path)
        upload_file_to_artifacts(model_path, "sac_model", "model")

    if cfg.save_replay_buffer:
        logging.info("Saving replay_buffer to artifacts")
        buffer_path = f"runs/{run.id}/buffers/buffer.pkl"
        model.save_replay_buffer(buffer_path)
        upload_file_to_artifacts(buffer_path, "sac_replay_buffer", "replay buffer")

    env.close()
    run.finish()


if __name__ == "__main__":
    main()
