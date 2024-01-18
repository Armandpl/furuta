import copy
import logging

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecVideoRecorder,
)

from furuta.rl.envs.furuta_real import FurutaReal
from furuta.rl.utils import (
    download_artifact_file,
    seed_everything,
    upload_file_to_artifacts,
)


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):

    logging_level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging_level)

    # setup wandb run
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        sync_tensorboard=True,
        monitor_gym=cfg.capture_video,
        save_code=True,
    )

    # setup env
    env = hydra.utils.instantiate(cfg.env, _recursive_=True)
    check_env(env)

    # seed everything
    seed_everything(env, cfg.seed, cfg.cudnn_deterministic)

    # setup wrappers
    for wrapper in cfg.wrappers.wrappers:
        env = hydra.utils.instantiate(wrapper, env=env)

    # don't paralelize if it's the real robot
    if isinstance(env.unwrapped, FurutaReal):
        vec_env = DummyVecEnv([lambda: env])
        if cfg.n_envs > 1:
            logging.warning("n_envs > 1 but using real robot, ignoring n_envs")
    elif cfg.n_envs == 1:
        vec_env = DummyVecEnv([lambda: env])
    else:
        vec_env = SubprocVecEnv([lambda: copy.deepcopy(env) for _ in range(cfg.n_envs)])

    # setup algo/model
    verbose = 2 if cfg.debug else 0
    model = hydra.utils.instantiate(
        cfg.algo, env=vec_env, tensorboard_log=f"runs/{run.id}", verbose=verbose, _convert_="all"
    )

    # load model/replay buffer
    if cfg.model_artifact:
        model.load(download_artifact_file(cfg.model_artifact, "model.zip"))

    if cfg.replay_buffer_artifact:
        rb_path = download_artifact_file(cfg.replay_buffer_artifact, "buffer.pkl")
        model.load_replay_buffer(rb_path)

    # Stop training when the model reaches the reward threshold
    eval_callback = None
    if cfg.evaluation.early_stopping_reward_threshold:
        # TODO seems like weird things happen when we use the same env for training and eval
        # e.g we get stuck in eval mode
        eval_env = copy.deepcopy(env)
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=cfg.evaluation.early_stopping_reward_threshold, verbose=1
        )
        # use same env for eval
        # TODO maybe do eval even when we don't want to early stop?
        # would it be useful?
        # accounting for vec envs

        eval_freq = max(cfg.evaluation.eval_freq // cfg.n_envs, 1)
        eval_callback = EvalCallback(
            eval_env,
            deterministic=cfg.evaluation.deterministic,
            n_eval_episodes=cfg.evaluation.n_eval_episodes,
            eval_freq=eval_freq,
            callback_on_new_best=callback_on_best,
            verbose=1,
        )

    try:
        logging.info("Starting to train")
        model.learn(total_timesteps=cfg.total_timesteps, callback=eval_callback)
    except KeyboardInterrupt:
        logging.info("Interupting training")

    # only save last video
    if cfg.capture_video:
        # TODO add headless arg, depends on the machine
        # import pyvirtualdisplay
        # pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
        video_length = 500
        env = DummyVecEnv([lambda: env])
        env = VecVideoRecorder(
            env,
            f"videos/{wandb.run.id}",
            record_video_trigger=lambda x: x == 0,
            video_length=video_length,
        )

        obs = env.reset()
        for _ in range(video_length + 1):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, _ = env.step(action)
            if done:
                break
        # Save the video
        env.close()

    if cfg.save_model:
        logging.info("Saving model to artifacts")
        model_path = f"runs/{run.id}/models/model.zip"
        model.save(model_path)
        upload_file_to_artifacts(model_path, "model", "model")

    if cfg.save_replay_buffer:
        logging.info("Saving replay_buffer to artifacts")
        buffer_path = f"runs/{run.id}/buffers/buffer.pkl"
        model.save_replay_buffer(buffer_path)
        upload_file_to_artifacts(buffer_path, "replay_buffer", "replay buffer")

    env.close()
    run.finish()


if __name__ == "__main__":
    main()
