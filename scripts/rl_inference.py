import os

import hydra
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.vec_env import DummyVecEnv

import wandb
from furuta.rl.wrappers import MCAPLogger


@hydra.main(version_base="1.3", config_path="configs", config_name="rl_inference.yaml")
def main(cfg: DictConfig):
    run = wandb.init(
        project=cfg.wandb_project,
        job_type="inference",
    )
    artifact = run.use_artifact(cfg.model_artifact)
    artifact_dir = artifact.download()
    producer_cfg = DictConfig(artifact.logged_by().config)

    if "env" in cfg:
        env = hydra.utils.instantiate(cfg.env, _recursive_=True)
    else:
        env = hydra.utils.instantiate(producer_cfg.env, _recursive_=True)

    if cfg.render:
        env.render_mode = "human"

    # setup wrappers
    wrappers = cfg.wrappers if "wrappers" in cfg else producer_cfg.wrappers
    for wrapper in wrappers:
        env = hydra.utils.instantiate(wrapper, env=env)

    env = MCAPLogger(env, use_sim_time=False, log_dir="./mcap_logs/")

    env = DummyVecEnv([lambda: env])

    model = hydra.utils.instantiate(producer_cfg.algo, env=env, _convert_="all")

    model = model.load(os.path.join(artifact_dir, "model.zip"))

    ep_returns = []
    for _ in range(cfg.nb_episodes):
        ep_return = 0
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _ = env.step(action)
            ep_return += reward
            if cfg.render:
                env.render()
            if done:
                ep_returns.append(ep_return)
                break

    print(f"Returns: {ep_returns}")
    print(f"Mean return: {np.mean(ep_returns)}")
    print(f"Std return: {np.std(ep_returns)}")

    # Save the video
    env.close()
    run.finish()


if __name__ == "__main__":
    main()
