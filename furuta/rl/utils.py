import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb


def seed_everything(env, seed, cudnn_deterministic):
    # got this from cleanrl
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    env.reset(seed=seed)


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
