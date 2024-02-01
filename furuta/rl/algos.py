import sb3_contrib
import sbx
import stable_baselines3


# wrapper class for stable-baselines3.SAC
# TODO is there a cleaner way to do this?
class BaseAlgoWrapper:
    def __init__(self, **kwargs):
        # sb3 expects tuple, omegaconf returns list
        # so we need to convert kwarg train_freq from tuple to list
        if "train_freq" in kwargs and isinstance(kwargs["train_freq"], list):
            kwargs.update({"train_freq": tuple(kwargs["train_freq"])})

        super().__init__(**kwargs)


class SAC(BaseAlgoWrapper, stable_baselines3.SAC):
    pass


class TQC(BaseAlgoWrapper, sb3_contrib.TQC):
    pass


class SBXTQC(BaseAlgoWrapper, sbx.TQC):
    pass
