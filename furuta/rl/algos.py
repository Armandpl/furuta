import stable_baselines3


# wrapper class for stable-baselines3.SAC
# TODO can we make one class for all algos?
# check if they all have the train freq param
# check if they have other tuple args
# check if it would be cleaner for sb3 to accept list instead of tuple?
# TODO also does having the sb3 import means i need sb3 when importing from everywhere else in the package?
class SAC(stable_baselines3.SAC):
    def __init__(self, **kwargs):
        # sb3 expects tuple, omegaconf returns list
        # so we need to convert kwarg train_freq from tuple to list
        kwargs.update({"train_freq": tuple(kwargs["train_freq"])})

        super().__init__(**kwargs)