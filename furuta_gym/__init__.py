from gym.envs.registration import register

register(
    id='FurutaReal-v0',
    entry_point='furuta_gym.envs:FurutaReal',
)
register(
    id='FurutaSim-v0',
    entry_point='furuta_gym.envs:FurutaSim',
)
