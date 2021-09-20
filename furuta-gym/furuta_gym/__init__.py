from gym.envs.registration import register

register(
    id='Furuta-v0',
    entry_point='furuta_gym.envs:FurutaEnv',
)
