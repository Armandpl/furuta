from gymnasium.envs.registration import register

register(
    id="FurutaReal-v0",
    entry_point="furuta.rl.envs.furuta_real:FurutaReal",
)
register(
    id="FurutaSim-v0",
    entry_point="furuta.rl.envs.furuta_sim:FurutaSim",
)
