from stable_baselines3.common.env_checker import check_env

from furuta.rl.envs.furuta_real import FurutaReal
from furuta.rl.envs.furuta_sim import FurutaSim


def test_sim_env():
    env = FurutaSim()
    check_env(env)


# TODO how to check real env?
# would need to mock the robot
