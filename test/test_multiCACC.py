import pytest
from marlcacc.rlenv.multiCACCenv import multiCACC
import numpy as np


kwargs = {"num_agents": 10,
          "initial_position": np.cumsum(np.ones(10)) * 20,
          "initial_speed": np.ones(10) * 30,
          "dt": 0.1,
          "acc_bound": (-5, 5),
          "track_length": 1000.0,
          "max_speed": 120.0, }


def test_init():
    return multiCACC(**kwargs)


def test_get_state():
    env = multiCACC(**kwargs)
    return env.get_state()


def test_reset():
    env = multiCACC(**kwargs)
    return env.reset()


def test_step():
    env = multiCACC(**kwargs)
    return env.step(np.ones(10))


def test_get_reward():
    env = multiCACC(**kwargs)
    return env.get_reward()


def test_env():
    kwargs = {"num_agents": 10,
              "initial_position": np.cumsum(np.ones(10)) * 20,
              "initial_speed": np.ones(10) * 30,
              "dt": 0.1,
              "acc_bound": (-5, 5),
              "track_length": 1000.0,
              "max_speed": 120.0/3.6,
              }

    from marlcacc.rlenv.multiCACCenv import multiCACC
    env = multiCACC(**kwargs)
    env.reset()
    for _ in range(1000):
        env.step(np.random.rand(env.num_agents, 1))
        env.render()

    return 1


# test_env()V
