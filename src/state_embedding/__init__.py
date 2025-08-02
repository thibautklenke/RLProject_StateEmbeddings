import gymnasium as gym

from state_embedding.embed import DQNWithEmbedLoss
from minatar.gym import register_envs
from stable_baselines3.common.env_util import make_vec_env

import torch as th
import numpy as np
import random
from stable_baselines3.common.utils import set_random_seed

SEED = 0


def hello() -> None:
    register_envs()  # Register minatar namespace for gymnasium
    set_random_seed(SEED)
    th.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)

    # vec_env = make_vec_env(lambda: gym.make('MinAtar/Seaquest-v1'), n_envs=2)

    dqn = DQNWithEmbedLoss("MlpPolicy", env)
    dqn.learn(total_timesteps=10000)
    print("Hello from hello-world!")
