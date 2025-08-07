import gymnasium as gym

from state_embedding.embed import DQNWithEmbedLoss
from minatar.gym import register_envs

import torch as th
import numpy as np
import random
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import ProgressBarCallback

from state_embedding.env import ContextEnv
from state_embedding.train import pretrain_qloss, pretrain_combined

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

    window_size = 5

    # TODO: Insert embedding module
    # embedding_module = StateEmbedNetwork(env.observation_space, embedding_size=8, window_size=window_size)

    # context_env = EmbeddingEnv(env, embedding_module, window_size)

    dqn = DQNWithEmbedLoss(
        "MlpPolicy", ContextEnv(env, window_size), learning_rate=1e-3
    )
    dqn.learn(total_timesteps=10000, callback=ProgressBarCallback())
    print("Hello from hello-world!")


def test_qloss():
    pretrain_qloss(
        gym.make("CartPole-v1"),
        embedding_kwargs={
            "features_dim": 8,
            "window_size": 5,
            "n_head": 2,
            "n_layers": 6,
        },
    )


def test_combined():
    pretrain_combined(
        gym.make("CartPole-v1"),
        embedding_kwargs={
            "features_dim": 8,
            "window_size": 5,
            "n_head": 2,
            "n_layers": 6,
        },
    )


if __name__ == "__main__":
    test_combined()
