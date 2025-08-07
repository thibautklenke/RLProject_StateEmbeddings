import random

import gymnasium as gym
import numpy as np
import torch as th
from minatar.gym import register_envs
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.dqn import DQN

from state_embedding.env import EmbeddingEnv
from state_embedding.train import pretrain_combined, pretrain_qloss

from state_embedding.env import ContextEnv

SEED = 0


def hello() -> None:
    # FIXME: currently unused
    register_envs()  # Register minatar namespace for gymnasium
    set_random_seed(SEED)
    th.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = "cuda" if th.cuda.is_available() else "cpu"
    
    env = gym.make("MinAtar/Seaquest-v1")
    env.reset(seed=SEED)

    dqn = pretrain_combined(
        env,
        embedding_kwargs={
            "features_dim": 1024,
            "window_size": 15,
            "n_head": 2,
            "n_layers": 6,
        },
        total_timesteps=100,
        device=device
    )
    
    embedding_net = dqn.q_net.features_extractor
    embedding_net.to(device)
    embedding_net.eval()

    embedding_env = EmbeddingEnv(
        env=ContextEnv(env, 5), embedding_module=embedding_net, window_size=embedding_net.window_size
    )
    dqn = DQN("MlpPolicy", embedding_env, device=device)
    dqn.learn(total_timesteps=5_000_000, progress_bar=True)


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
    import torch

    torch.autograd.set_detect_anomaly(True)
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
