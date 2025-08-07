import numpy as np
import breakout_experiment as be
import cartpole_experiment as ce
import seaquest_experiment as se
import seaquestmarkov_experiment as sem
import random
import gymnasium as gym
import numpy as np
import torch as th
from minatar.gym import register_envs
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.dqn import DQN

from state_embedding.env import EmbeddingEnv
from state_embedding.train import pretrain_combined, pretrain_qloss

def main() -> None:
    seeds = range(1)
    register_envs()  # Register minatar namespace for gymnasium

    for SEED in seeds:
        print(SEED)
        set_random_seed(SEED)
        th.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        ce.pretrain(SEED)
        ce.train(SEED)


if __name__ == '__main__':
    main()
