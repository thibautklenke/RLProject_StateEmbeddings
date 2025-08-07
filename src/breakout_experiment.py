import random
import gymnasium as gym
import numpy as np
import torch as th
from minatar.gym import register_envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.dqn import DQN

from state_embedding.env import EmbeddingEnv
from state_embedding.train import pretrain_combined, pretrain_qloss

pretrain_types = [
    ("pretrain_combined", pretrain_combined),
    ("pretrain_qloss", pretrain_qloss),
]

train_algorithm_types = [
    ("DQN", DQN),
    ("PPO", PPO),
]

embedding_kwargs={
                "features_dim": 1024,
                "window_size": 20,
                "n_head": 8,
                "n_layers": 6,
            }

env_name = "MinAtar/Breakout-v1"
env_name_short = "breakout"

def pretrain(seed) -> None:
    env = gym.make(env_name)
    env.reset(seed=seed)

    # Separate evaluation env
    eval_env = gym.make(env_name)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"./logs/{env_name_short}",
                                 log_path="./logs/", eval_freq=500,
                                 deterministic=True, render=False)

    for pretrain_name, pretrain_function in pretrain_types:
        dqn = pretrain_function(
            env,
            tensorboard_log="./logs/breakout/",
            embedding_kwargs=embedding_kwargs,
            total_timesteps=2000,
            callbacks=[ProgressBarCallback(), eval_callback],
        )
        embedding_net = dqn.q_net.features_extractor

        th.save(embedding_net.state_dict(), f"embedding_net_{pretrain_name}-{seed}.pth")


def train(seed) -> None:
    for pretrain_name, _ in pretrain_types:
        for train_algorithm_name, train_algorithm in train_algorithm_types:
            # Separate evaluation env
            eval_env = gym.make(env_name)
            # Use deterministic actions for evaluation
            eval_callback = EvalCallback(eval_env, best_model_save_path=f"./logs/{env_name_short}/",
                                         log_path="./logs/", eval_freq=500,
                                         deterministic=True, render=False)

            embedding_net = th.load(f"embedding_net_{pretrain_name}-{seed}.pth")

            embedding_net.eval()

            env = gym.make(env_name)
            env.reset(seed=seed+1)

            embedding_env = EmbeddingEnv(
                env=env, embedding_module=embedding_net, window_size=embedding_net.window_size
            )
            model = train_algorithm("MlpPolicy", embedding_env, callbacks=[ProgressBarCallback(), eval_callback])
            model.learn(total_timesteps=10, progress_bar=True)