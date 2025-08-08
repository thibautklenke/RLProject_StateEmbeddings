import random
import gymnasium as gym
import numpy as np
import torch as th
from minatar.gym import register_envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.dqn import DQN
from state_embedding.callbacks import EveryNSteps

import subprocess

from state_embedding.embedding import StateEmbedding
from state_embedding.env import EmbeddingEnv, ContextEnv
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
    "features_dim": 256,
    "window_size": 10,
    "n_head": 2,
    "n_layers": 2,
}

env_name = "MinAtar/Seaquest-v1"
env_name_short = "seaquest"

n_pretrain = 500_000
n_train = 4_000_000


def pretrain(seed=0) -> None:
    env = gym.make(env_name)
    env.reset(seed=seed)
    device = "cuda" if th.cuda.is_available() else "cpu"

    # Use deterministic actions for evaluation
    for pretrain_name, pretrain_function in pretrain_types:
        dqn = pretrain_function(
            env,
            tensorboard_log=f"./logs/{env_name_short}/",
            embedding_kwargs=embedding_kwargs,
            total_timesteps=n_pretrain,
            callbacks=[ProgressBarCallback(),
                       CheckpointCallback(save_freq=n_pretrain // 10, save_path=f"./saves/{env_name_short}/",
                                          name_prefix=f"{env_name_short}_embedding", save_replay_buffer=False,
                                          save_vecnormalize=False),
                       EveryNSteps(n_steps=n_pretrain // 10, callback=lambda: subprocess.call("/workspace/RLProject_StateEmbeddings/move_to_s3.sh"))
            ],
            device=device
        )
        embedding_net = dqn.q_net.features_extractor

        th.save(embedding_net, f"embedding_net_{env_name_short}-{pretrain_name}.pth")


def train(seed=0) -> None:
    device = "cuda" if th.cuda.is_available() else "cpu"
    for pretrain_name, _ in pretrain_types:
        for train_algorithm_name, train_algorithm in train_algorithm_types:
            
            # Use deterministic actions for evaluation
            embedding_net = th.load(f"embedding_net_{env_name_short}-{pretrain_name}.pth", weights_only=False)
            
            embedding_net.to(device)
            embedding_net.eval()

            env = gym.make(env_name)
            env.reset(seed=seed+1)

            embedding_env = EmbeddingEnv(
                env=ContextEnv(env, embedding_kwargs["window_size"]), embedding_module=embedding_net, window_size=embedding_net.window_size
            )

            # Separate evaluation env
            eval_env = EmbeddingEnv(
                env=ContextEnv(gym.make(env_name), embedding_kwargs["window_size"]), embedding_module=embedding_net, window_size=embedding_net.window_size
            )    
            eval_env.reset(seed=seed+1)
            eval_callback = EvalCallback(eval_env, best_model_save_path=f"./logs/{env_name_short}/{seed}/model/",
                                         log_path=f"./logs/{env_name_short}/{seed}/logs/", eval_freq=n_train//1000,
                                         deterministic=True, render=False)
            model = train_algorithm("MlpPolicy", embedding_env, device=device)
            model.learn(total_timesteps=n_train, progress_bar=True, callback=eval_callback)
