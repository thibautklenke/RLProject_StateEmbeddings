import random
import gymnasium as gym
import numpy as np
import torch as th
from minatar.gym import register_envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.dqn import DQN

from state_embedding.env import ContextEnv, EmbeddingEnv
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
    "window_size": 15,
    "n_head": 2,
    "n_layers": 4,
}

env_name = "CartPole-v1"
env_name_short = "cartpole"

n_pretrain = 2_000_000
n_train = 5_000_000

def pretrain(seed) -> None:
    env = gym.make(env_name)
    env.reset(seed=seed)
    device = "cuda" if th.cuda.is_available() else "cpu"

    # Separate evaluation env
    eval_env = gym.make(env_name)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"./logs/{env_name_short}",
                                 log_path="./logs/", eval_freq=500,
                                 deterministic=True, render=False)

    for pretrain_name, pretrain_function in pretrain_types:
        dqn = pretrain_function(
            env,
            tensorboard_log=f"./logs/{env_name_short}/",
            embedding_kwargs=embedding_kwargs,
            total_timesteps=n_pretrain,
            callbacks=[ProgressBarCallback()],#, eval_callback],
            device=device
        )
        embedding_net = dqn.q_net.features_extractor

        th.save(embedding_net, f"embedding_net_{env_name_short}-{pretrain_name}-{seed}.pth")


def train(seed) -> None:
    device = "cuda" if th.cuda.is_available() else "cpu"
    for pretrain_name, _ in pretrain_types:
        for train_algorithm_name, train_algorithm in train_algorithm_types:
            # Separate evaluation env
            eval_env = gym.make(env_name)
            # Use deterministic actions for evaluation
            eval_callback = EvalCallback(eval_env, best_model_save_path=f"./logs/{env_name_short}/",
                                         log_path="./logs/", eval_freq=500,
                                         deterministic=True, render=False)

            embedding_net = th.load(f"embedding_net_{env_name_short}-{pretrain_name}-{seed}.pth", weights_only=False)

            embedding_net.to(device)
            embedding_net.eval()

            env = gym.make(env_name)
            env.reset(seed=seed + 1)

            embedding_env = EmbeddingEnv(
                env=ContextEnv(env, embedding_kwargs["window_size"]), embedding_module=embedding_net, window_size=embedding_net.window_size
            )
            model = train_algorithm("MlpPolicy", embedding_env)#, callbacks=[ProgressBarCallback(), eval_callback])
            model.learn(total_timesteps=n_train, progress_bar=True)