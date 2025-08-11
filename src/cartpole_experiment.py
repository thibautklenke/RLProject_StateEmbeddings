import gymnasium as gym
import torch as th
from stable_baselines3.common.callbacks import (
    ProgressBarCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.dqn import DQN
from stable_baselines3.ppo import PPO
from state_embedding.callbacks import EveryNSteps

import subprocess

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

embedding_kwargs = {
    "features_dim": 3,
    "window_size": 10,
    "n_head": 2,
    "n_layers": 2,
}

env_name = "CartPole-v1"
env_name_short = "cartpole"

n_pretrain = 40_000
n_train = 100_000
net_arch = [64, 64]


def pretrain(seed: int = 0) -> None:
    """Pretrain embedding networks using different pretraining objectives in the CartPole environment.

    For each pretraining type, trains a DQN agent with the specified pretraining function,
    saves the resulting embedding network to disk, and logs progress and checkpoints.

    Parameters
    ----------
    seed : int, optional
        Random seed for environment initialization (default is 0).

    """
    env = gym.make(env_name)
    env.reset(seed=seed)
    device = "cuda" if th.cuda.is_available() else "cpu"

    # Use deterministic actions for evaluation
    for pretrain_name, pretrain_function in pretrain_types:
        dqn = pretrain_function(
            env,
            tensorboard_log=f"./logs/{env_name_short}/",
            embedding_kwargs=embedding_kwargs,
            policy_kwargs={"net_arch": net_arch},
            total_timesteps=n_pretrain,
            callbacks=[
                ProgressBarCallback(),
                CheckpointCallback(
                    save_freq=n_pretrain // 10,
                    save_path=f"./saves/{env_name_short}/",
                    name_prefix=f"{env_name_short}_{pretrain_name}_embedding",
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                ),
                EveryNSteps(
                    n_steps=n_pretrain // 10,
                    callback=lambda: subprocess.call(
                        "/workspace/RLProject_StateEmbeddings/move_to_s3.sh"
                    ),
                ),
            ],
            device=device,
            exploration_fraction=0.2,
            exploration_final_eps=0.1,
        )
        embedding_net = dqn.q_net.features_extractor

        th.save(embedding_net, f"embedding_net_{env_name_short}-{pretrain_name}.pth")


def train(seed: int = 0) -> None:
    """Train RL agents with and without pretrained embeddings in the CartPole environment.

    For each RL algorithm, first trains a baseline agent, then trains agents using
    each pretrained embedding network. Evaluation is performed during training.

    Parameters
    ----------
    seed : int, optional
        Random seed for environment initialization (default is 0).

    """
    device = "cuda" if th.cuda.is_available() else "cpu"
    for train_algorithm_name, train_algorithm in train_algorithm_types:
        # first train the algorithm normally so we know a benchmark
        env = gym.make(env_name)
        env.reset(seed=seed)

        eval_env = gym.make(env_name)
        eval_env.reset(seed=seed + 1)

        eval_callback = EvalCallback(
            eval_env,
            log_path=f"./logs/{env_name_short}/train/{seed}/normal/logs/",
            eval_freq=n_train // 200,
            deterministic=True,
            render=False,
        )

        model = train_algorithm(
            "MlpPolicy",
            env,
            device=device,
            tensorboard_log=f"./logs/{env_name_short}/train/{seed}/normal/128/logs/",
            policy_kwargs={
                "net_arch": net_arch,
            },
        )

        model.learn(total_timesteps=n_train, progress_bar=True, callback=eval_callback)

        # now train the algorithms with the pretrain types
        for pretrain_name, _ in pretrain_types:
            # Use deterministic actions for evaluation
            embedding_net = th.load(
                f"embedding_net_{env_name_short}-{pretrain_name}.pth",
                weights_only=False,
            )

            embedding_net.to(device)
            embedding_net.eval()

            env = gym.make(env_name)
            env.reset(seed=seed + 1)

            embedding_env = EmbeddingEnv(
                env=ContextEnv(env, embedding_kwargs["window_size"]),
                embedding_module=embedding_net,
                window_size=embedding_net.window_size,
            )

            # Separate evaluation env
            eval_env = EmbeddingEnv(
                env=ContextEnv(gym.make(env_name), embedding_kwargs["window_size"]),
                embedding_module=embedding_net,
                window_size=embedding_net.window_size,
            )
            eval_env.reset(seed=seed + 1)
            eval_callback = EvalCallback(
                eval_env,
                log_path=f"./logs/{env_name_short}/train/{seed}/{pretrain_name}/logs/",
                eval_freq=n_train // 200,
                deterministic=True,
                render=False,
            )
            model = train_algorithm(
                "MlpPolicy",
                embedding_env,
                policy_kwargs={"net_arch": net_arch},
                device=device,
                tensorboard_log=f"./logs/{env_name_short}/train/{seed}/{pretrain_name}/128/logs/",
            )

            model.learn(
                total_timesteps=n_train, progress_bar=True, callback=eval_callback
            )
