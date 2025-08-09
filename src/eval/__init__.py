from itertools import product
from typing import Any

import gymnasium as gym
import torch as th
from minatar.gym import register_envs
from stable_baselines3.common.utils import configure_logger

from state_embedding.embedding import StateEmbedding
from state_embedding.embedding_eval import evaluate_embedding
from state_embedding.env import ContextEnv


def load_embedding(path: str, embedding_kwargs: dict[str, Any]) -> StateEmbedding:
    policy = th.load(
        "./breakout/breakout_pretrain_combined_embedding_100000_steps/policy.pth",
        weights_only=False,
        map_location=th.device("cpu"),
    )
    # Only keep keys in dict with prefix "q_net.features_extractor."
    # and remove that prefix from the keys.
    policy = {
        k.replace("q_net.features_extractor.", ""): v
        for k, v in policy.items()
        if k.startswith("q_net.features_extractor.")
    }
    embedding = StateEmbedding(**embedding_kwargs)
    embedding.load_state_dict(policy)
    embedding.eval()
    return embedding


def eval_pretrain_breakout():
    label_steps = [50_000 * i for i in range(1, 11)]
    label_type = ["qloss", "combined"]

    register_envs()
    env = ContextEnv(gym.make("MinAtar/Breakout-v1"), window_size=10)
    env.reset(seed=0)

    # Initialize raw tensor board logger
    logger = {
        "qloss": configure_logger(
            verbose=1, tensorboard_log="./eval_logs", tb_log_name="qloss"
        ),
        "combined": configure_logger(
            verbose=1, tensorboard_log="./eval_logs", tb_log_name="combined"
        ),
    }

    for step, train_type in product(label_steps, label_type):
        embedding = load_embedding(
            f"./breakout/breakout_pretrain_{train_type}_embedding_{step}_steps/policy.pth",
            {
                "observation_space": env.observation_space,
                "features_dim": 256,
                "window_size": 10,
                "n_head": 2,
                "n_layers": 2,
            },
        )

        loss_mean, loss_std = evaluate_embedding(
            env, embedding, total_steps=1_000, deterministic=False, hidden_size=64
        )

        logger[train_type].record("eval/loss_mean", loss_mean)
        logger[train_type].record("eval/loss_std", loss_std)
        logger[train_type].dump(step // 50_000)
        print("Run done")


if __name__ == "__main__":
    eval_pretrain_breakout()
