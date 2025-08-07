from typing import Any, Optional

from stable_baselines3.common.type_aliases import GymEnv

from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.dqn import DQN

from state_embedding.dqn import DQNWithReconstruction
from state_embedding.qnetwork import EmbeddingPolicy
from state_embedding.env import ContextEnv


def pretrain_state_hook(env: GymEnv, embedding_kwargs: dict[str, Any]):
    pass


def pretrain_qloss(
    env: GymEnv,
    embedding_kwargs: dict[str, Any],
    policy_kwargs: Optional[dict[str, Any]] = None,
):
    window_size = embedding_kwargs.get("window_size", 5)
    dqn = DQN(
        env=ContextEnv(env, window_size=window_size),
        policy=EmbeddingPolicy,
        policy_kwargs={
            "feature_extractor_kwargs": embedding_kwargs,
        }.update(policy_kwargs or {}),
    )
    dqn.learn(total_timesteps=10000, callback=ProgressBarCallback())


def pretrain_combined(
    env: GymEnv,
    embedding_kwargs: dict[str, Any],
    policy_kwargs: Optional[dict[str, Any]] = None,
):
    window_size = embedding_kwargs.get("window_size", 5)
    dqn = DQNWithReconstruction(
        env=ContextEnv(env, window_size=window_size),
        policy_kwargs={
            "feature_extractor_kwargs": embedding_kwargs,
        }
        | (policy_kwargs or {}),
    )
    dqn.learn(total_timesteps=10000, callback=ProgressBarCallback())
