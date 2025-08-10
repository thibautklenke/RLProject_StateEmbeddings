from typing import Any, Optional

from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.dqn import DQN

from state_embedding.dqn import DQNWithReconstruction, EmbeddingPolicy
from state_embedding.env import ContextEnv


def pretrain_state_hook(env: GymEnv, embedding_kwargs: dict[str, Any]):
    pass


def pretrain_qloss(
    env: GymEnv,
    embedding_kwargs: dict[str, Any],
    policy_kwargs: Optional[dict[str, Any]] = None,
    total_timesteps: int = 10000,
    tensorboard_log: str = "./logs/",
    callbacks: list[BaseCallback] = [ProgressBarCallback()],
    device: str = "cpu",
    exploration_fraction: float = 0.2,
    exploration_final_eps: float = 0.1,
) -> DQN:
    window_size = embedding_kwargs.get("window_size", 5)
    dqn = DQN(
        env=ContextEnv(env, window_size=window_size),
        policy=EmbeddingPolicy,
        tensorboard_log=tensorboard_log,
        policy_kwargs={
            "feature_extractor_kwargs": embedding_kwargs,
        }
        | (policy_kwargs or {}),
        device=device,
        buffer_size=100_000,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
    )
    dqn.learn(total_timesteps=total_timesteps, callback=callbacks)
    return dqn


def pretrain_combined(
    env: GymEnv,
    embedding_kwargs: dict[str, Any],
    policy_kwargs: Optional[dict[str, Any]] = None,
    total_timesteps: int = 10000,
    tensorboard_log: str = "./logs/",
    callbacks: list[BaseCallback] = [ProgressBarCallback()],
    device: str = "cpu",
    exploration_fraction: float=0.2,
    exploration_final_eps: float=0.1,
) -> DQNWithReconstruction:
    window_size = embedding_kwargs.get("window_size", 5)
    dqn = DQNWithReconstruction(
        env=ContextEnv(env, window_size=window_size),
        policy_kwargs={
            "feature_extractor_kwargs": embedding_kwargs,
        }
        | (policy_kwargs or {}),
        device=device,
        tensorboard_log=tensorboard_log,
        buffer_size=100_000,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
    )
    dqn.learn(total_timesteps=total_timesteps, callback=callbacks)
    return dqn
