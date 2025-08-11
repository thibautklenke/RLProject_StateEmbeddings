from typing import Any, Optional

from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.dqn import DQN

from state_embedding.dqn import DQNWithReconstruction, EmbeddingPolicy
from state_embedding.env import ContextEnv


def pretrain_state_hook(env: GymEnv, embedding_kwargs: dict[str, Any]) -> None:
    # TODO: Implement reconstruction module training using just
    # reconstruction loss without embedding in Q-network
    # (see callbacks.py : EmbeddingTrainingCallback)
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
    """Pretrain an embedding network using Q-learning loss.

    Trains a DQN agent with a custom embedding policy in a context environment,
    using the specified embedding network configuration and training parameters.

    Parameters
    ----------
    env : GymEnv
        The environment to use for pretraining.
    embedding_kwargs : dict[str, Any]
        Keyword arguments for the embedding network.
    policy_kwargs : dict[str, Any], optional
        Additional keyword arguments for the policy.
    total_timesteps : int, optional
        Number of training timesteps (default is 10000).
    tensorboard_log : str, optional
        Path for TensorBoard logs (default is "./logs/").
    callbacks : list[BaseCallback], optional
        List of callbacks to use during training.
    device : str, optional
        Device to use for training ("cpu" or "cuda").
    exploration_fraction : float, optional
        Fraction of training where exploration rate is annealed (default is 0.2).
    exploration_final_eps : float, optional
        Final value of exploration rate (default is 0.1).

    Returns
    -------
    DQN
        The trained DQN agent.
    """
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
    exploration_fraction: float = 0.2,
    exploration_final_eps: float = 0.1,
) -> DQNWithReconstruction:
    """Pretrain an embedding network using a combined Q-learning and reconstruction loss.

    Trains a DQNWithReconstruction agent in a context environment, using the specified
    embedding network configuration and training parameters.

    Parameters
    ----------
    env : GymEnv
        The environment to use for pretraining.
    embedding_kwargs : dict[str, Any]
        Keyword arguments for the embedding network.
    policy_kwargs : dict[str, Any], optional
        Additional keyword arguments for the policy.
    total_timesteps : int, optional
        Number of training timesteps (default is 10000).
    tensorboard_log : str, optional
        Path for TensorBoard logs (default is "./logs/").
    callbacks : list[BaseCallback], optional
        List of callbacks to use during training.
    device : str, optional
        Device to use for training ("cpu" or "cuda").
    exploration_fraction : float, optional
        Fraction of training where exploration rate is annealed (default is 0.2).
    exploration_final_eps : float, optional
        Final value of exploration rate (default is 0.1).

    Returns
    -------
    DQNWithReconstruction
        The trained DQNWithReconstruction agent.
    """
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
