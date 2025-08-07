from typing import Any, Optional

import gymnasium as gym
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from torch import nn

from state_embedding.embedding import StateEmbedding
from state_embedding.env import ContextEnv


class EmbeddingPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        feature_extractor_kwargs: Optional[dict[str, Any]] = None,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=StateEmbedding,
            features_extractor_kwargs=feature_extractor_kwargs,
            normalize_images=True,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


def run():
    env = gym.make("CartPole-v1")
    dqn = DQN(
        policy=DQNPolicy,
        env=ContextEnv(env, window_size=5),
        policy_kwargs={
            "features_extractor_class": StateEmbedding,
            "features_extractor_kwargs": {
                "features_dim": 8,
                "window_size": 5,
                "n_head": 2,
                "n_layers": 6,
            },
        },
    )
    dqn.learn(total_timesteps=10000, callback=ProgressBarCallback())
