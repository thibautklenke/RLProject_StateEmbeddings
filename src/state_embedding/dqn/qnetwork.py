from typing import Any, Optional, Tuple

import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.dqn.dqn import QNetwork
from stable_baselines3.dqn.policies import DQNPolicy
from torch import nn
from torch.nn import functional as F

from state_embedding.embedding import StateDecoder, StateEmbedding


class ReconstructionQNetwork(QNetwork):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: StateEmbedding,
        features_dim: int,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )

        self._decoder = StateDecoder(
            observation_space=features_extractor._observation_space,
            features_dim=features_extractor.features_dim,
            window_size=features_extractor.window_size,
            n_head=features_extractor.n_head,
            n_layers=features_extractor.n_layers,
        )

    def combined_forward(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor]:
        embedding = self.extract_features(obs, self.features_extractor)

        reconstructed = self._decoder(embedding)
        reconstruction_loss = F.mse_loss(reconstructed, obs)

        return self.q_net(embedding), reconstruction_loss


class EmbeddingPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        use_reconstruction_loss: bool = False,
        feature_extractor_kwargs: Optional[dict[str, Any]] = None,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.use_reconstruction_loss = use_reconstruction_loss

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

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        if self.use_reconstruction_loss:
            return ReconstructionQNetwork(**net_args).to(self.device)
        else:
            return QNetwork(**net_args).to(self.device)
