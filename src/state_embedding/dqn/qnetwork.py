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
    """QNetwork with an additional state reconstruction decoder.

    This network uses a StateEmbedding as the feature extractor and adds a StateDecoder
    for reconstructing the input observation from the embedding. The combined_forward method
    returns both Q-values and the reconstruction loss.

    Parameters
    ----------
    observation_space : spaces.Space
        The observation space of the environment.
    action_space : spaces.Discrete
        The action space of the environment.
    features_extractor : StateEmbedding
        The embedding module used as feature extractor.
    features_dim : int
        Dimension of the embedding vector.
    net_arch : list[int], optional
        Architecture of the Q-network.
    activation_fn : type[nn.Module], optional
        Activation function for the Q-network.
    normalize_images : bool, optional
        Whether to normalize images.

    Methods
    -------
    combined_forward(obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor]
        Compute Q-values and reconstruction loss for a batch of observations.
    """

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
        """Initialize the ReconstructionQNetwork.

        Parameters
        ----------
        observation_space : spaces.Space
            The observation space of the environment.
        action_space : spaces.Discrete
            The action space of the environment.
        features_extractor : StateEmbedding
            The embedding module used as feature extractor.
        features_dim : int
            Dimension of the embedding vector.
        net_arch : list[int], optional
            Architecture of the Q-network.
        activation_fn : type[nn.Module], optional
            Activation function for the Q-network (default is nn.ReLU).
        normalize_images : bool, optional
            Whether to normalize images (default is True).
        """
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
        """Compute Q-values and reconstruction loss for a batch of observations.

        Parameters
        ----------
        obs : PyTorchObs
            Batch of observations.

        Returns
        -------
        Tuple[th.Tensor, th.Tensor]
            Q-values and reconstruction loss.
        """
        embedding = self.extract_features(obs, self.features_extractor)
        reconstructed = self._decoder(embedding)
        reconstruction_loss = F.mse_loss(reconstructed, obs)
        return self.q_net(embedding), reconstruction_loss


class EmbeddingPolicy(DQNPolicy):
    """DQNPolicy that supports using a StateEmbedding feature extractor and optional reconstruction loss.

    This policy can be configured to use a ReconstructionQNetwork for joint Q-learning and
    reconstruction, or a standard QNetwork for regular DQN.

    Parameters
    ----------
    observation_space : spaces.Space
        The observation space of the environment.
    action_space : spaces.Discrete
        The action space of the environment.
    lr_schedule : Schedule
        Learning rate schedule.
    use_reconstruction_loss : bool, optional
        Whether to use the reconstruction loss.
    feature_extractor_kwargs : dict[str, Any], optional
        Keyword arguments for the StateEmbedding feature extractor.
    net_arch : list[int], optional
        Architecture of the Q-network.
    activation_fn : type[nn.Module], optional
        Activation function for the Q-network.
    optimizer_class : type[th.optim.Optimizer], optional
        Optimizer class.
    optimizer_kwargs : dict[str, Any], optional
        Keyword arguments for the optimizer.

    Methods
    -------
    make_q_net() -> QNetwork
        Create the Q-network (optionally with reconstruction loss).
    """

    use_reconstruction_loss: bool

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
        """Initialize the EmbeddingPolicy.

        Parameters
        ----------
        observation_space : spaces.Space
            The observation space of the environment.
        action_space : spaces.Discrete
            The action space of the environment.
        lr_schedule : Schedule
            Learning rate schedule.
        use_reconstruction_loss : bool, optional
            Whether to use the reconstruction loss (default is False).
        feature_extractor_kwargs : dict[str, Any], optional
            Keyword arguments for the StateEmbedding feature extractor.
        net_arch : list[int], optional
            Architecture of the Q-network.
        activation_fn : type[nn.Module], optional
            Activation function for the Q-network (default is nn.ReLU).
        optimizer_class : type[th.optim.Optimizer], optional
            Optimizer class (default is th.optim.Adam).
        optimizer_kwargs : dict[str, Any], optional
            Keyword arguments for the optimizer.
        """
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
        """Create the Q-network (optionally with reconstruction loss).

        Returns
        -------
        QNetwork
            The Q-network instance.
        """
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        if self.use_reconstruction_loss:
            return ReconstructionQNetwork(**net_args).to(self.device)
        else:
            return QNetwork(**net_args).to(self.device)
