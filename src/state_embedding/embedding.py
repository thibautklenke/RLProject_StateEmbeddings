import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class StateEmbedding(BaseFeaturesExtractor):
    """Transformer-based state embedding module for RL environments.

    Encodes a window of observations into a fixed-size embedding vector using a transformer encoder.
    The output embedding is constrained to [0, 1] by a sigmoid activation.

    Parameters
    ----------
    observation_space : spaces.Space
        The observation space of the environment.
    features_dim : int, optional
        Dimension of the output embedding vector.
    window_size : int, optional
        Number of previous observations to stack.
    n_head : int, optional
        Number of attention heads in the transformer encoder.
    n_layers : int, optional
        Number of transformer encoder layers.

    Methods
    -------
    forward(observations: th.Tensor) -> th.Tensor
        Encode a batch of observation windows into embedding vectors.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 8,
        window_size: int = 10,
        n_head: int = 2,
        n_layers: int = 6,
    ):
        """Initialize the StateEmbedding module.

        Parameters
        ----------
        observation_space : spaces.Space
            The observation space of the environment.
        features_dim : int, optional
            Dimension of the output embedding vector (default is 8).
        window_size : int, optional
            Number of previous observations to stack (default is 10).
        n_head : int, optional
            Number of attention heads in the transformer encoder (default is 2).
        n_layers : int, optional
            Number of transformer encoder layers (default is 6).
        """
        super().__init__(observation_space, features_dim)

        self.window_size = window_size
        self.n_head = n_head
        self.n_layers = n_layers

        in_dim = int(np.prod(observation_space.shape[1:]))
        self._in_dim = max(n_head, 2 ** ((in_dim - 1).bit_length()))

        self.head = nn.Sequential(
            # [bs, window_size, *obs.shape]
            nn.Flatten(start_dim=2),
            # [bs, window_size, in_dim]
            nn.Linear(in_dim, self._in_dim),
            # [bs, window_size, self._in_dim]
        )

        self._encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self._in_dim, n_head, batch_first=True),
            n_layers,
        )
        self._encoder_compress = nn.Sequential(
            nn.Linear(self._in_dim, features_dim),
            nn.Sigmoid(),  # Force output components to be in [0, 1]
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Encode a batch of observation windows into embedding vectors.

        Parameters
        ----------
        observations : th.Tensor
            Tensor of shape [batch_size, window_size, *obs.shape].

        Returns
        -------
        th.Tensor
            Embedding tensor of shape [batch_size, features_dim].
        """
        # [bs, window_size, *obs.shape]
        x = self.head(observations)
        # [bs, window_size, self._in_dim]
        x = self._encoder(x)
        # [bs, window_size, self._in_dim]
        x = x.mean(dim=1)
        # [bs, self._in_dim]
        x = self._encoder_compress(x)
        # [bs, features_dim]
        return x


class StateDecoder(nn.Module):
    """Transformer-based state decoder for reconstructing observation windows from embeddings.

    Decodes a fixed-size embedding vector into a sequence of observations using a transformer decoder.

    Parameters
    ----------
    observation_space : spaces.Space
        The observation space of the environment.
    features_dim : int, optional
        Dimension of the input embedding vector.
    window_size : int, optional
        Number of observations to reconstruct.
    n_head : int, optional
        Number of attention heads in the transformer decoder.
    n_layers : int, optional
        Number of transformer decoder layers.

    Methods
    -------
    forward(embedding: th.Tensor) -> th.Tensor
        Decode a batch of embedding vectors into observation windows.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 8,
        window_size: int = 10,
        n_head: int = 2,
        n_layers: int = 6,
    ):
        """Initialize the StateDecoder module.

        Parameters
        ----------
        observation_space : spaces.Space
            The observation space of the environment.
        features_dim : int, optional
            Dimension of the input embedding vector (default is 8).
        window_size : int, optional
            Number of observations to reconstruct (default is 10).
        n_head : int, optional
            Number of attention heads in the transformer decoder (default is 2).
        n_layers : int, optional
            Number of transformer decoder layers (default is 6).
        """
        super().__init__()

        self._window_size = window_size

        in_dim = int(np.prod(observation_space.shape[1:]))
        self._in_dim = max(n_head, 2 ** ((in_dim - 1).bit_length()))

        self._decoder_decompress = nn.Sequential(
            nn.Linear(features_dim, self._in_dim),
        )
        self._decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self._in_dim, n_head, batch_first=True),
            n_layers,
        )
        self.reconstruct = nn.Sequential(
            nn.Linear(self._in_dim, in_dim),
            nn.Unflatten(dim=2, unflattened_size=observation_space.shape[1:]),
        )

    def forward(self, embedding: th.Tensor) -> th.Tensor:
        """Decode a batch of embedding vectors into observation windows.

        Parameters
        ----------
        embedding : th.Tensor
            Tensor of shape [batch_size, features_dim].

        Returns
        -------
        th.Tensor
            Reconstructed observations of shape [batch_size, window_size, *obs.shape].
        """
        # [bs, features_dim]
        x = self._decoder_decompress(embedding)
        # [bs, self._in_dim]

        memory = x.unsqueeze(dim=1)
        tgt = th.zeros(
            size=(x.shape[0], self._window_size, self._in_dim), device=x.device
        )
        tgt = self._decoder(tgt, memory)

        # [bs, window_size, self._in_dim]
        tgt = self.reconstruct(tgt)
        # [bs, window_size, *obs.shape]
        return tgt
