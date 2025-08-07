import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class StateEmbedding(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 8,
        window_size: int = 5,
        n_head: int = 2,
        n_layers: int = 6,
    ):
        super().__init__(observation_space, features_dim)

        self._window_size = window_size

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
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 8,
        window_size: int = 5,
        n_head: int = 2,
        n_layers: int = 6,
    ):
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
        # [bs, features_dim]
        x = self._decoder_decompress(embedding)
        # [bs, self._in_dim]

        # Use greedy decoding to reconstruct sequence
        memory = x.unsqueeze(dim=1)
        token_shape = (x.shape[0], 1, self._in_dim)
        tgt = th.zeros(shape=token_shape)
        # [bs, 1, self._in_dim]
        for i in range(self._window_size):
            recon = self._decoder(tgt, memory)
            # [bs, i, self._in_dim]
            tgt[:, i, :] = recon[:, i, :]

            if i < self._window_size - 1:
                # Prepare next token
                tgt = th.cat((tgt, th.zeros(shape=token_shape)), dim=1)

        # [bs, window_size, self._in_dim]
        tgt = self.reconstruct(tgt)
        # [bs, window_size, *obs.shape]
        return tgt
