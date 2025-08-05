from typing import Any, Optional

import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from torch.nn import functional as F

from state_embedding.embedding import StateEmbedding, StateDecoder


class EmbeddingTrainingCallback(BaseCallback):
    """
    Callback class to train the embedding model.

    As we are using a callback here we can use any model to explore the world
    """

    def __init__(
        self,
        embedding_kwargs: Optional[dict[str, Any]] = None,
        optim_kwargs: Optional[dict[str, Any]] = None,
        verbose=0,
    ):
        super(EmbeddingTrainingCallback, self).__init__(verbose)

        self.embedding_kwargs = embedding_kwargs or {}
        self.optim_kwargs = optim_kwargs or {}

        self.window_size = self.embedding_kwargs.get("window_size", 5)

    def _on_training_start(self) -> None:
        self.embedd_module = StateEmbedding(
            observation_space=self.model.observation_space,
            **self.embedding_kwargs,
        )
        self.decoder = StateDecoder(
            observation_space=self.model.observation_space,
            **self.embedding_kwargs,
        )

        th.optim.Adam(self.embedd_module.parameters(), **self.optim_kwargs)

    def _on_step(self):
        new_obs = self.locals["new_obs"]

        obs_tensor = th.tensor(new_obs)

        encoded = self.embedd_module(obs_tensor)
        decoded = self.decoder(obs_tensor, encoded)

        # calculate mse loss
        loss = F.mse_loss(obs_tensor, decoded)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger.record("train/embedding_callback_loss", loss.item())

        return True
