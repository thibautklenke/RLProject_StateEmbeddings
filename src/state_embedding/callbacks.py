from typing import Any, Optional

import torch as th
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from torch.nn import functional as F

from state_embedding.embedding import StateEmbedding, StateDecoder
from state_embedding.embedding_eval import EmbeddingEvalHead


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

        self.features_dim = self.embedding_kwargs.get("features_dim", 8)
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

        if self.n_calls % 500 == 0:
            # variables needed
            replay_buffer = self.locals["replay_buffer"]
            head = EmbeddingEvalHead(
                features_dim=self.features_dim * self.window_size, out_features=1
            )
            # FIXME: broken
            # eval_optimizer = th.optim.Adam(head.parameters(), lr=1e-3)

            total_loss = 0
            # train the head on a embeddings
            for i in range(500):
                sample = replay_buffer.sample(1)
                obs = sample.observations
                reward = sample.rewards

                with th.no_grad():
                    encoded_obs = self.embedd_module(obs).flatten(start_dim=1)

                # embedd the observation and calculate a reward with the linear probe/head
                _reward = head(encoded_obs)

                # Loss between real reward and head
                loss = F.mse_loss(reward, _reward)
                loss.backward()
                total_loss += loss.item()

                # print(f"Eval_loss: {loss.item()}")

            self.logger.record("eval/Total_Loss", total_loss / 500)

        return True

class EveryNSteps(EventCallback):
    """
    Trigger a callback every `n_steps` timesteps
    :param n_steps: (int) Number of timesteps between two trigger.
    """
    def __init__(self, n_steps: int, callback):
        super(EveryNSteps, self).__init__()
        self.n_steps = n_steps
        self.last_time_trigger = 0
        self._custom_callback = callback

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            self._custom_callback()
        return True