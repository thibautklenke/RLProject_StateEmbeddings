from typing import Any, Optional, Callable

import torch as th
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from torch.nn import functional as F

from state_embedding.embedding import StateEmbedding, StateDecoder
from state_embedding.embedding_eval import EmbeddingEvalHead


class EmbeddingTrainingCallback(BaseCallback):
    """Callback class to train the embedding model during RL training.

    This callback trains a StateEmbedding and StateDecoder module using reconstruction loss
    at each step. It also periodically evaluates the learned embedding using a linear probe
    (EmbeddingEvalHead) on reward prediction.

    Parameters
    ----------
    embedding_kwargs : dict[str, Any], optional
        Keyword arguments for the embedding network.
    optim_kwargs : dict[str, Any], optional
        Keyword arguments for the optimizer.
    verbose : int, optional
        Verbosity level (default is 0).

    Methods
    -------
    _on_training_start() -> None
        Initialize embedding and decoder modules and optimizer at training start.
    _on_step() -> bool
        Perform one step of embedding/decoder training and periodic evaluation.
    """

    def __init__(
        self,
        embedding_kwargs: Optional[dict[str, Any]] = None,
        optim_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
    ) -> None:
        """Initialize the EmbeddingTrainingCallback.

        Parameters
        ----------
        embedding_kwargs : dict[str, Any], optional
            Keyword arguments for the embedding network.
        optim_kwargs : dict[str, Any], optional
            Keyword arguments for the optimizer.
        verbose : int, optional
            Verbosity level (default is 0).
        """
        super(EmbeddingTrainingCallback, self).__init__(verbose)

        self.embedding_kwargs = embedding_kwargs or {}
        self.optim_kwargs = optim_kwargs or {}

        self.features_dim = self.embedding_kwargs.get("features_dim", 8)
        self.window_size = self.embedding_kwargs.get("window_size", 10)

    def _on_training_start(self) -> None:
        """
        Initialize embedding and decoder modules and optimizer at training start.
        """
        self.embedd_module = StateEmbedding(
            observation_space=self.model.observation_space,
            **self.embedding_kwargs,
        )
        self.decoder = StateDecoder(
            observation_space=self.model.observation_space,
            **self.embedding_kwargs,
        )

        th.optim.Adam(self.embedd_module.parameters(), **self.optim_kwargs)

    def _on_step(self) -> bool:
        """Perform one step of embedding/decoder training and periodic evaluation.

        Returns
        -------
        bool
            Whether training should continue (always True).
        """
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

            self.logger.record("eval/Total_Loss", total_loss / 500)

        return True


class EveryNSteps(EventCallback):
    """Trigger a callback every `n_steps` timesteps.

    Parameters
    ----------
    n_steps : int
        Number of timesteps between two triggers.
    callback : Callable
        The callback function to trigger.

    Methods
    -------
    _on_step() -> bool
        Trigger the callback if the required number of steps has passed.
    """

    def __init__(self, n_steps: int, callback: Callable) -> None:
        """Initialize the EveryNSteps callback.

        Parameters
        ----------
        n_steps : int
            Number of timesteps between two triggers.
        callback : callable
            The callback function to trigger.
        """
        super(EveryNSteps, self).__init__()
        self.n_steps = n_steps
        self.last_time_trigger = 0
        self._custom_callback = callback

    def _on_step(self) -> bool:
        """Trigger the callback if the required number of steps has passed.

        Returns
        -------
        bool
            Whether training should continue (always True).
        """
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            self._custom_callback()
        return True
