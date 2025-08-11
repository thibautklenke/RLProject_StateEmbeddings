from typing import Any, Optional, Union

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.dqn import DQN
from torch.nn import functional as F

from .qnetwork import EmbeddingPolicy


class DQNWithReconstruction(DQN):
    """DQN agent with an additional reconstruction loss for state embeddings.

    This class extends the standard DQN algorithm to jointly optimize for Q-learning and
    state reconstruction. It uses an EmbeddingPolicy with a StateEmbedding feature extractor
    and a StateDecoder for reconstructing the input observation from the embedding.

    Parameters
    ----------
    env : Union[GymEnv, str]
        The environment to learn from (can be string for gym.make or a GymEnv).
    learning_rate : Union[float, Schedule], optional
        Learning rate for the optimizer or a schedule function (default is 1e-4).
    buffer_size : int, optional
        Size of the replay buffer (default is 1_000_000).
    learning_starts : int, optional
        Number of steps before learning starts (default is 100).
    batch_size : int, optional
        Minibatch size for each gradient update (default is 32).
    tau : float, optional
        Soft update coefficient for the target network (default is 1.0).
    gamma : float, optional
        Discount factor (default is 0.99).
    train_freq : Union[int, tuple[int, str]], optional
        Update the model every `train_freq` steps (default is 4).
    gradient_steps : int, optional
        How many gradient steps to do after each rollout (default is 1).
    replay_buffer_class : type[ReplayBuffer], optional
        Replay buffer class to use.
    replay_buffer_kwargs : dict[str, Any], optional
        Keyword arguments for the replay buffer.
    optimize_memory_usage : bool, optional
        Enable a memory efficient variant of the replay buffer (default is False).
    target_update_interval : int, optional
        Update the target network every `target_update_interval` steps (default is 10000).
    exploration_fraction : float, optional
        Fraction of entire training period over which the exploration rate is reduced (default is 0.1).
    exploration_initial_eps : float, optional
        Initial value of random action probability (default is 1.0).
    exploration_final_eps : float, optional
        Final value of random action probability (default is 0.05).
    max_grad_norm : float, optional
        Maximum gradient norm for clipping (default is 10).
    stats_window_size : int, optional
        Window size for logging statistics (default is 100).
    tensorboard_log : str, optional
        Path for tensorboard logs.
    policy_kwargs : dict[str, Any], optional
        Additional keyword arguments for the policy.
    verbose : int, optional
        Verbosity level (default is 0).
    seed : int, optional
        Random seed.
    device : Union[th.device, str], optional
        Device for training ("cpu", "cuda", or torch.device).
    _init_setup_model : bool, optional
        Whether to build the network at initialization (default is True).
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        """Initialize the DQNWithReconstruction agent.

        Parameters
        ----------
        (see class docstring for parameter details)
        """
        super().__init__(
            EmbeddingPolicy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs={
                "use_reconstruction_loss": True,
            }
            | (policy_kwargs or {}),
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Perform a training update for the DQN agent, including reconstruction loss.

        Parameters
        ----------
        gradient_steps : int
            Number of gradient steps to perform.
        batch_size : int, optional
            Minibatch size for each gradient update (default is 100).

        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        reconstruction_losses = []
        for _ in range(gradient_steps):
            # Sample a batch from the replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]

            device = self.device
            replay_data = replay_data._replace(
                observations=replay_data.observations.to(device),
                next_observations=replay_data.next_observations.to(device),
                actions=replay_data.actions.to(device),
                rewards=replay_data.rewards.to(device),
                dones=replay_data.dones.to(device),
            )

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates and reconstruction loss
            current_q_values, reconstruction_loss = self.q_net.combined_forward(
                replay_data.observations
            )

            reconstruction_loss = reconstruction_loss.to(device)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            reconstruction_losses.append(reconstruction_loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            total_loss = loss + reconstruction_loss
            total_loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        # Logging
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/reconstruction_loss", np.mean(reconstruction_losses))
