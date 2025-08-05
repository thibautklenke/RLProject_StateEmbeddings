from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from torch.nn import functional as F
import torch as th
from torch import nn
import numpy as np
from gymnasium import spaces
from typing import Optional, Union, Any
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from state_embedding.replay_buffer import ContextualizedReplayBuffer


class DQNWithEmbedLoss(DQN):
    def __init__(
        self,
        policy: Union[str, type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
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
        window_size: int = 5,  # Embedding context
    ):
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {"window_size": window_size}
        else:
            replay_buffer_kwargs["window_size"] = window_size

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            ContextualizedReplayBuffer,
            replay_buffer_kwargs,
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

        self._window_size: int = window_size
        self._embedding_module = StateEmbedNetwork(
            observation_space=self.observation_space,
            window_size=window_size,
            embedding_size=8,
        )

    # Copied from super().train(), modified to include reconstruction loss of special state module
    # @override
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data, context, lengths = self.replay_buffer.sample_with_context(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]

            # [batch_size, window_size, *obs]
            context = th.flatten(context, start_dim=2)
            context = context.to(th.float32)
            # [batch_size, window_size, num_features]

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

            # Update embedding module
            decoded = self._embedding_module(context)
            loss_decoded = F.mse_loss(context, decoded)
            print(loss_decoded)

            # self.policy.optimizer.zero_grad()
            # loss_decoded.backward()

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            total_loss = loss_decoded + loss
            total_loss.backward()
            # loss.backward()
            # Clip gradient norm
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


class StateEmbedNetwork(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        n_head=8,
        n_layers_encoder=6,
        n_layers_decoder=6,
        window_size: int = 10,
        embedding_size=2**10,
    ):
        super().__init__()

        self._window_size = window_size  # Window size for contextual embedding
        num_features = np.prod(observation_space.shape)

        self._linear_encode = nn.Linear(num_features, embedding_size)
        self._linear_decode = nn.Linear(embedding_size, num_features)

        self._encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_size, n_head, batch_first=True),
            n_layers_encoder,
        )

        self._decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embedding_size, n_head, batch_first=True),
            n_layers_decoder,
        )

    # @override
    def forward(self, context: th.Tensor) -> th.Tensor:
        linear_encoded_context = self._linear_encode(context)

        encoded = self._encoder(linear_encoded_context)
        decoded = self._decoder(
            tgt=th.zeros_like(linear_encoded_context), memory=encoded
        )

        return self._linear_decode(decoded)

    def encode(self, context: th.Tensor) -> th.Tensor:
        return self._encoder(self._linear_encode(context))
