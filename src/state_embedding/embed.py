from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import QNetwork
from torch.nn import functional as F
from stable_baselines3.common.type_aliases import PyTorchObs
import torch as th
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Optional

class DQNWithEmbedLoss(DQN):
    # Copied from super().train(), modified to include reconstruction loss of special state module
    @override
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            # TODO: intercept state embedding from q_net

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            # TODO: compute reconstruction loss, if state module is in use

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


class StateEmbedNetwork(QNetwork):

    # TODO: introduce flag whether network uses "vanilla" or "custom" behaviour
    #   Needed because we want to introduce a second loss, but
    #   also to be able to insert this special state module into normal DQN

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True
    ):
        super().__init__( 
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch,
            activation_fn,
            normalize_images
        )

        self._custom = False

    @override
    def forward(self, obs: PyTorchObs) -> th.Tensor:
        # TODO: maybe use own feature extractor
        z = self.extract_features(obs, self.features_extractor)

        # TODO: put z through decoder (e.g. "unextract_features")
        #   to compute loss

        return self.q_net(z)