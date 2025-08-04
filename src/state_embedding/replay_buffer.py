from typing import Union

from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import ReplayBufferSamples
import numpy as np
import torch as th
from typing import Any, Tuple, Optional


class ContextualizedReplayBuffer(ReplayBuffer):
    # [window_size, n_envs, *obs_shape]
    _context_tracker: np.ndarray
    # [n_envs]
    _context_index: np.ndarray

    # [buffer_size, window_size, n_envs, *obs_shape]
    contexts: np.ndarray
    # [buffer_size, n_envs]
    contexts_length: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        window_size: int = 2,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )

        self._window_size: int = window_size

        self._context_tracker = np.zeros(
            (self._window_size, self.n_envs, *self.obs_shape),
            dtype=observation_space.dtype,
        )
        self._context_index = np.zeros((self.n_envs,), dtype=np.int64)

        self.contexts = np.zeros(
            (self.buffer_size, self._window_size, self.n_envs, *self.obs_shape),
            dtype=observation_space.dtype,
        )
        self.contexts_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ):
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Write current observations to context tracker
        for env in range(self.n_envs):
            idx = self._context_index[env]
            self._context_tracker[idx, env, :] = obs[env, :]

        # Put current context into the replay buffer
        self.contexts[self.pos, :] = self._context_tracker
        self.contexts_length[self.pos, :] = self._context_index + 1

        # Increment window index for each environment
        for env in range(self.n_envs):
            # If the index is at the end of the window, shift all contexts
            if self._context_index[env] == self._window_size - 1:
                for idx in range(self._window_size - 1):
                    self._context_tracker[idx, env, :] = self._context_tracker[
                        idx + 1, env, :
                    ]
            else:
                self._context_index[env] += 1

        # If environment is done, reset the context tracker for that environment
        for idx, terminate in enumerate(done):
            if terminate:
                self._context_tracker[:, idx, :] = 0
                self._context_index[idx] = 0

        super().add(obs, next_obs, action, reward, done, infos)

    def sample_with_context(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> Tuple[ReplayBufferSamples, th.Tensor, th.Tensor]:
        # Yes, we are smart
        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            return self._get_samples_with_context(batch_inds, env=env)

        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples_with_context(batch_inds, env=env)

    def _get_samples_with_context(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Tuple[ReplayBufferSamples, th.Tensor, th.Tensor]:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env
            ),
        )
        contexts = self.contexts[batch_inds, :, env_indices, :]
        contexts_length = self.contexts_length[batch_inds, env_indices]
        return (
            ReplayBufferSamples(*tuple(map(self.to_torch, data))),
            self.to_torch(contexts),
            self.to_torch(contexts_length),
        )
