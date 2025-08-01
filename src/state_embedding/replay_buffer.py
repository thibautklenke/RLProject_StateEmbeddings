from typing import Union

from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
import torch as th
from typing import Any


class ContextualizedReplayBuffer(ReplayBuffer):
    _context: list[list[np.ndarray]]

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

        self._context = list()

    def _resize_context(self, size: int):
        diff = size - len(self._context)
        if diff > 0:
            for _ in range(diff):
                self._context.append(list())
        elif diff < 0:
            for _ in range(-diff):
                self._context.pop(-1)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ):
        self._resize_context(obs.shape[0])

        n_envs = len(self._context)
        for i in range(n_envs):
            self._context[i].append(obs[i])
            if len(self._context[i]) > self._window_size:
                self._context[i].pop(0)

            infos[i]["context"] = self._context[i][:]  # Copy into infos

        for idx, terminate in enumerate(done):
            if terminate:
                self._context[idx] = list()

        super().add(obs, next_obs, action, reward, done, infos)
