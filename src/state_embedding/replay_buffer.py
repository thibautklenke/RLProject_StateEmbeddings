from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
from typing import Any

class ContextualizedReplayBuffer(ReplayBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ):
        # Here, we can use obs

        print(obs, done)

        super().add(obs, next_obs, action, reward, done, infos)