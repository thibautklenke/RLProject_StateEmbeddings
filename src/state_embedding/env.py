from gymnasium import Env
import torch as th
from gymnasium import spaces
from state_embedding.embed import StateEmbedNetwork
import numpy as np

class ContextualizedEnv(Env):
    def __init__(
        self,
        env: Env,
        embedding_module: StateEmbedNetwork = None,
        window_size: int = 5,
    ) -> None:
        super().__init__()

        self._env = env

        self.action_space = self._env.action_space

        self.metadata = self._env.metadata
        self.render_mode = self._env.render_mode
        self.spec = self._env.spec

        self._embedding_module = embedding_module

        self.observation_space = spaces.Box(
            0, 1, (self._embedding_module._embedding_size,)
        )

        self._window_size = window_size
        self._current_context = th.zeros(window_size, *self.observation_space.shape)
        self._insert_idx = 0

    def step(self, action):
        step_result = self._env.step(action)
        obs = th.tensor(step_result[0])

        # Append, no roll
        if self._insert_idx < self._window_size:
            self._current_context[self._insert_idx] = obs
            self._insert_idx += 1
        else:  # Roll all elements up by one, insert at last position
            self._current_context = th.roll(self._current_context, shifts=-1, dims=0)
            self._current_context[-1] = obs

        if self._embedding_module is not None:
            embedding = self._embedding_module.encode(self._current_context)
        else:
            embedding = obs

        return embedding, *step_result[1:]

    def reset(self, seed=None, options=None):
        reset_result = self._env.reset(seed=seed, options=options)

        # Here, we initialize the context with zeros
        # Alternatively, we could use window_size-many copies of the
        # initial state as the initial context:
        # obs = th.tensor(reset_result[0])
        # self._current_context = th.stack([obs] * self._window_size)

        self._current_context = th.zeros_like(self._current_context)
        self._insert_idx = 0

        return reset_result

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


class ContextEnv(ContextualizedEnv):
    """Env to be used with context as states, i.e. (window_size, *obs.shape)"""

    def __init__(
        self,
        env: Env,
        embedding_module: StateEmbedNetwork = None,
        window_size: int = 5,
    ) -> None:
        super().__init__(env, embedding_module, window_size)

        low = np.stack([self._env.observation_space.low] * self._window_size)
        high = np.stack([self._env.observation_space.high] * self._window_size)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(self._window_size, *self._env.observation_space.shape),
        )

        self._current_context = th.zeros(window_size, *self.observation_space.shape)
    

    def step(self, action):
        step_result = self._env.step(action)
        obs = th.tensor(step_result[0])

        # Append, no roll
        if self._insert_idx < self._window_size:
            self._current_context[self._insert_idx] = obs
            self._insert_idx += 1
        else:  # Roll all elements up by one, insert at last position
            self._current_context = th.roll(self._current_context, shifts=-1, dims=0)
            self._current_context[-1] = obs

        return self._current_context, *step_result[1:]


class EmbeddingEnv(ContextualizedEnv):
    """Env to be used with embeddings as states, i.e. (embedding_size, )"""

    def __init__(
        self,
        env: Env,
        embedding_module: StateEmbedNetwork = None,
        window_size: int = 5,
    ) -> None:
        super().__init__(env, embedding_module, window_size)

        self.observation_space = spaces.Box(
            0, 1, (self._embedding_module._embedding_size,)
        )

        self._current_context = th.zeros(window_size, *self.observation_space.shape)
