import numpy as np
import torch as th
from gymnasium import Env, spaces
from typing import Any, Optional, Tuple, TypeVar

from state_embedding.embedding import StateEmbedding

ActType = TypeVar("ActType")


class ContextEnv(Env):
    """Environment wrapper that augments observations with a context window of previous states.

    This environment maintains a rolling window of the last `window_size` observations,
    stacking them along a new axis. The resulting observation is a tensor of shape
    (window_size, *original_observation_shape).

    Attributes
    ----------
    _env : Env
        The underlying environment instance.
    _window_size : int
        Number of previous observations to stack.
    _insert_idx : int
        Current index for inserting the next observation.
    _current_context : torch.Tensor
        Tensor holding the stacked context observations.
    action_space : gymnasium.Space
        The action space of the environment.
    observation_space : gymnasium.Space
        The observation space of the environment.
    metadata : dict
        Metadata for the environment.
    render_mode : str or None
        The rendering mode of the environment.
    spec : gymnasium.envs.registration.EnvSpec
        The environment specification.

    Methods
    -------
    step(action: ActType) -> Tuple[th.Tensor, dict[str, Any]]
        Take a step in the environment and update the context window.
    reset(seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[th.Tensor, dict[str, Any]]
        Reset the environment and context window.
    render() -> Any
        Render the current state of the environment.
    close() -> Any
        Close the environment and release resources.
    """

    def __init__(
        self,
        env: Env,
        window_size: int = 10,
    ) -> None:
        """Initialize the ContextEnv.

        Parameters
        ----------
        env : Env
            The environment to wrap.
        window_size : int, optional
            Number of previous observations to stack (default is 10).
        """
        super().__init__()

        self._env = env

        self.action_space = self._env.action_space

        self.metadata = self._env.metadata
        self.render_mode = self._env.render_mode
        self.spec = self._env.spec

        self._window_size = window_size

        low = np.stack([self._env.observation_space.low] * self._window_size)
        high = np.stack([self._env.observation_space.high] * self._window_size)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(self._window_size, *self._env.observation_space.shape),
        )

        self._insert_idx = 0
        self._current_context = th.zeros(*self.observation_space.shape)

    def step(self, action: ActType) -> Tuple[th.Tensor, dict[str, Any]]:
        """Take a step in the environment and update the context window.

        Parameters
        ----------
        action : ActType
            The action to take in the environment.

        Returns
        -------
        Tuple[th.Tensor, dict[str, Any]]
            The stacked context observation and the rest of the environment's step output.
        """
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

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[th.Tensor, dict[str, Any]]:
        """Reset the environment and context window.

        Parameters
        ----------
        seed : int, optional
            Random seed for environment reset.
        options : dict, optional
            Additional options for environment reset.

        Returns
        -------
        Tuple[th.Tensor, dict[str, Any]]
            The initial stacked context observation and the rest of the environment's reset output.
        """
        reset_result = self._env.reset(seed=seed, options=options)

        self._insert_idx = 0

        self._current_context = th.zeros(*self.observation_space.shape)
        self._current_context[0] = th.tensor(reset_result[0])

        return self._current_context, *reset_result[1:]

    def render(self) -> Any:
        """Render the current state of the environment.

        Returns
        -------
        Any
            Rendered output as defined by the underlying environment.
        """
        return self._env.render()

    def close(self) -> Any:
        """Close the environment and release resources.

        Returns
        -------
        Any
            Output of the underlying environment's close method.
        """
        return self._env.close()


class EmbeddingEnv(Env):
    """Environment wrapper that replaces observations with learned state embeddings.

    This environment uses a provided StateEmbedding module to encode the context window
    of observations into a fixed-size embedding vector, which is returned as the observation.

    Attributes
    ----------
    _env : ContextEnv
        The underlying context environment instance.
    _embedding_module : StateEmbedding
        The embedding module used to encode observations.
    action_space : gymnasium.Space
        The action space of the environment.
    observation_space : gymnasium.Space
        The observation space of the environment (embedding vector).
    metadata : dict
        Metadata for the environment.
    render_mode : str or None
        The rendering mode of the environment.
    spec : gymnasium.envs.registration.EnvSpec
        The environment specification.

    Methods
    -------
    step(action: Any) -> Tuple[np.ndarray, dict[str, Any]]
        Take a step in the environment and return the embedding of the context window.
    reset(seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict[str, Any]]
        Reset the environment and return the embedding of the initial context window.
    render() -> Any
        Render the current state of the environment.
    close() -> Any
        Close the environment and release resources.
    """

    def __init__(
        self, env: ContextEnv, embedding_module: StateEmbedding, window_size: int = 10
    ) -> None:
        """Initialize the EmbeddingEnv.

        Parameters
        ----------
        env : ContextEnv
            The context environment to wrap.
        embedding_module : StateEmbedding
            The embedding module to use for encoding observations.
        window_size : int, optional
            Number of previous observations to stack (default is 10).
        """
        super().__init__()

        self._env = env

        self.action_space = self._env.action_space

        self.metadata = self._env.metadata
        self.render_mode = self._env.render_mode
        self.spec = self._env.spec

        self._embedding_module = embedding_module

        self.observation_space = spaces.Box(
            0, 1, (self._embedding_module.features_dim,)
        )

    def step(self, action: Any) -> Tuple[np.ndarray, dict[str, Any]]:
        """Take a step in the environment and return the embedding of the context window.

        Parameters
        ----------
        action : Any
            The action to take in the environment.

        Returns
        -------
        Tuple[np.ndarray, dict[str, Any]]
            The embedding vector and the rest of the environment's step output.
        """
        step_result = self._env.step(action)
        # obs.shape = [window_size, *observation_space.shape]
        obs = step_result[0].unsqueeze(0)
        device = next(self._embedding_module.parameters()).device
        obs_tensor = obs.to(device)
        # obs.shape = [1, window_size, *observation_space.shape]
        embedding = self._embedding_module(obs_tensor).squeeze(0)

        return embedding.detach().cpu().numpy(), *step_result[1:]

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return the embedding of the initial context window.

        Parameters
        ----------
        seed : int, optional
            Random seed for environment reset.
        options : dict, optional
            Additional options for environment reset.

        Returns
        -------
        Tuple[np.ndarray, dict[str, Any]]
            The embedding vector and the rest of the environment's reset output.
        """
        reset_result = self._env.reset(seed=seed, options=options)
        device = next(self._embedding_module.parameters()).device
        obs_tensor = reset_result[0].unsqueeze(0).to(device)
        encoded_init = self._embedding_module(obs_tensor).squeeze(0)

        return encoded_init.detach().cpu().numpy(), *reset_result[1:]

    def render(self) -> Any:
        """Render the current state of the environment.

        Returns
        -------
        Any
            Rendered output as defined by the underlying environment.
        """
        return self._env.render()

    def close(self) -> Any:
        """Close the environment and release resources.

        Returns
        -------
        Any
            Output of the underlying environment's close method.
        """
        return self._env.close()
