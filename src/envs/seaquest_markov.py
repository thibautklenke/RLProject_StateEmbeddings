from gymnasium import Env
from typing import TypeVar, Any, Optional, Tuple


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class SeaquestNoMarkov(Env):
    """A version of the MinAtar Seaquest environment that does not have the oxygen bar.

    This environment wraps the standard MinAtar Seaquest environment and masks out the oxygen bar
    from the observation, making the environment partially observable. All other environment
    dynamics and interfaces remain unchanged.

    Attributes
    ----------
    _env : Env
        The underlying MinAtar Seaquest environment instance.
    action_space : gymnasium.Space
        The action space of the environment.
    observation_space : gymnasium.Space
        The observation space of the environment.
    metadata : dict
        Metadata for the environment, such as supported render modes.
    render_mode : str or None
        The rendering mode of the environment.
    spec : gymnasium.envs.registration.EnvSpec
        The environment specification.

    Methods
    -------
    step(action: ActType) -> Tuple[ObsType, dict[str, Any]]
        Take a step in the environment, masking out the oxygen bar in the observation.
    reset(seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict[str, Any]]
        Reset the environment to an initial state.
    render() -> Any
        Render the current state of the environment.
    close() -> None
        Close the environment and release any resources.
    """

    def __init__(self, env: Env) -> None:
        """Initializes the evironment.

        Parameters
        ----------
        env : Env
            Instance of MinAtar Seaquest to wrap.

        Raises
        ------
        ValueError
            If the provided environment is not a MinAtar Seaquest environment.
        """
        super().__init__()

        self._env: Env = env

        if not self._env.spec.id.startswith("MinAtar/Seaquest-v"):
            raise ValueError("Environment must be MinAtar/Seaquest")

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self.metadata = self._env.metadata
        self.render_mode = self._env.render_mode
        self.spec = self._env.spec

    def step(self, action: ActType) -> Tuple[ObsType, dict[str, Any]]:
        """Take one step in the environment.

        Parameters
        ----------
        action : ActType
            The action to take in the environment.

        Returns
        -------
        tuple[ObsType, dict[str, Any]]
            The observation (with the oxygen bar removed) and additional info from the environment.
        """
        step_result = self._env.step(action)
        obs = step_result[0]

        # Remove oxygen bar
        obs[9, :, 7] = 0

        return step_result

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict[str, Any]]:
        """Reset the environment to an initial state.

        Parameters
        ----------
        seed : int, optional
            The random seed to use for the environment reset.
        options : dict, optional
            Additional options for the environment reset.

        Returns
        -------
        tuple[ObsType, dict[str, Any]]
            The initial observation and info as returned by the underlying environment.
        """
        return self._env.reset(seed=seed, options=options)

    def render(self) -> Any:
        """Render the current state of the environment.

        Returns
        -------
        Any
            Rendered output as defined by the underlying environment.
        """
        return self._env.render()

    def close(self) -> None:
        """Close the environment and release any resources."""
        return self._env.close()
