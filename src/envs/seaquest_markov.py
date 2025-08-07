from gymnasium import Env


class SeaquestNoMarkov(Env):
    def __init__(self, env: Env):
        super().__init__()

        self._env = env

        if not self._env.spec.id.startswith("MinAtar/Seaquest-v"):
            raise ValueError("Environment must be MinAtar/Seaquest")

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self.metadata = self._env.metadata
        self.render_mode = self._env.render_mode
        self.spec = self._env.spec

    def step(self, action):
        step_result = self._env.step(action)
        obs = step_result[0]

        # Remove oxygen bar
        obs[9, :, 7] = 0

        return step_result

    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()
