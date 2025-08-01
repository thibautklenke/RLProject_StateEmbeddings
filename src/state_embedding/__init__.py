import gymnasium as gym

from state_embedding.embed import DQNWithEmbedLoss


def hello() -> None:
    env = gym.make("CartPole-v1", render_mode="human")

    dqn = DQNWithEmbedLoss("MlpPolicy", env)
    dqn.learn(total_timesteps=1000)
    print("Hello from hello-world!")


hello()
