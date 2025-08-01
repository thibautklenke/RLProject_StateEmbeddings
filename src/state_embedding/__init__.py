from stable_baselines3 import DQN
import gymnasium as gym


def hello() -> None:
    env = gym.make("CartPole-v1", render_mode="human")

    dqn = DQN("MlpPolicy", env)
    dqn.learn(total_timesteps=1000)
    print("Hello from hello-world!")
