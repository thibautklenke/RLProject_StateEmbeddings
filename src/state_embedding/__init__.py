from stable_baselines3 import DQN
import gymnasium as gym
from state_embedding.replay_buffer import ContextualizedReplayBuffer

def hello() -> None:
    env = gym.make("CartPole-v1", render_mode="human")

    dqn = DQN("MlpPolicy", env, replay_buffer_class=ContextualizedReplayBuffer)
    dqn.learn(total_timesteps=1000)
    print("Hello from hello-world!")
