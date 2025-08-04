import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback

def call_eval_loop() -> None:

    # Parameter
    env_name = "CartPole-v1"
    SEED = 0

    # create the main variables
    env = gym.make(env_name)
    env.reset(seed=SEED)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/MlpPolicy/")

    # Separate evaluation env
    eval_env = gym.make(env_name)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                                 log_path="./logs/", eval_freq=500,
                                 deterministic=True, render=False)

    # run the model with the callbacks
    model.learn(total_timesteps=10000, callback=[ProgressBarCallback(), eval_callback])

def graphs() -> None:
    print("Done")


if __name__ == "__main__":
    call_eval_loop()