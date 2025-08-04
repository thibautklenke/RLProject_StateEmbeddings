import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback

def call_eval_loop() -> None:

    envs = ["CartPole-v1"]
    seeds = np.random.randint(low=0, high=10000, size=100)
    models = ["DQN"]

    # Parameter
    for i in range(100):
        ENV_NAME = np.random.choice(envs)
        SEED = int(np.random.choice(seeds))
        MODEL_NAME = np.random.choice(models)

        eval_loop(MODEL_NAME, ENV_NAME, SEED)

def eval_loop(model_name, env_name, seed)->None:

    # create the main variables
    env = gym.make(env_name)
    env.reset(seed=seed)

    if model_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/MlpPolicy/")
    elif model_name == "DQN":
        model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./logs/MlpPolicy/", learning_rate=1e-3)

    # Separate evaluation env
    eval_env = gym.make(env_name)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env,  eval_freq=500,
                                 deterministic=True, render=False)

    # run the model with the callbacks
    model.learn(total_timesteps=10000,
                callback=[ProgressBarCallback(), eval_callback],
                tb_log_name=f"{model_name}-{env_name}-{seed}",)


if __name__ == "__main__":
    call_eval_loop()