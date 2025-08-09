import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common import type_aliases
from stable_baselines3.common.callbacks import BaseCallback
from torch.nn import functional as F

from state_embedding.embedding import StateEmbedding
from state_embedding.env import ContextEnv


class EmbeddingEvalHead(nn.Module):
    def __init__(self, features_dim, out_features=1, hidden_size=64):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(features_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features),
        )

    # @override
    def forward(self, context: th.Tensor) -> th.Tensor:
        return self.head(context)


class EmbeddingEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        total_steps: int = 1000,
        eval_freq: int = 10000,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.total_steps = total_steps
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.best_mean_loss = 99999999999.0

    def _init_callback(self) -> None:
        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Extract current Embedding in use and create environment where evaluation is happening
            embedding = self.model.q_net.features_extractor
            window_size = embedding.window_size
            env = ContextEnv(env=self.eval_env, window_size=window_size)

            # Evaluate!
            # TODO: hidden size as parameter
            mean_loss, std_loss = evaluate_embedding(
                self.model,
                env,
                embedding=embedding,
                total_steps=self.total_steps,
                deterministic=self.deterministic,
            )

            # Add to current Logger
            self.logger.record("eval/mean_loss", mean_loss)
            self.logger.record("eval/std_loss", std_loss)

            # Save best model
            if mean_loss < self.best_mean_loss:
                if self.best_model_save_path is not None:
                    th.save(
                        embedding,
                        os.path.join(self.best_model_save_path, "best_model.pth"),
                    )

                self.best_mean_loss = float(mean_loss)

        return True


def evaluate_embedding(
    model: "type_aliases.PolicyPredictor",
    env: ContextEnv,
    embedding: StateEmbedding,
    total_steps: int = 1000,
    deterministic: bool = True,
    hidden_size: int = 64,
):
    eval_head = EmbeddingEvalHead(
        embedding.features_dim, out_features=1, hidden_size=hidden_size
    )
    # FIXME: parameter
    optim = th.optim.Adam(eval_head.parameters())

    # FIXME: seeding

    # FIXME: vector envs?

    def train(loss):
        optim.zero_grad()
        loss.backward()
        optim.step()

    embedding.eval()
    eval_head.train(True)
    _step_in_env(
        model,
        env,
        embedding,
        train,
        eval_head,
        total_steps=total_steps,
        deterministic=True,
    )

    losses = []

    def eval(loss):
        losses.append(loss.item())

    # FIXME: config
    eval_head.eval()
    _step_in_env(
        model,
        env,
        embedding,
        eval,
        eval_head,
        total_steps=total_steps,
        deterministic=True,
    )

    losses = np.array(losses)

    embedding.train(True)
    return losses.mean(), losses.std()


def _step_in_env(
    model: "type_aliases.PolicyPredictor",
    env: ContextEnv,
    embedding: StateEmbedding,
    loss_callback,
    eval_head: EmbeddingEvalHead,
    total_steps: int = 1000,
    deterministic: bool = True,
):
    obs, _ = env.reset()
    for _ in range(total_steps):
        actions, states = model.predict(obs, deterministic=deterministic)
        new_obs, rewards, terminated, truncated, infos = env.step(actions)

        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = new_obs

        with th.no_grad():
            embed = embedding(new_obs.unsqueeze(0)).squeeze(0)

        predicted_rewards = eval_head(embed)
        loss = F.mse_loss(predicted_rewards, th.tensor(rewards).unsqueeze(0))
        loss_callback(loss)
