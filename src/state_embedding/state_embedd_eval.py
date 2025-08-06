import torch.nn as nn
import torch as th
import gymnasium as gym


class StateEmbeddEvalModule(nn.Module):
    def __init__(self, in_features, out_features=1, hidden_size=64):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features),
        )

    # @override
    def forward(self, context: th.Tensor) -> th.Tensor:
        return self.head(context)
