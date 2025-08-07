import torch as th
import torch.nn as nn


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
