import torch as th
import torch.nn as nn


class EmbeddingEvalHead(nn.Module):
    """Simple MLP head for evaluating learned state embeddings.

    This module takes a state embedding as input and outputs a prediction (e.g., for reward, value, or classification).
    It consists of a single hidden layer with ReLU activation.

    Parameters
    ----------
    features_dim : int
        Dimension of the input embedding vector.
    out_features : int, optional
        Dimension of the output.
    hidden_size : int, optional
        Number of units in the hidden layer.

    Methods
    -------
    forward(context: th.Tensor) -> th.Tensor
        Compute the output prediction from the input embedding.
    """

    def __init__(self, features_dim: int, out_features: int = 1, hidden_size: int = 64):
        """Initialize the EmbeddingEvalHead.

        Parameters
        ----------
        features_dim : int
            Dimension of the input embedding vector.
        out_features : int, optional
            Dimension of the output (default is 1).
        hidden_size : int, optional
            Number of units in the hidden layer (default is 64).
        """
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(features_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features),
        )

    def forward(self, context: th.Tensor) -> th.Tensor:
        """Compute the output prediction from the input embedding.

        Parameters
        ----------
        context : th.Tensor
            Input embedding tensor of shape [batch_size, features_dim].

        Returns
        -------
        th.Tensor
            Output tensor of shape [batch_size, out_features].
        """
        return self.head(context)
