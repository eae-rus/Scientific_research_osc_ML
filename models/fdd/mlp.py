import torch
from torch import nn


class MLP(nn.Module):
    """
    Shallow MLP autoencoder for reconstruction.
    """
    def __init__(self, window_size: int, input_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        hidden_dim = window_size // 8 * input_dim
        self.model = nn.Sequential(
            nn.Linear(window_size * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.model(x)

        return x