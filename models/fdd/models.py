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


class CNN(nn.Module):
    def __init__(
        self, window_size: int, input_dim: int, output_dim: int
    ):
        super(CNN, self).__init__()
        hidden_size = 40
        self.conv = nn.Sequential(
            nn.Conv1d(
                input_dim,
                int(hidden_size / 2),
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                int(hidden_size / 2),
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.enc_fc = nn.Linear(320, hidden_size)
        self.out_fc = nn.Linear(hidden_size, output_dim)
        self.fft_fc = nn.Linear(hidden_size, 5)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = x.view(x.size(0), -1)
        x = self.enc_fc(x).relu()
        out = self.out_fc(x)
        return out