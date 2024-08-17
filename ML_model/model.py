import torch as torch
import torch.nn as nn


class CONV_MLP(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=4
    ):
        super(CONV_MLP, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                channel_num,
                int(hidden_size / 2),
                kernel_size=32,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                int(hidden_size / 2),
                hidden_size,
                kernel_size=6,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # TODO: исправить расчётывание выходного размера после свёрток
        # пока что задаю принудительно 128*16
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*7, hidden_size), # nn.Linear(hidden_size * 8, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),  # Add the sigmoid activation function
        )

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = x.view(x.size(0), x.size(1) * x.size(2))
        x = self.fc(x)
        return x


if __name__ == "__main__":
    print(CONV_MLP())
