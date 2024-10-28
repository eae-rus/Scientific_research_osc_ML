import torch.nn as nn


class CONV_MLP(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=3
    ):
        super(CONV_MLP, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                channel_num,
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

        self.enc_fc = nn.Linear(hidden_size * 8, hidden_size)
        self.out_fc = nn.Linear(hidden_size, output_size)
        self.fft_fc = nn.Linear(hidden_size, 5)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = x.view(x.size(0), -1)
        x = self.enc_fc(x).relu()
        out = self.out_fc(x)
        #x_fft = self.fft_fc(x)
        return out#, x_fft