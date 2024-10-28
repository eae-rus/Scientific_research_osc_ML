import torch.nn as nn


class CONV_AE_MLP(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=3
    ):
        super(CONV_AE_MLP, self).__init__()
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
        self.dec_fc = nn.Linear(hidden_size, hidden_size * 8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(
                40,
                20,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(True),
            nn.ConvTranspose1d(
                20,
                5,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Linear(32, 32),
        )

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = x.view(x.size(0), -1)
        x = self.enc_fc(x).relu()
        out = self.out_fc(x)
        x = self.dec_fc(x).relu()
        x = x.view(x.size(0), 40, 8)
        x = self.deconv(x)
        x = x.permute(0, 2, 1)
        return out, x


class CONV_AE(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=3
    ):
        super(CONV_AE, self).__init__()
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
        self.dec_fc = nn.Linear(hidden_size, hidden_size * 8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(
                40,
                20,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(True),
            nn.ConvTranspose1d(
                20,
                5,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Linear(32, 32),
        )

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = x.view(x.size(0), -1)
        x = self.enc_fc(x).relu()
        x = self.dec_fc(x).relu()
        x = x.view(x.size(0), 40, 8)
        x = self.deconv(x)
        x = x.permute(0, 2, 1)
        return x

    def latent(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = x.view(x.size(0), -1)
        x = self.enc_fc(x)
        return x


class MLP(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=3
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #x = x.relu()
        #x = self.fc1(x).relu()
        #x = self.fc2(x).relu()
        x = self.fc3(x)
        return x
