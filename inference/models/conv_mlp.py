import torch.nn as nn
import torch


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


class CONV_NPU(nn.Module):
    def __init__(
        self, channel_num=5, hidden_size=160, output_size=3
    ):
        super(CONV_NPU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                channel_num,
                int(hidden_size / 2),
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                padding_mode="circular",
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(
                int(hidden_size / 2),
                hidden_size,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                padding_mode="circular",
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.enc_fc = nn.Linear(hidden_size * 8, hidden_size)
        self.out_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.enc_fc(x).relu()
        x = self.out_fc(x)
        return x


class CONV_FFT_MLP(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=3
    ):
        super(CONV_FFT_MLP, self).__init__()
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
        x_fft = self.fft_fc(x)
        return out, x_fft


class GRU(nn.Module):
    '''
    Gated recurrent unit
    '''    
    def __init__(self, num_sensors=5, num_states=4, hidden_dim=40, num_layers=2, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(num_sensors, hidden_dim, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim*num_layers, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, num_states)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Forward step
        '''
        h = self.gru(x)[1].permute(1, 0, 2)
        h = h.reshape(h.size(0), -1)
        linear_out = self.linear1(h).relu()
        linear_out = self.dropout(linear_out)
        out = self.linear2(linear_out)
        return out

class MLP(nn.Module):

    def __init__(
        self,
        frame_size: int,
        input_size: int = 160,
        hidden_size: int = 68,
        output_size: int = 3,
    ) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
