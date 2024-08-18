import sys
import os
import torch as torch
import torch.nn as nn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)
from kan_convolutional.KANLinear import KANLinear

# Требуется разобраться с базовой моделью KANLinear, у меня не работала норально.
# У них можно график строить заполениня и другие фичи
# from kan import *
# KAN

class CONV_MLP(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=4
    ):
        # TODO: разобраться в схеме обработки сигналов
        # TODO: Подумать о создании более сложной схемы (2D свёртки, чтобы одновременно и I и U)
        super(CONV_MLP, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                channel_num,
                int(hidden_size),
                kernel_size=32,
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
            nn.Linear(hidden_size*17, hidden_size//2), # nn.Linear(hidden_size * 7, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size//2, output_size),
        )

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = x.view(x.size(0), x.size(1) * x.size(2))
        x = self.fc(x)
        return x

class KAN_firrst(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=4
    ):
        # пример взят у https://github.com/AntonioTepsich/Convolutional-KANs.git
        # https://arxiv.org/pdf/2406.13155
        super(KAN_firrst, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                channel_num,
                int(hidden_size),
                kernel_size=32,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # TODO: Расписать # KANLayer
        conv_output_size = (frame_size - 30) // 2  # Предполагаем, что размер фрейма уменьшается после свёртки и пулинга
        self.kan = KANLinear(in_features=conv_output_size * hidden_size,
                             out_features=output_size,
                             grid_size=10,
                             spline_order=3,
                             scale_noise=0.01,
                             scale_base=1,
                             scale_spline=1,
                             base_activation=nn.SiLU,
                             grid_eps=0.02,
                             grid_range=[0,1])

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = x.view(x.size(0), -1)
        x = self.kan(x)
        return x


if __name__ == "__main__":
    print(CONV_MLP())
