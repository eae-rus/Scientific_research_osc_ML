import torch
import torch.nn as nn
from osc_tools.ml.models.base import BaseModel
from osc_tools.ml.layers.complex_ops import cLeakyReLU, cMaxPool1d

def create_conv_block(in_channels, out_channels, maxPool_size = 2, kernel_size=3, stride=1, padding=1, padding_mode="circular", useComplex=False):
    if useComplex:
        conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, dtype=torch.cfloat)
        return nn.Sequential(conv, cLeakyReLU(), cMaxPool1d(kernel_size=maxPool_size, stride=maxPool_size))
    conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
    return nn.Sequential(conv, nn.LeakyReLU(True), nn.MaxPool1d(kernel_size=maxPool_size, stride=maxPool_size))

class Conv_3(BaseModel):
    def __init__(self, useComplex=False):
        super(Conv_3, self).__init__()
        # TODO: Добавить задание параметров
        self.layer = nn.Sequential(
            create_conv_block(1, 16, useComplex=useComplex), # 32*16 -> 16*16
            create_conv_block(16, 32, useComplex=useComplex), # 16*32 -> 8*32
            create_conv_block(32, 32, maxPool_size = 8, useComplex=useComplex) # 8*32 -> 1*32
        )

    def forward(self, x):
        # FIXME: Не задаётся "devise" и от этого падает при расчёте.
        return self.layer(x)
