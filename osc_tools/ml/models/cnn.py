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


class ResBlock1D(nn.Module):
    """
    Базовый блок ResNet для 1D сигналов.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=None):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(BaseModel):
    """
    ResNet для 1D временных рядов.
    Адаптировано из классической архитектуры ResNet.
    """
    def __init__(self, in_channels, num_classes, layers=[2, 2, 2, 2], base_filters=64):
        super(ResNet1D, self).__init__()
        self.inplanes = base_filters
        
        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(base_filters, layers[0])
        self.layer2 = self._make_layer(base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], stride=2)

        # Final classification
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters * 8, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

        layers = []
        layers.append(ResBlock1D(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResBlock1D(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
