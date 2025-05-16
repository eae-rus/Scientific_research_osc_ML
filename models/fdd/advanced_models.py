import torch
from torch import nn
import torch.nn.functional as F
import math


class LSTM_CNN_Hybrid(nn.Module):
    """
    Гибридная модель LSTM+CNN для выявления временных и частотных паттернов
    """

    def __init__(self, window_size: int, input_dim: int, output_dim: int):
        super(LSTM_CNN_Hybrid, self).__init__()

        # CNN ветка для извлечения локальных паттернов
        self.conv_branch = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)  # Фиксированный размер вместо window_size // 4
        )

        # LSTM ветка для временных зависимостей
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(0.3)

        # Вычисляем правильные размерности
        cnn_features = 256 * 16  # 256 каналов * 16 временных точек
        lstm_features = 256  # 128 * 2 (bidirectional)

        # Объединение признаков
        self.feature_fusion = nn.Linear(cnn_features + lstm_features, 512)
        self.final_layers = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # CNN ветка: (batch, time, features) -> (batch, features, time)
        cnn_out = self.conv_branch(x.permute(0, 2, 1))
        cnn_out = cnn_out.view(batch_size, -1)

        # LSTM ветка
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out[:, -1, :])  # Берем последний выход

        # Объединение
        combined = torch.cat([cnn_out, lstm_out], dim=1)
        fused = self.feature_fusion(combined)

        return self.final_layers(fused)


class TransformerFDD(nn.Module):
    """
    Transformer модель с вниманием для выявления сложных паттернов
    """

    def __init__(self, window_size: int, input_dim: int, output_dim: int):
        super(TransformerFDD, self).__init__()
        self.model_dim = 128
        self.input_dim = input_dim

        # Проекция входных данных
        self.input_projection = nn.Linear(input_dim, self.model_dim)
        self.pos_encoding = PositionalEncoding(self.model_dim, window_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.model_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # Проекция и позиционное кодирование
        x = self.input_projection(x)
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x)

        # Классификация
        x = x.permute(0, 2, 1)  # Для AdaptiveAvgPool1d
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class ResNet1D(nn.Module):
    """
    1D ResNet для временных рядов с остаточными связями
    """

    def __init__(self, window_size: int, input_dim: int, output_dim: int):
        super(ResNet1D, self).__init__()

        self.init_conv = nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.init_bn = nn.BatchNorm1d(64)
        self.init_relu = nn.ReLU()
        self.init_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Остаточные блоки
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Глобальный пулинг и классификатор
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)

        # Attention механизм для важных признаков
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(512, output_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Входной слой
        x = x.permute(0, 2, 1)  # (batch, features, time)
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_relu(x)
        x = self.init_pool(x)

        # Остаточные слои
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Глобальный пулинг
        x = self.global_pool(x).squeeze(-1)

        # Attention
        attention_weights = self.attention(x)
        x = x * attention_weights

        x = self.dropout(x)
        return self.classifier(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut соединение
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class DualBranchFDD(nn.Module):
    """
    Двухканальная модель: одна ветвь для электрических сигналов, другая для ML признаков
    Адаптирована для 5 входных признаков
    """

    def __init__(self, window_size: int, input_dim: int, output_dim: int):
        super(DualBranchFDD, self).__init__()

        # Для ваших данных все 5 признаков - электрические
        # Создаем две ветви обработки для разнообразия подходов

        # Первая ветвь - CNN для выявления локальных паттернов
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)  # Фиксированный размер
        )

        # Вторая ветвь - RNN для временных зависимостей
        self.rnn_branch = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=True)
        self.rnn_dropout = nn.Dropout(0.3)

        # Третья ветвь - Attention механизм
        self.attention_dim = 64
        self.attention = nn.Sequential(
            nn.Linear(input_dim, self.attention_dim),
            nn.ReLU(),
            nn.Linear(self.attention_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        self.attention_fc = nn.Linear(input_dim * window_size, 128)

        # Объединение и классификация
        cnn_features = 256 * 16  # 256 каналов * 16 временных точек
        rnn_features = 256  # 128 * 2 (bidirectional)
        attention_features = 128
        combined_features = cnn_features + rnn_features + attention_features

        self.fusion = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # CNN ветвь
        cnn_out = self.cnn_branch(x.permute(0, 2, 1))
        cnn_out = cnn_out.view(batch_size, -1)

        # RNN ветвь
        rnn_out, _ = self.rnn_branch(x)
        rnn_out = self.rnn_dropout(rnn_out[:, -1, :])  # Берем последний выход

        # Attention ветвь
        attention_weights = self.attention(x)
        attended = x * attention_weights
        att_out = self.attention_fc(attended.view(batch_size, -1))

        # Объединение всех признаков
        combined = torch.cat([cnn_out, rnn_out, att_out], dim=1)

        return self.fusion(combined)