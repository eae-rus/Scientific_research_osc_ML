"""
Гибридные модели для эксперимента 2.6.3.

Двухголовая архитектура (Two-Headed Model):
- Ветка 1 (Fast): обрабатывает Raw данные — ловит резкие фронты и паттерны
- Ветка 2 (Precise): обрабатывает спектральные признаки (Phase Polar) — точная классификация

Все 6 гибридных моделей соответствуют 6 базовым:
- HybridMLP = SimpleMLP + SimpleMLP → Fusion
- HybridCNN = SimpleCNN + SimpleCNN → Fusion
- HybridResNet = ResNet1D + ResNet1D → Fusion
- HybridSimpleKAN = SimpleKAN + SimpleKAN → Fusion
- HybridConvKAN = ConvKAN + ConvKAN → Fusion
- HybridPhysicsKAN = PhysicsKAN + PhysicsKAN → Fusion

Параметры по ширине уменьшены в 2 раза для каждой ветки, чтобы суммарное 
количество параметров оставалось примерно таким же как у базовых моделей.

Авторы: Scientific Research OSC ML Team
Дата: 2026-01-22
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List
from osc_tools.ml.models.base import BaseModel
from osc_tools.ml.layers.kan_layers import KANLinear, KANConv1d
from osc_tools.ml.kan_conv.arithmetic import MultiplicationLayer, DivisionLayer


# ============================================================================
# Вспомогательные компоненты
# ============================================================================

class SafeMaxPool1d(nn.Module):
    """Pooling layer that handles small input sizes gracefully."""
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.pool = nn.MaxPool1d(kernel_size, stride=self.stride, padding=padding)

    def forward(self, x):
        if x.shape[-1] < self.kernel_size:
            return x
        return self.pool(x)


class FusionHead(nn.Module):
    """
    Слой слияния двух веток и классификатор.
    Конкатенирует выходы двух веток и пропускает через FC слои.
    """
    def __init__(self, branch1_features: int, branch2_features: int, 
                 num_classes: int, fusion_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        total_features = branch1_features + branch2_features
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fusion_hidden, num_classes)
        )
        
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([feat1, feat2], dim=1)
        return self.fusion(combined)


# ============================================================================
# HybridMLP: SimpleMLP + SimpleMLP
# ============================================================================

class HybridMLP(BaseModel):
    """
    Гибридная MLP модель: две независимые MLP ветки + слияние.
    
    Аналог SimpleMLP, но с двумя входами (raw + features).
    Параметры hidden_sizes уменьшены вдвое для каждой ветки.
    """
    def __init__(
        self, 
        input_size: int = None,  # Для совместимости, игнорируется
        in_channels: int = 24,  # Общее количество каналов
        num_classes: int = 4,
        hidden_sizes: List[int] = [64, 32],  # Для каждой ветки (уже уменьшенные)
        dropout: float = 0.2,
        use_bn: bool = True,
        raw_channels: int = 8,
        features_channels: int = 16,
        seq_len: int = 64  # Длина последовательности
    ):
        super().__init__()
        
        self.raw_channels = raw_channels
        self.features_channels = features_channels
        self.seq_len = seq_len
        
        # Размеры входов для каждой ветки
        self.raw_input_size = raw_channels * seq_len
        self.features_input_size = features_channels * seq_len
        
        # Ветка 1: Raw данные
        self.raw_branch = self._make_mlp_branch(
            self.raw_input_size, hidden_sizes, dropout, use_bn
        )
        
        # Ветка 2: Features (спектральные признаки)
        self.features_branch = self._make_mlp_branch(
            self.features_input_size, hidden_sizes, dropout, use_bn
        )
        
        # Размер выхода каждой ветки
        branch_out = hidden_sizes[-1] if hidden_sizes else self.raw_input_size
        
        # Fusion + Classifier
        self.head = FusionHead(
            branch_out, branch_out, num_classes, 
            fusion_hidden=branch_out, dropout=dropout
        )
        
    def _make_mlp_branch(self, input_size, hidden_sizes, dropout, use_bn):
        layers = []
        prev_size = input_size
        
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if use_bn:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = size
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channels, Length) или (Batch, Features)
        if x.dim() == 3:
            # Разделяем по каналам
            x_raw = x[:, :self.raw_channels, :].flatten(start_dim=1)
            x_feat = x[:, self.raw_channels:, :].flatten(start_dim=1)
        else:
            # Уже flatten
            x_raw = x[:, :self.raw_input_size]
            x_feat = x[:, self.raw_input_size:]
        
        raw_out = self.raw_branch(x_raw)
        feat_out = self.features_branch(x_feat)
        
        return self.head(raw_out, feat_out)


# ============================================================================
# HybridCNN: SimpleCNN + SimpleCNN
# ============================================================================

class HybridCNN(BaseModel):
    """
    Гибридная CNN модель: две независимые CNN ветки + слияние.
    
    Аналог SimpleCNN, но с двумя входами.
    Параметры channels уменьшены вдвое для каждой ветки.
    """
    def __init__(
        self,
        in_channels: int = 24,  # Общее количество каналов (raw + features)
        num_classes: int = 4,
        channels: List[int] = [16, 32],  # Для каждой ветки (уже уменьшенные)
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.2,
        use_bn: bool = True,
        pool_every: int = 1,
        raw_channels: int = 8,
        features_channels: int = 16
    ):
        super().__init__()
        
        self.raw_channels = raw_channels
        self.features_channels = features_channels
        
        # Ветка 1: Raw данные
        self.raw_branch = self._make_cnn_branch(
            raw_channels, channels, kernel_size, stride, dropout, use_bn, pool_every
        )
        
        # Ветка 2: Features
        self.features_branch = self._make_cnn_branch(
            features_channels, channels, kernel_size, stride, dropout, use_bn, pool_every
        )
        
        # Размер выхода каждой ветки после AdaptiveAvgPool1d(1)
        branch_out = channels[-1] if channels else raw_channels
        
        # Fusion + Classifier
        self.head = FusionHead(
            branch_out, branch_out, num_classes,
            fusion_hidden=branch_out, dropout=dropout
        )
        
    def _make_cnn_branch(self, in_ch, channels, kernel_size, stride, dropout, use_bn, pool_every):
        layers = []
        curr_ch = in_ch
        
        for i, out_ch in enumerate(channels):
            s = stride if i == 0 else 1
            layers.append(nn.Conv1d(curr_ch, out_ch, kernel_size, stride=s, padding=kernel_size//2))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            
            if (i + 1) % pool_every == 0:
                layers.append(SafeMaxPool1d(2))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_ch = out_ch
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channels, Length)
        x_raw = x[:, :self.raw_channels, :]
        x_feat = x[:, self.raw_channels:, :]
        
        raw_out = self.raw_branch(x_raw)
        raw_out = nn.functional.adaptive_avg_pool1d(raw_out, 1).flatten(1)
        
        feat_out = self.features_branch(x_feat)
        feat_out = nn.functional.adaptive_avg_pool1d(feat_out, 1).flatten(1)
        
        return self.head(raw_out, feat_out)


# ============================================================================
# HybridResNet: ResNet1D + ResNet1D
# ============================================================================

class _ResBlock1D(nn.Module):
    """Базовый блок ResNet для 1D сигналов (локальный класс для HybridResNet)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=None):
        super().__init__()
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


class _ResNetBranch(nn.Module):
    """Одна ветка ResNet для гибридной модели."""
    def __init__(self, in_channels, layers, base_filters):
        super().__init__()
        self.inplanes = base_filters
        
        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = SafeMaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(base_filters, layers[0])
        self.layer2 = self._make_layer(base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out_features = base_filters * 8
        
    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(_ResBlock1D(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_ResBlock1D(self.inplanes, planes))
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
        return x.flatten(1)


class HybridResNet(BaseModel):
    """
    Гибридная ResNet модель: две независимые ResNet ветки + слияние.
    
    Аналог ResNet1D, но с двумя входами.
    Параметры base_filters уменьшены вдвое для каждой ветки.
    """
    def __init__(
        self,
        in_channels: int = 24,
        num_classes: int = 4,
        layers: List[int] = [1, 1, 1, 1],  # Количество блоков в каждом layer
        base_filters: int = 32,  # Уменьшено вдвое (было 64)
        raw_channels: int = 8,
        features_channels: int = 16
    ):
        super().__init__()
        
        self.raw_channels = raw_channels
        self.features_channels = features_channels
        
        # Ветка 1: Raw данные
        self.raw_branch = _ResNetBranch(raw_channels, layers, base_filters)
        
        # Ветка 2: Features
        self.features_branch = _ResNetBranch(features_channels, layers, base_filters)
        
        # Выход каждой ветки: base_filters * 8 (после layer4)
        branch_out = base_filters * 8
        
        # Fusion + Classifier
        self.head = FusionHead(
            branch_out, branch_out, num_classes,
            fusion_hidden=branch_out, dropout=0.2
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_raw = x[:, :self.raw_channels, :]
        x_feat = x[:, self.raw_channels:, :]
        
        raw_out = self.raw_branch(x_raw)
        feat_out = self.features_branch(x_feat)
        
        return self.head(raw_out, feat_out)


# ============================================================================
# HybridSimpleKAN: SimpleKAN + SimpleKAN
# ============================================================================

class HybridSimpleKAN(BaseModel):
    """
    Гибридная KAN модель: две независимые SimpleKAN ветки + слияние.
    
    Аналог SimpleKAN, но с двумя входами.
    Параметры hidden_sizes уменьшены вдвое для каждой ветки.
    """
    def __init__(
        self,
        input_size: int = None,  # Для совместимости, игнорируется
        in_channels: int = 24,
        num_classes: int = 4,
        hidden_sizes: List[int] = [32, 16],  # Уменьшено вдвое
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.0,
        base_activation=torch.nn.SiLU,
        raw_channels: int = 8,
        features_channels: int = 16,
        seq_len: int = 64
    ):
        super().__init__()
        
        self.raw_channels = raw_channels
        self.features_channels = features_channels
        self.seq_len = seq_len
        
        # Размеры входов
        self.raw_input_size = raw_channels * seq_len
        self.features_input_size = features_channels * seq_len
        
        # Ветка 1: Raw данные
        self.raw_branch = self._make_kan_branch(
            self.raw_input_size, hidden_sizes, grid_size, spline_order, dropout, base_activation
        )
        
        # Ветка 2: Features
        self.features_branch = self._make_kan_branch(
            self.features_input_size, hidden_sizes, grid_size, spline_order, dropout, base_activation
        )
        
        branch_out = hidden_sizes[-1] if hidden_sizes else self.raw_input_size
        
        # Fusion + Classifier (используем KANLinear для финального слияния)
        fusion_size = branch_out * 2
        self.fusion = nn.Sequential(
            KANLinear(fusion_size, branch_out, grid_size=grid_size, spline_order=spline_order, base_activation=base_activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            KANLinear(branch_out, num_classes, grid_size=grid_size, spline_order=spline_order, base_activation=base_activation)
        )
        
    def _make_kan_branch(self, input_size, hidden_sizes, grid_size, spline_order, dropout, base_activation):
        layers = []
        prev_size = input_size
        
        for size in hidden_sizes:
            layers.append(
                KANLinear(prev_size, size, grid_size=grid_size, spline_order=spline_order, base_activation=base_activation)
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = size
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x_raw = x[:, :self.raw_channels, :].flatten(start_dim=1)
            x_feat = x[:, self.raw_channels:, :].flatten(start_dim=1)
        else:
            x_raw = x[:, :self.raw_input_size]
            x_feat = x[:, self.raw_input_size:]
        
        raw_out = self.raw_branch(x_raw)
        feat_out = self.features_branch(x_feat)
        
        combined = torch.cat([raw_out, feat_out], dim=1)
        return self.fusion(combined)


# ============================================================================
# HybridConvKAN: ConvKAN + ConvKAN
# ============================================================================

class HybridConvKAN(BaseModel):
    """
    Гибридная ConvKAN модель: две независимые ConvKAN ветки + слияние.
    
    Аналог ConvKAN, но с двумя входами.
    Параметры channels уменьшены вдвое для каждой ветки.
    """
    def __init__(
        self,
        in_channels: int = 24,
        num_classes: int = 4,
        channels: List[int] = [8, 16],  # Уменьшено вдвое
        kernel_size: int = 3,
        stride: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.2,
        pool_every: int = 1,
        base_activation=torch.nn.SiLU,
        raw_channels: int = 8,
        features_channels: int = 16
    ):
        super().__init__()
        
        self.raw_channels = raw_channels
        self.features_channels = features_channels
        
        # Ветка 1: Raw данные
        self.raw_branch = self._make_convkan_branch(
            raw_channels, channels, kernel_size, stride, grid_size, spline_order, 
            dropout, pool_every, base_activation
        )
        
        # Ветка 2: Features
        self.features_branch = self._make_convkan_branch(
            features_channels, channels, kernel_size, stride, grid_size, spline_order,
            dropout, pool_every, base_activation
        )
        
        branch_out = channels[-1] if channels else raw_channels
        
        # Fusion + Classifier с KANLinear
        fusion_size = branch_out * 2
        self.classifier = nn.Sequential(
            KANLinear(fusion_size, branch_out, grid_size=grid_size, base_activation=base_activation),
            KANLinear(branch_out, num_classes, grid_size=grid_size, base_activation=base_activation)
        )
        
    def _make_convkan_branch(self, in_ch, channels, kernel_size, stride, grid_size, 
                              spline_order, dropout, pool_every, base_activation):
        layers = []
        curr_ch = in_ch
        
        for i, out_ch in enumerate(channels):
            curr_grid = grid_size[i] if isinstance(grid_size, list) else grid_size
            s = stride if i == 0 else 1
            
            layers.append(
                KANConv1d(curr_ch, out_ch, kernel_size=kernel_size, stride=s,
                          padding=kernel_size//2, grid_size=curr_grid, 
                          spline_order=spline_order, base_activation=base_activation)
            )
            layers.append(nn.BatchNorm1d(out_ch))
            
            if (i + 1) % pool_every == 0:
                layers.append(SafeMaxPool1d(2))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_ch = out_ch
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_raw = x[:, :self.raw_channels, :]
        x_feat = x[:, self.raw_channels:, :]
        
        raw_out = self.raw_branch(x_raw)
        raw_out = nn.functional.adaptive_avg_pool1d(raw_out, 1).flatten(1)
        
        feat_out = self.features_branch(x_feat)
        feat_out = nn.functional.adaptive_avg_pool1d(feat_out, 1).flatten(1)
        
        combined = torch.cat([raw_out, feat_out], dim=1)
        return self.classifier(combined)


# ============================================================================
# HybridPhysicsKAN: PhysicsKAN + PhysicsKAN
# ============================================================================

class HybridPhysicsKAN(BaseModel):
    """
    Гибридная PhysicsKAN модель: две независимые PhysicsKAN ветки + слияние.
    
    Каждая ветка применяет физические операции (Mult/Div для I*U и I/U).
    Аналог PhysicsKAN, но с двумя входами.
    """
    def __init__(
        self,
        in_channels: int = 24,
        num_classes: int = 4,
        channels: List[int] = [8, 16],  # Уменьшено вдвое
        kernel_size: int = 3,
        stride: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.2,
        pool_every: int = 1,
        base_activation=torch.nn.SiLU,
        raw_channels: int = 8,
        features_channels: int = 16
    ):
        super().__init__()
        
        self.raw_channels = raw_channels
        self.features_channels = features_channels
        
        # Физические слои для Raw ветки (предполагаем [I, U] структуру)
        # Raw: [IA, IB, IC, IN, UA, UB, UC, UN] -> half = 4 канала токов, 4 напряжений
        self.mult_raw = MultiplicationLayer()
        self.div_raw = DivisionLayer()
        raw_half = raw_channels // 2
        self.bn_mult_raw = nn.BatchNorm1d(raw_half)
        self.bn_div_raw = nn.BatchNorm1d(raw_half)
        
        # Физические слои для Features ветки
        self.mult_feat = MultiplicationLayer()
        self.div_feat = DivisionLayer()
        feat_half = features_channels // 2
        self.bn_mult_feat = nn.BatchNorm1d(feat_half)
        self.bn_div_feat = nn.BatchNorm1d(feat_half)
        
        # Вход для ConvKAN веток после физических операций:
        # Original (C) + Mult (C/2) + Div (C/2) = 2 * C
        raw_conv_in = raw_channels + raw_half * 2
        feat_conv_in = features_channels + feat_half * 2
        
        # ConvKAN режим
        self.raw_branch = self._make_convkan_branch(
            raw_conv_in, channels, kernel_size, stride, grid_size, 
            spline_order, dropout, pool_every, base_activation
        )
        self.features_branch = self._make_convkan_branch(
            feat_conv_in, channels, kernel_size, stride, grid_size,
            spline_order, dropout, pool_every, base_activation
        )
        
        branch_out = channels[-1] if channels else raw_conv_in
        
        # Fusion + Classifier
        fusion_size = branch_out * 2
        self.classifier = nn.Sequential(
            KANLinear(fusion_size, branch_out, grid_size=grid_size, base_activation=base_activation),
            KANLinear(branch_out, num_classes, grid_size=grid_size, base_activation=base_activation)
        )
        
    def _make_convkan_branch(self, in_ch, channels, kernel_size, stride, grid_size,
                              spline_order, dropout, pool_every, base_activation):
        layers = []
        curr_ch = in_ch
        
        for i, out_ch in enumerate(channels):
            curr_grid = grid_size[i] if isinstance(grid_size, list) else grid_size
            s = stride if i == 0 else 1
            
            layers.append(
                KANConv1d(curr_ch, out_ch, kernel_size=kernel_size, stride=s,
                          padding=kernel_size//2, grid_size=curr_grid,
                          spline_order=spline_order, base_activation=base_activation)
            )
            layers.append(nn.BatchNorm1d(out_ch))
            
            if (i + 1) % pool_every == 0:
                layers.append(SafeMaxPool1d(2))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_ch = out_ch
            
        return nn.Sequential(*layers)
    
    def _apply_physics(self, x, mult_layer, div_layer, bn_mult, bn_div):
        """Применяет физические операции к входу (каналы должны быть [I, U] попарно)."""
        # MultiplicationLayer и DivisionLayer ожидают тензор с парными каналами
        # и сами делят его пополам
        
        power = mult_layer(x)  # I * U (внутри делит тензор пополам)
        power = bn_mult(power)
        
        admittance = div_layer(x)  # I / U (внутри делит тензор пополам)
        admittance = bn_div(admittance)
        
        return torch.cat([x, power, admittance], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_raw = x[:, :self.raw_channels, :]
        x_feat = x[:, self.raw_channels:, :]
        
        # Физические операции
        x_raw = self._apply_physics(x_raw, self.mult_raw, self.div_raw, self.bn_mult_raw, self.bn_div_raw)
        x_feat = self._apply_physics(x_feat, self.mult_feat, self.div_feat, self.bn_mult_feat, self.bn_div_feat)
        
        raw_out = self.raw_branch(x_raw)
        feat_out = self.features_branch(x_feat)
        
        raw_out = nn.functional.adaptive_avg_pool1d(raw_out, 1).flatten(1)
        feat_out = nn.functional.adaptive_avg_pool1d(feat_out, 1).flatten(1)
        
        combined = torch.cat([raw_out, feat_out], dim=1)
        return self.classifier(combined)
