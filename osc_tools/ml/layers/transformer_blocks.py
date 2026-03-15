"""
Строительные блоки Physical KAN-Transformer (Фаза 4).

Содержит:
- DataSanitizer: обработка отсутствующих каналов (NaN/-1) + learnable MissingToken
- ComplexInteractionBlock: обучаемое комплексное умнож./деление через 4-группную схему
- PhysicalStem: rPhysicsKAN-кодирование + ComplexInteractionBlock → embedding
- SinusoidalPositionalEncoding: позиционное кодирование
- KANFeedForward: FastKAN-FFN для Transformer (без физических блоков)
- PhysicalKANFeedForward: KAN-FFN + малый ComplexInteractionBlock в каждом слое
- MLPFeedForward: стандартный MLP-FFN для baseline
- TransformerEncoderBlock: один блок Encoder (Pre-LN Attention + FFN)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# FastKAN из локального репозитория
import sys
import os

_fast_kan_path = os.path.join(os.path.dirname(__file__), '..', 'fast-kan-master')
if _fast_kan_path not in sys.path:
    sys.path.insert(0, _fast_kan_path)

# Pylance не видит этот путь (добавлен динамически), но импорт работает корректно
from fastkan import FastKANLayer  # noqa: E402


# ---------------------------------------------------------------------------
# DataSanitizer: обработка отсутствующих сигналов
# ---------------------------------------------------------------------------

class DataSanitizer(nn.Module):
    """Обработка отсутствующих каналов в данных.

    Поддерживает два варианта маркера отсутствия:
    - NaN (по умолчанию) — естественный маркер в данных, когда канал отсутствует
    - Конкретное число (например, -1) — если данные явно размечены

    Логика:
    1. Создаёт бинарную маску пропусков M_missing (True = отсутствует).
    2. Заменяет маркер на 0 для безопасных математических операций.
    3. Добавляет обучаемый MissingToken к позициям пропусков.

    Примечание: замена на 0 безопасна для деления, т.к. PhysicalStem
    дополнительно проверяет порог шума (amp_I > noise_threshold) и не делит
    на значения ниже порога. Так что 0 не попадёт в знаменатель.

    Args:
        num_channels: количество входных каналов (для learnable embedding)
        missing_marker: значение-маркер отсутствия. По умолчанию None (= NaN).
            Если передано число (например, -1.0), используется точное сравнение.
    """

    def __init__(self, num_channels: int, missing_marker: float | None = None) -> None:
        super().__init__()
        self.missing_marker = missing_marker
        # Обучаемый вектор-заглушка для отсутствующих каналов
        self.missing_token = nn.Parameter(torch.zeros(1, num_channels, 1))
        nn.init.normal_(self.missing_token, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T) — входные признаки (амплитуды+углы чередуются)

        Returns:
            x_safe: (B, C, T) — очищенные данные
            mask_missing: (B, C, T) — True где данные отсутствовали
        """
        if self.missing_marker is None:
            # NaN-маркер (по умолчанию)
            mask_missing = torch.isnan(x)
        else:
            # Числовой маркер (например, -1.0)
            mask_missing = (x == self.missing_marker)

        # Заменяем маркер на 0 для безопасных операций
        # (0 безопасен: PhysicalStem проверяет amp > noise_threshold перед делением)
        x_safe = torch.where(mask_missing, torch.zeros_like(x), x)

        # Добавляем learnable embedding для пропущенных каналов
        x_safe = x_safe + mask_missing.float() * self.missing_token

        return x_safe, mask_missing


# ---------------------------------------------------------------------------
# ComplexInteractionBlock: обучаемое комплексное умножение/деление
# ---------------------------------------------------------------------------

class ComplexInteractionBlock(nn.Module):
    """Обучаемый блок комплексных взаимодействий (умножение + деление).

    Работает полностью в комплексном пространстве:
    1. Входные сигналы — комплексные тензоры (z = A·exp(jφ) или re+j·im)
    2. Вещественная линейная комбинация: y_k = Σ_i(w_ki · z_i).
       Веса w_ki — вещественные скаляры, масштабирующие амплитуды.
       Углы (фазы) НЕ искажаются отдельным слоем — они сохраняются
       через комплексное векторное сложение.
    3. Промежуточные Y сигналов делятся на 4 группы по Y/4:
       - Группы 1 × 2 → попарное комплексное умножение
       - Группы 3 × 4 → попарное комплексное деление
    4. Итого num_interaction_pairs выходных комплексных сигналов.

    Args:
        num_input_pairs: число входных комплексных сигналов
        num_interaction_pairs: число выходных сигналов (должно быть чётным)
        division_epsilon: минимальный квадрат модуля знаменателя при делении
    """

    def __init__(
        self,
        num_input_pairs: int,
        num_interaction_pairs: int = 16,
        division_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        assert num_interaction_pairs % 2 == 0, (
            f"num_interaction_pairs должно быть чётным, получено {num_interaction_pairs}"
        )

        self.num_input_pairs = num_input_pairs
        self.num_interaction_pairs = num_interaction_pairs
        self.division_epsilon = division_epsilon

        # Каждая пара операций (mul/div) требует 2 операнда → 4 группы по group_size
        # group_size = n/2, Y_intermediate = 4 * group_size = 2n
        self.group_size = num_interaction_pairs // 2
        num_intermediate = 4 * self.group_size  # = 2 * num_interaction_pairs

        # Вещественная линейная комбинация для комплексных входов:
        # y = W @ z, где W ∈ R^{Y×X}, z ∈ C^{X} → y ∈ C^{Y}
        # Эквивалентно: y_k = Σ_i(w_ki · A_i · exp(jφ_i)) — масштабирование амплитуд
        # с сохранением фаз через комплексное сложение.
        self.selector = nn.Linear(num_input_pairs, num_intermediate, bias=False)
        nn.init.xavier_uniform_(self.selector.weight)

    def forward(
        self, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z: (B*T, num_input_pairs) complex64 — входные комплексные сигналы

        Returns:
            z_out: (B*T, num_interaction_pairs) complex64 — результат взаимодействий
        """
        # Вещественная линейная комбинация комплексных входов:
        # W @ z = W @ re(z) + j · W @ im(z)
        w = self.selector.weight  # (num_intermediate, num_input_pairs), вещественный
        re_combined = F.linear(z.real, w)  # (B*T, num_intermediate)
        im_combined = F.linear(z.imag, w)  # (B*T, num_intermediate)
        z_combined = torch.complex(re_combined, im_combined)

        # Разделяем на 4 группы
        gs = self.group_size
        g1 = z_combined[:, 0*gs : 1*gs]
        g2 = z_combined[:, 1*gs : 2*gs]
        g3 = z_combined[:, 2*gs : 3*gs]
        g4 = z_combined[:, 3*gs : 4*gs]

        # Комплексное умножение: g1 · g2
        z_mul = g1 * g2

        # Комплексное деление: g3 / g4 (безопасное)
        # z/w = z·conj(w) / |w|², с clamp на |w|² для защиты от деления на ≈0
        g4_mag_sq = g4.real ** 2 + g4.imag ** 2
        g4_mag_sq_safe = g4_mag_sq.clamp(min=self.division_epsilon ** 2)
        z_div = (g3 * g4.conj()) / g4_mag_sq_safe

        return torch.cat([z_mul, z_div], dim=-1)


# ---------------------------------------------------------------------------
# PhysicalStem: физические операции + KAN → embedding для Transformer
# ---------------------------------------------------------------------------

class PhysicalStem(nn.Module):
    """Входной Stem-блок с физическим индуктивным смещением.

    Реализует rPhysicsKAN-подход:
    1. Разделяет амплитуды и углы
    2. Обучаемый ComplexInteractionBlock: модель сама выбирает, какие пары
       сигналов умножать/делить (вместо жёстких P/Q/Z формул)
    3. KAN-кодирование с гейтированием углами
    4. Линейная проекция в d_model для Transformer

    Входной формат: (B, C, T) где C — чередующиеся [A₁, φ₁, A₂, φ₂, ...]
    Каналы идут парами: первая половина пар = токи, вторая = напряжения.

    Args:
        num_input_channels: полное число каналов (амплитуды + углы)
        d_model: размерность выходного embedding для Transformer
        num_current_pairs: число пар ток (A_I, φ_I) в первой половине каналов
        num_interaction_pairs: число выходных пар от ComplexInteractionBlock
        noise_threshold_current: порог шума для токов (отн. ед.)
        noise_threshold_voltage: порог шума для напряжений (отн. ед.)
        kan_grid_size: число узлов RBF-сетки для FastKAN
        dropout: вероятность dropout
    """

    def __init__(
        self,
        num_input_channels: int,
        d_model: int = 64,
        num_current_pairs: int | None = None,
        num_interaction_pairs: int = 16,
        noise_threshold_current: float = 1.0 / 2000,
        noise_threshold_voltage: float = 1.0 / 300,
        kan_grid_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert num_input_channels % 2 == 0, (
            f"num_input_channels должно быть чётным (A+φ пары), получено {num_input_channels}"
        )

        self.num_input_channels = num_input_channels
        self.num_pairs = num_input_channels // 2  # Количество (A, φ) пар
        self.d_model = d_model

        # Число пар токов (по умолчанию — ровно половина)
        if num_current_pairs is None:
            num_current_pairs = self.num_pairs // 2
        self.num_current_pairs = num_current_pairs
        self.num_voltage_pairs = self.num_pairs - num_current_pairs

        # Пороги шума
        self.noise_threshold_I = noise_threshold_current
        self.noise_threshold_U = noise_threshold_voltage

        # --- Обучаемый блок комплексных взаимодействий ---
        self.interaction = ComplexInteractionBlock(
            num_input_pairs=self.num_pairs,
            num_interaction_pairs=num_interaction_pairs,
        )
        # Число признаков от interaction: amp + angle = 2 * num_interaction_pairs
        num_interaction_features = num_interaction_pairs * 2

        # --- rPhysicsKAN: KAN для амплитуд + гейт из углов ---
        num_amplitudes = self.num_pairs  # Все амплитуды
        num_angles = self.num_pairs  # Все углы

        # KAN для амплитуд → половина d_model
        half_d = d_model // 2
        self.kan_amp = FastKANLayer(
            input_dim=num_amplitudes,
            output_dim=half_d,
            num_grids=kan_grid_size,
            use_base_update=True,
            use_layernorm=True,
        )

        # KAN для углов → гейт (sigmoid) для модулирования амплитуд
        self.kan_angle_gate = FastKANLayer(
            input_dim=num_angles,
            output_dim=half_d,
            num_grids=kan_grid_size,
            use_base_update=True,
            use_layernorm=True,
        )

        # KAN для углов → вторая половина embedding
        self.kan_angle = FastKANLayer(
            input_dim=num_angles,
            output_dim=half_d,
            num_grids=kan_grid_size,
            use_base_update=True,
            use_layernorm=True,
        )

        # Проекция: (half_d_modulated + half_d_angle + interaction_features) → d_model
        fusion_dim = half_d + half_d + num_interaction_features
        self.projection = nn.Linear(fusion_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _extract_amp_angle(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Разделить чередующийся формат [A₁, φ₁, A₂, φ₂, ...] на амплитуды и углы.

        Args:
            x: (B, C, T)
        Returns:
            amplitudes: (B, num_pairs, T)
            angles: (B, num_pairs, T)
        """
        amplitudes = x[:, 0::2, :]  # Чётные каналы — амплитуды
        angles = x[:, 1::2, :]      # Нечётные каналы — углы
        return amplitudes, angles

    def forward(
        self, x: torch.Tensor, mask_missing: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) — очищенные данные после DataSanitizer
            mask_missing: (B, C, T) — маска пропусков

        Returns:
            embedding: (B, T, d_model) — готовый для Transformer вход
        """
        B, C, T = x.shape

        # 1. Разделить на амплитуды и углы
        amplitudes, angles = self._extract_amp_angle(x)

        # 2. Переводим в формат (B*T, features) для FastKAN и interaction
        amp_flat = amplitudes.permute(0, 2, 1).reshape(B * T, -1)  # (B*T, num_pairs)
        ang_flat = angles.permute(0, 2, 1).reshape(B * T, -1)      # (B*T, num_pairs)

        # 3. Обучаемый комплексный блок: модель сама решает что с чем умножать/делить
        #    Конвертируем в комплексные числа: z = A·exp(jφ)
        z_input = torch.polar(amp_flat, ang_flat)  # (B*T, num_pairs) complex64
        z_inter = self.interaction(z_input)         # (B*T, num_interaction_pairs) complex64
        # Обратно в полярную форму (amp, angle) для конкатенации с остальными фичами
        inter_amp = z_inter.abs()
        inter_angle = z_inter.angle()
        inter_features = torch.stack([inter_amp, inter_angle], dim=-1)  # (B*T, n, 2)
        inter_features = inter_features.reshape(B * T, -1)  # (B*T, n*2)

        # 4. rPhysicsKAN: KAN для амплитуд + гейтирование углами
        h_amp = self.kan_amp(amp_flat)           # (B*T, half_d)
        g_angle = torch.sigmoid(self.kan_angle_gate(ang_flat))  # (B*T, half_d) — гейт [0, 1]
        h_modulated = h_amp * g_angle            # Модулирование амплитуд углами

        h_angle = self.kan_angle(ang_flat)       # (B*T, half_d)

        # 5. Объединение и проекция
        fused = torch.cat([h_modulated, h_angle, inter_features], dim=-1)
        embedding = self.projection(fused)       # (B*T, d_model)
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)

        # Обратно в (B, T, d_model)
        embedding = embedding.view(B, T, self.d_model)
        return embedding


# ---------------------------------------------------------------------------
# Positional Encoding (синусоидальное)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Классическое синусоидальное позиционное кодирование (Vaswani et al., 2017).

    Args:
        d_model: размерность модели
        max_len: максимальная длина последовательности
        dropout: dropout после добавления PE
    """

    def __init__(
        self, d_model: int, max_len: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) с добавленным позиционным кодированием
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# KANFeedForward: замена MLP-FFN на FastKAN
# ---------------------------------------------------------------------------

class KANFeedForward(nn.Module):
    """Feed-Forward Network на основе FastKAN для Transformer.

    Заменяет стандартный FFN (Linear → ReLU → Linear) на
    FastKANLayer → FastKANLayer с остаточным соединением.

    Args:
        d_model: размерность входа/выхода
        d_ff: размерность скрытого слоя (по умолчанию 4*d_model)
        kan_grid_size: число узлов RBF-сетки
        dropout: dropout после FFN
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        kan_grid_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.kan1 = FastKANLayer(
            input_dim=d_model,
            output_dim=d_ff,
            num_grids=kan_grid_size,
            use_base_update=True,
            use_layernorm=False,  # LayerNorm внешний (Pre-LN)
        )
        self.kan2 = FastKANLayer(
            input_dim=d_ff,
            output_dim=d_model,
            num_grids=kan_grid_size,
            use_base_update=True,
            use_layernorm=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, D = x.shape
        # FastKANLayer ожидает 2D → reshape
        x_flat = x.reshape(B * T, D)
        h = self.kan1(x_flat)
        h = self.dropout(h)
        h = self.kan2(h)
        h = self.dropout(h)
        return h.view(B, T, D)


# ---------------------------------------------------------------------------
# PhysicalKANFeedForward: KAN-FFN + ComplexInteractionBlock
# ---------------------------------------------------------------------------

class PhysicalKANFeedForward(nn.Module):
    """KAN-FFN с встроенным ComplexInteractionBlock для физических операций.

    Параллельно с основным KAN-путём (d_model → d_ff → d_model) работает
    малый блок комплексных взаимодействий. Это позволяет модели выполнять
    умножение/деление в комплексной плоскости на КАЖДОМ слое Transformer,
    а не только в Stem.

    Мотивация: физические комбинации сигналов (мощности, импедансы и т.д.)
    могут требовать нескольких последовательных операций, и одного Stem
    недостаточно для подготовки всех нужных признаков.

    Архитектура:
    - KAN путь: d_model → d_ff → d_model (основная нелинейность)
    - Комплексный путь: d_model/2 complex → interaction → project → d_model
    - Гейтированное сложение: KAN + gate * interaction → d_model

    Размерность d_model сохраняется для residual-соединений.

    Args:
        d_model: размерность входа/выхода
        d_ff: размерность скрытого слоя KAN (по умолчанию 4 * d_model)
        num_interaction_pairs: число выходных пар ComplexInteractionBlock
            (обычно 4-8, меньше чем в Stem)
        kan_grid_size: число узлов RBF-сетки FastKAN
        dropout: dropout
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        num_interaction_pairs: int = 4,
        kan_grid_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % 2 == 0, f"d_model должен быть чётным, получено {d_model}"

        if d_ff is None:
            d_ff = d_model * 4

        # TODO: Но эти числа тоже комплексные, т.е. там амплитуда и угол. Да, может
        # блок внимания как-то не так на это влияет и мешает... Не знаю вот так сходу,
        # надо обдумать, но в целом это тоже комплексные числа и их надо обрабатывать так же,
        # Как мы обрабатывали входные сигналы, ведь исходно там амплитуда и угол. 
        # --- KAN путь (основной) ---
        self.kan1 = FastKANLayer(
            input_dim=d_model,
            output_dim=d_ff,
            num_grids=kan_grid_size,
            use_base_update=True,
            use_layernorm=False,
        )
        self.kan2 = FastKANLayer(
            input_dim=d_ff,
            output_dim=d_model,
            num_grids=kan_grid_size,
            use_base_update=True,
            use_layernorm=False,
        )
        self.dropout = nn.Dropout(dropout)

        # --- Комплексный путь (малый) ---
        # d_model/2 фичей = real-часть, d_model/2 = imag-часть → d_model/2 комплексных
        num_complex_inputs = d_model // 2
        self.interaction = ComplexInteractionBlock(
            num_input_pairs=num_complex_inputs,
            num_interaction_pairs=num_interaction_pairs,
        )
        # Проекция: (num_interaction_pairs * 2 вещественных) → d_model
        self.interaction_proj = nn.Linear(num_interaction_pairs * 2, d_model)
        # Гейт: sigmoid(-2) ≈ 0.12 на старте — комплексный путь включается мягко
        self.interaction_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        # KAN путь
        h = self.kan1(x_flat)
        h = self.dropout(h)
        h = self.kan2(h)
        h = self.dropout(h)  # (B*T, d_model)

        # Комплексный путь: интерпретируем d_model как (re, im) пары
        half = D // 2
        z_in = torch.complex(x_flat[:, :half], x_flat[:, half:])  # (B*T, d_model/2)
        z_out = self.interaction(z_in)  # (B*T, num_interaction_pairs) complex

        # Обратно в вещественные: [re₁, im₁, re₂, im₂, ...]
        inter_real = torch.stack([z_out.real, z_out.imag], dim=-1)
        inter_flat = inter_real.reshape(B * T, -1)  # (B*T, num_interaction_pairs * 2)
        inter_proj = self.interaction_proj(inter_flat)  # (B*T, d_model)

        # Гейтированное сложение
        gate = torch.sigmoid(self.interaction_gate)
        combined = h + gate * inter_proj

        return combined.view(B, T, D)


# ---------------------------------------------------------------------------
# MLPFeedForward: стандартный MLP FFN для Baseline Transformer
# ---------------------------------------------------------------------------

class MLPFeedForward(nn.Module):
    """Стандартный FFN (Linear → GELU → Linear) для baseline сравнения.

    Args:
        d_model: размерность входа/выхода
        d_ff: размерность скрытого слоя
        dropout: dropout
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# TransformerEncoderBlock: один слой Encoder (Pre-LN архитектура)
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """Один блок Transformer Encoder с Pre-LayerNorm.

    Pre-LN: LayerNorm → Attention → residual → LayerNorm → FFN → residual.
    Стабильнее Post-LN для глубоких моделей.

    Args:
        d_model: размерность модели
        num_heads: число голов внимания
        ffn: Feed-Forward модуль (KANFeedForward или MLPFeedForward)
        dropout: dropout для attention
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn: nn.Module,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = ffn

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            key_padding_mask: (B, T) — True для позиций, которые нужно игнорировать

        Returns:
            (B, T, d_model)
        """
        # Pre-LN Attention
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out

        # Pre-LN FFN
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x
