"""
Строительные блоки Physical KAN-Transformer (Фаза 4).

Содержит:
- DataSanitizer: замена -1 на 0 + создание маски пропусков + learnable MissingToken
- PhysicalStem: физические операции (P, Q, Z) + KAN-кодирование + раздельная обработка
  амплитуд и углов (rPhysicsKAN-стиль) → итоговый embedding для Transformer
- PositionalEncoding: синусоидальное 1D позиционное кодирование
- KANFeedForward: замена MLP-FFN в Transformer на Fast-KAN
- TransformerEncoderBlock: один блок Encoder (Attention + KAN-FFN + LayerNorm)
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

# TODO: почему-то подсвечивается как неиспользуемый импорт, вероятно это связано с тем,
# что он скачан напрямую как библиотека... Хотя и не факт, в общем, надо проверить.
from fastkan import FastKANLayer


# ---------------------------------------------------------------------------
# DataSanitizer: обработка отсутствующих сигналов (-1)
# ---------------------------------------------------------------------------

# TODO: Фраза о том, что оно будет равно "-1", это лишь гипотеза, данные пока никак не размечались
# и там скорее будут получаться None или прост нули, это надо как раз в формировании данных проверить.
# И если удобнее задать какое-то другое число / шаблон, то давай поменяем, мы только начинаем эту работу.
class DataSanitizer(nn.Module):
    """Обработка специального значения -1 (отсутствующий канал).

    1. Создаёт бинарную маску пропусков M_missing (True = отсутствует).
    2. Заменяет -1 на 0 для безопасных математических операций.
    3. Добавляет обучаемый MissingToken к позициям пропусков.

    Args:
        num_channels: количество входных каналов (для learnable embedding)
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
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
            x_safe: (B, C, T) — очищенные данные (без -1)
            mask_missing: (B, C, T) — True где данные отсутствовали
        """
        # Для полярного формата: -1 стоит и в амплитуде, и в угле отсутствующего канала.
        # Достаточно проверить == -1 (точное сравнение, т.к. -1 ставится программно).
        mask_missing = (x == -1.0)

        # Заменяем -1 на 0 для безопасных операций
        # TODO: А будет ли 0 безопасен для деления? Не попадёт ли туда?
        x_safe = torch.where(mask_missing, torch.zeros_like(x), x)

        # Добавляем learnable embedding для пропущенных каналов
        x_safe = x_safe + mask_missing.float() * self.missing_token

        return x_safe, mask_missing


# ---------------------------------------------------------------------------
# PhysicalStem: физические операции + KAN → embedding для Transformer
# ---------------------------------------------------------------------------

class PhysicalStem(nn.Module):
    """Входной Stem-блок с физическим индуктивным смещением.

    Реализует rPhysicsKAN-подход:
    1. Разделяет амплитуды и углы
    2. Вычисляет физические признаки (P, Q, Z) с масками безопасности
    3. Формирует KAN-кодирование с гейтированием углами
    4. Линейная проекция в d_model для Transformer

    Входной формат: (B, C, T) где C — чередующиеся [A₁, φ₁, A₂, φ₂, ...]
    Каналы идут парами: первая половина пар = токи, вторая = напряжения.

    Args:
        num_input_channels: полное число каналов (амплитуды + углы)
        d_model: размерность выходного embedding для Transformer
        num_current_pairs: число пар ток (A_I, φ_I) в первой половине каналов
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
        # TODO: надо верно будет сформировать входные данные, не упустить этот момент,
        # чтобы данное утверждение было верным, иначе придётся менять кодировку и логику вычисления физических признаков.
        if num_current_pairs is None:
            num_current_pairs = self.num_pairs // 2
        self.num_current_pairs = num_current_pairs
        self.num_voltage_pairs = self.num_pairs - num_current_pairs

        # Пороги шума
        self.noise_threshold_I = noise_threshold_current
        self.noise_threshold_U = noise_threshold_voltage

        # Число физических признаков на каждую пару ток-напряжение
        # P, Q, Z_amp, Z_phase = 4 признака на пару
        self.num_phys_pairs = min(self.num_current_pairs, self.num_voltage_pairs)
        num_phys_features = self.num_phys_pairs * 4

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

        # Проекция: (half_d_modulated + half_d_angle + phys_features) → d_model
        fusion_dim = half_d + half_d + num_phys_features
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

    def _compute_physics(
        self,
        amplitudes: torch.Tensor,
        angles: torch.Tensor,
        mask_missing: torch.Tensor,
    ) -> torch.Tensor:
        """Вычислить физические признаки: P, Q, Z.

        Args:
            amplitudes: (B, num_pairs, T) — амплитуды
            angles: (B, num_pairs, T) — углы
            mask_missing: (B, C_original, T) — маска пропусков

        Returns:
            phys_features: (B, num_phys*4, T) — [P, Q, Z_amp, Z_phase] для каждой пары I-U
        """
        n_I = self.num_current_pairs
        n_phys = self.num_phys_pairs

        amp_I = amplitudes[:, :n_I, :][:, :n_phys, :]
        amp_U = amplitudes[:, n_I:, :][:, :n_phys, :]
        angle_I = angles[:, :n_I, :][:, :n_phys, :]
        angle_U = angles[:, n_I:, :][:, :n_phys, :]

        # Маска пропуска для амплитуд (проверяем по чётным каналам оригинала)
        mask_amp_I = mask_missing[:, 0::2, :][:, :n_I, :][:, :n_phys, :]
        mask_amp_U = mask_missing[:, 0::2, :][:, n_I:, :][:, :n_phys, :]
        mask_any_missing = mask_amp_I | mask_amp_U

        delta_phi = angle_I - angle_U


        # Псевдо-мощность: P = A_U * A_I * cos(φ_I - φ_U)
        # TODO: Подожди... не надо прям чётко и жёстко задавать формулы для P, Q, Z.
        # Исходная идея в том, чтобы это были блоки производящие умножение в комплексной плоскости и деление любых двух величин друг с другом. Да, каждую с каждой, наверное, очень дорого... Поэтому были желания сделать таких элементов не много, или как-то обыграть через маску внимания в самой КАН модели. Возможно в исходно эмбединге не будет это использовать (хотя может здесь уже не сами блоки, я не до конца пока понял). Причём чтобы можно было как I/U, так и наоборот U/I, прост за счёт каких-то весов модель выбирала, а что же такого мне сейчас посчитать, а другие прост зануляла, т.е. был некий комплексный аргегатор сигналов, который занулял амплитуды не нужных сигналов и/или двигал другие. Есть мысль это сделать вот как:
        #- стандартные КАН блоки даю Х комплексных сигналов (в комплексной форме).
        #- рядом аналогичная цепочка, можно чисто линейную с обрезанием только модулей (угол вообще не трогается) – из них получается Y числе в комплексном представлении из которых первая половина умножается (например у нас 32 сигналов, половина это 16 и значит 1-ый будет умножаться на 9, потом 2-ой на 10-ый и т.д.). А вторая половина уже будет делиться (так же 33 на 49-ый, 34 на 50 и т.д.). И т.е из этих сигналов можно получить ещё Y/2 сигналов на выходе которые посчитались умножением и делением. 
        # Примечание: Числа пока с потолка, вроде бы исходно у нас 64, значит можно будет взять Y=64, чтобы потом получилось 32 сигнала (половина от всего пространства). Ну или 32… Если 64 с учётом углов. В общем это надо пересчитать / проверить
        # П.С. Вероятно раньше у меня тут была ошибка - я описываю, как считаю верным.
        
        P = amp_U * amp_I * torch.cos(delta_phi)
        # Реактивная: Q = A_U * A_I * sin(φ_I - φ_U)
        Q = amp_U * amp_I * torch.sin(delta_phi)

        # Импеданс Z = U / I (безопасное деление)
        I_above_noise = amp_I > self.noise_threshold_I
        safe_amp_I = torch.where(I_above_noise, amp_I, torch.ones_like(amp_I))
        Z_amp = amp_U / safe_amp_I
        Z_phase = angle_U - angle_I

        # Маскируем Z где ток ниже порога шума
        Z_mask = (~I_above_noise) | mask_any_missing
        Z_amp = torch.where(Z_mask, torch.zeros_like(Z_amp), Z_amp)
        Z_phase = torch.where(Z_mask, torch.zeros_like(Z_phase), Z_phase)

        # Маскируем P, Q где каналы отсутствуют
        P = torch.where(mask_any_missing, torch.zeros_like(P), P)
        Q = torch.where(mask_any_missing, torch.zeros_like(Q), Q)

        # Собираем: (B, n_phys*4, T)
        return torch.cat([P, Q, Z_amp, Z_phase], dim=1)

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

        # 2. Физические признаки
        phys_features = self._compute_physics(amplitudes, angles, mask_missing)

        # 3. rPhysicsKAN: KAN для амплитуд + гейтирование углами
        # Переводим в формат (B*T, features) для FastKAN (работает с 2D)
        amp_flat = amplitudes.permute(0, 2, 1).reshape(B * T, -1)  # (B*T, num_pairs)
        ang_flat = angles.permute(0, 2, 1).reshape(B * T, -1)      # (B*T, num_pairs)

        h_amp = self.kan_amp(amp_flat)           # (B*T, half_d)
        g_angle = torch.sigmoid(self.kan_angle_gate(ang_flat))  # (B*T, half_d) — гейт [0, 1]
        h_modulated = h_amp * g_angle            # Модулирование амплитуд углами

        h_angle = self.kan_angle(ang_flat)       # (B*T, half_d)

        # 4. Физические признаки в плоский формат
        phys_flat = phys_features.permute(0, 2, 1).reshape(B * T, -1)  # (B*T, num_phys*4)

        # 5. Объединение и проекция
        fused = torch.cat([h_modulated, h_angle, phys_flat], dim=-1)  # (B*T, fusion_dim)
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
        
        # TODO: Ну и та идея с комплексными числами и всем вот этим - она касалась и вот этой части.
        # Может сейчас корректно тут работает, но всё упомяну на всякий случай, ибо во многом это ключевая идея.
        # Можно её сделать переключаемой из серии "физический блок", "простой КАН блок", "Стандартный МЛП блок". 
        # Если сейчас и так всё норм, то прост скажи об этом, не надо изобретать то, что уже работает.
        x_flat = x.reshape(B * T, D)
        h = self.kan1(x_flat)
        h = self.dropout(h)
        h = self.kan2(h)
        h = self.dropout(h)
        return h.view(B, T, D)


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
