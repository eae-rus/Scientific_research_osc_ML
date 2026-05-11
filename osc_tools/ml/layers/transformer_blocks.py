"""
Строительные блоки Physical KAN-Transformer (Фаза 4).

Представление d_model — полярное (amp | angle): первая половина вектора
содержит амплитудные признаки, вторая — фазовые (угловые). Конвертация
в комплексные числа (re + j·im) происходит ТОЛЬКО внутри
ComplexInteractionBlock для умножения/деления, после чего результат
возвращается в полярную форму (.abs(), .angle()).

Содержит:
- DataSanitizer: обработка отсутствующих каналов (NaN/-1) + learnable MissingToken
- ComplexInteractionBlock: обучаемое комплексное умнож./деление через 4-групповую схему
- ComplexMultiheadAttention: MHA с раздельными проекциями для amp/angle
- PhysicalStem: rPhysicsKAN-кодирование + ComplexInteractionBlock → (amp|angle) embedding
- SinusoidalPositionalEncoding: позиционное кодирование
- KANFeedForward: FastKAN-FFN для Transformer (без физических блоков)
- PhysicalKANFeedForward: KAN-FFN + малый ComplexInteractionBlock (полярная обработка)
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
# AmpOnlyLayerNorm: LayerNorm только для амплитудной половины вектора
# ---------------------------------------------------------------------------

class AmpOnlyLayerNorm(nn.Module):
    """LayerNorm, применяемый только к амплитудной половине полярного вектора.

    Вектор d_model структурирован как (amp | angle). Стандартный LayerNorm
    вычитает среднее и делит на σ по ВСЕМ компонентам, что уничтожает
    физический смысл углов (циклических величин от 0 до 2π).

    Решение: нормализуем ТОЛЬКО первую половину (amp), а угловую половину
    пропускаем без изменений.

    Args:
        d_model: полная размерность вектора (amp + angle)
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        assert d_model % 2 == 0, f"d_model должен быть чётным, получено {d_model}"
        self.d_half = d_model // 2
        self.norm_amp = nn.LayerNorm(self.d_half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, ..., d_model) → (B, ..., d_model). Нормализует только amp-половину."""
        amp = self.norm_amp(x[..., :self.d_half])
        angle = x[..., self.d_half:]
        return torch.cat([amp, angle], dim=-1)


# ---------------------------------------------------------------------------
# DirectionalRelayGate: дифференцируемый направленный орган
# ---------------------------------------------------------------------------

class DirectionalRelayGate(nn.Module):
    """Дифференцируемый направленный орган (мягкое реле).

    Реализует угловую зону срабатывания для каждого канала:
    - В прямой зоне: gate ≈ 1.0 (полная передача амплитуды)
    - В зоне «за спиной»: gate ≈ back_value (ослабление, по умолч. 0.5)
    - Плавный переход на границах (параметризован sharpness)

    Математика:
    1. deviation = angle - mta (отклонение от центра зоны)
    2. cos_dev = cos(deviation) — циклически корректная «проекция»
    3. threshold = cos(zone_width) — граница прямой зоны
    4. raw = sharpness · (cos_dev - threshold)
    5. smooth = sigmoid(raw)  → 0..1
    6. gate = back_value + (1 - back_value) · smooth → [back_value, 1.0]

    Все параметры (mta, zone_width, back_value, sharpness) — обучаемые,
    но с ограничениями через clamp/softplus для физической корректности.

    Args:
        num_channels: число каналов (per-channel параметры)
        default_zone_width: половина ширины прямой зоны, рад (по умолч. 5π/6 = 150°)
        default_back_value: значение gate за спиной (по умолч. 0.5)
        default_sharpness: резкость перехода на границе (по умолч. 10.0)
    """

    def __init__(
        self,
        num_channels: int,
        default_zone_width: float = 5 * math.pi / 6,
        default_back_value: float = 0.5,
        default_sharpness: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels

        # Угол максимальной чувствительности (центр зоны), по umolч. 0
        self.mta = nn.Parameter(torch.zeros(num_channels))

        # Половина ширины зоны (хранится в unconstrained-пространстве для clamp)
        # zone_width ∈ (0, π) — чтобы не превышать полуокружность
        self.zone_width_raw = nn.Parameter(
            torch.full((num_channels,), default_zone_width)
        )

        # Значение за спиной ∈ (0, 1), хранится через sigmoid⁻¹ (logit)
        _back_logit = math.log(default_back_value / (1 - default_back_value + 1e-7))
        self.back_value_logit = nn.Parameter(
            torch.full((num_channels,), _back_logit)
        )

        # Резкость перехода ≥ 1 (через softplus)
        _sharp_raw = math.log(math.exp(default_sharpness) - 1)
        self.sharpness_raw = nn.Parameter(
            torch.full((num_channels,), _sharp_raw)
        )

    def forward(self, angle: torch.Tensor) -> torch.Tensor:
        """Вычисляет gate-коэффициент для каждого канала.

        Args:
            angle: (..., num_channels) — углы в радианах

        Returns:
            gate: (..., num_channels) — значения в [back_value, 1.0]
        """
        # Constrained параметры
        zone_width = self.zone_width_raw.clamp(min=0.01, max=math.pi - 0.01)
        back_value = torch.sigmoid(self.back_value_logit)
        sharpness = F.softplus(self.sharpness_raw) + 1.0  # ≥ 1

        # Отклонение от центра зоны
        deviation = angle - self.mta
        cos_dev = torch.cos(deviation)

        # Порог: cos(zone_width). В прямой зоне cos_dev > threshold
        threshold = torch.cos(zone_width)

        # Плавный переход через sigmoid
        raw = sharpness * (cos_dev - threshold)
        smooth = torch.sigmoid(raw)  # → 1 внутри зоны, → 0 за спиной

        # gate = back_value + (1 - back_value) · smooth → [back_value, 1.0]
        gate = back_value + (1 - back_value) * smooth
        return gate


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
        # Полностью отключаем AMP — ComplexHalf не поддерживается
        with torch.amp.autocast('cuda', enabled=False):
            z = z.to(torch.complex64) if z.is_complex() else z
            w = self.selector.weight.float()  # (num_intermediate, num_input_pairs)
            re_combined = F.linear(z.real.float(), w)  # (B*T, num_intermediate)
            im_combined = F.linear(z.imag.float(), w)  # (B*T, num_intermediate)
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

        # Сжатие амплитуды результатов деления через tanh.
        # Деление может давать на порядки бо́льшие значения (Z = U/I при малом I),
        # что «взрывает» дисперсию и убивает полезные признаки при нормализации.
        # tanh(|z|) ∈ [0, 1) — сохраняет порядок, но ограничивает выбросы.
        # Угол при этом сохраняется без изменений.
        z_div_amp = z_div.abs()
        z_div_angle = z_div.angle()
        z_div = torch.polar(torch.tanh(z_div_amp), z_div_angle)

        return torch.cat([z_mul, z_div], dim=-1)


# ---------------------------------------------------------------------------
# ComplexMultiheadAttention: MHA для комплексно-структурированного d_model
# ---------------------------------------------------------------------------

class ComplexMultiheadAttention(nn.Module):
    """Multi-Head Attention для полярно-структурированных (amp/angle) представлений.

    Вектор d_model структурирован как (amp | angle): первая половина —
    амплитудные признаки, вторая — фазовые (угловые).

    Attention score = Σ_k (Q_amp·K_amp + Q_angle·K_angle):
    Раздельные проекции для amp и angle не смешивают эти две половины.
    Score высокий, когда и амплитудные, и фазовые паттерны похожи.

    Одинаковые веса attention применяются к V_amp и V_angle, гарантируя
    согласованную обработку обеих компонент.

    Параметры: 8 линейных слоёв по d_complex × d_complex (Q/K/V/out × amp/angle).
    При d_model=64 это ~8×16² = 32K params vs ~4×64² = 64K для стандартного MHA.

    Args:
        d_model: полная размерность (= 2 × d_complex)
        num_heads: число голов (d_model / num_heads должно быть чётным)
        dropout: dropout для attention weights
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % 2 == 0, f"d_model должен быть чётным, получено {d_model}"
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) не делится на num_heads ({num_heads})"
        )
        d_head = d_model // num_heads
        assert d_head % 2 == 0, (
            f"d_head={d_head} должен быть чётным (amp + angle в каждой голове)"
        )

        self.d_model = d_model
        self.d_complex = d_model // 2
        self.num_heads = num_heads
        self.d_head_complex = self.d_complex // num_heads

        # Раздельные проекции Q, K, V для amp и angle компонент
        self.W_Q_amp = nn.Linear(self.d_complex, self.d_complex, bias=False)
        self.W_Q_angle = nn.Linear(self.d_complex, self.d_complex, bias=False)
        self.W_K_amp = nn.Linear(self.d_complex, self.d_complex, bias=False)
        self.W_K_angle = nn.Linear(self.d_complex, self.d_complex, bias=False)
        self.W_V_amp = nn.Linear(self.d_complex, self.d_complex, bias=False)
        self.W_V_angle = nn.Linear(self.d_complex, self.d_complex, bias=False)

        # Выходная проекция (раздельно)
        self.out_proj_amp = nn.Linear(self.d_complex, self.d_complex, bias=False)
        self.out_proj_angle = nn.Linear(self.d_complex, self.d_complex, bias=False)

        self.dropout = nn.Dropout(dropout)
        # Score суммирует 2 * d_head_complex слагаемых → масштаб 1/sqrt(d_head)
        self.scale = (self.d_head_complex * 2) ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query, key, value: (B, T, d_model) — первая половина amp, вторая angle
            key_padding_mask: (B, T) — True для игнорируемых позиций
            need_weights: возвращать ли веса attention
            attn_mask: дополнительная маска (для совместимости)

        Returns:
            output: (B, T, d_model) — структурированный как (amp | angle)
            attn_weights: (B, num_heads, T, T) или None
        """
        B, T, _ = query.shape
        dc = self.d_complex
        nh = self.num_heads
        dhc = self.d_head_complex

        # Разделяем на amp / angle
        q_amp, q_angle = query[:, :, :dc], query[:, :, dc:]
        k_amp, k_angle = key[:, :, :dc], key[:, :, dc:]
        v_amp, v_angle = value[:, :, :dc], value[:, :, dc:]

        # Проекции Q, K, V → (B, num_heads, T, d_head_complex)
        def to_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, T, nh, dhc).transpose(1, 2)

        Q_amp = to_heads(self.W_Q_amp(q_amp))
        Q_angle = to_heads(self.W_Q_angle(q_angle))
        K_amp = to_heads(self.W_K_amp(k_amp))
        K_angle = to_heads(self.W_K_angle(k_angle))
        V_amp = to_heads(self.W_V_amp(v_amp))
        V_angle = to_heads(self.W_V_angle(v_angle))

        # Score = Σ(Q_amp·K_amp + Q_angle·K_angle) — раздельное скалярное произведение
        score = (
            Q_amp @ K_amp.transpose(-2, -1) + Q_angle @ K_angle.transpose(-2, -1)
        ) * self.scale  # (B, num_heads, T, T)

        # Маски
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) → (B, 1, 1, T)
            score = score.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        if attn_mask is not None:
            score = score + attn_mask

        attn_weights = torch.softmax(score, dim=-1)
        # Защита от NaN: если ВСЕ key замаскированы, softmax(-inf,...,-inf)=NaN
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        # Одинаковые веса для amp и angle → согласованная обработка компонент
        out_amp = (attn_weights @ V_amp).transpose(1, 2).reshape(B, T, dc)
        out_angle = (attn_weights @ V_angle).transpose(1, 2).reshape(B, T, dc)

        # Выходная проекция (раздельно для amp и angle)
        out_amp = self.out_proj_amp(out_amp)
        out_angle = self.out_proj_angle(out_angle)

        output = torch.cat([out_amp, out_angle], dim=-1)

        if need_weights:
            return output, attn_weights
        return output, None


# ---------------------------------------------------------------------------
# PhysicalStem: физические операции + KAN → embedding для Transformer
# ---------------------------------------------------------------------------

class PhysicalStem(nn.Module):
    """Входной Stem-блок с физическим индуктивным смещением.

    Архитектура (переработанная):
    1. Разделяет амплитуды и углы из чередующегося формата [A₁,φ₁, A₂,φ₂, ...]
    2. ComplexInteractionBlock: обучаемое комплексное умножение/деление
       (с tanh-сжатием результатов деления для стабильности)
    3. KAN для амплитуд (нелинейные уставки по отфильтрованным амплитудам)
    4. Углы НЕ проходят через KAN/линейные слои — только обучаемый сдвиг
       phase_shift_b (поворот вектора в комплексной плоскости, аналогично MTA)
    5. DirectionalRelayGate (опционально): мягкое угловое реле, модулирующее
       амплитуды в зависимости от зоны срабатывания
    6. Раздельная проекция в (amp | angle) для d_model
    7. AmpOnlyLayerNorm (опционально, по умолчанию ВЫКЛ): нормализует только amp

    Args:
        num_input_channels: полное число каналов (амплитуды + углы)
        d_model: размерность выходного embedding для Transformer
        num_current_pairs: число пар ток (A_I, φ_I) в первой половине каналов
        num_interaction_pairs: число выходных пар от ComplexInteractionBlock
        noise_threshold_current: порог шума для токов (отн. ед.)
        noise_threshold_voltage: порог шума для напряжений (отн. ед.)
        kan_grid_size: число узлов RBF-сетки для FastKAN
        use_angle_gate: включить DirectionalRelayGate (по умолч. True)
        use_layer_norm: включить AmpOnlyLayerNorm на выходе (по умолч. False)
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
        use_angle_gate: bool = True,
        use_layer_norm: bool = False,
        dropout: float = 0.1,
        # --- Параметры абляции ---
        disable_interaction: bool = False,
        disable_kan: bool = False,
        disable_phase_shift: bool = False,
    ) -> None:
        super().__init__()

        assert num_input_channels % 2 == 0, (
            f"num_input_channels должно быть чётным (A+φ пары), получено {num_input_channels}"
        )

        self.num_input_channels = num_input_channels
        self.num_pairs = num_input_channels // 2  # Количество (A, φ) пар
        self.d_model = d_model
        self.use_angle_gate = use_angle_gate

        # Флаги абляции: runtime-переключение через set_ablation()
        self.disable_interaction = disable_interaction
        self.disable_kan = disable_kan
        self.disable_phase_shift = disable_phase_shift

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
        # Число признаков от interaction: только amp (углы идут отдельно)
        num_interaction_amp_features = num_interaction_pairs

        # --- Обучаемый сдвиг фазы (поворот вектора) для каждого сигнала ---
        # phase_shift_b — аналог MTA (Maximum Torque Angle) в релейной защите
        self.phase_shift_b = nn.Parameter(torch.zeros(self.num_pairs))

        # --- DirectionalRelayGate (опционально) ---
        if self.use_angle_gate:
            self.angle_gate = DirectionalRelayGate(num_channels=self.num_pairs)

        # --- KAN для амплитуд → d_model // 2 ---
        half_d = d_model // 2
        num_amplitudes = self.num_pairs
        self.kan_amp = FastKANLayer(
            input_dim=num_amplitudes,
            output_dim=half_d,
            num_grids=kan_grid_size,
            use_base_update=True,
            use_layernorm=False,
        )
        # Линейный fallback для абляции без KAN
        self.linear_amp_fallback = nn.Linear(num_amplitudes, half_d)

        # --- Проекция в amp/angle структуру d_model ---
        d_complex = d_model // 2
        # amp-путь: KAN-выход + interaction amps
        amp_fusion_dim = half_d + num_interaction_amp_features
        self.proj_amp = nn.Linear(amp_fusion_dim, d_complex)
        # angle-путь: сырые углы (с phase_shift) + interaction angles
        angle_fusion_dim = self.num_pairs + num_interaction_pairs
        self.proj_angle = nn.Linear(angle_fusion_dim, d_complex, bias=False)

        # --- Нормализация (опционально, по умолч. ВЫКЛ) ---
        self.norm: nn.Module
        if use_layer_norm:
            self.norm = AmpOnlyLayerNorm(d_model)
        else:
            self.norm = nn.Identity()

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

    def set_ablation(
        self,
        disable_interaction: bool | None = None,
        disable_kan: bool | None = None,
        disable_phase_shift: bool | None = None,
        disable_angle_gate: bool | None = None,
    ) -> None:
        """Runtime-переключение абляционных флагов (Q4).

        Позволяет на уже обученной модели отключить физ. блоки
        для inference-абляции без переобучения.
        Передайте None, чтобы оставить текущее значение.
        """
        if disable_interaction is not None:
            self.disable_interaction = disable_interaction
        if disable_kan is not None:
            self.disable_kan = disable_kan
        if disable_phase_shift is not None:
            self.disable_phase_shift = disable_phase_shift
        if disable_angle_gate is not None:
            self.use_angle_gate = not disable_angle_gate

    def forward(
        self, x: torch.Tensor, mask_missing: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) — очищенные данные после DataSanitizer
            mask_missing: (B, C, T) — маска пропусков

        Returns:
            embedding: (B, T, d_model) — готовый для Transformer вход

        Note:
            mask_missing пока не используется внутри Stem — DataSanitizer уже
            заменил NaN→0 + learnable missing_token. Маска пробрасывается
            далее в Transformer как key_padding_mask (all_missing_mask).
            TODO: рассмотреть маскировку gate/KAN для пропущенных каналов,
            чтобы нулевые амплитуды не вносили шум через нелинейности.
        """
        # Отключаем AMP для всего Stem — внутри комплексные операции (не FP16)
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            B, C, T = x.shape

            # 1. Разделить на амплитуды и углы
            amplitudes, angles = self._extract_amp_angle(x)

            # 2. Переводим в формат (B*T, features) для FastKAN и interaction
            amp_flat = amplitudes.permute(0, 2, 1).reshape(B * T, -1)
            ang_flat = angles.permute(0, 2, 1).reshape(B * T, -1)

            # 3. Обучаемый сдвиг фазы (поворот вектора = MTA)
            if self.disable_phase_shift:
                rotated_angle = ang_flat
            else:
                rotated_angle = ang_flat + self.phase_shift_b

            # 4. DirectionalRelayGate: модулируем амплитуды по углу
            if self.use_angle_gate:
                angle_coeff = self.angle_gate(rotated_angle)
                gated_amp = amp_flat * angle_coeff
            else:
                gated_amp = amp_flat

            # 5. Обучаемый комплексный блок: z = A·exp(jφ)
            if self.disable_interaction:
                # Абляция: нулевые выходы interaction (сохраняем форму для concat)
                inter_amp = torch.zeros(B * T, self.interaction.num_interaction_pairs,
                                        device=x.device, dtype=x.dtype)
                inter_angle = torch.zeros_like(inter_amp)
            else:
                z_input = torch.polar(amp_flat, rotated_angle)  # complex64
                z_inter = self.interaction(z_input)
                inter_amp = z_inter.abs()     # (B*T, num_interaction_pairs)
                inter_angle = z_inter.angle() # (B*T, num_interaction_pairs)

            # 6. KAN для амплитуд (уже отфильтрованных через gate)
            if self.disable_kan:
                h_amp = self.linear_amp_fallback(gated_amp)
            else:
                h_amp = self.kan_amp(gated_amp)

            # 7. Объединение и раздельная проекция в (amp, angle)
            fused_amp = torch.cat([h_amp, inter_amp], dim=-1)
            fused_angle = torch.cat([rotated_angle, inter_angle], dim=-1)

            emb_amp = self.proj_amp(fused_amp)
            emb_angle = self.proj_angle(fused_angle)

            embedding = torch.cat([emb_amp, emb_angle], dim=-1)
            embedding = self.norm(embedding)
            embedding = self.dropout(embedding)

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
# PhysicalKANFeedForward: KAN-FFN + ComplexInteractionBlock (компл.структура)
# ---------------------------------------------------------------------------

class PhysicalKANFeedForward(nn.Module):
    """KAN-FFN с полярно-структурированной обработкой (amp/angle).

    d_model структурирован как (amp | angle): первая половина — амплитудные
    признаки, вторая — фазовые (угловые).

    Переработанная архитектура:
    1. KAN_amp: d_complex → d_ff/2 → d_complex (нелинейная обработка амплитуд)
    2. Угол: НЕ проходит через KAN — только обучаемый сдвиг phase_shift_b
       (для residual: angle_new = angle_old + phase_shift_b)
    3. DirectionalRelayGate (опционально): угол управляет амплитудой
    4. ComplexInteractionBlock: малый блок комплексного умножения/деления
       (вклад в amp-путь через гейтированное сложение)

    Args:
        d_model: размерность входа/выхода (должен быть чётным)
        d_ff: размерность скрытого слоя KAN (по умолчанию 4 * d_model).
        num_interaction_pairs: число выходных пар ComplexInteractionBlock
        kan_grid_size: число узлов RBF-сетки FastKAN
        use_angle_gate: включить DirectionalRelayGate (по умолч. True)
        dropout: dropout
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        num_interaction_pairs: int = 4,
        kan_grid_size: int = 5,
        use_angle_gate: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % 2 == 0, f"d_model должен быть чётным, получено {d_model}"

        d_complex = d_model // 2
        if d_ff is None:
            d_ff = d_model * 4
        d_ff_half = d_ff // 2

        self.d_model = d_model
        self.d_complex = d_complex
        self.use_angle_gate = use_angle_gate

        # --- KAN для амплитудных признаков ---
        self.kan_amp1 = FastKANLayer(
            input_dim=d_complex, output_dim=d_ff_half,
            num_grids=kan_grid_size, use_base_update=True, use_layernorm=False,
        )
        self.kan_amp2 = FastKANLayer(
            input_dim=d_ff_half, output_dim=d_complex,
            num_grids=kan_grid_size, use_base_update=True, use_layernorm=False,
        )

        self.dropout = nn.Dropout(dropout)

        # --- Обучаемый сдвиг фазы (поворот, аналог MTA) ---
        self.phase_shift_b = nn.Parameter(torch.zeros(d_complex))

        # --- DirectionalRelayGate (опционально) ---
        if self.use_angle_gate:
            self.angle_gate = DirectionalRelayGate(num_channels=d_complex)

        # --- Комплексный путь (малый) ---
        self.interaction = ComplexInteractionBlock(
            num_input_pairs=d_complex,
            num_interaction_pairs=num_interaction_pairs,
        )
        # Проекция interaction amp → d_complex (для amp-пути)
        self.interaction_proj_amp = nn.Linear(num_interaction_pairs, d_complex)
        # Гейт: sigmoid(-2) ≈ 0.12 на старте — комплексный путь включается мягко
        self.interaction_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) — первая половина: amp, вторая: angle
        Returns:
            (B, T, d_model)
        """
        # Отключаем AMP — комплексные операции не поддерживают FP16
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            B, T, D = x.shape
            dc = self.d_complex
            x_flat = x.reshape(B * T, D)

            x_amp = x_flat[:, :dc]
            x_angle = x_flat[:, dc:]

            # KAN для амплитудных признаков
            h_amp = self.kan_amp1(x_amp)
            h_amp = self.dropout(h_amp)
            h_amp = self.kan_amp2(h_amp)
            h_amp = self.dropout(h_amp)

            # DirectionalRelayGate: угол модулирует amp
            if self.use_angle_gate:
                gate = self.angle_gate(x_angle)
                h_amp = h_amp * gate

            # Комплексный путь: amp/angle → polar complex → interaction
            z_in = torch.polar(x_amp, x_angle)
            z_out = self.interaction(z_in)

            inter_proj_amp = self.interaction_proj_amp(z_out.abs())

            # Гейтированное сложение (только для amp)
            gate_inter = torch.sigmoid(self.interaction_gate)
            h_amp = h_amp + gate_inter * inter_proj_amp

            # Угловой путь: только обучаемый сдвиг (delta для residual в encoder)
            h_angle = self.phase_shift_b.unsqueeze(0).expand(B * T, -1)

            combined = torch.cat([h_amp, h_angle], dim=-1)
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

    Pre-LN: Norm → Attention → residual → Norm → FFN → residual.
    Стабильнее Post-LN для глубоких моделей.

    По умолчанию использует AmpOnlyLayerNorm (нормализует только
    амплитудную половину вектора, углы проходят без изменений).
    Через параметр use_mixed_layer_norm можно включить стандартный
    nn.LayerNorm, который нормализует весь вектор (amp+angle).

    Args:
        d_model: размерность модели
        num_heads: число голов внимания
        ffn: Feed-Forward модуль (KANFeedForward или MLPFeedForward)
        attn: модуль Attention (если None — стандартный MHA)
        use_mixed_layer_norm: True → стандартный nn.LayerNorm на всём d_model
            False (по умолч.) → AmpOnlyLayerNorm (норм. только amp-половину)
        dropout: dropout для attention
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn: nn.Module,
        attn: nn.Module | None = None,
        use_mixed_layer_norm: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Выбор нормализации: AmpOnly (по умолч.) или стандартный LayerNorm
        if use_mixed_layer_norm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
        else:
            self.ln1 = AmpOnlyLayerNorm(d_model)
            self.ln2 = AmpOnlyLayerNorm(d_model)

        if attn is not None:
            self.attn = attn
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
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
