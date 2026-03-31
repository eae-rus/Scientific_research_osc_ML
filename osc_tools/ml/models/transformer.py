"""
Physical KAN-Transformer для Фазы 4.

Единая модель, объединяющая:
- DataSanitizer: обработка отсутствующих каналов (NaN)
- PhysicalStem: физический индуктивный сдвиг (rPhysicsKAN) → embedding
- SinusoidalPositionalEncoding: позиционное кодирование
- TransformerEncoder: стек блоков с KAN-FFN или MLP-FFN
- SSL Head: реконструкция спектральных признаков
- Classification Head: для fine-tuning (зональная классификация)

Основная модель: PhysicalKANTransformer
Baseline для сравнения: BaselineTransformer (MLP Stem + MLP FFN)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from osc_tools.ml.models.base import BaseModel
from osc_tools.ml.layers.transformer_blocks import (
    AmpOnlyLayerNorm,
    ComplexMultiheadAttention,
    DataSanitizer,
    KANFeedForward,
    MLPFeedForward,
    PhysicalKANFeedForward,
    PhysicalStem,
    SinusoidalPositionalEncoding,
    TransformerEncoderBlock,
)

# FastKAN для классификационной головы
import sys as _sys
import os as _os
_fast_kan_path = _os.path.join(_os.path.dirname(__file__), '..', 'fast-kan-master')
if _fast_kan_path not in _sys.path:
    _sys.path.insert(0, _fast_kan_path)
from fastkan import FastKANLayer  # noqa: E402


class KANClassificationHead(nn.Module):
    """Классификационная голова на основе FastKAN.

    Заменяет nn.Linear: d_model → d_hidden (KAN) → num_classes (Linear).
    Обеспечивает интерпретируемость за счёт сплайновых активаций.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        kan_grid_size: int = 5,
        d_hidden: int | None = None,
    ) -> None:
        super().__init__()
        if d_hidden is None:
            d_hidden = d_model * 2
        self.kan = FastKANLayer(
            input_dim=d_model,
            output_dim=d_hidden,
            num_grids=kan_grid_size,
            use_base_update=True,
            use_layernorm=False,  # False: final_norm уже нормализует вход
        )
        # Финальная линейная проекция на классы (без нелинейности — logits)
        self.proj = nn.Linear(d_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, ..., d_model) → (B, ..., num_classes). Поддерживает 2D и 3D вход."""
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])  # (N, d_model)
        h = self.kan(x_flat)               # (N, d_hidden)
        out = self.proj(h)                 # (N, num_classes)
        return out.view(*shape[:-1], -1)   # восстанавливаем форму


class MLPClassificationHead(nn.Module):
    """Простая MLP-голова для абляции."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        d_hidden: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_hidden is None:
            d_hidden = d_model * 2
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_cls_head(
    cls_head_type: str,
    d_model: int,
    num_classes: int,
    kan_grid_size: int,
    dropout: float,
) -> nn.Module:
    """Фабрика классификационных голов для абляций и подбора архитектуры."""
    if cls_head_type == 'linear':
        return nn.Linear(d_model, num_classes)
    if cls_head_type == 'mlp':
        return MLPClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout,
        )
    if cls_head_type == 'kan':
        return KANClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            kan_grid_size=kan_grid_size,
        )
    raise ValueError(f"Неизвестный cls_head_type: {cls_head_type}")


class PhysicalKANTransformer(BaseModel):
    """Physical KAN-Transformer с физическим Stem и KAN-FFN.

    Архитектура:
    1. Sanitizer → PhysicalStem (rPhysicsKAN + DirectionalRelayGate) → d_model
    2. + Positional Encoding
    3. N × TransformerEncoderBlock (ComplexMHA + PhysicalKANFeedForward)
       В каждом блоке: KAN-FFN (только для амплитуд) + ComplexInteractionBlock
       + DirectionalRelayGate (опционально). Углы обрабатываются ТОЛЬКО
       как обучаемый сдвиг (phase_shift_b), без KAN/линейных преобразований.
    4. SSL Head (реконструкция) или Classification Head (fine-tuning)

    Ключевые принципы (из анализа в Thoughts_on_recycling_AI_transformers.md):
    - Углы НИКОГДА не проходят через KAN/линейные слои (только сдвиг b)
    - Углы НИКОГДА не нормализуются (AmpOnlyLayerNorm по умолч.)
    - Результаты деления сжимаются через tanh для стабильности
    - DirectionalRelayGate: дифференцируемый направленный орган (мягкое реле)

    Args:
        num_input_channels: число входных каналов (чередующихся A, φ пар)
        d_model: размерность скрытого состояния Transformer
        num_heads: число голов внимания
        num_layers: число блоков Transformer Encoder
        d_ff: размерность скрытого слоя FFN (None → 4*d_model)
        num_current_pairs: число пар тока в первой половине каналов
        num_classes: число классов для Classification Head (None → SSL-режим)
        ssl_output_channels: число каналов на выходе SSL Head (None → num_input_channels)
        zone_size: размер зоны для зональной классификации
        kan_grid_size: размер RBF-сетки в FastKAN
        noise_threshold_current: порог шума токов
        noise_threshold_voltage: порог шума напряжений
        num_stem_interaction_pairs: число пар в ComplexInteractionBlock Stem (16)
        num_ffn_interaction_pairs: число пар в ComplexInteractionBlock FFN (4)
        use_physical_ffn: True → PhysicalKANFeedForward, False → KANFeedForward
        use_angle_gate: включить DirectionalRelayGate в Stem и FFN (по умолч. True)
        use_mixed_layer_norm: True → стандартный LayerNorm, False → AmpOnlyLayerNorm
        cls_head_type: тип классификационной головы ('kan'|'mlp'|'linear')
        dropout: dropout для всех слоёв
        max_seq_len: максимальная длина последовательности для PE
    """

    def __init__(
        self,
        num_input_channels: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int | None = None,
        num_current_pairs: int | None = None,
        num_classes: int | None = None,
        ssl_output_channels: int | None = None,
        zone_size: int = 16,
        kan_grid_size: int = 5,
        noise_threshold_current: float = 1.0 / 2000,
        noise_threshold_voltage: float = 1.0 / 300,
        num_stem_interaction_pairs: int = 16,
        num_ffn_interaction_pairs: int = 4,
        use_physical_ffn: bool = True,
        use_angle_gate: bool = True,
        use_mixed_layer_norm: bool = False,
        cls_head_type: str = 'kan',
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()

        self.num_input_channels = num_input_channels
        self.d_model = d_model
        self.num_classes = num_classes
        self.zone_size = zone_size

        # --- 1. Sanitizer (NaN маркер отсутствующих каналов) ---
        self.sanitizer = DataSanitizer(num_channels=num_input_channels)

        # --- 2. Physical Stem → (B, T, d_model) ---
        self.stem = PhysicalStem(
            num_input_channels=num_input_channels,
            d_model=d_model,
            num_current_pairs=num_current_pairs,
            num_interaction_pairs=num_stem_interaction_pairs,
            noise_threshold_current=noise_threshold_current,
            noise_threshold_voltage=noise_threshold_voltage,
            kan_grid_size=kan_grid_size,
            use_angle_gate=use_angle_gate,
            use_layer_norm=use_mixed_layer_norm,
            dropout=dropout,
        )

        # --- 3. Positional Encoding ---
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=max_seq_len, dropout=dropout
        )

        # --- 4. Transformer Encoder (Physical KAN-FFN + ComplexMHA) ---
        self.encoder_blocks = nn.ModuleList()
        for _ in range(num_layers):
            complex_attn = ComplexMultiheadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
            if use_physical_ffn:
                ffn = PhysicalKANFeedForward(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_interaction_pairs=num_ffn_interaction_pairs,
                    kan_grid_size=kan_grid_size,
                    use_angle_gate=use_angle_gate,
                    dropout=dropout,
                )
            else:
                ffn = KANFeedForward(
                    d_model=d_model,
                    d_ff=d_ff,
                    kan_grid_size=kan_grid_size,
                    dropout=dropout,
                )
            block = TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn=ffn,
                attn=complex_attn,
                use_mixed_layer_norm=use_mixed_layer_norm,
                dropout=dropout,
            )
            self.encoder_blocks.append(block)

        # Финальная нормализация (AmpOnly по умолчанию)
        if use_mixed_layer_norm:
            self.final_norm: nn.Module = nn.LayerNorm(d_model)
        else:
            self.final_norm = AmpOnlyLayerNorm(d_model)

        # --- 5. Heads ---
        ssl_out = ssl_output_channels if ssl_output_channels else num_input_channels
        self.ssl_head = nn.Linear(d_model, ssl_out)

        self.cls_head: nn.Module | None = None
        if num_classes is not None:
            self.cls_head = _build_cls_head(
                cls_head_type=cls_head_type,
                d_model=d_model,
                num_classes=num_classes,
                kan_grid_size=kan_grid_size,
                dropout=dropout,
            )

    def set_ablation(self, **kwargs) -> None:
        """Переключает абляционные флаги в PhysicalStem.

        Поддерживаемые kwargs:
            disable_interaction, disable_kan, disable_phase_shift, disable_angle_gate
        """
        self.stem.set_ablation(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = 'ssl',
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, T) — входные спектральные признаки
            mode: 'ssl' — реконструкция, 'classify' — классификация по зонам

        Returns:
            dict с ключами:
                'ssl': (B, C_out, T) — реконструированные признаки (в режиме ssl)
                'classify': (B, num_zones, num_classes) — классификация по зонам
                'features': (B, T, d_model) — latent представление (всегда)
        """
        # 1. Sanitizer: обработка NaN (отсутствующие каналы)
        x_safe, mask_missing = self.sanitizer(x)

        # 2. Physical Stem → (B, T, d_model)
        embedding = self.stem(x_safe, mask_missing)

        # 3. Positional Encoding
        embedding = self.pos_encoder(embedding)

        # 4. Создаём attention padding mask из mask_missing
        # Если ВСЕ каналы на данном шаге = missing → маскируем
        all_missing_mask = mask_missing.all(dim=1)  # (B, T)

        # 5. Transformer Encoder
        h = embedding
        for block in self.encoder_blocks:
            h = block(h, key_padding_mask=all_missing_mask)

        h = self.final_norm(h)  # (B, T, d_model)

        result: dict[str, torch.Tensor] = {'features': h}

        if mode == 'ssl':
            # SSL: реконструкция по каждому временному шагу
            ssl_out = self.ssl_head(h)  # (B, T, C_out)
            ssl_out = ssl_out.permute(0, 2, 1)  # (B, C_out, T)
            result['ssl'] = ssl_out

        elif mode == 'classify' and self.cls_head is not None:
            # Зональная классификация: усреднение по зонам
            # Примечание: zone_size должен совпадать с downsampling_stride данных.
            # При stride=16 (полпериода) каждый временной шаг = одна зона.
            B, T, D = h.shape
            num_zones = T // self.zone_size
            if num_zones > 0:
                # Обрезаем до кратного zone_size
                h_trimmed = h[:, :num_zones * self.zone_size, :]
                # Группируем по зонам и усредняем
                h_zoned = h_trimmed.view(B, num_zones, self.zone_size, D).mean(dim=2)
                cls_out = self.cls_head(h_zoned)  # (B, num_zones, num_classes)
            else:
                # Если T < zone_size, используем всё окно как одну зону
                h_pooled = h.mean(dim=1, keepdim=True)  # (B, 1, D)
                cls_out = self.cls_head(h_pooled)        # (B, 1, num_classes)
            result['classify'] = cls_out

        return result


class BaselineTransformer(BaseModel):
    """Baseline Transformer без KAN и физических блоков.

    Для корректного сравнения с PhysicalKANTransformer:
    - MLPStem вместо PhysicalStem
    - MLP-FFN вместо KAN-FFN
    - Тот же Attention, PE, d_model

    Args:
        num_input_channels: число входных каналов
        d_model: размерность скрытого состояния
        num_heads: число голов внимания
        num_layers: число блоков Encoder
        d_ff: размерность FFN
        num_classes: число классов (None → SSL)
        ssl_output_channels: число каналов на выходе SSL Head
        zone_size: размер зоны классификации
        dropout: dropout
        max_seq_len: максимальная длина последовательности
    """

    def __init__(
        self,
        num_input_channels: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int | None = None,
        num_classes: int | None = None,
        ssl_output_channels: int | None = None,
        zone_size: int = 16,
        cls_head_type: str = 'linear',
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()

        self.num_input_channels = num_input_channels
        self.d_model = d_model
        self.num_classes = num_classes
        self.zone_size = zone_size

        # --- Sanitizer (NaN маркер отсутствующих каналов) ---
        self.sanitizer = DataSanitizer(num_channels=num_input_channels)

        # --- MLP Stem: простая линейная проекция каждого временного шага ---
        self.stem = nn.Sequential(
            nn.Linear(num_input_channels, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # --- Positional Encoding ---
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=max_seq_len, dropout=dropout
        )

        # --- Transformer Encoder (MLP-FFN) ---
        self.encoder_blocks = nn.ModuleList()
        for _ in range(num_layers):
            ffn = MLPFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
            block = TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn=ffn,
                use_mixed_layer_norm=True,  # Baseline не имеет полярной структуры
                dropout=dropout,
            )
            self.encoder_blocks.append(block)

        self.final_norm = nn.LayerNorm(d_model)

        # --- Heads ---
        ssl_out = ssl_output_channels if ssl_output_channels else num_input_channels
        self.ssl_head = nn.Linear(d_model, ssl_out)

        self.cls_head: nn.Module | None = None
        if num_classes is not None:
            self.cls_head = _build_cls_head(
                cls_head_type=cls_head_type,
                d_model=d_model,
                num_classes=num_classes,
                kan_grid_size=5,
                dropout=dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        mode: str = 'ssl',
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, T)
            mode: 'ssl' или 'classify'

        Returns:
            dict аналогично PhysicalKANTransformer
        """
        # 1. Sanitizer
        x_safe, mask_missing = self.sanitizer(x)

        # 2. MLP Stem: (B, C, T) → (B, T, d_model)
        x_t = x_safe.permute(0, 2, 1)  # (B, T, C)
        embedding = self.stem(x_t)      # (B, T, d_model)

        # 3. PE
        embedding = self.pos_encoder(embedding)

        # 4. Attention mask
        all_missing_mask = mask_missing.all(dim=1)

        # 5. Encoder
        h = embedding
        for block in self.encoder_blocks:
            h = block(h, key_padding_mask=all_missing_mask)

        h = self.final_norm(h)

        result: dict[str, torch.Tensor] = {'features': h}

        if mode == 'ssl':
            ssl_out = self.ssl_head(h).permute(0, 2, 1)
            result['ssl'] = ssl_out
        elif mode == 'classify' and self.cls_head is not None:
            B, T, D = h.shape
            num_zones = T // self.zone_size
            if num_zones > 0:
                h_trimmed = h[:, :num_zones * self.zone_size, :]
                h_zoned = h_trimmed.view(B, num_zones, self.zone_size, D).mean(dim=2)
                cls_out = self.cls_head(h_zoned)
            else:
                h_pooled = h.mean(dim=1, keepdim=True)
                cls_out = self.cls_head(h_pooled)
            result['classify'] = cls_out

        return result
