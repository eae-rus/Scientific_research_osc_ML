"""
Тесты для компонентов Фазы 4: Physical KAN-Transformer.

Layer 3: Experimental Zone (Smoke Tests Only)
- Forward pass без ошибок
- Форма выходных тензоров корректна
- Маскирование работает
- Loss функции вычисляются без NaN
- НЕ тестируем: веса, точность, конвергенцию
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.layers.transformer_blocks import (
    ComplexInteractionBlock,
    ComplexMultiheadAttention,
    DataSanitizer,
    KANFeedForward,
    MLPFeedForward,
    PhysicalKANFeedForward,
    PhysicalStem,
    SinusoidalPositionalEncoding,
    TransformerEncoderBlock,
)
from osc_tools.ml.models.transformer import (
    BaselineTransformer,
    PhysicalKANTransformer,
)
from osc_tools.ml.losses import ComplexMSELoss, SpectralReconstructionLoss


# ============================================================
# DataSanitizer
# ============================================================

class TestDataSanitizer:
    """Тесты Sanitizer для обработки отсутствующих каналов."""

    def test_instantiation(self):
        sanitizer = DataSanitizer(num_channels=16)
        assert isinstance(sanitizer, nn.Module)

    def test_instantiation_with_marker(self):
        sanitizer = DataSanitizer(num_channels=16, missing_marker=-1.0)
        assert sanitizer.missing_marker == -1.0

    def test_forward_shape(self):
        sanitizer = DataSanitizer(num_channels=16)
        x = torch.randn(2, 16, 10)
        x_safe, mask = sanitizer(x)
        assert x_safe.shape == x.shape
        assert mask.shape == x.shape

    def test_nan_default_marker(self):
        """По умолчанию детектируются NaN-значения."""
        sanitizer = DataSanitizer(num_channels=8)
        x = torch.randn(1, 8, 5)
        x[0, 0, 0] = float('nan')
        x[0, 3, 2] = float('nan')

        x_safe, mask = sanitizer(x)

        assert mask[0, 0, 0].item() is True
        assert mask[0, 3, 2].item() is True
        assert mask[0, 1, 0].item() is False
        assert not torch.isnan(x_safe).any()

    def test_numeric_marker(self):
        """Числовой маркер (-1) детектируется при передаче missing_marker=-1.0."""
        sanitizer = DataSanitizer(num_channels=8, missing_marker=-1.0)
        x = torch.randn(1, 8, 5)
        x[0, 0, 0] = -1.0
        x[0, 3, 2] = -1.0

        x_safe, mask = sanitizer(x)

        assert mask[0, 0, 0].item() is True
        assert mask[0, 3, 2].item() is True
        assert mask[0, 1, 0].item() is False
        assert x_safe[0, 0, 0].item() != -1.0

    def test_no_nan(self):
        sanitizer = DataSanitizer(num_channels=16, missing_marker=-1.0)
        x = torch.randn(4, 16, 20)
        x[:, :4, :5] = -1.0
        x_safe, mask = sanitizer(x)
        assert not torch.isnan(x_safe).any()


# ============================================================
# ComplexInteractionBlock
# ============================================================

class TestComplexInteractionBlock:
    """Тесты обучаемого комплексного блока взаимодействий (complex-интерфейс)."""

    def test_instantiation(self):
        block = ComplexInteractionBlock(num_input_pairs=8, num_interaction_pairs=16)
        assert isinstance(block, nn.Module)

    def test_forward_shape(self):
        block = ComplexInteractionBlock(num_input_pairs=8, num_interaction_pairs=16)
        z = torch.randn(20, 8) + 1j * torch.randn(20, 8)  # complex input
        z_out = block(z)
        assert z_out.shape == (20, 16)
        assert z_out.is_complex()

    def test_polar_roundtrip(self):
        """Проверка работы с полярными входами (как в PhysicalStem)."""
        block = ComplexInteractionBlock(num_input_pairs=8, num_interaction_pairs=16)
        amp = torch.randn(10, 8).abs()
        ang = torch.randn(10, 8)
        z = torch.polar(amp, ang)
        z_out = block(z)
        # Выходные амплитуды неотрицательны
        assert (z_out.abs() >= 0).all()

    def test_no_nan_with_zeros(self):
        block = ComplexInteractionBlock(num_input_pairs=4, num_interaction_pairs=8)
        z = torch.zeros(5, 4, dtype=torch.complex64)
        z_out = block(z)
        assert not torch.isnan(z_out.real).any()
        assert not torch.isnan(z_out.imag).any()

    def test_even_pairs_required(self):
        """Нечётное num_interaction_pairs должно вызвать AssertionError."""
        with pytest.raises(AssertionError):
            ComplexInteractionBlock(num_input_pairs=8, num_interaction_pairs=7)


# ============================================================
# PhysicalKANFeedForward
# ============================================================

class TestPhysicalKANFeedForward:
    """Тесты PhysicalKANFeedForward (KAN + ComplexInteractionBlock в FFN)."""

    def test_instantiation(self):
        ffn = PhysicalKANFeedForward(d_model=32, kan_grid_size=3)
        assert isinstance(ffn, nn.Module)

    def test_forward_shape(self):
        ffn = PhysicalKANFeedForward(d_model=32, d_ff=64, kan_grid_size=3)
        x = torch.randn(2, 10, 32)
        out = ffn(x)
        assert out.shape == (2, 10, 32)

    def test_no_nan(self):
        ffn = PhysicalKANFeedForward(d_model=32, kan_grid_size=3, dropout=0.0)
        x = torch.randn(4, 8, 32)
        out = ffn(x)
        assert not torch.isnan(out).any()

    def test_interaction_gate_exists(self):
        """Проверяем что гейт есть и начинается с малого значения."""
        ffn = PhysicalKANFeedForward(d_model=32, kan_grid_size=3)
        gate_val = torch.sigmoid(ffn.interaction_gate).item()
        assert gate_val < 0.2  # sigmoid(-2) ≈ 0.12

    def test_backward(self):
        ffn = PhysicalKANFeedForward(d_model=32, kan_grid_size=3, dropout=0.0)
        x = torch.randn(2, 5, 32)
        out = ffn(x)
        out.sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in ffn.parameters() if p.requires_grad)
        assert has_grad


# ============================================================
# PhysicalStem
# ============================================================

class TestPhysicalStem:
    """Тесты Physical Stem с rPhysicsKAN."""

    @pytest.fixture
    def stem(self):
        return PhysicalStem(
            num_input_channels=16,  # 8 пар (A, φ), 4 I + 4 U
            d_model=32,
            num_current_pairs=4,
            kan_grid_size=3,
        )

    def test_instantiation(self, stem):
        assert isinstance(stem, nn.Module)

    def test_forward_shape(self, stem):
        B, C, T = 2, 16, 10
        x = torch.randn(B, C, T)
        mask = torch.zeros(B, C, T, dtype=torch.bool)
        out = stem(x, mask)
        assert out.shape == (B, T, 32)  # (B, T, d_model)

    def test_with_missing_channels(self, stem):
        B, C, T = 2, 16, 10
        x = torch.randn(B, C, T)
        mask = torch.zeros(B, C, T, dtype=torch.bool)
        # Помечаем некоторые каналы как отсутствующие
        mask[0, :4, :] = True
        x[0, :4, :] = 0.0  # После Sanitizer
        out = stem(x, mask)
        assert out.shape == (B, T, 32)
        assert not torch.isnan(out).any()

    def test_no_nan_with_zeros(self, stem):
        """Проверяем безопасное деление при нулевых токах."""
        B, C, T = 2, 16, 10
        x = torch.zeros(B, C, T)  # Все нули
        mask = torch.zeros(B, C, T, dtype=torch.bool)
        out = stem(x, mask)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ============================================================
# SinusoidalPositionalEncoding
# ============================================================

class TestPositionalEncoding:

    def test_instantiation(self):
        pe = SinusoidalPositionalEncoding(d_model=32)
        assert isinstance(pe, nn.Module)

    def test_forward_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=32, max_len=100)
        x = torch.randn(4, 20, 32)
        out = pe(x)
        assert out.shape == (4, 20, 32)

    def test_adds_position_info(self):
        pe = SinusoidalPositionalEncoding(d_model=32, max_len=100, dropout=0.0)
        x = torch.zeros(1, 10, 32)
        out = pe(x)
        # Выход не должен быть нулевым — PE добавляет значения
        assert out.abs().sum() > 0


# ============================================================
# KANFeedForward
# ============================================================

class TestKANFeedForward:

    def test_instantiation(self):
        ffn = KANFeedForward(d_model=32, kan_grid_size=3)
        assert isinstance(ffn, nn.Module)

    def test_forward_shape(self):
        ffn = KANFeedForward(d_model=32, d_ff=64, kan_grid_size=3)
        x = torch.randn(2, 10, 32)
        out = ffn(x)
        assert out.shape == (2, 10, 32)

    def test_no_nan(self):
        ffn = KANFeedForward(d_model=32, kan_grid_size=3, dropout=0.0)
        x = torch.randn(4, 8, 32)
        out = ffn(x)
        assert not torch.isnan(out).any()


# ============================================================
# MLPFeedForward
# ============================================================

class TestMLPFeedForward:

    def test_instantiation(self):
        ffn = MLPFeedForward(d_model=32)
        assert isinstance(ffn, nn.Module)

    def test_forward_shape(self):
        ffn = MLPFeedForward(d_model=32, d_ff=64)
        x = torch.randn(2, 10, 32)
        out = ffn(x)
        assert out.shape == (2, 10, 32)


# ============================================================
# TransformerEncoderBlock
# ============================================================

class TestTransformerEncoderBlock:

    def test_instantiation(self):
        ffn = MLPFeedForward(d_model=32)
        block = TransformerEncoderBlock(d_model=32, num_heads=4, ffn=ffn)
        assert isinstance(block, nn.Module)

    def test_forward_shape(self):
        ffn = MLPFeedForward(d_model=32)
        block = TransformerEncoderBlock(d_model=32, num_heads=4, ffn=ffn)
        x = torch.randn(2, 10, 32)
        out = block(x)
        assert out.shape == (2, 10, 32)

    def test_with_padding_mask(self):
        ffn = MLPFeedForward(d_model=32)
        block = TransformerEncoderBlock(d_model=32, num_heads=4, ffn=ffn)
        x = torch.randn(2, 10, 32)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[0, 8:] = True  # Последние 2 шага замаскированы
        out = block(x, key_padding_mask=mask)
        assert out.shape == (2, 10, 32)

    def test_with_kan_ffn(self):
        ffn = KANFeedForward(d_model=32, kan_grid_size=3)
        block = TransformerEncoderBlock(d_model=32, num_heads=4, ffn=ffn)
        x = torch.randn(2, 10, 32)
        out = block(x)
        assert out.shape == (2, 10, 32)

    def test_with_complex_attention(self):
        """Блок с ComplexMultiheadAttention вместо стандартного MHA."""
        ffn = MLPFeedForward(d_model=32)
        attn = ComplexMultiheadAttention(d_model=32, num_heads=2, dropout=0.0)
        block = TransformerEncoderBlock(d_model=32, num_heads=2, ffn=ffn, attn=attn)
        x = torch.randn(2, 10, 32)
        out = block(x)
        assert out.shape == (2, 10, 32)
        assert not torch.isnan(out).any()


# ============================================================
# ComplexMultiheadAttention
# ============================================================

class TestComplexMultiheadAttention:
    """Тесты комплексного Multi-Head Attention."""

    def test_instantiation(self):
        attn = ComplexMultiheadAttention(d_model=32, num_heads=2)
        assert isinstance(attn, nn.Module)

    def test_forward_shape(self):
        attn = ComplexMultiheadAttention(d_model=32, num_heads=2, dropout=0.0)
        x = torch.randn(4, 10, 32)
        out, _ = attn(x, x, x)
        assert out.shape == (4, 10, 32)

    def test_with_padding_mask(self):
        attn = ComplexMultiheadAttention(d_model=32, num_heads=2, dropout=0.0)
        x = torch.randn(2, 8, 32)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, 6:] = True
        out, _ = attn(x, x, x, key_padding_mask=mask)
        assert out.shape == (2, 8, 32)
        assert not torch.isnan(out).any()

    def test_re_im_separation(self):
        """Проверяем что re и im обрабатываются раздельно (не смешиваются)."""
        attn = ComplexMultiheadAttention(d_model=32, num_heads=2, dropout=0.0)
        x = torch.randn(1, 5, 32)
        out, _ = attn(x, x, x)
        # Выход должен иметь ту же структуру (re | im)
        assert out.shape == (1, 5, 32)
        # Проверяем что re и im части не идентичны (разные проекции)
        assert not torch.allclose(out[:, :, :16], out[:, :, 16:])

    def test_backward(self):
        attn = ComplexMultiheadAttention(d_model=32, num_heads=2, dropout=0.0)
        x = torch.randn(2, 5, 32)
        out, _ = attn(x, x, x)
        out.sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in attn.parameters() if p.requires_grad)
        assert has_grad

    def test_d_head_must_be_even(self):
        """d_model=32, num_heads=8 → d_head=4, d_head_complex=2 — OK."""
        attn = ComplexMultiheadAttention(d_model=32, num_heads=8)
        assert attn.d_head_complex == 2


# ============================================================
# PhysicalKANTransformer (полная модель)
# ============================================================

class TestPhysicalKANTransformer:
    """Smoke-тесты полной модели."""

    @pytest.fixture
    def model(self):
        return PhysicalKANTransformer(
            num_input_channels=16,
            d_model=32,
            num_heads=2,
            num_layers=2,
            kan_grid_size=3,
            dropout=0.0,
            max_seq_len=64,
        )

    @pytest.fixture
    def model_with_cls(self):
        return PhysicalKANTransformer(
            num_input_channels=16,
            d_model=32,
            num_heads=2,
            num_layers=2,
            num_classes=4,
            zone_size=4,
            kan_grid_size=3,
            dropout=0.0,
            max_seq_len=64,
        )

    def test_instantiation(self, model):
        assert isinstance(model, nn.Module)

    def test_num_parameters(self, model):
        n = model.num_parameters()
        assert n > 0

    def test_ssl_forward(self, model):
        """Forward pass в SSL режиме."""
        x = torch.randn(2, 16, 20)
        out = model(x, mode='ssl')
        assert 'ssl' in out
        assert 'features' in out
        assert out['ssl'].shape == (2, 16, 20)
        assert out['features'].shape == (2, 20, 32)

    def test_classify_forward(self, model_with_cls):
        """Forward pass в classify режиме."""
        x = torch.randn(2, 16, 20)
        out = model_with_cls(x, mode='classify')
        assert 'classify' in out
        # T=20, zone_size=4 → num_zones=5
        assert out['classify'].shape == (2, 5, 4)

    def test_with_missing_channels(self, model):
        """Проверяем обработку NaN (отсутствующих каналов)."""
        x = torch.randn(2, 16, 20)
        x[0, :4, :] = float('nan')
        out = model(x, mode='ssl')
        assert not torch.isnan(out['ssl']).any()
        assert not torch.isnan(out['features']).any()

    def test_backward_pass(self, model):
        """Проверяем что градиенты текут."""
        x = torch.randn(2, 16, 10)
        out = model(x, mode='ssl')
        loss = out['ssl'].sum()
        loss.backward()
        # Хотя бы у одного параметра должен быть ненулевой градиент
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad


# ============================================================
# BaselineTransformer
# ============================================================

class TestBaselineTransformer:

    @pytest.fixture
    def model(self):
        return BaselineTransformer(
            num_input_channels=16,
            d_model=32,
            num_heads=2,
            num_layers=2,
            dropout=0.0,
            max_seq_len=64,
        )

    def test_instantiation(self, model):
        assert isinstance(model, nn.Module)

    def test_ssl_forward(self, model):
        x = torch.randn(2, 16, 20)
        out = model(x, mode='ssl')
        assert out['ssl'].shape == (2, 16, 20)

    def test_backward(self, model):
        x = torch.randn(2, 16, 10)
        out = model(x, mode='ssl')
        loss = out['ssl'].sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad


# ============================================================
# ComplexMSELoss
# ============================================================

class TestComplexMSELoss:

    def test_instantiation(self):
        loss = ComplexMSELoss()
        assert isinstance(loss, nn.Module)

    def test_forward_no_mask(self):
        loss_fn = ComplexMSELoss()
        B, C, T = 4, 8, 10
        pred_amp = torch.randn(B, C, T).abs()
        pred_phase = torch.randn(B, C, T)
        true_amp = torch.randn(B, C, T).abs()
        true_phase = torch.randn(B, C, T)
        loss = loss_fn(pred_amp, pred_phase, true_amp, true_phase)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_zero_loss_for_identical(self):
        """Если предсказание = цель, Loss = 0."""
        loss_fn = ComplexMSELoss()
        amp = torch.tensor([[[1.0, 2.0]]])
        phase = torch.tensor([[[0.5, 1.0]]])
        loss = loss_fn(amp, phase, amp, phase)
        assert loss.item() < 1e-6

    def test_with_mask(self):
        loss_fn = ComplexMSELoss()
        B, C, T = 2, 4, 5
        pred_amp = torch.randn(B, C, T).abs()
        pred_phase = torch.randn(B, C, T)
        true_amp = torch.randn(B, C, T).abs()
        true_phase = torch.randn(B, C, T)
        mask = torch.zeros(B, C, T, dtype=torch.bool)
        mask[:, :, -1] = True  # Маскируем последний шаг
        loss = loss_fn(pred_amp, pred_phase, true_amp, true_phase, mask=mask)
        assert not torch.isnan(loss)

    def test_phase_wrap_invariance(self):
        """Ошибка между 1° и 359° должна быть маленькой (≈2°), не 358°."""
        loss_fn = ComplexMSELoss()
        import math
        amp = torch.tensor([[[1.0]]])
        phase_1 = torch.tensor([[[math.radians(1)]]])
        phase_359 = torch.tensor([[[math.radians(359)]]])
        phase_3 = torch.tensor([[[math.radians(3)]]])

        # Расстояние 1°→359° должно быть малым
        loss_wrap = loss_fn(amp, phase_359, amp, phase_1)
        # Расстояние 1°→3° тоже малое
        loss_small = loss_fn(amp, phase_3, amp, phase_1)

        # Оба расстояния должны быть одного порядка и малы
        assert loss_wrap.item() < 0.01
        assert loss_small.item() < 0.01


# ============================================================
# SpectralReconstructionLoss
# ============================================================

class TestSpectralReconstructionLoss:

    def test_instantiation(self):
        loss = SpectralReconstructionLoss()
        assert isinstance(loss, nn.Module)

    def test_forward(self):
        loss_fn = SpectralReconstructionLoss(num_current_channels=4)
        B, C, T = 2, 8, 10
        pred_amp = torch.randn(B, C, T).abs()
        pred_phase = torch.randn(B, C, T)
        true_amp = torch.randn(B, C, T).abs()
        true_phase = torch.randn(B, C, T)
        loss = loss_fn(pred_amp, pred_phase, true_amp, true_phase, current_len=7)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_no_nan_with_zeros(self):
        """Проверяем что при нулевых сигналах нет NaN или Inf."""
        loss_fn = SpectralReconstructionLoss(num_current_channels=4)
        B, C, T = 2, 8, 5
        pred_amp = torch.zeros(B, C, T)
        pred_phase = torch.zeros(B, C, T)
        true_amp = torch.zeros(B, C, T)
        true_phase = torch.zeros(B, C, T)
        loss = loss_fn(pred_amp, pred_phase, true_amp, true_phase)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
