"""
Тесты для модуля TemporalPooling.

Layer 1: Stable Core — полное покрытие, т.к. это базовый строительный блок
для всех свёрточных моделей проекта.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.layers.temporal_pooling import TemporalPooling, TemporalAttentionPool


# ============================================================================
# Фикстуры
# ============================================================================

@pytest.fixture
def sample_input():
    """Стандартный тензор (Batch=4, Channels=16, Time=80)."""
    return torch.randn(4, 16, 80)

@pytest.fixture
def short_input():
    """Короткий тензор (Time=4) — проверка граничных случаев."""
    return torch.randn(2, 8, 4)

@pytest.fixture
def single_step_input():
    """Вход с одним временным шагом."""
    return torch.randn(2, 8, 1)


# ============================================================================
# Тесты TemporalAttentionPool
# ============================================================================

class TestTemporalAttentionPool:
    """Тесты обучаемого модуля внимания."""

    def test_instantiation(self):
        """Модуль создаётся без ошибок."""
        pool = TemporalAttentionPool(channels=16)
        assert isinstance(pool, nn.Module)

    def test_output_shape(self, sample_input):
        """Выходная форма: (Batch, Channels) — без временной оси."""
        pool = TemporalAttentionPool(channels=16)
        output = pool(sample_input)
        assert output.shape == (4, 16)

    def test_output_shape_short(self, short_input):
        """Работает на коротких последовательностях."""
        pool = TemporalAttentionPool(channels=8)
        output = pool(short_input)
        assert output.shape == (2, 8)

    def test_attention_weights_sum_to_one(self, sample_input):
        """Веса внимания нормализованы (softmax → сумма = 1)."""
        pool = TemporalAttentionPool(channels=16)
        scores = pool.attention_conv(sample_input)
        weights = torch.nn.functional.softmax(scores, dim=-1)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flows(self, sample_input):
        """Градиенты проходят через модуль."""
        pool = TemporalAttentionPool(channels=16)
        sample_input = sample_input.clone().requires_grad_(True)
        output = pool(sample_input)
        loss = output.sum()
        loss.backward()
        assert sample_input.grad is not None


# ============================================================================
# Тесты TemporalPooling — global_avg
# ============================================================================

class TestTemporalPoolingGlobalAvg:
    """Стратегия global_avg — обратная совместимость."""

    def test_output_shape(self, sample_input):
        pool = TemporalPooling(channels=16, strategy="global_avg")
        output = pool(sample_input)
        assert output.shape == (4, 16)

    def test_output_scale(self):
        pool = TemporalPooling(channels=16, strategy="global_avg")
        assert pool.output_scale == 1

    def test_matches_adaptive_avg_pool(self, sample_input):
        """Результат совпадает с nn.AdaptiveAvgPool1d(1) — обратная совместимость."""
        pool = TemporalPooling(channels=16, strategy="global_avg")
        reference = nn.AdaptiveAvgPool1d(1)(sample_input).flatten(1)
        output = pool(sample_input)
        assert torch.allclose(output, reference, atol=1e-6)


# ============================================================================
# Тесты TemporalPooling — attention
# ============================================================================

class TestTemporalPoolingAttention:
    """Стратегия attention — обучаемое внимание."""

    def test_output_shape(self, sample_input):
        pool = TemporalPooling(channels=16, strategy="attention")
        output = pool(sample_input)
        assert output.shape == (4, 16)

    def test_output_scale(self):
        pool = TemporalPooling(channels=16, strategy="attention")
        assert pool.output_scale == 1

    def test_has_learnable_params(self):
        """Стратегия attention имеет обучаемые параметры."""
        pool = TemporalPooling(channels=16, strategy="attention")
        params = list(pool.parameters())
        assert len(params) > 0

    def test_gradient_flows(self, sample_input):
        """Градиенты проходят при обратном проходе."""
        pool = TemporalPooling(channels=16, strategy="attention")
        sample_input = sample_input.clone().requires_grad_(True)
        output = pool(sample_input)
        output.sum().backward()
        assert sample_input.grad is not None

    def test_no_nan(self, sample_input):
        """Выход не содержит NaN."""
        pool = TemporalPooling(channels=16, strategy="attention")
        output = pool(sample_input)
        assert not torch.isnan(output).any()


# ============================================================================
# Общие тесты
# ============================================================================

class TestTemporalPoolingGeneral:
    """Общие тесты для всех стратегий."""

    @pytest.mark.parametrize("strategy", ["global_avg", "attention"])
    def test_all_strategies_produce_2d(self, strategy, sample_input):
        """Все стратегии выдают 2D тензор (Batch, Features) без NaN."""
        pool = TemporalPooling(channels=16, strategy=strategy)
        output = pool(sample_input)
        assert not torch.isnan(output).any()
        assert output.dim() == 2

    def test_invalid_strategy_raises(self):
        """Некорректная стратегия вызывает ошибку."""
        with pytest.raises(ValueError, match="Неизвестная стратегия"):
            TemporalPooling(channels=16, strategy="nonexistent")

    def test_extra_repr(self):
        """Проверка строкового представления."""
        pool = TemporalPooling(channels=16, strategy="attention")
        repr_str = pool.extra_repr()
        assert "attention" in repr_str
        assert "16" in repr_str

    @pytest.mark.parametrize("strategy", ["global_avg", "attention"])
    def test_single_step_all_strategies(self, strategy, single_step_input):
        """Все стратегии работают с одним временным шагом без ошибок."""
        pool = TemporalPooling(channels=8, strategy=strategy)
        output = pool(single_step_input)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("strategy", ["global_avg", "attention"])
    def test_short_input(self, strategy, short_input):
        """Работа с короткими последовательностями."""
        pool = TemporalPooling(channels=8, strategy=strategy)
        output = pool(short_input)
        assert output.shape == (2, 8)


# ============================================================================
# Интеграционные тесты: совместимость с моделями
# ============================================================================

class TestTemporalPoolingModelIntegration:
    """Проверка, что модели работают с обеими стратегиями."""

    def test_conv_kan_global_avg(self):
        """ConvKAN с global_avg (обратная совместимость)."""
        from osc_tools.ml.models.kan import ConvKAN
        model = ConvKAN(in_channels=8, num_classes=4, channels=[8, 16],
                        pooling_strategy="global_avg")
        x = torch.randn(2, 8, 80)
        y = model(x)
        assert y.shape == (2, 4)

    def test_conv_kan_attention(self):
        """ConvKAN с attention."""
        from osc_tools.ml.models.kan import ConvKAN
        model = ConvKAN(in_channels=8, num_classes=4, channels=[8, 16],
                        pooling_strategy="attention")
        x = torch.randn(2, 8, 80)
        y = model(x)
        assert y.shape == (2, 4)

    def test_simple_cnn_attention(self):
        """SimpleCNN с attention."""
        from osc_tools.ml.models.baseline import SimpleCNN
        model = SimpleCNN(in_channels=8, num_classes=4, channels=[16, 32],
                          pooling_strategy="attention")
        x = torch.randn(2, 8, 80)
        y = model(x)
        assert y.shape == (2, 4)

    def test_resnet1d_attention(self):
        """ResNet1D с attention."""
        from osc_tools.ml.models.cnn import ResNet1D
        model = ResNet1D(in_channels=8, num_classes=4, layers=[1, 1, 1, 1],
                         base_filters=16, pooling_strategy="attention")
        x = torch.randn(2, 8, 128)
        y = model(x)
        assert y.shape == (2, 4)

    def test_physics_kan_conditional_attention(self):
        """PhysicsKANConditional с attention."""
        from osc_tools.ml.models.kan import PhysicsKANConditional
        model = PhysicsKANConditional(in_channels=8, num_classes=4,
                                       channels=[8, 16],
                                       pooling_strategy="attention")
        x = torch.randn(2, 8, 128)
        y = model(x)
        assert y.shape == (2, 4)
