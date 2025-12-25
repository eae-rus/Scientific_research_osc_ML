"""
Smoke-тесты для ML моделей.
Layer 3: Experimental Zone (Smoke Tests Only)
"""
import pytest
import torch
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.models import PDR_MLP_v2

class TestMLModelsSmoke:
    """Базовые тесты для проверки работоспособности моделей."""

    def test_pdr_mlp_v2_instantiation(self):
        """Тест инициализации PDR_MLP_v2."""
        model = PDR_MLP_v2(input_features=10, block_neuron_config=[3, 2])
        assert model is not None
        assert len(model.blocks) == 2

    def test_pdr_mlp_v2_forward_pass(self):
        """Тест прямого прохода PDR_MLP_v2."""
        input_features = 10
        batch_size = 4
        model = PDR_MLP_v2(input_features=input_features, block_neuron_config=[3, 2])
        
        # Создаем случайный входной тензор
        x = torch.randn(batch_size, input_features)
        
        # Переводим в режим оценки
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        # Проверяем форму выхода (PDR_MLP обычно возвращает 1 значение на выходе)
        assert output.dim() == 2
        assert output.shape[0] == batch_size
