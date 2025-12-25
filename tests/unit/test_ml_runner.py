import pytest
import torch
import shutil
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from osc_tools.ml.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from osc_tools.ml.runner import ExperimentRunner

class TestExperimentRunner:
    
    @pytest.fixture
    def temp_dir(self):
        path = Path('tests/temp_experiments')
        path.mkdir(exist_ok=True)
        yield path
        if path.exists():
            shutil.rmtree(path)

    @pytest.fixture
    def simple_config(self, temp_dir):
        return ExperimentConfig(
            model=ModelConfig(
                name='SimpleMLP',
                params={'input_size': 10, 'output_size': 2}
            ),
            data=DataConfig(
                path='dummy',
                window_size=10,
                batch_size=2,
                mode='classification'
            ),
            training=TrainingConfig(
                epochs=1,
                learning_rate=0.01,
                save_dir=str(temp_dir),
                experiment_name='test_run',
                device='cpu'
            )
        )

    def test_init(self, simple_config):
        runner = ExperimentRunner(simple_config)
        assert runner.model is not None
        assert runner.optimizer is not None
        assert runner.criterion is not None

    def test_train_loop(self, simple_config):
        runner = ExperimentRunner(simple_config)
        
        # Создать фиктивные данные
        x = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=2)
        
        # Тренировка по бегу
        runner.train(loader, loader)
        
        # Проверьте, сохранена ли модель
        save_path = Path(simple_config.training.save_dir) / simple_config.training.experiment_name
        assert (save_path / 'best_model.pt').exists()
        assert (save_path / 'final_model.pt').exists()
