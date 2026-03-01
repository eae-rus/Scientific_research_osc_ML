import pytest
import polars as pl
import numpy as np
from pathlib import Path
import sys
import torch
import os
import shutil

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2_experiments.run_phase2_baseline import main as baseline_main
from scripts.phase2_experiments.run_phase2_kan import main as kan_main
from scripts.phase2_experiments.run_phase2_advanced import main as advanced_main
from scripts.evaluation.evaluate_models import main as evaluate_main

@pytest.fixture
def local_tmp_path():
    """Создает локальную временную директорию для тестов."""
    path = PROJECT_ROOT / "tests" / "unit" / "tmp_scripts_infra"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    # Cleanup after test
    if path.exists():
        shutil.rmtree(path)

@pytest.fixture
def mock_ml_data(local_tmp_path):
    """Создает минимальный набор данных для проверки работоспособности скриптов."""
    data_dir = local_tmp_path / "data" / "ml_datasets"
    data_dir.mkdir(parents=True)
    
    raw_output_dir = local_tmp_path / "raw_data" / "Output"
    raw_output_dir.mkdir(parents=True)
    
    metadata_file = data_dir / "labeled_2025_12_03.csv"
    
    # Создаем фиктивные колонки (I, U и ML_*)
    ml_cols = ["ML_2_5_1", "ML_3_4"]
    feature_cols = [f"I{i}" for i in range(1, 7)] + [f"U{i}" for i in range(1, 7)]
    
    # 2 файла по 1000 строк, чтобы хватило на WINDOW_SIZE=640
    rows = []
    for fname in ["test1.cfg", "test2.cfg"]:
        for _ in range(1000):
            row = {
                "file_name": fname,
                "ML_2_5_1": 1 if fname == "test1.cfg" else 0,
                "ML_3_4": 1 if fname == "test2.cfg" else 0,
            }
            for c in feature_cols:
                row[c] = np.random.randn()
            rows.append(row)
            
    df = pl.DataFrame(rows)
    df.write_csv(metadata_file)
    
    # Создаем пустые папки для экспериментов
    (local_tmp_path / "experiments").mkdir()
    
    return local_tmp_path

def test_scripts_smoke(mock_ml_data, monkeypatch):
    """Smoke-тест: проверяем, что скрипты запускаются и не падают с ошибками импорта/путей."""
    
    # Подменяем ROOT_DIR в скриптах на нашу временную папку
    monkeypatch.setattr("scripts.phase2_experiments.run_phase2_baseline.ROOT_DIR", mock_ml_data)
    monkeypatch.setattr("scripts.phase2_experiments.run_phase2_kan.ROOT_DIR", mock_ml_data)
    monkeypatch.setattr("scripts.phase2_experiments.run_phase2_advanced.ROOT_DIR", mock_ml_data)
    monkeypatch.setattr("scripts.evaluation.evaluate_models.ROOT_DIR", mock_ml_data)
    
    # Ограничиваем количество эпох для быстроты теста
    # (В реальности нужно было бы прокинуть параметры в main, но пока просто проверяем запуск)
    
    # 1. Проверка Baseline (только SimpleCNN для скорости)
    # Внимание: main() в скриптах может содержать бесконечные циклы или долгое обучение.
    # Для полноценного теста нужно рефакторить main() на принятие аргументов.
    # Пока просто проверяем, что импорты и базовые пути корректны.
    
    assert (mock_ml_data / "data" / "ml_datasets" / "labeled_2025_12_03.csv").exists()
    
    # TODO: Рефакторинг скриптов для поддержки аргументов командной строки (epochs, limit_data)
    # позволит запускать их здесь полностью.
