import pytest
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import sys
import torch

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ML_model.train import main as train_main
from ML_model.train_PDR import main as train_pdr_main
from osc_tools.core.constants import Features, PDRFeatures

@pytest.fixture
def local_tmp_path():
    """Создает локальную временную директорию для тестов."""
    path = PROJECT_ROOT / "tests" / "unit" / "tmp_ml_smoke"
    if path.exists():
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    # Cleanup after test
    if path.exists():
        import shutil
        shutil.rmtree(path)

@pytest.fixture
def dummy_train_data(local_tmp_path):
    """Создает фиктивные данные для train.py."""
    csv_path = local_tmp_path / "datset_simpl v2.csv"
    json_path = local_tmp_path / "test_files.json"
    
    # Колонки
    cols = ["file_name"] + Features.ALL + Features.TARGET
    
    # Данные (2 файла, по 100 точек)
    data = []
    for fname in ["file1.cfg", "file2.cfg"]:
        for i in range(100):
            # Гарантируем наличие всех классов для FastBalancedBatchSampler
            # Учитываем expand_right=31 и [:-FRAME_SIZE] (FRAME_SIZE=64)
            # Нам нужны индексы < 36 (100-64)
            if i == 0:
                target = [1, 0, 0] # opr_swch -> expands to 0-31
            elif i == 1:
                target = [0, 1, 0] # abnorm_evnt -> expands to 1-32
            elif i == 2:
                target = [0, 0, 1] # emerg_evnt -> expands to 2-33
            elif i == 35:
                target = [0, 0, 0] # no_event -> index 35 is safe (33 < 35 < 36)
            else:
                target = [0, 0, 0]
            
            row = [fname] + [np.random.rand()] * len(Features.ALL) + target
            data.append(row)
    
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(csv_path, index=False)
    
    # Тестовые файлы
    with open(json_path, "w") as f:
        json.dump({"test_files": ["file2.cfg"]}, f)
        
    return csv_path, json_path

@pytest.fixture
def dummy_pdr_data(local_tmp_path):
    """Создает фиктивные данные для train_PDR.py."""
    train_csv = local_tmp_path / "train_dataset_rPDR_norm_v1.csv"
    test_csv = local_tmp_path / "test_dataset_iPDR_1600_norm_v1.csv"
    
    # Колонки для трейна
    train_cols = ["file_name"] + PDRFeatures.ALL_MODEL_1 + PDRFeatures.TARGET_TRAIN
    # Колонки для теста
    test_cols = ["file_name"] + PDRFeatures.ALL_MODEL_1 + PDRFeatures.TARGET_TEST
    
    # Данные для трейна
    train_data = []
    for fname in ["file1.cfg"]:
        for i in range(100):
            row = [fname] + [np.random.rand()] * len(PDRFeatures.ALL_MODEL_1) + [np.random.randint(0, 2)]
            train_data.append(row)
    
    pd.DataFrame(train_data, columns=train_cols).to_csv(train_csv, index=False)
    
    # Данные для теста
    test_data = []
    for fname in ["file2.cfg"]:
        for i in range(100):
            row = [fname] + [np.random.rand()] * len(PDRFeatures.ALL_MODEL_1) + [np.random.randint(0, 2)]
            test_data.append(row)
            
    pd.DataFrame(test_data, columns=test_cols).to_csv(test_csv, index=False)
    
    return train_csv, test_csv

def test_train_script_smoke(dummy_train_data, monkeypatch, local_tmp_path):
    """Smoke test для train.py."""
    csv_path, json_path = dummy_train_data
    
    # Меняем рабочую директорию на local_tmp_path, чтобы скрипт искал файлы там
    monkeypatch.chdir(local_tmp_path)
    
    # Создаем структуру папок, которую ожидает скрипт
    os.makedirs("ML_model/trained_models", exist_ok=True)
    
    # Копируем файлы в нужное место
    import shutil
    shutil.copy(csv_path, "ML_model/datset_simpl v2.csv")
    shutil.copy(json_path, "ML_model/test_files.json")
    
    # Запускаем main с минимальными параметрами
    # Мы ожидаем, что он отработает без ошибок
    try:
        train_main(epochs=1, num_batches=2, batch_size_per_class=1)
    except Exception as e:
        pytest.fail(f"train_main failed with error: {e}")

def test_train_pdr_script_smoke(dummy_pdr_data, monkeypatch, local_tmp_path):
    """Smoke test для train_PDR.py."""
    train_csv, test_csv = dummy_pdr_data
    
    # Меняем рабочую директорию
    monkeypatch.chdir(local_tmp_path)
    
    # Создаем структуру папок
    os.makedirs("ML_model/trained_models", exist_ok=True)
    
    # Копируем файлы
    import shutil
    shutil.copy(train_csv, "ML_model/train_dataset_rPDR_norm_v1.csv")
    shutil.copy(test_csv, "ML_model/test_dataset_iPDR_1600_norm_v1.csv")
    
    # Запускаем main
    try:
        train_pdr_main(epochs=1, num_train_batches_per_epoch=2, num_files_per_batch=1, samples_per_file=1)
    except Exception as e:
        pytest.fail(f"train_pdr_main failed with error: {e}")
