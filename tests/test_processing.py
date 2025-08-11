import pytest
import json
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, '.')

from osc_tools.data_management.processing import (
    extract_frequencies,
    combining_json_hash_table,
    _deleting_confidential_information_in_one_file
)

# --- Тесты для extract_frequencies ---

def test_extract_frequencies_success(tmp_path: Path):
    """Проверяет корректное извлечение частот из валидного CFG файла."""
    cfg_file = tmp_path / "test.cfg"
    content = [
        "line 1",
        "2,1A,1D", # 2 сигнала всего
        "analog 1",
        "analog 2",
        "50", # f_network
        "1,1", # sample rates
        "1600,1000", # f_rate
    ]
    cfg_file.write_text("\n".join(content), encoding='utf-8')

    f_network, f_rate = extract_frequencies(str(cfg_file))

    assert f_network == 50
    assert f_rate == 1600

def test_extract_frequencies_malformed_file(tmp_path: Path):
    """Проверяет, что функция возвращает (-1, -1) для некорректного файла."""
    cfg_file = tmp_path / "test.cfg"
    cfg_file.write_text("not enough lines", encoding='utf-8')

    f_network, f_rate = extract_frequencies(str(cfg_file))

    assert f_network == -1
    assert f_rate == -1

# --- Тесты для combining_json_hash_table ---

def test_combining_json_hash_table(tmp_path: Path):
    """Проверяет корректное слияние нескольких JSON файлов в один."""
    # 1. Подготовка
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    json1_path = dir1 / "table1.json"
    json1_data = {"hash1": "file1", "hash2": "file2"}
    json1_path.write_text(json.dumps(json1_data))

    dir2 = tmp_path / "dir2"
    dir2.mkdir()
    json2_path = dir2 / "table2.json"
    json2_data = {"hash2": "file2_overwrite_attempt", "hash3": "file3"}
    json2_path.write_text(json.dumps(json2_data))

    # 2. Выполнение
    combining_json_hash_table(str(tmp_path))

    # 3. Проверка
    combined_file = tmp_path / "combine_hash_table.json"
    assert combined_file.exists()

    with open(combined_file, 'r') as f:
        combined_data = json.load(f)

    expected_data = {
        "hash1": "file1",
        "hash2": "file2", # Значение из первого файла должно остаться
        "hash3": "file3"
    }
    assert combined_data == expected_data

# --- Тесты для _deleting_confidential_information_in_one_file ---

def test_deleting_confidential_information(tmp_path: Path):
    """
    Проверяет удаление конфиденциальной информации и переименование
    файлов в их MD5 хеш.
    """
    # 1. Подготовка
    root_dir = tmp_path
    cfg_path = root_dir / "original.cfg"
    dat_path = root_dir / "original.dat"

    cfg_content = [
        "Station,Device,2024", # Конфиденциальная информация
        "1,1A,0D",
        "Analog1,A,,V,1,0,0,-1,1,1,1,S",
        "50",
        "1,1",
        "10/10/2024,10:10:10.000", # Старая дата
        "11/11/2024,11:11:11.000", # Старая дата
    ]
    dat_content = b"some binary data"

    cfg_path.write_text("\n".join(cfg_content), encoding='utf-8')
    dat_path.write_bytes(dat_content)

    # Вычисляем ожидаемый хеш
    expected_hash = hashlib.md5(dat_content).hexdigest()

    # 2. Выполнение
    _deleting_confidential_information_in_one_file(str(cfg_path), str(root_dir), 'utf-8')

    # 3. Проверка

    # Проверяем, что старые файлы удалены
    assert not cfg_path.exists()
    assert not dat_path.exists()

    # Проверяем, что новые (переименованные) файлы существуют
    new_cfg_path = root_dir / (expected_hash + ".cfg")
    new_dat_path = root_dir / (expected_hash + ".dat")
    assert new_cfg_path.exists()
    assert new_dat_path.exists()

    # Проверяем содержимое нового CFG файла
    new_cfg_content = new_cfg_path.read_text(encoding='utf-8').splitlines()
    assert new_cfg_content[0] == ",,2024" # Первая строка очищена
    # В нашем тестовом файле signals=1. Даты находятся на строках signals+5 и signals+6 (индексы 6 и 7)
    # Наш файл имеет 7 строк (индексы 0-6). Значит, только строка с индексом 6 будет изменена.
    assert new_cfg_content[5] == "10/10/2024,10:10:10.000" # Эта строка не должна измениться
    assert new_cfg_content[6] == "01/01/0001, 01:01:01.000000" # А эта должна
