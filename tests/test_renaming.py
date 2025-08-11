import pytest
import csv
from pathlib import Path
import sys

sys.path.insert(0, '.')

from osc_tools.data_management.renaming import rename_one_signals, combining_databases_of_unique_codes

# --- Тесты для rename_one_signals ---

def test_rename_one_signals(tmp_path: Path):
    """
    Проверяет, что rename_one_signals корректно заменяет имя сигнала в CFG файле.
    """
    # 1. Подготовка: Создаем временную директорию и тестовый CFG файл
    source_dir = tmp_path / "test_data"
    source_dir.mkdir()
    cfg_file = source_dir / "test.cfg"

    old_name = "Signal_To_Rename"
    new_name = "Signal_Renamed"

    cfg_content = [
        "station_name,rec_dev_id,rev_year",
        "1,2A,2D",
        f"1,{old_name},A,,V,1.0,0.0,0,-1,1,1.0,1,S",
        "1,Some_Other_Signal,B,,V,1.0,0.0,0,-1,1,1.0,1,S",
    ]
    cfg_file.write_text("\n".join(cfg_content), encoding='utf-8')

    # 2. Действие: Вызываем функцию
    # Подавляем вывод tqdm и print в тестах
    rename_one_signals(str(source_dir), old_name, new_name)

    # 3. Проверка: Читаем файл и проверяем, что замена произошла
    updated_content = cfg_file.read_text(encoding='utf-8')

    assert new_name in updated_content
    assert old_name not in updated_content
    assert "Some_Other_Signal" in updated_content # Убеждаемся, что другие строки не затронуты


# --- Тесты для combining_databases_of_unique_codes ---

def test_combining_databases_merge_and_sum(tmp_path: Path):
    """
    Проверяет, что combining_databases_of_unique_codes корректно объединяет
    два CSV файла, суммируя значения для одинаковых ключей.
    """
    # 1. Подготовка: Создаем два исходных CSV и путь для итогового
    old_csv = tmp_path / "old.csv"
    new_csv = tmp_path / "new.csv"
    merged_csv = tmp_path / "merged.csv"

    old_data = [
        {'Key': 'Signal_A', 'universal_code': 'U_A', 'Value': '10'},
        {'Key': 'Signal_B', 'universal_code': 'U_B', 'Value': '5'},
    ]
    new_data = [
        {'Key': 'Signal_B', 'universal_code': 'U_B_new', 'Value': '7'}, # Общий ключ
        {'Key': 'Signal_C', 'universal_code': 'U_C', 'Value': '12'}, # Новый ключ
    ]

    # Записываем данные в CSV файлы
    with open(old_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Key', 'universal_code', 'Value'])
        writer.writeheader()
        writer.writerows(old_data)

    with open(new_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Key', 'universal_code', 'Value'])
        writer.writeheader()
        writer.writerows(new_data)

    # 2. Действие: Вызываем функцию в режиме слияния
    combining_databases_of_unique_codes(
        str(old_csv), str(new_csv), str(merged_csv), is_merge_files=True
    )

    # 3. Проверка: Читаем итоговый файл и проверяем содержимое
    with open(merged_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        result_data = {row['Key']: row for row in reader}

    assert len(result_data) == 3
    assert result_data['Signal_A']['universal_code'] == 'U_A'
    assert result_data['Signal_A']['Value'] == '10'

    assert result_data['Signal_B']['universal_code'] == 'U_B' # universal_code берется из старого файла
    assert result_data['Signal_B']['Value'] == '12' # 5 + 7

    assert result_data['Signal_C']['universal_code'] == 'U_C'
    assert result_data['Signal_C']['Value'] == '12'
