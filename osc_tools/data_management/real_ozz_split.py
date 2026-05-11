"""
Разделение осциллограмм real_OZZ на подтверждённые ОЗЗ и ложные срабатывания.

Использует отчёт overvoltage_report_T1_with_com_v1.7.csv:
- Проверка == '+' → confirmed_ozz (подтверждённое ОЗЗ)
- Проверка == '-' → false_detection (ошибочно детектировано)

Структура выходных CSV:
  data/real_OZZ/confirmed_ozz.csv — filename, bus, group, overvoltage, comment
  data/real_OZZ/false_detection.csv — то же

Пример использования::

    from osc_tools.data_management.real_ozz_split import load_real_ozz_report, split_by_verification

    report = load_real_ozz_report()
    confirmed, false_det = split_by_verification(report)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl


# Путь к отчёту по умолчанию (относительно корня проекта)
_DEFAULT_REPORT_PATH = Path(__file__).resolve().parents[2] / 'data' / 'real_OZZ' / 'overvoltage_report_T1_with_com_v1.7.csv'
_DEFAULT_COMTRADE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'real_OZZ' / 'osc_comtrade'


def load_real_ozz_report(
    report_path: str | Path | None = None,
) -> pl.DataFrame:
    """Загружает отчёт о перенапряжениях.

    Args:
        report_path: путь к CSV (по умолч. data/real_OZZ/overvoltage_report_T1_with_com_v1.7.csv)

    Returns:
        DataFrame с колонками: filename, bus, group, overvoltage_group, overvoltage, verified, comment
    """
    if report_path is None:
        report_path = _DEFAULT_REPORT_PATH
    report_path = Path(report_path)

    if not report_path.exists():
        raise FileNotFoundError(f"Отчёт не найден: {report_path}")

    df = pl.read_csv(
        str(report_path),
        separator=';',
        infer_schema_length=5000,
    )

    # Стандартизация имён колонок
    rename_map = {
        'overvoltage_group': 'overvoltage_group',
        'filename': 'filename',
        'overvoltage': 'overvoltage',
        'bus': 'bus',
        'group': 'group',
        'Проверка': 'verified',
        'Комментарий': 'comment',
    }
    existing = set(df.columns)
    rename = {old: new for old, new in rename_map.items() if old in existing and old != new}
    if rename:
        df = df.rename(rename)

    return df


def split_by_verification(
    report: pl.DataFrame,
    comtrade_dir: str | Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Разделяет отчёт на подтверждённые ОЗЗ и ложные срабатывания.

    Args:
        report: DataFrame из load_real_ozz_report()
        comtrade_dir: папка с .cfg/.dat файлами (для проверки наличия)

    Returns:
        (confirmed_ozz, false_detection) — с теми же колонками
    """
    if comtrade_dir is None:
        comtrade_dir = _DEFAULT_COMTRADE_DIR
    comtrade_dir = Path(comtrade_dir)

    confirmed = report.filter(pl.col('verified') == '+')
    false_det = report.filter(pl.col('verified') == '-')

    # Проверяем наличие файлов на диске
    if comtrade_dir.exists():
        existing_files = {p.stem for p in comtrade_dir.glob('*.cfg')}

        confirmed_in = confirmed.filter(pl.col('filename').is_in(list(existing_files)))
        false_in = false_det.filter(pl.col('filename').is_in(list(existing_files)))

        n_conf_missing = len(confirmed) - len(confirmed_in)
        n_false_missing = len(false_det) - len(false_in)

        if n_conf_missing > 0 or n_false_missing > 0:
            print(f"  COMTRADE файлы не найдены: confirmed={n_conf_missing}, false_detection={n_false_missing}")

        confirmed = confirmed_in
        false_det = false_in

    return confirmed, false_det


def get_bus_for_file(
    report: pl.DataFrame,
    filename: str,
) -> list[str]:
    """Возвращает список секций (bus) для данного файла.

    Один файл COMTRADE может содержать несколько секций (bus=1,2,...).
    В отчёте каждая секция — отдельная строка.

    Args:
        filename: имя файла (без расширения, MD5-хэш)

    Returns:
        Список bus (например ['1', '2'])
    """
    rows = report.filter(pl.col('filename') == filename)
    if rows.is_empty():
        return []
    return sorted(rows['bus'].cast(str).unique().to_list())


def save_split(
    report: pl.DataFrame,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Сохраняет split в CSV файлы.

    Args:
        report: полный отчёт из load_real_ozz_report()
        output_dir: папка для сохранения (по умолч. data/real_OZZ/)

    Returns:
        (confirmed_path, false_detection_path)
    """
    if output_dir is None:
        output_dir = _DEFAULT_REPORT_PATH.parent
    output_dir = Path(output_dir)

    confirmed, false_det = split_by_verification(report)

    conf_path = output_dir / 'confirmed_ozz.csv'
    false_path = output_dir / 'false_detection.csv'

    confirmed.write_csv(str(conf_path))
    false_det.write_csv(str(false_path))

    print(f"Подтверждённые ОЗЗ: {len(confirmed)} записей → {conf_path}")
    print(f"Ложные срабатывания: {len(false_det)} записей → {false_path}")

    # Статистика
    print(f"\n--- Статистика подтверждённых ОЗЗ ---")
    if not confirmed.is_empty():
        print(f"  Уникальных файлов: {confirmed['filename'].n_unique()}")
        print(f"  По группам: {confirmed.group_by('group').len().sort('group')}")
        print(f"  По секциям (bus): {confirmed.group_by('bus').len().sort('bus')}")

    print(f"\n--- Статистика ложных срабатываний ---")
    if not false_det.is_empty():
        print(f"  Уникальных файлов: {false_det['filename'].n_unique()}")

    return conf_path, false_path


if __name__ == '__main__':
    report = load_real_ozz_report()
    save_split(report)
