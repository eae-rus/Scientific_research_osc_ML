"""
Скрипт сбора статистики по каналам из Simulated_OZZ_v1.

Для каждого файла и каждого канала (IA, IB, IC, IN, UA, UB, UC, UN) вычисляет:
  - max |мгновенное значение| по всему файлу
  - max |первая гармоника (h1)| — амплитуда фундаментальной частоты (50 Гц)

Результаты сохраняются:
  1) Подробная таблица: reports/sim_ozz_channel_stats.csv  (по каждому файлу)
  2) Сводка: reports/sim_ozz_channel_summary.txt  (min/max/mean/median по всем файлам)

Используется для определения номинальных коэффициентов ТТ/ТН при нормализации.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

DATA_DIR = Path('data/Simulated_OZZ_v1')
REPORT_DIR = Path('reports')
REPORT_DIR.mkdir(exist_ok=True)

CHANNELS = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']

# Переименование колонок RTDS
_ALIAS = {
    'S1_VA': 'UA', 'VA': 'UA',
    'S1_VB': 'UB', 'VB': 'UB',
    'S1_VC': 'UC', 'VC': 'UC',
    '3I0': 'IN',
    '3U0': 'UN',
}

F_NETWORK = 50.0  # Гц


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _rename_columns(df: pl.DataFrame) -> pl.DataFrame:
    """RTDS long names -> short names."""
    rename_map = {}
    for col in df.columns:
        short = col.split('|')[-1].strip()
        short = short.replace(') ', '_').replace(' ', '_').replace(')', '')
        short = _ALIAS.get(short, short)
        if short != col:
            rename_map[col] = short
    if rename_map:
        df = df.rename(rename_map)
    return df


def _compute_h1_amplitude(signal: np.ndarray, fs: float) -> float:
    """Вычисляет амплитуду первой гармоники (50 Гц) через DFT.
    
    Используем полный файл и DFT-бины ближайшие к 50 Гц.
    """
    n = len(signal)
    if n < 2:
        return 0.0
    
    # FFT
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    
    # Находим бин ближайший к F_NETWORK (50 Гц)
    idx = np.argmin(np.abs(freqs - F_NETWORK))
    
    # Амплитуда (RMS * sqrt(2) для пиковой, или просто |FFT| * 2/N)
    amplitude = 2.0 * np.abs(fft_vals[idx]) / n
    return float(amplitude)


def process_one_file(path: Path) -> dict | None:
    """Обрабатывает один CSV, возвращает dict со статистикой."""
    try:
        df = pl.read_csv(str(path), infer_schema_length=500)
    except Exception:
        return None
    
    df = _rename_columns(df)
    
    # Проверяем наличие каналов
    missing = set(CHANNELS) - set(df.columns)
    if missing:
        return None  # пропускаем некорректные файлы
    
    # dt и Fs
    t_col = df['Time'].to_numpy().astype(np.float64)
    if len(t_col) < 2:
        return None
    dt = float(t_col[1] - t_col[0])
    fs = 1.0 / dt
    
    row = {'filename': path.name, 'fs': round(fs, 1), 'n_samples': len(df)}
    
    for ch in CHANNELS:
        vals = df[ch].to_numpy().astype(np.float64)
        # Макс мгновенное
        max_inst = float(np.max(np.abs(vals)))
        # Амплитуда h1
        max_h1 = _compute_h1_amplitude(vals, fs)
        
        row[f'{ch}_max_inst'] = max_inst
        row[f'{ch}_max_h1'] = max_h1
    
    return row


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------

def main():
    csv_files = sorted(DATA_DIR.glob('OZZ_*.csv'))
    total = len(csv_files)
    print(f"Всего файлов: {total}")
    
    if total == 0:
        print("Файлы не найдены!")
        sys.exit(1)
    
    results = []
    t0 = time.time()
    errors = 0
    
    for i, f in enumerate(csv_files):
        row = process_one_file(f)
        if row is not None:
            results.append(row)
        else:
            errors += 1
        
        if (i + 1) % 500 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] "
                  f"{elapsed:.0f}с, {rate:.1f} файл/с, "
                  f"ETA ~{eta:.0f}с, ошибок={errors}")
    
    if not results:
        print("Нет результатов!")
        sys.exit(1)
    
    # --- Сохраняем подробную таблицу ---
    df_results = pl.DataFrame(results)
    csv_path = REPORT_DIR / 'sim_ozz_channel_stats.csv'
    df_results.write_csv(str(csv_path))
    print(f"\nПодробная таблица: {csv_path} ({len(results)} строк)")
    
    # --- Формируем сводку ---
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("СВОДКА: Статистика каналов Simulated_OZZ_v1")
    summary_lines.append(f"Файлов: {len(results)}, ошибок: {errors}")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Уникальные Fs
    fs_vals = df_results['fs'].to_numpy()
    unique_fs = np.unique(fs_vals)
    summary_lines.append(f"Уникальные Fs: {unique_fs}")
    summary_lines.append("")
    
    # По каждому каналу: глобальный max, mean, median макс. мгновенных / h1
    summary_lines.append(f"{'Канал':>6} | "
                         f"{'max|inst| global':>18} | "
                         f"{'mean max|inst|':>18} | "
                         f"{'median max|inst|':>18} | "
                         f"{'max|h1| global':>18} | "
                         f"{'mean max|h1|':>18} | "
                         f"{'median max|h1|':>18}")
    summary_lines.append("-" * 120)
    
    for ch in CHANNELS:
        inst_col = f'{ch}_max_inst'
        h1_col = f'{ch}_max_h1'
        
        inst_vals = df_results[inst_col].to_numpy()
        h1_vals = df_results[h1_col].to_numpy()
        
        summary_lines.append(
            f"{ch:>6} | "
            f"{np.max(inst_vals):>18.6f} | "
            f"{np.mean(inst_vals):>18.6f} | "
            f"{np.median(inst_vals):>18.6f} | "
            f"{np.max(h1_vals):>18.6f} | "
            f"{np.mean(h1_vals):>18.6f} | "
            f"{np.median(h1_vals):>18.6f}"
        )
    
    summary_lines.append("")
    summary_lines.append("=" * 80)
    summary_lines.append("ИНТЕРПРЕТАЦИЯ (сеть 10 кВ):")
    summary_lines.append("  Напряжения предположительно в кВ:")
    summary_lines.append("    Unom_phase = 10 / sqrt(3) = 5.77 кВ (RMS)")
    summary_lines.append("    Unom_peak = 5.77 * sqrt(2) = 8.16 кВ")
    summary_lines.append("  Для нормализации нужно выбрать делитель,")
    summary_lines.append("  чтобы нормированные значения были ~1.0 в нормальном режиме.")
    summary_lines.append("")
    summary_lines.append("  Для реальных осциллограмм используется:")
    summary_lines.append("    Ip_nominal = 20 * base_val (фазные токи)")
    summary_lines.append("    Iz_nominal = 5 * base_val  (ток нулевой посл.)")
    summary_lines.append("    Ub_nominal = 3 * base_val  (напряжения)")
    summary_lines.append("    где base_val - из norm_coef_all_v1.4.csv")
    summary_lines.append("=" * 80)
    
    summary_text = '\n'.join(summary_lines)
    summary_path = REPORT_DIR / 'sim_ozz_channel_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"Сводка: {summary_path}")
    print()
    print(summary_text)


if __name__ == '__main__':
    main()
