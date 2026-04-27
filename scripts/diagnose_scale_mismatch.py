"""
Диагностика масштабной разницы между SimOZZ и Real (NormOsc) нормализациями.

Сравнивает амплитуды первой гармоники (h1 FFT) после нормализации
для обоих пайплайнов, чтобы выявить domain shift.

Запуск:
    python scripts/diagnose_scale_mismatch.py
"""
import sys
from pathlib import Path
import numpy as np
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.simulated_ozz_dataset import (
    load_raw_csv, SIM_NOMINAL, RAW_CHANNELS,
)
from osc_tools.ml.augmented_dataset import _extract_fft_harmonics

# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------
SIM_OZZ_DIR = PROJECT_ROOT / 'data' / 'Simulated_OZZ_v1'
NORM_COEF_PATH = PROJECT_ROOT / 'raw_data' / 'norm_coef_all_v1.4.csv'
N_SAMPLE_FILES = 30       # количество SimOZZ файлов для анализа
F_NETWORK = 50.0          # промышленная частота, Гц

CHANNEL_NAMES = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']


def compute_h1_magnitudes(raw_8ch: np.ndarray, dt: float) -> np.ndarray:
    """Вычисляет медианную амплитуду h1 для каждого из 8 каналов.

    Args:
        raw_8ch: (T, 8) нормализованные данные
        dt: шаг дискретизации (сек)

    Returns:
        (8,) медианные амплитуды первой гармоники
    """
    fs = 1.0 / dt
    spp = round(fs / F_NETWORK)  # samples per period

    h1_per_channel = []
    for ch in range(8):
        signal = raw_8ch[:, ch]
        # Берём скользящие окна длиной 1 период
        if len(signal) < spp:
            h1_per_channel.append(0.0)
            continue

        # Выбираем позиции каждые stride отсчётов (4 на период)
        stride = max(1, spp // 4)
        positions = np.arange(0, len(signal) - spp + 1, stride)
        if len(positions) == 0:
            h1_per_channel.append(0.0)
            continue

        windows = np.lib.stride_tricks.sliding_window_view(signal, spp)
        selected_windows = windows[positions]
        harmonics = _extract_fft_harmonics(selected_windows, 1)  # (N, 1) complex
        mags = np.abs(harmonics[:, 0])
        h1_per_channel.append(float(np.median(mags)))

    return np.array(h1_per_channel)


def analyze_sim_ozz():
    """Анализирует несколько SimOZZ файлов, возвращает статистику h1."""
    csv_files = sorted(SIM_OZZ_DIR.glob('OZZ_*.csv'))
    if not csv_files:
        print(f"ОШИБКА: CSV файлы не найдены в {SIM_OZZ_DIR}")
        return None

    sample = random.sample(csv_files, min(N_SAMPLE_FILES, len(csv_files)))

    all_h1 = []  # list of (8,) arrays
    for path in sample:
        result = load_raw_csv(path, normalize=True)
        if result is None:
            continue
        raw = result['raw']       # (T, 8) нормализовано через SIM_NOMINAL
        dt = result['dt']
        h1 = compute_h1_magnitudes(raw, dt)
        all_h1.append(h1)

    if not all_h1:
        print("ОШИБКА: Не удалось загрузить SimOZZ файлы")
        return None

    all_h1 = np.array(all_h1)  # (N_files, 8)
    return all_h1


def analyze_real_theoretical():
    """Теоретический расчёт h1 для реальных данных на основе norm_coef CSV.

    NormOsc делит:
      - Напряжения (UA, UB, UC, UN) на 3 × Ub_base
      - Фазные токи (IA, IB, IC) на 20 × Ip_base
      - Ток нулевой (IN) на 5 × Iz_base

    Для оценки: h1_normalized ≈ h1_raw / divisor
    """
    import polars as pl

    if not NORM_COEF_PATH.exists():
        print(f"ОШИБКА: {NORM_COEF_PATH} не найден")
        return None

    df = pl.read_csv(str(NORM_COEF_PATH), infer_schema_length=None)
    # Фильтруем только разрешённые
    df_ok = df.filter(pl.col('norm').str.contains('YES'))
    print(f"  Всего файлов с norm=YES: {len(df_ok)}")

    # Собираем статистику по Bus 1 (самый популярный)
    bus = 1
    stats = {}

    # Напряжения
    ub_base_col = f'{bus}Ub_base'
    ub_h1_col = f'{bus}Ub_h1'
    if ub_base_col in df_ok.columns and ub_h1_col in df_ok.columns:
        ub_base = df_ok[ub_base_col].cast(pl.Float64, strict=False).drop_nulls().drop_nans()
        ub_h1 = df_ok[ub_h1_col].cast(pl.Float64, strict=False).drop_nulls().drop_nans()

        if len(ub_base) > 0:
            divisors = 3.0 * ub_base.to_numpy()
            h1_vals = ub_h1.to_numpy()
            # h1 в единицах COMTRADE, делитель = 3*base
            # НО h1 это значение до нормализации. После нормализации: h1/divisor
            # Нужно учесть, что h1 — это амплитуда первой гармоники
            # compute_spectral_from_raw даёт: |FFT[k=1]/N * 2| = peak amplitude
            # Значит h1 уже peak amplitude сигнала (если h1 = peak)
            # Или h1 = RMS (если вычислен через другой метод)
            # Принимаем h1 как амплитуду (peak), тогда:
            norm_h1 = h1_vals[:len(divisors)] / divisors[:len(h1_vals)]
            stats['UA_BB'] = norm_h1[norm_h1 > 0]

    # Токи
    ip_base_col = f'{bus}Ip_base'
    ip_h1_col = f'{bus}Ip_h1'
    if ip_base_col in df_ok.columns and ip_h1_col in df_ok.columns:
        ip_base = df_ok[ip_base_col].cast(pl.Float64, strict=False).drop_nulls().drop_nans()
        ip_h1 = df_ok[ip_h1_col].cast(pl.Float64, strict=False).drop_nulls().drop_nans()

        if len(ip_base) > 0:
            divisors = 20.0 * ip_base.to_numpy()
            h1_vals = ip_h1.to_numpy()
            norm_h1 = h1_vals[:len(divisors)] / divisors[:len(h1_vals)]
            stats['IA'] = norm_h1[norm_h1 > 0]

    return stats


def main():
    print("=" * 70)
    print("ДИАГНОСТИКА МАСШТАБНОЙ РАЗНИЦЫ: SimOZZ vs NormOsc (Real)")
    print("=" * 70)

    # --- 1. SimOZZ ---
    print("\n--- SimOZZ (SIM_NOMINAL нормализация) ---")
    print(f"SIM_NOMINAL: {SIM_NOMINAL}")
    print(f"Анализ {N_SAMPLE_FILES} случайных файлов из {SIM_OZZ_DIR.name}...")

    sim_h1 = analyze_sim_ozz()
    if sim_h1 is not None:
        print(f"\n  Результаты (медиана h1 по файлам):")
        print(f"  {'Канал':<8} {'Median':>10} {'P5':>10} {'P95':>10} {'Mean':>10}")
        print(f"  {'-'*48}")
        for i, ch_name in enumerate(CHANNEL_NAMES):
            col = sim_h1[:, i]
            print(f"  {ch_name:<8} {np.median(col):10.4f} {np.percentile(col, 5):10.4f} "
                  f"{np.percentile(col, 95):10.4f} {np.mean(col):10.4f}")

    # --- 2. Real (теоретический расчёт) ---
    print("\n--- Real (NormOsc, теоретический расчёт из norm_coef_all) ---")
    print(f"Формулы:")
    print(f"  U_divisor = 3 × Ub_base")
    print(f"  I_divisor = 20 × Ip_base")
    print(f"  I0_divisor = 5 × Iz_base")

    real_stats = analyze_real_theoretical()
    if real_stats:
        print(f"\n  h1_normalized = h1_raw / divisor (Bus 1):")
        for name, values in real_stats.items():
            if len(values) > 0:
                print(f"  {name:<8} median={np.median(values):.4f}  "
                      f"P5={np.percentile(values, 5):.4f}  "
                      f"P95={np.percentile(values, 95):.4f}  "
                      f"N={len(values)}")

    # --- 3. Сравнение ---
    print("\n" + "=" * 70)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 70)

    if sim_h1 is not None:
        sim_u_median = np.median(sim_h1[:, 4:7])  # UA, UB, UC
        sim_i_median = np.median(sim_h1[:, 0:3])  # IA, IB, IC
    else:
        sim_u_median = sim_i_median = 0

    if real_stats:
        real_u_median = np.median(real_stats.get('UA_BB', [0]))
        real_i_median = np.median(real_stats.get('IA', [0]))
    else:
        real_u_median = real_i_median = 0

    print(f"\n  Напряжения (h1 amplitude):")
    print(f"    SimOZZ median:  {sim_u_median:.4f}")
    print(f"    Real median:    {real_u_median:.4f}")
    if real_u_median > 0:
        ratio_u = sim_u_median / real_u_median
        print(f"    RATIO (sim/real): {ratio_u:.2f}x")
        print(f"    → Модель обучена на напряжениях в {ratio_u:.1f}× больше, чем видит при inference")

    print(f"\n  Токи (h1 amplitude):")
    print(f"    SimOZZ median:  {sim_i_median:.4f}")
    print(f"    Real median:    {real_i_median:.4f}")
    if real_i_median > 0:
        ratio_i = sim_i_median / real_i_median
        print(f"    RATIO (sim/real): {ratio_i:.2f}x")

    # --- 4. Физический анализ ---
    print(f"\n--- ФИЗИЧЕСКИЙ АНАЛИЗ ---")
    print(f"SimOZZ (10 кВ сеть):")
    print(f"  Фазное напряжение (RMS): 10/√3 ≈ {10/np.sqrt(3):.3f} кВ")
    print(f"  Peak: {10/np.sqrt(3)*np.sqrt(2):.3f} кВ")
    print(f"  После /10.0: peak = {10/np.sqrt(3)*np.sqrt(2)/10:.4f}")
    print(f"  FFT h1 magnitude ≈ {10/np.sqrt(3)*np.sqrt(2)/10:.4f}")

    print(f"\nReal (10 кВ, ТН 10000:100, NormOsc):")
    print(f"  Вторичное фазное (RMS): {100/np.sqrt(3):.2f} В")
    print(f"  Peak: {100/np.sqrt(3)*np.sqrt(2):.2f} В")
    print(f"  Divisor = 3 × 100 = 300 В")
    print(f"  После /300: peak = {100/np.sqrt(3)*np.sqrt(2)/300:.4f}")
    print(f"  FFT h1 magnitude ≈ {100/np.sqrt(3)*np.sqrt(2)/300:.4f}")
    print(f"  RATIO: {(10/np.sqrt(3)*np.sqrt(2)/10) / (100/np.sqrt(3)*np.sqrt(2)/300):.2f}x")

    print(f"\n--- РЕКОМЕНДУЕМЫЙ ФИКС ---")
    print(f"Вариант 1: Изменить SIM_NOMINAL (умножить напряжения на 3):")
    new_nom_u = 10.0 * 3
    new_nom_un = SIM_NOMINAL['UN'] * 3
    print(f"  UA/UB/UC: {SIM_NOMINAL['UA']} → {new_nom_u}")
    print(f"  UN:       {SIM_NOMINAL['UN']:.3f} → {new_nom_un:.3f}")
    print(f"  Новый h1 (voltage): {10/np.sqrt(3)*np.sqrt(2)/new_nom_u:.4f}")
    print(f"  vs Real:             {100/np.sqrt(3)*np.sqrt(2)/300:.4f}")
    print(f"  Ratio: {(10/np.sqrt(3)*np.sqrt(2)/new_nom_u)/(100/np.sqrt(3)*np.sqrt(2)/300):.2f}x (≈1.0 = OK)")

    print(f"\nВариант 2: Добавить масштабный множитель в inference_real_ozz.py:")
    print(f"  raw_8ch_model[:, 4:8] *= 3.0  # напряжения × 3")


if __name__ == '__main__':
    main()
