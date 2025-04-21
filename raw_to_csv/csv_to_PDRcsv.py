import pandas as pd
import numpy as np
import json # Для возможной конфигурации из файла
import os
import sys
import warnings
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm # Для прогресс-бара

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)

# Подавление предупреждений о делении на ноль и невалидных значениях,
# т.к. мы будем обрабатывать их явно позже
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Вспомогательные функции ---

def sliding_window_fft(signal: np.ndarray, window_size: int, num_harmonics: int, verbose: bool = False) -> np.ndarray:
    """
    Выполняет БПФ в скользящем окне.
    Возвращает комплексные значения для указанного числа гармоник для каждого окна.
    Результат относится к центру окна, поэтому первые window_size // 2 точек будут NaN.
    """
    n_points = len(signal)
    if n_points < window_size:
        if verbose:
            print(f"    Предупреждение: Длина сигнала ({n_points}) меньше окна FFT ({window_size}). Пропуск FFT.")
        # Возвращаем массив NaN нужной формы
        return np.full((n_points, num_harmonics), np.nan + 1j*np.nan, dtype=complex)

    fft_results = np.full((n_points, num_harmonics), np.nan + 1j*np.nan, dtype=complex)
    hanning_window = np.hanning(window_size)
    fft_start_offset = window_size # Смещение для записи результата в центр окна

    for i in range(n_points - window_size + 1):
        window_data = signal[i : i + window_size] * hanning_window
        fft_coeffs = np.fft.fft(window_data) / window_size

        # Берем нужные гармоники (индексы 1..num_harmonics) и умножаем на 2
        harmonics = fft_coeffs[1 : num_harmonics + 1] * 2
        
        center_index = i + fft_start_offset
        if center_index < n_points:
             num_calculated = len(harmonics)
             fft_results[center_index, :num_calculated] = harmonics[:num_harmonics]

    return fft_results # Форма (n_points, num_harmonics)

def calculate_symmetrical_components(phasor_a: np.ndarray, phasor_b: np.ndarray, phasor_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Расчет симметричных составляющих (прямая, обратная, нулевая)."""
    a = np.exp(1j * 2 * np.pi / 3)
    a2 = a * a
    
    phasor_0 = (phasor_a + phasor_b + phasor_c) / 3.0
    phasor_1 = (phasor_a + a * phasor_b + a2 * phasor_c) / 3.0
    phasor_2 = (phasor_a + a2 * phasor_b + a * phasor_c) / 3.0
    
    return phasor_1, phasor_2, phasor_0

def calculate_impedance(voltage: np.ndarray, current: np.ndarray, min_current_threshold: float = 1e-6) -> np.ndarray:
    """Расчет комплексного сопротивления Z = V / I."""
    current_safe = current.copy()
    zero_current_mask = np.abs(current_safe) < min_current_threshold
    current_safe[zero_current_mask] = np.nan # Замена на NaN для избежания деления на 0
    impedance = voltage / current_safe
    # Защита от слишком малых токов
    impedance[zero_current_mask] = 1/min_current_threshold + (1/min_current_threshold)*1j # Задаём максимальный порог, чтобы исключить nan
    return impedance

def calculate_power(voltage: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Расчет комплексной (S), активной (P) и реактивной (Q) мощности."""
    s_complex = voltage * np.conj(current)
    p_active = s_complex.real
    q_reactive = s_complex.imag
    # s_apparent = np.abs(s_complex) # Модуль можно получить позже из s_complex
    return s_complex, p_active, q_reactive

def calculate_linear_voltages(ua: np.ndarray, ub: np.ndarray, uc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Расчет линейных напряжений."""
    uab = ua - ub
    ubc = ub - uc
    uca = uc - ua
    return uab, ubc, uca

# TODO: Реализация расчета линейных сопротивлений требует уточнения методики.
# def calculate_linear_impedances(...): ...

def format_complex_to_mag_angle(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Форматирует комплексные столбцы в модуль и угол.
    Удаляет исходные комплексные столбцы.
    """
    # TODO: Реализация поддержки форматов 're_im' и 'complex' требует пересмотра
    # стратегии нормализации, поэтому пока используется только 'mag_angle'.
    new_df = df.copy()
    if not columns:
        return new_df
        
    for col in columns:
        if col in new_df.columns:
            complex_data = new_df[col].astype(complex) # Убедимся что тип комплексный
            mask_valid = pd.notna(complex_data)

            magnitudes = np.full(complex_data.shape, np.nan)
            angles = np.full(complex_data.shape, np.nan)

            magnitudes[mask_valid] = np.abs(complex_data[mask_valid])
            angles[mask_valid] = np.angle(complex_data[mask_valid]) # Угол в радианах [-pi, pi]

            new_df[f'{col}_mag'] = magnitudes
            new_df[f'{col}_angle'] = angles
            new_df = new_df.drop(columns=[col]) # Удаляем исходный комплексный столбец
        # else: # Не должно происходить, если columns содержит только существующие столбцы
        #     print(f"Предупреждение: Столбец {col} не найден для форматирования.")

    return new_df

def cols_exist_and_not_all_nan(df: pd.DataFrame, cols: List[str]) -> bool:
    """
    Возвращает True, если все колонки из cols есть в df и ни одна из них
    не заполнена целиком NaN.
    """
    for col in cols:
        if col not in df.columns:
            return False
        # если все значения в столбце — NaN, то отбрасываем
        if df[col].isna().all():
            return False
    return True


# --- Основная функция обработки ---

def process_oscillograms(
    csv_path: str,
    config: Dict,
    output_csv_path: str = "PDR_data.csv",
    mapping_csv_path: str = "filename_mapping.csv",
) -> None:
    """
    Обрабатывает CSV файл с осциллограммами согласно конфигурации.

    Args:
        csv_path (str): Путь к исходному CSV файлу.
        config (Dict): Словарь конфигурации. Содержит ключи:
            target_signal (str): 'iPDR PS' или 'rPDR PS'.
            use_voltage_source (str): 'BB' или 'CL' - источник напряжения по умолчанию.
            samples_per_period (int): Количество точек на период для FFT (размер окна).
            num_harmonics (int): Количество гармоник для расчета (начиная с 1-й).
            # complex_format (str): 'mag_angle' (пока только этот формат, так как имеются проблемы с нормализацией)
            output_signals (List[str]): Список сигналов для сохранения. 'ALL' добавляет все рассчитанные.
                                       Поддерживаются базовые сигналы (UA BB, IA и т.д.) и производные.
            rename_files (bool): Переименовывать ли file_name в номера.
            start_file_id (int): Начальный номер для переименования файлов (если rename_files=True).
            min_current_threshold (float): Порог тока для расчета сопротивления.
            verbose (bool): Выводить ли подробные сообщения во время обработки.
    """
    print("--- Начало обработки осциллограмм ---")
    
    # Проверка конфигурации
    required_keys = ['target_signal', 'use_voltage_source', 'samples_per_period', 
                     'num_harmonics', 'output_signals', 
                     'rename_files', 'min_current_threshold', 'verbose']
    default_config = {'verbose': False} # Значение по умолчанию для verbose
    
    # Установка значений по умолчанию
    for key, value in default_config.items():
        config.setdefault(key, value)
        
    if not all(key in config for key in required_keys):
        missing_keys = [k for k in required_keys if k not in config]
        raise ValueError(f"Конфигурация неполная. Отсутствуют ключи: {missing_keys}")
        
    verbose = config['verbose']
    window_size = config['samples_per_period']
    fft_start_offset = window_size # Сколько точек обрезать в начале

    if not os.path.exists(csv_path):
         raise FileNotFoundError(f"Исходный CSV файл не найден: {csv_path}")

    try:
        df_main = pd.read_csv(csv_path)
        if verbose: print(f"Загружен файл: {csv_path}, строк: {len(df_main)}")
    except Exception as e:
        print(f"Критическая ошибка загрузки CSV файла {csv_path}: {e}")
        return

    processed_groups = []
    filename_mapping = {}
    file_counter = 0
    
    # Группировка по имени файла
    grouped = df_main.groupby('file_name')
    total_groups = len(grouped)
    if verbose: print(f"Найдено {total_groups} уникальных групп (file_name).")

    # Инициализация прогресс-бара
    pbar = tqdm(grouped, total=total_groups, desc="Обработка групп")

    # Определяем, какие типы производных сигналов нужно считать
    output_signals_set = set(config['output_signals'])
    calc_linear_V = any(s in output_signals_set for s in ['UAB_h1', 'UBC_h1', 'UCA_h1'])
    calc_sym_comp_V = any(s in output_signals_set for s in ['V_pos_seq', 'V_neg_seq', 'V_zero_seq'])
    calc_sym_comp_I = any(s in output_signals_set for s in ['I_pos_seq', 'I_neg_seq', 'I_zero_seq'])
    calc_Z_phase = any(s in output_signals_set for s in ['Z_A', 'Z_B', 'Z_C'])
    calc_Z_linear = any(s in output_signals_set for s in ['Z_AB', 'Z_BC', 'Z_CA'])
    calc_Z_seq = any(s in output_signals_set for s in ['Z_pos_seq', 'Z_neg_seq'])
    calc_Power_phase = any(s.startswith(p) for p in ['P_A','Q_A','S_A','P_B','Q_B','S_B','P_C','Q_C','S_C','P_total','Q_total','S_total'] for s in output_signals_set)
    calc_Power_seq = any(s.endswith('_seq') and s[0] in 'PQS' for s in output_signals_set)

    for name, group in pbar:
        pbar.set_description(f"Обработка группы: {name[:20]}...") # Обновляем описание в прогресс-баре
        if verbose: print(f"\n--- Обработка группы: {name} ---")
        try:
            group_data = group.copy() # Работаем с копией данных группы
            
            # --- Шаг 1: Валидация и выбор сигналов ---
            if verbose: print("  Шаг 1: Валидация и выбор сигналов...")
            
            target_col = config['target_signal']
            if target_col not in group_data.columns or group_data[target_col].isnull().any():
                print(f"\nПредупреждение (Группа {name}): Целевой сигнал '{target_col}' отсутствует или содержит NaN. Пропуск группы.")
                continue
            
            # Определяем используемые колонки для U и I
            u_source = config['use_voltage_source']
            ua_col, ub_col, uc_col = f'UA {u_source}', f'UB {u_source}', f'UC {u_source}'
            
            if not cols_exist_and_not_all_nan(group_data, [ua_col, ub_col, uc_col]):
                alt_source = 'CL' if u_source == 'BB' else 'BB'
                ua_col_alt, ub_col_alt, uc_col_alt = f'UA {alt_source}', f'UB {alt_source}', f'UC {alt_source}'
                if cols_exist_and_not_all_nan(group_data, [ua_col_alt, ub_col_alt, uc_col_alt]):
                    ua_col, ub_col, uc_col = ua_col_alt, ub_col_alt, uc_col_alt
                    if verbose: print(f"    Предупреждение: Напряжения '{u_source}' не найдены, используются '{alt_source}'.")
                else:
                    print(f"\nПредупреждение (Группа {name}): Не найден полный набор фазных напряжений (ни {u_source}, ни {alt_source}). Пропуск группы.")
                    continue
            
            ia_col, ib_col, ic_col = 'IA', 'IB', 'IC'
            ia_present = ia_col in group_data.columns and not group_data[ia_col].isnull().any()
            ic_present = ic_col in group_data.columns and not group_data[ic_col].isnull().any()
            
            if not (ia_present and ic_present):
                print(f"\nПредупреждение (Группа {name}): Отсутствуют токи фаз A ('{ia_col}') или C ('{ic_col}'). Пропуск группы.")
                continue

            ib_present = ib_col in group_data.columns and not group_data[ib_col].isnull().any()
            
            # Собираем сигналы U и I в словарь для удобства
            signals = {}
            signals['UA'] = group_data[ua_col].values
            signals['UB'] = group_data[ub_col].values
            signals['UC'] = group_data[uc_col].values
            signals['IA'] = group_data[ia_col].values
            signals['IC'] = group_data[ic_col].values
            
            if not ib_present:
                if verbose: print(f"    Предупреждение: Ток фазы B ('{ib_col}') отсутствует или содержит NaN. Расчет как IB = -IA - IC.")
                signals['IB'] = -signals['IA'] - signals['IC']
                ib_col = 'IB_calculated' # Используем новое имя для потенциального вывода
            else:
                signals['IB'] = group_data[ib_col].values
            
            if verbose: print(f"    Используются напряжения: {ua_col}, {ub_col}, {uc_col}")
            if verbose: print(f"    Используются токи: {ia_col}, {ib_col}, {ic_col}")

            # --- Шаг 2: Преобразование Фурье и коррекция фаз ---
            if verbose: print(f"  Шаг 2: Расчет FFT (окно: {window_size}, гармоники: {config['num_harmonics']})...")
            num_harmonics = config['num_harmonics']
            
            fft_results = {} # Нескорректированные результаты
            for sig_name, sig_data in signals.items():
                fft_complex = sliding_window_fft(sig_data, window_size, num_harmonics, verbose)
                fft_results[sig_name] = fft_complex # Shape: (n_points, num_harmonics)

            # Коррекция фаз относительно первой гармоники UA
            ua_h1_complex = fft_results['UA'][:, 0] # Первая гармоника UA (индекс 0)
            ua_h1_angles = np.full(ua_h1_complex.shape, np.nan)
            valid_ua_mask = pd.notna(ua_h1_complex)
            ua_h1_angles[valid_ua_mask] = np.angle(ua_h1_complex[valid_ua_mask])

            if verbose: print("    Коррекция фаз относительно UA H1...")
            corrected_fft_results = {} # Словарь для хранения скорректированных комплексных результатов FFT
            all_complex_signal_names = [] # Собираем имена всех комплексных сигналов

            for sig_name, complex_data in fft_results.items():
                corrected_data = np.full_like(complex_data, np.nan + 1j*np.nan)
                for h in range(num_harmonics):
                    harmonic_phasors = complex_data[:, h]
                    valid_mask = pd.notna(harmonic_phasors) & pd.notna(ua_h1_angles)
                    
                    magnitudes = np.abs(harmonic_phasors[valid_mask])
                    original_angles = np.angle(harmonic_phasors[valid_mask])
                    corrected_angles = np.angle(np.exp(1j * (original_angles - ua_h1_angles[valid_mask]))) 
                    
                    corrected_data[valid_mask, h] = magnitudes * np.exp(1j * corrected_angles)
                    if sig_name == 'UA' and h == 0: # Угол UA H1 должен стать 0
                        corrected_data[valid_mask, h] = magnitudes * np.exp(1j * 0)

                corrected_fft_results[sig_name] = corrected_data
                # Добавляем имена гармоник в список комплексных сигналов
                for h in range(num_harmonics):
                     all_complex_signal_names.append(f"{sig_name}_h{h+1}")


            # Создаем DataFrame для обработанных данных этой группы
            # Изначально содержит только результаты FFT и позже другие расчеты
            processed_data = pd.DataFrame(index=group_data.index) # Сохраняем исходные индексы

            # Добавляем результаты FFT (пока комплексные)
            for sig_name, complex_data in corrected_fft_results.items():
                for h in range(num_harmonics):
                    col_name = f"{sig_name}_h{h+1}"
                    processed_data[col_name] = complex_data[:, h]
                    
            # --- Шаг 3: Расчет производных сигналов (только если они нужны) ---
            if verbose: print("  Шаг 3: Расчет производных сигналов (если запрошены)...")
            calculated_signals = {} # Словарь для новых сигналов (комплексных и действительных)
            
            # Берем фазоры 1-й гармоники (они нужны для большинства расчетов)
            h1_index = 0 # Индекс первой гармоники
            ua_h1 = corrected_fft_results['UA'][:, h1_index]
            ub_h1 = corrected_fft_results['UB'][:, h1_index]
            uc_h1 = corrected_fft_results['UC'][:, h1_index]
            ia_h1 = corrected_fft_results['IA'][:, h1_index]
            ib_h1 = corrected_fft_results['IB'][:, h1_index]
            ic_h1 = corrected_fft_results['IC'][:, h1_index]
            
            min_current = config['min_current_threshold']

            # 3.1 Симметричные составляющие
            if calc_sym_comp_V or calc_sym_comp_I or calc_Z_seq or calc_Power_seq:
                 V1, V2, V0 = calculate_symmetrical_components(ua_h1, ub_h1, uc_h1)
                 I1, I2, I0 = calculate_symmetrical_components(ia_h1, ib_h1, ic_h1)
                 if calc_sym_comp_V:
                      calculated_signals.update({'V_pos_seq': V1, 'V_neg_seq': V2, 'V_zero_seq': V0})
                      all_complex_signal_names.extend(['V_pos_seq', 'V_neg_seq', 'V_zero_seq'])
                 if calc_sym_comp_I:
                      calculated_signals.update({'I_pos_seq': I1, 'I_neg_seq': I2, 'I_zero_seq': I0})
                      all_complex_signal_names.extend(['I_pos_seq', 'I_neg_seq', 'I_zero_seq'])

            # 3.2 Сопротивления
            if calc_Z_phase:
                calculated_signals['Z_A'] = calculate_impedance(ua_h1, ia_h1, min_current)
                calculated_signals['Z_B'] = calculate_impedance(ub_h1, ib_h1, min_current)
                calculated_signals['Z_C'] = calculate_impedance(uc_h1, ic_h1, min_current)
                all_complex_signal_names.extend(['Z_A', 'Z_B', 'Z_C'])
            if calc_Z_seq:
                 # Убедимся, что V1, I1, V2, I2 рассчитаны
                 if 'V_pos_seq' not in calculated_signals: V1, V2, V0 = calculate_symmetrical_components(ua_h1, ub_h1, uc_h1)
                 if 'I_pos_seq' not in calculated_signals: I1, I2, I0 = calculate_symmetrical_components(ia_h1, ib_h1, ic_h1)
                 calculated_signals['Z_pos_seq'] = calculate_impedance(V1, I1, min_current)
                 calculated_signals['Z_neg_seq'] = calculate_impedance(V2, I2, min_current) 
                 all_complex_signal_names.extend(['Z_pos_seq', 'Z_neg_seq'])
            if calc_Z_linear:
                # Расчет линейных напряжений (разность фазных фазоров)
                vab_h1 = ua_h1 - ub_h1
                vbc_h1 = ub_h1 - uc_h1
                vca_h1 = uc_h1 - ua_h1

                # Расчет линейных (дифференциальных) токов (разность фазных фазоров)
                iab_h1 = ia_h1 - ib_h1
                ibc_h1 = ib_h1 - ic_h1
                ica_h1 = ic_h1 - ia_h1

                # Расчет линейных сопротивлений с использованием существующей функции
                calculated_signals['Z_AB'] = calculate_impedance(vab_h1, iab_h1, min_current)
                calculated_signals['Z_BC'] = calculate_impedance(vbc_h1, ibc_h1, min_current)
                calculated_signals['Z_CA'] = calculate_impedance(vca_h1, ica_h1, min_current)
                
                # Добавляем имена новых комплексных сигналов в общий список
                all_complex_signal_names.extend(['Z_AB', 'Z_BC', 'Z_CA'])

            # 3.3 Мощности
            if calc_Power_phase:
                S_A, P_A, Q_A = calculate_power(ua_h1, ia_h1)
                S_B, P_B, Q_B = calculate_power(ub_h1, ib_h1)
                S_C, P_C, Q_C = calculate_power(uc_h1, ic_h1)
                calculated_signals.update({'S_A': S_A, 'P_A': P_A, 'Q_A': Q_A,
                                           'S_B': S_B, 'P_B': P_B, 'Q_B': Q_B,
                                           'S_C': S_C, 'P_C': P_C, 'Q_C': Q_C})
                all_complex_signal_names.extend(['S_A', 'S_B', 'S_C'])
                # Суммарная мощность (рассчитываем если нужна любая фазная или суммарная)
                P_total = P_A + P_B + P_C
                Q_total = Q_A + Q_B + Q_C
                S_total_complex = S_A + S_B + S_C
                calculated_signals.update({'P_total': P_total, 'Q_total': Q_total, 'S_total': S_total_complex})
                all_complex_signal_names.append('S_total')
                
            if calc_Power_seq:
                 # Убедимся, что V1, I1 и т.д. рассчитаны
                 if 'V_pos_seq' not in calculated_signals: V1, V2, V0 = calculate_symmetrical_components(ua_h1, ub_h1, uc_h1)
                 if 'I_pos_seq' not in calculated_signals: I1, I2, I0 = calculate_symmetrical_components(ia_h1, ib_h1, ic_h1)
                 S1, P1, Q1 = calculate_power(3 * V1, I1)
                 S2, P2, Q2 = calculate_power(3 * V2, I2)
                 S0, P0, Q0 = calculate_power(3 * V0, I0)
                 calculated_signals.update({'S_pos_seq': S1, 'P_pos_seq': P1, 'Q_pos_seq': Q1,
                                            'S_neg_seq': S2, 'P_neg_seq': P2, 'Q_neg_seq': Q2,
                                            'S_zero_seq': S0,'P_zero_seq': P0,'Q_zero_seq': Q0})
                 all_complex_signal_names.extend(['S_pos_seq', 'S_neg_seq', 'S_zero_seq'])
                 
            # 3.4 Линейные напряжения (1-й гармоники)
            if calc_linear_V:
                uab_h1, ubc_h1, uca_h1 = calculate_linear_voltages(ua_h1, ub_h1, uc_h1)
                calculated_signals.update({'UAB_h1': uab_h1, 'UBC_h1': ubc_h1, 'UCA_h1': uca_h1})
                all_complex_signal_names.extend(['UAB_h1', 'UBC_h1', 'UCA_h1'])

            # Добавляем рассчитанные сигналы в processed_data
            for sig_name, sig_data in calculated_signals.items():
                 processed_data[sig_name] = sig_data

            # --- Добавление необходимых исходных и служебных столбцов ---
            # Добавляем file_name для последующей фильтрации и переименования
            processed_data['file_name'] = name 
            # Добавляем целевой сигнал и change_event
            processed_data[target_col] = group_data[target_col].values
            if 'change_event' in group_data.columns:
                 processed_data['change_event'] = group_data['change_event'].values

            # Добавляем исходные U, I если они запрошены в output_signals
            original_signals_to_consider = {
                ua_col: signals['UA'], ub_col: signals['UB'], uc_col: signals['UC'],
                ia_col: signals['IA'], ib_col: signals['IB'], ic_col: signals['IC']
            }
            for col_name, sig_data in original_signals_to_consider.items():
                 if config['output_signals'] == 'ALL' or col_name in output_signals_set:
                      processed_data[col_name] = sig_data

            # --- Фильтрация выходных сигналов ---
            available_cols = list(processed_data.columns)
            if config['output_signals'] != 'ALL':
                # Оставляем только запрошенные + обязательные (file_name, target, change_event)
                cols_to_keep_requested = [col for col in config['output_signals'] if col in available_cols]
                # Формируем итоговый список столбцов
                final_cols_to_keep = ['file_name', target_col]
                if 'change_event' in processed_data.columns:
                    final_cols_to_keep.append('change_event')
                # Добавляем запрошенные столбцы, избегая дубликатов
                final_cols_to_keep.extend([c for c in cols_to_keep_requested if c not in final_cols_to_keep])
            else:
                final_cols_to_keep = available_cols # Оставляем все сгенерированные

            processed_data = processed_data[final_cols_to_keep]
            if verbose: print(f"    Оставляем столбцы: {list(processed_data.columns)}")

            # --- Форматирование комплексных чисел в Mag/Angle ---
            # Определяем, какие из ОСТАВШИХСЯ колонок были комплексными
            complex_cols_in_output = [col for col in processed_data.columns if col in all_complex_signal_names]
            if verbose: print(f"    Форматирование комплексных чисел в 'mag_angle'...")
            processed_data = format_complex_to_mag_angle(processed_data, complex_cols_in_output)

            # --- Шаг 4 (внутри группы): Обрезка начальных точек из-за FFT ---
            if verbose: print(f"    Обрезка первых {fft_start_offset} точек из-за окна FFT...")
            if len(processed_data) > fft_start_offset:
                 processed_data = processed_data.iloc[fft_start_offset:].reset_index(drop=True)
            else:
                 if verbose: print(f"    Предупреждение: Длина данных ({len(processed_data)}) после FFT меньше смещения ({fft_start_offset}). Группа будет пустой.")
                 processed_data = processed_data.iloc[0:0] # Делаем DataFrame пустым

            # --- Добавление в общий список ---
            if not processed_data.empty:
                 processed_groups.append(processed_data)
                 file_counter += 1 # Счетчик успешно обработанных и непустых групп
                 if verbose: print(f"    Группа {name} успешно обработана и добавлена.")
            else:
                 if verbose: print(f"    Группа {name} пропущена (стала пустой после обрезки FFT).")


        except Exception as e:
            print(f"\nКритическая ошибка при обработке группы {name}: {e}")
            # import traceback # Для детальной отладки
            # traceback.print_exc() # Раскомментировать для вывода стека вызовов
            continue # Переходим к следующей группе

    # --- Сборка и пост-обработка ---
    if not processed_groups:
        print("\n--- Ни одна группа не была успешно обработана или все оказались пустыми. Выход. ---")
        return

    print("\n--- Сборка результатов ---")
    final_df = pd.concat(processed_groups, ignore_index=True)
    print(f"Итоговый датасет содержит {len(final_df)} строк и {len(final_df.columns)} столбцов.")

    # --- Шаг 4: Переименование файлов (если выбрано) ---
    filename_mapping_df = None
    if config['rename_files']:
        if 'file_name' not in final_df.columns:
             print("Предупреждение: Столбец 'file_name' не найден для переименования. Пропуск шага.")
        else:
            if verbose: print("  Шаг 4: Переименование file_name...")
            start_id = config.get('start_file_id', 0) # Получаем начальный ID, по умолчанию 0
            if not isinstance(start_id, int):
                print(f"Предупреждение: start_file_id ('{start_id}') не является целым числом. Используется 0.")
                start_id = 0

            unique_files = final_df['file_name'].unique()
            # Генерируем ID начиная с start_id
            filename_mapping = {i + start_id: old_name for i, old_name in enumerate(unique_files)}
            mapping = {old_name: i + start_id for i, old_name in enumerate(unique_files)}
            final_df['file_name'] = final_df['file_name'].map(mapping)

            filename_mapping_df = pd.DataFrame(list(filename_mapping.items()), columns=['new_id', 'original_name'])
            if verbose: print(f"    Имена файлов заменены на ID от {start_id} до {start_id + len(unique_files) - 1}.")

    # --- Сохранение результатов ---
    print("\n--- Сохранение результатов ---")
    try:
        if filename_mapping_df is not None:
             filename_mapping_df.to_csv(mapping_csv_path, index=False)
             print(f"Словарь имен файлов сохранен в: {mapping_csv_path}")
        
        # Замена Inf на очень большие числа или NaN перед сохранением
        final_df = final_df.replace([np.inf, -np.inf], np.nan) 
        # Можно заменить NaN на что-то другое, если требуется CSV без пустых полей
        # final_df.fillna('NaN', inplace=True) 
        final_df.to_csv(output_csv_path, index=False, float_format='%.6g') # Форматирование для float
        print(f"Обработанный датасет сохранен в: {output_csv_path}")
             
    except Exception as e:
        print(f"Критическая ошибка при сохранении результатов: {e}")

    print(f"--- Обработка завершена ({file_counter} групп обработано) ---")

def normalize(
    input_csv_path: str,
    config: Dict,
    output_normalized_csv_path: str,
) -> None:
    """
    Выполняет нормализацию файлов для обработанного CSV.

    Args:
        input_csv_path (str): Путь к CSV файлу, полученному от process_oscillograms.
        config (Dict): Словарь конфигурации. Ожидаются ключи:
            normalize (bool): Выполнять ли нормализацию.
            normalization_params_path (Optional[str]): Путь для сохранения CSV с коэфф. нормализации.
            verbose (bool): Выводить ли подробные сообщения.
        output_normalized_csv_path (str): Путь для сохранения итогового нормализованного. CSV.
    """
    print(f"\n--- Начало нормализации: {input_csv_path} ---")
    verbose = config.get('verbose', False) # Получаем verbose из конфига

    # Проверка наличия обязательных ключей в конфиге для этой функции
    required_keys = ['normalize', 'normalization_params_path']
    if not all(key in config for key in required_keys):
        missing = [k for k in required_keys if k not in config]
        raise ValueError(f"Конфигурация для normalize_and_rename неполная. Отсутствуют ключи: {missing}")

    if not os.path.exists(input_csv_path):
         raise FileNotFoundError(f"Входной CSV файл для нормализации не найден: {input_csv_path}")

    try:
        final_df = pd.read_csv(input_csv_path)
        if verbose: print(f"Загружен файл: {input_csv_path}, строк: {len(final_df)}")
    except Exception as e:
        print(f"Критическая ошибка загрузки CSV файла {input_csv_path}: {e}")
        return

    # --- Нормализация (если выбрано) ---
    normalization_params_list = [] # Инициализируем здесь
    if config['normalize']:
        if verbose: print("  Шаг 5: Нормализация данных...")

        # Определяем колонки для нормализации (из загруженного DataFrame)
        magnitude_cols = [col for col in final_df.columns if col.endswith('_mag')]
        angle_cols = [col for col in final_df.columns if col.endswith('_angle')]
        # Определяем P/Q/Other колонки из оставшихся числовых
        numeric_cols = final_df.select_dtypes(include=np.number).columns
        other_numeric_cols = [col for col in numeric_cols
                              if col not in magnitude_cols + angle_cols
                              and not final_df[col].name.lower().startswith('time') # Исключаем столбец времени, если есть
                              and col not in ['file_name', 'change_event'] # Исключаем служебные и целевые
                              and not col.endswith('_PS') # Исключаем целевые сигналы
                             ]

        # --- Вспомогательная функция normalize_magnitudes (внутри normalize_and_rename) ---
        def normalize_magnitudes(cols, df, group_name, params_list):
            if not cols: return df
            data_flat = df[cols].values.flatten()
            data_flat = data_flat[~np.isnan(data_flat) & np.isfinite(data_flat)]

            if len(data_flat) > 1:
                mean = np.mean(data_flat)
                std = np.std(data_flat)
                params_list.append({
                    'group_type': group_name,
                    'columns_normalized': ', '.join(cols),
                    'normalization_type': 'z_score',
                    'mean': mean,
                    'std_dev': std if std > 1e-9 else 0.0,
                    'divisor': np.nan
                })
                if std > 1e-9:
                    df[cols] = (df[cols] - mean) / std
                    if verbose: print(f"    Нормализованы {group_name} (mean={mean:.4f}, std={std:.4f}): {len(cols)} столбцов")
                else:
                     df[cols] = df[cols] - mean
                     if verbose: print(f"    Центрированы {group_name} (std близко к 0): {len(cols)} столбцов")
            elif verbose:
                 print(f"    Недостаточно данных для нормализации {group_name}: {len(cols)} столбцов")
            return df
        # --- Конец вспомогательной функции ---

        # Применяем к группам магнитуд и другим числовым
        current_mags = [c for c in magnitude_cols if c.startswith('I')]
        voltage_mags = [c for c in magnitude_cols if c.startswith('U') or c.startswith('V')]
        impedance_mags = [c for c in magnitude_cols if c.startswith('Z')]
        power_mags = [c for c in magnitude_cols if c.startswith('S')]
        power_p_cols = [c for c in other_numeric_cols if c.startswith('P')]
        power_q_cols = [c for c in other_numeric_cols if c.startswith('Q')]
        other_cols = [c for c in other_numeric_cols if not (c.startswith('P') or c.startswith('Q'))]

        final_df = normalize_magnitudes(current_mags, final_df, "Current_Magnitudes", normalization_params_list)
        final_df = normalize_magnitudes(voltage_mags, final_df, "Voltage_Magnitudes", normalization_params_list)
        final_df = normalize_magnitudes(impedance_mags, final_df, "Impedance_Magnitudes", normalization_params_list)
        final_df = normalize_magnitudes(power_mags, final_df, "Power_S_Magnitudes", normalization_params_list)
        final_df = normalize_magnitudes(power_p_cols, final_df, "Power_P_Values", normalization_params_list)
        final_df = normalize_magnitudes(power_q_cols, final_df, "Power_Q_Values", normalization_params_list)
        final_df = normalize_magnitudes(other_cols, final_df, "Other_Numeric", normalization_params_list)

        # Нормализация углов к [-1, 1]
        if angle_cols:
            for col in angle_cols:
                 if col in final_df.columns:
                     final_df[col] = final_df[col] / np.pi
            if verbose: print(f"    Углы нормализованы к [-1, 1]: {len(angle_cols)} столбцов")
            normalization_params_list.append({
                'group_type': 'Angles',
                'columns_normalized': ', '.join(angle_cols),
                'normalization_type': 'divide_by_pi',
                'mean': np.nan,
                'std_dev': np.nan,
                'divisor': np.pi
            })

    # --- Сохранение коэффициентов нормализации (если требуется) ---
    if config['normalize'] and config.get('normalization_params_path') and normalization_params_list:
        params_path = config['normalization_params_path']
        try:
            params_df = pd.DataFrame(normalization_params_list)
            with open(params_path, 'w', encoding='utf-8') as f:
                f.write("# Coefficients used for data normalization\n")
                f.write("# group_type: Category of signals normalized together.\n")
                f.write("# columns_normalized: Comma-separated list of columns in the processed dataset affected by this row.\n")
                f.write("# normalization_type: 'z_score' ((X-mean)/std) or 'divide_by_pi'.\n")
                f.write("# mean, std_dev: Parameters for z_score (std_dev=0 if original std was near zero).\n")
                f.write("# divisor: Parameter for angle normalization.\n")
                params_df.to_csv(f, index=False, lineterminator='\n')
            if verbose: print(f"  Коэффициенты нормализации сохранены в: {params_path}")
        except Exception as e:
            print(f"\nПредупреждение: Не удалось сохранить коэффициенты нормализации в {params_path}. Ошибка: {e}")

    # --- Сохранение результатов ---
    print("\n--- Сохранение итогового файла ---")
    try:
        # Замена Inf на NaN перед сохранением
        final_df = final_df.replace([np.inf, -np.inf], np.nan)
        final_df.to_csv(output_normalized_csv_path, index=False, float_format='%.6g')
        print(f"Нормализованный датасет сохранен в: {output_normalized_csv_path}")

    except Exception as e:
        print(f"Критическая ошибка при сохранении итоговых результатов: {e}")

    print(f"--- Нормализация файла {input_csv_path} завершены ---")

def apply_normalization(
    input_csv_path: str,
    params_csv_path: str,
    output_normalized_csv_path: str,
    verbose: bool = False
) -> None:
    """
    Применяет ранее рассчитанные параметры нормализации к новому CSV файлу.

    Args:
        input_csv_path (str): Путь к CSV файлу, который нужно нормализовать.
                               (Предполагается, что он имеет структуру,
                               аналогичную той, для которой считались параметры).
        params_csv_path (str): Путь к CSV файлу с параметрами нормализации
                               (созданному функцией normalize_and_rename).
        output_normalized_csv_path (str): Путь для сохранения нормализованного CSV файла.
        verbose (bool): Выводить ли подробные сообщения во время обработки.
    """
    print(f"\n--- Начало применения нормализации к файлу: {input_csv_path} ---")
    print(f"  Используются параметры из: {params_csv_path}")

    # --- Загрузка данных ---
    if not os.path.exists(input_csv_path):
         raise FileNotFoundError(f"Входной CSV файл для нормализации не найден: {input_csv_path}")
    if not os.path.exists(params_csv_path):
         raise FileNotFoundError(f"Файл с параметрами нормализации не найден: {params_csv_path}")

    try:
        df_to_normalize = pd.read_csv(input_csv_path)
        if verbose: print(f"Загружен файл для нормализации: {input_csv_path}, строк: {len(df_to_normalize)}")
    except Exception as e:
        print(f"Критическая ошибка загрузки CSV файла {input_csv_path}: {e}")
        return

    try:
        # Пропускаем строки с комментариями при чтении параметров
        params_df = pd.read_csv(params_csv_path, comment='#')
        if verbose: print(f"Загружены параметры нормализации: {params_csv_path}, строк: {len(params_df)}")
    except Exception as e:
        print(f"Критическая ошибка загрузки файла параметров {params_csv_path}: {e}")
        return

    # --- Применение нормализации ---
    df_normalized = df_to_normalize.copy() # Работаем с копией

    # Итерация по строкам (группам параметров) в файле параметров
    for index, params_row in params_df.iterrows():
        norm_type = params_row['normalization_type']
        # Получаем список столбцов из строки, убираем пробелы
        cols_str = params_row['columns_normalized']
        if pd.isna(cols_str):
            if verbose: print(f"  Предупреждение: Пустой список столбцов для группы {params_row['group_type']}. Пропуск.")
            continue
            
        target_cols = [col.strip() for col in cols_str.split(',')]
        
        # Отбираем только те столбцы, которые реально есть в нашем датафрейме
        existing_cols = [col for col in target_cols if col in df_normalized.columns]
        
        if not existing_cols:
            if verbose: print(f"  Предупреждение: Ни один из столбцов группы '{params_row['group_type']}' не найден в {input_csv_path}. Пропуск группы.")
            continue
        
        if verbose: print(f"  Применение '{norm_type}' к группе '{params_row['group_type']}' (столбцы: {', '.join(existing_cols)})...")

        # Применяем соответствующий тип нормализации
        if norm_type == 'z_score':
            mean = params_row['mean']
            std_dev = params_row['std_dev']
            
            if pd.isna(mean) or pd.isna(std_dev):
                 print(f"  Предупреждение: Отсутствуют mean или std_dev для группы '{params_row['group_type']}'. Пропуск.")
                 continue

            # Применяем к существующим столбцам
            if std_dev > 1e-9: # Порог, как и при расчете
                df_normalized[existing_cols] = (df_normalized[existing_cols] - mean) / std_dev
            else: # Если стандартное отклонение было нулевым (константа)
                df_normalized[existing_cols] = df_normalized[existing_cols] - mean
                
        elif norm_type == 'divide_by_pi':
            divisor = params_row['divisor']
            
            if pd.isna(divisor) or divisor == 0:
                 print(f"  Предупреждение: Некорректный 'divisor' для группы '{params_row['group_type']}'. Пропуск.")
                 continue
                 
            # Применяем к существующим столбцам
            df_normalized[existing_cols] = df_normalized[existing_cols] / divisor
            
        else:
            print(f"  Предупреждение: Неизвестный тип нормализации '{norm_type}' для группы '{params_row['group_type']}'. Пропуск.")

    # --- Сохранение результата ---
    print("\n--- Сохранение нормализованного файла ---")
    try:
        # Замена Inf на NaN перед сохранением (на всякий случай)
        df_normalized = df_normalized.replace([np.inf, -np.inf], np.nan)
        df_normalized.to_csv(output_normalized_csv_path, index=False, float_format='%.6g')
        print(f"Нормализованный датасет сохранен в: {output_normalized_csv_path}")
    except Exception as e:
        print(f"Критическая ошибка при сохранении нормализованного файла: {e}")

    print(f"--- Применение нормализации к файлу {input_csv_path} завершено ---")


# --- Пример использования ---
if __name__ == "__main__":

    # --- Конфигурация ---
    config = {
        'target_signal': 'iPDR PS',      # iPDR PS - идеальный сигнал, rPDR PS - реальные
        'use_voltage_source': 'BB',      
        'samples_per_period': 32,        # Окно FFT (32 точек для 50 Гц при Fd=1600 Гц)
        'num_harmonics': 1,              
        # 'complex_format': 'mag_angle', # Закомментировано, т.к. теперь только этот формат
        'output_signals': [              # Запрашиваем только нужные сигналы
             # Гармоники U/I (они всегда считаются для расчетов, но добавим если нужны на выход)
             # 'UA_h1', 'UB_h1', 'UC_h1', 'IA_h1', 'IB_h1', 'IC_h1',
             # Симметричные составляющие
             'V_pos_seq', 'V_neg_seq',    
             'I_pos_seq', 'I_neg_seq',    
             # Сопротивления
             # Фазыне: Z_A, Z_B, Z_C
             # Линейные: Z_AB, Z_BC, Z_CA
             'Z_pos_seq',               
             # Мощности
             'P_pos_seq', 'P_neg_seq',       
             'Q_pos_seq', 'Q_neg_seq',   
             # Линейные напряжения
             # 'UAB_h1', 'UBC_h1', 'UCA_h1' 
             # Исходный ток IB (если был рассчитан)
             # 'IB_calculated' # Имя изменится на IB при расчете, если не было исходного
        ],
        # 'output_signals': 'ALL', # Можно использовать для отладки
        'rename_files': True,
        'start_file_id': 0, # Начало нумерации при переименовывании файлов
        'min_current_threshold': 1e-7,
        'verbose': False,                # Ставим False для чистого вывода (только прогресс-бар и ошибки)
                                         # Ставим True для детальной информации по каждой группе
        # --- Параметры для normalize_and_rename ---
        'normalize': True,
        'normalization_params_path': 'output/normalization_params.csv' # путь, куда сохранянтся параметры нормализации
    }

    # --- Обработка данных ---
    #input_csv_file = "dataset_cut_out_PDR_1600_v1.2 — копия.csv"
    #input_csv_file = "dataset_cut_out_PDR_1200_v1.2 — копия.csv"   
    #input_csv_file = "dataset_iPDR_v1.1.csv"   
    #process_oscillograms(
    #    csv_path=input_csv_file,
    #    config=config,
    #    #output_csv_path="output/dataset_cut_out_PDR_1600_v2.csv",
    #    #mapping_csv_path="output/filename_map_PDR_1600_v2.csv"
    #    #output_csv_path="output/dataset_cut_out_PDR_1200_v2.csv",
    #    #mapping_csv_path="output/filename_map_PDR_1200_v2.csv"
    #    output_csv_path="output/dataset_iPDR_v1.csv",
    #    mapping_csv_path="output/filename_map_iPDR_v1.csv"
    #)
    
    # --- Нормализация ---
    #new_input_csv_file = "output/dataset_cut_out_PDR_v2.csv" 
    #final_output_path = "output/dataset_cut_out_PDR_norm_v1.csv"
    #normalize(
    #    input_csv_path=new_input_csv_file, # объединённый датасет
    #    config=config, # Передаем весь конфиг, функция возьмет нужное
    #    output_normalized_csv_path=final_output_path
    #)
    
    # --- Применение нормализации ---
    another_processed_file = "output/dataset_iPDR_v1.csv"
    params_file_to_apply = "output/normalization_params.csv" # Используем ранее сохраненные параметры
    output_path_for_applied = "output/dataset_iPDR_norm_v1.csv"
    apply_normalization(
        input_csv_path=another_processed_file,
        params_csv_path=params_file_to_apply,
        output_normalized_csv_path=output_path_for_applied,
    )