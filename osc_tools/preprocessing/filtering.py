import os
import sys
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)
from osc_tools.features.normalization import NormOsc
from dataflow.comtrade_processing import ReadComtrade

class ChannelType(Enum):
    CURRENT = 'current'
    VOLTAGE = 'voltage'
    NONE = 'none'

def sliding_window_fft(signal: np.ndarray, fft_window_size: int, num_harmonics: int, verbose: bool = False) -> np.ndarray:
    """
    Выполняет БПФ в скользящем окне.
    Возвращает комплексные значения для указанного числа гармоник для каждого окна.
    Результат относится к началу окна.
    Размерность выходного массива: (len(signal), num_harmonics).
    Если окно выходит за пределы сигнала или гармоника не может быть вычислена,
    соответствующие значения будут np.nan + 1j*np.nan.
    """
    n_points = len(signal)
    # Инициализируем результат NaNами комплексного типа
    fft_results = np.full((n_points, num_harmonics), np.nan + 1j*np.nan, dtype=complex)

    if n_points < fft_window_size:
        if verbose:
            print(f"    Предупреждение: Длина сигнала ({n_points}) меньше окна FFT ({fft_window_size}). Пропуск FFT.")
        return fft_results # Возвращаем массив NaNов нужной формы

    # Цикл по началу каждого окна
    for i in range(n_points - fft_window_size + 1):
        window_data = signal[i : i + fft_window_size]
        
        # Вычисляем FFT и нормируем для получения амплитуд (пока без *2)
        fft_coeffs = np.fft.fft(window_data) / fft_window_size 
        
        # Временный массив для хранения гармоник текущего окна
        current_window_harmonics = np.full(num_harmonics, np.nan + 1j*np.nan, dtype=complex)
        
        # Извлекаем запрошенные гармоники (1-based)
        # h_num - номер гармоники (1, 2, ...)
        for h_num in range(1, num_harmonics + 1):
            if h_num < len(fft_coeffs): # fft_coeffs[0] - DC, fft_coeffs[k] - k-ая гармоника. len(fft_coeffs) == fft_window_size
                                        # Убеждаемся, что запрашиваемый индекс h_num существует.
                # Умножаем на 2 для восстановления амплитуды из одностороннего спектра
                # (кроме DC и Найквиста, но мы начинаем с h_num=1 и обычно num_harmonics << fft_window_size/2)
                current_window_harmonics[h_num - 1] = fft_coeffs[h_num] * 2
            else:
                # Если запрашиваемая гармоника h_num выходит за пределы fft_coeffs (т.е. h_num >= fft_window_size),
                # то оставляем NaN. Дальнейшие гармоники также будут недоступны.
                break 
        
        fft_results[i, :] = current_window_harmonics
            
    return fft_results


class EmptyOscFilter:    
    def __init__(self,
                 comtrade_root_path: str,
                 output_csv_path: str,
                 config: Dict[str, Any],
                 norm_coef_csv_path: Optional[str] = None,
                 ):
        self.comtrade_root_path = comtrade_root_path
        self.output_csv_path = output_csv_path
        self.config = config
        self.norm_coef_csv_path = norm_coef_csv_path
        self.readComtrade = ReadComtrade()
        
        self.normalizer = None
        if self.norm_coef_csv_path and self.config.get('use_norm_osc', False):
            if os.path.exists(self.norm_coef_csv_path):
                self.normalizer = NormOsc(norm_coef_file_path=self.norm_coef_csv_path)
            else:
                print(f"Предупреждение: Файл коэффициентов нормализации {self.norm_coef_csv_path} не найден. Нормализация NormOsc будет отключена.")
                self.config['use_norm_osc'] = False # Отключаем, если файла нет
        else:
            self.config['use_norm_osc'] = False # Отключаем, если путь не задан или use_norm_osc изначально False

        self.fft_window_size = -1 # Должно считываться с файла
        
        self.channels_to_analyze_patterns_lower = [
            p.lower() for p in self.config.get('channels_to_analyze_patterns', ['i |', 'u |', 'phase: n'])
        ]
        self.current_id_patterns_lower = [
            p.lower() for p in self.config.get('current_channel_id_patterns', ['i |'])
        ]
        self.voltage_id_patterns_lower = [
            p.lower() for p in self.config.get('voltage_channel_id_patterns', ['u |'])
        ]
        self.verbose = self.config.get('verbose', False)

    def _find_osc_files(self) -> List[str]:
        """Находит все .cfg файлы рекурсивно."""
        cfg_files = []
        for root, _, files in os.walk(self.comtrade_root_path):
            for file in files:
                if file.lower().endswith(".cfg"):
                    cfg_files.append(os.path.join(root, file))
        return sorted(cfg_files)

    def _get_h1_amplitude_series(self, signal_values: np.ndarray) -> Optional[pd.Series]:
        """Рассчитывает временной ряд амплитуд первой гармоники."""
        if len(signal_values) < self.fft_window_size:
            if self.verbose:
                print(f"    Длина сигнала ({len(signal_values)}) меньше окна FFT ({self.fft_window_size}). Пропуск H1.")
            return None

        fft_complex_all_points = sliding_window_fft(signal_values, self.fft_window_size, num_harmonics=1, verbose=self.verbose)
        
        # fft_complex_all_points имеет форму (n_points, 1)
        # Извлекаем первую гармонику (единственный столбец)
        h1_complex_series_all_points = fft_complex_all_points[:, 0]
        
        h1_amplitude_series_all_points = np.abs(h1_complex_series_all_points) / np.sqrt(2)
        
        # Отбрасываем NaN значения, возникшие там, где окно FFT не могло быть сформировано
        # (в основном, в конце сигнала)
        valid_h1_amplitudes = h1_amplitude_series_all_points[~np.isnan(h1_amplitude_series_all_points)]

        if len(valid_h1_amplitudes) == 0:
             if self.verbose:
                print(f"    Не удалось рассчитать амплитуды H1 (все значения NaN).")
             return None
        
        return pd.Series(valid_h1_amplitudes)

    def is_oscillogram_not_empty(self, osc_df: pd.DataFrame, file_name_for_norm: str) -> bool:
        if osc_df is None or osc_df.empty:
            return False

        processed_df = osc_df.copy()
        use_normalized_thresholds_for_this_file = False

        if self.config.get('use_norm_osc', False) and self.normalizer:
            if self.verbose:
                print(f"  Попытка применения нормализации NormOsc для {file_name_for_norm}...")
            
            original_df_for_raw_analysis = processed_df.copy() 
            normalized_df_ext = self.normalizer.normalize_bus_signals(processed_df, file_name_for_norm, is_print_error=self.verbose)
            
            # Проверяем, была ли нормализация успешно применена (например, файл был в таблице norm_coef и norm='YES')
            # Эта проверка зависит от внутренней логики self.normalizer.normalize_bus_signals
            # Для упрощения, предполагаем, что если DataFrame изменился, нормализация прошла.
            # Более точная проверка может потребовать доступ к результатам из normalizer.
            # Здесь предполагаем, что normalize_bus_signals возвращает измененный df если нормализация удалась,
            # и исходный (или его копию) если нет.
            # Однако, по условию, он может вернуть исходный df если файл не найден.
            # Нужна более явная проверка статуса нормализации от NormOsc.
            # Пока что используем флаг, как в исходном коде, на основе norm_row:
            norm_applied_flag = False
            if self.normalizer.norm_coef is not None: # norm_coef это атрибут NormOsc, который должен быть DataFrame
                norm_row = self.normalizer.norm_coef[self.normalizer.norm_coef["name"] == file_name_for_norm]
                if not norm_row.empty and norm_row["norm"].values[0] == "YES":
                    norm_applied_flag = True
            
            if norm_applied_flag and normalized_df_ext is not None: # и сам DataFrame был возвращен
                processed_df = normalized_df_ext
                use_normalized_thresholds_for_this_file = True
                if self.verbose:
                    print(f"    Нормализация NormOsc успешно применена. Будут использованы *_normalized пороги.")
            else:
                if self.verbose:
                    print(f"    NormOsc не был применен (файл не найден в coef или norm != YES, или normalize_bus_signals вернул None). Переход к анализу 'сырых' данных.")
                processed_df = original_df_for_raw_analysis
                use_normalized_thresholds_for_this_file = False
        else:
            if self.verbose:
                print(f"  Нормализация NormOsc отключена или недоступна. Будет выполнен анализ 'сырых' данных.")
            use_normalized_thresholds_for_this_file = False

        at_least_one_channel_active = False

        for col_name in processed_df.columns:
            col_name_lower = col_name.lower()
            
            is_candidate_channel = any(
                patt in col_name_lower for patt in self.channels_to_analyze_patterns_lower
            )
            if not is_candidate_channel:
                continue
            
            try:
                signal_values_full_series = processed_df[col_name].astype(float).values
            except ValueError:
                if self.verbose: print(f"    Не удалось преобразовать канал {col_name} в числовой формат. Пропуск.")
                continue

            channel_type = ChannelType.NONE
            if any(p_curr in col_name_lower for p_curr in self.current_id_patterns_lower):
                channel_type = ChannelType.CURRENT
            elif any(p_volt in col_name_lower for p_volt in self.voltage_id_patterns_lower):
                channel_type = ChannelType.VOLTAGE
            
            if channel_type == ChannelType.NONE:
                if self.verbose: print(f"    Канал {col_name} не определен как ток/напряжение. Пропуск.")
                continue

            current_thresholds_set = None
            h1_for_relative_norm = None 

            if use_normalized_thresholds_for_this_file and not("_dup" in str(col_name)):
                current_thresholds_set = self.config[f"thresholds_{channel_type.name.lower()}_normalized"]
                if self.verbose: print(f"  Анализ канала: {col_name} (тип: {channel_type.name.lower()}, режим: NormOsc)")
            else: # Анализ "сырых" данных
                if self.verbose: print(f"  Анализ канала: {col_name} (тип: {channel_type.name.lower()}, режим: Raw)")
                is_clean, h1_initial_amplitude = self._is_initial_signal_clean(signal_values_full_series, channel_type.name.lower())
                if not is_clean:
                    if self.verbose: print(f"    Канал {col_name} не прошел проверку на 'чистоту' начального сигнала. Считается неактивным.")
                    continue 
                
                current_thresholds_set = self.config['raw_signal_analysis'][f'thresholds_raw_{channel_type.name.lower()}_relative']
                h1_for_relative_norm = h1_initial_amplitude
                if h1_for_relative_norm is None or h1_for_relative_norm < 1e-6: 
                    if self.verbose: print(f"    Начальная амплитуда H1 для канала {col_name} слишком мала ({h1_for_relative_norm}). Канал считается неактивным.")
                    continue

            h1_amplitudes_full_series = self._get_h1_amplitude_series(signal_values_full_series)

            if h1_amplitudes_full_series is None or h1_amplitudes_full_series.empty or h1_amplitudes_full_series.isna().all():
                if self.verbose: print(f"    Нет валидных данных первой гармоники для канала {col_name}.")
                continue
            
            h1_series_to_analyze = h1_amplitudes_full_series.copy()
            if (not use_normalized_thresholds_for_this_file or ("_dup" in str(col_name))) and h1_for_relative_norm is not None:
                # h1_for_relative_norm здесь не может быть 0 или слишком мал, т.к. отсеяли ранее
                h1_series_to_analyze = h1_series_to_analyze / h1_for_relative_norm
                if self.verbose: print(f"    Амплитуды H1 для '{col_name}' нормализованы по начальной H1={h1_for_relative_norm:.4f}.")

            if len(h1_series_to_analyze) < 1: # Должно быть отсеяно ранее, но на всякий случай
                if self.verbose: print(f"    Ряд H1 для '{col_name}' пуст после обработки. Пропуск.")
                continue
            
            stat_h1_max_abs = h1_series_to_analyze.max()
            if len(h1_series_to_analyze) == 1:
                stat_h1_delta = 0.0
                stat_h1_std_dev = 0.0 # std для одного элемента это NaN, но для логики лучше 0
            else:
                stat_h1_delta = h1_series_to_analyze.max() - h1_series_to_analyze.min()
                stat_h1_std_dev = h1_series_to_analyze.std()

            if self.verbose:
                mode_str = "NormOsc" if use_normalized_thresholds_for_this_file else "Raw/Relative"
                print(f"    Статистики H1 для {col_name} (режим {mode_str}): MaxAbs={stat_h1_max_abs:.4f}, Delta={stat_h1_delta:.4f}, StdDev={stat_h1_std_dev:.4f}")
                print(f"    Применяемые пороги: Delta={current_thresholds_set['delta']}, StdDev={current_thresholds_set['std_dev']}", end="")
                if 'max_abs' in current_thresholds_set:
                    print(f", MaxAbs={current_thresholds_set['max_abs']}")
                else:
                    print("")

            # Защита от необработанных сигналов при нормализации (Пока убрал: или вероятных ошибок по номиналу (220В при номинале 100В))
            # if "_dup" in str(col_name) and use_normalized_thresholds_for_this_file: # or stat_h1_max_abs > 0.6:
            #     continue
            # Перенёс её выше
            
            is_active_by_delta = stat_h1_delta > current_thresholds_set['delta']
            is_active_by_std = stat_h1_std_dev > current_thresholds_set['std_dev'] # np.nan > x is False
            is_active_by_max_abs = False
            if 'max_abs' in current_thresholds_set: 
                is_active_by_max_abs = stat_h1_max_abs > current_thresholds_set['max_abs']

            if (is_active_by_delta or is_active_by_std) and (is_active_by_max_abs or not norm_applied_flag):
                if self.verbose:
                    print(f"    Канал {col_name} активен (Delta: {is_active_by_delta}, StdDev: {is_active_by_std}, MaxAbs: {is_active_by_max_abs}). Осциллограмма не пустая.")
                at_least_one_channel_active = True
                break 

        return at_least_one_channel_active

    def run_filter(self):
        """Запускает процесс фильтрации."""
        cfg_files = self._find_osc_files()
        if not cfg_files:
            print("Осциллограммы (.cfg файлы) не найдены.")
            return

        non_empty_file_names = []
        unread_files_exceptions = []

        print(f"Найдено {len(cfg_files)} осциллограмм для анализа.")
        
        for cfg_file_path in tqdm(cfg_files, desc="Фильтрация осциллограмм"):
            file_basename = os.path.basename(cfg_file_path)
            file_name_for_norm = file_basename[:-4] 
            if self.verbose:
                print(f"\nОбработка файла: {cfg_file_path}")
            
            try:
                raw_date, osc_df_raw = self.readComtrade.read_comtrade(cfg_file_path)
                frequency = raw_date.cfg.frequency
                samples_rate = raw_date.cfg.sample_rates[0][0]
                self.fft_window_size = int(samples_rate / frequency)
                if raw_date is None or osc_df_raw is None or osc_df_raw.empty:
                     if self.verbose: print(f"  Файл {cfg_file_path} не содержит данных или ошибка чтения comtrade. Пропуск.")
                     continue

                original_analog_ids_from_rec = list(raw_date.analog_channel_ids) # Исходные ID из записи COMTRADE
                num_expected_analog_channels = len(original_analog_ids_from_rec)

                time_col_name = 'time' # Стандартное имя для столбца времени
                has_time_column_in_raw = time_col_name in osc_df_raw.columns
                
                # Определяем индекс первого аналогового столбца в osc_df_raw
                # Предполагаем, что аналоговые каналы идут подряд после столбца 'time' (если он есть)
                first_analog_col_idx_in_raw_df = 0
                if has_time_column_in_raw:
                    # Найдем позицию столбца 'time'. Следующий за ним - первый аналоговый.
                    time_col_position = osc_df_raw.columns.get_loc(time_col_name)
                    first_analog_col_idx_in_raw_df = time_col_position + 1
                
                # Проверка, достаточно ли столбцов в osc_df_raw для всех ожидаемых аналоговых каналов
                if first_analog_col_idx_in_raw_df + num_expected_analog_channels > len(osc_df_raw.columns):
                    if self.verbose:
                        print(f"  Ошибка: В DataFrame из файла {cfg_file_path} недостаточно столбцов ({len(osc_df_raw.columns)}) "
                              f"для ожидаемого количества аналоговых каналов ({num_expected_analog_channels}) "
                              f"начиная с индекса {first_analog_col_idx_in_raw_df}.")
                    unread_files_exceptions.append((cfg_file_path, "DataFrame column count mismatch for analog channels"))
                    continue
                
                # Генерируем целевые уникальные имена для аналоговых каналов в нашем итоговом osc_df
                target_unique_analog_names = self._make_unique_column_names(original_analog_ids_from_rec)

                # Создаем новый DataFrame osc_df с корректными данными и уникальными именами
                data_for_final_osc_df = {}
                
                # Копируем столбец времени, если он есть
                if has_time_column_in_raw:
                    data_for_final_osc_df[time_col_name] = osc_df_raw[time_col_name].copy()

                # Копируем данные аналоговых каналов, присваивая им новые уникальные имена
                for i, unique_name in enumerate(target_unique_analog_names):
                    # Имя столбца в osc_df_raw, соответствующего i-му аналоговому каналу
                    # (pandas мог уже переименовать дубликаты, например в 'Канал' и 'Канал.1')
                    # Мы берем столбцы по их *порядковому номеру* в osc_df_raw, который должен соответствовать порядку в rec.analog_channel_ids
                    original_df_column_for_this_analog_channel = osc_df_raw.columns[first_analog_col_idx_in_raw_df + i]
                    data_for_final_osc_df[unique_name] = osc_df_raw[original_df_column_for_this_analog_channel].copy()
                
                osc_df = pd.DataFrame(data_for_final_osc_df)

                # Устанавливаем индекс времени, если столбец времени был
                if has_time_column_in_raw and time_col_name in osc_df.columns:
                    osc_df = osc_df.set_index(time_col_name)
                elif not has_time_column_in_raw and osc_df_raw.index.name is not None:
                    # Если в исходном osc_df_raw время было индексом, восстанавливаем его
                    osc_df.index = osc_df_raw.index.copy() 
                    if osc_df.index.name is None and isinstance(osc_df_raw.index, pd.RangeIndex) == False : # Если имя индекса не было установлено, но это не RangeIndex
                        # Пытаемся дать имя 'time' по соглашению, если это не простой RangeIndex
                         osc_df.index.name = time_col_name
                
            except Exception as e:
                if self.verbose:
                    print(f"Ошибка обработки файла {cfg_file_path} на этапе подготовки DataFrame: {e}")
                unread_files_exceptions.append((cfg_file_path, f"DataFrame preparation error: {str(e)}"))
                continue # Переходим к следующему файлу
            
            if osc_df.empty and num_expected_analog_channels > 0: # Если ожидали каналы, но df пуст
                if self.verbose:
                     print(f"  DataFrame osc_df для {cfg_file_path} пуст после обработки, хотя ожидались каналы. Пропуск.")
                # Можно добавить в unread_files_exceptions, если это считать ошибкой
                continue
            elif osc_df.empty and num_expected_analog_channels == 0: # Пустой, т.к. не было аналоговых каналов
                 if self.verbose:
                     print(f"  В файле {cfg_file_path} нет аналоговых каналов для анализа. Пропуск.")
                 continue


            if self.is_oscillogram_not_empty(osc_df, file_name_for_norm):
                non_empty_file_names.append(file_basename) 

        if non_empty_file_names:
            result_df = pd.DataFrame({'non_empty_files': non_empty_file_names})
            try:
                result_df.to_csv(self.output_csv_path, index=False)
                print(f"\nСписок непустых осциллограмм сохранен в: {self.output_csv_path}")
                print(f"Найдено непустых осциллограмм: {len(non_empty_file_names)}")
            except Exception as e:
                print(f"\nОшибка сохранения CSV файла: {e}")
        else:
            print("\nНепустых осциллограмм не найдено.")

        if unread_files_exceptions:
            print(f"\nКоличество файлов, которые не удалось прочитать/обработать: {len(unread_files_exceptions)}")
            # Можно логировать unread_files_exceptions

    def _is_initial_signal_clean(self, signal_values: np.ndarray, channel_type: ChannelType) -> Tuple[bool, Optional[float]]:
            """
            Проверяет, является ли начальный участок сигнала "чистым" (доминирует первая гармоника).
            Возвращает (is_clean, h1_amplitude_at_start).
            """
            raw_analysis_config = self.config['raw_signal_analysis']
            periods_to_check = raw_analysis_config['initial_window_check_periods']
            # Предполагается, что fft_window_size соответствует одному периоду основной частоты
            points_to_check = periods_to_check * self.fft_window_size

            if len(signal_values) < points_to_check:
                if self.verbose:
                    print(f"    Недостаточно данных ({len(signal_values)}) для проверки чистоты ({points_to_check} точек).")
                return False, None
            
            window_data = signal_values[:points_to_check]
            fft_coeffs = np.fft.fft(window_data) / len(window_data) # Нормировка
            
            # Индекс первой гармоники (50 Гц) в окне `points_to_check`.
            # Если fft_window_size = 1 период, то в окне `periods_to_check * fft_window_size`
            # 1-я гармоника будет на индексе `periods_to_check`.
            idx_h1 = periods_to_check 
            
            if idx_h1 >= len(fft_coeffs) // 2: # Проверка, что такой индекс (и выше) существует
                if self.verbose: print(f"    Ошибка расчета индекса первой гармоники для проверки чистоты (idx_h1={idx_h1} слишком велик).")
                return False, None

            m1_amplitude = np.abs(fft_coeffs[idx_h1]) * 2 

            # Амплитуды высших гармоник (от следующего бина после H1 до Найквиста)
            # Это несколько отличается от "суммы высших гармоник", это максимум из всех бинов в этом диапазоне.
            start_higher_harm_idx = idx_h1 + 1 
            end_higher_harm_idx = len(fft_coeffs) // 2 

            mx_amplitude = 0.0
            if start_higher_harm_idx < end_higher_harm_idx : # Если есть диапазон для анализа
                higher_harm_amplitudes = np.abs(fft_coeffs[start_higher_harm_idx : end_higher_harm_idx + 1]) * 2
                if len(higher_harm_amplitudes) > 0:
                    mx_amplitude = np.max(higher_harm_amplitudes)
            
            if self.verbose:
                print(f"    Проверка чистоты: M1={m1_amplitude:.4f}, Max_HigherFreqComponent={mx_amplitude:.4f}")

            is_clean = False
            # Если mx_amplitude очень мал (избегаем деления на 0, и считаем сигнал зашумлённым)
            if mx_amplitude < 1e-9: 
                is_clean = False 
            else:
                ratio_h1_hx = m1_amplitude / mx_amplitude
                h1_vs_hx_ratio_threshold = raw_analysis_config['h1_vs_hx_ratio_threshold_U']
                if channel_type == ChannelType.CURRENT:
                    h1_vs_hx_ratio_threshold = raw_analysis_config['h1_vs_hx_ratio_threshold_I']
                is_clean = ratio_h1_hx > h1_vs_hx_ratio_threshold
                if self.verbose:
                    print(f"    Соотношение M1/Max_Hx = {ratio_h1_hx:.2f} (порог > {h1_vs_hx_ratio_threshold})")

            if is_clean:
                if self.verbose: print(f"    Начальный участок сигнала ({points_to_check} точек) 'чистый'.")
                return True, m1_amplitude
            else:
                if self.verbose: print(f"    Начальный участок сигнала 'шумный/искаженный'.")
                return False, None

    def _make_unique_column_names(self, name_list: List[str]) -> List[str]:
        """
        Генерирует список уникальных имен столбцов из исходного списка.
        К дубликатам добавляет суффикс _dupN.
        Например: ['A', 'B', 'A'] -> ['A', 'B', 'A_dup1']
        """
        counts = {}
        new_names = []
        for name in name_list:
            if name not in counts:
                counts[name] = 0
                new_names.append(name)
            else:
                counts[name] += 1
                new_names.append(f"{name}_dup{counts[name]}")
        return new_names

if __name__ == '__main__':
# --- Пример конфигурации ---
    config_filter = {
        'channels_to_analyze_patterns': ['I ', 'U ', 'phaze: N '],
        'current_channel_id_patterns': ['I ', 'ТОК'],
        'voltage_channel_id_patterns': ['U ', 'НАПРЯЖЕНИЕ'],
        'use_norm_osc': True, # Если False, будут использоваться 'thresholds_raw_*'

        # Пороги для ТОКОВЫХ каналов (когда use_norm_osc=True)
        'thresholds_current_normalized': {
            'delta': 0.1/20, 'std_dev': 0.05/20, 'max_abs': 0.005/20
        },
        # Пороги для каналов НАПРЯЖЕНИЯ (когда use_norm_osc=True)
        'thresholds_voltage_normalized': {
            'delta': 0.05/3, 'std_dev': 0.05/3/2, 'max_abs': 0.05/3
        },

        # Параметры для анализа "сырых" сигналов (когда use_norm_osc=False или NormOsc не применим)
        'raw_signal_analysis': {
            'initial_window_check_periods': 1, # Сколько периодов в начале проверять на "чистоту" синусоиды (1 период = fft_window_size точек)
            'h1_vs_hx_ratio_threshold_U': 10,
            'h1_vs_hx_ratio_threshold_I': 1.5, # m1 <= threshold * mx (из вашего CreateNormOsc NOISE_FACTOR) - порог для "чистоты"
                                               # чем БОЛЬШЕ это значение, тем ЧИЩЕ должен быть сигнал (меньше высших гармоник)
                                               # Если у вас NOISE_FACTOR = 1.5 (m1 <= 1.5 * mx), то здесь должно быть что-то вроде 1/1.5 ~ 0.67, если понимать как (mx / m1 < 1/1.5)
                                               # Либо если m1 > NOISE_FACTOR * mx, то h1_vs_hx_ratio_threshold = 1.5 (m1/mx > 1.5)
                                               # Давайте остановимся на h1_amplitude / hx_amplitude > threshold
                                               # То есть, первая гармоника должна быть ЗНАЧИТЕЛЬНО больше суммы высших.
                                               # В вашем коде CreateNormOsc: `if m1 > NOISE_FACTOR * mx else '?2'`
                                               # Это значит, что если m1 (амплитуда первой гармоники) > 1.5 * mx (амплитуда максимальной из высших), то сигнал считается "чистым".
                                               # Значит, h1_vs_hx_ratio_threshold = 1.5, и мы будем проверять m1/mx > 1.5

            # Пороги для "сырых" ТОКОВЫХ каналов, если начальная проверка пройдена
            # Эти пороги будут применяться к сигналам, нормализованным по своему локальному максимуму в окне проверки
            'thresholds_raw_current_relative': {
                'delta': 0.4, # Относительная дельта (0.15 = 15% от локального максимума)
                'std_dev': 0.2 # Относительное ст.отклонение
            },
            # Пороги для "сырых" каналов НАПРЯЖЕНИЯ, если начальная проверка пройдена
            'thresholds_raw_voltage_relative': {
                'delta': 0.05,
                'std_dev': 0.025
            }
        },
        'verbose': False
    }

    # --- Пути ---
    # Укажите правильные пути
    comtrade_data_directory = "Путь к папке с файлами осциллограмм"
    output_filtered_list_csv = "Путь к папке выходных докумнетов/filtered_oscillograms.csv" # имя итогово файла filtered_oscillograms
    normalization_coefficients_csv = "Путь к папке с номинальными коэффициентами/norm.csv" # Опционально, если use_norm_osc=True

    # --- Запуск фильтра ---
    if os.path.exists(comtrade_data_directory) and any(f.endswith(".cfg") for f in os.listdir(comtrade_data_directory)):
        filter_instance = EmptyOscFilter(
            comtrade_root_path=comtrade_data_directory,
            output_csv_path=output_filtered_list_csv,
            config=config_filter,
            norm_coef_csv_path=normalization_coefficients_csv if config_filter['use_norm_osc'] else None
        )
        filter_instance.run_filter()
    else:
        print(f"Папка с данными {comtrade_data_directory} не найдена или пуста. Пропустите создание демо-файлов или укажите правильный путь.")