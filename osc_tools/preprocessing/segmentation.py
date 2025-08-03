import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, Any, List, Optional, Tuple

# Утилитарная функция из предыдущего кода
def _sliding_window_fft(signal: np.ndarray, fft_window_size: int, num_harmonics: int) -> np.ndarray:
    n_points = len(signal)
    fft_results = np.full((n_points, num_harmonics), np.nan + 1j * np.nan, dtype=complex)
    if n_points < fft_window_size:
        return fft_results
    for i in range(n_points - fft_window_size + 1):
        window_data = signal[i : i + fft_window_size]
        fft_coeffs = np.fft.fft(window_data) / fft_window_size
        if num_harmonics >= 1 and 1 < len(fft_coeffs):
            fft_results[i, 0] = fft_coeffs[1] * 2
    return fft_results

class OscillogramEventSegmenter:
    """
    Класс для поиска и выделения "событий" в осциллограммах, представленных в виде DataFrame.
    Поддерживает два режима: для предварительно нормализованных и для "сырых" сигналов.
    """
    def __init__(self,
                 fft_window_size: int,
                 config: Dict[str, Any],
                 signals_are_normalized: bool = False
                ):
        """
        Инициализация сегментатора.

        Args:
            fft_window_size (int): Размер окна для БПФ (количество точек за один период).
            config (Dict[str, Any]): Словарь с конфигурационными параметрами.
            signals_are_normalized (bool): Флаг, указывающий, нормализованы ли входные сигналы.
        """
        self.fft_window_size = fft_window_size
        self.config = config
        self.signals_are_normalized = signals_are_normalized

        # Общие параметры
        self.detection_window_periods = self.config.get('detection_window_periods', 5)
        self.padding_periods = self.config.get('padding_periods', 10)
        self.current_patterns = [p.lower() for p in self.config.get('current_patterns', [])]
        self.voltage_patterns = [p.lower() for p in self.config.get('voltage_patterns', [])]

        # Параметры, зависящие от режима
        if self.signals_are_normalized:
            self.thresholds_current = self.config['thresholds_current_normalized']
            self.thresholds_voltage = self.config['thresholds_voltage_normalized']
        else:
            self.raw_analysis_config = self.config['raw_signal_analysis']
            self.thresholds_current = self.raw_analysis_config['thresholds_raw_current_relative']
            self.thresholds_voltage = self.raw_analysis_config['thresholds_raw_voltage_relative']

    def _get_target_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Находит столбцы для анализа в DataFrame по заданным паттернам."""
        target_cols = {'current': [], 'voltage': []}
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        for col_lower, orig_col_name in df_cols_lower.items():
            # Паттерны для тока
            if any(p in col_lower for p in self.current_patterns):
                target_cols['current'].append(orig_col_name)
            # Паттерны для напряжения
            elif any(p in col_lower for p in self.voltage_patterns):
                target_cols['voltage'].append(orig_col_name)
        
        return target_cols

    def _calculate_h1_amplitude_series(self, signal_values: np.ndarray) -> np.ndarray:
        """Рассчитывает временной ряд амплитуд первой гармоники."""
        if len(signal_values) < self.fft_window_size:
            return np.array([])
            
        fft_complex = _sliding_window_fft(signal_values, self.fft_window_size, num_harmonics=1)
        h1_complex_series = fft_complex[:, 0]
        h1_amplitude_series = np.abs(h1_complex_series)
        
        # Заполняем NaN в конце для непрерывности
        valid_mask = ~np.isnan(h1_amplitude_series)
        if np.any(valid_mask):
            last_valid_value = h1_amplitude_series[valid_mask][-1]
            h1_amplitude_series[~valid_mask] = last_valid_value
        else:
            h1_amplitude_series.fill(0)
            
        return h1_amplitude_series

    def _get_clean_signal_h1_amplitude(self, signal_values: np.ndarray, channel_type: str) -> Optional[float]:
        """Проверяет "чистоту" начального участка сигнала и возвращает базовую амплитуду H1."""
        cfg = self.raw_analysis_config
        periods_to_check = cfg['initial_window_check_periods']
        points_to_check = periods_to_check * self.fft_window_size

        if len(signal_values) < points_to_check:
            return None
        
        window_data = signal_values[:points_to_check]
        fft_coeffs = np.fft.fft(window_data) / len(window_data)
        
        idx_h1 = periods_to_check
        if idx_h1 >= len(fft_coeffs) // 2:
            return None

        m1_amplitude = np.abs(fft_coeffs[idx_h1]) * 2
        
        higher_harm_amplitudes = np.abs(fft_coeffs[idx_h1 + 1 : len(fft_coeffs) // 2 + 1]) * 2
        mx_amplitude = np.max(higher_harm_amplitudes) if len(higher_harm_amplitudes) > 0 else 0.0

        if mx_amplitude < 1e-9: # Избегаем деления на ноль и считаем сигнал зашумленным
            return None
        
        ratio_h1_hx = m1_amplitude / mx_amplitude
        threshold = cfg[f'h1_vs_hx_ratio_threshold_{"I" if channel_type == "current" else "U"}']

        if ratio_h1_hx > threshold:
            return m1_amplitude
        return None

    def _mark_interesting_points(self, h1_series: np.ndarray, thresholds: Dict[str, float], use_max_abs_check: bool) -> np.ndarray:
        """Создает булеву маску интересных точек для одного канала."""
        num_points = len(h1_series)
        mask = np.zeros(num_points, dtype=bool)
        detection_window_size = self.detection_window_periods * self.fft_window_size

        if num_points < detection_window_size:
            return mask

        for i in range(num_points - detection_window_size + 1):
            window_h1 = h1_series[i : i + detection_window_size]
            
            delta = window_h1.max() - window_h1.min()
            std_dev = window_h1.std()
            
            is_active_by_variation = (delta > thresholds['delta']) or (std_dev > thresholds['std_dev'])

            if use_max_abs_check:
                max_abs = window_h1.max()
                is_active_by_max = max_abs > thresholds['max_abs']
                is_active = is_active_by_variation and is_active_by_max
            else:
                is_active = is_active_by_variation

            if is_active:
                mask[i : i + detection_window_size] = True
        
        return mask

    def _find_and_merge_events(self, full_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Находит, расширяет и сливает непрерывные блоки в маске."""
        if not np.any(full_mask):
            return []

        d = np.diff(full_mask.astype(int))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if full_mask[0]: starts = np.insert(starts, 0, 0)
        if full_mask[-1]: ends = np.append(ends, len(full_mask))
        if len(starts) == 0: return []

        padding_points = self.padding_periods * self.fft_window_size
        padded_zones = [[max(0, s - padding_points), min(len(full_mask), e + padding_points)] for s, e in zip(starts, ends)]
        
        padded_zones.sort(key=lambda x: x[0])
        merged = [padded_zones[0]]
        for current in padded_zones[1:]:
            last = merged[-1]
            if current[0] < last[1]:
                last[1] = max(last[1], current[1])
            else:
                merged.append(current)
        
        return [tuple(zone) for zone in merged]

    def process_single_dataframe(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Обрабатывает один DataFrame, находя и вырезая события."""
        if df.empty:
            return []

        file_name = df['file_name'].iloc[0]
        target_cols = self._get_target_columns(df)
        if not target_cols['current'] and not target_cols['voltage']:
            return []

        full_mask = np.zeros(len(df), dtype=bool)

        for channel_type, columns in target_cols.items():
            thresholds = self.thresholds_current if channel_type == 'current' else self.thresholds_voltage
            for col in columns:
                signal = df[col].values.astype(float)
                h1_series = self._calculate_h1_amplitude_series(signal)
                if len(h1_series) == 0:
                    continue
                
                h1_to_analyze = h1_series
                
                if not self.signals_are_normalized:
                    base_amplitude = self._get_clean_signal_h1_amplitude(signal, channel_type)
                    if base_amplitude is None or base_amplitude < 1e-6:
                        continue # Пропускаем "грязный" или нулевой канал
                    h1_to_analyze = h1_series / base_amplitude
                
                channel_mask = self._mark_interesting_points(
                    h1_to_analyze, 
                    thresholds, 
                    use_max_abs_check=self.signals_are_normalized
                )
                full_mask |= channel_mask

        event_zones = self._find_and_merge_events(full_mask)

        result_dfs = []
        for i, (start, end) in enumerate(event_zones):
            event_df = df.iloc[start:end].copy()
            event_df['file_name'] = f"{file_name}_event_{i+1}"
            result_dfs.append(event_df)

        return result_dfs

    def process_csv_file(self, input_csv_path: str, output_csv_path: str):
        """Читает CSV, обрабатывает все файлы в нем и сохраняет результат в новый CSV.
           Также сохраняет лог-файлы со списком всех обработанных файлов и файлов с найденными событиями.
        """
        if not os.path.exists(input_csv_path):
            print(f"Ошибка: Входной файл не найден: {input_csv_path}")
            return
        
        try:
            df_full = pd.read_csv(input_csv_path, index_col=False)
        except Exception as e:
            print(f"Ошибка чтения CSV файла: {e}")
            return

        all_event_dfs = []
        processed_file_names = []
        event_found_file_names = []

        grouped = df_full.groupby('file_name')
        
        for file_name, group_df in tqdm(grouped, desc="Обработка файлов в CSV"):
            # Логируем, что этот файл был взят в обработку
            processed_file_names.append(file_name)
            
            group_df = group_df.sort_index() 
            event_dfs_for_file = self.process_single_dataframe(group_df)
            
            if event_dfs_for_file:
                # Если найдены события, добавляем их в общий список и логируем имя файла
                all_event_dfs.extend(event_dfs_for_file)
                event_found_file_names.append(file_name)
        
        # --- Сохранение основного результата (сегментированных событий) ---
        if not all_event_dfs:
            print("\nИтоговых событий для сохранения не найдено.")
            # Создаем пустой файл с заголовками, чтобы было понятно, что обработка завершилась
            pd.DataFrame(columns=df_full.columns).to_csv(output_csv_path, index=False)
        else:
            final_df = pd.concat(all_event_dfs, ignore_index=True)
            try:
                final_df.to_csv(output_csv_path, index=False)
                print(f"\nНайдено и сохранено событий: {len(all_event_dfs)}")
                print(f"Результат сохранен в: {output_csv_path}")
            except Exception as e:
                print(f"\nОшибка сохранения итогового CSV файла: {e}")

        # --- Сохранение лог-файлов ---
        output_dir = os.path.dirname(output_csv_path)
        base_name = os.path.splitext(os.path.basename(output_csv_path))[0]

        # 1. Лог всех обработанных файлов
        processed_log_path = os.path.join(output_dir, f"{base_name}_processed_log.csv")
        try:
            pd.DataFrame({'processed_files': processed_file_names}).to_csv(processed_log_path, index=False)
            print(f"Список всех обработанных файлов ({len(processed_file_names)}) сохранен в: {processed_log_path}")
        except Exception as e:
            print(f"Ошибка сохранения лога обработанных файлов: {e}")

        # 2. Лог файлов с найденными событиями
        events_found_log_path = os.path.join(output_dir, f"{base_name}_events_found_log.csv")
        try:
            pd.DataFrame({'event_files': event_found_file_names}).to_csv(events_found_log_path, index=False)
            print(f"Список файлов с событиями ({len(event_found_file_names)}) сохранен в: {events_found_log_path}")
        except Exception as e:
            print(f"Ошибка сохранения лога файлов с событиями: {e}")


if __name__ == '__main__':
    # --- Конфигурация ---
    config = {
        'detection_window_periods': 10, # 10 + 5 = 15 периодов до/после события (300 мс для 50Гц или 250 мс для 60Гц) 
        'padding_periods': 5,
        'current_patterns': ['ia', 'ib', 'ic', 'in'],
        'voltage_patterns': ['ua', 'ub', 'uc', 'un', 'uab', 'ubc', 'uca'],

        # --- Режим 1: Пороги для ПРЕДВАРИТЕЛЬНО НОРМАЛИЗОВАННЫХ сигналов ---
        'thresholds_current_normalized': {
            'delta': 0.1/20, 'std_dev': 0.05/20, 'max_abs': 0.005/20
        },
        'thresholds_voltage_normalized': {
            'delta': 0.05/3, 'std_dev': 0.025/3, 'max_abs': 0.05/3
        },

        # --- Режим 2: Параметры для анализа "СЫРЫХ" сигналов ---
        'raw_signal_analysis': {
            'initial_window_check_periods': 2,
            'h1_vs_hx_ratio_threshold_U': 10, # H1 должна быть в 10 раз больше высших гармоник
            'h1_vs_hx_ratio_threshold_I': 1.5,  # H1 должна быть в 1.5 раза больше высших гармоник
            # Пороги для ОТНОСИТЕЛЬНЫХ изменений (доли от базовой амплитуды)
            'thresholds_raw_current_relative': {
                'delta': 0.4, 'std_dev': 0.2
            },
            'thresholds_raw_voltage_relative': {
                'delta': 0.05, 'std_dev': 0.025
            }
        }
    }

    # --- ЗАПУСК: АНАЛИЗ НОРМАЛИЗОВАННЫХ ДАННЫХ ---   
    print("\n--- ЗАПУСК: АНАЛИЗ НОРМАЛИЗОВАННЫХ ДАННЫХ ---")
    ## !!ВАЖНО!! 
    # Требуется задать верное количество точек на период
    FFT_WINDOW_SIZE = 12 # 600 / 50
    # Укажите пути к файлам
    temp_normalized_input = "D://DataSet//_delete_unlabeled_50_600//unlabeled_50_600.csv"
    temp_segmented_from_norm = "D://DataSet//_delete_unlabeled_50_600//unlabeled_50_600_v1.csv" # имя итогово файла
    
    segmenter_norm = OscillogramEventSegmenter(
        fft_window_size=FFT_WINDOW_SIZE,
        config=config,
        signals_are_normalized=True # Явно указываем режим
    )
    segmenter_norm.process_csv_file(temp_normalized_input, temp_segmented_from_norm)
