"""
DatasetManager - Модуль для управления датасетами.

Функционал:
1. Разделение исходного датасета на train.csv и test.csv (80/20 по файлам)
2. Создание предрассчитанного CSV для тестовой выборки с:
   - 8 аналоговыми сигналами (IA, IB, IC, IN, UA, UB, UC, UN)
   - Расчётными признаками (FFT начиная с точки 32)
   - Метками

Использование:
    from osc_tools.data_management.dataset_manager import DatasetManager
    
    dm = DatasetManager('data/ml_datasets')
    dm.ensure_train_test_split()  # Создаёт train.csv/test.csv если их нет
    dm.create_precomputed_test_csv()  # Создаёт test_precomputed.csv
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from osc_tools.ml.labels import clean_labels, add_base_labels, get_target_columns, get_ml_columns
from osc_tools.features.pdr_calculator import sliding_window_fft
from osc_tools.features.phasor import calculate_symmetrical_components, calculate_power
from osc_tools.features.polar import calculate_polar_features


class DatasetManager:
    """
    Менеджер датасетов для проекта Scientific Research OSC ML.
    
    Управляет:
    - Разделением на train/test
    - Предварительным расчётом признаков для ускорения валидации
    """
    
    # Стандартные имена файлов
    MAIN_CSV = "labeled_2025_12_03.csv"
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    PRECOMPUTED_TEST_CSV = "test_precomputed.csv"
    NORM_COEF_CSV = "norm_coef_all_v1.4.csv"
    
    # Параметры FFT
    FFT_WINDOW = 32  # 20мс при 1600 Гц (1 период 50 Гц)
    SAMPLING_RATE = 1600
    
    # Аналоговые колонки (стандартизированные имена)
    ANALOG_COLS_RAW = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']
    
    def __init__(self, data_dir: str, norm_coef_path: Optional[str] = None):
        """
        Args:
            data_dir: Путь к директории с датасетами (обычно 'data/ml_datasets')
            norm_coef_path: Путь к файлу коэффициентов нормализации.
        """
        self.data_dir = Path(data_dir)
        self.custom_norm_path = Path(norm_coef_path) if norm_coef_path else None
        
        # Пути к файлам в родительской директории (для norm_coef)
        self.project_root = self.data_dir.parent.parent
        
    def _get_norm_coef_path(self) -> Path:
        """Возвращает путь к файлу коэффициентов нормализации."""
        if self.custom_norm_path and self.custom_norm_path.exists():
            return self.custom_norm_path
            
        # Пробуем несколько стандартных мест
        candidates = [
            self.data_dir / self.NORM_COEF_CSV,
            self.project_root / 'raw_data' / self.NORM_COEF_CSV,
            self.data_dir.parent / self.NORM_COEF_CSV,
        ]
        for p in candidates:
            if p.exists():
                return p
        return candidates[0]  # Вернём первый вариант даже если не найден
    
    def ensure_train_test_split(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42,
        force: bool = False
    ) -> Tuple[Path, Path]:
        """
        Гарантирует существование разделенных файлов train.csv и test.csv.
        
        Разделение происходит по уникальным файлам (file_name), чтобы избежать 
        утечки данных - все точки одного файла попадают в один набор.
        
        Args:
            test_size: Доля тестовой выборки (по умолчанию 0.2 = 20%)
            random_state: Seed для воспроизводимости
            force: Принудительно пересоздать файлы, даже если они существуют
            
        Returns:
            Кортеж путей (train_path, test_path)
        """
        train_path = self.data_dir / self.TRAIN_CSV
        test_path = self.data_dir / self.TEST_CSV
        
        if train_path.exists() and test_path.exists() and not force:
            print(f"[DatasetManager] Найдены существующие файлы: {train_path.name}, {test_path.name}")
            return train_path, test_path

        print(f"[DatasetManager] Создание разбиения из {self.MAIN_CSV}...")
        main_path = self.data_dir / self.MAIN_CSV
        
        if not main_path.exists():
            raise FileNotFoundError(f"Основной файл датасета не найден: {main_path}")

        df = pl.read_csv(main_path, infer_schema_length=50000, null_values=["NA", "nan", "null", ""])
        
        # Разбиваем по уникальным файлам
        unique_files = df['file_name'].unique().to_list()
        train_files, test_files = train_test_split(
            unique_files, test_size=test_size, random_state=random_state
        )
        
        train_df = df.filter(pl.col('file_name').is_in(train_files))
        test_df = df.filter(pl.col('file_name').is_in(test_files))
        
        train_df.write_csv(train_path)
        test_df.write_csv(test_path)
        
        print(f"[DatasetManager] Разбиение завершено:")
        print(f"  Train: {len(train_df):,} строк ({len(train_files)} файлов)")
        print(f"  Test:  {len(test_df):,} строк ({len(test_files)} файлов)")
        
        return train_path, test_path
    
    def _get_standardized_columns(self, df: pl.DataFrame) -> List[str]:
        """
        Определяет доступные аналоговые колонки в DataFrame.
        Возвращает список из 8 колонок в стандартном порядке.
        """
        available = set(df.columns)
        
        # Токи (всегда одинаковые)
        i_cols = ['IA', 'IB', 'IC', 'IN']
        
        # Напряжения (зависят от датасета)
        if 'UA BB' in available:
            u_cols = ['UA BB', 'UB BB', 'UC BB', 'UN BB']
        elif 'UA CL' in available:
            u_cols = ['UA CL', 'UB CL', 'UC CL', 'UN CL']
        elif 'UA' in available:
            u_cols = ['UA', 'UB', 'UC', 'UN']
        else:
            # Fallback на BB
            u_cols = ['UA BB', 'UB BB', 'UC BB', 'UN BB']
        
        return i_cols + u_cols
    
    def _compute_fft_phasors(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисляет комплексные фазоры первой гармоники (50 Гц) методом скользящего окна.
        
        ВАЖНО: Использует окно Ханнинга для согласованности с 
        pdr_calculator.sliding_window_fft, который применяется в OscillogramDataset.
        
        Args:
            signal: 1D массив сигнала (Time,)
            
        Returns:
            (real, imag) - Re и Im части фазора для каждой точки.
            Первые FFT_WINDOW точек содержат 0 (warmup).
        """
        phasors = sliding_window_fft(signal, self.FFT_WINDOW, num_harmonics=1)
        phasors = np.nan_to_num(phasors, nan=0.0, posinf=0.0, neginf=0.0)
        real = phasors[:, 0].real.astype(np.float32)
        imag = phasors[:, 0].imag.astype(np.float32)
        return real, imag

    def _harmonic_suffix(self, harmonic_idx: int) -> str:
        """Возвращает суффикс для гармоники (h=1 без суффикса)."""
        return "" if harmonic_idx == 1 else f"_h{harmonic_idx}"
    
    def _compute_symmetric_components(
        self, 
        phasors_a: complex, 
        phasors_b: complex, 
        phasors_c: complex
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисляет симметричные составляющие (прямая, обратная, нулевая).
        
        Args:
            phasors_a, phasors_b, phasors_c: Комплексные фазоры фаз (Time,)
            
        Returns:
            (s1, s2, s0) - комплексные массивы симм. составляющих
        """
        a = np.exp(1j * 2 * np.pi / 3)
        a2 = a * a
        
        s0 = (phasors_a + phasors_b + phasors_c) / 3.0
        s1 = (phasors_a + a * phasors_b + a2 * phasors_c) / 3.0
        s2 = (phasors_a + a2 * phasors_b + a * phasors_c) / 3.0
        
        return s1, s2, s0
    
    def create_precomputed_test_csv(self, force: bool = False, num_harmonics: int = 1) -> Path:
        """
        Создает CSV файл с предрассчитанными признаками для тестовой выборки.
        
        Структура колонок:
        1. Метаданные: sample, file_name
        2. Raw аналоговые (8): IA, IB, IC, IN, UA, UB, UC, UN  
        3. Phase Polar (16 * H): IA_mag/angle для h1, IA_h2_mag/angle для h2 и т.д.
        4. Symmetric Rect (12 * H): I1_re/im для h1, I1_h2_re/im для h2 и т.д.
        5. Symmetric Polar (12 * H): I1_mag/angle для h1, I1_h2_mag/angle для h2 и т.д.
        6. Phase Complex (16 * H): IA_re/im для h1, IA_h2_re/im для h2 и т.д.
        7. Power (8): P/Q для IA/IB/IC/IN
        8. Alpha-Beta (6): I_alpha/I_beta/I_zero, U_alpha/U_beta/U_zero
        9. Метки ML_*
        
        Первые 31 точка в расчётных колонках содержат 0 (warmup FFT).
        Валидные данные начинаются с точки 32 (индекс 31).
        
        Args:
            force: Принудительно пересоздать файл
            num_harmonics: Количество гармоник для предрасчёта
            
        Returns:
            Путь к созданному файлу
        """
        output_path = self.data_dir / self.PRECOMPUTED_TEST_CSV
        
        if output_path.exists() and not force:
            print(f"[DatasetManager] Найден существующий файл: {output_path.name}")
            return output_path
        
        num_harmonics = max(1, int(num_harmonics))
        print(f"[DatasetManager] Создание предрассчитанного тестового датасета (гармоники={num_harmonics})...")
        
        # Убеждаемся что train/test разделение существует
        _, test_path = self.ensure_train_test_split()
        
        # Загружаем тестовые данные
        test_df = pl.read_csv(test_path, infer_schema_length=50000, null_values=["NA", "nan", "null", ""])
        test_df = clean_labels(test_df)
        test_df = add_base_labels(test_df)
        
        # Определяем колонки
        analog_cols = self._get_standardized_columns(test_df)
        ml_cols = get_ml_columns(test_df)
        target_cols = get_target_columns('base')
        
        print(f"  Аналоговые колонки: {analog_cols}")
        print(f"  Количество ML меток: {len(ml_cols)}")
        
        # Загружаем коэффициенты нормализации
        norm_coef_path = self._get_norm_coef_path()
        norm_df = None
        if norm_coef_path.exists():
            try:
                norm_df = pl.read_csv(norm_coef_path, infer_schema_length=0)
                print(f"  Коэффициенты нормализации: {norm_coef_path.name}")
            except Exception as e:
                print(f"  Предупреждение: не удалось загрузить коэффициенты: {e}")
        
        # Имена колонок для расчётных признаков
        phase_polar_names: List[str] = []
        phase_complex_names: List[str] = []
        symmetric_rect_names: List[str] = []
        symmetric_polar_names: List[str] = []
        power_names = ['P_IA', 'Q_IA', 'P_IB', 'Q_IB', 'P_IC', 'Q_IC', 'P_IN', 'Q_IN']
        alpha_beta_names = ['I_alpha', 'I_beta', 'I_zero', 'U_alpha', 'U_beta', 'U_zero']

        for ch in ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']:
            for h in range(1, num_harmonics + 1):
                suffix = self._harmonic_suffix(h)
                phase_polar_names.extend([f'{ch}{suffix}_mag', f'{ch}{suffix}_angle'])
                phase_complex_names.extend([f'{ch}{suffix}_re', f'{ch}{suffix}_im'])
            
        for comp in ['I1', 'I2', 'I0', 'U1', 'U2', 'U0']:
            for h in range(1, num_harmonics + 1):
                suffix = self._harmonic_suffix(h)
                symmetric_rect_names.extend([f'{comp}{suffix}_re', f'{comp}{suffix}_im'])
                symmetric_polar_names.extend([f'{comp}{suffix}_mag', f'{comp}{suffix}_angle'])
        
        # Обрабатываем по файлам для прогресса
        unique_files = test_df['file_name'].unique().to_list()
        print(f"  Обработка {len(unique_files)} файлов...")
        
        all_rows = []
        
        for fname in tqdm(unique_files, desc="Предрасчёт признаков"):
            file_data = test_df.filter(pl.col('file_name') == fname)
            n_points = len(file_data)
            
            # Извлекаем сырые аналоговые данные
            raw_values = file_data.select(analog_cols).to_numpy().astype(np.float32)
            
            # Заполняем NaN нулями
            raw_values = np.nan_to_num(raw_values, nan=0.0)
            
            # Применяем нормализацию (если доступна)
            if norm_df is not None:
                raw_values = self._apply_normalization(raw_values, fname, norm_df)
            
            # Вычисляем фазоры для всех 8 каналов (несколько гармоник)
            phasors: List[np.ndarray] = []
            for i in range(8):
                p = sliding_window_fft(raw_values[:, i], self.FFT_WINDOW, num_harmonics)
                p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64)
                phasors.append(p)
            
            # Phase Polar: magnitude и относительный angle (опора UA/IA на 1-й гармонике)
            complex_features = np.stack(phasors, axis=1)  # (Time, 8, H)
            time_steps = complex_features.shape[0]
            complex_features_flat = complex_features.reshape(time_steps, 8 * num_harmonics)
            
            ua_phasor = phasors[4][:, 0]
            ia_phasor = phasors[0][:, 0]
            ua_mag = np.nanmean(np.abs(ua_phasor))
            if ua_mag > 1e-4:
                ref_phasor = ua_phasor
            else:
                ia_mag = np.nanmean(np.abs(ia_phasor))
                ref_phasor = ia_phasor if ia_mag > 1e-4 else None
            
            phase_polar = calculate_polar_features(complex_features_flat, ref_phasor)
            phase_polar = np.nan_to_num(phase_polar, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Phase Complex: Re/Im для каждого канала и гармоники
            phase_complex_list = []
            for i in range(8):
                for h in range(num_harmonics):
                    phase_complex_list.append(phasors[i][:, h].real)
                    phase_complex_list.append(phasors[i][:, h].imag)
            phase_complex = np.stack(phase_complex_list, axis=1).astype(np.float32)
            phase_complex = np.nan_to_num(phase_complex, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Symmetric Components (комплексные значения)
            i1, i2, i0 = calculate_symmetrical_components(phasors[0], phasors[1], phasors[2])
            if not np.allclose(raw_values[:, 3], 0):
                i0 = phasors[3] / 3
            
            u1, u2, u0 = calculate_symmetrical_components(phasors[4], phasors[5], phasors[6])
            if not np.allclose(raw_values[:, 7], 0):
                u0 = phasors[7] / 3
            
            components = [i1, i2, i0, u1, u2, u0]
            
            # Symmetric Rect (Re/Im)
            symmetric_rect_list = []
            for comp in components:
                for h in range(num_harmonics):
                    symmetric_rect_list.append(comp[:, h].real)
                    symmetric_rect_list.append(comp[:, h].imag)
            symmetric_rect = np.stack(symmetric_rect_list, axis=1).astype(np.float32)
            symmetric_rect = np.nan_to_num(symmetric_rect, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Symmetric Polar (Mag/Angle)
            symmetric_complex = np.stack(components, axis=1)  # (Time, 6, H)
            symmetric_complex_flat = symmetric_complex.reshape(time_steps, 6 * num_harmonics)
            symmetric_polar = calculate_polar_features(symmetric_complex_flat, ref_phasor)
            symmetric_polar = np.nan_to_num(symmetric_polar, nan=0.0, posinf=0.0, neginf=0.0)

            # Power (P/Q) для пар IA/UA, IB/UB, IC/UC, IN/UN
            power_features = []
            for i_idx, u_idx in zip(range(4), range(4, 8)):
                i_phasor = phasors[i_idx][:, 0]
                u_phasor = phasors[u_idx][:, 0]
                _, p_act, q_react = calculate_power(u_phasor, i_phasor)
                power_features.append(np.stack([p_act, q_react], axis=1))
            power_data = np.concatenate(power_features, axis=1).astype(np.float32)
            power_data = np.nan_to_num(power_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Alpha-Beta (для токов и напряжений)
            a_i, b_i, c_i = raw_values[:, 0], raw_values[:, 1], raw_values[:, 2]
            i_alpha = (2 / 3) * (a_i - 0.5 * b_i - 0.5 * c_i)
            i_beta = (2 / 3) * (np.sqrt(3) / 2 * (b_i - c_i))
            i_zero = (1 / 3) * (a_i + b_i + c_i)

            a_u, b_u, c_u = raw_values[:, 4], raw_values[:, 5], raw_values[:, 6]
            u_alpha = (2 / 3) * (a_u - 0.5 * b_u - 0.5 * c_u)
            u_beta = (2 / 3) * (np.sqrt(3) / 2 * (b_u - c_u))
            u_zero = (1 / 3) * (a_u + b_u + c_u)

            alpha_beta = np.stack([i_alpha, i_beta, i_zero, u_alpha, u_beta, u_zero], axis=1).astype(np.float32)
            alpha_beta = np.nan_to_num(alpha_beta, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Извлекаем метки
            labels = file_data.select(target_cols + ml_cols).to_numpy()
            
            # Извлекаем метаданные
            samples = file_data['sample'].to_numpy()
            file_names = file_data['file_name'].to_list()
            
            # Собираем все данные для этого файла
            for idx in range(n_points):
                row = {
                    'sample': int(samples[idx]),
                    'file_name': file_names[idx],
                }
                # Raw (8)
                for j, col in enumerate(['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']):
                    row[col] = float(raw_values[idx, j])
                # Phase Polar (16 * H)
                for j, col in enumerate(phase_polar_names):
                    row[col] = float(phase_polar[idx, j])
                # Symmetric Rect (12 * H)
                for j, col in enumerate(symmetric_rect_names):
                    row[col] = float(symmetric_rect[idx, j])
                # Symmetric Polar (12 * H)
                for j, col in enumerate(symmetric_polar_names):
                    row[col] = float(symmetric_polar[idx, j])
                # Phase Complex (16 * H)
                for j, col in enumerate(phase_complex_names):
                    row[col] = float(phase_complex[idx, j])
                # Power (8)
                for j, col in enumerate(power_names):
                    row[col] = float(power_data[idx, j])
                # Alpha-Beta (6)
                for j, col in enumerate(alpha_beta_names):
                    row[col] = float(alpha_beta[idx, j])
                # Labels
                for j, col in enumerate(target_cols + ml_cols):
                    row[col] = int(labels[idx, j])
                
                all_rows.append(row)
        
        # Создаём DataFrame и сохраняем
        print("  Сохранение CSV...")
        result_df = pl.DataFrame(all_rows)
        result_df.write_csv(output_path)
        
        print(f"[DatasetManager] Сохранено {len(result_df):,} строк в {output_path.name}")
        print(f"  Структура: {result_df.shape[1]} колонок")
        
        # Выводим информацию о колонках
        print(f"  - Метаданные: sample, file_name")
        print(f"  - Raw аналоговые (8): IA...UN")
        print(f"  - Phase Polar (16 * H): *_mag, *_angle")
        print(f"  - Symmetric Rect (12 * H): *_re, *_im")
        print(f"  - Symmetric Polar (12 * H): *_mag, *_angle")
        print(f"  - Phase Complex (16 * H): *_re, *_im")
        print("  - Power (8): P/Q для IA/IB/IC/IN")
        print("  - Alpha-Beta (6): I_alpha/I_beta/I_zero, U_alpha/U_beta/U_zero")
        print(f"  - Метки ({len(target_cols + ml_cols)}): Target_*, ML_*")
        
        return output_path
    
    def _apply_normalization(
        self, 
        raw_values: np.ndarray, 
        file_name: str, 
        norm_df: pl.DataFrame
    ) -> np.ndarray:
        """
        Применяет физическую нормализацию к сырым данным.
        
        Args:
            raw_values: Массив (Time, 8) с сырыми значениями
            file_name: Имя файла для поиска коэффициентов
            norm_df: DataFrame с коэффициентами
            
        Returns:
            Нормализованный массив
        """
        try:
            # Парсинг имени файла: hash_Bus X
            parts = file_name.split('_Bus ')
            if len(parts) != 2:
                return raw_values
                
            file_hash = parts[0]
            bus_num = int(parts[1].split('_')[0])
            
            # Поиск коэффициентов
            norm_row = norm_df.filter(pl.col("name") == file_hash)
            if norm_row.is_empty():
                return raw_values
            
            # Проверка флага
            norm_val = str(norm_row['norm'][0])
            if "YES" not in norm_val:
                return raw_values
            
            def safe_get(col_name: str) -> float:
                if col_name not in norm_row.columns:
                    return 0.0
                val = norm_row[col_name][0]
                try:
                    return float(val) if val is not None else 0.0
                except:
                    return 0.0
            
            # Применяем коэффициенты
            ip_nom = 20 * safe_get(f"{bus_num}Ip_base")
            iz_nom = 5 * safe_get(f"{bus_num}Iz_base")
            ub_nom = 3 * safe_get(f"{bus_num}Ub_base")
            
            result = raw_values.copy()
            
            if ip_nom > 0:
                result[:, 0:3] /= ip_nom  # IA, IB, IC
            if iz_nom > 0:
                result[:, 3] /= iz_nom     # IN
            if ub_nom > 0:
                result[:, 4:8] /= ub_nom   # UA, UB, UC, UN
                
            return result
            
        except Exception as e:
            # Если что-то пошло не так, возвращаем исходные данные
            return raw_values
    
    def get_precomputed_feature_columns(self, feature_mode: str, num_harmonics: int = 1) -> List[str]:
        """
        Возвращает имена колонок для указанного режима признаков.
        
        Args:
            feature_mode: 'raw', 'phase_polar', 'symmetric', 'symmetric_polar', 'phase_complex', 'power', 'alpha_beta'
            num_harmonics: Количество гармоник
            
        Returns:
            Список имён колонок
        """
        num_harmonics = max(1, int(num_harmonics))

        if feature_mode == 'raw':
            return ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']
        
        elif feature_mode == 'phase_polar':
            cols = []
            for ch in ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']:
                for h in range(1, num_harmonics + 1):
                    suffix = self._harmonic_suffix(h)
                    cols.extend([f'{ch}{suffix}_mag', f'{ch}{suffix}_angle'])
            return cols
        
        elif feature_mode == 'symmetric':
            cols = []
            for comp in ['I1', 'I2', 'I0', 'U1', 'U2', 'U0']:
                for h in range(1, num_harmonics + 1):
                    suffix = self._harmonic_suffix(h)
                    cols.extend([f'{comp}{suffix}_re', f'{comp}{suffix}_im'])
            return cols
        
        elif feature_mode == 'symmetric_polar':
            cols = []
            for comp in ['I1', 'I2', 'I0', 'U1', 'U2', 'U0']:
                for h in range(1, num_harmonics + 1):
                    suffix = self._harmonic_suffix(h)
                    cols.extend([f'{comp}{suffix}_mag', f'{comp}{suffix}_angle'])
            return cols
        
        elif feature_mode == 'phase_complex':
            cols = []
            for ch in ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']:
                for h in range(1, num_harmonics + 1):
                    suffix = self._harmonic_suffix(h)
                    cols.extend([f'{ch}{suffix}_re', f'{ch}{suffix}_im'])
            return cols

        elif feature_mode == 'power':
            return ['P_IA', 'Q_IA', 'P_IB', 'Q_IB', 'P_IC', 'Q_IC', 'P_IN', 'Q_IN']

        elif feature_mode == 'alpha_beta':
            return ['I_alpha', 'I_beta', 'I_zero', 'U_alpha', 'U_beta', 'U_zero']
        
        else:
            raise ValueError(f"Неизвестный feature_mode: {feature_mode}")
    
    def load_train_df(self) -> pl.DataFrame:
        """Загружает и подготавливает тренировочный DataFrame."""
        self.ensure_train_test_split()
        train_path = self.data_dir / self.TRAIN_CSV
        df = pl.read_csv(train_path, infer_schema_length=50000, null_values=["NA", "nan", "null", ""])
        df = clean_labels(df)
        df = add_base_labels(df)
        return df
    
    def load_test_df(self, precomputed: bool = False) -> pl.DataFrame:
        """
        Загружает и подготавливает тестовый DataFrame.
        
        Args:
            precomputed: Если True, загружает предрассчитанный файл
            
        Returns:
            DataFrame с тестовыми данными
        """
        self.ensure_train_test_split()
        
        if precomputed:
            precomputed_path = self.data_dir / self.PRECOMPUTED_TEST_CSV
            if not precomputed_path.exists():
                self.create_precomputed_test_csv()
            df = pl.read_csv(precomputed_path, infer_schema_length=50000)
            return df
        else:
            test_path = self.data_dir / self.TEST_CSV
            df = pl.read_csv(test_path, infer_schema_length=50000, null_values=["NA", "nan", "null", ""])
            df = clean_labels(df)
            df = add_base_labels(df)
            return df


def initialize_datasets(data_dir: str = "data/ml_datasets", force: bool = False) -> DatasetManager:
    """
    Утилитная функция для инициализации всех датасетов.
    
    Вызывается в начале экспериментов для гарантии существования:
    - train.csv
    - test.csv
    - test_precomputed.csv
    
    Args:
        data_dir: Путь к директории с датасетами
        force: Принудительно пересоздать файлы
        
    Returns:
        Настроенный DatasetManager
    """
    dm = DatasetManager(data_dir)
    dm.ensure_train_test_split(force=force)
    dm.create_precomputed_test_csv(force=force)
    return dm
