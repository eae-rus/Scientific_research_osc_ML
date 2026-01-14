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
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Путь к директории с датасетами (обычно 'data/ml_datasets')
        """
        self.data_dir = Path(data_dir)
        
        # Пути к файлам в родительской директории (для norm_coef)
        self.project_root = self.data_dir.parent.parent
        
    def _get_norm_coef_path(self) -> Path:
        """Возвращает путь к файлу коэффициентов нормализации."""
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
        n = len(signal)
        real = np.zeros(n, dtype=np.float32)
        imag = np.zeros(n, dtype=np.float32)
        
        if n < self.FFT_WINDOW:
            return real, imag
        
        # Используем sliding_window_view для эффективности
        windows = np.lib.stride_tricks.sliding_window_view(signal.astype(np.float32), self.FFT_WINDOW)
        
        # Применяем окно Ханнинга (как в pdr_calculator.sliding_window_fft)
        hanning_window = np.hanning(self.FFT_WINDOW).astype(np.float32)
        windows_windowed = windows * hanning_window
        
        # FFT
        fft_coeffs = np.fft.fft(windows_windowed, axis=-1) / self.FFT_WINDOW
        
        # Извлекаем первую гармонику и умножаем на 2
        first_harmonic = fft_coeffs[:, 1] * 2
        
        # Заполняем результат
        # sliding_window_view даёт n - FFT_WINDOW + 1 окон
        # Результат начинается с индекса FFT_WINDOW - 1 (последняя точка первого окна)
        n_windows = len(first_harmonic)
        end_idx = min(self.FFT_WINDOW - 1 + n_windows, n)
        actual_windows = end_idx - (self.FFT_WINDOW - 1)
        
        real[self.FFT_WINDOW - 1 : end_idx] = first_harmonic[:actual_windows].real.astype(np.float32)
        imag[self.FFT_WINDOW - 1 : end_idx] = first_harmonic[:actual_windows].imag.astype(np.float32)
        
        return real, imag
    
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
    
    def create_precomputed_test_csv(self, force: bool = False) -> Path:
        """
        Создает CSV файл с предрассчитанными признаками для тестовой выборки.
        
        Структура колонок:
        1. Метаданные: sample, file_name
        2. Raw аналоговые (8): IA, IB, IC, IN, UA, UB, UC, UN  
        3. Phase Polar (16): IA_mag, IA_angle, IB_mag, ..., UN_angle
        4. Symmetric (12): I1_mag, I1_angle, I2_mag, I2_angle, I0_mag, I0_angle,
                           U1_mag, U1_angle, U2_mag, U2_angle, U0_mag, U0_angle
        5. Phase Complex (16): IA_re, IA_im, IB_re, ..., UN_im
        6. Метки ML_*
        
        Первые 31 точка в расчётных колонках содержат 0 (warmup FFT).
        Валидные данные начинаются с точки 32 (индекс 31).
        
        Args:
            force: Принудительно пересоздать файл
            
        Returns:
            Путь к созданному файлу
        """
        output_path = self.data_dir / self.PRECOMPUTED_TEST_CSV
        
        if output_path.exists() and not force:
            print(f"[DatasetManager] Найден существующий файл: {output_path.name}")
            return output_path
        
        print(f"[DatasetManager] Создание предрассчитанного тестового датасета...")
        
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
        phase_polar_names = []
        phase_complex_names = []
        for ch in ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']:
            phase_polar_names.extend([f'{ch}_mag', f'{ch}_angle'])
            phase_complex_names.extend([f'{ch}_re', f'{ch}_im'])
            
        symmetric_names = []
        for comp in ['I1', 'I2', 'I0', 'U1', 'U2', 'U0']:
            symmetric_names.extend([f'{comp}_mag', f'{comp}_angle'])
        
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
            
            # Вычисляем фазоры для всех 8 каналов
            phasors_re = []
            phasors_im = []
            for i in range(8):
                re, im = self._compute_fft_phasors(raw_values[:, i])
                phasors_re.append(re)
                phasors_im.append(im)
            
            phasors_re = np.stack(phasors_re, axis=1)  # (Time, 8)
            phasors_im = np.stack(phasors_im, axis=1)  # (Time, 8)
            
            # Phase Polar: magnitude и ОТНОСИТЕЛЬНЫЙ angle для каждого канала
            # Опорный сигнал - UA (канал 4)
            complex_phasors = phasors_re + 1j * phasors_im
            ua_phasor = complex_phasors[:, 4]  # UA
            
            # Вычисляем опорный угол (используем UA, если он достаточно большой)
            ua_mag = np.abs(ua_phasor)
            ref_angle = np.zeros(n_points, dtype=np.float32)
            valid_mask = ua_mag > 1e-6
            ref_angle[valid_mask] = np.angle(ua_phasor[valid_mask])
            
            phase_polar = np.zeros((n_points, 16), dtype=np.float32)
            for i in range(8):
                phase_polar[:, 2*i] = np.abs(complex_phasors[:, i])
                # Относительный угол (от UA)
                raw_angle = np.angle(complex_phasors[:, i]) - ref_angle
                # Нормализация в [-pi, pi]
                phase_polar[:, 2*i + 1] = np.arctan2(np.sin(raw_angle), np.cos(raw_angle))
            
            # Phase Complex: Re и Im для каждого канала
            phase_complex = np.zeros((n_points, 16), dtype=np.float32)
            for i in range(8):
                phase_complex[:, 2*i] = phasors_re[:, i]
                phase_complex[:, 2*i + 1] = phasors_im[:, i]
            
            # Symmetric Components
            # Токи: каналы 0, 1, 2
            I_abc = complex_phasors[:, 0:3]
            I1, I2, I0 = self._compute_symmetric_components(I_abc[:, 0], I_abc[:, 1], I_abc[:, 2])
            
            # Напряжения: каналы 4, 5, 6
            U_abc = complex_phasors[:, 4:7]
            U1, U2, U0 = self._compute_symmetric_components(U_abc[:, 0], U_abc[:, 1], U_abc[:, 2])
            
            # Опорный угол для симметричных компонент - U1 (прямая последовательность напряжения)
            u1_mag = np.abs(U1)
            sym_ref_angle = np.zeros(n_points, dtype=np.float32)
            sym_valid_mask = u1_mag > 1e-6
            sym_ref_angle[sym_valid_mask] = np.angle(U1[sym_valid_mask])
            
            def relative_angle(phasor: np.ndarray, ref: np.ndarray) -> np.ndarray:
                """Вычисляет относительный угол в [-pi, pi]."""
                raw = np.angle(phasor) - ref
                return np.arctan2(np.sin(raw), np.cos(raw)).astype(np.float32)
            
            symmetric = np.zeros((n_points, 12), dtype=np.float32)
            symmetric[:, 0] = np.abs(I1)
            symmetric[:, 1] = relative_angle(I1, sym_ref_angle)
            symmetric[:, 2] = np.abs(I2)
            symmetric[:, 3] = relative_angle(I2, sym_ref_angle)
            symmetric[:, 4] = np.abs(I0)
            symmetric[:, 5] = relative_angle(I0, sym_ref_angle)
            symmetric[:, 6] = np.abs(U1)
            symmetric[:, 7] = relative_angle(U1, sym_ref_angle)  # Должен быть ~0
            symmetric[:, 8] = np.abs(U2)
            symmetric[:, 9] = relative_angle(U2, sym_ref_angle)
            symmetric[:, 10] = np.abs(U0)
            symmetric[:, 11] = relative_angle(U0, sym_ref_angle)
            
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
                # Phase Polar (16)
                for j, col in enumerate(phase_polar_names):
                    row[col] = float(phase_polar[idx, j])
                # Symmetric (12)
                for j, col in enumerate(symmetric_names):
                    row[col] = float(symmetric[idx, j])
                # Phase Complex (16)
                for j, col in enumerate(phase_complex_names):
                    row[col] = float(phase_complex[idx, j])
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
        print(f"  - Phase Polar (16): *_mag, *_angle")
        print(f"  - Symmetric (12): I1, I2, I0, U1, U2, U0 (mag, angle)")
        print(f"  - Phase Complex (16): *_re, *_im")
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
    
    def get_precomputed_feature_columns(self, feature_mode: str) -> List[str]:
        """
        Возвращает имена колонок для указанного режима признаков.
        
        Args:
            feature_mode: 'raw', 'phase_polar', 'symmetric', 'phase_complex'
            
        Returns:
            Список имён колонок
        """
        if feature_mode == 'raw':
            return ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']
        
        elif feature_mode == 'phase_polar':
            cols = []
            for ch in ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']:
                cols.extend([f'{ch}_mag', f'{ch}_angle'])
            return cols
        
        elif feature_mode in ['symmetric', 'symmetric_polar']:
            cols = []
            for comp in ['I1', 'I2', 'I0', 'U1', 'U2', 'U0']:
                cols.extend([f'{comp}_mag', f'{comp}_angle'])
            return cols
        
        elif feature_mode == 'phase_complex':
            cols = []
            for ch in ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']:
                cols.extend([f'{ch}_re', f'{ch}_im'])
            return cols
        
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
