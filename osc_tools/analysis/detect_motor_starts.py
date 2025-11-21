import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(ROOT_DIR)

import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm
import re


from osc_tools.data_management.comtrade_processing import ReadComtrade

class MotorStartDetector:
    """
    Класс для анализа осциллограмм и выявления пусков двигателей
    по заданным критериям (факторам).
    """
    def __init__(self, osc_folder_path: str, norm_coef_path: str, output_path: str, log_path: str):
        """
        Инициализация детектора.

        Args:
            osc_folder_path (str): Путь к папке с исходными осциллограммами.
            norm_coef_path (str): Путь к CSV файлу с коэффициентами нормализации (для Iном).
            output_path (str): Путь к папке, куда будут сохраняться отсортированные осциллограммы.
            log_path (str): Путь к файлу для логирования ошибок.
        """
        self.osc_folder_path = osc_folder_path
        self.norm_coef_path = norm_coef_path
        self.output_path = output_path
        self.log_path = log_path

        # --- Гиперпараметры для алгоритмов детекции ---
        # Фактор 1 и 4: Порог "отсутствия" тока (шум)
        self.NOISE_THRESHOLD_PU = 0.01
        # Фактор 2 и 3: Порог начального тока для нагруженной секции
        self.LOADED_BUS_THRESHOLD_PU = 0.2
        # Фактор 1 и 3: Длительность "плато" пускового тока в мс
        self.PLATEAU_DURATION_MS = 300
        # Фактор 1, 2, 3: Коэффициент допустимого снижения тока на "плато"
        self.PLATEAU_DROP_RATIO = 1.5
        # Фактор 1, 2, 4: Коэффициент снижения тока до установившегося режима
        self.FINAL_DROP_RATIO = 3.0
        # Фактор 2: Коэффициент, определяющий "бросок" тока
        self.INRUSH_JUMP_RATIO = 3.0
        # --- Конец гиперпараметров ---

        self.norm_coef_df = None
        self.results = []
        self.error_files = []
        self.processed_files = set() # Множество для хранения имен уже классифицированных файлов

        # Загружаем инструменты
        self.readComtrade = ReadComtrade()
        # Создаем базовую выходную папку
        os.makedirs(self.output_path, exist_ok=True)
        
        # Кэш исходных файлов (маппинг: имя -> {'.cfg': path, '.dat': path})
        self._source_file_map = None

    def _load_norm_coefficients(self):
        """Загружает единый файл с коэффициентами нормализации."""
        try:
            self.norm_coef_df = pd.read_csv(self.norm_coef_path)
            print(f"Файл коэффициентов нормализации '{self.norm_coef_path}' успешно загружен.")
        except FileNotFoundError:
            print(f"Ошибка: Файл коэффициентов нормализации не найден: {self.norm_coef_path}")
            sys.exit("Критическая ошибка: отсутствует файл с номинальными токами.")

    def _build_source_file_map(self):
        """Кэширует маппинг исходных файлов для быстрого поиска."""
        if self._source_file_map is not None:
            return  # Уже построена
        
        self._source_file_map = {}
        for root, _, files in os.walk(self.osc_folder_path):
            for file in files:
                name, ext = os.path.splitext(file)
                if name not in self._source_file_map:
                    self._source_file_map[name] = {}
                if ext.lower() in ['.cfg', '.dat']:
                    self._source_file_map[name][ext.lower()] = os.path.join(root, file)

    def _copy_file_to_factor_folder(self, filename: str, factor: int, channel_name: str) -> None:
        """
        Копирует файлы осциллограммы в папку соответствующего фактора и логирует результат.
        Вызывается сразу после обнаружения матча, для incremental сохранения.
        """
        # Если маппинг ещё не построен, строим
        if self._source_file_map is None:
            self._build_source_file_map()
        
        # Добавляем запись в результаты
        self.results.append({'filename': filename, 'factor': factor, 'channel': channel_name})
        
        # Создаём папку для фактора
        dest_folder = os.path.join(self.output_path, f"factor_{factor}")
        os.makedirs(dest_folder, exist_ok=True)
        
        # Копируем файлы
        if filename in self._source_file_map:
            file_pair = self._source_file_map[filename]
            try:
                if '.cfg' in file_pair:
                    shutil.copy2(file_pair['.cfg'], dest_folder)
                if '.dat' in file_pair:
                    shutil.copy2(file_pair['.dat'], dest_folder)
            except Exception as e:
                error_msg = f"Ошибка копирования {filename}: {e}"
                print(error_msg)
                self.error_files.append((filename, error_msg))
        else:
            error_msg = f"Не найдены исходные файлы для {filename}"
            print(error_msg)
            self.error_files.append((filename, error_msg))

    def _get_nominal_current(self, filename: str, signal_name: str) -> float:
        """
        Извлекает номинальный ток для заданного сигнала из файла коэффициентов.
        Пример: для сигнала 'IA 1' ищет столбец '1Ib_base' в norm_coef_df.
        """
        # Попробуем извлечь номер секции по шаблону 'Bus-<n>' из имени сигнала,
        # например: 'I | Bus-1 | phase: A' -> 1, 'I | Bus-20 | phase: A' -> 20.
        match = re.search(r"Bus-(\d+)", signal_name)
        if not match:
            # fallback: последняя встреченная цифра в строке (как раньше), но
            # предпочтение отдаём явному шаблону 'Bus-'
            match = re.search(r"\b(\d+)\b", signal_name)
            if not match:
                return None

        bus_idx = match.group(1)
        nominal_current_col = f"{bus_idx}Ip_base"

        # На некоторых CSV-ах 'name' может быть не строкой, поэтому приводим к str для надежности
        if self.norm_coef_df is None or 'name' not in self.norm_coef_df.columns:
            return None

        norm_row = self.norm_coef_df[self.norm_coef_df['name'].astype(str) == str(filename)]
        if norm_row.empty:
            return None

        # Используем .get как в других частях проекта — это возвращает Series или None
        nominal_series = norm_row.get(nominal_current_col)
        if nominal_series is None or pd.isna(nominal_series.values[0]):
            return None

        # В проекте в нескольких местах реальный номинал строится как 20 * xIp_base
        # (см. normalization.normalize_bus_signals и marking_up_oscillograms).
        try:
            i_nom = float(nominal_series.values[0])
        except Exception:
            return None

        return i_nom if pd.notna(i_nom) and i_nom > 0 else None

    @staticmethod
    def _calculate_first_harmonic(signal: pd.Series, samples_per_period: int) -> np.ndarray:
        """
        Рассчитывает амплитуду первой гармоники для сигнала с использованием скользящего окна.
        """
        if not isinstance(signal, np.ndarray):
            signal = signal.to_numpy()
            
        num_points = len(signal)
        amplitudes = np.zeros(num_points)
        
        # Используем rfft для реальных сигналов (быстрее)
        # Амплитуда = 2/N * |X_k|, где k=1 для первой гармоники
        normalization_factor = 2.0 / samples_per_period

        for i in range(num_points):
            start = max(0, i - samples_per_period + 1)
            window = signal[start : i + 1]
            
            # Дополняем окно нулями, если оно короче периода (в начале осциллограммы)
            if len(window) < samples_per_period:
                window = np.pad(window, (samples_per_period - len(window), 0), 'constant')

            fft_coeffs = np.fft.rfft(window)
            
            # Индекс первой гармоники в результате rfft всегда 1
            if len(fft_coeffs) > 1:
                amplitudes[i] = np.abs(fft_coeffs[1]) * normalization_factor
            
        return amplitudes

    # --------------------------------------------------------------------------
    # Функции-детекторы для каждого фактора
    # --------------------------------------------------------------------------

    def _check_factor_1(self, i1_rms_pu: np.ndarray, plateau_len_samples: int):
        """
        Математическая логика отбора:
        1. Проверяем, что в начале (первый период) ток практически отсутствует (ниже порога шума).
        2. Находим максимальное значение (амплитуду) тока за всю осциллограмму.
        3. Убеждаемся, что после точки максимума ток держится на высоком уровне ("плато")
           в течение заданного времени, не падая ниже, чем в `PLATEAU_DROP_RATIO` раз.
        4. Проверяем, что в конце осциллограммы ток значительно снизился (успешный выход на
           номинальный режим), упав ниже, чем в `FINAL_DROP_RATIO` раз от максимума.
        Это соответствует пуску двигателя на "пустой" шине.
        """
        samples_per_period = int(plateau_len_samples * 50 // (self.PLATEAU_DURATION_MS / 20)) # Приближенно

        # 1. Ток в начале отсутствует
        if np.mean(i1_rms_pu[:samples_per_period]) > self.NOISE_THRESHOLD_PU:
            return False

        # 2. Находим бросок тока
        max_current_pu = np.max(i1_rms_pu)
        max_idx = np.argmax(i1_rms_pu)

        # Пусковой ток должен быть значительным
        if max_current_pu < self.LOADED_BUS_THRESHOLD_PU:
            return False

        # 3. Проверяем плато после максимума
        plateau_end_idx = min(max_idx + plateau_len_samples, len(i1_rms_pu))
        if plateau_end_idx <= max_idx: # Если осциллограмма слишком коротка
             return False
        
        plateau_zone = i1_rms_pu[max_idx:plateau_end_idx]
        if np.min(plateau_zone) < max_current_pu / self.PLATEAU_DROP_RATIO:
            return False

        # 4. Проверяем падение тока в конце
        # Берем последнюю 1/5 часть осциллограммы для анализа установившегося режима
        end_zone_start_idx = int(len(i1_rms_pu) * 0.8)
        if np.mean(i1_rms_pu[end_zone_start_idx:]) > max_current_pu / self.FINAL_DROP_RATIO:
            return False

        return True

    def _check_factor_2(self, i1_rms_pu: np.ndarray, plateau_len_samples: int):
        """
        Математическая логика отбора:
        1. Проверяем, что в начале есть ненулевой, но не очень большой ток (существующая нагрузка).
        2. Находим максимум тока и убеждаемся, что он значительно (в `INRUSH_JUMP_RATIO` раз)
           превышает начальный ток. Это идентифицирует именно "бросок", а не просто колебания.
        3. Логика проверки "плато" и "конечного падения" аналогична Фактору 1.
        Это соответствует пуску двигателя на уже нагруженную секцию.
        """
        samples_per_period = int(plateau_len_samples * 50 // (self.PLATEAU_DURATION_MS / 20))
        
        # 1. Есть начальный ток, но не слишком большой
        initial_current_pu = np.mean(i1_rms_pu[:samples_per_period])
        if not (self.NOISE_THRESHOLD_PU < initial_current_pu < self.LOADED_BUS_THRESHOLD_PU):
            return False

        # 2. Происходит значительный бросок тока
        max_current_pu = np.max(i1_rms_pu)
        max_idx = np.argmax(i1_rms_pu)
        if max_current_pu < initial_current_pu * self.INRUSH_JUMP_RATIO:
            return False

        # 3. Плато
        plateau_end_idx = min(max_idx + plateau_len_samples, len(i1_rms_pu))
        if plateau_end_idx <= max_idx:
             return False
        plateau_zone = i1_rms_pu[max_idx:plateau_end_idx]
        if np.min(plateau_zone) < max_current_pu / self.PLATEAU_DROP_RATIO:
            return False

        # 4. Падение в конце
        end_zone_start_idx = int(len(i1_rms_pu) * 0.8)
        if np.mean(i1_rms_pu[end_zone_start_idx:]) > max_current_pu / self.FINAL_DROP_RATIO:
            return False

        return True

    def _check_factor_3(self, i1_rms_pu: np.ndarray, plateau_len_samples: int):
        """
        Математическая логика отбора:
        1. Начальные условия по току и логика определения броска аналогичны Фактору 2.
        2. Проверка "плато" также аналогична.
        3. Ключевое отличие: проверяем, что в конце осциллограммы ток НЕ УСПЕЛ снизиться,
           то есть остался на высоком уровне (не ниже, чем в `PLATEAU_DROP_RATIO` раз от максимума).
        Это позволяет сохранить "короткие" осциллограммы, где зафиксирован только пуск.
        """
        samples_per_period = int(plateau_len_samples * 50 // (self.PLATEAU_DURATION_MS / 20))
        
        # 1. Начальный ток (может быть и нулевым, и небольшим)
        initial_current_pu = np.mean(i1_rms_pu[:samples_per_period])
        if initial_current_pu > self.LOADED_BUS_THRESHOLD_PU:
            return False
            
        # 2. Бросок тока
        max_current_pu = np.max(i1_rms_pu)
        max_idx = np.argmax(i1_rms_pu)
        
        # Пусковой ток должен быть значительным
        if max_current_pu < self.LOADED_BUS_THRESHOLD_PU * self.INRUSH_JUMP_RATIO:
            return False
        
        if initial_current_pu > self.NOISE_THRESHOLD_PU and max_current_pu < initial_current_pu * self.INRUSH_JUMP_RATIO:
            return False # Если был начальный ток, скачок должен быть значительным
            
        # 3. Плато
        plateau_end_idx = min(max_idx + plateau_len_samples, len(i1_rms_pu))
        if plateau_end_idx <= max_idx:
             return False
        plateau_zone = i1_rms_pu[max_idx:plateau_end_idx]
        if np.min(plateau_zone) < max_current_pu / self.PLATEAU_DROP_RATIO:
            return False
            
        # 4. В конце ток НЕ упал
        end_zone_start_idx = int(len(i1_rms_pu) * 0.8)
        if np.mean(i1_rms_pu[end_zone_start_idx:]) < max_current_pu / self.PLATEAU_DROP_RATIO:
            return False

        return True

    def _check_factor_4(self, i1_rms_pu: np.ndarray, plateau_len_samples: int):
        """
        Математическая логика отбора:
        1. Проверяем, что осциллограмма начинается сразу с высокого тока (выше `LOADED_BUS_THRESHOLD_PU`).
        2. Убеждаемся, что этот высокий ток держится в течение заданного времени (`PLATEAU_DURATION_MS`)
           с самого начала осциллограммы.
        3. Проверяем, что после этого начального плато ток значительно снижается к концу.
        4. Дополнительно проверяем, что ток никогда не падает до уровня шума, т.е. нагрузка не отключалась.
        Это соответствует "продолжению" пуска, когда запись началась в середине процесса.
        """
        # 1. В начале осциллограммы высокий ток
        initial_current_pu = np.mean(i1_rms_pu[:plateau_len_samples])
        if initial_current_pu < self.LOADED_BUS_THRESHOLD_PU:
            return False

        # 2. Этот ток держится (начальное плато)
        plateau_zone = i1_rms_pu[:plateau_len_samples]
        if np.min(plateau_zone) < initial_current_pu / self.PLATEAU_DROP_RATIO:
            return False
            
        # 3. Затем ток падает
        end_zone_start_idx = int(len(i1_rms_pu) * 0.8)
        if np.mean(i1_rms_pu[end_zone_start_idx:]) > initial_current_pu / self.FINAL_DROP_RATIO:
            return False
            
        # 4. Ток никогда не опускается до нуля
        if np.min(i1_rms_pu) < self.NOISE_THRESHOLD_PU:
            return False

        return True


    def _analyze_file(self, cfg_file_path: str):
        """Анализирует один файл осциллограммы и копирует его сразу при обнаружении матча."""
        filename_without_ext = os.path.basename(cfg_file_path)[:-4]

        # Пропускаем уже обработанные файлы (важно, если один файл копируется по фактору 1,
        # чтобы он не проверялся для других факторов)
        if filename_without_ext in self.processed_files:
            return

        try:
            raw_date, osc_df_raw = self.readComtrade.read_comtrade(cfg_file_path)
            if osc_df_raw is None or osc_df_raw.empty:
                self.error_files.append((filename_without_ext, "Не удалось прочитать или файл пуст"))
                return
        except Exception as e:
            self.error_files.append((filename_without_ext, f"Ошибка чтения Comtrade: {e}"))
            return

        frequency = raw_date.cfg.frequency
        sample_rate = raw_date.cfg.sample_rates[0][0]
        samples_per_period = int(sample_rate / frequency)
        if samples_per_period < 6:
            # Защита от несинусоидальных данных
            return
        plateau_len_samples = int(self.PLATEAU_DURATION_MS / 1000 * sample_rate)

        # Ищем только токовые каналы (обычно начинаются с 'I')
        current_channels = [col for col in osc_df_raw.columns if col.upper().strip().startswith('I ')]

        for channel_name in current_channels:
            i_nom = self._get_nominal_current(filename_without_ext, channel_name)
            if i_nom is None:
                continue # Не нашли номинальный ток для этого канала, пропускаем

            # Расчет первой гармоники
            i1_amplitude = self._calculate_first_harmonic(osc_df_raw[channel_name], samples_per_period)
            
            # Переводим в относительные единицы (p.u.)
            # Номинальный ток - это RMS, а мы получили амплитуду, поэтому делим на sqrt(2)
            i1_rms_pu = i1_amplitude / (i_nom * np.sqrt(2))

            # Последовательно проверяем факторы. Если один сработал,
            # осциллограмма классифицируется, копируется и мы переходим к следующему файлу.
            
            # Проверка Фактора 1
            if self._check_factor_1(i1_rms_pu, plateau_len_samples):
                self._copy_file_to_factor_folder(filename_without_ext, 1, channel_name)
                self.processed_files.add(filename_without_ext)
                return # Выходим из функции, т.к. файл классифицирован и скопирован
            
            # Проверка Фактора 2
            if self._check_factor_2(i1_rms_pu, plateau_len_samples):
                self._copy_file_to_factor_folder(filename_without_ext, 2, channel_name)
                self.processed_files.add(filename_without_ext)
                return
            
            # Проверка Фактора 3
            if self._check_factor_3(i1_rms_pu, plateau_len_samples):
                self._copy_file_to_factor_folder(filename_without_ext, 3, channel_name)
                self.processed_files.add(filename_without_ext)
                return
            
            # Проверка Фактора 4
            if self._check_factor_4(i1_rms_pu, plateau_len_samples):
                self._copy_file_to_factor_folder(filename_without_ext, 4, channel_name)
                self.processed_files.add(filename_without_ext)
                return


    def _finalize_report_and_log(self):
        """Сохраняет итоговый отчет и логирует ошибки (файлы уже скопированы ранее)."""
        if not self.results:
            print("\nОсциллограммы, соответствующие критериям пуска двигателя, не найдены.")
            return

        report_df = pd.DataFrame(self.results)
        report_path = os.path.join(self.output_path, "motor_start_report.csv")
        report_df.to_csv(report_path, index=False)
        
        print(f"\n--- Итоговый отчет сохранен в: {report_path} ---")
        print("Распределение найденных осциллограмм по факторам:")
        print(report_df['factor'].value_counts().sort_index())

        if self.error_files:
            with open(self.log_path, 'w', encoding='utf-8') as f:
                for file, error in self.error_files:
                    f.write(f"{file};{error}\n")
            print(f"\nОбнаружены ошибки во время работы. Лог сохранен в: {self.log_path}")

    def run_detection(self):
        """Основной метод для запуска полного цикла анализа."""
        self._load_norm_coefficients()

        osc_files = []
        for root, _, files in os.walk(self.osc_folder_path):
            for file in files:
                if file.lower().endswith('.cfg'):
                    osc_files.append(os.path.join(root, file))
        
        if not osc_files:
            print(f"В папке '{self.osc_folder_path}' и её подпапках не найдено .cfg файлов.")
            return

        print(f"Начинается анализ {len(osc_files)} осциллограмм...")
        for file_path in tqdm(osc_files, desc="Детекция пусков двигателей"):
            self._analyze_file(file_path)
        
        self._finalize_report_and_log()


# --- Пример использования ---
if __name__ == '__main__':
    # Укажите ваши пути здесь
    # ПУТЬ К ПАПКЕ С ИСХОДНЫМИ ОСЦИЛЛОГРАММАМИ
    OSC_FOLDER = "D:\\DataSet\\__Open EE osc Dataset v1.2 — копия\\osc_comtrade\\f_network = 50\\f_ADC = 1600 v1.4"
    OSC_FOLDER = "D:\\DataSet\\__Open EE osc Dataset v1.2 — копия\\osc_comtrade"
    # ПУТЬ К ФАЙЛУ С НОМИНАЛЬНЫМИ ЗНАЧЕНИЯМИ (Iном)
    NORM_COEF_FILE = "D:\\DataSet\\__Open EE osc Dataset v1.2 — копия\\norm_coef_all_v1.4.csv"
    # ПУТЬ К ПАПКЕ, КУДА БУДУТ СОХРАНЕНЫ РЕЗУЛЬТАТЫ
    OUTPUT_FOLDER = "D:\\DataSet\\_detect_motor_starts_v1.0"
    # ПУТЬ К ЛОГ-ФАЙЛУ
    LOG_FILE = os.path.join(OUTPUT_FOLDER, "errors.log")

    # Проверка, что пути существуют (для примера)
    if not os.path.isdir(OSC_FOLDER) or not os.path.isfile(NORM_COEF_FILE):
        print("="*50)
        print("ВНИМАНИЕ: Указаны некорректные пути в секции `if __name__ == '__main__':`")
        print(f"Проверьте, что папка '{OSC_FOLDER}' и файл '{NORM_COEF_FILE}' существуют.")
        print("Работа скрипта прервана. Отредактируйте пути и запустите снова.")
        print("="*50)
    else:
        detector = MotorStartDetector(
            osc_folder_path=OSC_FOLDER,
            norm_coef_path=NORM_COEF_FILE,
            output_path=OUTPUT_FOLDER,
            log_path=LOG_FILE
        )
        detector.run_detection()