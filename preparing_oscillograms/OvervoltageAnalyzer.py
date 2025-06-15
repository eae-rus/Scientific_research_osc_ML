import pandas as pd
import numpy as np
import os
import sys
import shutil
from tqdm import tqdm
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)
from normalization.normalization import NormOsc, CreateNormOsc
from dataflow.comtrade_processing import ReadComtrade
from raw_to_csv.raw_to_csv import RawToCSV 

class OvervoltageAnalyzer:
    """
    Класс для анализа осциллограмм на предмет максимальных перенапряжений при ОЗЗ.
    """
    def __init__(self, osc_folder_path: str, norm_coef_path: str, output_path: str, log_path: str):
        self.osc_folder_path = osc_folder_path
        self.norm_coef_path = norm_coef_path
        self.output_path = output_path
        self.log_path = log_path
        
        self.norm_coef_df = None
        self.results = []
        self.error_files = []
        
        # Загружаем инструменты
        self.readComtrade = ReadComtrade()
        self.rawToCSV = RawToCSV()
        self.normalizer = NormOsc(norm_coef_file_path=self.norm_coef_path)
        
        # Конфигурация
        # TODO: подумать над параметром номиналов
        # VALID_NOMINAL_VOLTAGES - пока что так сделано, чтобы отсеивать осциллограммы с глухозаземлённой нейтралью (400В, 110кВ и т.д.)
        # Но всё же это решение не очень, когда датасет разрастётся.
        self.VALID_NOMINAL_VOLTAGES = {0.221, 0.312, 0.33342, 100.0, 400.0, 6000.0, 10000.0, 20000.0, 35000.0}
        self.SPEF_THRESHOLD_U0 = 0.1/np.sqrt(3) / 3 # Порог для нормализованного U0
        self.SPEF_THRESHOLD_Un = 0.05 / 3 # 0.1/3 Порог для нормализованного Un
        self.SPEF_MIN_DURATION_PERIODS = 1
        self.MAX_BUS_COUNT = 10

    def _load_norm_coefficients(self):
        """Загружает единый файл с коэффициентами нормализации."""
        try:
            self.norm_coef_df = pd.read_csv(self.norm_coef_path)
            print(f"Файл коэффициентов нормализации '{self.norm_coef_path}' успешно загружен.")
        except FileNotFoundError:
            print(f"Ошибка: Файл коэффициентов нормализации не найден: {self.norm_coef_path}")
            self.norm_coef_df = None

    def _find_spef_zones(self, df: pd.DataFrame, group_prefix: str, samples_per_period: int) -> list:
        """
        Находит зоны ОЗЗ для группы сигналов (СШ или КЛ).
        Возвращает список кортежей (start_index, end_index).
        """
        u0_calculated = None
        un_measured = None

        # 1. Получаем сигналы U0
        # Пытаемся рассчитать из фазных
        phase_cols = [f'U{ph} {group_prefix}' for ph in ['A', 'B', 'C']]
        if all(col in df.columns for col in phase_cols):
            u0_calculated = (df[phase_cols[0]] + df[phase_cols[1]] + df[phase_cols[2]]) / 3
        
        # Пытаемся взять измеренный
        un_col = f'UN {group_prefix}'
        if un_col in df.columns:
            un_measured = df[un_col]

        if u0_calculated is None and un_measured is None:
            return []

        # 2. Создаем общую маску превышения порога
        combined_mask = pd.Series(False, index=df.index)
        if u0_calculated is not None:
            combined_mask |= (u0_calculated.abs() > self.SPEF_THRESHOLD_U0)
        if un_measured is not None:
            combined_mask |= (un_measured.abs() > self.SPEF_THRESHOLD_Un)

        if not combined_mask.any():
            return []
            
        # 3. Эффективный поиск непрерывных блоков нужной длины
        min_len = self.SPEF_MIN_DURATION_PERIODS * samples_per_period
        
        # Находим группы и их длины
        blocks = combined_mask.ne(combined_mask.shift()).cumsum()
        block_lengths = blocks.map(blocks.value_counts())
        
        # Оставляем только те блоки, которые 'True' и имеют достаточную длину
        valid_zones_mask = combined_mask & (block_lengths >= min_len)
        
        if not valid_zones_mask.any():
            return []
            
        # Находим начало и конец каждого истинного блока в итоговой маске
        zone_starts = valid_zones_mask.ne(valid_zones_mask.shift()) & valid_zones_mask
        zone_ends = valid_zones_mask.ne(valid_zones_mask.shift(-1)) & valid_zones_mask
        
        start_indices = df.index[zone_starts]
        end_indices = df.index[zone_ends]
        
        return list(zip(start_indices, end_indices))

    def _analyze_file(self, cfg_file_path: str):
        """Анализирует один файл осциллограммы."""
        filename_without_ext = os.path.basename(cfg_file_path)[:-4]
        
        # 1. Чтение и базовая подготовка
        raw_date, osc_df_raw = self.readComtrade.read_comtrade(cfg_file_path)
        if osc_df_raw is None or osc_df_raw.empty:
            return

        samples_per_period = int(raw_date.cfg.sample_rates[0][0] / raw_date.cfg.frequency)
        
        # Создаем копию для нормализации, чтобы не портить оригинал
        df_to_norm = osc_df_raw.copy()

        # 2. Нормализация (используем существующую функцию)
        # Важно: normalize_bus_signals модифицирует DataFrame на месте
        normalized_df = self.normalizer.normalize_bus_signals(df_to_norm, filename_without_ext, yes_prase="YES")
        if normalized_df is None:
            return # Нормализация не удалась или не разрешена

        # 3. Разделение на шины и переименование столбцов
        # Теперь на вход подается уже нормализованный DataFrame.
        # Эта функция вернет DataFrame в "длинном" формате с короткими именами столбцов (UA BB и т.д.)
        buses_df = self.rawToCSV.split_buses(normalized_df.reset_index(), os.path.basename(cfg_file_path))
        if buses_df.empty:
            return

        max_overvoltage_for_file = -1.0
        best_result_for_file = {}
        
        # Получаем строку с коэффициентами для этой осциллограммы один раз
        norm_row_series = self.norm_coef_df[self.norm_coef_df['name'] == filename_without_ext]
        if norm_row_series.empty:
            return # Нет коэффициентов для этого файла
        norm_row = norm_row_series.iloc[0]

        # 4. Перебор групп сигналов, созданных функцией split_buses
        for group_full_name, group_df in buses_df.groupby('file_name'):
            # Извлекаем номер секции и тип группы (bb/cl) из имени группы
            match = re.search(r'_bus (\d+)$', group_full_name.lower())
            if not match:
                continue # Если имя не соответствует ожидаемому формату, пропускаем
            
            bus_idx = int(match.group(1)) # Получаем номер секции (например, 1)

            # Теперь, внутри этой группы (для определенной секции, например, Bus 1),
            # мы обрабатываем отдельно СШ и КЛ сигналы, так как group_full_name не содержит этой информации.
            for group_name_inner, group_prefix_inner in [("СШ", "BB"), ("КЛ", "CL")]:
                group_name = group_name_inner
                group_prefix = group_prefix_inner
                
                # norm_base_col будет f"1Ub_base" или f"1Uc_base", что соответствует norm_coef.csv
                norm_base_col = f"{bus_idx}U{'b' if group_name == 'СШ' else 'c'}_base"

                if norm_base_col not in norm_row or pd.isna(norm_row[norm_base_col]):
                    continue
                
                try:
                    nominal_voltage = float(norm_row[norm_base_col])
                except (ValueError, TypeError):
                    continue

                if nominal_voltage not in self.VALID_NOMINAL_VOLTAGES:
                    continue
                
                # 5. Поиск зон ОЗЗ в текущей группе
                spef_zones = self._find_spef_zones(group_df, group_prefix, samples_per_period)
                if not spef_zones:
                    continue

                # 6. Поиск макс. перенапряжения в зонах
                phase_cols = [f'U{ph} {group_prefix}' for ph in ['A', 'B', 'C']]
                if not all(col in group_df.columns for col in phase_cols):
                    continue
                
                max_inst_val_in_zones = 0
                for start, end in spef_zones:
                    zone_df = group_df.loc[start:end, phase_cols]
                    max_in_zone = zone_df.abs().max().max()
                    if max_in_zone > max_inst_val_in_zones:
                        max_inst_val_in_zones = max_in_zone

                current_overvoltage = max_inst_val_in_zones / np.sqrt(2) / (1/(3*np.sqrt(3))) # Исходно при нормализации номинал завышается в 3 раза + мы обрабатываем фазные значения

                if current_overvoltage > max_overvoltage_for_file:
                    max_overvoltage_for_file = current_overvoltage
                    best_result_for_file = {
                        "filename": filename_without_ext,
                        "overvoltage": max_overvoltage_for_file,
                        "bus": bus_idx,
                        "group": group_name
                    }
        
        if best_result_for_file:
            self.results.append(best_result_for_file)

    def _save_results(self):
        """Сохраняет результаты в CSV и выводит статистику."""
        if not self.results:
            print("\nОсциллограммы с ОЗЗ и перенапряжениями не найдены.")
            return

        df = pd.DataFrame(self.results)
        
        bins = [0, 1.2, 1.71, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, np.inf]
        labels = [
            "< 1.2", "1.2 - 1.71", "1.71 - 1.75", "1.75 - 2.0", "2.0 - 2.5", "2.5 - 3.0", "3.0 - 3.5",
            "3.5 - 4.0", "4.0 - 4.5", "4.5 - 5.0", "> 5.0"
        ]
        
        df["overvoltage_group"] = pd.cut(df["overvoltage"], bins=bins, labels=labels, right=False)
        
        # Сортировка для красивого вывода
        df['overvoltage_group'] = pd.Categorical(df['overvoltage_group'], categories=labels, ordered=True)
        df.sort_values("overvoltage_group", inplace=True)
        
        # Форматируем итоговый CSV
        output_df = df[["overvoltage_group", "filename", "overvoltage", "bus", "group"]]
        output_df.to_csv(self.output_path, index=False, float_format='%.3f')
        
        # Сохраняем лог ошибок
        if self.error_files:
            with open(self.log_path, 'w') as f:
                for file, error in self.error_files:
                    f.write(f"{file};{error}\n")
            print(f"\nОбнаружены ошибки. Лог сохранен в: {self.log_path}")

        # Вывод статистики
        print("\n--- Сводная статистика ---")
        print(f"Всего осциллограмм обработано (попыток): {self.total_files_processed}")
        print(f"Найдено осциллограмм с ОЗЗ: {len(df)}")
        print("\nРаспределение по группам перенапряжений:")
        print(df["overvoltage_group"].value_counts().sort_index())
        print(f"\nРезультаты сохранены в: {self.output_path}")

    def run_analysis(self):
        """Основной метод для запуска полного цикла анализа."""
        self._load_norm_coefficients()
        if self.norm_coef_df is None:
            return

        osc_files = []
        for root, _, files in os.walk(self.osc_folder_path):
            for file in files:
                if file.lower().endswith('.cfg'):
                    osc_files.append(os.path.join(root, file))
        osc_files.sort() # Сортируем для стабильного порядка обхода
        self.total_files_processed = len(osc_files)

        if not osc_files:
            print(f"В папке '{self.osc_folder_path}' и её подпапках не найдено .cfg файлов.")
            return

        print(f"Начинается анализ {self.total_files_processed} осциллограмм...")
        for file_path in tqdm(osc_files, desc="Анализ перенапряжений"):
            try:
                self._analyze_file(file_path)
            except Exception as e:
                self.error_files.append((os.path.basename(file_path), str(e)))
        
        self._save_results()
        
    def copy_spef_oscillograms(report_csv_path: str, source_osc_folder_path: str, destination_folder_path: str):
        """
        Копирует файлы осциллограмм (.cfg и .dat) из исходной папки в целевую
        на основе списка файлов из CSV-отчета.

        Args:
            report_csv_path (str): Путь к CSV-файлу с отчетом (например, "overvoltage_report.csv"),
                                который содержит столбец "filename" с именами осциллограмм без расширения.
            source_osc_folder_path (str): Путь к корневой папке, где находятся исходные
                                        осциллограммы (поиск будет рекурсивным).
            destination_folder_path (str): Путь к папке, куда будут скопированы осциллограммы.
                                        Папка будет создана, если не существует.
        """
        print("--- Начало операции копирования осциллограмм ОЗЗ ---")
        # 1. Чтение CSV-файла с отчетом
        try:
            report_df = pd.read_csv(report_csv_path)
            print(f"Файл отчета '{report_csv_path}' успешно загружен.")
        except FileNotFoundError:
            print(f"Ошибка: Файл отчета '{report_csv_path}' не найден. Операция прервана.")
            return
        except Exception as e:
            print(f"Ошибка при чтении файла отчета '{report_csv_path}': {e}. Операция прервана.")
            return

        if 'filename' not in report_df.columns:
            print(f"Ошибка: В файле отчета '{report_csv_path}' отсутствует столбец 'filename'. Операция прервана.")
            return

        filenames_to_copy = report_df['filename'].unique().tolist()
        if not filenames_to_copy:
            print("В файле отчета нет имен файлов для копирования. Операция завершена.")
            return
        
        print(f"Обнаружено {len(filenames_to_copy)} уникальных имен осциллограмм для копирования.")

        # 2. Создание целевой папки
        try:
            os.makedirs(destination_folder_path, exist_ok=True)
            print(f"Целевая папка '{destination_folder_path}' готова.")
        except OSError as e:
            print(f"Ошибка при создании целевой папки '{destination_folder_path}': {e}. Операция прервана.")
            return

        # 3. Сканирование исходной папки и создание карты файлов
        # {filename_without_ext: {'cfg': cfg_path, 'dat': dat_path}}
        source_file_map = {} 
        print(f"Сканирование исходной папки '{source_osc_folder_path}' для поиска файлов осциллограмм...")
        
        # Подсчет общего количества файлов для tqdm (приблизительно)
        total_files_to_scan = sum([len(files) for r, d, files in os.walk(source_osc_folder_path)])
        
        with tqdm(total=total_files_to_scan, desc="Сканирование исходных файлов", unit="файл") as pbar_scan:
            for root, _, files in os.walk(source_osc_folder_path):
                for file in files:
                    pbar_scan.update(1)
                    name, ext = os.path.splitext(file)
                    ext_lower = ext.lower()
                    
                    if ext_lower == '.cfg':
                        if name not in source_file_map:
                            source_file_map[name] = {}
                        source_file_map[name]['cfg'] = os.path.join(root, file)
                    elif ext_lower == '.dat':
                        if name not in source_file_map:
                            source_file_map[name] = {}
                        source_file_map[name]['dat'] = os.path.join(root, file)
        
        print(f"Сканирование завершено. Найдено {len(source_file_map)} уникальных имен осциллограмм (пар .cfg/.dat или одиночных файлов) в исходной папке.")

        # 4. Копирование файлов
        copied_count = 0
        errors_during_copy = 0
        skipped_missing_in_source_map = 0
        skipped_missing_cfg_in_map = 0
        skipped_missing_dat_in_map = 0

        print(f"Начинается копирование {len(filenames_to_copy)} выбранных осциллограмм...")
        for filename_no_ext in tqdm(filenames_to_copy, desc="Копирование осциллограмм", unit="осц."):
            if filename_no_ext in source_file_map:
                source_paths = source_file_map[filename_no_ext]
                
                cfg_source_path = source_paths.get('cfg')
                dat_source_path = source_paths.get('dat')

                if not cfg_source_path:
                    # print(f"Предупреждение: .cfg файл для '{filename_no_ext}' не найден в карте исходных файлов. Пропуск.")
                    skipped_missing_cfg_in_map += 1
                    continue
                if not dat_source_path:
                    # print(f"Предупреждение: .dat файл для '{filename_no_ext}' не найден в карте исходных файлов. Пропуск.")
                    skipped_missing_dat_in_map += 1
                    continue

                # Формируем целевые пути, сохраняя исходные имена файлов
                cfg_dest_path = os.path.join(destination_folder_path, os.path.basename(cfg_source_path))
                dat_dest_path = os.path.join(destination_folder_path, os.path.basename(dat_source_path))

                try:
                    shutil.copy2(cfg_source_path, cfg_dest_path)
                    shutil.copy2(dat_source_path, dat_dest_path)
                    copied_count += 1
                except FileNotFoundError:
                    # Эта ошибка более вероятна, если путь в source_file_map некорректен,
                    # или файл был удален между сканированием и копированием.
                    print(f"Ошибка FileNotFoundError: Один из файлов для '{filename_no_ext}' не найден по пути при копировании. Пропуск.")
                    errors_during_copy += 1
                except Exception as e:
                    print(f"Ошибка при копировании файлов для '{filename_no_ext}': {e}. Пропуск.")
                    errors_during_copy += 1
            else:
                # print(f"Предупреждение: Осциллограмма '{filename_no_ext}' из отчета не найдена среди просканированных файлов в '{source_osc_folder_path}'. Пропуск.")
                skipped_missing_in_source_map += 1
                
        # 5. Вывод статистики
        print("\n--- Статистика копирования ---")
        print(f"Всего уникальных имен осциллограмм в отчете: {len(filenames_to_copy)}")
        print(f"Успешно скопировано пар файлов (cfg+dat): {copied_count}")
        print(f"Пропущено (осциллограмма из отчета не найдена в исходной папке при сканировании): {skipped_missing_in_source_map}")
        print(f"Пропущено (в карте исходных файлов отсутствовал .cfg для имени из отчета): {skipped_missing_cfg_in_map}")
        print(f"Пропущено (в карте исходных файлов отсутствовал .dat для имени из отчета): {skipped_missing_dat_in_map}")
        print(f"Ошибок во время фактического копирования (файлы были в карте, но не скопировались): {errors_during_copy}")
        print(f"Итоговые файлы сохранены в: '{destination_folder_path}'")
        print("--- Завершение операции копирования ---")
