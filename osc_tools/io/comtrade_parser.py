import polars as pl
import numpy as np
import os
import sys
import json
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)

from osc_tools.features.normalization import NormOsc
from osc_tools.data_management.comtrade_processing import ReadComtrade


class ComtradeParser():
    """
    Этот класс реализован для преобразования необработанных файлов comtrade в файл csv.
    """
    def __init__(self, raw_path='raw_data/', csv_path='', uses_buses = ['1', '2', '12']):
        if not os.path.exists(raw_path):
            raise FileNotFoundError("Путь для необработанных файлов не существует")
         
        # FIXME: Вам нужно добавить обход проблемы, когда эти файлы отсутствуют. Потому что их нужно откуда-то брать.
        with open("dict_analog_names.json", "r") as file:
            analog_names = json.load(file)
        with open("dict_discrete_names.json", "r") as file:
            discrete_names = json.load(file)
        self.analog_names = analog_names
        self.discrete_names = discrete_names

        self.analog_names_dict = self.get_bus_names(analog=True, discrete=False)
        self.discrete_names_dict = self.get_bus_names(analog=False, discrete=True)
        self.all_names = self.get_all_names()
        self.raw_path = raw_path
        self.csv_path = csv_path
        self.readComtrade = ReadComtrade()
        self.unread_files = set()
        self.uses_buses = uses_buses # 12 - межсекционный, никак не учитывается, при его добавлении требуется скорректировать проверку дискреции
        # пока все сохранено, подумайте, как их удобнее задавать
        # TODO: по используемым переменным - добавить усечение сигналов из массивов.
        self.uses_CT_B, self.uses_CT_zero = True, True
        self.uses_VT_ph, self.uses_VT_iph, self.uses_VT_zero  = True, True, True
        self.use_VT_CL, self.use_VT_BB = True, True
        # TODO: подумать о большей понятности и оптимальности, так как use_PDR нужна только сейчас, а требует проверки всех дискрет
        self.use_PDR = True
        # TODO: Добавить переменные для объединения уровней аварий (сигналы ML)
        self.number_periods = 10 # TODO: Количество выборок устанавливается сейчас. Подумайте о задаче с привязкой ко времени или чем-то подобном.
        self.ml_all, self.ml_opr_swch, self.ml_abnorm_evnt, self.ml_emerg_evnt  = self.get_short_names_ml_signals()

    def create_csv(self, csv_name='dataset.csv', is_cut_out_area = False, is_simple_csv = False):
        """
        Эта функция DataFrame и сохраняет csv-файл из необработанных данных comtrade.

        Args:
            csv_name (str): Имя файла csv.
        """
        dataset_dfs = []
        raw_files = sorted([file for file in os.listdir(self.raw_path)
                            if 'cfg' in file])
        with tqdm(total=len(raw_files), desc="Преобразование Comtrade в CSV") as pbar:
            for file in raw_files:
                # TODO: нерационально использовать две переменные, а необходимые глобальные данные.
                # необходимо будет подумать об оптимизации этого вопроса.
                raw_date, raw_df = self.readComtrade.read_comtrade(self.raw_path + file)
                
                # TODO: Добавить нормировку значений по аналогии с PDR и SPEF
                # Но с параметром выбора (чтобы можно было её и не осуществлять)
                
                # TODO: Добавить проверки вторичных/первичных данных при чтении и обработке набора данных.
                self.check_columns(raw_df)
                if not raw_df.is_empty():
                    # raw_df = raw_df.reset_index() # Not needed in Polars
                    buses_df = self.split_buses(raw_df, file)

                    frequency = raw_date.cfg.frequency
                    samples_rate = raw_date.cfg.sample_rates[0][0]
                    number_samples = int(samples_rate / frequency) # TODO: Это не всегда будет целое число, но это редкость.
                    samples_before, samples_after = number_samples * self.number_periods, number_samples * self.number_periods
                    if is_cut_out_area:
                        buses_df = self.cut_out_area(buses_df, samples_before, samples_after)
                        dataset_dfs.append(buses_df)
                    else:
                        dataset_dfs.append(buses_df)
                    pbar.update(1)
        
        if dataset_dfs:
            dataset_df = pl.concat(dataset_dfs, how="diagonal")
        else:
            dataset_df = pl.DataFrame()
        
        # TODO: Подумайте об организации самой обработки по-другому, а не о добавлении функции.
        dataset_df = self.structure_columns(dataset_df)
        if is_simple_csv:
            # Мы не перезаписываем исходные данные, а только создаем еще один csv-файл
            self.get_simple_dataset(dataset_df.clone())
        
        dataset_df.write_csv(self.csv_path + csv_name)
        return dataset_df
    
    # TODO: подумать об универсанолизации данной функции с create_csv (основные замечания указаны там)
    def create_csv_for_PDR(self, csv_name='dataset.csv', signal_check_results_path='signal_check_results.csv', norm_coef_file_path='norm_coef.csv',
                           is_check_PDR = True, is_cut_out_area = False, yes_prase = "YES", is_print_error = False):
        """
        Эта функция DataFrame и сохраняет csv-файл из необработанных данных comtrade.

        Args:
            csv_name (str): Имя файла csv.
        """
        signal_check_df = pl.DataFrame()
        if (is_check_PDR and (not os.path.exists(signal_check_results_path))):
            raise FileNotFoundError("Путь для результатов проверки сигнала не существует")
            # TODO: Пока что это локальный файл, но надо подумать о том, чтобы он был глобальный, но пока может мешать.
        if is_check_PDR:
            signal_check_df = pl.read_csv(signal_check_results_path) # Загрузка файла проверки сигналов
        
        dataset_dfs = []
        raw_files = sorted([file for file in os.listdir(self.raw_path)
                            if 'cfg' in file])
        number_ocs_found = 0
        normOsc = NormOsc(norm_coef_file_path = norm_coef_file_path)
        with tqdm(total=len(raw_files), desc="Преобразование Comtrade в CSV") as pbar:
            for file in raw_files:
                filename_without_ext = file[:-4] # Имя файла без расширения
                # FIXME: На будущее - возможно этот файл и лишний, ибо проверки заложил в "_process_signals_for_PDR"
                if is_check_PDR:
                    check_result_row = signal_check_df.filter(pl.col('filename') == filename_without_ext)
                    if check_result_row.is_empty() or not (str(check_result_row['contains_required_signals'][0]).upper()  == yes_prase):
                        pbar.update(1) # Пропускаем файл, если его нет в signal_check_results_csv или contains_required_signals False
                        continue
                
                raw_date, raw_df = self.readComtrade.read_comtrade(self.raw_path + file)
                raw_df = normOsc.normalize_bus_signals(raw_df, filename_without_ext, yes_prase="YES", is_print_error=False)
                if raw_df is None:
                    pbar.update(1)
                    if is_print_error:
                        print(f"Предупреждение: В {filename_without_ext} нормализацию провести нельзя, значения не используем.")
                    continue
                    
                if not raw_df.is_empty():
                    buses_df = self.split_buses_for_PDR(raw_df, file)
                    
                    if buses_df is None:
                        if is_print_error:
                            print(f"Предупреждение: В {filename_without_ext} ошибка обработки осциллограммы.")
                        pbar.update(1)
                        continue
                    
                    frequency = raw_date.cfg.frequency
                    samples_rate = raw_date.cfg.sample_rates[0][0]
                    number_samples = int(samples_rate / frequency) # TODO: Это не всегда будет целое число, но это редкость.
                    samples_before, samples_after = number_samples * self.number_periods, number_samples * self.number_periods
                    if is_cut_out_area and is_check_PDR:
                        buses_df = self.cut_out_area_for_PDR(buses_df, samples_before, samples_after)
                        dataset_dfs.append(buses_df)
                    else:
                        dataset_dfs.append(buses_df)
                    number_ocs_found += 1
                    pbar.update(1)
        
        print(f"Количество найденных образцов = {number_ocs_found}")
        if dataset_dfs:
            dataset_df = pl.concat(dataset_dfs, how="diagonal")
        else:
            dataset_df = pl.DataFrame()
        dataset_df.write_csv(self.csv_path + csv_name)
        return dataset_df
    
    # TODO: подумать об универсанолизации данной функции с create_csv (основные замечания указаны там)
    def create_csv_for_SPEF(self, csv_name='dataset_spef.csv', signal_check_results_path='find_oscillograms_with_spef.csv', norm_coef_file_path='norm_coef.csv', is_cut_out_area = False, is_print_error = False):
        """
        Эта функция находит осциллограммы SPEF, опционально вырезает область и сохраняет данные в CSV.

        Args:
            csv_name (str): Имя CSV-файла для сохранения данных.
            signal_check_results_path (str): Путь для сохранения CSV-файла с именами файлов SPEF.
            norm_coef_file_path (str): Путь к CSV-файлу с коэффициентами нормализации.
            is_cut_out_area (bool): Вырезать ли область вокруг события SPEF.
        """
        spef_files_df = pl.read_csv(signal_check_results_path)
        spef_filenames_dict = {}
        for row in spef_files_df.iter_rows(named=True):
            filename = row['filename']
            file_name_bus = row['file_name_bus']
            if filename not in spef_filenames_dict:
                spef_filenames_dict[filename] = set()
            spef_filenames_dict[filename].add(file_name_bus)

        dataset_dfs = []
        raw_files = sorted([file for file in os.listdir(self.raw_path)
                            if 'cfg' in file])
        number_spef_found = 0
        normOsc = NormOsc(norm_coef_file_path = norm_coef_file_path)

        with tqdm(total=len(raw_files), desc="Преобразование Comtrade в CSV для SPEF") as pbar:
            for file in raw_files:
                filename_without_ext = file[:-4]
                if filename_without_ext not in spef_filenames_dict: # Оптимизация: пропускать файлы, отсутствующие в списке SPEF
                    pbar.update(1)
                    continue

                raw_date, raw_df = self.readComtrade.read_comtrade(self.raw_path + file)
                if not raw_df.is_empty():
                    raw_df = normOsc.normalize_bus_signals(raw_df, filename_without_ext, yes_prase="YES", is_print_error=False)
                    if raw_df is None:
                        pbar.update(1)
                        if is_print_error:
                            print(f"Предупреждение: В {filename_without_ext} нормализацию провести нельзя, значения не используем.")
                        continue
                        
                    buses_df = self.split_buses(raw_df, file) # Использовать общий split_buses согласно вашему комментарию
                    processed_bus_dfs = []
                    
                    unique_files = buses_df['file_name'].unique().to_list()
                    for file_name_bus in unique_files:
                        bus_df = buses_df.filter(pl.col('file_name') == file_name_bus)
                        if file_name_bus in spef_filenames_dict[filename_without_ext]: # Дальнейшая фильтрация по имени шины
                            bus_df = self._process_signals_for_SPEF(bus_df, is_print_error=False) # Оставить обработку как есть или заменить при необходимости
                            if bus_df is not None:
                                processed_bus_dfs.append(bus_df)

                    if not processed_bus_dfs: # Нет обработанных датафреймов шин для этого файла
                        pbar.update(1)
                        continue
                    buses_df = pl.concat(processed_bus_dfs, how="diagonal")

                    # TODO: Написать cut_out_area_for_SPEF
                    # В теории, можно по аналогии с поисковой функцией, но пока обойдусь без этого.
                    # Либо можно по MLsignal_y_2_1 и всем наследникам - но пока что не много размечено осциллограмм.
                    dataset_dfs.append(buses_df)

                    number_spef_found += 1
                    pbar.update(1)

        print(f"Количество найденных образцов SPEF = {number_spef_found}")
        if dataset_dfs:
            dataset_df = pl.concat(dataset_dfs, how="diagonal")
        else:
            dataset_df = pl.DataFrame()
        dataset_df.write_csv(self.csv_path + csv_name)
        return dataset_df
    
    # TODO: подумать об унификации данной вещи, пока это локальная реализация
    def _process_signals_for_PDR(self, buses_df, is_print_error = False, is_check_PDR = True):
        """Обработка аналоговых сигналов: выбор шинных сборок или кабельных линий, расчет Ib."""
        
        u_bb_names = {"UA BB", "UB BB", "UC BB"}
        u_cl_names = {"UA CL", "UB CL", "UC CL"}
        # i_names = {"IA", "IB", "IC"}
        # pdr_PS_names = {"PDR PS"}
        pdr_phase_names = {"rPDR A", "rPDR B", "rPDR C"}
        
        # Список для накопления обработанных групп
        processed_groups = []
        
        # Группировка по имени файла
        unique_files = buses_df['file_name'].unique().to_list()
        for file_name in unique_files:
            group_df = buses_df.filter(pl.col('file_name') == file_name)
            
            # 1. Определяем источник напряжения из первой строки группы
            row0 = group_df.row(0, named=True)
            
            has_bb = all(col in group_df.columns for col in u_bb_names)
            has_cl = all(col in group_df.columns for col in u_cl_names)
            
            use_bb = has_bb and all(row0.get(col) is not None for col in u_bb_names)
            use_cl = has_cl and all(row0.get(col) is not None for col in u_cl_names)

            if use_bb:
                # Используем шинные сборки
                group_df = group_df.with_columns([
                    pl.col("UA BB").alias("UA"),
                    pl.col("UB BB").alias("UB"),
                    pl.col("UC BB").alias("UC")
                ])
            elif use_cl:
                # Используем кабельные линии
                group_df = group_df.with_columns([
                    pl.col("UA CL").alias("UA"),
                    pl.col("UB CL").alias("UB"),
                    pl.col("UC CL").alias("UC")
                ])
            else:
                if is_print_error:
                    print(f"Предупреждение: фазные напряжения A/B/C не найдены в {file_name}.")
                return None

            # 2. Фазные токи: IA и IC копируются напрямую, если столбцы присутствуют
            if not ("IA" in group_df.columns) or not ("IC" in group_df.columns):
                if is_print_error:
                    print(f"Предупреждение: фазные токи A/C не найдены в {file_name}.")
                return None

            # 3. Фазный ток B: если столбец "IB" существует и значение не NaN, используем его,
            # иначе вычисляем как -(IA + IC), если IA и IC заданы.
            if "IB" in group_df.columns:
                group_df = group_df.with_columns(
                    pl.when(pl.col("IB").is_null())
                    .then(-(pl.col("IA") + pl.col("IC")))
                    .otherwise(pl.col("IB"))
                    .alias("IB_proc")
                )
            else:
                group_df = group_df.with_columns(
                    (-(pl.col("IA") + pl.col("IC"))).alias("IB_proc")
                )

            # 4. Сигнал PDR:
            # Если столбец "PDR PS" существует и первая строка не NaN, берем его для всей группы.
            if is_check_PDR:
                if "rPDR PS" in group_df.columns and row0.get("rPDR PS") is not None:
                    group_df = group_df.with_columns(pl.col("rPDR PS").alias("PDR_proc"))
                elif all(col in group_df.columns for col in pdr_phase_names):
                    # Векторная проверка для каждой строки: если все столбцы pdr_phase_names существуют и равны 1, то PDR = 1, иначе 0.
                    expr = pl.lit(True)
                    for col in pdr_phase_names:
                        expr = expr & (pl.col(col) == 1)
                    group_df = group_df.with_columns(expr.cast(pl.Int32).alias("PDR_proc"))
                else:
                    if is_print_error:
                        print(f"Предупреждение: сигналы фазного напряжения PDR не найдены в {file_name}.")
                        # Но это может случиться, это не системная ошибка.
                    continue
                
                # Выбираем необходимые столбцы и переименовываем временные столбцы
                processed_group = group_df.select(["file_name", "UA", "UB", "UC", "IA", "IB_proc", "IC", "PDR_proc"]).rename(
                    {"IB_proc": "IB", "PDR_proc": "rPDR PS"}
                )
            else: # Без сигналов PDR
                processed_group = group_df.select(["file_name", "UA", "UB", "UC", "IA", "IB_proc", "IC"]).rename(
                    {"IB_proc": "IB"}
                )
            processed_groups.append(processed_group)
        
        # Объединяем все группы в итоговый DataFrame
        if len(processed_groups) > 0:
            signal_for_PDR_df = pl.concat(processed_groups, how="diagonal")
            return signal_for_PDR_df
        else:
            return None
        
    # TODO: подумать об унификации данной вещи, пока это локальная реализация
    def _process_signals_for_SPEF(self, buses_df, is_print_error = False):
        """Обработка аналоговых сигналов: выбор шинных сборок или кабельных линий, расчет Ib."""
        file_name = {"file_name"}
        u_bb_names = {"UA BB", "UB BB", "UC BB", "UN BB"}
        u_cl_names = {"UA CL", "UB CL", "UC CL", "UN CL"}
        i_names = {"IA", "IB", "IC", "IN"}
        ml_names = set(self.get_short_names_ml_signals()[0])
        
        #ml_names = {col for col in self.get_short_names_ml_signals()[0]}
        
        # Объединяем имена столбцов в единое множество
        all_signals = file_name | u_bb_names | u_cl_names | i_names | ml_names

        # Фильтруем столбцы, которые действительно присутствуют в DataFrame
        available_signals = [col for col in all_signals if col in buses_df.columns]

        # Возвращаем набор данных, состоящий только из необходимых столбцов
        return buses_df.select(available_signals)
        
    def create_one_df(self, file_path, file_name) -> pl.DataFrame:
        """
        Функция преобразования одного файла Comtrade в pd.DataFrame().

        Аргументы:
            file_path (str): путь к файлу comtrade.
            file_name (str): имя файла comtrade.
            
        Возвращает:
            pd.DataFrame: DataFrame файла comtrade.
        """
        dataset_df = pl.DataFrame()
        _, raw_df = self.readComtrade.read_comtrade(file_path)
        self.check_columns(raw_df)
        if not raw_df.is_empty():
            # raw_df = raw_df.reset_index()
            dataset_df = self.split_buses(raw_df, file_name)

        return dataset_df
    
    def split_buses(self, raw_df, file_name):
        """Реализовано только для шины 1 и шины 2"""
        # TODO: Подумайте/рассмотрите обработку случаев, когда имена сигналов совпадают. Это недостатки стандартизации имен
        # Это может происходить часто. Для случаев "U | BusBar-1 | phase: A" и подобных, это часто разделение на
        # BusBar / CableLine.
        # Для токов - иногда это разделение на секции
        # English:
        # Think about/consider handling cases where the names of the signals are the same. These are flaws in the standardization of names
        # This can happen often. For cases of "U|BusBar-1 | phase: A" and similar, this is often a division into
        # BusBar / CableLine.
        # For currents - sometimes it is divided into sections
        
        buses_dfs = []
        buses_cols = dict()
        raw_cols = set(raw_df.columns)
        for bus, cols in self.analog_names_dict.items():
            cols = raw_cols.intersection(cols)
            for i_bus in self.uses_buses:
                if bus[-1] == i_bus or bus[-2] == i_bus:
                    ml_all = self.get_ml_signals(i_bus)
                    raw_ml = raw_cols.intersection(ml_all)
                    cols = cols.union(raw_ml)
            if cols:
                buses_cols[bus] = cols
        for bus, columns in buses_cols.items():
            bus_df = raw_df.select(list(columns))
            bus_df = bus_df.with_columns(pl.lit(file_name[:-4] + "_" + bus).alias('file_name'))
            bus_df = self.rename_bus_columns(bus_df)
            buses_dfs.append(bus_df)
            
        if buses_dfs:
            return pl.concat(buses_dfs, how="diagonal")
        else:
            return pl.DataFrame()
    
    def split_buses_for_PDR(self, raw_df, file_name):
        """Реализовано только для шины 1 и шины 2"""
        buses_dfs = []
        buses_cols = dict()
        raw_cols = set(raw_df.columns)
        for bus, cols in self.analog_names_dict.items():
            cols = raw_cols.intersection(cols)
            for i_bus in self.uses_buses:
                if bus[-1] == i_bus or bus[-2] == i_bus:
                    # TODO: подумайте о том, что можно использовать любой другой дискретный сигнал, а не только PDR
                    # и такой генератор (get_PDR_signals) является временным решением в перечислении аналоговых сигналов
                    ml_all = self.get_PDR_signals(i_bus)
                    raw_ml = raw_cols.intersection(ml_all)
                    cols = cols.union(raw_ml)
            if cols:
                buses_cols[bus] = cols    
        
        for bus, columns in buses_cols.items():
            bus_df = raw_df.select(list(columns))
            bus_df = bus_df.with_columns(pl.lit(file_name[:-4] + "_" + bus).alias('file_name'))
            bus_df = self.rename_bus_columns(bus_df, is_use_ML=False, is_use_discrete=True)
            buses_dfs.append(bus_df)
            
        if buses_dfs:
            return pl.concat(buses_dfs, how="diagonal")
        else:
            return pl.DataFrame()

    def get_bus_names(self, analog=True, discrete=False):
        """
        Эта функция создает словарь аналоговых и дискретных имен
        для каждой шины.

        Аргументы:
            discrete (bool): False - только аналоговые имена.

        Возвращает:
            dict: словарь аналоговых и дискретных имен для каждой шины.
        """
        bus_names = dict()
        if analog:
            for bus in self.analog_names.keys():
                bus_names[bus] = set()
                for v in self.analog_names[bus].values():
                    bus_names[bus] = bus_names[bus].union(v)
        if discrete:
            for bus in self.discrete_names.keys():
                if bus not in bus_names:
                    bus_names[bus] = set()
                for v in self.discrete_names[bus].values():
                    bus_names[bus] = bus_names[bus].union(v)
        return bus_names

    def get_all_names(self,):
        """
        Эта функция создает набор всех аналоговых и дискретных имен.

        Возвращает:
            set: набор всех аналоговых и дискретных имен.
        """
        all_names = set()
        buses_names = self.get_bus_names(discrete=True)
        for bus, names in buses_names.items():
            all_names = all_names.union(names)
        return all_names

    def rename_raw_columns(self, raw_df):
        # TODO: вероятно, устарело в текущей реализации. Проверьте, будет ли вызываться хотя бы один раз.
        """
        Эта функция переименовывает столбцы в необработанном DataFrame.

        Аргументы:
            raw_df (pandas.DataFrame): DataFrame необработанного файла comtrade.

        Возвращает:
            pandas.DataFrame: DataFrame с переименованными столбцами.
        """
        raw_columns_to_rename = {'Mlsignal_2_2_1_1': 'MLsignal_2_2_1_1',
                                 'Mlsignal_12_2_1_1': 'MLsignal_12_2_1_1',
                                 'Mlsignal_2_3': 'MLsignal_2_3'}
        return raw_df.rename(raw_columns_to_rename)

    def rename_bus_columns(self, bus_df, is_use_ML = True, is_use_discrete = False):
        """
        Эта функция переименовывает столбцы в DataFrame шины.

        Аргументы:
            buses_df (pandas.DataFrame): DataFrame с одной шиной.

        Возвращает:
            pandas.DataFrame: DataFrame с переименованными столбцами.
        """
        bus_columns_to_rename = {}

        # Генерируем переименование для сигналов ML
        if is_use_ML:
            for i_bus in self.uses_buses:
                ml_signals = self.get_ml_signals(i_bus)
                for signal in ml_signals:
                    new_name = signal.replace(f'MLsignal_{i_bus}_', 'ML_')
                    bus_columns_to_rename[signal] = new_name
        
        # Генерируем переименование для аналоговых сигналов
        for bus, names in self.analog_names_dict.items():
            for name in names:
                if 'I | Bus' in name:
                    phase = name.split(': ')[-1]
                    if (phase == 'B' and not self.uses_CT_B) or (phase == 'N' and not self.uses_CT_zero):
                        continue
                    bus_columns_to_rename[name] = f'I{phase}'
                    
                elif self.use_VT_BB and 'U | BusBar' in name:
                    phase = name.split(': ')[-1]
                    if (((phase == 'A' or phase == 'B' or phase == 'C') and not self.uses_VT_ph) or 
                        ((phase == 'AB' or phase == 'BC' or phase == 'CA') and not self.uses_VT_iph) or
                         (phase == 'N' and not self.uses_VT_zero)):
                        continue
                    bus_columns_to_rename[name] = f'U{phase} BB'
                    
                elif self.use_VT_CL and 'U | CableLine' in name:
                    phase = name.split(': ')[-1]
                    if (((phase == 'A' or phase == 'B' or phase == 'C') and not self.uses_VT_ph) or 
                        ((phase == 'AB' or phase == 'BC' or phase == 'CA') and not self.uses_VT_iph) or
                         (phase == 'N' and not self.uses_VT_zero)):
                        continue 
                    bus_columns_to_rename[name] = f'U{phase} CL'
        
        # Генерируем переименование для дискретных сигналов
        if is_use_discrete:
            for bus, names in self.discrete_names_dict.items():
                for name in names:
                    if self.use_PDR and 'PDR | Bus' in name:
                        phase = name.split(': ')[-1]
                        bus_columns_to_rename[name] = f'rPDR {phase}'
                    elif self.use_PDR and 'PDR_ideal | Bus' in name:
                        phase = name.split(': ')[-1]
                        bus_columns_to_rename[name] = f'iPDR {phase}' 
                   
        # TODO: сигналы I_raw, U_raw, I|dif-1, I | braking-1 не учитываются

        return bus_df.rename(bus_columns_to_rename)
    
    def check_columns(self, raw_df):
        """проверка на наличие неизвестных столбцов (только логирование, без выброса исключений)"""
        ml_signals = set()
        for i_bus in self.uses_buses:
            ml_signals.update(self.get_ml_signals(i_bus))

        all_names = self.all_names.union(ml_signals)
        columns = raw_df.columns
        unknown_columns = []
        for c in columns:
            if c not in all_names:
                unknown_columns.append(c)
        
        # Просто логируем неизвестные столбцы без выброса исключения
        if unknown_columns:
            pass  # Молча пропускаем неизвестные столбцы (например, 'Time' при построении графиков)

    def get_ml_signals(self, i_bus, use_operational_switching=True, use_abnormal_event=True, use_emergency_event=True):
        """
        Эта функция возвращает набор сигналов ML для данной шины.

        Аргументы:
            i_bus (str): номер шины.
            use_operational_switching (bool): включать сигналы оперативного переключения.
            use_abnormal_event (bool): включать сигналы анормальных событий.
            use_emergency_even (bool): включать сигналы аварийных событий.

        Возвращает:
            set: набор сигналов ML для данной шины.
        """
        # FIXME: переписать так, чтобы он записывался в самом начале и считался 1 раз, а не при каждом запросе
        ml_signals = set()

        ml_operational_switching = {
            #--- Рабочие переключения ---
            f'MLsignal_{i_bus}_1',      # Рабочее переключение, без уточнения
            f'MLsignal_{i_bus}_1_1',    # Оперативное включение, без уточнения
            f'MLsignal_{i_bus}_1_1_1',  # Пуск двигателя, пуск двигателя
            f'MLsignal_{i_bus}_1_2',    # Оперативное отключение, без уточнения
        }

        ml_abnormal_event = {
            # --- Анормальные события
            f'MLsignal_{i_bus}_2',      # Аномалия, без уточнения
            f'MLsignal_{i_bus}_2_1',    # Однофазное замыкание на землю, без уточнения
            f'MLsignal_{i_bus}_2_1_1',  # Устойчивое однофазное замыкание на землю
            f'MLsignal_{i_bus}_2_1_2',  # Устойчивое затухающее однофазное замыкание на землю с редкими пробоями
            f'MLsignal_{i_bus}_2_1_3',  # Дуговое прерывистое однофазное замыкание на землю
            f'MLsignal_{i_bus}_2_2',    # Затухающие колебания от аварийных процессов
            f'MLsignal_{i_bus}_2_3',    # Просадка напряжения
            f'MLsignal_{i_bus}_2_3_1',  # Просадка напряжения при пуске двигателя
            f'MLsignal_{i_bus}_2_4',    # Колебания тока, без уточнения
            f'MLsignal_{i_bus}_2_4_1',  # Колебания тока при пуске двигателя
            f'MLsignal_{i_bus}_2_4_2',  # Колебания тока от частотно-регулируемых двигателей
        }

        ml_emergency_event = {
            # --- Аварийные события ----
            f'MLsignal_{i_bus}_3',      # Аварийные события, без уточнения
            f'MLsignal_{i_bus}_3_1',    # Авария из-за неправильной работы устройства, без уточнения
            f'MLsignal_{i_bus}_3_2',    # Неисправность терминала
            f'MLsignal_{i_bus}_3_3'     # Двухфазное замыкание на землю
        }
            
        ml_signals = set()
        if use_operational_switching:
            ml_signals.update(ml_operational_switching)
        if use_abnormal_event:
            ml_signals.update(ml_abnormal_event)
        if use_emergency_event:
            ml_signals.update(ml_emergency_event)

        return ml_signals
    
    def get_PDR_signals(self, i_bus):
        """
        Эта функция возвращает набор сигналов ML для данной шины.

        Аргументы:
            i_bus (str): номер шины.

        Возвращает:
            set: набор PDR для сигналов ML для данной шины.
        """

        ml_signals = {
            #--- Рабочие переключения ---
            f'PDR | Bus-{i_bus} | phase: A',
            f'PDR | Bus-{i_bus} | phase: B',
            f'PDR | Bus-{i_bus} | phase: C',
            f'PDR | Bus-{i_bus} | phase: PS',
            f'PDR_ideal | Bus-{i_bus} | phase: A',
            f'PDR_ideal | Bus-{i_bus} | phase: B',
            f'PDR_ideal | Bus-{i_bus} | phase: C',
            f'PDR_ideal | Bus-{i_bus} | phase: PS',
        }

        return ml_signals
    
    def get_short_names_ml_signals(self, use_operational_switching: bool =True, use_abnormal_event: bool = True, use_emergency_event: bool = True) -> list:
        """
        Эта функция возвращает набор коротких имен сигналов ML для (без i_bus).

        Аргументы:
            use_operational_switching (bool): включать сигналы оперативного переключения.
            use_abnormal_event (bool): включать сигналы анормальных событий.
            use_emergency_event (bool): включать сигналы аварийных событий.

        Возвращает:
            list: список сигналов ML для данной шины.
        """
        # FIXME: переписать так, чтобы он записывался в самом начале и считался 1 раз, а не при каждом запросе

        ml_operational_switching = [
            #--- Рабочие переключения ---
            'ML_1',      # Рабочее переключение, без уточнения
            'ML_1_1',    # Оперативное включение, без уточнения
            'ML_1_1_1',  # Пуск двигателя, пуск двигателя
            'ML_1_2'    # Оперативное отключение, без уточнения
        ]
        ml_abnormal_event = [
            # --- Анормальные события
            'ML_2',      # Аномалия, без уточнения
            'ML_2_1',    # Однофазное замыкание на землю, без уточнения
            'ML_2_1_1',  # Устойчивое однофазное замыкание на землю
            'ML_2_1_2',  # Устойчивое затухающее однофазное замыкание на землю с редкими пробоями
            'ML_2_1_3',  # Дуговое прерывистое однофазное замыкание на землю
            'ML_2_2',    # Затухающие колебания от аварийных процессов
            'ML_2_3',    # Просадка напряжения
            'ML_2_3_1',  # Просадка напряжения при пуске двигателя
            'ML_2_4',    # Колебания тока, без уточнения
            'ML_2_4_1',  # Колебания тока при пуске двигателя
            'ML_2_4_2'  # Колебания тока от частотно-регулируемых двигателей
            ]

        ml_emergency_event = [
            # --- Аварийные события ----
            'ML_3',      # Аварийные события, без уточнения
            'ML_3_1',    # Авария из-за неправильной работы устройства, без уточнения
            'ML_3_2',    # Неисправность терминала
            'ML_3_3'     # Двухфазное замыкание на землю
        ]
        
        ml_signals = []
        if use_operational_switching:
            ml_signals.extend(ml_operational_switching)
        if use_abnormal_event:
            ml_signals.extend(ml_abnormal_event)
        if use_emergency_event:
            ml_signals.extend(ml_emergency_event)
        
        return ml_signals, ml_operational_switching, ml_abnormal_event, ml_emergency_event


    def get_short_names_ml_analog_signals(self) -> list:
        """
        Эта функция возвращает набор коротких имен аналоговых сигналов ML для (без i_bus).

        Аргументы:

        Возвращает:
            list: набор сигналов ML для данной шины.
        """

        ml_current = [
            'IA', 'IB', 'IC', 'IN'
        ]
        ml_votage_BB = [
             'UA BB', 'UB BB', 'UC BB', 'UN BB', 'UAB BB', 'UBC BB', 'UCA BB',
        ]
        ml_votage_CL = [
             'UA CL', 'UB CL', 'UC CL', 'UN CL', 'UAB CL', 'UBC CL', 'UCA CL',
        ]  
        # TODO: сигналы I_raw, U_raw, I|dif-1, I | braking-1 не учитываются
        
        ml_signals = []
        ml_signals.extend(ml_current)
        ml_signals.extend(ml_votage_BB)
        ml_signals.extend(ml_votage_CL)
        
        return ml_signals

    def cut_out_area(self, buses_df: pl.DataFrame, samples_before: int, samples_after: int) -> pl.DataFrame:
        """
        Функция отсекает участки, не содержащие сигналов ML, оставляя до и после них заданную границу.

        Аргументы:
            buses_df (pl.DataFrame): DataFrame для обработки
            samples_before (int): количество отсчетов для отсечения до первого сигнала ML.
            samples_after (int): количество отсчетов для отсечения после последнего сигнала ML.
            
        Возвращает:
            pl.DataFrame: DataFrame с вырезанными участками.
        """
        dataset_dfs = []
        unique_files = buses_df['file_name'].unique().to_list()
        
        for file_name in unique_files:
            bus_df = buses_df.filter(pl.col('file_name') == file_name)
            
            filtered_column_names = [col for col in bus_df.columns if col in self.ml_all]
            
            if not filtered_column_names:
                # No ML columns, so no events.
                # Fallback to middle slice logic
                if len(bus_df) > samples_before + samples_after:
                    middle = len(bus_df) // 2
                    start = max(0, middle - samples_before)
                    end = min(len(bus_df), middle + samples_after + 1)
                    truncated_dataset = bus_df.slice(start, end - start)
                else:
                    truncated_dataset = bus_df
                dataset_dfs.append(truncated_dataset)
                continue

            expr = pl.lit(False)
            for col in filtered_column_names:
                expr = expr | (pl.col(col) == 1)
            
            bus_df = bus_df.with_columns(expr.alias("is_save_base"))
            
            bus_df = bus_df.with_row_index("row_idx")
            save_indices = bus_df.filter(pl.col("is_save_base"))["row_idx"].to_list()
            
            if not save_indices:
                 # No events
                if len(bus_df) > samples_before + samples_after:
                    middle = len(bus_df) // 2
                    start = max(0, middle - samples_before)
                    end = min(len(bus_df), middle + samples_after + 1)
                    truncated_dataset = bus_df.slice(start, end - start)
                else:
                    truncated_dataset = bus_df
                dataset_dfs.append(truncated_dataset)
                continue
            
            # Create mask
            is_save = np.zeros(len(bus_df), dtype=bool)
            
            for idx in save_indices:
                start = max(0, idx - samples_before)
                end = min(len(bus_df) - 1, idx + samples_after)
                is_save[start:end+1] = True
                
            truncated_dataset = bus_df.filter(pl.lit(is_save))
            
            # Assign event numbers
            if not truncated_dataset.is_empty():
                 truncated_dataset = truncated_dataset.with_columns(
                     (pl.col("row_idx").diff().fill_null(1) != 1).cum_sum().alias("event_group")
                 )
                 
                 truncated_dataset = truncated_dataset.with_columns(
                     (pl.col("file_name") + " _event N" + pl.col("event_group").cast(pl.Utf8)).alias("file_name")
                 )
            
            # Drop temp cols
            cols_to_drop = ["is_save_base", "row_idx", "event_group"]
            existing_cols_to_drop = [col for col in cols_to_drop if col in truncated_dataset.columns]
            if existing_cols_to_drop:
                 truncated_dataset = truncated_dataset.drop(existing_cols_to_drop)
                 
            dataset_dfs.append(truncated_dataset)
            
        if dataset_dfs:
            return pl.concat(dataset_dfs, how="diagonal")
        else:
            return pl.DataFrame()
    
    # TODO: переписать функции, обобщить. Чтобы набор / обработка столбцов была параметром.
    # К тому же, раньше функция прост расширяла. А это регистрирует изменения (0-1 и 1-0) и расширяет вокруг них.
    def cut_out_area_for_PDR(self, buses_df: pl.DataFrame, samples_before: int, samples_after: int) -> pl.DataFrame:
        """
        Обрабатывает DataFrame, регистрируя факты изменения сигнала (0→1 и 1→0)
        и расширяя зону вокруг изменений на заданное число отсчётов.

        Сигнал для обнаружения изменений определяется по следующей логике:
        1. Если столбец "rPDR PS" существует и содержит не только NaN, используется он (1 если значение == 1, иначе 0).
        2. Иначе, если столбцы "rPDR A", "rPDR B", "rPDR C" существуют, 
           сигнал равен 1 только если все три столбца равны 1, иначе 0.
        3. Иначе (если ни один из источников сигнала не доступен), сигнал считается равным 0 для всех строк.

        Аргументы:
            buses_df (pl.DataFrame): Исходный DataFrame с данными. Должен содержать столбец 'file_name'.
            samples_before (int): Количество строк до точки изменения, которые надо включить.
            samples_after (int): Количество строк после точки изменения, которые надо включить.

        Возвращает:
            pl.DataFrame: Обработанный DataFrame с выделенными зонами и новым столбцом change_event.
                          Строки из разных исходных файлов могут быть перемешаны, но сгруппированы по событиям.
        """
        dataset_dfs = []
        
        primary_signal_col = "rPDR PS"
        composite_signal_cols = ["rPDR A", "rPDR B", "rPDR C"]

        unique_files = buses_df['file_name'].unique().to_list()
        for file_name in unique_files:
            bus_df = buses_df.filter(pl.col('file_name') == file_name)
            
            signal_calculated = False
            
            # 1. Try primary signal "rPDR PS"
            if primary_signal_col in bus_df.columns and bus_df[primary_signal_col].is_not_null().any():
                bus_df = bus_df.with_columns((pl.col(primary_signal_col) == 1).alias("signal"))
                signal_calculated = True

            # 2. If primary signal not used, try composite signals
            if not signal_calculated:
                if all(col in bus_df.columns for col in composite_signal_cols):
                    # Calculate composite signal: 1 only if ALL are 1
                    expr = pl.lit(True)
                    for col in composite_signal_cols:
                        expr = expr & (pl.col(col) == 1)
                    
                    bus_df = bus_df.with_columns(expr.alias(primary_signal_col))
                    bus_df = bus_df.with_columns((pl.col(primary_signal_col) == 1).alias("signal"))
                    signal_calculated = True
                    
            # 3. Default to False if no signal source was found/usable
            if not signal_calculated:
                 continue 

            # Determine the previous signal state to detect changes
            bus_df = bus_df.with_columns(pl.col("signal").shift(1).fill_null(False).alias("prev_signal"))

            # Register changes in 'change_event': "0-1" or "1-0"
            bus_df = bus_df.with_columns(
                pl.when((pl.col("prev_signal") == False) & (pl.col("signal") == True))
                .then(pl.lit("0-1"))
                .when((pl.col("prev_signal") == True) & (pl.col("signal") == False))
                .then(pl.lit("1-0"))
                .otherwise(pl.lit(None))
                .alias("change_event")
            )

            # Find indices where a change occurred
            bus_df = bus_df.with_row_index("row_idx")
            change_indices = bus_df.filter(pl.col("change_event").is_not_null())["row_idx"].to_list()

            ranges = []
            for idx in change_indices:
                 start_idx = max(0, idx - samples_before)
                 end_idx = min(len(bus_df) - 1, idx + samples_after)
                 ranges.append((start_idx, end_idx))
            
            if not ranges:
                # Handle case where no events were found
                if len(bus_df) > samples_before + samples_after:
                    middle = len(bus_df) // 2
                    start = max(0, middle - samples_before)
                    end = min(len(bus_df), middle + samples_after + 1)
                    truncated_dataset = bus_df.slice(start, end - start)
                else:
                    truncated_dataset = bus_df
            else:
                # Create mask
                is_save = np.zeros(len(bus_df), dtype=bool)
                for start, end in ranges:
                    is_save[start:end+1] = True
                
                truncated_dataset = bus_df.filter(pl.lit(is_save))

            # Assign event numbers
            if not truncated_dataset.is_empty():
                 truncated_dataset = truncated_dataset.with_columns(
                     (pl.col("row_idx").diff().fill_null(1) != 1).cum_sum().alias("event_group")
                 )
                 
                 truncated_dataset = truncated_dataset.with_columns(
                     (pl.col("file_name") + " _event N" + pl.col("event_group").cast(pl.Utf8)).alias("file_name")
                 )

            # Drop temporary columns
            cols_to_drop = ["is_save", "prev_signal", "signal", "rPDR A", "rPDR B", "rPDR C", "row_idx", "event_group", "change_event"]
            existing_cols_to_drop = [col for col in cols_to_drop if col in truncated_dataset.columns]
            if existing_cols_to_drop:
                 truncated_dataset = truncated_dataset.drop(existing_cols_to_drop)

            dataset_dfs.append(truncated_dataset)

        if dataset_dfs:
            return pl.concat(dataset_dfs, how="diagonal")
        else:
            return pl.DataFrame()
    
    def get_simple_dataset(self, dataset_df: pl.DataFrame, csv_name='dataset_simpl.csv'):
        """ 
        Создание новых столбцов для упрощенной версии. Формируется только 4 группы сигналов:
        1) Оперативные переключения
        2) Анормальные события
        3) Аварийные события
        4) Нормальные (без событий)
        """
        column_names = set(dataset_df.columns)
        ml_opr_swch = set(self.ml_opr_swch).intersection(column_names)
        ml_abnorm_evnt = set(self.ml_abnorm_evnt).intersection(column_names)
        ml_emerg_evnt = set(self.ml_emerg_evnt).intersection(column_names)
        ml_all = ml_opr_swch.union(ml_abnorm_evnt).union(ml_emerg_evnt)
        
        if ml_emerg_evnt:
            dataset_df = dataset_df.with_columns(
                pl.when(pl.any_horizontal(list(ml_emerg_evnt)))
                .then(pl.lit("1"))
                .otherwise(pl.lit(""))
                .alias("emerg_evnt")
            )
        else:
            dataset_df = dataset_df.with_columns(pl.lit("").alias("emerg_evnt"))

        if ml_abnorm_evnt:
            dataset_df = dataset_df.with_columns(
                pl.when(pl.any_horizontal(list(ml_abnorm_evnt)))
                .then(pl.lit("1"))
                .otherwise(pl.lit(""))
                .alias("abnorm_evnt")
            )
        else:
            dataset_df = dataset_df.with_columns(pl.lit("").alias("abnorm_evnt"))
            
        dataset_df = dataset_df.with_columns(
            pl.when(pl.col("emerg_evnt") == "1")
            .then(pl.lit(""))
            .otherwise(pl.col("abnorm_evnt"))
            .alias("abnorm_evnt")
        )
        
        if ml_all:
            dataset_df = dataset_df.with_columns(
                pl.when(pl.any_horizontal(list(ml_all)))
                .then(pl.lit(""))
                .otherwise(pl.lit(1))
                .alias("normal")
            )
        else:
            dataset_df = dataset_df.with_columns(pl.lit(1).alias("normal"))

        dataset_df = dataset_df.with_columns(
            pl.when((pl.col("abnorm_evnt") == "1") | (pl.col("emerg_evnt") == "1"))
            .then(pl.lit(""))
            .otherwise(pl.lit(1))
            .alias("no_event")
        )
        
        # Удаляем сигналы ML
        dataset_df = dataset_df.drop(list(ml_all))
        
        dataset_df.write_csv(self.csv_path + csv_name)

    def structure_columns(self, dataset_df: pl.DataFrame) -> pl.DataFrame:
        """
        Эта функция переструктурирует столбцы набора данных для более простого анализа.
        """
        # TODO: некоторые имена могут отсутствовать, поэтому подумайте об их обработке.
        # Укажите желаемый порядок столбцов
        desired_order = ["file_name"]
        desired_order.extend(self.get_short_names_ml_analog_signals())
        desired_order.extend(self.get_short_names_ml_signals()[0])
        desired_order.extend(["opr_swch", "abnorm_evnt", "emerg_evnt", "normal"])
        
        # Проверяем, существует ли каждый столбец в желаемом порядке в DataFrame
        existing_columns = [col for col in desired_order if col in dataset_df.columns]

        # Переупорядочиваем столбцы DataFrame на основе существующих столбцов
        dataset_df = dataset_df.select(existing_columns)
        
        return dataset_df
