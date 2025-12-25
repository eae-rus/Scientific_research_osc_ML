import pandas as pd
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
        dataset_df = pd.DataFrame()
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
                if not raw_df.empty:
                    raw_df = raw_df.reset_index()
                    buses_df = self.split_buses(raw_df, file)

                    frequency = raw_date.cfg.frequency
                    samples_rate = raw_date.cfg.sample_rates[0][0]
                    number_samples = int(samples_rate / frequency) # TODO: Это не всегда будет целое число, но это редкость.
                    samples_before, samples_after = number_samples * self.number_periods, number_samples * self.number_periods
                    if is_cut_out_area:
                        buses_df = self.cut_out_area(buses_df, samples_before, samples_after)
                        dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=False)
                    else:
                        dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=False)
                    pbar.update(1)
        
        # TODO: Подумайте об организации самой обработки по-другому, а не о добавлении функции.
        dataset_df = self.structure_columns(dataset_df)
        if is_simple_csv:
            # Мы не перезаписываем исходные данные, а только создаем еще один csv-файл
            self.get_simple_dataset(dataset_df.copy())
        
        dataset_df.to_csv(self.csv_path + csv_name, index=False)
        return dataset_df
    
    # TODO: подумать об универсанолизации данной функции с create_csv (основные замечания указаны там)
    def create_csv_for_PDR(self, csv_name='dataset.csv', signal_check_results_path='signal_check_results.csv', norm_coef_file_path='norm_coef.csv',
                           is_check_PDR = True, is_cut_out_area = False, yes_prase = "YES", is_print_error = False):
        """
        Эта функция DataFrame и сохраняет csv-файл из необработанных данных comtrade.

        Args:
            csv_name (str): Имя файла csv.
        """
        signal_check_df = pd.DataFrame()
        if (is_check_PDR and (not os.path.exists(signal_check_results_path))):
            raise FileNotFoundError("Путь для результатов проверки сигнала не существует")
            # TODO: Пока что это локальный файл, но надо подумать о том, чтобы он был глобальный, но пока может мешать.
        if is_check_PDR:
            signal_check_df = pd.read_csv(signal_check_results_path) # Загрузка файла проверки сигналов
        
        dataset_df = pd.DataFrame()
        raw_files = sorted([file for file in os.listdir(self.raw_path)
                            if 'cfg' in file])
        number_ocs_found = 0
        normOsc = NormOsc(norm_coef_file_path = norm_coef_file_path)
        with tqdm(total=len(raw_files), desc="Преобразование Comtrade в CSV") as pbar:
            for file in raw_files:
                filename_without_ext = file[:-4] # Имя файла без расширения
                # FIXME: На будущее - возможно этот файл и лишний, ибо проверки заложил в "_process_signals_for_PDR"
                if is_check_PDR:
                    check_result_row = signal_check_df[signal_check_df['filename'] == filename_without_ext]
                    if check_result_row.empty or not (str(check_result_row['contains_required_signals'].iloc[0]).upper()  == yes_prase):
                        pbar.update(1) # Пропускаем файл, если его нет в signal_check_results_csv или contains_required_signals False
                        continue
                
                raw_date, raw_df = self.readComtrade.read_comtrade(self.raw_path + file)
                raw_df = normOsc.normalize_bus_signals(raw_df, filename_without_ext, yes_prase="YES", is_print_error=False)
                if raw_df is None:
                    pbar.update(1)
                    if is_print_error:
                        print(f"Предупреждение: В {filename_without_ext} нормализацию провести нельзя, значения не используем.")
                    continue
                    
                if not raw_df.empty:
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
                        dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=False)
                    else:
                        dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=False)
                    dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=False)
                    number_ocs_found += 1
                    pbar.update(1)
        
        print(f"Количество найденных образцов = {number_ocs_found}")
        dataset_df.to_csv(self.csv_path + csv_name, index=False)
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
        spef_files_df = pd.read_csv(signal_check_results_path)
        spef_filenames_dict = {}
        for _, row in spef_files_df.iterrows():
            filename = row['filename']
            file_name_bus = row['file_name_bus']
            if filename not in spef_filenames_dict:
                spef_filenames_dict[filename] = set()
            spef_filenames_dict[filename].add(file_name_bus)

        columns = ["file_name", "IA", "IB", "IC", "IN", "UA BB", "UB BB", "UC BB", "UN BB", "UA CL", "UB CL", "UC CL", "UN CL"]
        dataset_df = pd.DataFrame(columns=columns)
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
                if not raw_df.empty:
                    raw_df = normOsc.normalize_bus_signals(raw_df, filename_without_ext, yes_prase="YES", is_print_error=False)
                    if raw_df is None:
                        pbar.update(1)
                        if is_print_error:
                            print(f"Предупреждение: В {filename_without_ext} нормализацию провести нельзя, значения не используем.")
                        continue
                        
                    buses_df = self.split_buses(raw_df.reset_index(), file) # Использовать общий split_buses согласно вашему комментарию
                    processed_bus_dfs = []
                    for file_name_bus, bus_df in buses_df.groupby('file_name'):
                        if file_name_bus in spef_filenames_dict[filename_without_ext]: # Дальнейшая фильтрация по имени шины
                            bus_df = self._process_signals_for_SPEF(bus_df, is_print_error=False) # Оставить обработку как есть или заменить при необходимости
                            if bus_df is not None:
                                processed_bus_dfs.append(bus_df)

                    if not processed_bus_dfs: # Нет обработанных датафреймов шин для этого файла
                        pbar.update(1)
                        continue
                    buses_df = pd.concat(processed_bus_dfs, ignore_index=True)

                    # TODO: Написать cut_out_area_for_SPEF
                    # В теории, можно по аналогии с поисковой функцией, но пока обойдусь без этого.
                    # Либо можно по MLsignal_y_2_1 и всем наследникам - но пока что не много размечено осциллограмм.
                    dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=False)

                    number_spef_found += 1
                    pbar.update(1)

        print(f"Количество найденных образцов SPEF = {number_spef_found}")
        dataset_df.to_csv(self.csv_path + csv_name, index=False)
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
        for file_name, group_df in buses_df.groupby("file_name"):
            # Мы будем работать с копией группы
            group_df = group_df.copy()
            
            # 1. Определяем источник напряжения из первой строки группы
            row0 = group_df.iloc[0]
            if all(col in group_df.columns and pd.notna(row0[col]) for col in u_bb_names):
                # Используем шинные сборки
                group_df["UA"] = group_df["UA BB"]
                group_df["UB"] = group_df["UB BB"]
                group_df["UC"] = group_df["UC BB"]
            elif all(col in group_df.columns and pd.notna(row0[col]) for col in u_cl_names):
                # Используем кабельные линии
                group_df["UA"] = group_df["UA CL"]
                group_df["UB"] = group_df["UB CL"]
                group_df["UC"] = group_df["UC CL"]
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
                IB_series = group_df["IB"].copy()
            else:
                IB_series = pd.Series(np.nan, index=group_df.index)
            # Находим строки, где IB отсутствует (NaN) и где IA и IC заданы
            mask_replace = IB_series.isna() & group_df["IA"].notna() & group_df["IC"].notna()
            IB_series.loc[mask_replace] = -(group_df.loc[mask_replace, "IA"] + group_df.loc[mask_replace, "IC"])
            group_df["IB_proc"] = IB_series

            # 4. Сигнал PDR:
            # Если столбец "PDR PS" существует и первая строка не NaN, берем его для всей группы.
            if is_check_PDR:
                if "rPDR PS" in group_df.columns and pd.notna(row0["rPDR PS"]):
                    group_df["PDR_proc"] = group_df["rPDR PS"]
                elif all(col in group_df.columns and pd.notna(row0[col]) for col in pdr_phase_names):
                    # Векторная проверка для каждой строки: если все столбцы pdr_phase_names существуют и равны 1, то PDR = 1, иначе 0.
                    cond = pd.Series(True, index=group_df.index)
                    for col in pdr_phase_names:
                        if col in group_df.columns:
                            cond = cond & (group_df[col] == 1)
                        else:
                            cond = cond & False
                    group_df["PDR_proc"] = cond.astype(int)
                else:
                    if is_print_error:
                        print(f"Предупреждение: сигналы фазного напряжения PDR не найдены в {file_name}.")
                        # Но это может случиться, это не системная ошибка.
                    continue
                
                # Выбираем необходимые столбцы и переименовываем временные столбцы
                processed_group = group_df[["file_name", "UA", "UB", "UC", "IA", "IB_proc", "IC", "PDR_proc"]].rename(
                    columns={"IB_proc": "IB", "PDR_proc": "rPDR PS"}
                )
            else: # Без сигналов PDR
                processed_group = group_df[["file_name", "UA", "UB", "UC", "IA", "IB_proc", "IC"]].rename(
                    columns={"IB_proc": "IB"}
                )
            processed_groups.append(processed_group)
        
        # Объединяем все группы в итоговый DataFrame
        if len(processed_groups) > 0:
            signal_for_PDR_df = pd.concat(processed_groups, ignore_index=True)
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
        return buses_df[available_signals]
        
    def create_one_df(self, file_path, file_name) -> pd.DataFrame:
        """
        Функция преобразования одного файла Comtrade в pd.DataFrame().

        Аргументы:
            file_path (str): путь к файлу comtrade.
            file_name (str): имя файла comtrade.
            
        Возвращает:
            pd.DataFrame: DataFrame файла comtrade.
        """
        dataset_df = pd.DataFrame()
        _, raw_df = self.readComtrade.read_comtrade(file_path)
        self.check_columns(raw_df)
        if not raw_df.empty:
            raw_df = raw_df.reset_index()
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
        
        buses_df = pd.DataFrame()
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
            bus_df = raw_df.loc[:, list(columns)]
            bus_df.insert(0, 'file_name',  [file_name[:-4] + "_" + bus] * bus_df.shape[0])
            bus_df = self.rename_bus_columns(bus_df)
            buses_df = pd.concat([buses_df, bus_df], axis=0,
                                 ignore_index=False)    
        return buses_df
    
    def split_buses_for_PDR(self, raw_df, file_name):
        """Реализовано только для шины 1 и шины 2"""
        buses_df = pd.DataFrame()
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
            bus_df = raw_df.loc[:, list(columns)]
            bus_df.insert(0, 'file_name',  [file_name[:-4] + "_" + bus] * bus_df.shape[0])
            bus_df = self.rename_bus_columns(bus_df, is_use_ML=False, is_use_discrete=True)
            buses_df = pd.concat([buses_df, bus_df], axis=0,
                                 ignore_index=False)    
        return buses_df

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
        raw_df.rename(columns=raw_columns_to_rename, inplace=True)
        return raw_df

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

        bus_df.rename(columns=bus_columns_to_rename, inplace=True)
        return bus_df
    
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

    def cut_out_area(self, buses_df: pd.DataFrame, samples_before: int, samples_after: int) -> pd.DataFrame:
        """
        Функция отсекает участки, не содержащие сигналов ML, оставляя до и после них заданную границу.

        Аргументы:
            buses_df (pd.DataFrame): DataFrame для обработки
            samples_before (int): количество отсчетов для отсечения до первого сигнала ML.
            samples_after (int): количество отсчетов для отсечения после последнего сигнала ML.
            
        Возвращает:
            pd.DataFrame: DataFrame с вырезанными участками.
        """
        dataset_df = pd.DataFrame()
        for _, bus_df in buses_df.groupby("file_name"):
            truncated_dataset = pd.DataFrame()
            # Сбрасываем индекс перед нарезкой
            bus_df = bus_df.reset_index(drop=True)

            bus_df["is_save"] = False
            filtered_column_names = [col for col in bus_df.columns if col in self.ml_all]

            # Определяем строки с сигналами ML
            bus_df["is_save"] = bus_df[filtered_column_names].notna().any(axis=1) & (bus_df[filtered_column_names] == 1).any(axis=1)

            # TODO: требует ускорения
            # Прямое заполнение: расширяем "is_save" для охвата участков перед сигналами ML
            for index, row in bus_df.iterrows():
                if index == 0:
                    continue
                if bus_df.loc[index, "is_save"] and bus_df.loc[index-1, "is_save"]:
                    if index >= samples_before:
                        bus_df.loc[index-samples_before:index, "is_save"] = True
                    else:
                        bus_df.loc[0:index, "is_save"] = True

            # TODO: требует ускорения
            # Обратное заполнение: расширяем "is_save" для охвата участков после сигналов ML
            for index, row in reversed(list(bus_df.iterrows())):
                if index == len(bus_df) - 1:
                    continue
                if bus_df.loc[index, "is_save"] and bus_df.loc[index+1, "is_save"]:
                    if index + samples_after < len(bus_df):
                        bus_df.loc[index+1:index+samples_after+1, "is_save"] = True
                    else:
                        bus_df.loc[index+1:len(bus_df), "is_save"] = True

            # Добавляем номера событий к именам файлов
            event_number = 0
            for index, row in bus_df.iterrows():
                if bus_df.loc[index, "is_save"]:
                    if (index == 0 or not bus_df.loc[index - 1, "is_save"]):
                        event_number += 1
                    bus_df.loc[index, 'file_name'] = bus_df.loc[index, 'file_name'] + " _event N" + str(event_number)

            truncated_dataset = bus_df[bus_df["is_save"]]
            # Если массив пуст, то берем из куска посередине, равного сумме длин до и после
            if len(truncated_dataset) == 0:
                if len(bus_df) > samples_before + samples_after:
                    middle = len(bus_df) // 2
                    truncated_dataset = bus_df.iloc[middle-samples_before:middle+samples_after+1]
                else:
                    truncated_dataset = bus_df

            truncated_dataset = truncated_dataset.drop(columns=["is_save"])
            dataset_df = pd.concat([dataset_df, truncated_dataset], axis=0, ignore_index=False)
            
        return dataset_df
    
    # TODO: переписать функции, обобщить. Чтобы набор / обработка столбцов была параметром.
    # К тому же, раньше функция прост расширяла. А это регистрирует изменения (0-1 и 1-0) и расширяет вокруг них.
    def cut_out_area_for_PDR(self, buses_df: pd.DataFrame, samples_before: int, samples_after: int) -> pd.DataFrame:
        """
        Обрабатывает DataFrame, регистрируя факты изменения сигнала (0→1 и 1→0)
        и расширяя зону вокруг изменений на заданное число отсчётов.

        Сигнал для обнаружения изменений определяется по следующей логике:
        1. Если столбец "rPDR PS" существует и содержит не только NaN, используется он (1 если значение == 1, иначе 0).
        2. Иначе, если столбцы "rPDR A", "rPDR B", "rPDR C" существуют, 
           сигнал равен 1 только если все три столбца равны 1, иначе 0.
        3. Иначе (если ни один из источников сигнала не доступен), сигнал считается равным 0 для всех строк.

        Аргументы:
            buses_df (pd.DataFrame): Исходный DataFrame с данными. Должен содержать столбец 'file_name'.
            samples_before (int): Количество строк до точки изменения, которые надо включить.
            samples_after (int): Количество строк после точки изменения, которые надо включить.

        Возвращает:
            pd.DataFrame: Обработанный DataFrame с выделенными зонами и новым столбцом change_event.
                          Строки из разных исходных файлов могут быть перемешаны, но сгруппированы по событиям.
        """
        dataset_df = pd.DataFrame()
        
        # Define signal column names
        primary_signal_col = "rPDR PS"
        composite_signal_cols = ["rPDR A", "rPDR B", "rPDR C"]

        # Process each file (group) separately
        for file_name, bus_df in buses_df.groupby("file_name"):
            # Reset index for correct positional access and slicing
            bus_df = bus_df.reset_index(drop=True)
            
            signal_calculated = False
            
            # --- Determine the signal source ---
            # 1. Try primary signal "rPDR PS"
            if primary_signal_col in bus_df.columns and bus_df[primary_signal_col].notna().any():
                # Use primary signal if it exists and is not all NaN
                # Ensure comparison handles potential NaNs gracefully (NaN == 1 is False)
                bus_df["signal"] = (bus_df[primary_signal_col] == 1)
                signal_calculated = True
                # print(f"File {file_name}: Using primary signal '{primary_signal_col}'.") # Optional: for debugging

            # 2. If primary signal not used, try composite signals
            if not signal_calculated:
                # Check if all required composite columns exist
                if all(col in bus_df.columns for col in composite_signal_cols):
                    # Check if there's *any* non-NaN data in these columns before calculating
                    if bus_df[composite_signal_cols].notna().any().any():
                        # Calculate composite signal: 1 only if ALL are 1
                        bus_df[primary_signal_col] = (bus_df[composite_signal_cols] == 1).all(axis=1)
                        bus_df["signal"] = (bus_df[primary_signal_col] == 1)
                        signal_calculated = True
                        # print(f"File {file_name}: Using composite signals {composite_signal_cols}.") # Optional: for debugging
                    else:
                         # Composite columns exist but are all NaN
                         # print(f"File {file_name}: Composite columns {composite_signal_cols} exist but are all NaN. Signal set to False.") # Optional: for debugging
                         pass # signal_calculated remains False, will default to False below

                else:
                    # Not all composite columns are present
                    missing_cols = [col for col in composite_signal_cols if col not in bus_df.columns]
                    # print(f"File {file_name}: Primary signal '{primary_signal_col}' not usable. Composite columns missing: {missing_cols}. Signal set to False.") # Optional: for debugging
                    pass # signal_calculated remains False, will default to False below

            # 3. Default to False if no signal source was found/usable
            if not signal_calculated:
                 # print(f"File {file_name}: No usable signal source found. Skipping this file.") # Optional: for debugging
                 continue 

            # --- Continue with the original logic using the calculated 'signal' ---

            # Determine the previous signal state to detect changes
            bus_df["prev_signal"] = bus_df["signal"].shift(1).fillna(False)

            # Register changes in 'change_event': "0-1" or "1-0"
            bus_df["change_event"] = np.nan # Initialize with NaN instead of None for better pandas compatibility
            bus_df.loc[(bus_df["prev_signal"] == False) & (bus_df["signal"] == True), "change_event"] = "0-1"
            bus_df.loc[(bus_df["prev_signal"] == True) & (bus_df["signal"] == False), "change_event"] = "1-0"

            # Initialize column to mark rows to keep
            bus_df["is_save"] = False

            # Find indices where a change occurred (ignoring potential NaN in change_event)
            change_indices = bus_df.index[bus_df["change_event"].notna()]

            # Expand the area around each change event
            for idx in change_indices:
                 # Check idx > 0 is implicitly handled by shift(1) for change_event detection
                 start_idx = max(0, idx - samples_before)
                 # end_idx needs to be inclusive for .loc, so +1 compared to slicing
                 end_idx = min(len(bus_df) - 1, idx + samples_after) # Adjust end index for .loc
                 bus_df.loc[start_idx:end_idx, "is_save"] = True

            # Filter rows marked for saving
            truncated_dataset = bus_df[bus_df["is_save"]].copy() # Use .copy() to avoid SettingWithCopyWarning

            # Assign event numbers within the file_name for saved rows
            if not truncated_dataset.empty:
                 event_number = 0
                 # Identify the start of each contiguous block of 'is_save' == True
                 # A block starts if 'is_save' is True and the previous was False (or it's the first row)
                 is_start_of_event = truncated_dataset.index.to_series().diff().fillna(1) != 1
                 event_groups = is_start_of_event.cumsum()
                 
                 # Efficiently apply event numbers using the calculated groups
                 truncated_dataset['file_name'] = truncated_dataset['file_name'] + " _event N" + event_groups.astype(str)

            # Handle case where no events were found in this file_df
            if len(truncated_dataset) == 0 and len(bus_df) > 0: # Check if bus_df wasn't empty
                if len(bus_df) > samples_before + samples_after:
                    middle = len(bus_df) // 2
                    # Use iloc for slicing by position after reset_index
                    truncated_dataset = bus_df.iloc[max(0, middle - samples_before) : min(len(bus_df), middle + samples_after + 1)].copy()
                else:
                    truncated_dataset = bus_df.copy()
                # Ensure the columns added during processing are present even if no events were found, set to default values
                if "signal" not in truncated_dataset.columns: truncated_dataset["signal"] = False # Or appropriate default
                if "prev_signal" not in truncated_dataset.columns: truncated_dataset["prev_signal"] = False
                if "change_event" not in truncated_dataset.columns: truncated_dataset["change_event"] = np.nan


            # Drop temporary columns not needed in the final DataFrame
            # Check if columns exist before dropping to avoid errors if no events were found and columns weren't added
            cols_to_drop = ["is_save", "prev_signal", "signal", "rPDR A", "rPDR B", "rPDR C"]
            existing_cols_to_drop = [col for col in cols_to_drop if col in truncated_dataset.columns]
            if existing_cols_to_drop:
                 truncated_dataset = truncated_dataset.drop(columns=existing_cols_to_drop)

            # Append the processed data for this file to the overall dataset
            # Use ignore_index=True if you want a clean 0..N index in the final result
            # Set ignore_index=False if you want to preserve original indices (might be less useful here)
            dataset_df = pd.concat([dataset_df, truncated_dataset], axis=0, ignore_index=True) 

        return dataset_df
    
    def get_simple_dataset(self, dataset_df: pd.DataFrame, csv_name='dataset_simpl.csv'):
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
        
        dataset_df["emerg_evnt"] = dataset_df[list(ml_emerg_evnt)].apply(lambda x: 1 if x.any() else "", axis=1)
        dataset_df["abnorm_evnt"] = dataset_df[list(ml_abnorm_evnt)].apply(lambda x: 1 if x.any() else "", axis=1)
        dataset_df.loc[dataset_df['emerg_evnt'] == 1, 'abnorm_evnt'] = ""
        dataset_df["emerg_evnt"] = dataset_df[list(ml_emerg_evnt)].apply(lambda x: 1 if x.any() else "", axis=1)
        dataset_df["normal"] = dataset_df[list(ml_all)].apply(lambda x: "" if 1 in x.values else 1, axis=1)
        dataset_df["no_event"] = dataset_df[["abnorm_evnt", 'emerg_evnt']].apply(lambda x: "" if 1 in x.values else 1, axis=1)
        
        # Удаляем сигналы ML
        dataset_df = dataset_df.drop(columns=ml_all)
        
        dataset_df.to_csv(self.csv_path + csv_name, index=False)

    def structure_columns(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
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
        existing_columns = [col for col in desired_order if col in list(dataset_df.columns)]

        # Переупорядочиваем столбцы DataFrame на основе существующих столбцов
        dataset_df = dataset_df[existing_columns]
        
        return dataset_df
