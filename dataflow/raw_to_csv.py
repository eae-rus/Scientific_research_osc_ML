from dataflow.comtrade_processing import ReadComtrade
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

class RawToCSV():
    """
    This class implemented to convert raw comtrade files to csv file.
    """
    def __init__(self, raw_path=None, csv_path=None, uses_buses=['1', '2', '12']):
        # Относительный путь от директории скрипта к директории с данными
        # FIXME: You need to add a workaround to the problem when these files are missing. Because they have to be taken from somewhere.
        default_raw_path = os.path.join('..', '..', 'data', 'raw')
        default_csv_path = os.path.join('..', '..', 'data', 'csv')

        self.raw_path = raw_path if raw_path is not None else default_raw_path
        self.csv_path = csv_path if csv_path is not None else default_csv_path

        # Добавляем разделитель в конец пути, если его нет
        if not self.raw_path.endswith(os.path.sep):
            self.raw_path += os.path.sep
        if not self.csv_path.endswith(os.path.sep):
            self.csv_path += os.path.sep

        # Проверка существования директории с сырыми данными
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"Path for raw files does not exist: {self.raw_path}")

        # Создание директории для CSV, если она не существует
        os.makedirs(self.csv_path, exist_ok=True)

        # Инициализация компонента для чтения Comtrade файлов
        self.readComtrade = ReadComtrade()

        # Получение словарей имен сигналов из readComtrade
        self.analog_names_dict = self.readComtrade.get_bus_names(analog=True, discrete=False)
        self.discrete_names_dict = self.readComtrade.get_bus_names(analog=False, discrete=True)
        self.all_names = self.readComtrade.get_all_names()

        # Настройки для обработки данных
        self.uses_buses = uses_buses  # 12 - intersectional, is not taken into account in any way
        self.unread_files = set()

        # Настройки для аналоговых и дискретных сигналов
        # TODO: by uses variables - add truncation of signals from arrays.
        self.uses_CT_B, self.uses_CT_zero = True, True
        self.uses_VT_ph, self.uses_VT_iph, self.uses_VT_zero = True, True, True
        self.use_VT_CL, self.use_VT_BB = True, True
        # TODO: подумать о большей понятности и оптимальности, так как use_PDR нужна только сейчас, а требует проверки всех дискрет
        self.use_PDR = True

        # Параметры для обработки сигналов
        # TODO: Add variables for combining accident levels (ML signals)
        self.number_periods = 10  # TODO: The number of samples is being set now. Think about a time-to-date task, or something similar.

        # Инициализация списков ML сигналов
        self.ml_all, self.ml_opr_swch, self.ml_abnorm_evnt, self.ml_emerg_evnt = self.get_short_names_ml_signals()


    def create_csv(self, csv_name='dataset.csv', is_cut_out_area=False):
        """
        This function DataFrame and save csv file from raw comtrade data.

        Args:
            csv_name (str): The name of csv file.
        """
        dataset_df = pd.DataFrame()
        raw_files = sorted([file for file in os.listdir(self.raw_path)
                            if 'cfg' in file])
        with tqdm(total=len(raw_files), desc="Convert Comtrade to CSV") as pbar:
            for file in raw_files:
                # TODO: it is not rational to use two variables, but the necessary global data.
                # it will be necessary to think about optimizing this issue.
                raw_date, raw_df = self.readComtrade.read_comtrade(self.raw_path + file)

                # TODO: Добавить нормировку значений по аналогии с PDR и SPEF
                # Но с параметром выбора (чтобы можно было её и не осуществлять)

                # TODO: Add secondary/primary checks to the dataset reading and processing.
                self.check_columns(raw_df)
                if not raw_df.empty:
                    raw_df = raw_df.reset_index()
                    buses_df = self._split_buses(raw_df, file)

                    frequency = raw_date.cfg.frequency
                    samples_rate = raw_date.cfg.sample_rates[0][0]
                    number_samples = int(samples_rate / frequency)  # TODO: It won't always be whole, but it's rare.
                    samples_before, samples_after = number_samples * self.number_periods, number_samples * self.number_periods
                    if is_cut_out_area:
                        buses_df = self.cut_out_area(buses_df, samples_before, samples_after)
                        dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=False)
                    else:
                        dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=False)
                    pbar.update(1)

        # TODO: Think about organizing the processing itself differently and not adding a function.
        dataset_df = self.structure_columns(dataset_df)
        # if is_simple_csv:
        #     # We do not overwrite the original data, but only create another csv file
        #     self.get_simple_dataset(dataset_df.copy())

        dataset_df.to_csv(self.csv_path + csv_name, index=False)
        return dataset_df

    def check_columns(self, raw_df):
        """check for unknown columns"""
        ml_signals = set()
        for i_bus in self.uses_buses:
            ml_signals.update(self.readComtrade.get_ml_signals(i_bus))

        all_names = self.all_names.union(ml_signals)
        columns = raw_df.columns
        for c in columns:
            if c not in all_names:
                raise NameError("Unknown column: " + c)

    # def _create_one_df(self, file_path, file_name) -> pd.DataFrame:
    #     pass
    def _create_one_df(self, file_path, file_name) -> pd.DataFrame:
        """
        The function of converting a single Comtrade file to pd.DataFrame().

        Args:
            file_path (str): The path to comtrade file.
            file_name (str): The name of comtrade file.

        Returns:
            pd.DataFrame: The DataFrame of the comtrade file.
        """
        dataset_df = pd.DataFrame()
        _, raw_df = self.readComtrade.read_comtrade(file_path)
        self.check_columns(raw_df)
        if not raw_df.empty:
            raw_df = raw_df.reset_index()
            dataset_df = self._split_buses(raw_df, file_name)

        return dataset_df

    # def _split_buses(self, raw_df, file_name):
    #     pass
    def _split_buses(self, raw_df, file_name):
        """Implemented for bus 1 and bus 2 only"""
        # TODO: Подумать / учесть обработку случаев, когда имена сигналов одинаковы. Это недоработки стандартизации имён
        # Такого может попадаться часто. Для случаев "U | BusBar-1 | phase: A" и похожих, это часто раделение на
        # BusBar / CableLine.
        # Для токов - порой является разделением по секциям
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
                    ml_all = self.readComtrade.get_ml_signals(i_bus)
                    raw_ml = raw_cols.intersection(ml_all)
                    cols = cols.union(raw_ml)
            if cols:
                buses_cols[bus] = cols
        for bus, columns in buses_cols.items():
            bus_df = raw_df.loc[:, list(columns)]
            bus_df.insert(0, 'file_name', [file_name[:-4] + "_" + bus] * bus_df.shape[0])
            bus_df = self.rename_bus_columns(bus_df)
            buses_df = pd.concat([buses_df, bus_df], axis=0,
                                 ignore_index=False)
        return buses_df

    def _split_buses_for_PDR(self, raw_df, file_name):
        """Implemented for bus 1 and bus 2 only"""
        buses_df = pd.DataFrame()
        buses_cols = dict()
        raw_cols = set(raw_df.columns)
        for bus, cols in self.analog_names_dict.items():
            cols = raw_cols.intersection(cols)
            for i_bus in self.uses_buses:
                if bus[-1] == i_bus or bus[-2] == i_bus:
                    # TODO: подумать о том, что может и любой другой дискретный сигнал использоваться, а не только PDR
                    # и такой генератор (get_PDR_signals) - это временное решение в переборе аналоговых сигналов
                    ml_all = self.get_PDR_signals(i_bus)
                    raw_ml = raw_cols.intersection(ml_all)
                    cols = cols.union(raw_ml)
            if cols:
                buses_cols[bus] = cols

        for bus, columns in buses_cols.items():
            bus_df = raw_df.loc[:, list(columns)]
            bus_df.insert(0, 'file_name', [file_name[:-4] + "_" + bus] * bus_df.shape[0])
            bus_df = self.rename_bus_columns(bus_df, is_use_ML=False, is_use_discrete=True)
            buses_df = pd.concat([buses_df, bus_df], axis=0,
                                 ignore_index=False)
        return buses_df

    def rename_bus_columns(self, bus_df, is_use_ML=True, is_use_discrete=False):
        """
        This function renames columns in bus DataFrame.

        Args:
            buses_df (pandas.DataFrame): DataFrame with one bus.

        Returns:
            pandas.DataFrame: Dataframe with renamed columns.
        """
        bus_columns_to_rename = {}

        # Generate renaming for ML signals
        if is_use_ML:
            for i_bus in self.uses_buses:
                ml_signals = self.readComtrade.get_ml_signals(i_bus)
                for signal in ml_signals:
                    new_name = signal.replace(f'MLsignal_{i_bus}_', 'ML_')
                    bus_columns_to_rename[signal] = new_name

        # Generate renaming for analog signals
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

        # Generate renaming for discret signals
        if is_use_discrete:
            for bus, names in self.discrete_names_dict.items():
                for name in names:
                    if self.use_PDR and 'PDR | Bus' in name:
                        phase = name.split(': ')[-1]
                        bus_columns_to_rename[name] = f'PDR {phase}'

                        # TODO: signals I_raw, U_raw, I|dif-1, I | braking-1 are not taken into account

        bus_df.rename(columns=bus_columns_to_rename, inplace=True)
        return bus_df

    def get_PDR_signals(self, i_bus):
        """
        This function returns a set of ML signals for a given bus.

        Args:
            i_bus (str): The bus number.

        Returns:
            set: A set of PDR for ML signals for the given bus.
        """

        ml_signals = {
            #--- Working switching ---
            f'PDR | Bus-{i_bus} | phase: A',
            f'PDR | Bus-{i_bus} | phase: B',
            f'PDR | Bus-{i_bus} | phase: C',
            f'PDR | Bus-{i_bus} | phase: PS',
        }

        return ml_signals

    # def cut_out_area(self, buses_df: pd.DataFrame, samples_before: int, samples_after: int) -> pd.DataFrame:
    #      pass
    def cut_out_area(self, buses_df: pd.DataFrame, samples_before: int, samples_after: int) -> pd.DataFrame:
        """
        The function cuts off sections that do not contain ML signals, leaving before and after them at a given boundary.

        Args:
            buses_df (pd.DataFrame): DataFrame for processing
            samples_before (int): Number of samples to cut off before the first ML signal.
            samples_after (int): Number of samples to cut off after the last ML signal.

        Returns:
            pd.DataFrame: The DataFrame with cut-out sections.
        """
        dataset_df = pd.DataFrame()

        for _, bus_df in buses_df.groupby("file_name"):
            truncated_dataset = pd.DataFrame()
            # Reset the index before slicing
            bus_df = bus_df.reset_index(drop=True)

            bus_df["is_save"] = False
            filtered_column_names = [col for col in bus_df.columns if col in self.ml_all]

            # Identify rows with ML signals
            bus_df["is_save"] = bus_df[filtered_column_names].notna().any(axis=1) & (
                    bus_df[filtered_column_names] == 1).any(axis=1)

            # Получаем булевую маску "is_save" в виде массива numpy
            is_save_array = bus_df["is_save"].values

            # Forward fill: Extend "is_save" to cover sections before ML signals
            # Находим индексы, где начинаются последовательности True
            # (текущее значение True и предыдущее тоже True)
            consecutive_mask = np.logical_and(is_save_array[1:], is_save_array[:-1])
            consecutive_indices = np.where(consecutive_mask)[0] + 1  # +1 так как сравниваем с предыдущим

            # Расширяем "is_save" на samples_before точек перед каждой последовательностью
            for idx in consecutive_indices:
                start_idx = max(0, idx - samples_before)
                is_save_array[start_idx:idx] = True

            # Backward fill: Extend "is_save" to cover sections after ML signals
            # Находим индексы, где заканчиваются последовательности True
            # (текущее значение True и следующее тоже True)
            consecutive_mask_backward = np.logical_and(is_save_array[:-1], is_save_array[1:])
            consecutive_indices_backward = np.where(consecutive_mask_backward)[0]

            # Расширяем "is_save" на samples_after точек после каждой последовательности
            for idx in consecutive_indices_backward:
                end_idx = min(len(is_save_array) - 1, idx + samples_after + 1)
                is_save_array[idx + 1:end_idx] = True

            # Записываем обработанный массив обратно в DataFrame
            bus_df["is_save"] = is_save_array

            # Add event numbers to file names
            event_number = 0
            for index, row in bus_df.iterrows():
                if bus_df.loc[index, "is_save"]:
                    if (index == 0 or not bus_df.loc[index - 1, "is_save"]):
                        event_number += 1
                    bus_df.loc[index, 'file_name'] = bus_df.loc[index, 'file_name'] + " _event N" + str(event_number)

            truncated_dataset = bus_df[bus_df["is_save"]]
            # If the array is empty, then we take from the piece in the middle equal to the sum of the lengths before and after
            if len(truncated_dataset) == 0:
                if len(bus_df) > samples_before + samples_after:
                    middle = len(bus_df) // 2
                    truncated_dataset = bus_df.iloc[middle - samples_before:middle + samples_after + 1]
                else:
                    truncated_dataset = bus_df

            truncated_dataset = truncated_dataset.drop(columns=["is_save"])
            dataset_df = pd.concat([dataset_df, truncated_dataset], axis=0, ignore_index=False)

        return dataset_df

    def structure_columns(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
        """
        This function restructures the columns of the dataset for easier analysis.
        """
        # TODO: some names may be missing, so think about processing them.
        # Specify the desired order of columns
        desired_order = ["file_name"]
        desired_order.extend(self.get_short_names_ml_analog_signals())
        desired_order.extend(self.get_short_names_ml_signals()[0])
        desired_order.extend(["opr_swch", "abnorm_evnt", "emerg_evnt", "normal"])

        # Check if each column in the desired order exists in the DataFrame
        existing_columns = [col for col in desired_order if col in list(dataset_df.columns)]

        # Reorder the DataFrame columns based on the existing columns
        dataset_df = dataset_df[existing_columns]

        return dataset_df

    def get_short_names_ml_signals(self, use_operational_switching: bool = True, use_abnormal_event: bool = True,
                                   use_emergency_event: bool = True) -> list:
        """
        This function returns a set of short names ML signals for (without i_bus).

        Args:
            use_operational_switching (bool): Include operational switching signals.
            use_abnormal_event (bool): Include abnormal event signals.
            use_emergency_event (bool): Include emergency event signals.

        Returns:
            list: A list of ML signals for the given bus.
        """
        # FIXME: rewrite so that it is recorded at the very beginning and counted 1 time, and not at every request

        ml_operational_switching = [
            # --- Working switching ---
            'ML_1',  # Working switching, without specification
            'ML_1_1',  # Operational activation, without specification
            'ML_1_1_1',  # Operating start-up, engine start-up
            'ML_1_2'  # Operational shutdown, without specification
        ]
        ml_abnormal_event = [
            # --- Abnormal events
            'ML_2',  # Anomaly, without clarification
            'ML_2_1',  # Single phase-to-ground fault, without specification
            'ML_2_1_1',  # Sustainable single phase-to-ground fault
            'ML_2_1_2',  # Steady attenuating single phase-to-ground fault, with rare breakouts
            'ML_2_1_3',  # Arc intermittent single phase-to-ground fault
            'ML_2_2',  # Damping fluctuations from emergency processes
            'ML_2_3',  # Voltage drawdown
            'ML_2_3_1',  # Voltage drawdown when starting the engine
            'ML_2_4',  # Current fluctuations, without specification
            'ML_2_4_1',  # Current fluctuations when starting the engine
            'ML_2_4_2'  # Current fluctuations from frequency-driven motors
        ]

        ml_emergency_event = [
            # --- Emergency events ----
            'ML_3',  # Emergency events, without clarification
            'ML_3_1',  # An accident due to incorrect operation of the device, without clarification
            'ML_3_2',  # Terminal malfunction
            'ML_3_3'  # Two-phase earth fault
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
        This function returns a set of short names ML analog signals for (without i_bus).

        Args:

        Returns:
            list: A set of ML signals for the given bus.
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
        # TODO: signals I_raw, U_raw, I|dif-1, I | braking-1 are not taken into account

        ml_signals = []
        ml_signals.extend(ml_current)
        ml_signals.extend(ml_votage_BB)
        ml_signals.extend(ml_votage_CL)

        return ml_signals