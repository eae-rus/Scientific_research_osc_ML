import pandas as pd
import os
import json
import comtrade # comtrade 0.1.2


class RawToCSV():
    """
    This class implemented to convert raw comtrade files to csv file.
    """
    def __init__(self, raw_path='raw_data/', csv_path='', uses_buses = ['1', '2', '12']):
        if not os.path.exists(raw_path):
            raise FileNotFoundError("Path for raw files does not exist")
        with open("dict_analog_names.json", "r") as file:
            analog_names = json.load(file)
        with open("dict_discrete_names.json", "r") as file:
            discrete_names = json.load(file)
        self.analog_names = analog_names
        self.discrete_names = discrete_names
        self.buses_names = self.get_bus_names()
        self.all_names = self.get_all_names()
        self.raw_path = raw_path
        self.csv_path = csv_path
        self.unread_files = set()
        self.uses_buses = uses_buses # 12 - intersectional, is not taken into account in any way, when adding it, it is necessary to correct the discretionary check
        # while everything is saved, think about how to set them more conveniently
        # TODO: by uses variables - add truncation of signals from arrays.
        self.uses_CT_B, self.uses_CT_zero = True, True
        self.uses_VT_ph, self.uses_VT_iph, self.uses_VT_zero  = True, True, True
        self.use_VT_CL, self.use_VT_BB = True, True
        # TODO: Add variables for combining accident levels (ML signals)

    def create_csv(self, csv_name='datset.csv'):
        """
        This function DataFrame and save csv file from raw comtrade data.

        Args:
            csv_name (str): The name of csv file.
        """
        dataset_df = pd.DataFrame()
        raw_files = sorted([file for file in os.listdir(self.raw_path)
                            if 'cfg' in file])
        for file in raw_files:
            raw_df = self.read_comtrade(self.raw_path + file)
            self.check_columns(raw_df)
            if not raw_df.empty:
                raw_df = self.rename_raw_columns(raw_df)
                raw_df = raw_df.reset_index()
                buses_df = self.split_buses(raw_df, file)
                dataset_df = pd.concat([dataset_df, buses_df],
                                       axis=0, ignore_index=False)
        dataset_df.to_csv(self.csv_path + csv_name, index_label='time')
        return dataset_df

    def read_comtrade(self, file_name):
        """
        This function load the comtrade file to pandas DataFrame.

        Args:
            file_name (str): The name of comtrade file.

        Returns:
            pandas.DataFrame: DataFrame of raw comtrade file.
        """
        raw_df = None
        try:
            raw_df = comtrade.load_as_dataframe(file_name)
        except Exception as ex:
            self.unread_files.add((file_name, ex))
        return raw_df

    def split_buses(self, raw_df, file_name):
        """Implemented for bus 1 and bus 2 only"""
        buses_df = pd.DataFrame()
        buses_cols = dict()
        raw_cols = set(raw_df.columns)
        for bus, cols in self.buses_names.items():
            cols = raw_cols.intersection(cols)
            for i_bus in self.uses_buses:
                if bus[-1] == i_bus or bus[-2] == i_bus:
                    ml = self.get_ml_signals(i_bus)
                    raw_ml = raw_cols.intersection(ml)
                    cols = cols.union(raw_ml)
            if cols:
                buses_cols[bus] = cols
        for bus, columns in buses_cols.items():
            bus_df = raw_df.loc[:, list(columns)]
            bus_df.insert(0, 'file_name',  [file_name + bus] * bus_df.shape[0])
            bus_df = self.rename_bus_columns(bus_df)
            buses_df = pd.concat([buses_df, bus_df], axis=0,
                                 ignore_index=False)    
        return buses_df

    def get_bus_names(self, discrete=False):
        """
        This function makes a dict of analog and discrete names
        for each bus.

        Args:
            discrete (bool): False - only analog names.

        Returns:
            dict: dict of analog and discrete names for each bus.
        """
        bus_names = dict()
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
        This function makes a set of all analog and discrete names.

        Returns:
            set: set of all analog and discrete names.
        """
        all_names = set()
        buses_names = self.get_bus_names(discrete=True)
        for bus, names in buses_names.items():
            all_names = all_names.union(names)
        return all_names

    def rename_raw_columns(self, raw_df):
        # TODO: probably outdated in the current implementation. Check if it will be called at least once.
        """
        This function renames columns in raw DataFrame.

        Args:
            raw_df (pandas.DataFrame): DataFrame of raw comtrade file.

        Returns:
            pandas.DataFrame: Dataframe with renamed columns.
        """
        raw_columns_to_rename = {'Mlsignal_2_2_1_1': 'MLsignal_2_2_1_1',
                                 'Mlsignal_12_2_1_1': 'MLsignal_12_2_1_1',
                                 'Mlsignal_2_3': 'MLsignal_2_3'}
        raw_df.rename(columns=raw_columns_to_rename, inplace=True)
        return raw_df

    def rename_bus_columns(self, bus_df):
        """
        This function renames columns in bus DataFrame.

        Args:
            buses_df (pandas.DataFrame): DataFrame with one bus.

        Returns:
            pandas.DataFrame: Dataframe with renamed columns.
        """
        bus_columns_to_rename = {}

        # Generate renaming for ML signals
        for i_bus in self.uses_buses:
            ml_signals = self.get_ml_signals(i_bus)
            for signal in ml_signals:
                new_name = signal.replace(f'MLsignal_{i_bus}_', 'ML_')
                bus_columns_to_rename[signal] = new_name
        
        # Generate renaming for analog signals
        for bus, names in self.buses_names.items():
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
                        
        # TODO: signals I_raw, U_raw, I|dif-1, I | braking-1 are not taken into account

        bus_df.rename(columns=bus_columns_to_rename, inplace=True)
        return bus_df

    def check_columns(self, raw_df):
        """check for unknown columns"""
        ml_signals = set()
        for i_bus in self.uses_buses:
            ml_signals.update(self.get_ml_signals(i_bus))

        all_names = self.all_names.union(ml_signals)
        columns = raw_df.columns
        for c in columns:
            if c not in all_names:
                raise NameError("Unknown column: " + c)

    def get_ml_signals(self, i_bus, use_operational_switching=True, use_abnormal_event=True, use_emergency_even=True):
        """
        This function returns a set of ML signals for a given bus.

        Args:
            i_bus (str): The bus number.
            use_operational_switching (bool): Include operational switching signals.
            use_abnormal_event (bool): Include abnormal event signals.
            use_emergency_even (bool): Include emergency event signals.

        Returns:
            set: A set of ML signals for the given bus.
        """
        # FIXME: rewrite so that it is recorded at the very beginning and counted 1 time, and not at every request
        ml_signals = set()

        if use_operational_switching:
            ml_signals.update({
                #--- Working switching ---
                f'MLsignal_{i_bus}_1',      # Working switching, without specification
                f'MLsignal_{i_bus}_1_1',    # Operational activation, without specification
                f'MLsignal_{i_bus}_1_1_1',  # Operating start-up, engine start-up
                f'MLsignal_{i_bus}_1_2',    # Operational shutdown, without specification
            })

        if use_abnormal_event:
            ml_signals.update({
                # --- Abnormal events
                f'MLsignal_{i_bus}_2',      # Аномалия, без уточнения
                f'MLsignal_{i_bus}_2_1',    # Однофазное замыкание на землю (ОЗЗ), без уточнения
                f'MLsignal_{i_bus}_2_1_1',  # Устойчивое ОЗЗ
                f'MLsignal_{i_bus}_2_1_2',  # Устойчивое затухающее ОЗЗ, с редкими пробоями
                f'MLsignal_{i_bus}_2_1_3',  # Дуговое перемежающее однофазное замыкание на землю (ДПОЗЗ)
                f'MLsignal_{i_bus}_2_2',    # Затухающие колебания от аварийных процессов
                f'MLsignal_{i_bus}_2_3',    # Просадка напряжения
                f'MLsignal_{i_bus}_2_3_1',  # Просадка напряжения при пуске двигателя
                f'MLsignal_{i_bus}_2_4',    # Колебания тока, без уточнения
                f'MLsignal_{i_bus}_2_4_1',  # Колебания тока при пуске двигателя
                f'MLsignal_{i_bus}_2_4_2',  # Колебания тока, от двигателей с частотным приводом
                
                f'MLsignal_{i_bus}_2',      # Anomaly, without clarification
                f'MLsignal_{i_bus}_2_1',    # Single phase-to-ground fault, without specification
                f'MLsignal_{i_bus}_2_1_1',  # Sustainable single phase-to-ground fault
                f'MLsignal_{i_bus}_2_1_2',  # Steady attenuating single phase-to-ground fault, with rare breakouts
                f'MLsignal_{i_bus}_2_1_3',  # Arc intermittent single phase-to-ground fault
                f'MLsignal_{i_bus}_2_2',    # Damping fluctuations from emergency processes
                f'MLsignal_{i_bus}_2_3',    # Voltage drawdown
                f'MLsignal_{i_bus}_2_3_1',  # Voltage drawdown when starting the engine
                f'MLsignal_{i_bus}_2_4',    # Current fluctuations, without specification
                f'MLsignal_{i_bus}_2_4_1',  # Current fluctuations when starting the engine
                f'MLsignal_{i_bus}_2_4_2',  # Current fluctuations from frequency-driven motors
            })

        if use_emergency_even:
            ml_signals.update({
                # --- Emergency events ----
                f'MLsignal_{i_bus}_3',      # Emergency events, without clarification
                f'MLsignal_{i_bus}_3_1',    # An accident due to incorrect operation of the device, without clarification
                f'MLsignal_{i_bus}_3_2',    # Terminal malfunction
                f'MLsignal_{i_bus}_3_3'     # Two-phase earth fault
            })

        return ml_signals