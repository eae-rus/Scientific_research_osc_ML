import pandas as pd
import os
import json
import comtrade


class RawToCSV():
    """
    This class implemented to convert raw comtrade files to csv file.
    """
    def __init__(self, raw_path='raw_data/', csv_path=''):
        self.raw_path = raw_path
        self.csv_path = raw_path
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
        raw_df = self.rename_raw_columns(raw_df)
        for bus, cols in self.buses_names.items():
            cols = raw_cols.intersection(cols)
            if bus[-1] == '1':
                ml = {'MLsignal_1_1_1',
                      'MLsignal_1_1_2',
                      'MLsignal_1_1',
                      'MLsignal_1_2',
                      'MLsignal_1_2_1_1',
                      'MLsignal_1_2_1_2',
                      'MLsignal_1_2_1_3',
                      'MLsignal_1_3',
                      'MLsignal_1_3_1',
                      'MLsignal_1_3_2',
                      'MLsignal_1_3_3',
                      'MLsignal_12_1_1',
                      'MLsignal_12_1_2',
                      'MLsignal_12_2_1_1',
                      'MLsignal_12_3'}
                raw_ml = raw_cols.intersection(ml)
                cols = cols.union(raw_ml)
            if bus[-1] == '2':
                ml = {'MLsignal_2_1_1',
                      'MLsignal_2_1_2',
                      'MLsignal_2_1',
                      'MLsignal_2_2_1_1',
                      'MLsignal_2_2_1_2',
                      'MLsignal_2_2_1_3',
                      'MLsignal_2_3',
                      'MLsignal_2_3_2',
                      'MLsignal_2_3_3',
                      'MLsignal_12_1_1',
                      'MLsignal_12_1_2',
                      'MLsignal_12_2_1_1',
                      'MLsignal_12_3'}
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
        bus_columns_to_rename = {'MLsignal_1_1_1': 'ML_1_1',
                                 'MLsignal_2_1_1': 'ML_1_1',
                                 'MLsignal_12_1_1': 'ML_12_1_1',
                                 'MLsignal_1_1_2': 'ML_1_2',
                                 'MLsignal_2_1_2': 'ML_1_2',
                                 'MLsignal_12_1_2': 'ML_12_1_2',
                                 'MLsignal_1_1': 'ML_1',
                                 'MLsignal_2_1': 'ML_1',
                                 'MLsignal_1_2': 'ML_2',
                                 'MLsignal_1_2_1_1': 'ML_2_1_1',
                                 'MLsignal_2_2_1_1': 'ML_2_1_1',
                                 'MLsignal_12_2_1_1': 'ML_12_2_1_1',
                                 'MLsignal_1_2_1_2': 'ML_2_1_2',
                                 'MLsignal_2_2_1_2': 'ML_2_1_2',
                                 'MLsignal_1_2_1_3': 'ML_2_1_3',
                                 'MLsignal_2_2_1_3': 'ML_2_1_3',
                                 'MLsignal_1_3': 'ML_3',
                                 'MLsignal_2_3': 'ML_3',
                                 'MLsignal_12_3': 'ML_12_3',
                                 'MLsignal_1_3_1': 'ML_3_1',
                                 'MLsignal_1_3_2': 'ML_3_2',
                                 'MLsignal_1_3_3': 'ML_3_3',
                                 'MLsignal_2_3_2': 'ML_3_2',
                                 'MLsignal_2_3_3': 'ML_3_3',
                                 'I | Bus-1 | phase: A': 'IA',
                                 'I | Bus-2 | phase: A': 'IA',
                                 'I | Bus-1 | phase: C': 'IC',
                                 'I | Bus-2 | phase: C': 'IC',
                                 'U | BusBar-1 | phase: A': 'UA',
                                 'U | BusBar-2 | phase: A': 'UA',
                                 'U | BusBar-1 | phase: B': 'UB',
                                 'U | BusBar-2 | phase: B': 'UB',
                                 'U | BusBar-1 | phase: C': 'UC',
                                 'U | BusBar-2 | phase: C': 'UC',
                                 'U | BusBar-1 | phase: N': 'UN',
                                 'U | BusBar-2 | phase: N': 'UN'}
        bus_df.rename(columns=bus_columns_to_rename, inplace=True)
        return bus_df

    def check_columns(self, raw_df):
        """check for unknown columns"""
        ml_signals = {'MLsignal_1_1_1',
                      'MLsignal_1_1_2',
                      'MLsignal_1_1',
                      'MLsignal_1_2',
                      'MLsignal_1_2_1_1',
                      'MLsignal_1_2_1_2',
                      'MLsignal_1_2_1_3',
                      'MLsignal_1_3',
                      'MLsignal_1_3_1',
                      'MLsignal_1_3_2',
                      'MLsignal_1_3_3',
                      'MLsignal_2_1_1',
                      'MLsignal_2_1_2',
                      'MLsignal_2_1',
                      'MLsignal_2_2_1_1',
                      'MLsignal_2_2_1_2',
                      'MLsignal_2_2_1_3',
                      'MLsignal_2_3',
                      'MLsignal_2_3_2',
                      'MLsignal_2_3_3',
                      'MLsignal_12_1_1',
                      'MLsignal_12_1_2',
                      'MLsignal_12_2_1_1',
                      'MLsignal_12_3'}
        all_names = self.all_names.union(ml_signals)
        columns = raw_df.columns
        for c in columns:
            if c not in all_names:
                raise NameError("Unknown column: " + c)
