import comtrade
import json
import os

class ReadComtrade:
    def __init__(self):
        """Initialize ReadComtrade class."""

        self.unread_files = set()

        # Получаем директорию текущего файла (comtrade_processing.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Формируем полные пути к JSON файлам
        analog_path = os.path.join(current_dir, "dict_analog_names.json")
        discrete_path = os.path.join(current_dir, "dict_discrete_names.json")

        try:
            # Загружаем данные из JSON
            with open(analog_path, "r") as file:
                self.analog_names = json.load(file)
            with open(discrete_path, "r") as file:
                self.discrete_names = json.load(file)
        except FileNotFoundError as e:
            print(f"Error: Could not find JSON files: {e}")
            # Создаем пустые словари в случае ошибки
            self.analog_names = {}
            self.discrete_names = {}

    def read_comtrade(self, file_name):
        """
        Load and read comtrade files contents.

        Args:
            file_name (str): The name of comtrade file.

        Returns:
            tuple:
            - raw_date (Comtrade): The raw comtrade
            - raw_df (pandas.DataFrame): DataFrame of raw comtrade file.
        """
        try:
            raw_date = comtrade.load(file_name)
            raw_df = raw_date.to_dataframe()
            return raw_date, raw_df
        except Exception as ex:
            self.unread_files.add((file_name, ex))
            return None, None

    def get_bus_names(self, analog=True, discrete=False):
        """
        This function makes a dict of analog and discrete names
        for each bus.

        Args:
            analog (bool): True - include analog names.
            discrete (bool): True - include discrete names.

        Returns:
            dict: dict of analog and discrete names for each bus.
        """
        bus_names = dict()
        if analog and hasattr(self, 'analog_names'):
            for bus in self.analog_names.keys():
                bus_names[bus] = set()
                for v in self.analog_names[bus].values():
                    bus_names[bus] = bus_names[bus].union(v)
        if discrete and hasattr(self, 'discrete_names'):
            for bus in self.discrete_names.keys():
                if bus not in bus_names:
                    bus_names[bus] = set()
                for v in self.discrete_names[bus].values():
                    bus_names[bus] = bus_names[bus].union(v)
        return bus_names

    def get_all_names(self):
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

    def get_ml_signals(self, i_bus, use_operational_switching=True, use_abnormal_event=True, use_emergency_event=True):
        """
        This function returns a set of ML signals for a given bus.

        Args:
            i_bus (str): The bus number.
            use_operational_switching (bool): Include operational switching signals.
            use_abnormal_event (bool): Include abnormal event signals.
            use_emergency_event (bool): Include emergency event signals.

        Returns:
            set: A set of ML signals for the given bus.
        """
        # FIXME: rewrite so that it is recorded at the very beginning and counted 1 time, and not at every request
        ml_operational_switching = {
            # --- Working switching ---
            f'MLsignal_{i_bus}_1',  # Working switching, without specification
            f'MLsignal_{i_bus}_1_1',  # Operational activation, without specification
            f'MLsignal_{i_bus}_1_1_1',  # Operating start-up, engine start-up
            f'MLsignal_{i_bus}_1_2',  # Operational shutdown, without specification
        }

        ml_abnormal_event = {
            # --- Abnormal events
            f'MLsignal_{i_bus}_2',  # Anomaly, without clarification
            f'MLsignal_{i_bus}_2_1',  # Single phase-to-ground fault, without specification
            f'MLsignal_{i_bus}_2_1_1',  # Sustainable single phase-to-ground fault
            f'MLsignal_{i_bus}_2_1_2',  # Steady attenuating single phase-to-ground fault, with rare breakouts
            f'MLsignal_{i_bus}_2_1_3',  # Arc intermittent single phase-to-ground fault
            f'MLsignal_{i_bus}_2_2',  # Damping fluctuations from emergency processes
            f'MLsignal_{i_bus}_2_3',  # Voltage drawdown
            f'MLsignal_{i_bus}_2_3_1',  # Voltage drawdown when starting the engine
            f'MLsignal_{i_bus}_2_4',  # Current fluctuations, without specification
            f'MLsignal_{i_bus}_2_4_1',  # Current fluctuations when starting the engine
            f'MLsignal_{i_bus}_2_4_2',  # Current fluctuations from frequency-driven motors
        }

        ml_emergency_event = {
            # --- Emergency events ----
            f'MLsignal_{i_bus}_3',  # Emergency events, without clarification
            f'MLsignal_{i_bus}_3_1',  # An accident due to incorrect operation of the device, without clarification
            f'MLsignal_{i_bus}_3_2',  # Terminal malfunction
            f'MLsignal_{i_bus}_3_3'  # Two-phase earth fault
        }

        ml_signals = set()
        if use_operational_switching:
            ml_signals.update(ml_operational_switching)
        if use_abnormal_event:
            ml_signals.update(ml_abnormal_event)
        if use_emergency_event:
            ml_signals.update(ml_emergency_event)

        return ml_signals