import pandas as pd
import numpy as np
import os
import sys
import json
# import comtrade # REMOVE
from tqdm import tqdm
import pandas as pd
import numpy as np
import re # Already imported in original, ensure it's kept if used by helpers

# Add ROOT_DIR and sys.path.append if they are specific to this file's execution context
# For the subtask, assume imports will resolve if ROOT_DIR setup is handled project-wide.

# MODIFIED IMPORTS
from core.oscillogram import Oscillogram
from normalization.normalization import OscillogramNormalizer # RENAMED
# from dataflow.comtrade_processing import ReadComtrade # REMOVE

class OscillogramToCsvConverter(): # RENAMED from RawToCSV
    def __init__(self, normalizer: OscillogramNormalizer,
                 raw_path: str = 'raw_data/', csv_path: str = '',
                 uses_buses: list = None,
                 dict_analog_names_path="dict_analog_names.json",
                 dict_discrete_names_path="dict_discrete_names.json",
                 is_print_message: bool = False
                ):

        self.normalizer = normalizer
        self.raw_path = raw_path
        self.csv_path = csv_path
        self.uses_buses = uses_buses if uses_buses is not None else ['1', '2', '12']
        self.is_print_message = is_print_message

        self.analog_names = {}
        self.discrete_names = {}
        try:
            # Ensure paths are absolute or correctly relative if needed
            # For now, assume dict_analog_names_path and dict_discrete_names_path are accessible
            with open(dict_analog_names_path, "r", encoding='utf-8') as file: # Specify encoding
                self.analog_names = json.load(file)
        except FileNotFoundError:
            if self.is_print_message: print(f"Warning: Analog names JSON not found at {dict_analog_names_path}. Initializing to empty.")
        except json.JSONDecodeError:
            if self.is_print_message: print(f"Warning: Could not decode JSON from {dict_analog_names_path}. Initializing to empty.")

        try:
            with open(dict_discrete_names_path, "r", encoding='utf-8') as file: # Specify encoding
                self.discrete_names = json.load(file)
        except FileNotFoundError:
            if self.is_print_message: print(f"Warning: Discrete names JSON not found at {dict_discrete_names_path}. Initializing to empty.")
        except json.JSONDecodeError:
            if self.is_print_message: print(f"Warning: Could not decode JSON from {dict_discrete_names_path}. Initializing to empty.")

        self.unread_files = set()

        self.analog_names_dict = self.get_bus_names(analog=True, discrete=False)
        self.discrete_names_dict = self.get_bus_names(analog=False, discrete=True)
        self.all_names = self.get_all_names() # Based on loaded JSONs

        self.uses_CT_B, self.uses_CT_zero = True, True # Defaults from original
        self.uses_VT_ph, self.uses_VT_iph, self.uses_VT_zero  = True, True, True
        self.use_VT_CL, self.use_VT_BB = True, True
        self.use_PDR = True
        self.number_periods = 10
        self.ml_all, self.ml_opr_swch, self.ml_abnorm_evnt, self.ml_emerg_evnt  = self.get_short_names_ml_signals()

    def create_csv(self, csv_name='dataset.csv', is_cut_out_area = False, is_simple_csv = False):
        dataset_df = pd.DataFrame()
        if not os.path.isdir(self.raw_path):
            if self.is_print_message: print(f"Error: Raw path '{self.raw_path}' does not exist.")
            return dataset_df

        raw_files = sorted([f for f in os.listdir(self.raw_path) if f.lower().endswith('.cfg')])

        with tqdm(total=len(raw_files), desc="Convert Comtrade to CSV") as pbar:
            for file_cfg_name in raw_files:
                osc = None # Define osc here for broader scope if needed after try-except
                try:
                    cfg_full_path = os.path.join(self.raw_path, file_cfg_name)
                    osc = Oscillogram(cfg_full_path)
                    if osc.data_frame is None or osc.data_frame.empty:
                        self.unread_files.add(file_cfg_name)
                        pbar.update(1)
                        continue
                    raw_df = osc.data_frame
                except FileNotFoundError:
                    self.unread_files.add(file_cfg_name)
                    if self.is_print_message: print(f"File not found: {cfg_full_path}")
                    pbar.update(1)
                    continue
                except RuntimeError as e:
                    self.unread_files.add(file_cfg_name)
                    if self.is_print_message: print(f"Runtime error loading {cfg_full_path}: {e}")
                    pbar.update(1)
                    continue
                except Exception as e:
                    self.unread_files.add(file_cfg_name)
                    if self.is_print_message: print(f"Unexpected error loading {cfg_full_path}: {e}")
                    pbar.update(1)
                    continue
                
                # raw_df doesn't need reset_index if osc.data_frame 'time' isn't index.
                # Oscillogram.to_dataframe() currently makes 'time' a column.
                buses_df = self.split_buses(raw_df, file_cfg_name)

                if osc.frequency is None or osc.frequency == 0 or \
                   not osc.cfg.sample_rates or len(osc.cfg.sample_rates[0]) < 1 or osc.cfg.sample_rates[0][0] == 0:
                    if self.is_print_message: print(f"Warning: Invalid frequency ({osc.frequency}) or sample_rates for {file_cfg_name}. Skipping area cut/further processing dependent on it.")
                    dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=True)
                    pbar.update(1)
                    continue

                samples_per_period = int(osc.cfg.sample_rates[0][0] / osc.frequency)
                if samples_per_period <= 0:
                    if self.is_print_message: print(f"Warning: Invalid samples_per_period ({samples_per_period}) for {file_cfg_name}. Skipping area cut.")
                    dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=True)
                    pbar.update(1)
                    continue

                samples_before = samples_per_period * self.number_periods
                samples_after = samples_per_period * self.number_periods

                if is_cut_out_area and not buses_df.empty: # Ensure buses_df is not empty
                    buses_df = self.cut_out_area(buses_df, samples_before, samples_after)

                if not buses_df.empty: # Concat only if buses_df has data
                    dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=True)
                pbar.update(1)
        
        if not dataset_df.empty:
            dataset_df = self.structure_columns(dataset_df)
            if is_simple_csv:
                self.get_simple_dataset(dataset_df.copy(), csv_name=os.path.splitext(csv_name)[0] + "_simple.csv")

            output_path = os.path.join(self.csv_path, csv_name)
            # Ensure self.csv_path directory exists
            if self.csv_path and not os.path.exists(self.csv_path):
                os.makedirs(self.csv_path, exist_ok=True)
            dataset_df.to_csv(output_path, index=False)
            if self.is_print_message: print(f"CSV created at {output_path}")
        elif self.is_print_message:
            print("No data processed, CSV not created.")

        if self.unread_files and self.is_print_message:
            print(f"Files that could not be read or were skipped: {self.unread_files}")
        return dataset_df

    def create_csv_for_PDR(self, csv_name='dataset_pdr.csv',
                           signal_check_results_path='signal_check_results.csv',
                           is_check_PDR = True, is_cut_out_area = False,
                           yes_prase_norm = "YES",
                           yes_prase_sigcheck = "YES"): # Removed is_print_error, use self.is_print_message

        signal_check_df = pd.DataFrame()
        if is_check_PDR:
            if not os.path.exists(signal_check_results_path):
                if self.is_print_message: print(f"Error: Path for signal check results does not exist: {signal_check_results_path}")
                return pd.DataFrame() # Return empty if critical file missing
            try:
                signal_check_df = pd.read_csv(signal_check_results_path)
            except Exception as e:
                if self.is_print_message: print(f"Error reading signal check results {signal_check_results_path}: {e}")
                return pd.DataFrame()

        dataset_df = pd.DataFrame()
        if not os.path.isdir(self.raw_path):
            if self.is_print_message: print(f"Error: Raw path '{self.raw_path}' does not exist.")
            return dataset_df

        raw_files = sorted([f for f in os.listdir(self.raw_path) if f.lower().endswith('.cfg')])
        number_ocs_found = 0

        with tqdm(total=len(raw_files), desc="Convert Comtrade to CSV for PDR") as pbar:
            for file_cfg_name in raw_files:
                filename_without_ext = file_cfg_name[:-4]
                osc = None

                if is_check_PDR:
                    check_result_row = signal_check_df[signal_check_df['filename'] == filename_without_ext]
                    if check_result_row.empty or not (str(check_result_row['contains_required_signals'].iloc[0]).upper()  == yes_prase_sigcheck.upper()):
                        pbar.update(1)
                        continue
                
                try:
                    cfg_full_path = os.path.join(self.raw_path, file_cfg_name)
                    osc = Oscillogram(cfg_full_path)
                    if osc.data_frame is None or osc.data_frame.empty:
                        self.unread_files.add(file_cfg_name)
                        pbar.update(1)
                        continue
                    raw_df = osc.data_frame
                except Exception as e: # Catch-all for Oscillogram loading
                    self.unread_files.add(file_cfg_name)
                    if self.is_print_message: print(f"Error loading {file_cfg_name} with Oscillogram: {e}")
                    pbar.update(1)
                    continue

                # Normalization using self.normalizer
                normalized_df = self.normalizer.normalize_bus_signals(raw_df.copy(), filename_without_ext,
                                                                    yes_prase=yes_prase_norm,
                                                                    is_print_error=self.is_print_message)
                if normalized_df is None:
                    if self.is_print_message: print(f"Warning: Normalization failed for {filename_without_ext}.")
                    pbar.update(1)
                    continue
                    
                # Use normalized_df for further processing
                # Assuming osc.data_frame is not indexed by 'time' from Oscillogram
                buses_df = self.split_buses_for_PDR(normalized_df, file_cfg_name)

                if buses_df is None or buses_df.empty:
                    if self.is_print_message: print(f"Warning: Bus splitting for PDR failed for {filename_without_ext}.")
                    pbar.update(1)
                    continue

                if osc.frequency is None or osc.frequency == 0 or \
                   not osc.cfg.sample_rates or len(osc.cfg.sample_rates[0]) < 1 or osc.cfg.sample_rates[0][0] == 0:
                    if self.is_print_message: print(f"Warning: Invalid frequency/sample_rate for {file_cfg_name}. Skipping area cut.")
                    dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=True)
                    number_ocs_found += 1
                    pbar.update(1)
                    continue

                samples_per_period = int(osc.cfg.sample_rates[0][0] / osc.frequency)
                if samples_per_period <= 0 :
                     if self.is_print_message: print(f"Warning: Invalid samples_per_period for {file_cfg_name}. Skipping area cut.")
                     dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=True)
                     number_ocs_found += 1
                     pbar.update(1)
                     continue

                samples_before = samples_per_period * self.number_periods
                samples_after = samples_per_period * self.number_periods

                if is_cut_out_area: # is_check_PDR is implicitly true if we are here due to earlier check
                    buses_df = self.cut_out_area_for_PDR(buses_df, samples_before, samples_after)

                if not buses_df.empty:
                    dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=True)
                    number_ocs_found += 1
                pbar.update(1)
        
        if self.is_print_message: print(f"Number of PDR-relevant oscillograms processed = {number_ocs_found}")
        if not dataset_df.empty:
            output_path = os.path.join(self.csv_path, csv_name)
            if self.csv_path and not os.path.exists(self.csv_path): os.makedirs(self.csv_path, exist_ok=True)
            dataset_df.to_csv(output_path, index=False)
            if self.is_print_message: print(f"PDR dataset saved to {output_path}")
        elif self.is_print_message:
            print("No PDR data processed, CSV not created.")
        return dataset_df

    def create_csv_for_SPEF(self, csv_name='dataset_spef.csv',
                            spef_results_path='find_oscillograms_with_spef.csv', # Changed name
                            is_cut_out_area = False,
                            yes_prase_norm = "YES"): # Removed is_print_error

        if not os.path.exists(spef_results_path):
            if self.is_print_message: print(f"Error: SPEF results file not found: {spef_results_path}")
            return pd.DataFrame()

        try:
            spef_files_df = pd.read_csv(spef_results_path)
        except Exception as e:
            if self.is_print_message: print(f"Error reading SPEF results {spef_results_path}: {e}")
            return pd.DataFrame()

        spef_filenames_dict = {}
        if 'filename' in spef_files_df.columns and 'file_name_bus' in spef_files_df.columns:
            for _, row in spef_files_df.iterrows():
                filename = str(row['filename'])
                file_name_bus = str(row['file_name_bus']) # This is like "bus1", "bus2"
                if filename not in spef_filenames_dict:
                    spef_filenames_dict[filename] = set()
                # We need to match bus_name_part from split_buses which is "hash_busX"
                # So, store "hash_busX" if file_name_bus is "busX"
                spef_filenames_dict[filename].add(f"{filename}_{file_name_bus}")
        else:
            if self.is_print_message: print(f"Error: Required columns ('filename', 'file_name_bus') not in {spef_results_path}")
            return pd.DataFrame()

        columns = ["file_name", "IA", "IB", "IC", "IN", "UA BB", "UB BB", "UC BB", "UN BB", "UA CL", "UB CL", "UC CL", "UN CL"]
        # Add ML columns as they are processed by _process_signals_for_SPEF
        columns.extend(self.ml_all) # self.ml_all contains short names like 'ML_1'
        dataset_df = pd.DataFrame(columns=columns) # Initialize with all potential columns

        if not os.path.isdir(self.raw_path):
            if self.is_print_message: print(f"Error: Raw path '{self.raw_path}' does not exist.")
            return dataset_df

        raw_files = sorted([f for f in os.listdir(self.raw_path) if f.lower().endswith('.cfg')])
        number_spef_found = 0

        with tqdm(total=len(raw_files), desc="Convert Comtrade to CSV for SPEF") as pbar:
            for file_cfg_name in raw_files:
                filename_without_ext = file_cfg_name[:-4]
                osc = None

                if filename_without_ext not in spef_filenames_dict:
                    pbar.update(1)
                    continue

                try:
                    cfg_full_path = os.path.join(self.raw_path, file_cfg_name)
                    osc = Oscillogram(cfg_full_path)
                    if osc.data_frame is None or osc.data_frame.empty:
                        self.unread_files.add(file_cfg_name)
                        pbar.update(1)
                        continue
                    raw_df = osc.data_frame
                except Exception as e:
                    self.unread_files.add(file_cfg_name)
                    if self.is_print_message: print(f"Error loading {file_cfg_name} with Oscillogram: {e}")
                    pbar.update(1)
                    continue

                normalized_df = self.normalizer.normalize_bus_signals(raw_df.copy(), filename_without_ext,
                                                                    yes_prase=yes_prase_norm,
                                                                    is_print_error=self.is_print_message)
                if normalized_df is None:
                    if self.is_print_message: print(f"Warning: Normalization failed for {filename_without_ext}.")
                    pbar.update(1)
                    continue

                # split_buses now returns df with 'file_name' like 'hash_busX'
                buses_df = self.split_buses(normalized_df, file_cfg_name)

                processed_bus_dfs = []
                if not buses_df.empty:
                    for bus_specific_name_from_splitter, bus_group_df in buses_df.groupby('file_name'):
                        # bus_specific_name_from_splitter is like "hash_bus1", "hash_bus2"
                        if bus_specific_name_from_splitter in spef_filenames_dict[filename_without_ext]:
                            # _process_signals_for_SPEF expects renamed columns
                            # rename_bus_columns is called inside split_buses.
                            # So bus_group_df here already has renamed columns.
                            processed_group = self._process_signals_for_SPEF(bus_group_df, is_print_error=self.is_print_message)
                            if processed_group is not None and not processed_group.empty:
                                processed_bus_dfs.append(processed_group)

                if not processed_bus_dfs:
                    pbar.update(1)
                    continue

                final_buses_df = pd.concat(processed_bus_dfs, ignore_index=True)

                # TODO: Implement cut_out_area_for_SPEF if needed, or adapt existing cut_out_area
                if is_cut_out_area and not final_buses_df.empty:
                    if osc.frequency is None or osc.frequency == 0 or \
                       not osc.cfg.sample_rates or len(osc.cfg.sample_rates[0]) < 1 or osc.cfg.sample_rates[0][0] == 0:
                        if self.is_print_message: print(f"Warning: Invalid frequency/sample_rate for {file_cfg_name}. Cannot cut area for SPEF.")
                    else:
                        samples_per_period = int(osc.cfg.sample_rates[0][0] / osc.frequency)
                        if samples_per_period > 0:
                            samples_before = samples_per_period * self.number_periods
                            samples_after = samples_per_period * self.number_periods
                            # cut_out_area expects ML columns which might not be relevant for SPEF specific cut
                            # For now, let's assume SPEF events might not need ML-based cutting, or use a generic cut if any.
                            # final_buses_df = self.cut_out_area(final_buses_df, samples_before, samples_after) # This might not be appropriate
                            if self.is_print_message: print(f"Warning: is_cut_out_area=True for SPEF but no specific SPEF cut logic implemented. Data not cut.")
                        else:
                            if self.is_print_message: print(f"Warning: Invalid samples_per_period for {file_cfg_name}. Cannot cut area for SPEF.")


                if not final_buses_df.empty:
                    dataset_df = pd.concat([dataset_df, final_buses_df], ignore_index=True) # Use True for clean index
                    number_spef_found += 1
                pbar.update(1)

        if self.is_print_message: print(f"Number of SPEF oscillograms processed = {number_spef_found}")
        if not dataset_df.empty:
            dataset_df = self.structure_columns(dataset_df) # Ensure consistent column order
            output_path = os.path.join(self.csv_path, csv_name)
            if self.csv_path and not os.path.exists(self.csv_path): os.makedirs(self.csv_path, exist_ok=True)
            dataset_df.to_csv(output_path, index=False)
            if self.is_print_message: print(f"SPEF dataset saved to {output_path}")
        elif self.is_print_message:
            print("No SPEF data processed, CSV not created.")
        return dataset_df

    def create_one_df(self, cfg_full_file_path: str, file_cfg_name: str) -> pd.DataFrame: # Changed first param name
        dataset_df = pd.DataFrame()
        osc = None
        try:
            osc = Oscillogram(cfg_full_file_path)
            if osc.data_frame is None or osc.data_frame.empty:
                if self.is_print_message: print(f"Warning: No data in {file_cfg_name} for create_one_df.")
                return dataset_df

            # self.check_columns(osc.data_frame) # check_columns might be too strict or need update
            # osc.data_frame is expected to have 'time' as a regular column
            dataset_df = self.split_buses(osc.data_frame, file_cfg_name)
        except FileNotFoundError:
            if self.is_print_message: print(f"File not found for create_one_df: {cfg_full_file_path}")
        except RuntimeError as e:
            if self.is_print_message: print(f"Runtime error for create_one_df {cfg_full_file_path}: {e}")
        except Exception as e:
            if self.is_print_message: print(f"Unexpected error for create_one_df {cfg_full_file_path}: {e}")
        return dataset_df
    
    # --- Helper Methods (Ensure these are consistent with the Python block from the prompt) ---

    def split_buses(self, raw_df: pd.DataFrame, file_cfg_name: str) -> pd.DataFrame:
        # This is the refined version from the prompt
        buses_df = pd.DataFrame()
        buses_cols = dict()
        raw_cols_set = set(raw_df.columns) # More efficient for intersection

        # Iterate through bus definitions (e.g., 'bus1', 'bus1_cl', 'bus2_bb') from self.analog_names_dict
        for bus_key, defined_signal_name_set in self.analog_names_dict.items():
            current_bus_cols = raw_cols_set.intersection(defined_signal_name_set)

            bus_num_match = re.search(r'(\d+)', bus_key) # Extracts first number found in bus_key like 'bus1', 'bus12'
            if bus_num_match:
                bus_num_str = bus_num_match.group(1)
                if bus_num_str in self.uses_buses:
                    ml_signals_for_this_bus_type = self.get_ml_signals(bus_num_str)
                    raw_ml_cols = raw_cols_set.intersection(ml_signals_for_this_bus_type)
                    current_bus_cols.update(raw_ml_cols)

            if current_bus_cols:
                buses_cols[bus_key] = current_bus_cols

        for bus_identifier_key, columns_to_select in buses_cols.items():
            bus_specific_df = raw_df.loc[:, list(columns_to_select)].copy()

            # Create a unique file_name for this bus segment: "original_cfg_name_without_ext_bus_key"
            output_file_name_col_val = os.path.splitext(file_cfg_name)[0] + "_" + bus_identifier_key
            bus_specific_df.insert(0, 'file_name', [output_file_name_col_val] * bus_specific_df.shape[0])

            bus_specific_df = self.rename_bus_columns(bus_specific_df)
            buses_df = pd.concat([buses_df, bus_specific_df], axis=0, ignore_index=True)
        return buses_df

    def split_buses_for_PDR(self, raw_df: pd.DataFrame, file_cfg_name: str) -> pd.DataFrame:
        # This is the refined version from the prompt
        buses_df = pd.DataFrame()
        buses_cols = dict()
        raw_cols_set = set(raw_df.columns)
        # analog_names_dict should contain all analog signals per bus definition
        for bus_key, analog_signal_set in self.analog_names_dict.items():
            current_bus_analog_cols = raw_cols_set.intersection(analog_signal_set)

            bus_num_match = re.search(r'(\d+)', bus_key)
            if bus_num_match:
                bus_num_str = bus_num_match.group(1)
                if bus_num_str in self.uses_buses:
                    pdr_signals_for_this_bus = self.get_PDR_signals(bus_num_str) # Assumes get_PDR_signals is defined
                    raw_pdr_cols = raw_cols_set.intersection(pdr_signals_for_this_bus)
                    current_bus_analog_cols.update(raw_pdr_cols)

            if current_bus_analog_cols:
                buses_cols[bus_key] = current_bus_analog_cols
        
        for bus_identifier_key, columns_to_select in buses_cols.items():
            bus_specific_df = raw_df.loc[:, list(columns_to_select)].copy()
            output_file_name_col_val = os.path.splitext(file_cfg_name)[0] + "_" + bus_identifier_key
            bus_specific_df.insert(0, 'file_name',  [output_file_name_col_val] * bus_specific_df.shape[0])
            bus_specific_df = self.rename_bus_columns(bus_specific_df, is_use_ML=False, is_use_discrete=True)
            buses_df = pd.concat([buses_df, bus_specific_df], axis=0, ignore_index=True)
        return buses_df

    def get_bus_names(self, analog=True, discrete=False):
        # Refined version from prompt
        bus_names = dict()
        if analog and self.analog_names: # self.analog_names is loaded from JSON
            for bus_key, names_config_or_list in self.analog_names.items():
                current_bus_set = bus_names.get(bus_key, set())
                if isinstance(names_config_or_list, dict):
                    for signal_group_names_set in names_config_or_list.values():
                        current_bus_set.update(signal_group_names_set)
                elif isinstance(names_config_or_list, list):
                     current_bus_set.update(names_config_or_list)
                if current_bus_set: bus_names[bus_key] = current_bus_set

        if discrete and self.discrete_names: # self.discrete_names is loaded from JSON
            for bus_key, names_config_or_list in self.discrete_names.items():
                current_bus_set = bus_names.get(bus_key, set())
                if isinstance(names_config_or_list, dict):
                    for signal_group_names_set in names_config_or_list.values():
                        current_bus_set.update(signal_group_names_set)
                elif isinstance(names_config_or_list, list):
                     current_bus_set.update(names_config_or_list)
                if current_bus_set: bus_names[bus_key] = current_bus_set
        return bus_names

    def get_all_names(self,):
        all_names = set()
        buses_names_map = self.get_bus_names(analog=True, discrete=True)
        for bus_key, names_set in buses_names_map.items():
            all_names.update(names_set)
        return all_names

    def rename_bus_columns(self, bus_df: pd.DataFrame, is_use_ML = True, is_use_discrete = False):
        # Refined version from prompt
        bus_columns_to_rename = {}
        # Ensure self.uses_buses contains strings for matching with i_bus_str
        # self.uses_buses should be like ['1', '2', '12']

        # ML Signals Renaming
        if is_use_ML:
            for i_bus_str in self.uses_buses:
                # get_ml_signals should return original long names like 'MLsignal_1_xyz'
                ml_signals_original_names = self.get_ml_signals(i_bus_str)
                for original_ml_name in ml_signals_original_names:
                    if original_ml_name in bus_df.columns:
                        # New short name will be like 'ML_xyz'
                        new_ml_name = original_ml_name.replace(f'MLsignal_{i_bus_str}_', 'ML_')
                        bus_columns_to_rename[original_ml_name] = new_ml_name
        
        # Analog and Discrete Signals Renaming (based on patterns)
        for existing_col_name in bus_df.columns:
            if existing_col_name in bus_columns_to_rename: continue # Already handled by ML rename

            if 'I | Bus' in existing_col_name:
                phase = existing_col_name.split(': ')[-1]
                if not ((phase == 'B' and not self.uses_CT_B) or (phase == 'N' and not self.uses_CT_zero)):
                    bus_columns_to_rename[existing_col_name] = f'I{phase}'
            elif self.use_VT_BB and 'U | BusBar' in existing_col_name:
                phase = existing_col_name.split(': ')[-1]
                if not (((phase in ['A', 'B', 'C']) and not self.uses_VT_ph) or \
                        ((phase in ['AB', 'BC', 'CA']) and not self.uses_VT_iph) or \
                        (phase == 'N' and not self.uses_VT_zero)):
                    bus_columns_to_rename[existing_col_name] = f'U{phase} BB'
            elif self.use_VT_CL and 'U | CableLine' in existing_col_name:
                phase = existing_col_name.split(': ')[-1]
                if not (((phase in ['A', 'B', 'C']) and not self.uses_VT_ph) or \
                        ((phase in ['AB', 'BC', 'CA']) and not self.uses_VT_iph) or \
                        (phase == 'N' and not self.uses_VT_zero)):
                    bus_columns_to_rename[existing_col_name] = f'U{phase} CL'

            if is_use_discrete: # This flag controls if PDR signals are renamed
                if self.use_PDR and 'PDR | Bus' in existing_col_name:
                    phase = existing_col_name.split(': ')[-1]
                    bus_columns_to_rename[existing_col_name] = f'rPDR {phase}'
                elif self.use_PDR and 'PDR_ideal | Bus' in existing_col_name:
                    phase = existing_col_name.split(': ')[-1]
                    bus_columns_to_rename[existing_col_name] = f'iPDR {phase}'
        
        bus_df.rename(columns=bus_columns_to_rename, inplace=True)
        return bus_df

    def check_columns(self, raw_df: pd.DataFrame):
        # Refined version from prompt
        ml_signals_to_check = set()
        for i_bus_str in self.uses_buses: # Ensure uses_buses are strings
            ml_signals_to_check.update(self.get_ml_signals(i_bus_str))

        expected_names = self.all_names.union(ml_signals_to_check)
        # self.all_names is built from JSONs. Oscillogram.data_frame columns are the source of truth from CFG.
        # This check is more about validating if JSON definitions cover all seen signals.

        unknown_columns_found = []
        for c in raw_df.columns:
            if c not in expected_names and c != 'time': # 'time' might be index or column
                 unknown_columns_found.append(c)

        if unknown_columns_found and self.is_print_message:
            print(f"Warning (check_columns): Unknown columns found: {unknown_columns_found}. These might be missing from JSON definitions or are unexpected.")


    def get_ml_signals(self, i_bus_str: str, use_operational_switching=True, use_abnormal_event=True, use_emergency_event=True):
        # Version from prompt (ensure i_bus_str is used in f-strings)
        ml_signals = set()
        ml_ops = {f'MLsignal_{i_bus_str}_1', f'MLsignal_{i_bus_str}_1_1', f'MLsignal_{i_bus_str}_1_1_1', f'MLsignal_{i_bus_str}_1_2'}
        ml_abn = {f'MLsignal_{i_bus_str}_2', f'MLsignal_{i_bus_str}_2_1', f'MLsignal_{i_bus_str}_2_1_1', f'MLsignal_{i_bus_str}_2_1_2', f'MLsignal_{i_bus_str}_2_1_3', f'MLsignal_{i_bus_str}_2_2', f'MLsignal_{i_bus_str}_2_3', f'MLsignal_{i_bus_str}_2_3_1', f'MLsignal_{i_bus_str}_2_4', f'MLsignal_{i_bus_str}_2_4_1', f'MLsignal_{i_bus_str}_2_4_2'}
        ml_emg = {f'MLsignal_{i_bus_str}_3', f'MLsignal_{i_bus_str}_3_1', f'MLsignal_{i_bus_str}_3_2', f'MLsignal_{i_bus_str}_3_3'}
        if use_operational_switching: ml_signals.update(ml_ops)
        if use_abnormal_event: ml_signals.update(ml_abn)
        if use_emergency_event: ml_signals.update(ml_emg)
        return ml_signals
        
    def get_PDR_signals(self, i_bus_str: str):
        return {
            f'PDR | Bus-{i_bus_str} | phase: A', f'PDR | Bus-{i_bus_str} | phase: B',
            f'PDR | Bus-{i_bus_str} | phase: C', f'PDR | Bus-{i_bus_str} | phase: PS',
            f'PDR_ideal | Bus-{i_bus_str} | phase: A', f'PDR_ideal | Bus-{i_bus_str} | phase: B',
            f'PDR_ideal | Bus-{i_bus_str} | phase: C', f'PDR_ideal | Bus-{i_bus_str} | phase: PS',
        }

    def get_short_names_ml_signals(self, use_operational_switching: bool =True, use_abnormal_event: bool = True, use_emergency_event: bool = True) -> tuple[list,list,list,list]:
        # Version from prompt
        ml_op_sw = ['ML_1', 'ML_1_1', 'ML_1_1_1', 'ML_1_2']
        ml_ab_ev = ['ML_2', 'ML_2_1', 'ML_2_1_1', 'ML_2_1_2', 'ML_2_1_3', 'ML_2_2', 'ML_2_3', 'ML_2_3_1', 'ML_2_4', 'ML_2_4_1', 'ML_2_4_2']
        ml_em_ev = ['ML_3', 'ML_3_1', 'ML_3_2', 'ML_3_3']
        ml_all_short = []
        if use_operational_switching: ml_all_short.extend(ml_op_sw)
        if use_abnormal_event: ml_all_short.extend(ml_ab_ev)
        if use_emergency_event: ml_all_short.extend(ml_em_ev)
        return ml_all_short, ml_op_sw, ml_ab_ev, ml_em_ev

    def get_short_names_ml_analog_signals(self) -> list:
        # Version from prompt
        return ['IA', 'IB', 'IC', 'IN', 'UA BB', 'UB BB', 'UC BB', 'UN BB', 'UAB BB', 'UBC BB', 'UCA BB', 'UA CL', 'UB CL', 'UC CL', 'UN CL', 'UAB CL', 'UBC CL', 'UCA CL']

    def cut_out_area(self, buses_df: pd.DataFrame, samples_before: int, samples_after: int) -> pd.DataFrame:
        # Version from prompt
        dataset_df_cut = pd.DataFrame()
        for _, bus_df_group in buses_df.groupby("file_name"): # Group by the new 'file_name' (hash_busX)
            temp_bus_df = bus_df_group.reset_index(drop=True)
            
            # Ensure is_save column is fresh for each group
            if "is_save" in temp_bus_df.columns: temp_bus_df.drop(columns=["is_save"], inplace=True)
            temp_bus_df["is_save"] = False

            # self.ml_all contains short names like 'ML_1'. bus_df_group should have these names
            # if rename_bus_columns was called (it is, in split_buses).
            ml_cols_in_df = [col for col in self.ml_all if col in temp_bus_df.columns]

            if ml_cols_in_df:
                # Check for any ML signal == 1 (assuming they are 0 or 1 after processing)
                # Handle potential NaN by filling with 0 before comparison, then checking any(axis=1)
                temp_bus_df["is_save"] = (temp_bus_df[ml_cols_in_df].fillna(0) == 1).any(axis=1)

                saved_indices = temp_bus_df.index[temp_bus_df["is_save"]].tolist()
                final_indices_to_save = set(saved_indices)

                for idx in saved_indices:
                    start_mark_idx = max(0, idx - samples_before)
                    end_mark_idx = min(len(temp_bus_df) -1, idx + samples_after)
                    final_indices_to_save.update(range(start_mark_idx, end_mark_idx + 1))

                temp_bus_df["is_save"] = False # Reset
                if final_indices_to_save:
                    temp_bus_df.loc[list(final_indices_to_save), "is_save"] = True

            final_cut_df = temp_bus_df[temp_bus_df["is_save"]].copy()

            # Add event numbers
            if not final_cut_df.empty:
                # Identify change points to define events
                final_cut_df['event_change'] = final_cut_df['is_save'].astype(int).diff().fillna(0)
                event_starts = final_cut_df[final_cut_df['event_change'] == 1].index
                if final_cut_df.iloc[0]['is_save']: # Handle if event starts at row 0
                    event_starts = event_starts.insert(0,0)

                event_number = 0
                current_event_id_col = []
                current_event_num_for_row = 0

                for idx in final_cut_df.index:
                    if idx in event_starts:
                        event_number +=1
                    current_event_num_for_row = event_number if final_cut_df.loc[idx, 'is_save'] else 0

                    if current_event_num_for_row > 0 :
                         current_event_id_col.append(final_cut_df.loc[idx, 'file_name'] + " _event N" + str(current_event_num_for_row))
                    else: # Should not happen if only is_save=True rows are kept
                         current_event_id_col.append(final_cut_df.loc[idx, 'file_name'])
                final_cut_df['file_name'] = current_event_id_col
                if 'event_change' in final_cut_df.columns: final_cut_df.drop(columns=['event_change'], inplace=True)


            if final_cut_df.empty and not temp_bus_df.empty:
                if len(temp_bus_df) > samples_before + samples_after:
                    middle = len(temp_bus_df) // 2
                    final_cut_df = temp_bus_df.iloc[max(0, middle-samples_before) : min(len(temp_bus_df), middle+samples_after+1)].copy()
                else:
                    final_cut_df = temp_bus_df.copy()
            
            if "is_save" in final_cut_df.columns:
                 final_cut_df = final_cut_df.drop(columns=["is_save"])
            dataset_df_cut = pd.concat([dataset_df_cut, final_cut_df], ignore_index=True)
        return dataset_df_cut

    def cut_out_area_for_PDR(self, buses_df: pd.DataFrame, samples_before: int, samples_after: int) -> pd.DataFrame:
        # Version from prompt (ensure it's adapted for current class structure if needed)
        dataset_df_pdr_cut = pd.DataFrame()
        
        primary_signal_col = "rPDR PS" # Renamed from PDR_PS in rename_bus_columns
        composite_signal_cols = ["rPDR A", "rPDR B", "rPDR C"] # Renamed

        for file_name_val, bus_df_group in buses_df.groupby("file_name"):
            temp_bus_df = bus_df_group.reset_index(drop=True)
            
            current_signal_series = pd.Series(False, index=temp_bus_df.index) # Default to no signal
            
            if primary_signal_col in temp_bus_df.columns and temp_bus_df[primary_signal_col].notna().any():
                current_signal_series = (temp_bus_df[primary_signal_col] == 1)
            elif all(col in temp_bus_df.columns for col in composite_signal_cols) and \
                 temp_bus_df[composite_signal_cols].notna().any().any():
                current_signal_series = (temp_bus_df[composite_signal_cols] == 1).all(axis=1)

            if not current_signal_series.any(): # No PDR signal activity found
                # if self.is_print_message: print(f"  No PDR activity in {file_name_val}, taking middle/all.")
                if len(temp_bus_df) > samples_before + samples_after:
                    middle = len(temp_bus_df) // 2
                    final_cut_df = temp_bus_df.iloc[max(0, middle-samples_before) : min(len(temp_bus_df), middle+samples_after+1)].copy()
                else:
                    final_cut_df = temp_bus_df.copy()
                if "change_event" in final_cut_df.columns: final_cut_df.drop(columns=["change_event"], inplace=True)
                dataset_df_pdr_cut = pd.concat([dataset_df_pdr_cut, final_cut_df], ignore_index=True)
                continue

            temp_bus_df['change_event'] = current_signal_series.astype(int).diff().fillna(0)
            change_indices = temp_bus_df.index[temp_bus_df['change_event'] != 0].tolist()

            if not change_indices: # Constant signal, or no changes
                 # If signal is constantly 1, treat whole segment as event. Otherwise, take middle.
                if current_signal_series.all(): # All 1s
                    final_cut_df = temp_bus_df.copy()
                    final_cut_df['file_name'] = final_cut_df['file_name'] + " _event N1"
                elif len(temp_bus_df) > samples_before + samples_after:
                    middle = len(temp_bus_df) // 2
                    final_cut_df = temp_bus_df.iloc[max(0, middle-samples_before) : min(len(temp_bus_df), middle+samples_after+1)].copy()
                else:
                    final_cut_df = temp_bus_df.copy()
                if "change_event" in final_cut_df.columns: final_cut_df.drop(columns=["change_event"], inplace=True)
                dataset_df_pdr_cut = pd.concat([dataset_df_pdr_cut, final_cut_df], ignore_index=True)
                continue

            rows_to_save_indices = set()
            for idx in change_indices:
                start_save_idx = max(0, idx - samples_before)
                end_save_idx = min(len(temp_bus_df) - 1, idx + samples_after)
                rows_to_save_indices.update(range(start_save_idx, end_save_idx + 1))

            final_cut_df = temp_bus_df.loc[sorted(list(rows_to_save_indices))].copy()

            if not final_cut_df.empty:
                final_cut_df['event_group_id'] = (final_cut_df['change_event'] != 0).cumsum()
                # Assign event numbers based on these groups
                # This needs to map group_id to a sequential event number for file_name
                event_number_map = {group_id: i+1 for i, group_id in enumerate(final_cut_df['event_group_id'].unique())}
                final_cut_df['event_num_str'] = final_cut_df['event_group_id'].map(lambda x: " _event N" + str(event_number_map[x]))
                final_cut_df['file_name'] = final_cut_df['file_name'] + final_cut_df['event_num_str']
                final_cut_df.drop(columns=['change_event', 'event_group_id', 'event_num_str'], inplace=True)

            dataset_df_pdr_cut = pd.concat([dataset_df_pdr_cut, final_cut_df], ignore_index=True)
        return dataset_df_pdr_cut

    def get_simple_dataset(self, dataset_df: pd.DataFrame, csv_name='dataset_simpl.csv'):
        # Version from prompt
        column_names = set(dataset_df.columns)
        # Use the lists of short names stored in self
        ml_opr_swch_cols = [col for col in self.ml_opr_swch if col in column_names]
        ml_abnorm_evnt_cols = [col for col in self.ml_abnorm_evnt if col in column_names]
        ml_emerg_evnt_cols = [col for col in self.ml_emerg_evnt if col in column_names]

        dataset_df["opr_swch"] = dataset_df[ml_opr_swch_cols].any(axis=1).astype(int) if ml_opr_swch_cols else 0
        dataset_df["abnorm_evnt"] = dataset_df[ml_abnorm_evnt_cols].any(axis=1).astype(int) if ml_abnorm_evnt_cols else 0
        dataset_df["emerg_evnt"] = dataset_df[ml_emerg_evnt_cols].any(axis=1).astype(int) if ml_emerg_evnt_cols else 0
        
        dataset_df.loc[dataset_df['emerg_evnt'] == 1, 'abnorm_evnt'] = 0
        dataset_df.loc[dataset_df['emerg_evnt'] == 1, 'opr_swch'] = 0
        dataset_df.loc[dataset_df['abnorm_evnt'] == 1, 'opr_swch'] = 0
        
        dataset_df["normal"] = 0
        dataset_df.loc[(dataset_df["opr_swch"] == 0) & (dataset_df["abnorm_evnt"] == 0) & (dataset_df["emerg_evnt"] == 0), "normal"] = 1
        
        cols_to_drop_in_simple = [col for col in self.ml_all if col in dataset_df.columns]
        if cols_to_drop_in_simple:
            dataset_df = dataset_df.drop(columns=cols_to_drop_in_simple)

        output_simple_path = os.path.join(self.csv_path, csv_name)
        # Ensure self.csv_path directory exists
        if self.csv_path and not os.path.exists(self.csv_path):
            os.makedirs(self.csv_path, exist_ok=True)
        dataset_df.to_csv(output_simple_path, index=False)
        if self.is_print_message: print(f"Simple dataset saved to {output_simple_path}")

    def structure_columns(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
        # Version from prompt
        desired_order = ["file_name"]
        # get_short_names_ml_analog_signals() returns a list of strings
        desired_order.extend(self.get_short_names_ml_analog_signals())
        # self.ml_all is already a list of short ML signal names
        desired_order.extend(self.ml_all)
        
        simplified_cols = ["opr_swch", "abnorm_evnt", "emerg_evnt", "normal"]
        
        # Determine which set of columns to prioritize based on presence of simplified_cols
        current_cols_in_df = set(dataset_df.columns)
        is_simplified_dataset = any(s_col in current_cols_in_df for s_col in simplified_cols)

        final_ordered_list = []
        if is_simplified_dataset:
            # Order: file_name, analog, then simplified event types
            for col in ["file_name"] + self.get_short_names_ml_analog_signals() + simplified_cols:
                if col in current_cols_in_df and col not in final_ordered_list:
                    final_ordered_list.append(col)
        else:
            # Order: file_name, analog, then detailed ML event types
            for col in desired_order: # desired_order already has file_name, analog, ml_all
                if col in current_cols_in_df and col not in final_ordered_list:
                    final_ordered_list.append(col)

        # Add any remaining columns from the DataFrame that weren't in the desired_order
        other_cols = [col for col in dataset_df.columns if col not in final_ordered_list]
        final_ordered_list.extend(other_cols)

        return dataset_df[final_ordered_list]
