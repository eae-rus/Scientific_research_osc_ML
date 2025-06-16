import os
import numpy as np
import pandas as pd
from scipy.fft import fft
from core.oscillogram import Oscillogram
from normalization.normalization import NormOsc # Will be OscillogramNormalizer
from raw_to_csv.raw_to_csv import RawToCSV # For bus splitting

class SPEFAnalyzer:
    def __init__(self, normalizer: NormOsc, bus_splitter: RawToCSV,
                 threshold_3u0: float = (30/400/3), spef_period_count: int = 3):
        self.normalizer = normalizer
        self.bus_splitter = bus_splitter
        self.threshold_3u0 = threshold_3u0
        self.spef_period_count = spef_period_count
        if self.threshold_3u0 <= 0:
            raise ValueError("threshold_3u0 must be positive.")
        if self.spef_period_count <= 0:
            raise ValueError("spef_period_count must be positive.")

    def _calculate_sliding_window_first_harmonic(self, signal_values: np.ndarray,
                                                 samples_per_period: int) -> np.ndarray:
        if not isinstance(signal_values, np.ndarray):
            signal_values = np.array(signal_values, dtype=float)
        else:
            # Ensure it's float type, copy if not to avoid changing original array type
            if not np.issubdtype(signal_values.dtype, np.floating):
                 signal_values = signal_values.astype(float)
            # No copy needed if already float. astype(float, copy=False) is tricky if already float.

        num_samples = len(signal_values)
        if num_samples < samples_per_period or samples_per_period == 0:
            return np.array([])

        harmonics = np.zeros(num_samples - samples_per_period + 1, dtype=float)

        for i in range(num_samples - samples_per_period + 1):
            window = signal_values[i : i + samples_per_period]

            if samples_per_period > 1 :
                 window_fft_complex = fft(window)
                 window_fft_magnitudes = np.abs(window_fft_complex) / samples_per_period
                 harmonics[i] = window_fft_magnitudes[1] # First harmonic
            elif samples_per_period == 1:
                 harmonics[i] = 0.0
        return harmonics

    def detect_spef_on_bus_dataframe(self, bus_df: pd.DataFrame, samples_per_period: int,
                                     is_print_message: bool = False) -> bool:
        if bus_df.empty:
            if is_print_message: print("SPEFAnalyzer: Input DataFrame is empty for SPEF detection.")
            return False
        if not isinstance(samples_per_period, int) or samples_per_period <= 0: # Ensure samples_per_period is int
            if is_print_message: print("SPEFAnalyzer: samples_per_period must be a positive integer.")
            return False

        # min_signal_length is the raw data length needed to produce enough harmonic points
        # To get `self.spef_period_count` harmonic points, we need data for
        # `samples_per_period + (self.spef_period_count - 1)` points.
        # Each harmonic point represents a window of `samples_per_period`.
        # `self.spef_period_count` consecutive harmonic points means the event spans `self.spef_period_count` data periods.
        # So, the data signal must be at least `self.spef_period_count * samples_per_period` long.
        min_data_len_for_spef_duration = self.spef_period_count * samples_per_period
        if min_data_len_for_spef_duration == 0 : # handles samples_per_period=0 if not caught above, or spef_period_count=0
             if is_print_message: print("SPEFAnalyzer: Calculated min_data_len_for_spef_duration is zero.")
             return False


        consecutive_harmonic_windows_needed = self.spef_period_count

        # Check UN BB
        if 'UN BB' in bus_df.columns:
            un_bb_signal = bus_df['UN BB'].fillna(0).values
            if len(un_bb_signal) >= min_data_len_for_spef_duration:
                harmonics_un_bb = self._calculate_sliding_window_first_harmonic(un_bb_signal, samples_per_period)
                if len(harmonics_un_bb) >= consecutive_harmonic_windows_needed:
                    for i in range(len(harmonics_un_bb) - consecutive_harmonic_windows_needed + 1):
                        if np.all(harmonics_un_bb[i : i + consecutive_harmonic_windows_needed] >= self.threshold_3u0):
                            if is_print_message: print("SPEFAnalyzer: SPEF detected on UN BB.")
                            return True

        # Check UN CL
        if 'UN CL' in bus_df.columns:
            un_cl_signal = bus_df['UN CL'].fillna(0).values
            if len(un_cl_signal) >= min_data_len_for_spef_duration:
                harmonics_un_cl = self._calculate_sliding_window_first_harmonic(un_cl_signal, samples_per_period)
                if len(harmonics_un_cl) >= consecutive_harmonic_windows_needed:
                    for i in range(len(harmonics_un_cl) - consecutive_harmonic_windows_needed + 1):
                        if np.all(harmonics_un_cl[i : i + consecutive_harmonic_windows_needed] >= self.threshold_3u0):
                            if is_print_message: print("SPEFAnalyzer: SPEF detected on UN CL.")
                            return True

        phase_cols_bb = ['UA BB', 'UB BB', 'UC BB']
        if all(col in bus_df.columns for col in phase_cols_bb):
            ua_bb = bus_df[phase_cols_bb[0]].fillna(0).values
            ub_bb = bus_df[phase_cols_bb[1]].fillna(0).values
            uc_bb = bus_df[phase_cols_bb[2]].fillna(0).values
            if all(len(sig) >= min_data_len_for_spef_duration for sig in [ua_bb, ub_bb, uc_bb]):
                u0_3_bb_signal = (ua_bb + ub_bb + uc_bb) / np.sqrt(3)
                harmonics_3u0_bb = self._calculate_sliding_window_first_harmonic(u0_3_bb_signal, samples_per_period)
                if len(harmonics_3u0_bb) >= consecutive_harmonic_windows_needed:
                    for i in range(len(harmonics_3u0_bb) - consecutive_harmonic_windows_needed + 1):
                        if np.all(harmonics_3u0_bb[i : i + consecutive_harmonic_windows_needed] >= self.threshold_3u0):
                            if is_print_message: print("SPEFAnalyzer: SPEF detected on 3U0 from BusBar phases.")
                            return True

        phase_cols_cl = ['UA CL', 'UB CL', 'UC CL']
        if all(col in bus_df.columns for col in phase_cols_cl):
            ua_cl = bus_df[phase_cols_cl[0]].fillna(0).values
            ub_cl = bus_df[phase_cols_cl[1]].fillna(0).values
            uc_cl = bus_df[phase_cols_cl[2]].fillna(0).values
            if all(len(sig) >= min_data_len_for_spef_duration for sig in [ua_cl, ub_cl, uc_cl]):
                u0_3_cl_signal = (ua_cl + ub_cl + uc_cl) / np.sqrt(3)
                harmonics_3u0_cl = self._calculate_sliding_window_first_harmonic(u0_3_cl_signal, samples_per_period)
                if len(harmonics_3u0_cl) >= consecutive_harmonic_windows_needed:
                    for i in range(len(harmonics_3u0_cl) - consecutive_harmonic_windows_needed + 1):
                        if np.all(harmonics_3u0_cl[i : i + consecutive_harmonic_windows_needed] >= self.threshold_3u0):
                            if is_print_message: print("SPEFAnalyzer: SPEF detected on 3U0 from CableLine phases.")
                            return True

        # if is_print_message: print("SPEFAnalyzer: No SPEF detected on this bus DataFrame based on the defined criteria.") # Only print if no SPEF overall for the file
        return False

    def find_spef_in_directory(self, source_dir: str, output_csv_path: str,
                               norm_coef_file_path: str, filter_txt_path: str = None,
                               is_print_message: bool = False) -> None:

        if not os.path.isdir(source_dir):
            if is_print_message: print(f"SPEFAnalyzer Error: Source directory '{source_dir}' not found.")
            return

        try:
            # Assuming NormOsc can have its norm_coef DataFrame set this way or has a method to load it.
            # If self.normalizer is instantiated outside and expected to be ready, this line might be conditional
            # or handled by a dedicated method in NormOsc.
            self.normalizer.norm_coef = pd.read_csv(norm_coef_file_path)
            if is_print_message: print(f"SPEFAnalyzer: Loaded normalization coefficients from {norm_coef_file_path}")
        except Exception as e:
            if is_print_message: print(f"SPEFAnalyzer Error: Could not load norm_coef_file from {norm_coef_file_path}: {e}")
            return

        filter_set = None
        if filter_txt_path and os.path.exists(filter_txt_path):
            try:
                with open(filter_txt_path, 'r', encoding='utf-8') as f: # Specify encoding for filter file
                    filter_set = set(line.strip() for line in f if line.strip())
                if is_print_message: print(f"SPEFAnalyzer: Loaded filter set from {filter_txt_path} with {len(filter_set)} entries.")
            except Exception as e:
                if is_print_message: print(f"SPEFAnalyzer Warning: Could not read filter file {filter_txt_path}: {e}")

        spef_results = []

        cfg_files_to_process = [os.path.join(r, f) for r, _, fs in os.walk(source_dir) for f in fs if f.lower().endswith(".cfg")]

        if is_print_message: print(f"SPEFAnalyzer: Found {len(cfg_files_to_process)} .cfg files to analyze for SPEF.")

        for i, cfg_file_path in enumerate(cfg_files_to_process):
            base_cfg_name = os.path.basename(cfg_file_path)
            file_name_hash_only = base_cfg_name[:-4] # Assumes .cfg extension

            if filter_set and file_name_hash_only not in filter_set:
                if is_print_message and i < 5 : print(f"SPEFAnalyzer: Skipping {base_cfg_name} due to filter.") # Print for first few
                continue

            if is_print_message and i % 10 == 0 : print(f"SPEFAnalyzer: Processing {base_cfg_name} ({i+1}/{len(cfg_files_to_process)})...")

            try:
                osc = Oscillogram(cfg_file_path) # Oscillogram handles its own file reading errors
                if osc.data_frame is None or osc.data_frame.empty:
                    if is_print_message: print(f"  No data in {base_cfg_name}, skipping.")
                    continue

                if osc.frequency is None or osc.frequency <= 0 or not osc.cfg.sample_rates or osc.cfg.sample_rates[0][0] <=0:
                     if is_print_message: print(f"  Invalid frequency ({osc.frequency}) or sample_rate ({osc.cfg.sample_rates[0][0] if osc.cfg.sample_rates else 'N/A'}) for {base_cfg_name}. Skipping.")
                     continue
                samples_per_period = int(osc.cfg.sample_rates[0][0] / osc.frequency)
                if samples_per_period <= 0:
                    if is_print_message: print(f"  Invalid samples_per_period ({samples_per_period}) for {base_cfg_name}. Skipping.")
                    continue

                # Bus Splitting: RawToCSV.split_buses expects raw_df (DataFrame) and file_name (str, e.g. "hash.cfg")
                # It returns a single DataFrame with a 'file_name' column like 'hash_bus1', 'hash_bus2'
                # osc.data_frame might have 'time' as index. split_buses might need 'time' as column.
                df_for_splitting = osc.data_frame.reset_index()
                all_buses_df = self.bus_splitter.split_buses(df_for_splitting, base_cfg_name)

                if all_buses_df is None or all_buses_df.empty:
                    if is_print_message: print(f"  Bus splitting resulted in no data for {base_cfg_name}.")
                    continue

                file_had_any_spef = False
                for bus_specific_name_from_splitter, bus_group_df in all_buses_df.groupby('file_name'):
                    # bus_specific_name_from_splitter is like "hash_bus1". We only need "bus1" part.
                    bus_name_part = bus_specific_name_from_splitter.replace(file_name_hash_only, "", 1).lstrip('_') # gives "bus1"

                    if is_print_message: print(f"    Analyzing bus: {bus_name_part} from {bus_specific_name_from_splitter}")

                    normalized_bus_df = self.normalizer.normalize_bus_signals(
                        bus_group_df.copy(),
                        file_name_hash_only,
                        yes_prase="YES",
                        is_print_error=False # Reduce noise, main loop prints errors
                    )

                    if normalized_bus_df is None or normalized_bus_df.empty:
                        if is_print_message: print(f"      Normalization failed or resulted in empty data for bus {bus_name_part}.")
                        continue

                    is_spef = self.detect_spef_on_bus_dataframe(normalized_bus_df, samples_per_period, is_print_message)

                    if is_spef:
                        spef_results.append([file_name_hash_only, bus_name_part]) # Use just "busX"
                        file_had_any_spef = True
                        if is_print_message: print(f"      SPEF DETECTED for {file_name_hash_only} on bus {bus_name_part}!")

                if file_had_any_spef and is_print_message:
                    print(f"  Finished processing {base_cfg_name}. SPEF found.")
                elif is_print_message:
                    print(f"  Finished processing {base_cfg_name}. No SPEF found.")


            except FileNotFoundError: # Should be caught by Oscillogram, but as a fallback.
                if is_print_message: print(f"  SPEFAnalyzer: CFG File not found error for {base_cfg_name}.")
            except RuntimeError as e:
                 if is_print_message: print(f"  SPEFAnalyzer: Runtime error loading/processing {base_cfg_name}: {e}")
            except ValueError as e: # E.g. from bad freq/sample rate
                 if is_print_message: print(f"  SPEFAnalyzer: Value error processing {base_cfg_name}: {e}")
            except Exception as e:
                if is_print_message: print(f"  SPEFAnalyzer: Unexpected error processing {base_cfg_name}: {e}")

        # Write results to CSV
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                if is_print_message: print(f"SPEFAnalyzer Error: Could not create output directory {output_dir}: {e}")
                return # Cannot write CSV

        try:
            results_df = pd.DataFrame(spef_results, columns=['filename', 'file_name_bus'])
            results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            if is_print_message: print(f"SPEFAnalyzer: Analysis complete. Results saved to {output_csv_path}. Found {len(spef_results)} SPEF instances.")
        except Exception as e:
            if is_print_message: print(f"SPEFAnalyzer Error: Could not write results to CSV {output_csv_path}: {e}")
