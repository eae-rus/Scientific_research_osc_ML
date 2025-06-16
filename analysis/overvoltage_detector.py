import os
import numpy as np
import pandas as pd
import re
from typing import Dict, Any, List, Optional, Tuple, Set
import shutil # Added import
from tqdm import tqdm # Added import
import csv # Added import

from core.oscillogram import Oscillogram
from normalization.normalization import OscillogramNormalizer
from raw_to_csv.raw_to_csv import OscillogramToCsvConverter # For bus splitting

class OvervoltageDetector:
    def __init__(self, config: Dict[str, Any],
                 normalizer: OscillogramNormalizer,
                 bus_splitter: OscillogramToCsvConverter,
                 norm_coef_df: pd.DataFrame):
        self.config = config
        self.normalizer = normalizer
        self.bus_splitter = bus_splitter
        self.norm_coef_df = norm_coef_df # DataFrame of normalization coefficients
        self.verbose = config.get('verbose', False)

        # Extract config values with defaults
        self.VALID_NOMINAL_VOLTAGES: Set[float] = set(config.get('VALID_NOMINAL_VOLTAGES', {6000.0, 10000.0, 35000.0}))
        self.SPEF_THRESHOLD_U0: float = config.get('SPEF_THRESHOLD_U0', 0.1 / np.sqrt(3) / 3) # Approx 0.019 for Ubase=1pu_phase_peak
        self.SPEF_THRESHOLD_Un: float = config.get('SPEF_THRESHOLD_Un', 0.05 / 3) # Approx 0.016 for Ubase=1pu_phase_peak
        self.SPEF_MIN_DURATION_PERIODS: int = config.get('SPEF_MIN_DURATION_PERIODS', 1)
        self.SIMILAR_AMPLITUDES_FILTER_ENABLED: bool = config.get('SIMILAR_AMPLITUDES_FILTER_ENABLED', True)
        self.SIMILAR_AMPLITUDES_MAX_RELATIVE_DIFFERENCE: float = config.get('SIMILAR_AMPLITUDES_MAX_RELATIVE_DIFFERENCE', 0.05)

        if self.SPEF_MIN_DURATION_PERIODS <= 0:
            raise ValueError("SPEF_MIN_DURATION_PERIODS must be positive.")

    def _find_spef_zones(self, group_df: pd.DataFrame, group_prefix: str, samples_per_period: int) -> List[Tuple[int, int]]:
        if group_df.empty or samples_per_period <= 0:
            if self.verbose and group_df.empty: print(f"    _find_spef_zones: Empty group_df for {group_prefix}.")
            if self.verbose and samples_per_period <= 0: print(f"    _find_spef_zones: Invalid samples_per_period ({samples_per_period}) for {group_prefix}.")
            return []

        u0_calculated: Optional[pd.Series] = None
        un_measured: Optional[pd.Series] = None

        phase_cols_candidate = [f'U{ph} {group_prefix}' for ph in ['A', 'B', 'C']]
        available_phase_cols = [col for col in phase_cols_candidate if col in group_df.columns]

        if len(available_phase_cols) == 3:
            try:
                u0_calculated = (group_df[available_phase_cols[0]].astype(float) +
                                 group_df[available_phase_cols[1]].astype(float) +
                                 group_df[available_phase_cols[2]].astype(float)) / 3.0
            except Exception as e:
                if self.verbose: print(f"    Warning (_find_spef_zones): Could not calculate U0 for {group_prefix}: {e}")

        un_col_name = f'UN {group_prefix}'
        if un_col_name in group_df.columns:
            try:
                un_measured = group_df[un_col_name].astype(float)
            except Exception as e:
                if self.verbose: print(f"    Warning (_find_spef_zones): Could not process UN for {group_prefix}: {e}")

        if u0_calculated is None and un_measured is None:
            if self.verbose: print(f"    _find_spef_zones: No U0 or UN signals available/processable for {group_prefix}.")
            return []

        combined_mask = pd.Series(False, index=group_df.index)
        if u0_calculated is not None:
            combined_mask |= (u0_calculated.abs() > self.SPEF_THRESHOLD_U0)
        if un_measured is not None:
            combined_mask |= (un_measured.abs() > self.SPEF_THRESHOLD_Un)

        if not combined_mask.any():
            return [] # No points above threshold

        min_len = self.SPEF_MIN_DURATION_PERIODS * samples_per_period
        if min_len <=0:
            if self.verbose: print(f"    _find_spef_zones: min_len ({min_len}) is not positive. SPEF_MIN_DURATION_PERIODS: {self.SPEF_MIN_DURATION_PERIODS}, samples_per_period: {samples_per_period}")
            return []

        candidate_spef_zones = []
        is_in_block = False
        current_start_idx = 0
        for i, value_is_true in enumerate(combined_mask):
            actual_index = group_df.index[i] # Get original DataFrame index
            if value_is_true and not is_in_block:
                is_in_block = True
                current_start_idx = actual_index
            elif not value_is_true and is_in_block:
                is_in_block = False
                # end index is the last True index, which is group_df.index[i-1]
                if (group_df.index[i-1] - current_start_idx + 1) >= min_len:
                    candidate_spef_zones.append((current_start_idx, group_df.index[i-1]))
        if is_in_block: # If series ends in a True block
            if (group_df.index[-1] - current_start_idx + 1) >= min_len:
                candidate_spef_zones.append((current_start_idx, group_df.index[-1]))

        if not candidate_spef_zones: return []

        actual_spef_zones = []
        for start_idx, end_idx in candidate_spef_zones:
            # Similarity filter applies only if enabled and all 3 phase voltages are available
            apply_similarity_filter = self.SIMILAR_AMPLITUDES_FILTER_ENABLED and (len(available_phase_cols) == 3)
            is_zone_valid_after_similarity = True # Assume valid unless filtered

            if apply_similarity_filter:
                zone_df_segment = group_df.loc[start_idx:end_idx, available_phase_cols]
                if zone_df_segment.empty or zone_df_segment.isnull().all().all():
                    # If segment is empty or all NaNs, similarity filter can't apply meaningfully, keep zone for now
                    pass
                else:
                    amplitudes = []
                    valid_amplitudes = True
                    for phase_col in available_phase_cols:
                        try:
                            max_abs_val = zone_df_segment[phase_col].abs().max()
                            if pd.isna(max_abs_val): # If any phase max is NaN, can't compute similarity
                                valid_amplitudes = False; break
                            amplitudes.append(max_abs_val)
                        except KeyError: # Should not happen due to available_phase_cols check
                            valid_amplitudes = False; break

                    if valid_amplitudes and len(amplitudes) == 3:
                        mean_amp = np.mean(amplitudes)
                        if mean_amp < 1e-9: # Avoid division by zero if all amplitudes are near zero
                            relative_difference = 0.0 if (np.max(amplitudes) - np.min(amplitudes)) < 1e-9 else float('inf')
                        else:
                            relative_difference = (np.max(amplitudes) - np.min(amplitudes)) / mean_amp

                        if relative_difference < self.SIMILAR_AMPLITUDES_MAX_RELATIVE_DIFFERENCE:
                            is_zone_valid_after_similarity = False
                            if self.verbose: print(f"    Zone {start_idx}-{end_idx} for {group_prefix} filtered out by phase similarity (RelDiff: {relative_difference:.3f})")
                    elif self.verbose and not valid_amplitudes:
                         print(f"    Warning (_find_spef_zones): Could not get valid amplitudes for similarity check in zone {start_idx}-{end_idx} for {group_prefix}.")


            if is_zone_valid_after_similarity:
                actual_spef_zones.append((start_idx, end_idx))

        return actual_spef_zones

    def _calculate_max_overvoltage_in_zones(self, group_df: pd.DataFrame, group_prefix: str,
                                           spef_zones: List[Tuple[int, int]],
                                           nominal_voltage_ph_rms: float) -> Optional[float]:
        if not spef_zones or group_df.empty:
            return None
        if nominal_voltage_ph_rms <= 1e-6 : # Check for non-positive or extremely small nominal voltage
             if self.verbose: print(f"    _calc_max_ov: Invalid nominal_voltage_ph_rms ({nominal_voltage_ph_rms:.3f}) for {group_prefix}.")
             return None


        phase_cols = [f'U{ph} {group_prefix}' for ph in ['A', 'B', 'C']]
        available_phase_cols = [col for col in phase_cols if col in group_df.columns]
        if not available_phase_cols:
            if self.verbose: print(f"    _calc_max_ov: No phase columns ({phase_cols}) found in group_df for {group_prefix}")
            return None

        max_inst_val_overall = 0.0
        found_any_value_in_any_zone = False

        for start_idx, end_idx in spef_zones:
            # Ensure start and end indices are valid for the DataFrame's index
            if start_idx not in group_df.index or end_idx not in group_df.index:
                if self.verbose: print(f"    _calc_max_ov: Invalid zone indices [{start_idx}-{end_idx}] for DataFrame. Skipping zone.")
                continue

            zone_df_segment = group_df.loc[start_idx:end_idx, available_phase_cols]
            if zone_df_segment.empty:
                continue

            try:
                # Get max absolute value across all available phase columns in the current zone segment
                max_in_zone_for_segment = zone_df_segment.abs().max().max() # Max over columns, then max over those maxes
                if pd.notna(max_in_zone_for_segment):
                    if max_in_zone_for_segment > max_inst_val_overall:
                        max_inst_val_overall = max_in_zone_for_segment
                    found_any_value_in_any_zone = True
            except Exception as e:
                if self.verbose: print(f"    _calc_max_ov: Error finding max in zone {start_idx}-{end_idx} for {group_prefix}: {e}")

        if not found_any_value_in_any_zone or max_inst_val_overall < 1e-9 :
            if self.verbose: print(f"    _calc_max_ov: No significant phase voltage values found in valid SPEF zones for {group_prefix}.")
            return None

        overvoltage_factor = (max_inst_val_overall / np.sqrt(2)) / nominal_voltage_ph_rms

        if self.verbose:
            print(f"    _calc_max_ov: Max inst val for {group_prefix} across SPEF zones: {max_inst_val_overall:.4f}, Nom Ph RMS: {nominal_voltage_ph_rms:.2f}, OV Factor: {overvoltage_factor:.3f}")
        return overvoltage_factor

    def analyze_oscillogram(self, oscillogram: Oscillogram) -> Optional[Dict[str, Any]]:
        if self.verbose: print(f"  Analyzing oscillogram: {oscillogram.file_hash}")

        if oscillogram.frequency is None or oscillogram.frequency <= 0 or \
           not oscillogram.cfg.sample_rates or len(oscillogram.cfg.sample_rates[0]) < 1 or \
           oscillogram.cfg.sample_rates[0][0] <= 0:
            if self.verbose: print(f"    Invalid frequency or sample_rate for {oscillogram.file_hash}. Skipping analysis.")
            return None
        samples_per_period = int(oscillogram.cfg.sample_rates[0][0] / oscillogram.frequency)
        if samples_per_period <= 0:
            if self.verbose: print(f"    Invalid samples_per_period ({samples_per_period}) for {oscillogram.file_hash}. Skipping.")
            return None

        # 1. Normalization
        if self.normalizer is None:
            if self.verbose: print(f"    Normalizer not provided. Skipping normalization for {oscillogram.file_hash}.")
            normalized_df = oscillogram.data_frame.copy() # Use raw data if no normalizer
        else:
            normalized_df = self.normalizer.normalize_bus_signals(
                oscillogram.data_frame.copy(),
                oscillogram.file_hash,
                yes_prase=self.config.get('norm_yes_phrase', "YES"),
                is_print_error=self.verbose
            )

        if normalized_df is None:
            if self.verbose: print(f"    Normalization failed or not permitted for {oscillogram.file_hash}. Skipping analysis.")
            return None

        # 2. Bus Splitting
        # Ensure 'time' is a column if split_buses expects it (Oscillogram.data_frame has 'time' as column)
        # The `file_name` arg for split_buses is the original CFG filename (e.g., "hash.cfg")
        buses_grouped_df = self.bus_splitter.split_buses(normalized_df, os.path.basename(oscillogram.filepath))

        if buses_grouped_df is None or buses_grouped_df.empty:
            if self.verbose: print(f"    Bus splitting yielded no data for {oscillogram.file_hash}. Skipping analysis.")
            return None

        # 3. Retrieve norm_row for physical nominal voltage checks
        if self.norm_coef_df.empty or 'name' not in self.norm_coef_df.columns:
            if self.verbose: print(f"    Normalization coefficient DataFrame is empty or missing 'name' column. Cannot get physical nominals.")
            return None

        norm_row_matches = self.norm_coef_df[self.norm_coef_df["name"].astype(str) == str(oscillogram.file_hash)]
        if norm_row_matches.empty:
            if self.verbose: print(f"    No normalization coefficients found for {oscillogram.file_hash} in provided norm_coef_df. Skipping.")
            return None
        norm_row = norm_row_matches.iloc[0]

        max_overvoltage_for_file = -1.0
        best_result_for_file: Dict[str, Any] = {}

        for bus_specific_name, bus_group_df in buses_grouped_df.groupby('file_name'):
            # bus_specific_name is like "hash_bus1", "hash_bus2_cl", etc.
            # Extract bus_idx and group_type (BB/CL)
            match = re.search(r"_bus(\d+)(_?CL)?$", bus_specific_name) # Matches _bus<number> or _bus<number>_CL
            bus_idx_str = ""
            group_type = "BB" # Default to BusBar

            if match:
                bus_idx_str = match.group(1)
                if match.group(2) and match.group(2) == '_CL': # Check if _CL part exists
                    group_type = "CL"
            else: # Fallback if pattern is just _<name> e.g. _bus1, _bus2
                bus_name_part = bus_specific_name.replace(oscillogram.file_hash, "",1).lstrip('_') # e.g. "bus1", "busCL2"
                num_match = re.search(r'(\d+)', bus_name_part)
                if num_match: bus_idx_str = num_match.group(1)
                if "cl" in bus_name_part.lower() : group_type = "CL"

            if not bus_idx_str:
                if self.verbose: print(f"    Could not parse bus index from '{bus_specific_name}'. Skipping this bus group.")
                continue

            # Determine nominal voltage column and get physical nominal voltage
            # This depends on how norm_coef_df stores nominals (e.g. phase or line, kV or V)
            # Assuming norm_coef_df stores physical phase RMS voltage in Volts directly in these columns.
            nominal_voltage_col_key = f"{bus_idx_str}U{group_type[0].lower()}_base" # e.g. 1Ub_base or 1Uc_base

            physical_nominal_voltage_coeff = norm_row.get(nominal_voltage_col_key)

            if pd.isna(physical_nominal_voltage_coeff):
                if self.verbose: print(f"    Nominal voltage coefficient not found for {bus_idx_str}{group_type} ('{nominal_voltage_col_key}') in norm_coef for {oscillogram.file_hash}.")
                continue
            try:
                physical_nominal_voltage_value = float(physical_nominal_voltage_coeff)
            except ValueError:
                if self.verbose: print(f"    Invalid nominal voltage coefficient '{physical_nominal_voltage_coeff}' for {bus_idx_str}{group_type}. Skipping.")
                continue

            if physical_nominal_voltage_value not in self.VALID_NOMINAL_VOLTAGES:
                if self.verbose: print(f"    Physical nominal voltage {physical_nominal_voltage_value}V for {bus_idx_str}{group_type} is not in VALID_NOMINAL_VOLTAGES. Skipping.")
                continue

            # Since data is normalized (1 PU = Nominal Phase RMS), for _calculate_max_overvoltage_in_zones, nominal_voltage_ph_rms is 1.0
            spef_zones = self._find_spef_zones(bus_group_df, group_type, samples_per_period)
            if spef_zones:
                overvoltage_factor = self._calculate_max_overvoltage_in_zones(bus_group_df, group_type, spef_zones, 1.0)

                if overvoltage_factor is not None and overvoltage_factor > max_overvoltage_for_file:
                    max_overvoltage_for_file = overvoltage_factor
                    best_result_for_file = {
                        'filename': oscillogram.file_hash,
                        'overvoltage': round(max_overvoltage_for_file, 3),
                        'bus': bus_idx_str,
                        'group': group_type # 'BB' or 'CL'
                    }
                    if self.verbose: print(f"    New max overvoltage for {oscillogram.file_hash}: {max_overvoltage_for_file:.3f} on Bus {bus_idx_str} {group_type}")

        return best_result_for_file if best_result_for_file else None

    def _save_results_and_log(self, results: List[Dict[str, Any]], error_log: List[Tuple[str, str]],
                              output_csv_path: str, log_path: str) -> None:
        # Create DataFrame from results
        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # Define bins and labels for overvoltage grouping
            bins = self.config.get('overvoltage_report_bins', [0, 1.1, 1.3, 1.5, 2.0, np.inf])
            labels = self.config.get('overvoltage_report_labels', ["<1.1", "1.1-1.3", "1.3-1.5", "1.5-2.0", ">2.0"])
            if 'overvoltage' in results_df.columns:
                 results_df['overvoltage_group'] = pd.cut(results_df['overvoltage'], bins=bins, labels=labels, right=False)
            else: # Handle case where no 'overvoltage' column if all results were None
                 results_df['overvoltage_group'] = pd.NA


            # Ensure output directory exists
            output_dir = os.path.dirname(output_csv_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            if self.verbose:
                print(f"\nOvervoltage analysis results saved to: {output_csv_path}")
                print("Overvoltage Group Counts:")
                print(results_df['overvoltage_group'].value_counts().sort_index())
        elif self.verbose:
            print("\nNo overvoltages detected or results to save.")

        if error_log:
            log_dir = os.path.dirname(log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                for err_file, err_msg in error_log:
                    f.write(f"{err_file}: {err_msg}\n")
            if self.verbose: print(f"Error log saved to: {log_path}")

    def analyze_directory(self, osc_folder_path: str, output_csv_path: str, log_path: str) -> None:
        if not os.path.isdir(osc_folder_path):
            if self.verbose: print(f"Error: Source directory '{osc_folder_path}' not found.")
            return

        all_results: List[Dict[str, Any]] = []
        error_files_log: List[Tuple[str, str]] = []

        cfg_files = [os.path.join(r, f) for r, _, fs in os.walk(osc_folder_path) for f in fs if f.lower().endswith(".cfg")]

        if self.verbose: print(f"Found {len(cfg_files)} .cfg files for overvoltage analysis.")

        for cfg_file_path in tqdm(cfg_files, desc="Analyzing oscillograms for overvoltage", disable=not self.verbose):
            base_name = os.path.basename(cfg_file_path)
            if self.verbose and len(cfg_files) < 20: print(f"  Processing: {base_name}") # Print if few files
            try:
                osc = Oscillogram(cfg_file_path)
                result = self.analyze_oscillogram(osc)
                if result:
                    all_results.append(result)
            except FileNotFoundError:
                error_files_log.append((base_name, "CFG or DAT file not found."))
            except RuntimeError as e:
                error_files_log.append((base_name, f"RuntimeError loading: {e}"))
            except Exception as e:
                error_files_log.append((base_name, f"Unexpected error: {e}"))

        self._save_results_and_log(all_results, error_files_log, output_csv_path, log_path)
        if self.verbose: print("Overvoltage analysis of directory complete.")

    @staticmethod
    def copy_spef_oscillograms(report_csv_path: str, source_osc_folder_path: str,
                               destination_folder_path: str, is_print_message: bool = False) -> None:
        if not os.path.exists(report_csv_path):
            if is_print_message: print(f"Error: Report CSV file not found: {report_csv_path}")
            return
        if not os.path.isdir(source_osc_folder_path):
            if is_print_message: print(f"Error: Source oscillogram folder not found: {source_osc_folder_path}")
            return

        os.makedirs(destination_folder_path, exist_ok=True)

        try:
            report_df = pd.read_csv(report_csv_path)
        except Exception as e:
            if is_print_message: print(f"Error reading report CSV {report_csv_path}: {e}")
            return

        if 'filename' not in report_df.columns:
            if is_print_message: print(f"Error: 'filename' column not found in {report_csv_path}.")
            return

        copied_count = 0
        skipped_count = 0
        filenames_to_copy = report_df['filename'].unique()

        for filename_hash in tqdm(filenames_to_copy, desc="Copying SPEF oscillograms", disable=not is_print_message):
            cfg_file_name = f"{filename_hash}.cfg"
            dat_file_name = f"{filename_hash}.dat"

            source_cfg_path = os.path.join(source_osc_folder_path, cfg_file_name)
            source_dat_path = os.path.join(source_osc_folder_path, dat_file_name)

            dest_cfg_path = os.path.join(destination_folder_path, cfg_file_name)
            dest_dat_path = os.path.join(destination_folder_path, dat_file_name)

            copied_pair = False
            if os.path.exists(source_cfg_path):
                try:
                    shutil.copy2(source_cfg_path, dest_cfg_path)
                    copied_pair = True
                except Exception as e:
                    if is_print_message: print(f"  Error copying CFG {source_cfg_path}: {e}")
            elif is_print_message:
                print(f"  Warning: CFG file not found: {source_cfg_path}")

            if os.path.exists(source_dat_path):
                try:
                    shutil.copy2(source_dat_path, dest_dat_path)
                    copied_pair = copied_pair and True # Only true if both copied
                except Exception as e:
                    if is_print_message: print(f"  Error copying DAT {source_dat_path}: {e}")
                    copied_pair = False # If DAT copy fails, pair is not fully copied
            elif is_print_message:
                print(f"  Warning: DAT file not found: {source_dat_path}")
                copied_pair = False # If DAT is missing, pair is not fully copied

            if copied_pair:
                copied_count +=1
            elif os.path.exists(source_cfg_path) or os.path.exists(source_dat_path): # If at least one existed but pair failed
                skipped_count +=1


        if is_print_message:
            print(f"Finished copying. Copied {copied_count} pairs of CFG/DAT files.")
            if skipped_count > 0:
                print(f"Skipped {skipped_count} files (one part of pair missing or copy error).")
