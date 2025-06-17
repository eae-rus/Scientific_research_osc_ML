import os
import numpy as np
import pandas as pd
from scipy.fft import fft # Explicitly import fft from scipy.fft
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm # Assuming tqdm is available

from core.oscillogram import Oscillogram
from normalization.normalization import OscillogramNormalizer # Actual import

# Forward declaration for type hinting if OscillogramNormalizer is in a separate module
# and there's a risk of circular imports. For now, we'll use a string or Any if needed.
# from normalization.normalization import OscillogramNormalizer # Actual import if no circular dependency risk

class ChannelType(Enum):
    CURRENT = 'current'
    VOLTAGE = 'voltage'
    NONE = 'none'

def sliding_window_fft(signal: np.ndarray, fft_window_size: int, num_harmonics: int, verbose: bool = False) -> np.ndarray:
    """
    Calculates specified number of harmonics using a sliding window FFT.

    Args:
        signal (np.ndarray): Input signal.
        fft_window_size (int): Size of the FFT window.
        num_harmonics (int): Number of harmonics to calculate (e.g., 1 for H1, 2 for H1 & H2).
        verbose (bool): If True, prints debug messages.

    Returns:
        np.ndarray: Array where each row corresponds to a window start time,
                    and columns correspond to complex FFT coefficients of H1, H2, ... H_num_harmonics.
                    Shape: (n_points - fft_window_size + 1, num_harmonics).
                    Returns an empty array if input is invalid.
    """
    n_points = len(signal)

    # Initialize results array with NaNs (complex type)
    # Shape: (number_of_windows, num_harmonics)
    num_windows = n_points - fft_window_size + 1
    if num_windows < 0: num_windows = 0 # handles n_points < fft_window_size case early

    fft_results = np.full((num_windows, num_harmonics), np.nan + 1j*np.nan, dtype=complex)

    if n_points == 0:
        if verbose: print("    sliding_window_fft: Empty signal provided.")
        return np.array([]).reshape(0,num_harmonics) # Ensure consistent shape for empty result
    if fft_window_size <= 0:
        if verbose: print(f"    sliding_window_fft: Invalid fft_window_size ({fft_window_size}).")
        return np.array([]).reshape(0,num_harmonics)

    if n_points < fft_window_size:
        if verbose: print(f"    sliding_window_fft: Signal length ({n_points}) < FFT window ({fft_window_size}).")
        return np.array([]).reshape(0,num_harmonics)

    for i in range(num_windows):
        window_data = signal[i : i + fft_window_size]

        # Normalized FFT coefficients (amplitude = |X[k]| * 2 / N for k!=0, DC = |X[0]|/N)
        # We store complex coefficients first, then derive amplitude.
        # fft_coeffs_normalized = np.fft.fft(window_data) / fft_window_size
        # For amplitude of harmonic h_num (1-indexed), it's usually abs(fft_coeffs_normalized[h_num]) * 2

        fft_coeffs_raw = np.fft.fft(window_data) # Raw FFT output

        current_window_harmonics_complex = np.full(num_harmonics, np.nan + 1j*np.nan, dtype=complex)
        for h_idx in range(num_harmonics): # h_idx is 0-based (for H1, h_idx=0)
            harmonic_number = h_idx + 1   # Actual harmonic number (1st, 2nd, ...)

            if harmonic_number < len(fft_coeffs_raw): # Ensure index is valid
                # Amplitude of X_k is |fft_coeffs_raw[k]| / N for DC (k=0) or Nyquist (k=N/2)
                # Amplitude of X_k is 2 * |fft_coeffs_raw[k]| / N for 0 < k < N/2
                # We store the complex value X[k] * (2/N) to get amplitude later via np.abs()
                # or X[k]/N if we only want one-sided spectrum value from two-sided result.
                # For consistency with previous example that used `abs(fft(window))/N` for H1,
                # let's assume the threshold comparison is based on that scaling.
                # If H1 amplitude is `abs(fft(window)[1])/N`, then this is what we store.
                # The `*2` is if you want the full amplitude of that frequency component.
                # Let's provide the complex coefficient scaled by 1/N. Amplitude can be derived by caller.
                # The prompt's _get_h1_amplitude_series used: `np.abs(fft_coeffs_complex[:, 0]) / np.sqrt(2)`
                # which suggests fft_coeffs_complex was RMS.
                # Original EmptyOscFilter used: `np.abs(fft(window)) / window_size` then `h1_rms = h1_amplitude / np.sqrt(2)`
                # Let's provide complex X[k]/N.
                current_window_harmonics_complex[h_idx] = fft_coeffs_raw[harmonic_number] / fft_window_size
            else:
                # Not enough FFT coefficients for this harmonic (should not happen if num_harmonics is reasonable)
                break
        fft_results[i, :] = current_window_harmonics_complex
    return fft_results


class OscillogramActivityFilter:
    def __init__(self, config: Dict[str, Any], normalizer: Optional[OscillogramNormalizer] = None, show_progress_bars: bool = True):
        self.config = config
        self.normalizer = normalizer
        self.fft_window_size = -1 # To be set per oscillogram based on its frequency and sample rate
        self.show_progress_bars = show_progress_bars

        # Default patterns if not in config
        default_analyze_patterns = ['i |', 'u |', 'phase: n'] # Matches current, voltage, neutral signals
        default_current_patterns = ['i |']
        default_voltage_patterns = ['u |']

        self.channels_to_analyze_patterns_lower = [
            p.lower() for p in self.config.get('channels_to_analyze_patterns', default_analyze_patterns)
        ]
        self.current_id_patterns_lower = [
            p.lower() for p in self.config.get('current_channel_id_patterns', default_current_patterns)
        ]
        self.voltage_id_patterns_lower = [
            p.lower() for p in self.config.get('voltage_channel_id_patterns', default_voltage_patterns)
        ]
        self.verbose = self.config.get('verbose', False)

    def _make_unique_column_names(self, name_list: List[str]) -> List[str]:
        counts = {}
        new_names = []
        for name in name_list:
            if name not in counts:
                counts[name] = 0
                new_names.append(name)
            else:
                counts[name] += 1
                new_names.append(f"{name}_dup{counts[name]}")
        return new_names

    def _get_h1_amplitude_series(self, signal_values: np.ndarray) -> Optional[pd.Series]:
        if self.fft_window_size <= 0:
            if self.verbose: print(f"    _get_h1_amplitude_series: fft_window_size not set or invalid ({self.fft_window_size}).")
            return None
        if signal_values is None or len(signal_values) == 0:
            if self.verbose: print(f"    _get_h1_amplitude_series: Empty or None signal provided.")
            return None

        # sliding_window_fft returns complex X[1]/N (for num_harmonics=1)
        fft_complex_h1_per_window = sliding_window_fft(signal_values, self.fft_window_size, num_harmonics=1, verbose=self.verbose)

        if fft_complex_h1_per_window.shape[0] == 0 or fft_complex_h1_per_window.shape[1] == 0 :
             if self.verbose: print(f"    _get_h1_amplitude_series: sliding_window_fft returned no H1 data.")
             return None

        # sliding_window_fft returns complex X[h_num]/fft_window_size.
        # For peak amplitude of H1, it's 2 * abs(X[1]/N).
        h1_peak_amplitude_per_window = np.abs(fft_complex_h1_per_window[:, 0]) * 2

        # Convert peak to RMS for threshold comparison, assuming thresholds are RMS based
        h1_rms_amplitude_per_window = h1_peak_amplitude_per_window / np.sqrt(2)

        valid_h1_amplitudes = h1_rms_amplitude_per_window[~np.isnan(h1_rms_amplitude_per_window)]

        if len(valid_h1_amplitudes) == 0:
             if self.verbose: print(f"    _get_h1_amplitude_series: No valid H1 amplitudes calculated (all NaN or empty).")
             return None

        return pd.Series(valid_h1_amplitudes)

    def _is_initial_signal_clean(self, signal_values: np.ndarray, channel_type_str: str) -> Tuple[bool, Optional[float]]:
        if self.fft_window_size <= 0:
            if self.verbose: print(f"    _is_initial_signal_clean: fft_window_size not set or invalid ({self.fft_window_size}).")
            return False, None
        if signal_values is None or len(signal_values) == 0:
             if self.verbose: print(f"    _is_initial_signal_clean: Empty or None signal provided.")
             return False, None

        raw_analysis_config = self.config.get('raw_signal_analysis', {})
        periods_to_check = raw_analysis_config.get('initial_window_check_periods', 1)
        # Ensure points_to_check is at least fft_window_size for a single FFT
        points_to_check = max(self.fft_window_size, periods_to_check * self.fft_window_size)


        if len(signal_values) < points_to_check:
            if self.verbose:
                print(f"    _is_initial_signal_clean: Insufficient data ({len(signal_values)}) for clean check ({points_to_check} points).")
            return False, None # Not enough data for the configured check window

        window_data = signal_values[:points_to_check]

        if window_data.size == 0: # Should be caught by len(signal_values) check already
            if self.verbose: print(f"    _is_initial_signal_clean: Window data is empty for FFT.")
            return False, None

        # Normalized FFT: fft_coeffs[k] = X[k]/N
        fft_coeffs_normalized = np.fft.fft(window_data) / len(window_data)

        # For a window of `periods_to_check` fundamental periods, the 1st harmonic of the fundamental
        # corresponds to index `periods_to_check` in this specific FFT result.
        idx_h1 = periods_to_check

        if idx_h1 == 0 or idx_h1 >= len(fft_coeffs_normalized): # H1 index must be > 0 and within bounds
            if self.verbose: print(f"    _is_initial_signal_clean: Invalid H1 index ({idx_h1}) for FFT results length ({len(fft_coeffs_normalized)}).")
            return False, None

        # Amplitude of H1 = 2 * |X[H1_index]/N|
        m1_amplitude = np.abs(fft_coeffs_normalized[idx_h1]) * 2

        # Higher harmonics start from H2 relative to the fundamental (whose H1 is at idx_h1)
        # So, H2 is at 2*idx_h1, H3 at 3*idx_h1, etc.
        # Max of these up to Nyquist.
        # This is different from original EmptyOscFilter's Hx which was sum of H2-H(N/2) of the window itself.
        # Let's use the EmptyOscFilter's approach: sum of higher harmonics of the window FFT.
        start_higher_harm_idx_of_window = 2 # 2nd harmonic of the window FFT
        end_higher_harm_idx_of_window = len(fft_coeffs_normalized) // 2

        mx_amplitude = 0.0
        if start_higher_harm_idx_of_window <= end_higher_harm_idx_of_window and \
           start_higher_harm_idx_of_window < len(fft_coeffs_normalized) :
            # Amplitudes of higher harmonics of the window signal
            higher_harm_amplitudes = np.abs(fft_coeffs_normalized[start_higher_harm_idx_of_window : end_higher_harm_idx_of_window + 1]) * 2
            if len(higher_harm_amplitudes) > 0:
                mx_amplitude = np.max(higher_harm_amplitudes) # Max of H2, H3,... of the window

        if self.verbose:
            print(f"    _is_initial_signal_clean: Window M1 (at index {idx_h1})={m1_amplitude:.4f}, Window Max_HigherComponents (H2 onwards)={mx_amplitude:.4f}")

        is_clean = False
        # Use channel_type_str ('current' or 'voltage') to get specific threshold
        key_suffix = 'U' if channel_type_str == ChannelType.VOLTAGE.value else 'I'
        h1_vs_hx_ratio_threshold = raw_analysis_config.get(f'h1_vs_hx_ratio_threshold_{key_suffix}', 1.5)

        if mx_amplitude < 1e-9: # If higher harmonics are virtually zero for the window
            is_clean = m1_amplitude > 1e-6 # Clean if M1 is not also zero (i.e., it's a pure sine of fundamental)
        else:
            ratio_h1_hx = m1_amplitude / mx_amplitude
            is_clean = ratio_h1_hx > h1_vs_hx_ratio_threshold
            if self.verbose:
                print(f"    _is_initial_signal_clean: M1/Max_WindowHx ratio = {ratio_h1_hx:.2f} (threshold > {h1_vs_hx_ratio_threshold})")

        if is_clean:
            if self.verbose: print(f"    _is_initial_signal_clean: Initial signal segment is 'clean'. H1_start_amplitude={m1_amplitude:.4f}")
            return True, m1_amplitude # Return the M1 amplitude of the (fundamental within the) long window
        else:
            if self.verbose: print(f"    _is_initial_signal_clean: Initial signal segment is 'noisy/distorted'.")
            return False, None

    def check_activity(self, oscillogram: Oscillogram) -> bool:
        if oscillogram.frequency is None or oscillogram.frequency <= 0 or \
           not oscillogram.cfg.sample_rates or len(oscillogram.cfg.sample_rates[0]) < 1 or \
           oscillogram.cfg.sample_rates[0][0] <= 0:
            if self.verbose: print(f"  check_activity: Invalid frequency ({oscillogram.frequency}) or sample_rate ({oscillogram.cfg.sample_rates[0][0] if osc.cfg.sample_rates else 'N/A'}) for {oscillogram.file_hash}. Skipping.")
            return False

        self.fft_window_size = int(oscillogram.cfg.sample_rates[0][0] / oscillogram.frequency)
        if self.fft_window_size <= 0:
            if self.verbose: print(f"  check_activity: Invalid fft_window_size ({self.fft_window_size}) for {oscillogram.file_hash}. Skipping.")
            return False

        current_df_orig = oscillogram.data_frame
        if current_df_orig is None or current_df_orig.empty:
            if self.verbose: print(f"  check_activity: DataFrame is empty for {oscillogram.file_hash}.")
            return False

        current_df = current_df_orig.copy() # Work on a copy
        # Assuming Oscillogram class already ensures unique column names from comtrade_APS
        # If not, uncommenting this is a good safety measure:
        # current_df.columns = self._make_unique_column_names(list(current_df.columns))

        use_normalized_thresholds = False
        processed_df_for_analysis = current_df # Default to original (or uniquely named) df

        if self.config.get('use_norm_osc', False) and self.normalizer:
            if self.verbose: print(f"  Attempting normalization for {oscillogram.file_hash}...")
            normalized_df = self.normalizer.normalize_bus_signals(
                current_df.copy(),
                oscillogram.file_hash,
                yes_prase=self.config.get('norm_yes_phrase', "YES"), # Get yes_prase from config
                is_print_error=self.verbose
            )

            if normalized_df is not None:
                # Check if normalization was effectively applied (e.g., 'norm' status was YES)
                # This logic might need refinement based on how OscillogramNormalizer signals success.
                # For now, assume if it returns a df, it was applied as per its internal logic.
                # A more robust check would involve inspecting the norm_coef table if accessible.
                if self.normalizer.norm_coef is not None and not self.normalizer.norm_coef.empty:
                    norm_row = self.normalizer.norm_coef[
                        self.normalizer.norm_coef["name"].astype(str) == str(oscillogram.file_hash)
                    ]
                    if not norm_row.empty and self.config.get('norm_yes_phrase', "YES") in str(norm_row["norm"].iloc[0]):
                        processed_df_for_analysis = normalized_df
                        use_normalized_thresholds = True
                        if self.verbose: print(f"    Normalization successfully applied for {oscillogram.file_hash}.")
                    elif self.verbose:
                        print(f"    Normalization conditions not met or 'norm' status not YES for {oscillogram.file_hash}.")
                else: # Normalizer has no coefficients loaded
                     if self.verbose: print(f"    Normalizer has no coefficients loaded, using raw data for {oscillogram.file_hash}.")
            elif self.verbose:
                print(f"    Normalization returned None for {oscillogram.file_hash}.")

        for col_name in processed_df_for_analysis.columns:
            if col_name == 'time': continue

            col_name_lower = col_name.lower()
            if not any(patt in col_name_lower for patt in self.channels_to_analyze_patterns_lower):
                continue

            signal_values = processed_df_for_analysis[col_name].astype(float).values

            channel_type = ChannelType.NONE
            if any(p in col_name_lower for p in self.current_id_patterns_lower): channel_type = ChannelType.CURRENT
            elif any(p in col_name_lower for p in self.voltage_id_patterns_lower): channel_type = ChannelType.VOLTAGE
            if channel_type == ChannelType.NONE: continue

            h1_for_relative_norm_peak = None # Store peak H1 for relative normalization
            current_threshold_set_key = ""
            is_raw_analysis_path = False

            if use_normalized_thresholds and "_dup" not in col_name:
                current_threshold_set_key = f"thresholds_{channel_type.value}_normalized"
                if self.verbose: print(f"  Analyzing channel: {col_name} (Type: {channel_type.value}, Mode: Normalized)")
            else:
                is_raw_analysis_path = True
                if self.verbose:
                    if "_dup" in col_name: print(f"  Channel {col_name} is a duplicate, using Raw/Relative analysis.")
                    else: print(f"  Analyzing channel: {col_name} (Type: {channel_type.value}, Mode: Raw/Relative)")

                is_clean, h1_initial_peak_val = self._is_initial_signal_clean(signal_values, channel_type.value)
                if not is_clean:
                    if self.verbose: print(f"    Channel {col_name} failed initial clean check. Skipping.")
                    continue
                if h1_initial_peak_val is None or h1_initial_peak_val < self.config.get('raw_signal_analysis',{}).get('min_initial_h1_amplitude_for_rel_norm', 1e-6):
                    if self.verbose: print(f"    Initial H1 peak for {col_name} too small ({h1_initial_peak_val}). Skipping.")
                    continue
                h1_for_relative_norm_peak = h1_initial_peak_val
                current_threshold_set_key = f"thresholds_raw_{channel_type.value}_relative"

            current_thresholds = self.config.get(current_threshold_set_key)
            if not current_thresholds:
                if self.verbose: print(f"    Warning: No threshold set found for key '{current_threshold_set_key}' in config.")
                continue

            h1_series_rms = self._get_h1_amplitude_series(signal_values)
            if h1_series_rms is None or h1_series_rms.empty or h1_series_rms.isna().all():
                if self.verbose: print(f"    No valid H1 RMS series for {col_name}.")
                continue

            h1_to_analyze = h1_series_rms.copy()
            if is_raw_analysis_path and h1_for_relative_norm_peak is not None:
                h1_initial_rms_for_norm = h1_for_relative_norm_peak / np.sqrt(2)
                if h1_initial_rms_for_norm < 1e-9: # Avoid division by zero/very small
                     if self.verbose: print(f"    Initial H1 RMS for relative norm of {col_name} is near zero. Skipping relative norm.")
                     continue
                h1_to_analyze = h1_to_analyze / h1_initial_rms_for_norm
                if self.verbose: print(f"    H1 RMS series for {col_name} relativized by initial H1_RMS={h1_initial_rms_for_norm:.4f}")

            stat_h1_max_abs_rms = h1_to_analyze.max()
            stat_h1_delta_rms = h1_to_analyze.max() - h1_to_analyze.min() if len(h1_to_analyze) > 0 else 0
            stat_h1_std_dev_rms = h1_to_analyze.std() if len(h1_to_analyze) > 1 else 0

            if self.verbose:
                print(f"    Stats for H1_RMS (analyzed) of {col_name}: Max={stat_h1_max_abs_rms:.4f}, Delta={stat_h1_delta_rms:.4f}, StdDev={stat_h1_std_dev_rms:.4f}")
                print(f"    Thresholds: Delta > {current_thresholds['delta']:.4f}, StdDev > {current_thresholds['std_dev']:.4f}", end="")
                if 'max_abs' in current_thresholds: print(f", MaxAbs > {current_thresholds['max_abs']:.4f}")
                else: print("")

            is_active_by_delta = stat_h1_delta_rms > current_thresholds['delta']
            is_active_by_std = stat_h1_std_dev_rms > current_thresholds['std_dev']

            passes_primary_activity = is_active_by_delta or is_active_by_std

            if use_normalized_thresholds and not is_raw_analysis_path : # Normalized path
                passes_max_abs_check = True # Default true if not defined
                if 'max_abs' in current_thresholds:
                    passes_max_abs_check = stat_h1_max_abs_rms > current_thresholds['max_abs']
                if passes_primary_activity and passes_max_abs_check:
                    if self.verbose: print(f"    Channel {col_name} is ACTIVE (Normalized criteria).")
                    return True
            else: # Raw/Relative path
                # In raw/relative, max_abs is optional; if not present, primary activity is enough.
                # If max_abs is present, it must also be met if primary activity is met.
                # This interpretation matches original: (delta OR std) AND (max_abs OR (not is_normalized_path))
                # Which simplifies to: if raw, (delta OR std) is enough if max_abs not specified.
                # If max_abs IS specified for raw, it must pass.
                if 'max_abs' in current_thresholds:
                    if passes_primary_activity and (stat_h1_max_abs_rms > current_thresholds['max_abs']):
                        if self.verbose: print(f"    Channel {col_name} is ACTIVE (Raw/Relative criteria with MaxAbs).")
                        return True
                elif passes_primary_activity: # No max_abs threshold for raw, delta/std is enough
                    if self.verbose: print(f"    Channel {col_name} is ACTIVE (Raw/Relative criteria w/o MaxAbs).")
                    return True

        if self.verbose: print(f"  No active channels found in {oscillogram.file_hash}. Oscillogram considered inactive.")
        return False

    def filter_directory(self, source_dir: str, output_csv_path: str) -> None:
        if not os.path.isdir(source_dir):
            if self.verbose: print(f"OscillogramActivityFilter Error: Source directory '{source_dir}' not found.")
            return

        active_file_names = [] # Store only basenames (hash.cfg)
        error_logs = []

        cfg_files = []
        for root, _, files in os.walk(source_dir):
            for file_name in files:
                if file_name.lower().endswith(".cfg"):
                    cfg_files.append(os.path.join(root, file_name))

        if self.verbose: print(f"OscillogramActivityFilter: Found {len(cfg_files)} .cfg files to filter.")

        for cfg_file_path in tqdm(cfg_files, desc="Filtering oscillograms for activity", disable=not self.show_progress_bars):
            base_name = os.path.basename(cfg_file_path)
            # if self.verbose: print(f"Processing file: {base_name}") # tqdm provides progress
            try:
                osc = Oscillogram(cfg_file_path)
                if self.check_activity(osc):
                    active_file_names.append(base_name)
            except FileNotFoundError:
                error_logs.append((cfg_file_path, "File not found during Oscillogram init."))
                if self.verbose: print(f"  Error: File not found for {base_name}")
            except RuntimeError as e:
                error_logs.append((cfg_file_path, f"RuntimeError: {e}"))
                if self.verbose: print(f"  Error: Runtime error loading {base_name}: {e}")
            except ValueError as e: # E.g. from bad freq/sample rate in check_activity
                 error_logs.append((cfg_file_path, f"ValueError: {e}"))
                 if self.verbose: print(f"  Error: Value error processing {base_name}: {e}")
            except Exception as e:
                error_logs.append((cfg_file_path, f"Unexpected error: {e}"))
                if self.verbose: print(f"  Error: Unexpected error with {base_name}: {e}")

        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                print(f"OscillogramActivityFilter Error: Could not create output directory {output_dir}: {e}")
                return

        try:
            result_df = pd.DataFrame({'active_files': sorted(list(set(active_file_names)))}) # Changed column name
            result_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            if self.verbose: print(f"\nActive oscillogram list saved to: {output_csv_path}")
            print(f"Total active oscillograms found: {len(active_file_names)}")
        except Exception as e:
            print(f"\nOscillogramActivityFilter Error: Could not write results to CSV {output_csv_path}: {e}")

        if error_logs and self.verbose:
            print(f"\nEncountered {len(error_logs)} errors during processing. Example error: {error_logs[0]}")
            # error_log_path = os.path.join(source_dir, "activity_filter_errors.txt") # Example error log path
            # with open(error_log_path, 'w', encoding='utf-8') as f_err:
            #    for err_path, err_msg in error_logs:
            #        f_err.write(f"{err_path}: {err_msg}\n")
            # if self.verbose: print(f"Error details saved to {error_log_path}")
