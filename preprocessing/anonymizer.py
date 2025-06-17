import os
import hashlib
from tqdm import tqdm # Added import

class DataAnonymizer:
    def __init__(self, verbose_logging: bool = False, show_progress_bars: bool = True): # Modified parameters
        self.verbose_logging = verbose_logging
        self.show_progress_bars = show_progress_bars

    def anonymize_file(self, cfg_file_path: str, encoding: str) -> bool:
        try:
            # Determine DAT file path
            dat_file_path = os.path.splitext(cfg_file_path)[0] + ".dat"
            if not os.path.exists(dat_file_path):
                print(f"Error (anonymize_file): DAT file {dat_file_path} not found for CFG {cfg_file_path} (encoding {encoding}).")
                return False

            # Read CFG lines
            lines = []
            try:
                with open(cfg_file_path, 'r', encoding=encoding) as file:
                    lines = file.readlines()
            except UnicodeDecodeError:
                # This specific error is important for the retry logic in anonymize_directory
                print(f"Error (anonymize_file): Failed to decode {cfg_file_path} with encoding {encoding}.")
                return False
            except IOError as e: # Catch other IO errors specifically if needed
                print(f"Error (anonymize_file): IO error reading {cfg_file_path} with encoding {encoding}: {e}.")
                return False


            if not lines:
                print(f"Error (anonymize_file): CFG file {cfg_file_path} (encoding {encoding}) is empty.")
                return False

            # Anonymize station/device ID (line 0)
            parts = lines[0].split(',')
            if len(parts) >= 3: # station,id,rev_year (or more if commas in name)
                # Preserve revision year, anonymize station and id
                lines[0] = ",," + parts[-1].strip() + "\n"
            else:
                # Fallback for unexpected format (e.g. already anonymized or malformed)
                lines[0] = ",,\n" # Default anonymized first line

            # Anonymize dates (lines after channel definitions)
            # This logic assumes nrates = 1 from the original script.
            # Line indices: 0=station, 1=counts, 2..analog_end=analog, analog_end+1..status_end=status,
            # status_end+1=freq, status_end+2=nrates, status_end+3=samp1, status_end+4=endsamp1,
            # status_end+5=start_datetime, status_end+6=trigger_datetime
            if len(lines) > 1:
                signals_line_parts = lines[1].split(',')
                if len(signals_line_parts) > 0 and signals_line_parts[0].strip().isdigit():
                    # total_channels is the first part of line 1 (e.g., "6,3A,3D")
                    total_channels = int(signals_line_parts[0].strip())

                    # Assuming nrates = 1, so 1 line for sample rate + 1 line for end sample for that rate.
                    # The original code implies this structure:
                    # line 0: station, id, rev_year
                    # line 1: total_chn, analog_chn, status_chn
                    # line 2 to (2 + total_chn - 1): channel definitions
                    # line (2 + total_chn): frequency
                    # line (2 + total_chn + 1): nrates
                    # line (2 + total_chn + 2): samp, endsamp (assuming nrates=1)
                    # line (2 + total_chn + 3): start_timestamp_line (this is lines[total_channels + 4] if 0-indexed)
                    # line (2 + total_chn + 4): trigger_timestamp_line (this is lines[total_channels + 5] if 0-indexed)
                    # The original code used lines[signals + 5] and lines[signals + 6].
                    # If 'signals' means total_channels, and lines are 0-indexed:
                    # lines[0] = header
                    # lines[1] = counts
                    # lines[2...1+total_channels] = channel info
                    # lines[2+total_channels] = freq
                    # lines[3+total_channels] = nrates
                    # Assuming nrates = 1:
                    # lines[4+total_channels] = samp, endsamp
                    # lines[5+total_channels] = start_datetime
                    # lines[6+total_channels] = trigger_datetime

                    start_date_line_index = total_channels + 1 + 1 + 1 + 1 + 1 # 1 for 0-index, 1 for counts line, 1 for freq, 1 for nrates, 1 for samp/endsamp
                    trigger_date_line_index = start_date_line_index + 1

                    new_date_line = '01/01/0001,00:00:00.000000\n' # Adjusted time slightly from prompt for consistency

                    if len(lines) > trigger_date_line_index : # trigger_date_line_index is the higher index
                        lines[start_date_line_index] = new_date_line
                        lines[trigger_date_line_index] = new_date_line
                    else:
                        print(f"Warning (anonymize_file): Not enough lines in {cfg_file_path} (encoding {encoding}) to anonymize dates based on channel count {total_channels}.")
                else:
                    print(f"Warning (anonymize_file): Could not parse total channel count from {cfg_file_path} (encoding {encoding}). Dates not anonymized.")
            else:
                print(f"Warning (anonymize_file): CFG file {cfg_file_path} (encoding {encoding}) has too few lines for full anonymization.")

            # Calculate MD5 hash of the DAT file
            dat_hash = ""
            try:
                with open(dat_file_path, 'rb') as file:
                    dat_hash = hashlib.md5(file.read()).hexdigest()
            except IOError as e:
                print(f"Error (anonymize_file): Could not read DAT file {dat_file_path} for hashing: {e}")
                return False


            # Write modified CFG content back to the original file path
            try:
                with open(cfg_file_path, 'w', encoding=encoding) as file:
                    file.writelines(lines)
            except IOError as e:
                print(f"Error (anonymize_file): Could not write modified CFG to {cfg_file_path} (encoding {encoding}): {e}")
                return False

            # Define new hashed file names
            root_dir = os.path.dirname(cfg_file_path)
            new_cfg_hashed_path = os.path.join(root_dir, dat_hash + '.cfg')
            new_dat_hashed_path = os.path.join(root_dir, dat_hash + '.dat')

            # Rename CFG file
            current_cfg_path_abs = os.path.abspath(cfg_file_path)
            new_cfg_hashed_path_abs = os.path.abspath(new_cfg_hashed_path)

            if current_cfg_path_abs != new_cfg_hashed_path_abs:
                if os.path.exists(new_cfg_hashed_path):
                    print(f"Warning (anonymize_file): Target CFG file {new_cfg_hashed_path} already exists. Removing.")
                    os.remove(new_cfg_hashed_path)
                os.rename(cfg_file_path, new_cfg_hashed_path)
                # print(f"  Successfully renamed {cfg_file_path} to {new_cfg_hashed_path}") # Verbosity for anonymize_directory
            else:
                # print(f"  CFG file {cfg_file_path} is already named with its hash. Content updated.") # Verbosity for anonymize_directory
                pass


            # Rename DAT file
            current_dat_path_abs = os.path.abspath(dat_file_path)
            new_dat_hashed_path_abs = os.path.abspath(new_dat_hashed_path)

            if os.path.exists(dat_file_path): # Ensure DAT file exists before trying to rename
                if current_dat_path_abs != new_dat_hashed_path_abs:
                    if os.path.exists(new_dat_hashed_path):
                        print(f"Warning (anonymize_file): Target DAT file {new_dat_hashed_path} already exists. Removing.")
                        os.remove(new_dat_hashed_path)
                    os.rename(dat_file_path, new_dat_hashed_path)
                    # print(f"  Successfully renamed {dat_file_path} to {new_dat_hashed_path}") # Verbosity for anonymize_directory
                else:
                    # print(f"  DAT file {dat_file_path} is already named with its hash.") # Verbosity for anonymize_directory
                    pass
            else:
                # This case should have been caught much earlier, but as a safeguard:
                print(f"Error (anonymize_file): DAT file {dat_file_path} disappeared before renaming. This is critical.")
                # Attempt to revert CFG rename if possible? Or mark as total failure.
                # For now, this indicates a severe issue. The new CFG (hash.cfg) might exist without a DAT file.
                return False

            return True

        except FileNotFoundError: # Should primarily be caught by the initial DAT check or CFG read attempt.
            print(f"Error (anonymize_file): File not found during processing of {cfg_file_path} (encoding {encoding}).")
            return False
        # UnicodeDecodeError is caught specifically above.
        # General IOError for write operations or unexpected read issues not covered by specific catches.
        except IOError as e:
            print(f"Error (anonymize_file): File IO error for {cfg_file_path} (encoding {encoding}): {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred in anonymize_file for {cfg_file_path} (encoding {encoding}): {e}")
            return False

    def anonymize_directory(self, source_dir: str):
        failed_files_log = []
        processed_files_count = 0
        total_cfg_files_found = 0

        cfg_files_to_process = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(".cfg"):
                    cfg_files_to_process.append(os.path.join(root, file))

        total_cfg_files_found = len(cfg_files_to_process)
        if self.verbose_logging: # Changed to verbose_logging
            print(f"Found {total_cfg_files_found} .cfg files to process in directory {source_dir}.")

        for cfg_file_path in tqdm(cfg_files_to_process, total=total_cfg_files_found, desc="Anonymizing files", unit="file", disable=not self.show_progress_bars): # Added tqdm
            original_path_for_logging = cfg_file_path

            if self.verbose_logging: # Changed to verbose_logging
                print(f"Processing {original_path_for_logging}...")
            successfully_processed = False
            encodings_to_try = ['utf-8', 'windows-1251', 'cp866']

            # Path to check for existence during retries should be the original one.
            # If anonymize_file succeeds, it renames the file. The next iteration for this file (if any)
            # would correctly not find original_path_for_logging if it was renamed.

            current_path_to_try = original_path_for_logging

            for encoding_attempt in encodings_to_try:
                if not os.path.exists(current_path_to_try):
                    # This means the file was successfully processed and renamed by a previous encoding attempt.
                    # (or deleted, but success is implied if successfully_processed is True)
                    if successfully_processed:
                        if self.verbose_logging: print(f"  File {original_path_for_logging} already processed and renamed.")
                    else:
                        if self.verbose_logging: print(f"  File {original_path_for_logging} no longer exists, but wasn't marked as successfully processed. Skipping further attempts.")
                    break

                if self.verbose_logging: print(f"  Trying encoding: {encoding_attempt} for {current_path_to_try}")
                if self.anonymize_file(current_path_to_try, encoding=encoding_attempt):
                    successfully_processed = True
                    processed_files_count += 1
                    if self.verbose_logging: print(f"  Successfully anonymized {original_path_for_logging} using {encoding_attempt}. Files are now hash-named.")
                    break
                else:
                    if self.verbose_logging: print(f"  Failed with encoding {encoding_attempt} for {current_path_to_try}.")

            if not successfully_processed:
                failed_files_log.append(f"{original_path_for_logging} - Could not process with any supported encoding or crucial error (e.g., DAT missing or unreadable).")
                if self.verbose_logging: print(f"  Failed to anonymize {original_path_for_logging} after all attempts.")

        if failed_files_log:
            log_path = os.path.join(source_dir, 'protected_files.txt')
            try:
                with open(log_path, 'w', encoding='utf-8') as log_file:
                    for entry in failed_files_log:
                        log_file.write(entry + "\n")
                if self.verbose_logging: print(f"Wrote {len(failed_files_log)} failure logs to {log_path}")
            except IOError as e:
                if self.verbose_logging: print(f"Error writing protected_files.txt: {e}")

        if self.verbose_logging: print(f"Anonymization of directory {source_dir} complete. Successfully processed {processed_files_count} out of {total_cfg_files_found} found .cfg files.")
