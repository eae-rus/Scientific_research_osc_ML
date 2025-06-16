import os
import datetime
import re # Not strictly needed for these two methods yet, but good for the class overall
import shutil
import csv
import hashlib

class FileOrganizer:
    def __init__(self):
        pass

    def update_modification_times(self, target_dir: str, new_mod_time: datetime.datetime = None) -> None:
        if not os.path.isdir(target_dir):
            print(f"Error: Directory not found: {target_dir}")
            return

        if new_mod_time is None:
            new_mod_time_ts = datetime.datetime.now().timestamp()
        else:
            new_mod_time_ts = new_mod_time.timestamp()

        total_files = 0
        # First pass to count total files for progress reporting
        for r, d, files in os.walk(target_dir):
            total_files += len(files)

        if total_files == 0:
            print(f"No files found in {target_dir} to update.")
            return

        print(f"Total files to update modification time: {total_files}")

        current_file_count = 0
        for root, dirs, files in os.walk(target_dir):
            for filename in files:
                current_file_count += 1
                file_path = os.path.join(root, filename)
                try:
                    # For os.utime, the first element of the tuple is access time, second is modification time.
                    # We preserve the current access time and update only the modification time.
                    current_stat = os.stat(file_path)
                    os.utime(file_path, times=(current_stat.st_atime, new_mod_time_ts))
                    if current_file_count % 100 == 0 or current_file_count == total_files: # Print progress
                        print(f"Updated {current_file_count}/{total_files} files...")
                except Exception as e:
                    print(f"Error updating time for {file_path}: {e}")
        print(f"Finished updating modification times for {total_files} files in {target_dir}.")

    def extract_frequencies_from_cfg(self, cfg_file_path: str, encoding: str, threshold: float = 0.1) -> tuple[int | None, int | None]:
        f_network_val: int | None = None
        f_rate_val: int | None = None

        try:
            with open(cfg_file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()

            if len(lines) < 2: # Basic check for minimum lines
                return None, None

            # Get total channels from the second line (index 1)
            signal_counts_parts = lines[1].strip().split(',')
            if not signal_counts_parts or not signal_counts_parts[0].strip().isdigit():
                return None, None # Cannot parse signal count
            count_all_signals = int(signal_counts_parts[0].strip())

            # Line indexing based on 0-indexed `lines` list:
            # lines[0]: Station info
            # lines[1]: Channel counts (e.g., "6,3A,3D")
            # lines[2] to lines[1 + count_all_signals]: Channel definitions
            # lines[2 + count_all_signals]: Network frequency (f_network)
            # lines[3 + count_all_signals]: nrates (number of sample rates)
            # lines[4 + count_all_signals]: First sample rate (samp, endsamp) - this is f_rate
            # This assumes nrates=1 for simplicity, replicating original logic.

            f_network_line_idx = 2 + count_all_signals
            f_rate_line_idx = 4 + count_all_signals

            if len(lines) > f_network_line_idx:
                f_network_str = lines[f_network_line_idx].strip()
                if f_network_str:
                    try:
                        f_network_float = float(f_network_str)
                        if abs(f_network_float - round(f_network_float)) < threshold:
                            f_network_val = round(f_network_float)
                        # else: value is not close to an integer, considered invalid by this logic for these specific fields
                        if f_network_val == 0: f_network_val = None
                    except ValueError:
                        pass # Could not parse, f_network_val remains None
            # else: Not enough lines, f_network_val remains None

            if len(lines) > f_rate_line_idx:
                f_rate_line_content = lines[f_rate_line_idx].strip()
                f_rate_str_parts = f_rate_line_content.split(',')
                if f_rate_str_parts and f_rate_str_parts[0].strip():
                    try:
                        f_rate_float = float(f_rate_str_parts[0].strip())
                        if abs(f_rate_float - round(f_rate_float)) < threshold:
                            f_rate_val = round(f_rate_float)
                        # else: value is not close to an integer
                        if f_rate_val == 0: f_rate_val = None
                    except ValueError:
                        pass # Could not parse, f_rate_val remains None
            # else: Not enough lines, f_rate_val remains None

        except UnicodeDecodeError:
            return None, None
        except IOError:
            return None, None
        except Exception:
            return None, None

        return f_network_val, f_rate_val

    def group_files_by_frequency_and_rate(self, source_dir: str, threshold: float = 0.1, is_print_message: bool = False) -> None:
        if not os.path.isdir(source_dir):
            if is_print_message:
                print(f"Error: Source directory not found: {source_dir}")
            return

        cfg_files_to_process = []
        # Collect all CFG files first, excluding those in potential target subdirectories
        for root, _, files in os.walk(source_dir):
            # Avoid processing files in already created frequency/rate subfolders
            # This check assumes subfolders are direct children of source_dir and named specifically.
            if root == source_dir or not os.path.basename(root).startswith("f_network ="):
                for file in files:
                    if file.lower().endswith(".cfg"):
                        cfg_files_to_process.append(os.path.join(root, file))
            elif is_print_message and os.path.basename(root).startswith("f_network ="):
                 print(f"Skipping already processed directory: {root}")


        total_files_to_scan = len(cfg_files_to_process)
        if is_print_message:
            print(f"Found {total_files_to_scan} .cfg files to potentially group in {source_dir} (excluding sub-folders like 'f_network = ...').")

        processed_count = 0
        moved_files_count = 0
        for cfg_file_path in cfg_files_to_process:
            processed_count +=1
            if is_print_message and processed_count % 50 == 0:
                print(f"Processing file {processed_count}/{total_files_to_scan}: {os.path.basename(cfg_file_path)}")

            encodings_to_try = ['utf-8', 'windows-1251', 'cp866']
            extracted_f_network, extracted_f_rate = None, None
            used_encoding = None

            for encoding in encodings_to_try:
                f_network, f_rate = self.extract_frequencies_from_cfg(cfg_file_path, encoding=encoding, threshold=threshold)
                if f_network is not None and f_rate is not None:
                    extracted_f_network, extracted_f_rate = f_network, f_rate
                    used_encoding = encoding
                    break

            if extracted_f_network is not None and extracted_f_rate is not None:
                dest_folder_name = f'f_network = {extracted_f_network} and f_rate = {extracted_f_rate}'
                # Place new folders directly under source_dir
                dest_folder_path = os.path.join(source_dir, dest_folder_name)

                try:
                    os.makedirs(dest_folder_path, exist_ok=True)
                except OSError as e:
                    if is_print_message:
                        print(f"  Error creating directory {dest_folder_path}: {e}. Skipping file {os.path.basename(cfg_file_path)}.")
                    continue # Skip to next file if directory creation fails

                dat_file_path = os.path.splitext(cfg_file_path)[0] + ".dat"

                if os.path.exists(dat_file_path):
                    try:
                        # Ensure target filenames are just basenames, not full paths
                        target_cfg_path = os.path.join(dest_folder_path, os.path.basename(cfg_file_path))
                        target_dat_path = os.path.join(dest_folder_path, os.path.basename(dat_file_path))

                        shutil.move(cfg_file_path, target_cfg_path)
                        shutil.move(dat_file_path, target_dat_path)
                        moved_files_count +=1
                        if is_print_message:
                            print(f"  Moved {os.path.basename(cfg_file_path)} (and .dat) to {dest_folder_name} (encoding: {used_encoding})")
                    except Exception as e:
                        if is_print_message:
                            print(f"  Error moving files for {os.path.basename(cfg_file_path)}: {e}")
                elif is_print_message:
                    print(f"  Warning: DAT file not found for {cfg_file_path} (needed for grouping). CFG file not moved.")
            elif is_print_message:
                print(f"  Could not determine valid frequencies for {cfg_file_path} after trying all encodings. File not moved.")

        if is_print_message:
            print(f"Finished grouping files. Moved {moved_files_count} pairs of CFG/DAT files into frequency/rate specific subdirectories.")

    def generate_dat_hash_inventory(self, root_dir: str, output_csv_path: str) -> None:
        if not os.path.isdir(root_dir):
            print(f"Error: Root directory for inventory not found: {root_dir}")
            return

        results = []
        file_counter = 1
        # Regex to extract f_network and f_rate from folder names like "f_network = 50 and f_rate = 1000"
        pattern = re.compile(r"f_network\s*=\s*(\d+)\s*and\s*f_rate\s*=\s*(\d+)")

        print(f"Generating DAT hash inventory for directory: {root_dir}")

        for item_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, item_name)
            if os.path.isdir(folder_path):
                match = pattern.match(item_name)
                if not match:
                    # print(f"Skipping folder (name does not match pattern): {item_name}")
                    continue

                network_frequency, sampling_frequency = match.groups()
                # print(f"Processing folder: {item_name} (Net: {network_frequency}, Rate: {sampling_frequency})")

                for dat_file_name in os.listdir(folder_path):
                    if dat_file_name.lower().endswith('.dat'):
                        dat_file_full_path = os.path.join(folder_path, dat_file_name)
                        try:
                            with open(dat_file_full_path, 'rb') as file_content:
                                file_hash = hashlib.md5(file_content.read()).hexdigest()

                            results.append([
                                file_counter,
                                str(network_frequency), # Ensure frequencies are strings for CSV consistency if needed
                                str(sampling_frequency),
                                dat_file_name,  # Store only the DAT filename
                                file_hash
                            ])
                            file_counter += 1
                        except IOError as e:
                            print(f"Error reading DAT file {dat_file_full_path}: {e}")
                        except Exception as e:
                            print(f"Unexpected error processing DAT file {dat_file_full_path}: {e}")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty string
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")
            except OSError as e:
                print(f"Error creating output directory {output_dir}: {e}")
                return # Cannot write CSV if output dir creation fails

        csv_header = ['Номер', 'Частота сети', 'Частота дискретизации', 'Имя файла', 'Хеш']
        try:
            with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(csv_header)
                if results:
                    writer.writerows(results)

            if results:
                print(f"DAT hash inventory successfully saved to: {output_csv_path} with {len(results)} entries.")
            else:
                print(f"No DAT files found in matching subdirectories. Empty inventory CSV with headers created at: {output_csv_path}")

        except IOError as e:
            print(f"Error writing inventory CSV to {output_csv_path}: {e}")
        except Exception as e:
            print(f"Unexpected error writing inventory CSV {output_csv_path}: {e}")
