import os
import shutil
import hashlib
import datetime
import csv
import json
from tqdm import tqdm

class ProcessingOscillograms():
    """
    A class for processing COMTRADE waveforms for further public use.
    """
    def __init__(self):
        pass
        # There is nothing yet. In the future, it will be necessary to rework the approach of passed variables.
    

    def deleting_confidential_information_in_all_files(self, source_dir: str) -> None:
        """
        Identifies and processes confidential information in files ".cfg" in the specified source directory, 
        and also renames files with the extension.cfg in .data.

        The function searches for ".cfg" files in a given directory, tries to determine their encoding (utf-8, windows-1251 or ОЕМ 866)
        and collects file paths with potential confidential information. 
        It saves a list of protected files and any processing errors to a 'protected_files.txt file in the source directory.

        Args:
            source_dir (str): A directory containing .cfg files for verifying confidential information.

        Returns:
            None
        """
        protected_files = []
        print("We count the total number of files in the source directory...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Total number of files: {total_files}, starting processing...")
        with tqdm(total=total_files, desc="Deleting confidential information") as pbar:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    pbar.update(1)
                    if file.endswith(".cfg"):
                        file_path = os.path.join(root, file)
                        # TODO: to determine the encoding and resave the file in utf-8, you need to create a separate function
                        try: 
                            self._deleting_confidential_information_in_one_file(file_path, root, 'utf-8')
                        except Exception as e:
                            try:
                                self._deleting_confidential_information_in_one_file(file_path, root, 'windows-1251')  
                            except Exception as e:
                                try:
                                    self._deleting_confidential_information_in_one_file(file_path, root, 'ОЕМ 866') # ОЕМ - Russian
                                except Exception as e:
                                    protected_files.append(file_path)
                                    protected_files.append(f"An error occurred while processing the cfg file: {e}")
        with open(os.path.join(source_dir, 'protected_files.txt'), 'w') as file:
            file.write('\n'.join(protected_files))

    def _deleting_confidential_information_in_one_file(self, file_path: str, root: str, encoding_name: str) -> None:
        """
        The function reads the contents of the file. cfg performs encoding correction, 
        deletes local information, updates certain strings, calculates the hash of the data and
        renames the .cfg files and the corresponding files.dat based on the calculated hash.
        
        Args:
            file_path (str): The path to the .cfg file.
            root (str): the path to the root directory.
            encoding_name (str): The encoding format used to read the file.

        Returns:
            None
        """
        with open(file_path, 'r', encoding=encoding_name) as file:
            lines = file.readlines()
            # Deleting information about local information in the cfg file
            parts = lines[0].split(',')
            if len(parts) >= 2:
                lines[0] = ",," + parts[-1].strip() + "\n"
            # reading the number of signals
            signals, analog_signals, digital_signals = lines[1].split(',')
            signals = int(signals)
            new_date = '01/01/0001, 01:01:01.000000\n'
            # FIXME: The point is incorrectly determined when there are several sampling frequencies.
            lines[signals + 5] = new_date
            lines[signals + 6] = new_date

        with open(file_path, 'w', encoding='utf-8') as file:
            for line in lines:
                file.write(line)
        dat_file_path = file_path[:-4] + ".dat"
        with open(dat_file_path, 'rb') as file:
            file_hash = hashlib.md5(file.read()).hexdigest()
        os.rename(file_path, os.path.join(root, file_hash + '.cfg'))
        os.rename(dat_file_path, os.path.join(root, file_hash + '.dat'))


    def date_of_change_replacement(self, source_dir: str) -> None:
        """
        The function goes through all the files in this directory and sets the modification time of each file to the current date and time.

        Args:
            source_dir (str): The directory containing the files to update.

        Returns:
            None
        """
        current_date = datetime.datetime.now()

        print("We count the total number of files in the source directory...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Total number of files: {total_files}, starting processing...")
        with tqdm(total=total_files, desc="Deleting confidential information") as pbar:
            for root, dirs, files in os.walk(source_dir):
                for filename in files:
                    pbar.update(1)
                    file_path = os.path.join(root, filename)
                    file_stat = os.stat(file_path)
                    os.utime(file_path, times=(file_stat.st_atime, current_date.timestamp()))
            
            
    def grouping_by_sampling_rate_and_network(self, source_dir: str) -> None:
        """
        The function groups files by sampling rate and network frequency.

        Args:
            source_dir (str): The directory containing the files to update.

        Returns:
            None
        """
        print("We count the total number of files in the source directory...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Total number of files: {total_files}, starting processing...")
        with tqdm(total=total_files, desc="Grouping by sampling rate and network") as pbar:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    pbar.update(1)
                    if file.lower().endswith(".cfg"):
                        file_path = os.path.join(root, file)
                        dat_file = file[:-4] + ".dat"
                        dat_file_path = os.path.join(root, dat_file)
                        is_exist = os.path.exists(dat_file_path) 
                        if is_exist:
                            f_network, f_rate = self._extract_frequencies(file_path)

                            if f_network and f_rate:
                                dest_folder = os.path.join(source_dir, 'f_network = ' + str(f_network) + ' and f_rate = ' + str(f_rate))
                                if not os.path.exists(dest_folder):
                                    os.makedirs(dest_folder)
                                    
                                shutil.move(file_path, os.path.join(dest_folder, file))
                                shutil.move(dat_file_path, os.path.join(dest_folder, dat_file))

    def _extract_frequencies(self, file_path: str) -> tuple:
        """
        Extracts the network frequency (f_network) and sampling rate (f_rate) from the specified ".cfg" file.

        Args:
            source_dir (str): The path to the ".cfg" file.

        Returns:
            tuple: A tuple containing the extracted network frequency and sampling rate.
        """
        f_network, f_rate = 0, 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # FIXME: No protection against protected and/or erroneous files
                lines = file.readlines()
                if len(lines) >= 2:
                    count_all_signals_str, count_analog_signals_str, count_digital_signals_str = lines[1].split(',')
                    count_all_signals = int(count_all_signals_str)
                    f_network = lines[count_all_signals + 2][:-1]
                    f_rate, count = lines[count_all_signals + 4].split(',')
                    f_network, f_rate = int(f_network), int(f_rate)
        except Exception as e:
            f_network, f_rate = 1, 1 # TODO: In future versions, the invalid frequency may need to be adjusted
            print(e)

        return f_network, f_rate

    def find_all_name_analog_signals(self, source_dir: str) -> None:
        """
        The function searches for all the names of analog signals in the comtrade file and sorts them by frequency of use.

        Args:
            source_dir (str): The directory containing the files to update.

        Returns:
            None
        """
        analog_signals_name = {}
        print("We count the total number of files in the source directory...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Total number of files: {total_files}, starting processing...")
        with tqdm(total=total_files, desc="Find all name analog signals") as pbar:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    pbar.update(1)
                    if file.lower().endswith(".cfg"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as file:
                            # FIXME: No protection against protected and/or erroneous files
                            lines = file.readlines()
                            if len(lines) >= 2:
                                count_all_signals_str, count_analog_signals_str, count_digital_signals_str = lines[1].split(',')
                                count_analog_signals = int(count_analog_signals_str[:-1])
                                for i in range(count_analog_signals):
                                    analog_signal = lines[2 + i].split(',') # getting an analog signal
                                    # TODO: add a single function for generating a combined signal name
                                    name, phase = analog_signal[1], analog_signal[2] # we get the name, phase and unit of measurement
                                    name, phase = name.replace(' ', ''), phase.replace(' ', '')
                                    signal_name = name + ' | phase:' + phase # creating a combined signal name
                                    if signal_name not in analog_signals_name:
                                        analog_signals_name[signal_name] = 1
                                    else:
                                        analog_signals_name[signal_name] += 1
        
        sorted_analog_signals_name = {k: v for k, v in sorted(analog_signals_name.items(), key=lambda item: item[1], reverse=True)}      
        csv_file = os.path.join(source_dir, 'sorted_analog_signals_name.csv')
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Key', "universal_code", 'Value'])  # Write header
            for key, value in sorted_analog_signals_name.items():
                writer.writerow([key, "-", value])

    def find_all_name_digital_signals(self, source_dir: str) -> None:
        """
        The function searches for all the names of discrete signals in the comtrade file and sorts them by frequency of use.

        Args:
            source_dir (str): The directory containing the files to update.

        Returns:
            None
        """
        digital_signals_name = {}
        print("We count the total number of files in the source directory...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Total number of files: {total_files}, starting processing...")
        with tqdm(total=total_files, desc="Find all name digital signals") as pbar:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    pbar.update(1)
                    if file.lower().endswith(".cfg"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as file:
                            # FIXME: No protection against protected and/or erroneous files
                            lines = file.readlines()
                            if len(lines) >= 2:
                                count_all_signals_str, count_analog_signals_str, count_digital_signals_str = lines[1].split(',')
                                count_analog_signals = int(count_analog_signals_str[:-1])
                                count_digital_signals = int(count_digital_signals_str[:-2])
                                for i in range(count_digital_signals):
                                    digital_signal = lines[2 + count_analog_signals + i].split(',') # getting an analog signal
                                    if len(digital_signal) == 1:# protection against incorrect number of signals
                                        break
                                    signal_name = digital_signal[1] # getting the name
                                    if signal_name not in digital_signals_name:
                                        digital_signals_name[signal_name] = 1
                                    else:
                                        digital_signals_name[signal_name] += 1
        
        sorted_digital_signals_name = {k: v for k, v in sorted(digital_signals_name.items(), key=lambda item: item[1], reverse=True)}      
        csv_file = os.path.join(source_dir, 'sorted_digital_signals_name.csv')
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Key', "universal_code", 'Value'])  # Write header
            for key, value in sorted_digital_signals_name.items():
                writer.writerow([key, "-", value])


    def rename_analog_signals(self, source_dir: str, csv_dir: str, encoding: str = 'utf8', delimiter: str = ',') -> None:
        """
        The function searches for all the name analog signals that are present in the database and renames them to the standard codes.
        
        Args:
            source_dir (str): The directory containing the files to update.
            csv_dir (str): The address of the csv file.
            encoding (str, optional): encoding of the csv file. Defaults to 'utf8'.
            delimiter (str, optional): delimiter in the csv file. Defaults to ','.

        Returns:
            None
        """
        code_map = {}
        with open(csv_dir, mode='r', encoding=encoding) as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                key = row['Key']
                universal_code = row['universal_code']
                is_name_determined = universal_code != '-' and universal_code != '?'
                if is_name_determined:
                    code_map[key] = universal_code
        
        print("We count the total number of files in the source directory...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Total number of files: {total_files}, starting processing...")
        with tqdm(total=total_files, desc="Rename analog signals") as pbar:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    pbar.update(1)
                    if file.lower().endswith(".cfg"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as file:
                            # FIXME: No protection against protected and/or erroneous files
                            lines = file.readlines()
                            if len(lines) >= 2:
                                count_all_signals_str, count_analog_signals_str, count_digital_signals_str = lines[1].split(',')
                                count_analog_signals = int(count_analog_signals_str[:-1])
                                for i in range(count_analog_signals):
                                    analog_signal = lines[2 + i].split(',') # getting an analog signal
                                    # TODO: add a single function for generating a combined signal name
                                    name, phase = analog_signal[1], analog_signal[2] # we get the name, phase and unit of measurement
                                    name, phase = name.replace(' ', ''), phase.replace(' ', ''),
                                    signal_name = name + ' | phase:' + phase # creating a combined signal name
                                    if signal_name in code_map:
                                        analog_signal[1] = code_map[signal_name]
                                        lines[2 + i] = ','.join(analog_signal)

                        with open(file_path, 'w', encoding='utf-8') as file:
                            file.writelines(lines)

    def rename_digital_signals(self, source_dir: str, csv_dir: str, encoding: str = 'utf8', delimiter: str = ',') -> None:
        """
        The function searches for all the name discrete signals that are present in the database and renames them to the standard codes.
        
        Args:
            source_dir (str): The directory containing the files to update.
            csv_dir (str): The address of the csv file.
            encoding (str, optional): encoding of the csv file. Defaults to 'utf8'.
            delimiter (str, optional): delimiter in the csv file. Defaults to ','.

        Returns:
            None
        """
        code_map = {}
        with open(csv_dir, mode='r', encoding=encoding) as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                key = row['Key']
                universal_code = row['universal_code']
                is_name_determined = universal_code != '-' and universal_code != '?'
                if is_name_determined:
                    code_map[key] = universal_code
        
        print("We count the total number of files in the source directory...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Total number of files: {total_files}, starting processing...")
        with tqdm(total=total_files, desc="Rename digital signals") as pbar:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    pbar.update(1)
                    if file.lower().endswith(".cfg"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as file:
                            # FIXME: No protection against protected and/or erroneous files
                            lines = file.readlines()
                            if len(lines) >= 2:
                                count_all_signals_str, count_analog_signals_str, count_digital_signals_str = lines[1].split(',')
                                count_analog_signals = int(count_analog_signals_str[:-1])
                                count_digital_signals = int(count_digital_signals_str[:-2])
                                for i in range(count_digital_signals):
                                    digital_signal = lines[2 + count_analog_signals + i].split(',') # getting an analog signal
                                    if len(digital_signal) == 1: # protection against incorrect number of signals
                                        break
                                    signal_name = digital_signal[1] # getting the name
                                    if signal_name in code_map:
                                        digital_signal[1] = code_map[signal_name]
                                        lines[2 + count_analog_signals + i] = ','.join(digital_signal)

                        with open(file_path, 'w', encoding='utf-8') as file:
                            file.writelines(lines)
                        
    def rename_one_signals(self, source_dir: str, old_name: str, new_name: str) -> None:
        """
        The function searches for the entire signal name with one name and changes it to a new one.
        
        Args:
            source_dir (str): The directory containing the files to update.
            old_name (str): the old name of the signal.
            new_name (str): the new name of the signal.

        Returns:
            None
        """
        print("We count the total number of files in the source directory...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Total number of files: {total_files}, starting processing...")
        with tqdm(total=total_files, desc="Rename signal") as pbar:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    pbar.update(1)
                    if file.lower().endswith(".cfg"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as file:
                            # FIXME: No protection against protected and/or erroneous files
                            lines = file.readlines()
                            if len(lines) >= 2:
                                for i in range(len(lines)):
                                    if old_name in lines[i]:
                                        lines[i] = lines[i].replace(old_name, new_name)

                        with open(file_path, 'w', encoding='utf-8') as file:
                            file.writelines(lines)

    def combining_databases_of_unique_codes(self, old_csv_file_path: str, new_csv_file_path: str, merged_csv_file_path: str,
                                            encoding_old_csv: str = 'utf-8', encoding_new_csv: str = 'utf-8',
                                            deelimed_old_csv_file: str = ',', deelimed_new_csv_file: str = ',', is_merge_files: bool = True) -> None:
        """
        The function combines csv files with unique signal codes
        
        Args:
            old_csv_file_path (str): The address of the csv file with unique signal codes.
            new_csv_file_path (str): The address of the csv file with unique signal codes.
            encoding_old_csv (str): encoding of the old csv file.
            encoding_new_csv (str): encoding of the new csv file.
            deelimed_old_csv_file (str): separator of the old csv file.
            deelimed_new_csv_file (str): delimiter of the new csv file.

        Returns:
            None
        """
        old_code_map = {}
        with open(old_csv_file_path, mode='r', encoding=encoding_old_csv) as file:
            reader = csv.DictReader(file, delimiter=deelimed_old_csv_file)
            for row in reader:
                key = row['Key']
                universal_code = row['universal_code']
                value = row['Value']
                is_name_determined = universal_code != '-' and universal_code != '?'
                if is_name_determined or is_merge_files:
                    old_code_map[key] = (universal_code, value)

        new_code_map = {}
        with open(new_csv_file_path, mode='r', encoding=encoding_new_csv) as file:
            reader = csv.DictReader(file, delimiter=deelimed_new_csv_file)
            for row in reader:
                key = row['Key']
                universal_code = row['universal_code']
                value = row['Value']
                new_code_map[key] = (universal_code, value)
                

        # If is_merge_files is set to True, combine the array with the summed values in the value field
        merged_code_map = dict()
        if is_merge_files:
            merged_code_map = old_code_map.copy()
            
        for key, value in new_code_map.items():
            if is_merge_files:
                if key not in merged_code_map:
                    merged_code_map[key] = value
                else:
                    old_value = merged_code_map[key][1]
                    new_value = value[1]
                    merged_value = int(old_value) + int(new_value)
                    merged_code_map[key] = (merged_code_map[key][0], str(merged_value))
            else:
                if key in old_code_map:
                    merged_code_map[key] = (old_code_map[key][0] , value[1])
                else:
                    merged_code_map[key] = value
        
        sorted_code_map = dict(sorted(merged_code_map.items(), key=lambda item: int(item[1][1]), reverse=True))
        with open(merged_csv_file_path, mode='w', encoding='utf-8', newline='') as new_file:
            writer = csv.writer(new_file, delimiter=deelimed_new_csv_file)
            writer.writerow(['Key', 'universal_code', 'Value'])
            for key, (universal_code, value) in sorted_code_map.items():
                writer.writerow([key, universal_code, value])

    def combining_json_hash_table(self, source_dir: str, encoding: str = 'utf-8') -> None:
        """
        !!ATTENTION!!
        It is worth using "hash_table" only for combining files. The function combines json files with unique names of waveforms
        
        Args:
            source_dir (str): The directory containing the files to update.

        Returns:
            None
        """
        combine_hash_table = {}
        print("We count the total number of files in the source directory...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Total number of files: {total_files}, starting processing...")
        with tqdm(total=total_files, desc="Combining json hash table") as pbar:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    pbar.update(1)
                    if file.lower().endswith(".json"):
                        try:
                            path = os.path.join(root, file)
                            with open(path, 'r', encoding=encoding) as file:
                                hash_table = json.load(file)
                                for key, value in hash_table.items():
                                    if key not in combine_hash_table:
                                        combine_hash_table[key] = value
                        except:
                            print("Failed to read hash_table from JSON file")
                            
        try:
            combine_hash_table_file_path = os.path.join(source_dir, 'combine_hash_table.json')
            with open(combine_hash_table_file_path, 'w') as file:
                json.dump(combine_hash_table, file)
        except:
            print("Failed to save new_hash_table to a JSON file")


    def research_coorect_encoding_in_cfg(self, source_dir: str, act_function = None) -> None:
        """
        The function searches for .cfg files in the specified directory, 
        tries to determine their encoding (utf-8, windows-1251 or ОЕМ 866) and collects file paths. 
        It saves a list of protected files and any processing errors to a 'protected_files.txt file in the source directory.
        
        The function specified in the input data is applied to the required files

        Args:
            source_dir (str): A directory containing .cfg files for verifying confidential information.
            act_function (function): The function that will be applied to the files.

        Returns:
            None
        """
        protected_files = []
        print("We count the total number of files in the source directory...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Total number of files: {total_files}, starting processing...")
        with tqdm(total=total_files, desc="Finding the correct encoding and determining the date") as pbar:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    pbar.update(1)
                    if file.endswith(".cfg"):
                        file_path = os.path.join(root, file)
                        # TODO: to determine the encoding and resave a file in utf-8, you need to create a separate function
                        try: 
                            act_function(file_path, root, 'utf-8')
                        except Exception as e:
                            try:
                                act_function(file_path, root, 'windows-1251')  
                            except Exception as e:
                                try:
                                    act_function(file_path, root, 'ОЕМ 866') # ОЕМ - Russian
                                except Exception as e:
                                    protected_files.append(file_path)
                                    protected_files.append(f"An error occurred while processing the cfg file: {e}")
        with open(os.path.join(source_dir, 'protected_files.txt'), 'w', encoding='utf-8') as file:
            file.write('\n'.join(protected_files))

    def detect_date(self, file_path: str, root: str, encoding_name: str, dict_all_dates = {}) -> None:
        """
        The function determines the date of the event in the cfg file.
        
        Args:
            file_path (str): The path to the .cfg file.
            root (str): the path to the root directory.
            encoding_name (str): The encoding format used to read the file.

        Returns:
            None
        """
        with open(file_path, 'r', encoding=encoding_name) as file:
            lines = file.readlines()
            count_all_signals_str, count_analog_signals_str, count_digital_signals_str = lines[1].split(',')
            count_all_signals = int(count_all_signals_str)
            
            # We only change the year in the event date
            start_date = lines[count_all_signals + 5].split(',')
            start_date_parts = start_date[0].split('/')
            start_time_parts = start_date[1].split(':')
            end_date = lines[count_all_signals + 6].split(',')
            end_date_parts = end_date[0].split('/')
            end_time_parts = end_date[1].split(':')

            dict_date = {
                'start_date': {
                    'year': start_date_parts[2],
                    'month': start_date_parts[1],
                    'day': start_date_parts[0],
                    'hour': start_time_parts[0],
                    'minute': start_time_parts[1],
                    'second': start_time_parts[2]
                },
                'end_date': {
                    'year': end_date_parts[2],
                    'month': end_date_parts[1],
                    'day': end_date_parts[0],
                    'hour': end_time_parts[0],
                    'minute': end_time_parts[1],
                    'second': end_time_parts[2]
                }
            }
            dat_file_path = file_path[:-4] + ".dat"
            with open(dat_file_path, 'rb') as file:
                file_hash = hashlib.md5(file.read()).hexdigest()

            dict_all_dates[file_hash] = dict_date
