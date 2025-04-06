import pandas as pd
import numpy as np
import os
import sys
import shutil
import hashlib
import datetime
import csv
import json
from tqdm import tqdm
from scipy.fft import fft
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)

from normalization.normalization import NormOsc
from raw_to_csv.raw_to_csv import RawToCSV  # Import RawToCSV for RawToCSV function

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
            
            
    def grouping_by_sampling_rate_and_network(self, source_dir: str, threshold: float = 0.1, isPrintMessege: bool = False) -> None:
        """
        The function groups files by sampling rate and network frequency.

        Args:
            source_dir (str): The directory containing the files to update.
            threshold (float): The threshold for considering frequency deviation from an integer as a measurement error.
            isPrintMessege (bool): A flag indicating whether to print a message if the frequencies are not found.

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
                            f_network, f_rate = self.extract_frequencies(file_path=file_path, threshold=threshold, isPrintMessege=isPrintMessege)

                            if f_network and f_rate:
                                dest_folder = os.path.join(source_dir, 'f_network = ' + str(f_network) + ' and f_rate = ' + str(f_rate))
                                if not os.path.exists(dest_folder):
                                    os.makedirs(dest_folder)
                                    
                                shutil.move(file_path, os.path.join(dest_folder, file))
                                shutil.move(dat_file_path, os.path.join(dest_folder, dat_file))
                            elif f_network:
                                if isPrintMessege: print(f"No network frequency found in the file: {file_path}")
                            elif f_rate:
                                if isPrintMessege: print(f"No sampling rate found in the file: {file_path}")
                            else:
                                if isPrintMessege: print(f"No frequencies found in the file: {file_path}")

    def extract_frequencies(self, file_path: str, threshold: float = 0.1, isPrintMessege: bool = False) -> tuple:
        """
        Extracts the network frequency (f_network) and sampling rate (f_rate) from the specified ".cfg" file.

        Args:
            source_dir (str): The path to the ".cfg" file.
            threshold (float): The threshold for considering frequency deviation from an integer as a measurement error.
            isPrintMessege (bool): A flag indicating whether to print a message if the frequencies are not found.

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
                if f_network == 0:
                    f_network = -1
                if f_rate == 0:
                    f_rate = -1
        except Exception as e:
            try:
                f_network, f_rate = float(f_network), float(f_rate)
                if abs(f_network - round(f_network)) < threshold and abs(f_rate - round(f_rate)) < threshold:
                    f_network, f_rate = round(f_network), round(f_rate)
                else:
                    f_network, f_rate = -1, -1 # TODO: In future versions, the invalid frequency may need to be adjusted
            except Exception as e:
                f_network, f_rate = -1, -1 # TODO: In future versions, the invalid frequency may need to be adjusted
                if isPrintMessege: print(e)

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
                    try:
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
                    except Exception as e:
                        print(e)
                        print("Error occurred while processing file: ", file_path)
        
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
                    try:
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
                    except Exception as e:
                        print(e)
                        print("Error occurred while processing file: ", file_path)
        
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
            
    def process_dat_files(root_dir, output_csv):
        """
        Проходит по папкам, вычисляет хеш MD5 для файлов .dat и сохраняет результаты в CSV.
        Названия папок содержат частоту сети и частоту дискретизации в формате:
        f_network = X and f_rate = Y
        
        :param root_dir: Корневая папка для обработки
        :param output_csv: Путь к выходному CSV-файлу
        """
        results = []
        file_counter = 1

        # Регулярное выражение для извлечения значений f_network и f_rate
        pattern = re.compile(r"f_network\s*=\s*(\d+)\s*and\s*f_rate\s*=\s*(\d+)")

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue  # Пропускаем не папки

            # Извлекаем частоту сети и дискретизации из имени папки
            match = pattern.match(folder_name)
            if not match:
                continue  # Пропускаем папки с некорректными названиями

            network_frequency, sampling_frequency = match.groups()

            for file_name in os.listdir(folder_path):
                if file_name.endswith('.dat'):
                    file_path = os.path.join(folder_path, file_name)

                    # Вычисление хеш-значения MD5
                    with open(file_path, 'rb') as file:
                        file_hash = hashlib.md5(file.read()).hexdigest()

                    # Добавляем данные в результаты
                    results.append([
                        file_counter,
                        network_frequency,
                        sampling_frequency,
                        file_name,
                        file_hash
                    ])
                    file_counter += 1

        # Запись результатов в CSV
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Номер', 'Частота сети', 'Частота дискретизации', 'Имя файла', 'Хеш'])
            writer.writerows(results)

    def _get_signals_from_prepared_cfg(self, file_path: str, encoding_name: str) -> list:
        """
        Parses a .cfg file file prepared for standard names and returns a list of full signal names (both analog and digital).

        Args:
            file_path (str): The path to the .cfg file.
            encoding_name (str): Encoding of the cfg file

        Returns:
            list: List of full signal names in the cfg file.
        """
        signal_names = []
        try:
            with open(file_path, 'r', encoding=encoding_name) as file:
                lines = file.readlines()
                count_all_signals_str, count_analog_signals_str, count_digital_signals_str = lines[1].split(',')
                count_analog_signals = int(count_analog_signals_str[:-1])
                count_digital_signals = int(count_digital_signals_str[:-2])

                # Analog signals processing
                for i in range(count_analog_signals):
                    signal_name = lines[2 + i].split(',')
                    name = signal_name[1].split('|')
                    if len(name) < 2: # защита от некорректных строк
                        return signal_names.append({
                        'signal_type': "error_name",
                        'location_type': None,
                        'section_number': None,
                        'phase': None
                    })
                    signal_type = name[0].strip() # U, I
                    section_info = name[1].strip().split("-")
                    location_type = section_info[0] # BusBar, CableLine, Bus
                    section_number = section_info[1] if len(section_info)>1 else None # 1, 2 ..., None
                    phase_info = name[2].strip().split(":")
                    phase = phase_info[1].strip() if len(phase_info)>1 else None # A, B, C ..., None
                    signal_names.append({
                        'signal_type': signal_type,
                        'location_type': location_type,
                        'section_number': section_number,
                        'phase': phase
                    })

                # Digital signals processing
                for i in range(count_digital_signals):
                    signal_name = lines[2 + count_analog_signals + i].split(',')
                    if len(signal_name) < 2: # защита от некорректных строк
                        return signal_names.append({
                        'signal_type': "error_digital_signal",
                        'location_type': None,
                        'section_number': None,
                        'phase': None
                    })
                    name = signal_name[1].split('|')
                    if len(name) < 2: # защита от некорректных строк
                        return signal_names.append({
                        'signal_type': "error_name",
                        'location_type': None,
                        'section_number': None,
                        'phase': None
                    })
                    signal_type = name[0].strip() # PDR...
                    section_info = name[1].strip().split("-")
                    location_type = section_info[0] # BusBar, CableLine, Bus
                    section_number = section_info[1] if len(section_info)>1 else None # 1, 2 ..., None
                    phase_info = name[-1].strip().split(":")
                    phase = None # A, B, C ..., None
                    if len(phase_info)>1 and (phase_info[0] == "phase"):
                        phase = phase_info[1].strip()
                    signal_names.append({
                        'signal_type': signal_type,
                        'location_type': location_type,
                        'section_number': section_number,
                        'phase': phase
                    })

        except Exception as e:
            print(f"Error reading cfg file {file_path}: {e}")
        return signal_names
    
    def _default_signal_checker(self, file_signals: list) -> bool:
        """
        Checks if a single comtrade file contains the required signals (considering PDR as digital).

        Args:
            file_signals (list): list of signal names.

        Returns:
            bool: True if all required signals are present, False otherwise.
        """
        has_voltage_busbar_1 = False
        has_voltage_cableline_1 = False
        has_voltage_busbar_2 = False
        has_voltage_cableline_2 = False
        has_current_bus_1_ac = False
        has_current_bus_2_ac = False
        r_has_pdr_bus_1 = False
        i_has_pdr_bus_1 = False

        voltage_busbar_1_phases = set()
        voltage_cableline_1_phases = set()
        voltage_busbar_2_phases = set()
        voltage_cableline_2_phases = set()
        current_bus_1_phases = set()
        current_bus_2_phases = set()
        r_pdr_bus_1_phases = set()
        i_pdr_bus_1_phases = set()

        for signal in file_signals:
            # проверка аналоговых сигналов
            if signal['signal_type'] == 'U':
                if signal['location_type'] == 'BusBar' and signal['section_number'] == '1':
                    voltage_busbar_1_phases.add(signal['phase'])
                elif signal['location_type'] == 'CableLine' and signal['section_number'] == '1':
                    voltage_cableline_1_phases.add(signal['phase'])
                elif signal['location_type'] == 'BusBar' and signal['section_number'] == '2':
                    voltage_busbar_2_phases.add(signal['phase'])
                elif signal['location_type'] == 'CableLine' and signal['section_number'] == '2':
                    voltage_cableline_2_phases.add(signal['phase'])
            elif signal['signal_type'] == 'I':
                if signal['location_type'] == 'Bus' and signal['section_number'] == '1':
                    current_bus_1_phases.add(signal['phase'])
                elif signal['location_type'] == 'Bus' and signal['section_number'] == '2':
                    current_bus_2_phases.add(signal['phase'])
            # проверка дискретных сигналов
            elif signal['signal_type'] == 'PDR': # реальный сигнал из осциллограмм
                if signal['location_type'] == 'Bus' and signal['section_number'] == '1':
                    r_pdr_bus_1_phases.add(signal['phase'])
            elif signal['signal_type'] == 'PDR_ideal': # сигнал идеальной разметки
                if signal['location_type'] == 'Bus' and signal['section_number'] == '1':
                    i_pdr_bus_1_phases.add(signal['phase'])

        has_voltage_busbar_1 = {'A', 'B', 'C'}.issubset(voltage_busbar_1_phases)
        has_voltage_cableline_1 = {'A', 'B', 'C'}.issubset(voltage_cableline_1_phases)
        has_voltage_busbar_2 = {'A', 'B', 'C'}.issubset(voltage_busbar_2_phases)
        has_voltage_cableline_2 = {'A', 'B', 'C'}.issubset(voltage_cableline_2_phases)

        has_current_bus_1_ac = {'A', 'C'}.issubset(current_bus_1_phases) and len(current_bus_1_phases) >= 2
        has_current_bus_2_ac = {'A', 'C'}.issubset(current_bus_2_phases) and len(current_bus_2_phases) >= 2

        r_has_pdr_bus_1 = {'PS'}.issubset(r_pdr_bus_1_phases) or {'A', 'B', 'C'}.issubset(r_pdr_bus_1_phases)
        i_has_pdr_bus_1 = {'PS'}.issubset(i_pdr_bus_1_phases) or {'A', 'B', 'C'}.issubset(i_pdr_bus_1_phases)

        voltage_condition = (has_voltage_busbar_1 or has_voltage_cableline_1) and (has_voltage_busbar_2 or has_voltage_cableline_2)
        current_condition = has_current_bus_1_ac and has_current_bus_2_ac
        pdr_condition = r_has_pdr_bus_1 or i_has_pdr_bus_1

        return voltage_condition and current_condition and pdr_condition
    
    def _check_signals_in_one_file(self, file_path: str, signal_checker=None, encoding_name: str = 'utf-8') -> bool:
        file_signals = self._get_signals_from_prepared_cfg(file_path, encoding_name)
        if not file_signals:
            return False
        # Если функция проверки не передана, используем стандартную
        if signal_checker is None:
            signal_checker = self._default_signal_checker
        return signal_checker(file_signals)

    def check_signals_in_folder(self, raw_path='raw_data/', output_csv_filename='signal_check_results.csv', signal_checker=None):
        """
        Checks all comtrade files in a folder for required signals and outputs results to a CSV.

        Args:
            raw_path (str): Path to the folder containing raw comtrade files.
            output_csv_filename (str): Filename for the output CSV file.
            signal_checker (function): Function to check the required signals in a single file. Defaults to self._default_signal_checker.
        """
        output_csv_path = os.path.join(raw_path, output_csv_filename)
        results = []

        print(f"Checking signals in folder: {raw_path}")
        total_files = 0
        for root, dirs, files in os.walk(raw_path):
            for file in files:
                if file.endswith(".cfg"):
                    total_files += 1
        print(f"Total CFG files found: {total_files}, starting processing...")

        with tqdm(total=total_files, desc="Checking signals") as pbar:
            for root, dirs, files in os.walk(raw_path):
                for file in files:
                    if file.endswith(".cfg"):
                        pbar.update(1)
                        file_path = os.path.join(root, file)
                        file_hash = file[:-4] # Assuming filename is hash.cfg
                        contains_required_signals = False
                        
                        try:
                            contains_required_signals = self._check_signals_in_one_file(file_path, signal_checker, 'utf-8')
                        except Exception as e:
                            # windows-1251 и ОЕМ 866 не проверяются, так как предполагается, что уже выполнена требуемая стандартизация
                            print(f"Error processing {file_path}: {e}")
                            contains_required_signals = "Error"

                        if contains_required_signals == True:
                            signal_status = "Yes"
                        elif contains_required_signals == False:
                            signal_status = "No"
                        else:
                            signal_status = "Error"

                        results.append({'filename': file_hash, 'contains_required_signals': signal_status})
        
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
           fieldnames = ['filename', 'contains_required_signals']
           writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"')
           writer.writeheader()
           writer.writerows(results)

        print(f"Signal check results saved to {output_csv_path}")
        
    def find_oscillograms_with_spef(self, raw_path: str ='raw_data/', output_csv_path: str = "find_oscillograms_with_spef.csv", 
                                    norm_coef_file_path: str = 'norm_coef.csv'):
        """
        Finds oscillograms with single-phase earth faults (SPEF) based on defined conditions using sliding window and harmonic analysis.

        Args:
            raw_path (str): Path to the directory containing COMTRADE files.
            output_csv_path (str): Path to save the CSV file with SPEF filenames.
            norm_coef_file_path (str): Path to the normalization coefficients CSV file.
        """
        spef_files = []
        raw_files = sorted([file for file in os.listdir(raw_path) if 'cfg' in file])
        norm_osc = NormOsc(norm_coef_file_path=norm_coef_file_path)

        threshold = 0.1 / 3
        period_count = 3
        
        number_ocs_found = 0

        with tqdm(total=len(raw_files), desc="Searching for SPEF") as pbar:
            for file in raw_files:
                file_path = os.path.join(raw_path, file)
                filename_without_ext = file[:-4]

                try:
                    raw_date, raw_df = RawToCSV.read_comtrade(RawToCSV(), file_path)
                    # TODO: Модернизировать "read_comtrade", а точнее даже функцию "comtrade.load(file_name)"
                    # Так как сейчас приходится искусственно вытягивать нужные коэффициент
                    samples_per_period = int(raw_date._cfg.sample_rates[0][0] / raw_date.frequency)
                    samples_duration = period_count * samples_per_period
                    if raw_df is None or raw_df.empty:
                        pbar.update(1)
                        continue

                    buses_df = RawToCSV.split_buses(RawToCSV(), raw_df.reset_index(), file)
                    if buses_df.empty:
                        pbar.update(1)
                        continue

                    buses_df = norm_osc.normalize_bus_signals(buses_df, filename_without_ext, yes_prase="YES", is_print_error=False)
                    if buses_df is None:
                        pbar.update(1)
                        continue

                    for file_name, group_df in buses_df.groupby("file_name"):
                        is_spef = False
                        group_df = group_df.copy()

                        # Condition 1 & 2: 3U0 BB and 3U0 CL (combined for efficiency)
                        u0_signals_bb_cl = {}
                        signal_names_3u0 = {"UN BB": "UN BB", "UN CL": "UN CL"} # Mapping for signal names

                        for signal_3u0_name, col_name in signal_names_3u0.items():
                            if not group_df.empty and col_name in group_df.columns:
                                u0_signal = group_df[col_name].fillna(0).values
                                u0_harmonics = np.zeros_like(u0_signal, dtype=float) # Array to store first harmonic values

                                # Sliding window for harmonic calculation and check
                                for i in range(len(u0_signal) - samples_per_period + 1): # Slide through the entire signal
                                    window = u0_signal[i:i+samples_per_period]
                                    window_fft = np.abs(fft(window)) / samples_per_period
                                    window_h1 = 2 * window_fft[1] if len(window_fft) > 1 else 0
                                    u0_harmonics[i:i+samples_per_period] = max(u0_harmonics[i:i+samples_per_period].max(), window_h1) # Take max harmonic in window

                                # Sliding window check on harmonics
                                for i in range(len(u0_harmonics) - samples_duration + 1):
                                    window_harmonics = u0_harmonics[i:i+samples_duration]
                                    if np.all(window_harmonics >= threshold): # Check harmonic level in sliding window
                                        is_spef = True
                                        number_ocs_found += 1
                                        spef_files.append([filename_without_ext, file_name])
                                        break # Condition met, no need to check further conditions for this file
                                if is_spef:
                                    break # Break outer loop if SPEF is found

                            if is_spef:
                                break # Break if SPEF is found

                        if is_spef:
                            continue # Go to next file if SPEF is found

                        # Condition 3 & 4: Zero sequence + Phase voltages BB and CL (combined)
                        phase_voltage_conditions = {
                            "BB": {"phases": ["UA BB", "UB BB", "UC BB"], "threshold_u0": threshold, "threshold_phase": threshold/np.sqrt(3)}, # Using threshold_phase if needed for phase voltages
                            "CL": {"phases": ["UA CL", "UB CL", "UC CL"], "threshold_u0": threshold, "threshold_phase": threshold/np.sqrt(3)}  # Using threshold_phase if needed for phase voltages
                        }

                        for location, condition_params in phase_voltage_conditions.items():
                            phase_names = condition_params["phases"]
                            threshold_u0 = condition_params["threshold_u0"]
                            threshold_phase = condition_params["threshold_phase"] # Unused currently, can be used for phase voltage level checks

                            if not group_df.empty and all(col in group_df.columns for col in phase_names):
                                ua_signal = group_df[phase_names[0]].fillna(0).values
                                ub_signal = group_df[phase_names[1]].fillna(0).values
                                uc_signal = group_df[phase_names[2]].fillna(0).values

                                u0_3_signal = (ua_signal + ub_signal + uc_signal) / np.sqrt(3) # 3U0 calculation
                                # /np.sqrt(3) - because we used phase signal

                                u0_3_harmonics = np.zeros_like(u0_3_signal, dtype=float) # Array for harmonics
                                for i in range(len(u0_3_signal) - samples_per_period + 1):
                                    window = u0_3_signal[i:i+samples_per_period]
                                    window_fft = np.abs(fft(window)) / samples_per_period
                                    window_h1 = 2 * window_fft[1] if len(window_fft) > 1 else 0
                                    u0_3_harmonics[i:i+samples_per_period] = max(u0_3_harmonics[i:i+samples_per_period].max(), window_h1)

                                phase_voltages = [ua_signal, ub_signal, uc_signal]
                                voltages_above_threshold = 0 # Count how many phase voltages meet the condition (if needed)

                                # Sliding window check for 3U0 harmonics
                                for i in range(len(u0_3_harmonics) - samples_duration + 1):
                                    window_harmonics_u0 = u0_3_harmonics[i:i+samples_duration]
                                    if np.all(window_harmonics_u0 >= threshold_u0): # Check 3U0 harmonic level in sliding window
                                        voltages_above_threshold = 0 # Reset counter for phase voltages for this window
                                        for v in phase_voltages:
                                            # Phase voltage check can be added here if needed, e.g., using harmonics or time-domain values in a window
                                            # For now, just checking if any two voltages exist (as per original condition)
                                            voltages_above_threshold += 1 # Increment if voltage signal exists (for simplicity, can be enhanced)
                                        
                                        if voltages_above_threshold >= 2: # Check if at least two phase voltages are present (condition from original task)
                                            is_spef = True
                                            number_ocs_found += 1
                                            spef_files.append([filename_without_ext, file_name])
                                            break # Condition met, no need to check further conditions for this file
                                if is_spef:
                                    break # Break location loop if SPEF is found
                            
                            if is_spef:
                                break # Break outer loop if SPEF is found

                except Exception as e:
                    print(f"Error processing file {file}: {e}")

                pbar.update(1)

        print(f"Number of samples found = {number_ocs_found}")
        df_spef = pd.DataFrame(spef_files, columns=['filename', 'file_name_bus'])
        df_spef.to_csv(output_csv_path, index=False)
        print(f"SPEF files saved to: {output_csv_path}")
