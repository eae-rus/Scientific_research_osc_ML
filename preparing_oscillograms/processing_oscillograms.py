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
from collections import Counter

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

    def _extract_signal_names_from_cfg_lines(self, lines: list[str], file_path_for_error_msg: str, include_digital_signals: bool = True) -> tuple[list[str], str | None]:
        """
        Extracts all signal names from the lines of a CFG file.

        Args:
            lines (list[str]): Content of the CFG file as a list of strings.
            file_path_for_error_msg (str): Path to the file, used for error messages.

        Returns:
            tuple[list[str], str | None]: A tuple containing a list of signal names
                                           and an error message string if any error occurred, otherwise None.
        """
        signal_names = []
        
        if len(lines) < 2:
            return [], f"File {file_path_for_error_msg} has less than 2 lines."

        # Parse second line for signal counts
        try:
            parts = lines[1].split(',')
            if len(parts) < 3:
                return [], f"Second line in {file_path_for_error_msg} has incorrect format: {lines[1].strip()}"
            
            # total_signals_str = parts[0].strip() # Не используется напрямую для извлечения имен
            
            analog_signals_str = parts[1].strip().upper()
            digital_signals_str = parts[2].strip().upper()

            # Извлекаем числа, удаляя нечисловые символы (например, 'A' или 'D')
            count_analog_signals = int(re.sub(r'\D', '', analog_signals_str))
            count_digital_signals = int(re.sub(r'\D', '', digital_signals_str))

        except ValueError:
            return [], f"Could not parse signal counts from second line in {file_path_for_error_msg}: {lines[1].strip()}"
        except Exception as e:
            return [], f"Unexpected error parsing signal counts in {file_path_for_error_msg}: {e}"

        # Check if we have enough lines for analog signals
        if len(lines) < 2 + count_analog_signals:
            return [], f"File {file_path_for_error_msg} does not have enough lines for declared analog signals ({count_analog_signals}). Has {len(lines)} lines."

        # Extract analog signal names
        for i in range(count_analog_signals):
            line_index = 2 + i
            signal_line_parts = lines[line_index].split(',')
            if len(signal_line_parts) > 1:
                signal_names.append(signal_line_parts[1].strip())
            else:
                return [], f"Malformed analog signal line {line_index+1} in {file_path_for_error_msg}: {lines[line_index].strip()}"
        
        # Check if we have enough lines for digital signals
        if len(lines) < 2 + count_analog_signals + count_digital_signals:
            return [], f"File {file_path_for_error_msg} does not have enough lines for declared digital signals ({count_digital_signals})."

        # Extract digital signal names
        if include_digital_signals:
            # Check if we have enough lines for digital signals
            if len(lines) < 2 + count_analog_signals + count_digital_signals:
                # Если include_digital_signals=True, но строк не хватает, это ошибка.
                # Если include_digital_signals=False, эта проверка и извлечение не нужны.
                return [], f"File {file_path_for_error_msg} does not have enough lines for declared digital signals ({count_digital_signals})."

            for i in range(count_digital_signals):
                line_index = 2 + count_analog_signals + i
                signal_line_parts = lines[line_index].split(',')
                if len(signal_line_parts) > 1:
                    signal_names.append(signal_line_parts[1].strip())
                else:
                    return [], f"Malformed digital signal line {line_index+1} in {file_path_for_error_msg}: {lines[line_index].strip()}"
                    
        return signal_names, None
    
    def _parse_analog_signal_name_for_section(self, signal_name: str) -> dict | None:
            """
            Parses an analog signal name to extract its components, focusing on the section number.
            Example: "U | BusBar-1 | phase: A" -> {'prefix': "U | BusBar-", 'section': "1", 'suffix': " | phase: A"}
            Example: "I | Bus-12" -> {'prefix': "I | Bus-", 'section': "12", 'suffix': ""}
            Args:
                signal_name (str): The full name of the analog signal.
            Returns:
                dict | None: A dictionary with 'prefix', 'section', 'suffix' if parsable, else None.
            """
            parts = signal_name.split('|')
            if len(parts) < 2:
                return None

            signal_type_part = parts[0].strip()
            location_section_part = parts[1].strip()

            # Regex to find 'LocationType-Number' at the end of the location_section_part
            # It captures (anything before)-(digits)
            match = re.search(r"^(.*?)-(\d+)$", location_section_part)
            if not match:
                # Try another pattern if location type itself has hyphens, e.g. "Some-Location-Type-1"
                # This pattern looks for the last hyphen followed by digits.
                match_alternative = re.match(r"^(.*[A-Za-z_])-(\d+)$", location_section_part)
                if not match_alternative:
                    return None
                location_base = match_alternative.group(1)
                section_number_str = match_alternative.group(2)
            else:
                location_base = match.group(1)
                section_number_str = match.group(2)
                
            prefix = f"{signal_type_part} | {location_base}-"
            
            suffix = ""
            if len(parts) > 2:
                suffix_parts = [p.strip() for p in parts[2:]]
                suffix = " | " + " | ".join(suffix_parts)

            return {
                "prefix": prefix,
                "section": section_number_str,
                "suffix": suffix,
                "original_name": signal_name
            }

    def _rename_duplicate_analog_signals_in_lines(self, cfg_lines: list[str], file_path_for_log: str) -> tuple[list[str], bool, list[dict]]:
        """
        Identifies duplicate analog signal names in the provided CFG lines and renames them
        by assigning new, unused section numbers.
        Operates only on analog signals.

        Args:
            cfg_lines (list[str]): The content of the CFG file as a list of strings.
            file_path_for_log (str): The path of the file, for logging purposes.

        Returns:
            tuple:
                - list[str]: Modified list of CFG lines.
                - bool: True if any changes were made, False otherwise.
                - list[dict]: A log of renamed signals [{'file_path', 'line_index', 'old_name', 'new_name'}].
        """
        modified_lines = list(cfg_lines)
        made_changes = False
        rename_log = []

        if len(modified_lines) < 2:
            return modified_lines, False, rename_log

        try:
            parts = modified_lines[1].split(',')
            analog_signals_str = parts[1].strip().upper()
            count_analog_signals = int(re.sub(r'\D', '', analog_signals_str))
        except (ValueError, IndexError):
            return modified_lines, False, rename_log
        
        analog_signals_data = []
        current_used_section_numbers = set()
        all_current_analog_signal_names = set() 

        for i in range(count_analog_signals):
            line_idx = 2 + i
            if line_idx >= len(modified_lines):
                break
            
            line_content = modified_lines[line_idx]
            line_parts = line_content.split(',')
            if len(line_parts) <= 1:
                continue

            original_name = line_parts[1].strip()
            all_current_analog_signal_names.add(original_name)
            parsed_components = self._parse_analog_signal_name_for_section(original_name)
            
            analog_signals_data.append({
                "line_index": line_idx,
                "name": original_name,
                "parsed": parsed_components
            })
            if parsed_components and parsed_components['section'].isdigit():
                current_used_section_numbers.add(int(parsed_components['section']))

        signals_by_name = {}
        for data in analog_signals_data:
            if not data['parsed']:
                continue
            name = data['name']
            if name not in signals_by_name:
                signals_by_name[name] = []
            signals_by_name[name].append(data)

        for name, instances in signals_by_name.items():
            if len(instances) > 1:
                instances.sort(key=lambda x: x['line_index'])
                
                # Сохраняем информацию о первом (оставляемом) экземпляре
                first_instance_section = int(instances[0]['parsed']['section']) if instances[0]['parsed'] and instances[0]['parsed']['section'].isdigit() else None

                for k in range(1, len(instances)):
                    signal_to_rename_info = instances[k]
                    parsed_parts = signal_to_rename_info['parsed']
                    original_section_of_duplicate = int(parsed_parts['section']) if parsed_parts['section'].isdigit() else -1 # -1 если не число, чтобы не совпало

                    chosen_section_number = None
                    new_signal_name = None

                    # Попытка 1: Найти существующую секцию (не оригинальную для этого дубля и не секцию первого экземпляра, если они разные)
                    # где можно разместить сигнал без создания нового дубликата.
                    # Сортируем номера секций для предсказуемого поведения.
                    sorted_existing_sections = sorted(list(current_used_section_numbers))

                    for target_section in sorted_existing_sections:
                        # Не перемещаем в ту же секцию, откуда дубликат, если это не секция первого экземпляра,
                        # и не перемещаем в секцию первого экземпляра, если это не та же самая секция, откуда дубликат.
                        # Это условие сложное, проще: не перемещать в секцию первого экземпляра, если он там и остался.
                        # И не перемещать в секцию, откуда мы "выселяем" дубликат, если это не секция первого экземпляра.
                        # Главное - не создавать конфликт с УЖЕ СУЩЕСТВУЮЩИМИ сигналами в target_section.
                        
                        # Если мы пытаемся переназначить сигнал, который был в той же секции, что и первый (оставленный)
                        # экземпляр, то мы не можем использовать эту секцию снова для этого же типа сигнала.
                        # Пропускаем секцию, если она является секцией первого (оставленного) экземпляра *этого же имени*
                        if first_instance_section is not None and target_section == first_instance_section:
                            # Проверяем, не пытаемся ли мы создать дубликат с первым экземпляром
                            potential_check_name_against_first = f"{parsed_parts['prefix']}{target_section}{parsed_parts['suffix']}"
                            if potential_check_name_against_first == instances[0]['name']: # Сравниваем с именем первого экземпляра
                                continue # Нельзя, создаст дубликат с первым экземпляром

                        potential_new_name_in_existing_section = f"{parsed_parts['prefix']}{target_section}{parsed_parts['suffix']}"
                        
                        # Проверяем, существует ли УЖЕ такое имя в файле (после предыдущих переименований на этом шаге)
                        # или оно было изначально. all_current_analog_signal_names содержит начальные имена.
                        # Для проверки текущего состояния нужен более динамический список, либо просто проверка по modified_lines.
                        # Проще всего будет проверять по текущему состоянию modified_lines.
                        
                        is_slot_free = True
                        temp_all_names_in_modified_lines = set()
                        for line_idx_check in range(2, 2 + count_analog_signals):
                            if line_idx_check >= len(modified_lines): break
                            cfg_line_parts_check = modified_lines[line_idx_check].split(',')
                            if len(cfg_line_parts_check) > 1:
                                temp_all_names_in_modified_lines.add(cfg_line_parts_check[1].strip())
                        
                        if potential_new_name_in_existing_section in temp_all_names_in_modified_lines:
                            is_slot_free = False
                        
                        if is_slot_free:
                            chosen_section_number = target_section
                            new_signal_name = potential_new_name_in_existing_section
                            break # Нашли подходящую существующую секцию

                    # Попытка 2: Если не нашли места в существующих, создаем новую секцию
                    if chosen_section_number is None:
                        new_section_candidate = 1
                        while new_section_candidate in current_used_section_numbers:
                            new_section_candidate += 1
                        chosen_section_number = new_section_candidate
                        new_signal_name = f"{parsed_parts['prefix']}{chosen_section_number}{parsed_parts['suffix']}"
                    
                    # Обновляем строку и логи
                    line_idx_to_change = signal_to_rename_info['line_index']
                    cfg_line_parts = modified_lines[line_idx_to_change].split(',')
                    cfg_line_parts[1] = new_signal_name
                    modified_lines[line_idx_to_change] = ",".join(cfg_line_parts)
                    
                    made_changes = True
                    current_used_section_numbers.add(chosen_section_number) # Добавляем новую или подтверждаем использование существующей
                    # Если имя было изменено, его нужно обновить и в all_current_analog_signal_names для последующих проверок (если нужно)
                    # Но проще пересобирать temp_all_names_in_modified_lines на каждой итерации, как сделано выше.

                    rename_log.append({
                        'file_path': file_path_for_log,
                        'line_index': line_idx_to_change,
                        'old_name': signal_to_rename_info['name'],
                        'new_name': new_signal_name
                    })
        
        return modified_lines, made_changes, rename_log

    def find_duplicate_signal_names_in_cfg(self, 
                                           source_dir: str, 
                                           output_csv_duplicates_path: str, 
                                           include_digital_signals: bool = True,
                                           auto_rename_analog_duplicates: bool = False,
                                           output_csv_rename_log_path: str = "rename_log.csv"
                                           ) -> None:
        """
        Scans .cfg files for duplicate signal names. Optionally, renames duplicate *analog* signals.

        Args:
            source_dir (str): Directory containing .cfg files.
            output_csv_duplicates_path (str): Path to save CSV of files with duplicates.
            include_digital_signals (bool): Whether to include digital signals in the duplicate search.
            auto_rename_analog_duplicates (bool): If True, automatically renames duplicate analog signals.
            output_csv_rename_log_path (str): Path to save CSV log of renaming actions.
        """
        files_with_duplicates_overall = [] # Для CSV со списком файлов с дубликатами
        error_log_scan = [] 
        all_renaming_actions_log = [] # Для CSV с логом переименований

        total_cfg_files = 0
        for _, _, files_in_dir in os.walk(source_dir):
            for file_name in files_in_dir:
                if file_name.lower().endswith(".cfg"):
                    total_cfg_files += 1
        
        print(f"Total .cfg files to scan: {total_cfg_files}")
        encodings_to_try = ['utf-8', 'windows-1251', 'cp866']

        with tqdm(total=total_cfg_files, desc="Scanning CFG for duplicates") as pbar:
            for root, _, files_in_dir in os.walk(source_dir):
                for file_name in files_in_dir:
                    if not file_name.lower().endswith(".cfg"):
                        continue
                    
                    pbar.update(1)
                    file_path = os.path.join(root, file_name)
                    
                    file_content_lines = None
                    used_encoding = None

                    for enc in encodings_to_try:
                        try:
                            with open(file_path, 'r', encoding=enc) as f:
                                file_content_lines = f.readlines()
                            used_encoding = enc
                            break 
                        except Exception: # Более общее исключение для простоты
                            continue
                    
                    if file_content_lines is None:
                        error_log_scan.append(f"Could not read file {file_path} with any attempted encodings.")
                        continue

                    # 1. Анализ на дубликаты (как и раньше)
                    # Используем include_digital_signals для определения, какие сигналы считать
                    signal_names_for_duplicate_check, error_msg_extract = self._extract_signal_names_from_cfg_lines(
                        file_content_lines, file_path, include_digital_signals
                    )

                    if error_msg_extract:
                        error_log_scan.append(error_msg_extract)
                        # Продолжаем, чтобы попытаться переименовать, если auto_rename включен
                        # т.к. _extract_signal_names_from_cfg_lines мог споткнуться на цифровых,
                        # а аналоговые еще могут быть обработаны для переименования
                    
                    has_duplicates_in_file_for_log = False
                    if signal_names_for_duplicate_check: # Если извлечение имен для поиска дублей прошло успешно
                        name_counts = Counter(signal_names_for_duplicate_check)
                        for count in name_counts.values():
                            if count > 1:
                                has_duplicates_in_file_for_log = True
                                break
                    
                    if has_duplicates_in_file_for_log:
                        files_with_duplicates_overall.append({
                            'file_path': file_path,
                            'file_name': file_name
                        })

                    # 2. Автоматическое переименование АНАЛОГОВЫХ дубликатов (если включено)
                    if auto_rename_analog_duplicates:
                        # Важно: _rename_duplicate_analog_signals_in_lines работает с оригинальным file_content_lines
                        # и сама определяет аналоговые сигналы.
                        modified_lines, made_changes_flag, current_file_rename_log = \
                            self._rename_duplicate_analog_signals_in_lines(file_content_lines, file_path)
                        
                        if made_changes_flag:
                            try:
                                with open(file_path, 'w', encoding=used_encoding) as f_write:
                                    f_write.writelines(modified_lines)
                                all_renaming_actions_log.extend(current_file_rename_log)
                            except IOError as e:
                                error_log_scan.append(f"Error writing changes to {file_path}: {e}")
                        elif current_file_rename_log: # Если были ошибки внутри переименования, но флаг false
                             error_log_scan.append(f"Rename function reported issues for {file_path} but no changes made. Log: {current_file_rename_log}")


        # Сохранение CSV со списком файлов, где найдены дубликаты
        try:
            # Убедимся, что директория для output_csv_duplicates_path существует
            os.makedirs(os.path.dirname(output_csv_duplicates_path), exist_ok=True)
            with open(output_csv_duplicates_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['file_path', 'file_name']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(files_with_duplicates_overall)
            print(f"Successfully saved list of files with duplicate signal names to: {output_csv_duplicates_path}")
        except IOError as e:
            error_log_scan.append(f"Error writing duplicates CSV to {output_csv_duplicates_path}: {e}")

        # Сохранение CSV с логом переименований (если были)
        if auto_rename_analog_duplicates and all_renaming_actions_log:
            try:
                # Убедимся, что директория для output_csv_rename_log_path существует
                # Если путь относительный, он будет относительно текущей рабочей директории
                # Если путь абсолютный, то все ок.
                # Если output_csv_rename_log_path это просто имя файла, он создастся в CWD.
                # Для большей предсказуемости можно передавать полный путь или путь относительно source_dir
                rename_log_full_path = output_csv_rename_log_path
                if not os.path.isabs(rename_log_full_path): # Если путь не абсолютный
                     rename_log_full_path = os.path.join(os.getcwd(), output_csv_rename_log_path) # Сохраняем в текущей рабочей папке
                
                os.makedirs(os.path.dirname(rename_log_full_path), exist_ok=True)

                with open(rename_log_full_path, 'w', newline='', encoding='utf-8') as csvfile_rename:
                    fieldnames_rename = ['file_path', 'line_index', 'old_name', 'new_name']
                    writer_rename = csv.DictWriter(csvfile_rename, fieldnames=fieldnames_rename)
                    writer_rename.writeheader()
                    writer_rename.writerows(all_renaming_actions_log)
                print(f"Successfully saved renaming log to: {rename_log_full_path}")
            except IOError as e:
                error_log_scan.append(f"Error writing rename log CSV to {rename_log_full_path}: {e}")
        elif auto_rename_analog_duplicates:
             print("Auto-renaming was enabled, but no signals were renamed.")


        if error_log_scan:
            error_log_path = os.path.join(os.path.dirname(output_csv_duplicates_path), "scan_and_rename_errors.txt")
            try:
                with open(error_log_path, 'w', encoding='utf-8') as err_file:
                    for err in error_log_scan:
                        err_file.write(f"{err}\n")
                print(f"Scan/rename errors logged to: {error_log_path}")
            except IOError as e:
                 print(f"Could not write error log to {error_log_path}: {e}")
