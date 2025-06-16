import os
import sys
import pandas as pd
from pathlib import Path
import glob
from typing import List
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)

class Utils():
    def __init__(self):
        """
        Initialize the class
        """
        pass
    
    def merge_csv_files(self, input_files: list, output_file: str, chunksize=100000, is_print_messege = False):
        """
        Reads multiple large CSV files in chunks, merges them, and saves the result to a new file.
        
        :param input_files: list of paths to CSV files
        :param output_file: path to the output CSV file
        :param chunksize: number of rows to read at a time (default: 100000)
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)  # Create directory if needed
        
        with open(output_file, 'w', newline='') as outfile:
            header_written = False
            for file in input_files:
                try:
                    for chunk in pd.read_csv(file, chunksize=chunksize):
                        chunk.to_csv(outfile, index=False, header=not header_written, mode='a')
                        header_written = True  # Ensure header is written only once
                except Exception as e:
                    print(f"Error reading {file}: {e}")
        
        print(f"File saved: {output_file}")
        
    def process_cfg_files(self, input_folder, output_folder, mapping_file_path, file_list_path="processed_files.txt"):
        """
        Обрабатывает файлы с расширением .cfg:
        - Загружает маппинг замен из файла mapping_file_path. Каждая строка должна иметь формат:
            <искомая подстрока> -> <заменяемая подстрока>
        - Для каждого файла из input_folder проверяет каждую строку и, если находится одна из подстрок для замены, 
        выполняет замену (может быть более одной замены на файл).
        - Если в файле произведены изменения, сохраняет его в output_folder под тем же именем и добавляет имя файла
        в список processed_files.
        - Записывает список имён обработанных файлов в file_list_path.
        """
        # Создаём выходную папку, если её нет
        os.makedirs(output_folder, exist_ok=True)

        # Загружаем маппинг замен из файла
        mapping = {}
        with open(mapping_file_path, 'r', encoding='utf-8') as mf:
            for line in mf:
                line = line.strip()
                # Пропускаем пустые строки и комментарии
                if not line or line.startswith("#"):
                    continue
                if "->" in line:
                    old, new = line.split("->", 1)
                    mapping[old.strip()] = new.strip()
        
        processed_files = []

        # Обрабатываем все файлы с расширением .cfg во входной папке
        for file_path in glob.glob(os.path.join(input_folder, '*.cfg')):
            file_name = os.path.basename(file_path)

            # Чтение содержимого файла
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            file_modified = False
            # Проходим по всем строкам файла
            for i, line in enumerate(lines):
                # Для каждой подстроки из маппинга выполняем проверку и замену
                for old_substr, new_substr in mapping.items():
                    if old_substr in line:
                        # Если нужно заменить только первое вхождение, можно использовать параметр count=1
                        lines[i] = line.replace(old_substr, new_substr, 1)
                        file_modified = True
                        # Если требуется делать только одну замену в строке, можно добавить break здесь:
                        # break

            # Если в файле были изменения, сохраняем его и фиксируем имя файла
            if file_modified:
                processed_files.append(file_name)
                output_file_path = os.path.join(output_folder, file_name)
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

        # Записываем список имён обработанных файлов в указанный файл
        with open(file_list_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_files))
            
    # update_dataframe, correct_df, merge_normalization_csvs removed from here

    def filter_noise(self, csv_source: str, threshold: int, resave: bool = False, output_dir: str = "output"):
        """
        Фильтрует шумные группы в CSV-файле и сохраняет результаты.

        :param csv_source: путь или URL к исходному CSV-файлу
        :param threshold: максимальное число непустых change_event до того, как группа считается шумной
        :param resave: флаг, сохранять ли очищенный CSV с удалёнными записями
        :param output_dir: директория для сохранения выходных файлов
        :return: список префиксов, отмеченных для удаления
        """
        # Создаём папку для вывода, если её нет
        os.makedirs(output_dir, exist_ok=True)

        # Загружаем данные
        df = pd.read_csv(csv_source)

        # Выделяем префикс (часть до первого пробела)
        df['prefix'] = df['file_name'].astype(str).map(lambda x: x.split(' ')[0])

        # Группируем и считаем непустые значения change_event
        counts = (
            df.groupby('prefix')['change_event']
            .apply(lambda s: s.fillna('')
                            .astype(str)
                            .str.strip()
                            .astype(bool)
                            .sum())
        )

        # Определяем шумные префиксы
        noisy_prefixes = counts[counts > threshold].index.tolist()

        # Сохраняем текстовый файл со списком префиксов для удаления
        delete_list_path = os.path.join(output_dir, 'to_delete.txt')
        with open(delete_list_path, 'w') as f:
            for prefix in noisy_prefixes:
                f.write(f"{prefix}\n")

        # При необходимости сохраняем очищенный CSV
        if resave:
            cleaned = df[~df['prefix'].isin(noisy_prefixes)].drop(columns=['prefix'])
            cleaned_path = os.path.join(output_dir, 'cleaned.csv')
            cleaned.to_csv(cleaned_path, index=False)

        return noisy_prefixes

    @staticmethod
    def combine_json_hash_tables(source_dir: str, output_filename: str = "combined_hash_table.json",
                                 encoding: str = 'utf-8', is_print_message: bool = False) -> None:
        if not os.path.isdir(source_dir):
            if is_print_message: print(f"Error: Source directory '{source_dir}' not found.")
            return

        combined_table = {}
        json_files_found = []
        for root, _, files in os.walk(source_dir):
            # Avoid going into subdirectories that might be created by other processes if source_dir is reused for output
            if root == source_dir:
                for file_name in files:
                    if file_name.lower().endswith(".json") and file_name.lower() != output_filename.lower():
                        json_files_found.append(os.path.join(root, file_name))

        if is_print_message:
            print(f"Found {len(json_files_found)} JSON files to combine in '{source_dir}'.")

        for json_file_path in json_files_found:
            if is_print_message:
                print(f"Processing JSON file: {json_file_path}")
            try:
                with open(json_file_path, 'r', encoding=encoding) as file:
                    data = json.load(file)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key not in combined_table: # First-wins logic
                                combined_table[key] = value
                    elif is_print_message:
                        print(f"  Warning: Content of {json_file_path} is not a dictionary. Skipping.")
            except json.JSONDecodeError:
                if is_print_message:
                    print(f"  Error: Failed to decode JSON from {json_file_path}. Skipping.")
            except IOError as e:
                if is_print_message:
                    print(f"  Error reading file {json_file_path}: {e}. Skipping.")
            except Exception as e:
                if is_print_message:
                    print(f"  An unexpected error occurred with {json_file_path}: {e}. Skipping.")

        output_file_path = os.path.join(source_dir, output_filename)
        try:
            with open(output_file_path, 'w', encoding=encoding) as file:
                json.dump(combined_table, file, indent=4, ensure_ascii=False) # ensure_ascii=False for wider char support
            if is_print_message:
                print(f"Successfully combined JSON files. Output saved to: {output_file_path} with {len(combined_table)} unique keys.")
        except IOError as e:
            if is_print_message:
                print(f"Error writing combined JSON to {output_file_path}: {e}")
        except Exception as e:
            if is_print_message:
                print(f"An unexpected error occurred while writing {output_file_path}: {e}")
