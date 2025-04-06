import os
import sys
import pandas as pd
from pathlib import Path
import glob

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
            
    def update_dataframe(self, df_path, processed_files_path, output_df_path):
        """
        Обновляет DataFrame на основе списка обработанных файлов.
        
        Аргументы:
        df_path: путь к файлу с исходным DataFrame (например, CSV)
        processed_files_path: путь к файлу, содержащему список имён файлов (по одному в строке)
        output_df_path: путь для сохранения обновлённого DataFrame
        
        Функция:
        - Загружает DataFrame
        - Загружает список имён файлов
        - Находит строки, где столбец "name" совпадает с одним из имён из списка
        - Обновляет значения столбцов "2Ip_PS", "2Ip_base", "2Ip_h1" и "2Ip_hx"
        - Сохраняет изменённый DataFrame по указанному пути
        """
        # Загружаем DataFrame
        df = pd.read_csv(df_path)
        
        # Читаем список обработанных файлов
        with open(processed_files_path, 'r', encoding='utf-8') as f:
            processed_files = [line.strip()[:-4] for line in f if line.strip()[:-4]]
        
        # Определяем маску для строк, где столбец "name" содержится в списке processed_files
        mask = df['name'].isin(processed_files)
        
        # Выполняем замену значений для найденных строк
        df.loc[mask, '2Ip_PS'] = "s"
        df.loc[mask, '2Ip_base'] = "5"
        df.loc[mask, '2Ip_h1'] = "5"
        df.loc[mask, '2Ip_hx'] = "1"
        
        # Сохраняем обновлённый DataFrame по указанному пути
        df.to_csv(output_df_path, index=False)

    def correct_df(self, input_path: str, output_path: str):
        df = pd.read_csv(input_path)
        
        # Фильтруем строки, где norm равно 'NO' или 'hz'
        mask = df['norm'].isin(['NO', 'hz'])
        
        # Функция для корректировки значений в столбцах
        def correct_values(row, prefix, target_value, default_ps, default_base):
            candidates = [f"{i}{prefix}_base" for i in range(1, 7)]
            values = [row[col] for col in candidates if str(row[col]) in target_value]
            
            # Определяем заменяемое значение
            if '100' in values:
                replacement = '100'
            elif '400' in values:
                replacement = '400'
            else:
                replacement = default_base
            
            for i in range(1, 7):
                base_col = f"{i}{prefix}_base"
                ps_col = f"{i}{prefix}_PS"
                if str(row[base_col]) in target_value:
                    row[base_col] = replacement
                    row[ps_col] = default_ps
            return row
        
        def correct_ip_values(row):
            return correct_values(row, 'Ip', {'?1', '?2', 'Noise'}, 's', '5')
        
        df.loc[mask] = df.loc[mask].apply(lambda row: correct_values(row, 'Ub', {'?1', '?2', 'Noise'}, 's', '100'), axis=1)
        df.loc[mask] = df.loc[mask].apply(lambda row: correct_values(row, 'Uc', {'?1', '?2', 'Noise'}, 's', '100'), axis=1)
        df.loc[mask] = df.loc[mask].apply(correct_ip_values, axis=1)
        
        df.loc[mask, 'norm'] = 'YES'
        
        df.to_csv(output_path, index=False)
