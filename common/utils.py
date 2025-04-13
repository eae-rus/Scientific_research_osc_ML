import os
import sys
import pandas as pd
from pathlib import Path
import glob
from typing import List

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
        
    def merge_normalization_csvs(self, file_paths: List[str], output_path: str = "new_norm_coef.csv"):
        """
        Объединяет CSV файлы с коэффициентами нормализации, добавляя только строки с новыми 'name'.

        Функция читает первый файл из списка, сохраняет его данные и уникальные значения
        из столбца 'name'. Затем итерирует по остальным файлам, читает их,
        и добавляет только те строки, чьи значения 'name' еще не встречались.
        Столбец 'Column1', если он присутствует, игнорируется и удаляется.

        Args:
            file_paths (List[str]): Список путей к CSV файлам для объединения.
                                    Порядок файлов в списке важен: первый файл
                                    считается основным, из последующих добавляются
                                    только новые записи.

        Returns:
            Сохраняет файл по пути output_path
        """
        #TODO: На подумать. Вероятно её стоит пернести в класс производящий нормализацию, так как напрямую работает с тоблицами создаваемыми там 
        if not file_paths:
            print("Ошибка: Список путей к файлам пуст.")
            return None

        master_df = None
        known_names = set()
        expected_columns = None

        # Обрабатываем первый файл
        first_file_path = file_paths[0]
        try:
            print(f"Обработка основного файла: {first_file_path}...")
            df = pd.read_csv(first_file_path)

            # Проверяем наличие ключевого столбца 'name'
            if 'name' not in df.columns:
                df = pd.read_csv(file_path, delimiter=';')
                if 'name' not in df.columns:
                    print(f"Ошибка: Ключевой столбец 'name' не найден в файле {first_file_path}. Невозможно продолжить.")
                    return None

            # Удаляем 'Column1', если он существует
            if 'Column1' in df.columns:
                print(f"  Найден и будет удален столбец 'Column1' в {first_file_path}")
                df = df.drop('Column1', axis=1)
            
            # Проверяем наличие дубликатов 'name' в первом файле (опционально, но полезно)
            if df['name'].duplicated().any():
                print(f"Предупреждение: Найдены дублирующиеся значения 'name' в основном файле {first_file_path}. Будут сохранены все строки.")
                # Можно добавить логику обработки дубликатов, если нужно (например, брать первую встреченную)
                # df = df.drop_duplicates(subset=['name'], keep='first')

            master_df = df
            known_names = set(master_df['name'].astype(str).unique()) # Приводим к строке на всякий случай и берем уникальные
            expected_columns = list(master_df.columns)
            print(f"  Загружено {len(master_df)} строк. Уникальных имен: {len(known_names)}.")
            print(f"  Ожидаемые столбцы: {expected_columns}")

        except FileNotFoundError:
            print(f"Ошибка: Файл не найден: {first_file_path}. Невозможно инициализировать процесс.")
            return None
        except Exception as e:
            print(f"Ошибка при чтении или обработке основного файла {first_file_path}: {e}")
            return None

        # Обрабатываем остальные файлы
        for file_path in file_paths[1:]:
            try:
                print(f"\nОбработка дополнительного файла: {file_path}...")
                current_df = pd.read_csv(file_path)

                # Проверяем наличие 'name'
                if 'name' not in current_df.columns:
                    current_df = pd.read_csv(file_path, delimiter=';')
                    if 'name' not in current_df.columns:
                        print(f"  Предупреждение: столбец 'name' отсутствует в файле {file_path}. Файл пропущен.")
                        continue

                # Запоминаем оригинальные колонки для проверки перед удалением 'Column1'
                original_cols = list(current_df.columns)
                cols_to_check = original_cols

                # Удаляем 'Column1', если он есть
                column1_present = 'Column1' in current_df.columns
                if column1_present:
                    print(f"  Найден и будет удален столбец 'Column1' в {file_path}")
                    current_df = current_df.drop('Column1', axis=1)
                    cols_to_check = list(current_df.columns)

                # Сравниваем столбцы текущего файла со столбцами мастер-файла НА ДАННЫЙ МОМЕНТ
                master_cols = set(master_df.columns)
                current_cols = set(current_df.columns)

                # Столбцы, которые есть в текущем, но нет в мастере
                new_cols_in_current = list(current_cols - master_cols)
                # Столбцы, которые есть в мастере, но нет в текущем
                missing_cols_in_current = list(master_cols - current_cols)

                # Добавляем НОВЫЕ столбцы (из текущего файла) в master_df, заполняя NA
                if new_cols_in_current:
                    print(f"  Обнаружены новые столбцы в {file_path}: {new_cols_in_current}. Добавляю их в основную таблицу.")
                    for col in new_cols_in_current:
                        # Добавляем столбец в master_df, заполняя его pd.NA или np.nan
                        # pd.NA - новый рекомендуемый способ для обозначения пропусков,
                        # который лучше работает с разными типами данных (int, bool)
                        master_df[col] = pd.NA
                    # Обновляем master_cols для следующего шага
                    master_cols.update(new_cols_in_current)


                # Добавляем НЕДОСТАЮЩИЕ столбцы (из мастера) в current_df, заполняя NA
                if missing_cols_in_current:
                    print(f"  В файле {file_path} отсутствуют столбцы: {missing_cols_in_current}. Добавляю их.")
                    for col in missing_cols_in_current:
                        current_df[col] = pd.NA

                # После добавления столбцов, УПОРЯДОЧИВАЕМ столбцы в current_df
                # так же, как в master_df, чтобы concat работал корректно
                # Убедимся, что master_df содержит все столбцы перед упорядочиванием
                all_columns = list(master_df.columns) # Теперь это полный набор столбцов
                current_df = current_df[all_columns]

                # Отбираем строки с НОВЫМИ именами
                # Приводим 'name' к строке перед сравнением для надежности
                new_rows = current_df[~current_df['name'].astype(str).isin(known_names)]

                if not new_rows.empty:
                    print(f"  Найдено {len(new_rows)} строк с новыми именами.")
                    # Добавляем новые строки в основной DataFrame
                    master_df = pd.concat([master_df, new_rows], ignore_index=True)
                    # Обновляем множество известных имен
                    new_names_found = set(new_rows['name'].astype(str).unique())
                    known_names.update(new_names_found)
                    print(f"  Добавлены новые имена")
                    print(f"  Общее количество строк теперь: {len(master_df)}. Уникальных имен: {len(known_names)}")
                else:
                    print(f"  Новых имен в файле {file_path} не найдено.")

            except FileNotFoundError:
                print(f"  Предупреждение: Файл не найден {file_path}. Файл пропущен.")
                continue
            except Exception as e:
                print(f"  Предупреждение: Ошибка при чтении или обработке файла {file_path}: {e}. Файл пропущен.")
                continue

        print("\nОбъединение завершено.")
        master_df.to_csv(output_path, index=False)
        print(f"\nСохранение завершено, файл имеет имя: {output_path}.")
