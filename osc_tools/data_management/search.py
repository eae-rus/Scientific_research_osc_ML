"""Совместимый интерфейс поиска и организации осциллограмм."""
import os
import shutil
import json
from .oscillogram_search import SafeOscillogramSearch


class SearchOscillograms(SafeOscillogramSearch):
    """Поиск кандидатов и прежние операции организации по терминалам."""

    def find_terminal_hashes_from_json(
        self,
        input_json_path: str,
        terminal_numbers_to_find: list[int],
        output_json_path: str
    ) -> None:
        """
        Ищет хеши осциллограмм для указанных номеров терминалов в JSON-файле
        и сохраняет результат в новый JSON.

        Args:
            input_json_path (str): Путь к исходному JSON-файлу с данными осциллограмм.
                                   Формат: {"hash": ["filename.cfg", "path/to/filename.cfg"], ...}
            terminal_numbers_to_find (list[int]): Список целочисленных номеров терминалов для поиска.
            output_json_path (str): Путь для сохранения выходного JSON-файла.
                                    Формат: {"terminal_num_str": ["hash1", "hash2"], ...}
        """
        SHIPPED_MARKER = "ОТГРУЖЕННЫЕ ТЕРМИНАЛЫ И ШКАФЫ" # Имя папки хранения осциллограмм
        from tqdm import tqdm
        FILENAME_PREFIX = "t" # Префикс для поиска по имени файла

        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                oscillogram_data = json.load(f)
        except FileNotFoundError:
            print(f"Ошибка: Входной JSON-файл не найден: {input_json_path}")
            return
        except json.JSONDecodeError:
            print(f"Ошибка: Не удалось декодировать JSON из файла: {input_json_path}")
            return

        # Инициализируем словарь для результатов.
        # Ключи - строковые представления номеров терминалов, значения - списки хешей.
        # Используем уникальные отсортированные номера терминалов для предсказуемого порядка ключей.
        unique_sorted_term_numbers = sorted(list(set(terminal_numbers_to_find)))
        found_hashes_by_terminal = {str(tn): [] for tn in unique_sorted_term_numbers}

        # Оборачиваем внешний цикл в tqdm для отображения прогресса
        for hash_code, (filename, filepath) in tqdm(oscillogram_data.items(), desc="Обработка осциллограмм"):
            filename_lower = filename.lower() # Для регистронезависимого сравнения имени файла

            for term_num in unique_sorted_term_numbers:
                # Паттерн 1: Поиск по имени файла
                # Имя файла осциллограммы начинается с "t" + номер терминала (5 цифр, дополненных нулями).
                filename_search_pattern = f"{FILENAME_PREFIX}{term_num:05d}".lower()

                found_for_current_hash_and_term = False

                if filename_lower.startswith(filename_search_pattern):
                    if hash_code not in found_hashes_by_terminal[str(term_num)]:
                        found_hashes_by_terminal[str(term_num)].append(hash_code)
                    found_for_current_hash_and_term = True
                    # Для данной пары (hash_code, term_num) совпадение найдено по имени.
                    # По ТЗ, если нашли по имени, по пути для этой же пары искать не нужно.

                # Паттерн 2: Поиск по пути к файлу (если не найдено по имени для текущего term_num)
                if not found_for_current_hash_and_term:
                    # Проверяем наличие ключевой фразы в пути (регистрочувствительно, как в ТЗ)
                    if SHIPPED_MARKER in filepath:
                        # Готовим паттерны номера для поиска в пути
                        path_number_patterns = [f"{term_num:05d}"]
                        if term_num < 10:
                            path_number_patterns.append(f"{term_num:04d}")

                        # Нормализуем разделители и разбиваем путь на компоненты
                        path_components = filepath.replace("\\", "/").split("/")

                        for num_pattern_for_path in path_number_patterns:
                            if num_pattern_for_path in path_components:
                                if hash_code not in found_hashes_by_terminal[str(term_num)]:
                                    found_hashes_by_terminal[str(term_num)].append(hash_code)
                                # Нашли по одному из паттернов в пути, дальше для этой пары (hash_code, term_num)
                                # и этого типа паттерна (путь) искать не нужно.
                                break # Выход из цикла по path_number_patterns

        # Сохраняем результаты в выходной JSON-файл
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(found_hashes_by_terminal, f, indent=4, ensure_ascii=False)
            print(f"Результаты успешно сохранены в: {output_json_path}")
        except IOError:
            print(f"Ошибка: Не удалось записать результаты в файл: {output_json_path}")
    

    def organize_oscillograms_by_terminal(self, source_dir: str, dest_dir: str, terminal_list: list, terminal_oscillogram_names: dict, is_hashes: bool = True) -> None:
        """
        Копирует осциллограммы, структурируя их по номерам терминалов.
          
        Args:
            source_dir (str): исходный каталог.
            dest_dir (str): каталог для сохранения скопированных файлов.
            terminal_list (list): список имен терминалов.
            terminal_oscillogram_names (dict): словарь для хранения терминалов и хеш-имен осциллограмм.
            is_hashes (bool): использовать ли хеш-имена.
        Returns:
            None
        """
        from tqdm import tqdm
        osc_terminal_dict = {}  # словарь для хранения имен осциллограмм и их принадлежности к терминалам
        # Функция, которая формирует обратный список к словарю osc_name_dict (из "терминал -> имя осциллограммы" в "имя осциллограммы -> терминал")
        for terminal_name in terminal_list:
            for osc_name in terminal_oscillogram_names[terminal_name]:
                osc_terminal_dict[osc_name] = terminal_name
        
        print("Подсчитываем общее количество файлов в исходном каталоге...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Общее количество файлов: {total_files}, начинаем обработку...")
        with tqdm(total=total_files, desc="Организация осциллограмм") as pbar:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    pbar.update(1)
                    if file.endswith(self.CFG_EXTENSION):
                        file_name = file[:-4]
                        if not is_hashes:
                            dat_file = file[:-4] + self.DAT_EXTENSION
                            dat_path = os.path.join(root, dat_file)
                            try:
                                with open(dat_path, 'rb') as f:
                                    file_name = hashlib.md5(f.read()).hexdigest()
                            except FileNotFoundError:
                                continue  # пропустить, если .dat файл не найден

                        if file_name in osc_terminal_dict:
                            cfg_file_name = file
                            dat_file_name = cfg_file_name[:-4] + self.DAT_EXTENSION
                            cfg_file_path = os.path.join(root, file)
                            dat_file_path = cfg_file_path[:-4] + self.DAT_EXTENSION
                            if os.path.exists(cfg_file_path) and os.path.exists(dat_file_path):
                                cfg_dest_path = os.path.join(dest_dir, osc_terminal_dict[file_name], cfg_file_name)
                                dat_dest_path = os.path.join(dest_dir, osc_terminal_dict[file_name], dat_file_name)
                                if not os.path.exists(cfg_dest_path):
                                    os.makedirs(os.path.dirname(cfg_dest_path), exist_ok=True)
                                    shutil.copy2(cfg_file_path, cfg_dest_path)

                                if not os.path.exists(dat_dest_path):
                                    os.makedirs(os.path.dirname(dat_dest_path), exist_ok=True)
                                    shutil.copy2(dat_file_path, dat_dest_path)


if __name__ == '__main__':
    # --- Вызов класса ---
    f = SearchOscillograms()

    # ---!!! Вызов новой функции !!!---
    # ---find_terminal_hashes_from_json---    
    # Укажите правильные пути
    comtrade_directory = "Путь к папке с файлами осциллограмм"
    input_json_path = "Путь к файлу с исходным JSON файлом содержащем информацию о всех осциллограммах"
    output_json_path = "Путь к итоговому файлу"
    # список искомых терминалов
    terminals_to_search = [1, 2, 3] # список искомых терминалов
    # f.find_terminal_hashes_from_json(input_json_path, terminals_to_search, output_json_path)
