import os
import csv
import re
from collections import Counter
from tqdm import tqdm

# Этот файл будет содержать функции для работы с именами сигналов в файлах осциллограмм,
# включая поиск, переименование и разрешение конфликтов.

def find_all_name_analog_signals(source_dir: str) -> None:
    """
    Функция ищет все имена аналоговых сигналов в файле comtrade и сортирует их по частоте использования.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    analog_signals_name = {}
    print("Подсчитываем общее количество файлов в исходном каталоге...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
    print(f"Общее количество файлов: {total_files}, начинаем обработку...")
    with tqdm(total=total_files, desc="Поиск всех имен аналоговых сигналов") as pbar:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)
                try:
                    if file.lower().endswith(".cfg"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as file_content:
                            # FIXME: Нет защиты от защищенных и/или ошибочных файлов
                            lines = file_content.readlines()
                            if len(lines) >= 2:
                                count_all_signals_str, count_analog_signals_str, count_digital_signals_str = lines[1].split(',')
                                count_analog_signals = int(count_analog_signals_str[:-1])
                                for i in range(count_analog_signals):
                                    analog_signal = lines[2 + i].split(',') # получение аналогового сигнала
                                    # TODO: добавить единую функцию для генерации комбинированного имени сигнала
                                    name, phase = analog_signal[1], analog_signal[2] # получаем имя, фазу и единицу измерения
                                    name, phase = name.replace(' ', ''), phase.replace(' ', '')
                                    signal_name = name + ' | phase:' + phase # создание комбинированного имени сигнала
                                    if signal_name not in analog_signals_name:
                                        analog_signals_name[signal_name] = 1
                                    else:
                                        analog_signals_name[signal_name] += 1
                except Exception as e:
                    print(e)
                    print("Произошла ошибка при обработке файла: ", file_path)

    sorted_analog_signals_name = {k: v for k, v in sorted(analog_signals_name.items(), key=lambda item: item[1], reverse=True)}
    csv_file = os.path.join(source_dir, 'sorted_analog_signals_name.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', "universal_code", 'Value'])  # Запись заголовка
        for key, value in sorted_analog_signals_name.items():
            writer.writerow([key, "-", value])

def find_all_name_digital_signals(source_dir: str) -> None:
    """
    Функция ищет все имена дискретных сигналов в файле comtrade и сортирует их по частоте использования.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    digital_signals_name = {}
    print("Подсчитываем общее количество файлов в исходном каталоге...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
    print(f"Общее количество файлов: {total_files}, начинаем обработку...")
    with tqdm(total=total_files, desc="Поиск всех имен цифровых сигналов") as pbar:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)
                try:
                    if file.lower().endswith(".cfg"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as file_content:
                            # FIXME: Нет защиты от защищенных и/или ошибочных файлов
                            lines = file_content.readlines()
                            if len(lines) >= 2:
                                count_all_signals_str, count_analog_signals_str, count_digital_signals_str = lines[1].split(',')
                                count_analog_signals = int(count_analog_signals_str[:-1])
                                count_digital_signals = int(count_digital_signals_str[:-2])
                                for i in range(count_digital_signals):
                                    digital_signal = lines[2 + count_analog_signals + i].split(',') # получение аналогового сигнала
                                    if len(digital_signal) == 1:# защита от неверного количества сигналов
                                        break
                                    signal_name = digital_signal[1] # получение имени
                                    if signal_name not in digital_signals_name:
                                        digital_signals_name[signal_name] = 1
                                    else:
                                        digital_signals_name[signal_name] += 1
                except Exception as e:
                    print(e)
                    print("Произошла ошибка при обработке файла: ", file_path)

    sorted_digital_signals_name = {k: v for k, v in sorted(digital_signals_name.items(), key=lambda item: item[1], reverse=True)}
    csv_file = os.path.join(source_dir, 'sorted_digital_signals_name.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', "universal_code", 'Value'])  # Запись заголовка
        for key, value in sorted_digital_signals_name.items():
            writer.writerow([key, "-", value])


def rename_analog_signals(source_dir: str, csv_dir: str, encoding: str = 'utf8', delimiter: str = ',') -> None:
    """
    Функция ищет все имена аналоговых сигналов, которые присутствуют в базе данных, и переименовывает их в стандартные коды.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.
        csv_dir (str): адрес CSV-файла.
        encoding (str, optional): кодировка CSV-файла. По умолчанию 'utf8'.
        delimiter (str, optional): разделитель в CSV-файле. По умолчанию ','.

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

    print("Подсчитываем общее количество файлов в исходном каталоге...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
    print(f"Общее количество файлов: {total_files}, начинаем обработку...")
    with tqdm(total=total_files, desc="Переименование аналоговых сигналов") as pbar:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)
                if file.lower().endswith(".cfg"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as file_content:
                        # FIXME: Нет защиты от защищенных и/или ошибочных файлов
                        lines = file_content.readlines()
                        if len(lines) >= 2:
                            count_all_signals_str, count_analog_signals_str, count_digital_signals_str = lines[1].split(',')
                            count_analog_signals = int(count_analog_signals_str[:-1])
                            for i in range(count_analog_signals):
                                analog_signal = lines[2 + i].split(',') # получение аналогового сигнала
                                # TODO: добавить единую функцию для генерации комбинированного имени сигнала
                                name, phase = analog_signal[1], analog_signal[2] # получаем имя, фазу и единицу измерения
                                name, phase = name.replace(' ', ''), phase.replace(' ', ''),
                                signal_name = name + ' | phase:' + phase # создание комбинированного имени сигнала
                                if signal_name in code_map:
                                    analog_signal[1] = code_map[signal_name]
                                    lines[2 + i] = ','.join(analog_signal)

                    with open(file_path, 'w', encoding='utf-8') as file_content:
                        file_content.writelines(lines)

def rename_digital_signals(source_dir: str, csv_dir: str, encoding: str = 'utf8', delimiter: str = ',') -> None:
    """
    Функция ищет все имена дискретных сигналов, которые присутствуют в базе данных, и переименовывает их в стандартные коды.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.
        csv_dir (str): адрес CSV-файла.
        encoding (str, optional): кодировка CSV-файла. По умолчанию 'utf8'.
        delimiter (str, optional): разделитель в CSV-файле. По умолчанию ','.

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

    print("Подсчитываем общее количество файлов в исходном каталоге...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
    print(f"Общее количество файлов: {total_files}, начинаем обработку...")
    with tqdm(total=total_files, desc="Переименование цифровых сигналов") as pbar:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)
                if file.lower().endswith(".cfg"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as file_content:
                        # FIXME: Нет защиты от защищенных и/или ошибочных файлов
                        lines = file_content.readlines()
                        if len(lines) >= 2:
                            count_all_signals_str, count_analog_signals_str, count_digital_signals_str = lines[1].split(',')
                            count_analog_signals = int(count_analog_signals_str[:-1])
                            count_digital_signals = int(count_digital_signals_str[:-2])
                            for i in range(count_digital_signals):
                                digital_signal = lines[2 + count_analog_signals + i].split(',') # получение аналогового сигнала
                                if len(digital_signal) == 1: # защита от неверного количества сигналов
                                    break
                                signal_name = digital_signal[1] # получение имени
                                if signal_name in code_map:
                                    digital_signal[1] = code_map[signal_name]
                                    lines[2 + count_analog_signals + i] = ','.join(digital_signal)

                    with open(file_path, 'w', encoding='utf-8') as file_content:
                        file_content.writelines(lines)

def rename_one_signals(source_dir: str, old_name: str, new_name: str) -> None:
    """
    Функция ищет все сигналы с одним именем и заменяет его на новое.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.
        old_name (str): старое имя сигнала.
        new_name (str): новое имя сигнала.

    Returns:
        None
    """
    print("Подсчитываем общее количество файлов в исходном каталоге...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
    print(f"Общее количество файлов: {total_files}, начинаем обработку...")
    with tqdm(total=total_files, desc="Переименование сигнала") as pbar:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)
                if file.lower().endswith(".cfg"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as file_content:
                        # FIXME: Нет защиты от защищенных и/или ошибочных файлов
                        lines = file_content.readlines()
                        if len(lines) >= 2:
                            for i in range(len(lines)):
                                if old_name in lines[i]:
                                    lines[i] = lines[i].replace(old_name, new_name)

                    with open(file_path, 'w', encoding='utf-8') as file_content:
                        file_content.writelines(lines)

def combining_databases_of_unique_codes(old_csv_file_path: str, new_csv_file_path: str, merged_csv_file_path: str,
                                        encoding_old_csv: str = 'utf-8', encoding_new_csv: str = 'utf-8',
                                        deelimed_old_csv_file: str = ',', deelimed_new_csv_file: str = ',', is_merge_files: bool = True) -> None:
    """
    Функция объединяет csv-файлы с уникальными кодами сигналов

    Args:
        old_csv_file_path (str): адрес csv-файла с уникальными кодами сигналов.
        new_csv_file_path (str): адрес csv-файла с уникальными кодами сигналов.
        encoding_old_csv (str): кодировка старого csv-файла.
        encoding_new_csv (str): кодировка нового csv-файла.
        deelimed_old_csv_file (str): разделитель старого csv-файла.
        deelimed_new_csv_file (str): разделитель нового csv-файла.

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


    # Если is_merge_files имеет значение True, объединить массив с суммированными значениями в поле value
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


def _extract_signal_names_from_cfg_lines(lines: list[str], file_path_for_error_msg: str, include_digital_signals: bool = True) -> tuple[list[str], str | None]:
    """
    Извлекает все имена сигналов из строк файла CFG.

    Args:
        lines (list[str]): содержимое файла CFG в виде списка строк.
        file_path_for_error_msg (str): путь к файлу, используемый для сообщений об ошибках.

    Returns:
        tuple[list[str], str | None]: кортеж, содержащий список имен сигналов
                                       и строку с сообщением об ошибке, если произошла ошибка, в противном случае None.
    """
    signal_names = []

    if len(lines) < 2:
        return [], f"Файл {file_path_for_error_msg} имеет менее 2 строк."

    # Разбор второй строки для подсчета сигналов
    try:
        parts = lines[1].split(',')
        if len(parts) < 3:
            return [], f"Вторая строка в {file_path_for_error_msg} имеет неверный формат: {lines[1].strip()}"

        # total_signals_str = parts[0].strip() # Не используется напрямую для извлечения имен

        analog_signals_str = parts[1].strip().upper()
        digital_signals_str = parts[2].strip().upper()

        # Извлекаем числа, удаляя нечисловые символы (например, 'A' или 'D')
        count_analog_signals = int(re.sub(r'\D', '', analog_signals_str))
        count_digital_signals = int(re.sub(r'\D', '', digital_signals_str))

    except ValueError:
        return [], f"Не удалось разобрать количество сигналов из второй строки в {file_path_for_error_msg}: {lines[1].strip()}"
    except Exception as e:
        return [], f"Неожиданная ошибка при разборе количества сигналов в {file_path_for_error_msg}: {e}"

    # Проверяем, достаточно ли у нас строк для аналоговых сигналов
    if len(lines) < 2 + count_analog_signals:
        return [], f"Файл {file_path_for_error_msg} не содержит достаточного количества строк для заявленных аналоговых сигналов ({count_analog_signals}). Содержит {len(lines)} строк."

    # Извлечение имен аналоговых сигналов
    for i in range(count_analog_signals):
        line_index = 2 + i
        signal_line_parts = lines[line_index].split(',')
        if len(signal_line_parts) > 1:
            signal_names.append(signal_line_parts[1].strip())
        else:
            return [], f"Неверно сформированная строка аналогового сигнала {line_index+1} в {file_path_for_error_msg}: {lines[line_index].strip()}"

    # Проверяем, достаточно ли у нас строк для цифровых сигналов
    if len(lines) < 2 + count_analog_signals + count_digital_signals:
        return [], f"Файл {file_path_for_error_msg} не содержит достаточного количества строк для заявленных цифровых сигналов ({count_digital_signals})."

    # Извлечение имен цифровых сигналов
    if include_digital_signals:
        # Проверяем, достаточно ли у нас строк для цифровых сигналов
        if len(lines) < 2 + count_analog_signals + count_digital_signals:
            # Если include_digital_signals=True, но строк не хватает, это ошибка.
            # Если include_digital_signals=False, эта проверка и извлечение не нужны.
            return [], f"Файл {file_path_for_error_msg} не содержит достаточного количества строк для заявленных цифровых сигналов ({count_digital_signals})."

        for i in range(count_digital_signals):
            line_index = 2 + count_analog_signals + i
            signal_line_parts = lines[line_index].split(',')
            if len(signal_line_parts) > 1:
                signal_names.append(signal_line_parts[1].strip())
            else:
                return [], f"Неверно сформированная строка цифрового сигнала {line_index+1} в {file_path_for_error_msg}: {lines[line_index].strip()}"

    return signal_names, None

def _parse_analog_signal_name_for_section(signal_name: str) -> dict | None:
        """
        Разбирает имя аналогового сигнала для извлечения его компонентов, уделяя особое внимание номеру секции.
        Пример: "U | BusBar-1 | phase: A" -> {'prefix': "U | BusBar-", 'section': "1", 'suffix': " | phase: A"}
        Пример: "I | Bus-12" -> {'prefix': "I | Bus-", 'section': "12", 'suffix': ""}
        Args:
            signal_name (str): полное имя аналогового сигнала.
        Returns:
            dict | None: словарь с 'prefix', 'section', 'suffix', если его можно разобрать, иначе None.
        """
        parts = signal_name.split('|')
        if len(parts) < 2:
            return None

        signal_type_part = parts[0].strip()
        location_section_part = parts[1].strip()

        # Регулярное выражение для поиска 'LocationType-Number' в конце location_section_part
        # Оно захватывает (все, что до)-(цифры)
        match = re.search(r"^(.*?)-(\d+)$", location_section_part)
        if not match:
            # Попробуйте другой шаблон, если сам тип местоположения содержит дефисы, например, "Some-Location-Type-1"
            # Этот шаблон ищет последний дефис, за которым следуют цифры.
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

def _rename_duplicate_analog_signals_in_lines(cfg_lines: list[str], file_path_for_log: str) -> tuple[list[str], bool, list[dict]]:
    """
    Определяет дубликаты имен аналоговых сигналов в предоставленных строках CFG и переименовывает их
    путем присвоения новых, неиспользуемых номеров секций.
    Работает только с аналоговыми сигналами.

    Args:
        cfg_lines (list[str]): содержимое файла CFG в виде списка строк.
        file_path_for_log (str): путь к файлу для целей ведения журнала.

    Returns:
        tuple:
            - list[str]: измененный список строк CFG.
            - bool: True, если были внесены какие-либо изменения, иначе False.
            - list[dict]: журнал переименованных сигналов [{'file_path', 'line_index', 'old_name', 'new_name'}].
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
        parsed_components = _parse_analog_signal_name_for_section(original_name)

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

def find_duplicate_signal_names_in_cfg(
                                       source_dir: str,
                                       output_csv_duplicates_path: str,
                                       include_digital_signals: bool = True,
                                       auto_rename_analog_duplicates: bool = False,
                                       output_csv_rename_log_path: str = "rename_log.csv"
                                       ) -> None:
    """
    Сканирует файлы .cfg на наличие дублирующихся имен сигналов. При необходимости переименовывает дубликаты *аналоговых* сигналов.

    Args:
        source_dir (str): каталог, содержащий файлы .cfg.
        output_csv_duplicates_path (str): путь для сохранения CSV-файла со списком файлов с дубликатами.
        include_digital_signals (bool): включать ли цифровые сигналы в поиск дубликатов.
        auto_rename_analog_duplicates (bool): если True, автоматически переименовывает дубликаты аналоговых сигналов.
        output_csv_rename_log_path (str): путь для сохранения CSV-журнала действий по переименованию.
    """
    files_with_duplicates_overall = [] # Для CSV со списком файлов с дубликатами
    error_log_scan = []
    all_renaming_actions_log = [] # Для CSV с логом переименований

    total_cfg_files = 0
    for _, _, files_in_dir in os.walk(source_dir):
        for file_name in files_in_dir:
            if file_name.lower().endswith(".cfg"):
                total_cfg_files += 1

    print(f"Всего файлов .cfg для сканирования: {total_cfg_files}")
    encodings_to_try = ['utf-8', 'windows-1251', 'cp866']

    with tqdm(total=total_cfg_files, desc="Сканирование CFG на наличие дубликатов") as pbar:
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
                    error_log_scan.append(f"Не удалось прочитать файл {file_path} ни с одной из предпринятых кодировок.")
                    continue

                # 1. Анализ на дубликаты (как и раньше)
                # Используем include_digital_signals для определения, какие сигналы считать
                signal_names_for_duplicate_check, error_msg_extract = _extract_signal_names_from_cfg_lines(
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
                        _rename_duplicate_analog_signals_in_lines(file_content_lines, file_path)

                    if made_changes_flag:
                        try:
                            with open(file_path, 'w', encoding=used_encoding) as f_write:
                                f_write.writelines(modified_lines)
                            all_renaming_actions_log.extend(current_file_rename_log)
                        except IOError as e:
                            error_log_scan.append(f"Ошибка записи изменений в {file_path}: {e}")
                    elif current_file_rename_log: # Если были ошибки внутри переименования, но флаг false
                         error_log_scan.append(f"Функция переименования сообщила о проблемах для {file_path}, но никаких изменений не было внесено. Журнал: {current_file_rename_log}")


    # Сохранение CSV со списком файлов, где найдены дубликаты
    try:
        # Убедимся, что директория для output_csv_duplicates_path существует
        os.makedirs(os.path.dirname(output_csv_duplicates_path), exist_ok=True)
        with open(output_csv_duplicates_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file_path', 'file_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(files_with_duplicates_overall)
        print(f"Список файлов с дублирующимися именами сигналов успешно сохранен в: {output_csv_duplicates_path}")
    except IOError as e:
        error_log_scan.append(f"Ошибка записи CSV-файла с дубликатами в {output_csv_duplicates_path}: {e}")

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
            print(f"Журнал переименований успешно сохранен в: {rename_log_full_path}")
        except IOError as e:
            error_log_scan.append(f"Ошибка записи CSV-файла журнала переименований в {rename_log_full_path}: {e}")
    elif auto_rename_analog_duplicates:
         print("Автоматическое переименование было включено, но ни один сигнал не был переименован.")


    if error_log_scan:
        error_log_path = os.path.join(os.path.dirname(output_csv_duplicates_path), "scan_and_rename_errors.txt")
        try:
            with open(error_log_path, 'w', encoding='utf-8') as err_file:
                for err in error_log_scan:
                    err_file.write(f"{err}\n")
            print(f"Ошибки сканирования/переименования записаны в: {error_log_path}")
        except IOError as e:
             print(f"Не удалось записать журнал ошибок в {error_log_path}: {e}")
