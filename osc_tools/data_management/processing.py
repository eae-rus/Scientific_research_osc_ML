import os
import shutil
import hashlib
import datetime
import json
import re
import csv
from tqdm import tqdm

# Этот файл содержит функции для общей обработки файлов осциллограмм,
# такие как удаление конфиденциальной информации и группировка.

def deleting_confidential_information_in_all_files(source_dir: str) -> None:
    """
    Выявляет и обрабатывает конфиденциальную информацию в файлах ".cfg" в указанном исходном каталоге,
    а также переименовывает файлы с расширением .cfg в .data.

    Функция ищет файлы ".cfg" в заданном каталоге, пытается определить их кодировку (utf-8, windows-1251 или ОЕМ 866)
    и собирает пути к файлам с потенциальной конфиденциальной информацией.
    Она сохраняет список защищенных файлов и любых ошибок обработки в файл 'protected_files.txt' в исходном каталоге.

    Args:
        source_dir (str): каталог, содержащий файлы .cfg для проверки конфиденциальной информации.

    Returns:
        None
    """
    protected_files = []
    print("Подсчитываем общее количество файлов в исходном каталоге...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
    print(f"Общее количество файлов: {total_files}, начинаем обработку...")
    with tqdm(total=total_files, desc="Удаление конфиденциальной информации") as pbar:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)
                if file.endswith(".cfg"):
                    file_path = os.path.join(root, file)
                    # TODO: для определения кодировки и пересохранения файла в utf-8 нужно создать отдельную функцию
                    try:
                        _deleting_confidential_information_in_one_file(file_path, root, 'utf-8')
                    except Exception as e:
                        try:
                            _deleting_confidential_information_in_one_file(file_path, root, 'windows-1251')
                        except Exception as e:
                            try:
                                _deleting_confidential_information_in_one_file(file_path, root, 'ОЕМ 866') # ОЕМ - русский
                            except Exception as e:
                                protected_files.append(file_path)
                                protected_files.append(f"Произошла ошибка при обработке файла cfg: {e}")
    with open(os.path.join(source_dir, 'protected_files.txt'), 'w') as file:
        file.write('\n'.join(protected_files))

def _deleting_confidential_information_in_one_file(file_path: str, root: str, encoding_name: str) -> None:
    """
    Функция считывает содержимое файла. cfg выполняет коррекцию кодировки,
    удаляет локальную информацию, обновляет определенные строки, вычисляет хеш данных и
    переименовывает файлы .cfg и соответствующие файлы .dat на основе вычисленного хеша.

    Args:
        file_path (str): путь к файлу .cfg.
        root (str): путь к корневому каталогу.
        encoding_name (str): формат кодировки, используемый для чтения файла.

    Returns:
        None
    """
    with open(file_path, 'r', encoding=encoding_name) as file:
        lines = file.readlines()
        # Удаление информации о локальной информации в файле cfg
        parts = lines[0].split(',')
        if len(parts) >= 2:
            lines[0] = ",," + parts[-1].strip() + "\n"
        # чтение количества сигналов
        signals, analog_signals, digital_signals = lines[1].split(',')
        signals = int(signals)
        new_date = '01/01/0001, 01:01:01.000000\n'
        # FIXME: Неправильно определяется точка при наличии нескольких частот дискретизации.
        if len(lines) > signals + 5:
            lines[signals + 5] = new_date
        if len(lines) > signals + 6:
            lines[signals + 6] = new_date

    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line)
    dat_file_path = file_path[:-4] + ".dat"
    with open(dat_file_path, 'rb') as file:
        file_hash = hashlib.md5(file.read()).hexdigest()
    os.rename(file_path, os.path.join(root, file_hash + '.cfg'))
    os.rename(dat_file_path, os.path.join(root, file_hash + '.dat'))


def date_of_change_replacement(source_dir: str) -> None:
    """
    Функция проходит по всем файлам в этом каталоге и устанавливает время модификации каждого файла на текущую дату и время.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    current_date = datetime.datetime.now()

    print("Подсчитываем общее количество файлов в исходном каталоге...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
    print(f"Общее количество файлов: {total_files}, начинаем обработку...")
    with tqdm(total=total_files, desc="Удаление конфиденциальной информации") as pbar:
        for root, dirs, files in os.walk(source_dir):
            for filename in files:
                pbar.update(1)
                file_path = os.path.join(root, filename)
                file_stat = os.stat(file_path)
                os.utime(file_path, times=(file_stat.st_atime, current_date.timestamp()))


def grouping_by_sampling_rate_and_network(source_dir: str, threshold: float = 0.1, isPrintMessege: bool = False) -> None:
    """
    Функция группирует файлы по частоте дискретизации и частоте сети.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.
        threshold (float): порог для рассмотрения отклонения частоты от целого числа как ошибки измерения.
        isPrintMessege (bool): флаг, указывающий, нужно ли выводить сообщение, если частоты не найдены.

    Returns:
        None
    """
    print("Подсчитываем общее количество файлов в исходном каталоге...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
    print(f"Общее количество файлов: {total_files}, начинаем обработку...")
    with tqdm(total=total_files, desc="Группировка по частоте дискретизации и сети") as pbar:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)
                if file.lower().endswith(".cfg"):
                    file_path = os.path.join(root, file)
                    dat_file = file[:-4] + ".dat"
                    dat_file_path = os.path.join(root, dat_file)
                    is_exist = os.path.exists(dat_file_path)
                    if is_exist:
                        f_network, f_rate = extract_frequencies(file_path=file_path, threshold=threshold, isPrintMessege=isPrintMessege)

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

def extract_frequencies(file_path: str, threshold: float = 0.1, isPrintMessege: bool = False) -> tuple:
    """
    Извлекает частоту сети (f_network) и частоту дискретизации (f_rate) из указанного файла ".cfg".

    Args:
        source_dir (str): путь к файлу ".cfg".
        threshold (float): порог для рассмотрения отклонения частоты от целого числа как ошибки измерения.
        isPrintMessege (bool): флаг, указывающий, нужно ли выводить сообщение, если частоты не найдены.

    Returns:
        tuple: кортеж, содержащий извлеченную частоту сети и частоту дискретизации.
    """
    f_network, f_rate = 0, 0

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # FIXME: Нет защиты от защищенных и/или ошибочных файлов
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
                f_network, f_rate = -1, -1 # TODO: В будущих версиях может потребоваться скорректировать неверную частоту
        except Exception as e:
            f_network, f_rate = -1, -1 # TODO: В будущих версиях может потребоваться скорректировать неверную частоту
            if isPrintMessege: print(e)

    return f_network, f_rate

def combining_json_hash_table(source_dir: str, encoding: str = 'utf-8') -> None:
    """
    !!ВНИМАНИЕ!!
    Стоит использовать "hash_table" только для объединения файлов. Функция объединяет json-файлы с уникальными именами осциллограмм

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    combine_hash_table = {}
    print("Подсчитываем общее количество файлов в исходном каталоге...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
    print(f"Общее количество файлов: {total_files}, начинаем обработку...")
    with tqdm(total=total_files, desc="Объединение хеш-таблицы json") as pbar:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)
                if file.lower().endswith(".json"):
                    try:
                        path = os.path.join(root, file)
                        with open(path, 'r', encoding=encoding) as f:
                            hash_table = json.load(f)
                            for key, value in hash_table.items():
                                if key not in combine_hash_table:
                                    combine_hash_table[key] = value
                    except:
                        print("Не удалось прочитать hash_table из файла JSON")

    try:
        combine_hash_table_file_path = os.path.join(source_dir, 'combine_hash_table.json')
        with open(combine_hash_table_file_path, 'w') as f:
            json.dump(combine_hash_table, f)
    except:
        print("Не удалось сохранить new_hash_table в файл JSON")

def detect_date(file_path: str, root: str, encoding_name: str, dict_all_dates = {}) -> None:
    """
    Функция определяет дату события в файле cfg.

    Args:
        file_path (str): путь к файлу .cfg.
        root (str): путь к корневому каталогу.
        encoding_name (str): формат кодировки, используемый для чтения файла.

    Returns:
        None
    """
    with open(file_path, 'r', encoding=encoding_name) as file:
        lines = file.readlines()
        count_all_signals_str, count_analog_signals_str, count_digital_signals_str = lines[1].split(',')
        count_all_signals = int(count_all_signals_str)

        # Мы меняем только год в дате события
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
