import os
import shutil
import hashlib
import datetime

def copy_cfg_and_dat_files_in_one_dir(source_dir, dest_dir): #FIXME: исправить наименование
    """
    Копирует файлы .cfg и соответствующие файлы .dat из дирректории "source_dir" в "dest_dir".

    Args:
        source_dir (str): исходный каталог, содержащий файлы .cfg и .dat.
        dest_dir (str): каталог назначения для копирования файлов.

    Returns:
        None
    """
    for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
        for file in files:  # Имя каждого файла
            if file.lower().endswith(".cfg"):  # Если файл имеет расширение .cfg
                file = file[:-4] + ".cfg" # изменяем шрифт типа файла на строчный.
                file_path = os.path.join(root, file)  # Получаем полный путь к cfg файлу
                dat_file = file[:-4] + ".dat"  # Формируем имя dat файла на основе имени cfg файла
                dat_file_path = os.path.join(root, dat_file)  # Получаем полный путь к dat файлу
                is_exist = os.path.exists(dat_file_path) 
                if is_exist:
                    dest_path = os.path.join(dest_dir, file)  # Формируем путь для копирования cfg файла
                    if not os.path.exists(dest_path):
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Создаем все несуществующие директории для целевого файла
                        shutil.copy2(file_path, dest_path)  # Копируем cfg файл в целевую директорию

                    dat_dest_path = os.path.join(dest_dir, dat_file)  # Формируем путь для копирования dat файла
                    if not os.path.exists(dat_dest_path):
                        os.makedirs(os.path.dirname(dat_dest_path), exist_ok=True)  # Создаем все несуществующие директории для целевого dat файла
                        shutil.copy2(dat_file_path, dat_dest_path)  # Копируем dat файл в целевую директорию

def deleting_confidential_information_in_cfg_files(source_dir): #FIXME: исправить наименование
    """
    Идентифицирует и обрабатывает конфиденциальную информацию в файлах. cfg в указанном исходном каталоге.

    Функция ищет файлы .cfg в заданном каталоге, пытается определить их кодировку (utf-8, windows-1251 или ОЕМ 866) 
    и собирает пути к файлам с потенциальной конфиденциальной информацией. 
    Он сохраняет список защищенных файлов и любые ошибки обработки в 'protected_files.txt-файл в исходном каталоге.

    Args:
        source_dir (str): каталог, содержащий файлы .cfg для проверки конфиденциальной информации.

    Returns:
        None
    """
    protected_files = []  # Создаем список для хранения путей к защищенным файлам
    for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
        for file in files:  # Имя каждого файла
            if file.endswith(".cfg"):  # Если файл имеет расширение .cfg
                file_path = os.path.join(root, file)  # Получаем полный путь к cfg файлу
                try:
                    diff_encoding(file_path, root, 'utf-8')
                except Exception as e:
                    try:
                        diff_encoding(file_path, root, 'windows-1251')  
                    except Exception as e:
                        try:
                            diff_encoding(file_path, root, 'ОЕМ 866') # ОЕМ - русский язык
                        except Exception as e:
                            protected_files.append(file_path)  # Добавляем защищенный файл в список
                            protected_files.append(f"Произошла ошибка при обработке cfg файла: {e}")
    with open(os.path.join(source_dir, 'protected_files.txt'), 'w') as file:
        file.write('\n'.join(protected_files))  # Сохраняем список защищенных файлов в txt файл в корне папки

def diff_encoding(file_path, root, encoding_name): #FIXME: исправить наименование
    """
    Функция считывает содержимое файла. cfg, выполняет коррекцию кодировки, 
    удаляет локальную информацию, обновляет определенные строки, вычисляет хэш данных 
    и переименовывает файлы .cfg и соответствующие файлы .dat на основе вычисленного хэша.
    
    Args:
        file_path (str): путь к файлу .cfg.
        root (str): путь к корневому каталогу.
        encoding_name (str): формат кодировки, используемый для чтения файла.

    Returns:
        None
    """
    with open(file_path, 'r', encoding=encoding_name) as file:
            lines = file.readlines()
            # Удаляем информацию о локальной информации в cfg файле
            parts = lines[0].split(',')
            if len(parts) >= 2:
                lines[0] = ",," + parts[-1].strip() + "\n"
            # считываем количество сигналов
            signals, analog_signals, digital_signals = lines[1].split(',')
            signals = int(signals)
            new_date = '01/01/0001, 01:01:01.000000\n'
            lines[signals + 5] = new_date
            lines[signals + 6] = new_date

    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line)
    dat_file_path = file_path[:-4] + ".dat"  # Получаем полный путь к dat файлу 
    with open(dat_file_path, 'rb') as file: # решил делать по dat файлу, так как он точно отличается даже после всех корректировок
        file_hash = hashlib.md5(file.read()).hexdigest()  # Вычисляем хэш-сумму cfg файла
    os.rename(file_path, os.path.join(root, file_hash + '.cfg'))
    os.rename(dat_file_path, os.path.join(root, file_hash + '.dat'))


def date_replacement(source_dir):
    """
    Функция перебирает все файлы в данном каталоге и устанавливает время изменения каждого файла на текущую дату и время.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    # Получаем текущую дату и время
    current_date = datetime.datetime.now()

    # Проходим по всем файлам в папке
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        file_stat = os.stat(file_path)
        os.utime(file_path, times=(file_stat.st_atime, current_date.timestamp()))


# Пример использования функции
# Путь к исходной директории
source_directory = 'C:/Users/User/Desktop/Буфер (Алексей)/Банк осциллограмм/_до обработки/Удалить'
source_directory = '//192.168.87.199/документы/ОТГРУЖЕННЫЕ ТЕРМИНАЛЫ И ШКАФЫ/Терминалы/БАВР/00203'
source_directory = 'C://Users/User/Desktop/Буфер (Алексей)/Банк осциллограмм/Локальное (Алексея)'
# Путь к целевой директории
destination_directory = 'C:/Users/User/Desktop/Буфер (Алексей)/Банк осциллограмм/_до обработки/_ALL_OSC'

# FIXME: Оформить описание
# 1) сперва рекомендуется переносить в одну папку
# 2) потом проходиться алгоритмом обезличивания
# 3) а затем удалять дату.

# copy_cfg_and_dat_files_in_one_dir(source_directory, destination_directory)
# deleting_confidential_information_in_cfg_files(source_directory)
date_replacement(source_directory)