import os
import shutil
import hashlib
import json
import datetime
import py7zr
import zipfile
import aspose.zip as az
from tqdm import tqdm
from enum import Enum


# создаю exe файл через 
# pip install auto-py-to-exe
# и затем опять же в командной строке
# python -m auto_py_to_exe

SOURCE_DIR = 'Пусть 1'
DEST_DIR = 'Путь 2'

class TYPE_OSC(Enum):
    # TODO: написать полноценные описание типов
    COMTRADE_CFG_DAT = "comtrade_cfg_dat"
    COMTRADE_CFF = "comtrade_cff"
    BRESLER = "bresler"
    BLACK_BOX = "black_box"
    RES_3 = "res_3"
    EKRA = "ekra"
    PARMA = "parma"
    NEVA = "neva"
    OSC = "osc"

CFG_EXTENSION, DAT_EXTENSION = '.cfg', '.dat'
CFF_EXTENSION = '.cff' # Форматы Comtrade (cff - новый) 
BRS_EXTENSION = '.brs' # Бреслер
EKRA_EXTENSION = '.dfr' # Внутренний формат ЭКРЫ
OSC_EXTENSION = '.osc' # Пока не выясинл что за тип
NEVA_EXTENSION = '.os' # Внутрении НЕВЫ. В реальности они os1, os2...
BLACK_BOX_EXTENSION = '.bb' # Чёрный ящик
RES_3_EXTENSION = '.sg2' # РЭС 3, пока не научился их открывать.
PARMA_O_EXTENSION, PARMA_0_EXTENSION, PARMA_1_EXTENSION, PARMA_2_EXTENSION, PARMA_3_EXTENSION = '.do', '.d0', '.d1', '.d2', '.d3' # в реальности они DO, D01 и D02...
ARCHIVE_7Z_EXTENSION, ARCHIVE_ZIP_EXTENSION, ARCHIVE_RAR_EXTENSION = '.7z', '.zip', '.rar'
WORD_1_EXTENSION, WORD_2_EXTENSION, WORD_3_EXTENSION = '.doc', '.docx', '.dot'


# Функция для обхода файловой системы
def copy_new_oscillograms(source_dir: str, dest_dir: str, copied_hashes: dict = {},
                          preserve_dir_structure: bool = True, use_hashes: bool = True,
                          use_comtrade: bool = True, use_new_comtrade: bool = True, use_brs: bool = True,
                          use_neva: bool = True, use_ekra: bool = True, use_parma: bool = True,
                          use_black_box: bool = True, use_res_3: bool = True, use_osc: bool = True,
                          _new_copied_hashes: dict = {}, _first_run: bool = True, _path_temp = None) -> int:
    """
    Копирует файлы осциллограмм из исходного каталога в целевой каталог, отслеживая скопированные файлы.

    Args:
        source_dir (str): путь к исходному каталогу.
        dest_dir (str): путь к целевому каталогу.
        copied_hashes (dict): хэш-таблица для отслеживания скопированных файлов (по умолчанию - пустой словарь).
        preserve_structure (bool): сохранять ли структуру директорий в целевом каталоге?
        use_hashes (bool): использовать ли хэш-суммы для проверки уникальности файлов?
        copy_comtrade (bool): копировать ли файлы Comtrade (.cfg и .dat)?
        copy_brs (bool): копировать ли файлы Bresler (.brs)?
        
        local variables
        _new_copied_hashes (dict): The dictionary of new copied file hashes.
        _first_run (bool): The flag indicating the first run.
        _path_temp (str): The temporary path.
    Returns:
        Возвращает колличество сохранённых осциллограм.
        При этом, создаются новые файлы в целевом каталоге:
        - обновляется файл "_hash_table"
        - создаётся новый файл "_new_hash_table"
    """
    # hash_table - хэш-таблица для отслеживания скопированных файлов
    count_new_files = 0
    print("Считаем общее количество файлов в исходной директории...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])  # Подсчет общего количества файлов
    print(f"Общее количество файлов: {total_files}, запускаем обработку...")
    with tqdm(total=total_files, desc="Copying files") as pbar:  # Инициализация прогресс-бара
        for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
            for file in files:  # Имя каждого файла
                pbar.update(1)  # Обновление прогресс-бара на один файл
                file_lower = file.lower()
                # TODO: переписать функции в одну / две, чтобы не повторяться
                if use_comtrade and file_lower.endswith(CFG_EXTENSION):  # Если файл имеет расширение .cfg
                    count_new_files += process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                    preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                    type_osc=TYPE_OSC.COMTRADE_CFG_DAT)
                
                elif use_new_comtrade and file_lower.endswith(CFF_EXTENSION):  # Если файл имеет расширение .cff (характерно для новых форматов Comtrade)
                    count_new_files += process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                    preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                    type_osc=TYPE_OSC.COMTRADE_CFF)
                
                elif use_brs and file_lower.endswith(BRS_EXTENSION):  # Если файл имеет расширение .brs (характерно для Бреслера)
                    count_new_files += process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                    preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                    type_osc=TYPE_OSC.BRESLER)
                 
                elif use_black_box and file_lower.endswith(BLACK_BOX_EXTENSION):  # Если файл имеет расширение .brs (характерно для Бреслера)
                    count_new_files += process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                    preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                    type_osc=TYPE_OSC.BLACK_BOX)
                    
                elif use_res_3 and file_lower.endswith(RES_3_EXTENSION):  # Если файл имеет расширение .brs (характерно для Бреслера)
                    count_new_files += process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                    preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                    type_osc=TYPE_OSC.RES_3)
                                    
                elif use_osc and file_lower.endswith(OSC_EXTENSION):  # Если файл имеет расширение .osc (Пока не выяснил, для кого характерно),
                    count_new_files += process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                    preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                    type_osc=TYPE_OSC.OSC)
                
                elif use_neva and NEVA_EXTENSION in file_lower and not file_lower.endswith('.xml'):  # Если файл имеет расширение .os (характерно для НЕВА),
                    # TODO: по сбору данных, проверить, всегда ли они имеют тип вида ".os1", где последняя цифра есть и может меняться.
                    # И если это так, то каков её размер? Может быть скорректировать поиск, так как сейчас не оптимален.
                    count_new_files += process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                    preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                    type_osc=TYPE_OSC.NEVA)
                
                elif use_ekra and file_lower.endswith(EKRA_EXTENSION):  # Если файл имеет расширение .DFR (характерно для специальных от ЭКРЫ)
                    count_new_files += process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                    preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                    type_osc=TYPE_OSC.EKRA)
                
                elif (use_parma and 
                      not (file_lower.endswith(WORD_1_EXTENSION) or file_lower.endswith(WORD_2_EXTENSION) or file_lower.endswith(WORD_3_EXTENSION) or 
                           ".doc" in file_lower) and
                      (PARMA_O_EXTENSION in file_lower or PARMA_0_EXTENSION in file_lower or PARMA_1_EXTENSION in file_lower or
                       PARMA_2_EXTENSION in file_lower) or PARMA_3_EXTENSION in file_lower):  # Если файл имеет расширение .do, d0, ?d1? (характерно для специальных от ПАРМЫ)
                    # TODO: по сбору данных, проверить, всегда ли они имеют тип вида .do, d01, d02, ?d11? , где последние цифра есть и может меняться.
                    # И если это так, то каков её размер? Может быть скорректировать поиск, так как сейчас не оптимален.
                    count_new_files += process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                    preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                    type_osc=TYPE_OSC.PARMA)
                
                elif (file_lower.endswith(ARCHIVE_7Z_EXTENSION) or file_lower.endswith(ARCHIVE_ZIP_EXTENSION) or
                    file_lower.endswith(ARCHIVE_RAR_EXTENSION) ):  # Если файл является архивом
                    count_new_files += process_archive_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                            preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                            _first_run=_first_run, _path_temp=_path_temp,
                                                            use_comtrade=use_comtrade, use_new_comtrade=use_new_comtrade, use_neva=use_neva, use_ekra=use_ekra,
                                                            use_brs=use_brs, use_black_box=use_black_box, use_res_3=use_res_3, use_parma=use_parma, use_osc=use_osc)
    
    if _first_run:
        print(f"Количество новых скопированных файлов: {count_new_files}") 
        # Сохранение JSON файлов hash_table и new_hash_table                        
        hash_table_file_path = os.path.join(dest_dir, '_hash_table.json')  # Формируем путь для сохранения hash_table
        if use_hashes: 
            try:
                with open(hash_table_file_path, 'w') as file:
                    json.dump(copied_hashes, file)  # Сохраняем hash_table в JSON файл
            except:
                print("Не удалось сохранить hash_table в JSON файл")

            data_now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            new_copied_hashes_file_path = os.path.join(dest_dir, f'new_hash_table_{data_now}.json')
            try:
                with open(new_copied_hashes_file_path, 'w') as file:
                    json.dump(_new_copied_hashes, file)
            except:
                print("Не удалось сохранить new_hash_table в JSON файл")
        return count_new_files
    else:
        return count_new_files

def process_file(file: str, root: str, source_dir: str, dest_dir: str, copied_hashes: dict = {}, 
                 preserve_dir_structure: bool = True, use_hashes: bool = True, _new_copied_hashes: dict = {},
                 new_folder: str = "New folder", type_osc: TYPE_OSC = TYPE_OSC.BRESLER) -> int:
    """
    Processes a single file, copying it to the destination directory and updating the copied_hashes dictionary.
    
    Args:
        file (str): The name of the file to process.
        root (str): The root directory of the file.
        dest_dir (str): The destination directory for the copied files.
        copied_hashes (dict): The dictionary of copied file hashes.
        preserve_dir_structure (bool): Whether to preserve the directory structure.
        use_hashes (bool): Whether to use file hashes for comparison.
        new_folder (str): the name of the new folder where the file will be saved
        type_osc (TYPE_OSC): The type of osc to process.
        
        local variables
        _new_copied_hashes (dict): The dictionary of new copied file hashes.
    Returns:
        int: The number of new files copied.
    """
    match type_osc:
        case TYPE_OSC.COMTRADE_CFG_DAT:
            return _process_comtrade_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                         preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes)
        case TYPE_OSC.COMTRADE_CFF:
            file = file[:-4] + CFF_EXTENSION # изменяем шрифт типа файла на строчный.
            if new_folder == "New folder": new_folder = "COMTRADE_CFF"
        case TYPE_OSC.BRESLER:
            file = file[:-4] + BRS_EXTENSION
            if new_folder == "New folder": new_folder = "BRESLER"
        case TYPE_OSC.EKRA:
            file = file[:-4] + EKRA_EXTENSION
            if new_folder == "New folder": new_folder = "EKRA"
        case TYPE_OSC.BLACK_BOX:
            file = file[:-3] + BLACK_BOX_EXTENSION
            if new_folder == "New folder": new_folder = "BLACK_BOX"
        case TYPE_OSC.RES_3:
            file = file[:-4] + RES_3_EXTENSION
            if new_folder == "New folder": new_folder = "RES_3"
        case TYPE_OSC.OSC:
            file = file[:-4] + OSC_EXTENSION
            if new_folder == "New folder": new_folder = "OSC"
        case TYPE_OSC.PARMA:
            # Тип файла не меняется, т.к. имеет множество вариантов
            if new_folder == "New folder": new_folder = "PARMA"
        case TYPE_OSC.NEVA:
            # Тип файла не меняется, т.к. имеет множество вариантов
            if new_folder == "New folder": new_folder = "NEVA"

    file_path = os.path.join(root, file)  # Получаем полный путь к файлу
    with open(file_path, 'rb') as f: # Открываем brs файл для чтения в бинарном режиме
        file_hash = hashlib.md5(f.read()).hexdigest()  # Вычисляем хэш-сумму dat файла
        if file_hash not in copied_hashes or not use_hashes:
            dest_subdir = os.path.relpath(root, source_dir)  # Получаем относительный путь от исходной директории до текущей директории
            if preserve_dir_structure:
                dest_path = os.path.join(dest_dir, new_folder, dest_subdir, file)  # Формируем путь для копирования файла
            else:
                dest_path = os.path.join(dest_dir, new_folder, file)  # Формируем путь для копирования файла
            if not os.path.exists(dest_path):
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Создаем все несуществующие директории для целевого файла
                shutil.copy2(file_path, dest_path)  # Копируем файл в целевую директорию
            
            if use_hashes:  
                copied_hashes[file_hash] = (file, file_path)  # Добавляем хэш-сумму файла в хэш-таблицу
                _new_copied_hashes[file_hash] = (file, file_path)
                return 1 # новый файл скопирован
    return 0 # новый файл НЕ скопирован

def _process_comtrade_file(file: str, root: str, source_dir: str, dest_dir: str, copied_hashes: dict = {}, 
                          preserve_dir_structure: bool = True, use_hashes: bool = True, _new_copied_hashes: dict = {}) -> int:
    """
    Processes a single file, copying it to the destination directory and updating the copied_hashes dictionary.
    
    Args:
        file (str): The name of the file to process.
        root (str): The root directory of the file.
        dest_dir (str): The destination directory for the copied files.
        copied_hashes (dict): The dictionary of copied file hashes.
        preserve_dir_structure (bool): Whether to preserve the directory structure.
        use_hashes (bool): Whether to use file hashes for comparison.
        
        local variables
        _new_copied_hashes (dict): The dictionary of new copied file hashes.
    Returns:
        int: The number of new files copied.
    """
    cfg_file = file[:-4] + CFG_EXTENSION # изменяем шрифт типа файла на строчный.
    cfg_file_path = os.path.join(root, cfg_file)  # Получаем полный путь к cfg файлу
    dat_file = cfg_file[:-4] + DAT_EXTENSION  # Формируем имя dat файла на основе имени cfg файла
    dat_file_path = os.path.join(root, dat_file)  # Получаем полный путь к dat файлу
    is_exist = os.path.exists(dat_file_path)  
    if is_exist:
        with open(dat_file_path, 'rb') as f:  # Открываем dat файл для чтения в бинарном режиме
            file_hash = hashlib.md5(f.read()).hexdigest()  # Вычисляем хэш-сумму dat файла
            if file_hash not in copied_hashes or not use_hashes:
                dest_subdir = os.path.relpath(root, source_dir)  # Получаем относительный путь от исходной директории до текущей директории
                
                if preserve_dir_structure:
                    dest_path = os.path.join(dest_dir, "COMTRADE_CFG_DAT", dest_subdir, cfg_file)  # Формируем путь для копирования cfg файла
                    dat_dest_path = os.path.join(dest_dir, "COMTRADE_CFG_DAT", dest_subdir, dat_file)  # Формируем путь для копирования dat файла
                else:
                    dest_path = os.path.join(dest_dir, "COMTRADE_CFG_DAT", cfg_file)
                    dat_dest_path = os.path.join(dest_dir, "COMTRADE_CFG_DAT", dat_file)
                
                if not os.path.exists(dest_path):
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Создаем все несуществующие директории для целевого файла
                    shutil.copy2(cfg_file_path, dest_path)  # Копируем cfg файл в целевую директорию
                
                if not os.path.exists(dat_dest_path):
                    os.makedirs(os.path.dirname(dat_dest_path), exist_ok=True)  # Создаем все несуществующие директории для целевого dat файла
                    shutil.copy2(dat_file_path, dat_dest_path)  # Копируем dat файл в целевую директорию
                    
                copied_hashes[file_hash] = (cfg_file, cfg_file_path)  # Добавляем хэш-сумму файла в хэш-таблицу
                _new_copied_hashes[file_hash] = (cfg_file, cfg_file_path)
                return 1 # новый файл скопирован
    return 0 # новый файл НЕ скопирован

def process_archive_file(file: str, root: str, source_dir: str, dest_dir: str, copied_hashes: dict = {}, 
                         preserve_dir_structure: bool = True, use_hashes: bool = True, _new_copied_hashes: dict = {},
                         _first_run: bool = False, _path_temp = None,
                         use_comtrade: bool = True, use_new_comtrade: bool = True, use_brs: bool = True,
                         use_neva: bool = True, use_ekra: bool = True, use_parma: bool = True,
                         use_black_box: bool = True, use_res_3: bool = True,) -> int:
    """
    Processes a single file, copying it to the destination directory and updating the copied_hashes dictionary.
    
    Args:
        file (str): The name of the file to process.
        root (str): The root directory of the file.
        dest_dir (str): The destination directory for the copied files.
        copied_hashes (dict): The dictionary of copied file hashes.
        preserve_dir_structure (bool): Whether to preserve the directory structure.
        use_hashes (bool): Whether to use file hashes for comparison.
        
        local variables
        _new_copied_hashes (dict): The dictionary of new copied file hashes.
    Returns:
        int: The number of new files copied.
    """
    count_new_files = 0
    file_path = os.path.join(root, file)  # Получаем полный путь к cfg файлу
    if _path_temp is None:
        _path_temp = os.path.join(dest_dir,'temp')  # Формируем путь для копирования
        source_dir_temp = _path_temp
        if preserve_dir_structure:
            dest_subdir = os.path.relpath(root, source_dir)  # Получаем относительный путь от исходной директории до текущей директории
            _path_temp = os.path.join(dest_dir,'temp', dest_subdir)  # Формируем путь для копирования
    else:
        _path_temp = os.path.join(_path_temp, 'temp')
        source_dir_temp = _path_temp
        
    # CСоздание дирректории, если она не сущетсвует
    os.makedirs(_path_temp, exist_ok=True)
    
    # определение типа и разархивация
    try:
        if file.lower().endswith(ARCHIVE_7Z_EXTENSION):
            with py7zr.SevenZipFile(file_path, mode='r') as archive:
                archive.extractall(_path_temp)  # Извлекаем все файлы из архива в целевую директорию
                dest_dir = os.path.join(dest_dir, "archive_7z")
        elif file.lower().endswith(ARCHIVE_ZIP_EXTENSION):
            with zipfile.ZipFile(file_path, 'r') as zf:
                zf.extractall(_path_temp)  # Извлекаем все файлы из архива в целевую директорию
                dest_dir = os.path.join(dest_dir, "archive_zip")
        elif file.lower().endswith(ARCHIVE_RAR_EXTENSION):
            with az.rar.RarArchive(file_path) as archive:
                archive.extract_to_directory(_path_temp)
            dest_dir = os.path.join(dest_dir, "archive_rar")
    except Exception as e:
        # FIXME: подумать над оформлением нормального лога.
        print(f"Ошибка при разархивации файла {_path_temp}: {e}")
    
    # FIXME: подумать над сохранением пути внутри архивов - пока эта функция не работает в полной мере.
    count_new_files += copy_new_oscillograms(source_dir=_path_temp, dest_dir=dest_dir, copied_hashes=copied_hashes, preserve_dir_structure=preserve_dir_structure,
                                             use_hashes=use_hashes, 
                                             _new_copied_hashes=_new_copied_hashes, _first_run = False, _path_temp=_path_temp,
                                             use_comtrade=use_comtrade, use_new_comtrade=use_new_comtrade, use_neva=use_neva, use_ekra=use_ekra,
                                             use_brs=use_brs, use_black_box=use_black_box, use_res_3=use_res_3, use_parma=use_parma)
    try:
        shutil.rmtree(source_dir_temp)  # Удаляем временную директорию
    except Exception as e:
        # FIXME: подумать над оформлением нормального лога.
        print(f"Ошибка при разархивации файла {_path_temp}: {e}")
        
    if _first_run:
        _path_temp = None
    else:
        _path_temp = _path_temp[:-5] # вычитает /temp

    return count_new_files

def match_oscillograms_to_terminals(dest_dir: str, copied_hashes: dict, terminal_oscillogram_names: dict) -> None:
    """
    Ищет коды осциллограмм в хэш-таблице и добавляет их в новый хэш-список.
      
    Args:
        dest_dir (str): каталог, для сохранения словаря осциллограмм.
        copied_hashes (dict): хэш-таблица для отслеживания скопированных файлов.
        terminal_oscillogram_names (dict): словарь для хранения кодов осциллограмм и их хэш-сумм.
    Returns:
        None
    """
    SCM_NAME = '\\ИПМ'
    new_osc_name_dict = {}
    new_osc_name_arr = [] # для ускорения работы циклов
    for osc_name in terminal_oscillogram_names.keys():
        new_osc_name_arr.append(osc_name)
        new_osc_name_dict[osc_name] = '\\'+osc_name[1:]
        
    # Проходим по всем файлам в папке
    with tqdm(total=len(copied_hashes) * len(new_osc_name_arr), desc="Matching oscillograms") as pbar:  # Инициализация прогресс-бара
        for key in copied_hashes.keys():  # имён по хэш-суммам в разы больше, чем терминалов
            for osc_name in new_osc_name_arr:
                pbar.update(1)  # Обновление прогресс-бара на один файл
                if osc_name in copied_hashes[key][0]:
                    terminal_oscillogram_names[osc_name].append(key)
                    break # если найдено имя осциллограммы, то прерываем цикл.
                elif ('ОТГРУЖЕННЫЕ ТЕРМИНАЛЫ И ШКАФЫ' in copied_hashes[key][1] and new_osc_name_dict[osc_name] in copied_hashes[key][1] and 
                      not SCM_NAME in copied_hashes[key][1]):
                    # сделано раздельно, чтобы отследить, что проверка добавлена не зря.
                    terminal_oscillogram_names[osc_name].append(key)
                    break
                
    
    osc_name_dict_file_path = os.path.join(dest_dir, '_osc_name_dict.json')  # Формируем путь для сохранения osc_name_dict
    try:
        with open(osc_name_dict_file_path, 'w') as file:
            json.dump(terminal_oscillogram_names, file)  # Сохраняем hash_table в JSON файл
    except:
        print("Не удалось сохранить osc_name_dict в JSON файл")

def organize_oscillograms_by_terminal(source_dir: str, dest_dir: str, terminal_list: list, terminal_oscillogram_names: dict) -> None:
    """
    Копирует осциллограммы структуируя их по номерам терминалов.
      
    Args:
        source_dir (str): исходный каталог.
        dest_dir (str): каталог, для сохранения скопированных файлов.
        terminal_list (list): список имён терминалов.
        osc_name_dict (dict): словарь для хранения терминалов и хеш имён осциллограмм.
    Returns:
        None
    """
    osc_terminal_dict = {}  # словарь для хранения имён осциллограмм и их принадлежности к терминалу
    # Функция формирующая обратный список к словарю osc_name_dict (из "терминал -> имя осц" в "имя осц -> терминал")
    for terminal_name in terminal_list:
        for osc_name in terminal_oscillogram_names[terminal_name]:
            osc_terminal_dict[osc_name] = terminal_name
    
    print("Считаем общее количество файлов в исходной директории...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])  # Подсчет общего количества файлов
    print(f"Общее количество файлов: {total_files}, запускаем обработку...")
    with tqdm(total=total_files, desc="Organize oscillograms") as pbar:  # Инициализация прогресс-бара
        for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
            for file in files:  # Имя каждого файла
                pbar.update(1)  # Обновление прогресс-бара на один файл
                if file.endswith(CFG_EXTENSION) and file[:-4] in osc_terminal_dict:  # Если файл имеет расширение .cfg и есть в списке имён интересующих осциллограмм
                    file_name = file[:-4]
                    cfg_file_name = file
                    dat_file_name = cfg_file_name[:-4] + DAT_EXTENSION
                    cfg_file_path = os.path.join(root, file)  # Формируем полный путь к cfg файлу
                    dat_file_path = cfg_file_path[:-4] + DAT_EXTENSION  # Формируем полный путь к dat файлу
                    # копируем файлы в соответствующу папку
                    if os.path.exists(cfg_file_path) and os.path.exists(dat_file_path):
                        cfg_dest_path = os.path.join(dest_dir, osc_terminal_dict[file_name], cfg_file_name)  # Формируем путь для копирования dat файла
                        dat_dest_path = os.path.join(dest_dir, osc_terminal_dict[file_name], dat_file_name)  # Формируем путь для копирования dat файла
                        if not os.path.exists(cfg_dest_path):
                            os.makedirs(os.path.dirname(cfg_dest_path), exist_ok=True)  # Создаем все несуществующие директории для целевого файла
                            shutil.copy2(cfg_file_path, cfg_dest_path)  # Копируем cfg файл в целевую директорию

                        if not os.path.exists(dat_dest_path):
                            os.makedirs(os.path.dirname(dat_dest_path), exist_ok=True)  # Создаем все несуществующие директории для целевого dat файла
                            shutil.copy2(dat_file_path, dat_dest_path)  # Копируем dat файл в целевую директорию              


# Пример использования функции

copied_hashes = {}
hash_table_path = DEST_DIR +  '/_hash_table.json'
# source_directory = input("Введите путь в которой искать: ")
# destination_directory = input("Введите путь в которую сохранять: ")
# destination_directory_hash_table = input("Введите путь к папке с файлом '_hash_table.json': ")
try:
    with open(hash_table_path, 'r') as file:
        copied_hashes = json.load(file)
except:
    print("Не удалось прочитать hash_table из JSON файла")


copy_new_oscillograms(source_dir=SOURCE_DIR, dest_dir=DEST_DIR, copied_hashes=copied_hashes, use_parma=False, use_neva=False)
# terminal_oscillogram_names = {}
# terminal_oscillogram_names["t00108"], terminal_oscillogram_names["t00209"], terminal_oscillogram_names["t00331"], terminal_oscillogram_names["t00363"] = [], [], [], []
# for i in range(1, 500): # пока лишь до 500, потом расширим
#     terminal_oscillogram_names[f"t{i:05}"] = []
# match_oscillograms_to_terminals(destination_directory, copied_hashes, terminal_oscillogram_names)

#
#  # Перенос файлов терминала в папку с осциллограммами
#  osc_name_dict = {}
#  terminal_oscillogram_names_path = DEST_DIR +  '/_osc_name_dict (по терминалам) v1.json'
#  # Путь к целевой директории
#  dataset_for_primary_research_directory = 'D:/DataSet/dataset_for_primary_research'
#  try:
#      with open(terminal_oscillogram_names_path, 'r') as file:
#          osc_name_dict = json.load(file)
#  except:
#      print("Не удалось прочитать hash_table из JSON файла")
#  terminal_list = []
#  for i in range(1, 100): # пока лишь до 100, потом расширим
#      terminal_name = f"t{i:05}"
#      terminal_list.append(terminal_name)
#  # organize_oscillograms_by_terminal(destination_directory, dataset_for_primary_research_directory, terminal_list, osc_name_dict)
    