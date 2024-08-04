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

# TODO: добавить логирование ошибок

# создаю exe файл через 
# pip install auto-py-to-exe
# и затем опять же в командной строке
# python -m auto_py_to_exe

class TYPE_OSC(Enum):
        # TODO: написать полноценные описание типов
        COMTRADE_CFG_DAT = "comtrade_cfg_dat"
        COMTRADE_CFF = "comtrade_cff"
        BRESLER = "bresler"
        BLACK_BOX = "black_box"
        RES_3 = "res_3"
        EKRA = "ekra"
        PARMA = "parma"
        PARMA_TO = "parma_to"
        NEVA = "neva"
        OSC = "osc"

class SearchOscillograms():
    """
    Библиотека для первичного поиска осциллограмм в папках и первичной обработки.
    Поддерживает форматы файлов: 'cfg'+'dat', 'cff', 'brs', 'dfr', 'osc', 'os2' и аналоги, 'bb', 'sg2', 'do' и аналоги.
    """
    
    def __init__(self):
        self.CFG_EXTENSION, self.DAT_EXTENSION = '.cfg', '.dat'
        self.CFF_EXTENSION = '.cff' # Форматы Comtrade (cff - новый) 
        self.BRS_EXTENSION = '.brs' # Бреслер
        self.EKRA_EXTENSION = '.dfr' # Внутренний формат ЭКРЫ
        self.OSC_EXTENSION = '.osc' # Пока не выясинл что за тип
        self.NEVA_EXTENSION = '.os' # Внутрении НЕВЫ. В реальности они os1, os2...
        self.BLACK_BOX_EXTENSION = '.bb' # Чёрный ящик
        self.RES_3_EXTENSION = '.sg2' # РЭС 3, пока не научился их открывать.
        self.PARMA_O_EXTENSION, self.PARMA_ZERO_EXTENSION = '.do', '.d0' # в реальности они DO, D01 и D02...
        self.PARMA_TO_EXTENSION, self.PARMA_T_ZERO_EXTENSION = '.to', '.t0' # открывается просмотрщиком TO, T01...
        self.ARCHIVE_7Z_EXTENSION, self.ARCHIVE_ZIP_EXTENSION, self.ARCHIVE_RAR_EXTENSION = '.7z', '.zip', '.rar'
        self.WORD_1_EXTENSION, self.WORD_2_EXTENSION, self.WORD_3_EXTENSION, self.PDF_EXTENSION = '.doc', '.docx', '.dot', '.pdf'


    def copy_new_oscillograms(self, source_dir: str, dest_dir: str, copied_hashes: dict = {},
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
         
        if _first_run:
            # FIXME: Переписать формирование прогресс бара, пока это очень криво
            
            print("Считаем общее количество файлов в исходной директории...")
            total_files = sum([len(files) for r, d, files in os.walk(source_dir)])  # Подсчет общего количества файлов
            print(f"Общее количество файлов: {total_files}, запускаем обработку...")
            with tqdm(total=total_files, desc="Copying files") as pbar:  # Инициализация прогресс-бара
                for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
                    for file in files:  # Имя каждого файла
                        pbar.update(1)  # Обновление прогресс-бара на один файл
                        file_lower = file.lower()
                        # TODO: переписать функции в одну / две, чтобы не повторяться
                        if use_comtrade and file_lower.endswith(self.CFG_EXTENSION):  # Если файл имеет расширение .cfg
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.COMTRADE_CFG_DAT)
    
                        elif use_new_comtrade and file_lower.endswith(self.CFF_EXTENSION):  # Если файл имеет расширение .cff (характерно для новых форматов Comtrade)
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.COMTRADE_CFF)
    
                        elif use_brs and file_lower.endswith(self.BRS_EXTENSION):  # Если файл имеет расширение .brs (характерно для Бреслера)
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.BRESLER)
    
                        elif use_black_box and file_lower.endswith(self.BLACK_BOX_EXTENSION):  # Если файл имеет расширение .brs (характерно для Бреслера)
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.BLACK_BOX)
    
                        elif use_res_3 and file_lower.endswith(self.RES_3_EXTENSION):  # Если файл имеет расширение .brs (характерно для Бреслера)
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.RES_3)
    
                        elif use_osc and file_lower.endswith(self.OSC_EXTENSION):  # Если файл имеет расширение .osc (Пока не выяснил, для кого характерно),
                            count_new_files +=self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                 preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                 type_osc=TYPE_OSC.OSC)
    
                        elif use_neva and self.NEVA_EXTENSION in file_lower and not file_lower.endswith('.xml'):  # Если файл имеет расширение .os (характерно для НЕВА),
                            # TODO: по сбору данных, проверить, всегда ли они имеют тип вида ".os1", где последняя цифра есть и может меняться.
                            # И если это так, то каков её размер? Может быть скорректировать поиск, так как сейчас не оптимален.
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.NEVA)
    
                        elif use_ekra and file_lower.endswith(self.EKRA_EXTENSION):  # Если файл имеет расширение .DFR (характерно для специальных от ЭКРЫ)
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.EKRA)
    
                        elif use_parma and (self.PARMA_TO_EXTENSION in file_lower or self.PARMA_T_ZERO_EXTENSION in file_lower):  # Если файл имеет расширение .TO или .T0x (характерно для специальных от ПАРМЫ)
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.PARMA_TO)
    
                        elif (use_parma and 
                              not (file_lower.endswith(self.WORD_1_EXTENSION) or file_lower.endswith(self.WORD_2_EXTENSION) or file_lower.endswith(self.WORD_3_EXTENSION) or 
                                   file_lower.endswith(self.PDF_EXTENSION) or ".doc" in file_lower) and
                              (self.PARMA_O_EXTENSION in file_lower or self.PARMA_ZERO_EXTENSION in file_lower)):  # Если файл имеет расширение .do, d0, ?d1? (характерно для специальных от ПАРМЫ)
                            # TODO: по сбору данных, проверить, всегда ли они имеют тип вида .do, d01, d02, где последние цифра есть и может меняться.
                            # И если это так, то каков её размер? Может быть скорректировать поиск, так как сейчас не оптимален.
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.PARMA)
    
                        elif (file_lower.endswith(self.ARCHIVE_7Z_EXTENSION) or file_lower.endswith(self.ARCHIVE_ZIP_EXTENSION) or
                            file_lower.endswith(self.ARCHIVE_RAR_EXTENSION) ):  # Если файл является архивом
                            count_new_files += self._process_archive_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                          preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                          _first_run=_first_run, _path_temp=_path_temp,
                                                                          use_comtrade=use_comtrade, use_new_comtrade=use_new_comtrade, use_neva=use_neva, use_ekra=use_ekra,
                                                                          use_brs=use_brs, use_black_box=use_black_box, use_res_3=use_res_3, use_parma=use_parma, use_osc=use_osc)
    
            
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
            for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
                for file in files:  # Имя каждого файла
                    file_lower = file.lower()
                    # TODO: переписать функции в одну / две, чтобы не повторяться
                    if use_comtrade and file_lower.endswith(self.CFG_EXTENSION):  # Если файл имеет расширение .cfg
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.COMTRADE_CFG_DAT)
    
                    elif use_new_comtrade and file_lower.endswith(self.CFF_EXTENSION):  # Если файл имеет расширение .cff (характерно для новых форматов Comtrade)
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.COMTRADE_CFF)
    
                    elif use_brs and file_lower.endswith(self.BRS_EXTENSION):  # Если файл имеет расширение .brs (характерно для Бреслера)
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.BRESLER)
    
                    elif use_black_box and file_lower.endswith(self.BLACK_BOX_EXTENSION):  # Если файл имеет расширение .brs (характерно для Бреслера)
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.BLACK_BOX)
    
                    elif use_res_3 and file_lower.endswith(self.RES_3_EXTENSION):  # Если файл имеет расширение .brs (характерно для Бреслера)
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.RES_3)
    
                    elif use_osc and file_lower.endswith(self.OSC_EXTENSION):  # Если файл имеет расширение .osc (Пока не выяснил, для кого характерно),
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.OSC)
    
                    elif use_neva and self.NEVA_EXTENSION in file_lower and not file_lower.endswith('.xml'):  # Если файл имеет расширение .os (характерно для НЕВА),
                        # TODO: по сбору данных, проверить, всегда ли они имеют тип вида ".os1", где последняя цифра есть и может меняться.
                        # И если это так, то каков её размер? Может быть скорректировать поиск, так как сейчас не оптимален.
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.NEVA)
    
                    elif use_ekra and file_lower.endswith(self.EKRA_EXTENSION):  # Если файл имеет расширение .DFR (характерно для специальных от ЭКРЫ)
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.EKRA)

                    elif use_parma and (self.PARMA_TO_EXTENSION in file_lower or self.PARMA_T_ZERO_EXTENSION in file_lower):  # Если файл имеет расширение .TO или .T0x (характерно для специальных от ПАРМЫ)
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.PARMA_TO)

                    elif (use_parma and 
                          not (file_lower.endswith(self.WORD_1_EXTENSION) or file_lower.endswith(self.WORD_2_EXTENSION) or file_lower.endswith(self.WORD_3_EXTENSION) or 
                               ".doc" in file_lower) and
                          (self.PARMA_O_EXTENSION in file_lower or self.PARMA_ZERO_EXTENSION)):  # Если файл имеет расширение .do, d0, ?d1? (характерно для специальных от ПАРМЫ)
                        # TODO: по сбору данных, проверить, всегда ли они имеют тип вида .do, d01, d02, ?d11? , где последние цифра есть и может меняться.
                        # И если это так, то каков её размер? Может быть скорректировать поиск, так как сейчас не оптимален.
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.PARMA)
    
                    elif (file_lower.endswith(self.ARCHIVE_7Z_EXTENSION) or file_lower.endswith(self.ARCHIVE_ZIP_EXTENSION) or
                        file_lower.endswith(self.ARCHIVE_RAR_EXTENSION) ):  # Если файл является архивом
                        count_new_files += self._process_archive_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                 preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                 _first_run=_first_run, _path_temp=_path_temp,
                                                                 use_comtrade=use_comtrade, use_new_comtrade=use_new_comtrade, use_neva=use_neva, use_ekra=use_ekra,
                                                                 use_brs=use_brs, use_black_box=use_black_box, use_res_3=use_res_3, use_parma=use_parma, use_osc=use_osc)
            
            return count_new_files

    def _process_file(self, file: str, root: str, source_dir: str, dest_dir: str, copied_hashes: dict = {}, 
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
                return self._process_comtrade_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                   preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes)
            case TYPE_OSC.COMTRADE_CFF:
                file = file[:-4] + self.CFF_EXTENSION # изменяем шрифт типа файла на строчный.
                if new_folder == "New folder": new_folder = "COMTRADE_CFF"
            case TYPE_OSC.BRESLER:
                file = file[:-4] + self.BRS_EXTENSION
                if new_folder == "New folder": new_folder = "BRESLER"
            case TYPE_OSC.EKRA:
                file = file[:-4] + self.EKRA_EXTENSION
                if new_folder == "New folder": new_folder = "EKRA"
            case TYPE_OSC.BLACK_BOX:
                file = file[:-3] + self.BLACK_BOX_EXTENSION
                if new_folder == "New folder": new_folder = "BLACK_BOX"
            case TYPE_OSC.RES_3:
                file = file[:-4] + self.RES_3_EXTENSION
                if new_folder == "New folder": new_folder = "RES_3"
            case TYPE_OSC.OSC:
                file = file[:-4] + self.OSC_EXTENSION
                if new_folder == "New folder": new_folder = "OSC"
            case TYPE_OSC.PARMA:
                # Тип файла не меняется, т.к. имеет множество вариантов
                if new_folder == "New folder": new_folder = "PARMA"
            case TYPE_OSC.PARMA_TO:
                # Тип файла не меняется, т.к. имеет множество вариантов
                if new_folder == "New folder": new_folder = "PARMA_TO"
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

    def _process_comtrade_file(self, file: str, root: str, source_dir: str, dest_dir: str, copied_hashes: dict = {}, 
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
        cfg_file = file[:-4] + self.CFG_EXTENSION # изменяем шрифт типа файла на строчный.
        cfg_file_path = os.path.join(root, cfg_file)  # Получаем полный путь к cfg файлу
        dat_file = cfg_file[:-4] + self.DAT_EXTENSION  # Формируем имя dat файла на основе имени cfg файла
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

    def _process_archive_file(self, file: str, root: str, source_dir: str, dest_dir: str, copied_hashes: dict = {}, 
                             preserve_dir_structure: bool = True, use_hashes: bool = True, _new_copied_hashes: dict = {},
                             _first_run: bool = False, _path_temp = None,
                             use_comtrade: bool = True, use_new_comtrade: bool = True, use_brs: bool = True,
                             use_neva: bool = True, use_ekra: bool = True, use_parma: bool = True,
                             use_black_box: bool = True, use_res_3: bool = True, use_osc: bool = True) -> int:
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
            if file.lower().endswith(self.ARCHIVE_7Z_EXTENSION):
                with py7zr.SevenZipFile(file_path, mode='r') as archive:
                    archive.extractall(_path_temp)  # Извлекаем все файлы из архива в целевую директорию
                    dest_dir = os.path.join(dest_dir, "archive_7z")
            elif file.lower().endswith(self.ARCHIVE_ZIP_EXTENSION):
                with zipfile.ZipFile(file_path, 'r') as zf:
                    zf.extractall(_path_temp)  # Извлекаем все файлы из архива в целевую директорию
                    dest_dir = os.path.join(dest_dir, "archive_zip")
            elif file.lower().endswith(self.ARCHIVE_RAR_EXTENSION):
                with az.rar.RarArchive(file_path) as archive:
                    archive.extract_to_directory(_path_temp)
                dest_dir = os.path.join(dest_dir, "archive_rar")
        except Exception as e:
            # FIXME: подумать над оформлением нормального лога.
            print(f"Ошибка при разархивации файла {_path_temp}: {e}")

        # FIXME: подумать над сохранением пути внутри архивов - пока эта функция не работает в полной мере.
        count_new_files += self.copy_new_oscillograms(source_dir=_path_temp, dest_dir=dest_dir, copied_hashes=copied_hashes, preserve_dir_structure=preserve_dir_structure,
                                                      use_hashes=use_hashes, 
                                                      _new_copied_hashes=_new_copied_hashes, _first_run = False, _path_temp=_path_temp,
                                                      use_comtrade=use_comtrade, use_new_comtrade=use_new_comtrade, use_neva=use_neva, use_ekra=use_ekra,
                                                      use_brs=use_brs, use_black_box=use_black_box, use_res_3=use_res_3, use_parma=use_parma, use_osc=use_osc)
        try:
            shutil.rmtree(source_dir_temp)  # Удаляем временную директорию
        except Exception as e:
            # FIXME: подумать над оформлением нормального лога.
            print(f"Ошибка удаления файлов файла {_path_temp}: {e}")

        if _first_run:
            _path_temp = None
        else:
            _path_temp = _path_temp[:-5] # вычитает /temp

        return count_new_files

    def match_oscillograms_to_terminals(self, source_dir: str, copied_hashes: dict, terminal_oscillogram_names: dict) -> None:
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
        with tqdm(total=len(copied_hashes), desc="Matching oscillograms") as pbar:  # Инициализация прогресс-бара
            for key in copied_hashes.keys():  # имён по хэш-суммам в разы больше, чем терминалов
                pbar.update(1)  # Обновление прогресс-бара на один файл
                for osc_name in new_osc_name_arr:
                    if osc_name in copied_hashes[key][0]:
                        terminal_oscillogram_names[osc_name].append(key)
                        break # если найдено имя осциллограммы, то прерываем цикл.
                    elif ('ОТГРУЖЕННЫЕ ТЕРМИНАЛЫ И ШКАФЫ' in copied_hashes[key][1] and new_osc_name_dict[osc_name] in copied_hashes[key][1] and 
                          not SCM_NAME in copied_hashes[key][1]):
                        # сделано раздельно, чтобы отследить, что проверка добавлена не зря.
                        terminal_oscillogram_names[osc_name].append(key)
                        break
                    
                    
        osc_name_dict_file_path = os.path.join(source_dir, '_osc_name_dict.json')  # Формируем путь для сохранения osc_name_dict
        try:
            with open(osc_name_dict_file_path, 'w') as file:
                json.dump(terminal_oscillogram_names, file)  # Сохраняем hash_table в JSON файл
        except:
            print("Не удалось сохранить osc_name_dict в JSON файл")

    def organize_oscillograms_by_terminal(self, source_dir: str, dest_dir: str, terminal_list: list, terminal_oscillogram_names: dict, is_hashes: bool = True) -> None:
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
                    if file.endswith(self.CFG_EXTENSION):
                        file_name = file[:-4]
                        if not is_hashes:
                            dat_file = file[:-4] + self.DAT_EXTENSION
                            dat_path = os.path.join(root, dat_file)
                            with open(dat_path, 'rb') as f:
                                file_name = hashlib.md5(f.read()).hexdigest()

                        if file_name in osc_terminal_dict:  # Если файл имеет расширение .cfg и есть в списке имён интересующих осциллограмм
                            cfg_file_name = file
                            dat_file_name = cfg_file_name[:-4] + self.DAT_EXTENSION
                            cfg_file_path = os.path.join(root, file)  # Формируем полный путь к cfg файлу
                            dat_file_path = cfg_file_path[:-4] + self.DAT_EXTENSION  # Формируем полный путь к dat файлу
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