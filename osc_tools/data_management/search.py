import os
import shutil
import hashlib
import json
import datetime
import py7zr
import zipfile
import aspose.zip as az
from tqdm import tqdm

from osc_tools.core.constants import TYPE_OSC

# TODO: добавить логирование в файл

# Создание файла ".exe":
# 1) в командной строке ввести "pip install auto-py-to-exe"
# 2) далее в командной строке "python -m auto_py_to_exe"

class SearchOscillograms():
    """
    Библиотека для первичного поиска осциллограмм в папках и первичной обработки.
    Поддерживает форматы файлов: 'cfg'+'dat', 'cff', 'brs', 'dfr', 'osc', 'os2' и аналоги, 'bb', 'sg2', 'do' и аналоги, 'to'.
    """

    def __init__(self):
        self.CFG_EXTENSION, self.DAT_EXTENSION = '.cfg', '.dat'
        self.CFF_EXTENSION = '.cff'
        self.BRS_EXTENSION = '.brs'
        self.EKRA_EXTENSION = '.dfr'
        self.OSC_EXTENSION = '.osc'
        self.NEVA_EXTENSION = '.os'
        self.BLACK_BOX_EXTENSION = '.bb'
        self.RES_3_EXTENSION = '.sg2'
        self.PARMA_O_EXTENSION, self.PARMA_ZERO_EXTENSION = '.do', '.d0' # в реальности это DO, D01 и D02...
        self.PARMA_TO_EXTENSION, self.PARMA_T_ZERO_EXTENSION = '.to', '.t0' # открывается просмотрщиком ТО, вероятно, файлы T01 лишние...
        self.ARCHIVE_7Z_EXTENSION, self.ARCHIVE_ZIP_EXTENSION, self.ARCHIVE_RAR_EXTENSION = '.7z', '.zip', '.rar'
        self.WORD_1_EXTENSION, self.WORD_2_EXTENSION, self.WORD_3_EXTENSION, self.PDF_EXTENSION = '.doc', '.docx', '.dot', '.pdf'


    def copy_new_oscillograms(self, source_dir: str, dest_dir: str, copied_hashes: dict = {},
                            preserve_dir_structure: bool = True, use_hashes: bool = True,
                            types_to_copy: list[TYPE_OSC] = None,
                            _new_copied_hashes: dict = {}, _first_run: bool = True, _path_temp = None, 
                            progress_callback = None, stop_processing_fn = None, is_write_names = None, **kwargs) -> int:
        """
        Копирует файлы осциллограмм из исходного каталога в целевой, отслеживая скопированные файлы.
    
        Args:
            source_dir (str): путь к исходному каталогу.
            dest_dir (str): путь к целевому каталогу.
            copied_hashes (dict): хеш-таблица для отслеживания скопированных файлов (по умолчанию пустой словарь).
            preserve_dir_structure (bool): сохранять ли структуру каталогов в целевом каталоге?
            use_hashes (bool): использовать ли хеш-суммы для проверки уникальности файлов?
            types_to_copy (list[TYPE_OSC], optional): список типов осциллограмм (enum TYPE_OSC) для копирования.
                                                     Если None или пуст, будут скопированы все типы. По умолчанию None.
            
            локальные переменные
            _new_copied_hashes (dict): словарь хешей вновь скопированных файлов.
            _first_run (bool): флаг, указывающий на первый запуск.
            _path_temp (str): временный путь.
            
            Внешние переменные
            progress_callback (callback): вызывается при обратном вызове прогресса (для внешнего чтения программой)
            stop_processing_fn (callback): вызывается, когда программа прекращает обработку
            is_write_names_fn (bool): флаг, указывающий, нужно ли записывать имена обработанных файлов.
        Returns:
            Возвращает количество сохраненных осциллограмм.
            При этом в целевом каталоге создаются новые файлы:
            - обновляется файл "_hash_table" - хеш-таблица для отслеживания скопированных файлов
            - создается новый файл "_new_hash_table"
        """
        count_new_files = 0
        actual_types_to_copy = list(TYPE_OSC) if not types_to_copy else types_to_copy
         
        if _first_run:
            # TODO: переписать функции в одну, чтобы не повторяться
            if progress_callback:
                progress_callback("Начинается процесс подсчета общего количества файлов в исходном каталоге...")
            print("Подсчитываем общее количество файлов в исходном каталоге...")
            total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
            print(f"Общее количество файлов: {total_files}, начинаем обработку...")
            if progress_callback:
                progress_callback(f"Общее количество файлов: {total_files}, начинаем обработку...")
            with tqdm(total=total_files, desc="Копирование файлов") as pbar:
                for root, dirs, files in os.walk(source_dir):
                    if stop_processing_fn and stop_processing_fn():
                        progress_callback("Процесс был прерван пользователем.")
                        break
                    for file in files:
                        if stop_processing_fn and stop_processing_fn():
                            progress_callback("Процесс был прерван пользователем.")
                            break
                        
                        pbar.update(1)
                        file_lower = file.lower()
                        if progress_callback and is_write_names:
                            if is_write_names:
                                progress_callback(f"Обработка файла: {root} / {file}")
                            else:
                                progress_callback()
                        if TYPE_OSC.COMTRADE_CFG_DAT in actual_types_to_copy and file_lower.endswith(self.CFG_EXTENSION):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.COMTRADE_CFG_DAT)
    
                        elif TYPE_OSC.COMTRADE_CFF in actual_types_to_copy and file_lower.endswith(self.CFF_EXTENSION):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.COMTRADE_CFF)
    
                        elif TYPE_OSC.BRESLER in actual_types_to_copy and file_lower.endswith(self.BRS_EXTENSION):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.BRESLER)
    
                        elif TYPE_OSC.BLACK_BOX in actual_types_to_copy and file_lower.endswith(self.BLACK_BOX_EXTENSION):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.BLACK_BOX)
    
                        elif TYPE_OSC.RES_3 in actual_types_to_copy and file_lower.endswith(self.RES_3_EXTENSION):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.RES_3)
    
                        elif TYPE_OSC.OSC in actual_types_to_copy and file_lower.endswith(self.OSC_EXTENSION):  # Я пока не выяснил, для кого типичен формат osc
                            count_new_files +=self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                 preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                 type_osc=TYPE_OSC.OSC)
    
                        elif TYPE_OSC.NEVA in actual_types_to_copy and self.NEVA_EXTENSION in file_lower and not file_lower.endswith('.xml'):
                            # TODO: Поиск нужно доработать. файлы не всегда имеют цифру 1 в конце (.os1).
                            # Возможно, потребуется скорректировать поиск, так как сейчас он не оптимален.
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.NEVA)
    
                        elif TYPE_OSC.EKRA in actual_types_to_copy and file_lower.endswith(self.EKRA_EXTENSION):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.EKRA)
    
                        elif TYPE_OSC.PARMA_TO in actual_types_to_copy and (self.PARMA_TO_EXTENSION in file_lower or self.PARMA_T_ZERO_EXTENSION in file_lower):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.PARMA_TO)
    
                        elif (TYPE_OSC.PARMA in actual_types_to_copy and
                              not (file_lower.endswith(self.WORD_1_EXTENSION) or file_lower.endswith(self.WORD_2_EXTENSION) or file_lower.endswith(self.WORD_3_EXTENSION) or 
                                   file_lower.endswith(self.PDF_EXTENSION) or ".doc" in file_lower) and
                              (self.PARMA_O_EXTENSION in file_lower or self.PARMA_ZERO_EXTENSION in file_lower)):  # Если файл имеет расширение .do, d0, ?d1? (характерно для специальных от ПАРМЫ)
                            # TODO: Поиск нужно доработать. Возможно, потребуется скорректировать поиск, так как сейчас он не оптимален.
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.PARMA)
    
                        elif (file_lower.endswith(self.ARCHIVE_7Z_EXTENSION) or file_lower.endswith(self.ARCHIVE_ZIP_EXTENSION) or
                            file_lower.endswith(self.ARCHIVE_RAR_EXTENSION) ):
                            count_new_files += self._process_archive_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                          preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                          _first_run=_first_run, _path_temp=_path_temp,
                                                                          types_to_copy=actual_types_to_copy,
                                                                          progress_callback=progress_callback, stop_processing_fn=stop_processing_fn, is_write_names=is_write_names)
    
            
            print(f"Количество вновь скопированных файлов: {count_new_files}")
            if progress_callback:
                progress_callback(f"Количество вновь скопированных файлов: {count_new_files}")
            # Сохранение JSON-файлов hash_table и new_hash_table
            hash_table_file_path = os.path.join(dest_dir, '_hash_table.json')
            if use_hashes: 
                try:
                    with open(hash_table_file_path, 'w') as file:
                        json.dump(copied_hashes, file) # Сохранение файла
                    if progress_callback:
                        progress_callback("Успешно сохранено _hash_table.json")
                except:
                    print("Failed to save hash_table to JSON file")
                    if progress_callback:
                        progress_callback("Не удалось сохранить _hash_table.json")
    
                data_now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                new_copied_hashes_file_path = os.path.join(dest_dir, f'new_hash_table_{data_now}.json')
                try:
                    with open(new_copied_hashes_file_path, 'w') as file:
                        json.dump(_new_copied_hashes, file)
                        if progress_callback:
                            progress_callback(f"Успешно сохранена новая хеш-таблица как {new_copied_hashes_file_path}")
                except:
                    print("Не удалось сохранить new_hash_table в JSON-файл")
                    if progress_callback:
                        progress_callback("Не удалось сохранить JSON-файл new_hash_table")
            return count_new_files
        else:
            # Копия работы внутри архивов
            # TODO: переписать функции в одну, чтобы не повторяться
            for root, dirs, files in os.walk(source_dir):
                if stop_processing_fn and stop_processing_fn():
                    progress_callback("Процесс был прерван пользователем.")
                    break
                for file in files:
                    if stop_processing_fn and stop_processing_fn():
                        progress_callback("Процесс был прерван пользователем.")
                        break
                    file_lower = file.lower()
                    if TYPE_OSC.COMTRADE_CFG_DAT in actual_types_to_copy and file_lower.endswith(self.CFG_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.COMTRADE_CFG_DAT)
    
                    elif TYPE_OSC.COMTRADE_CFF in actual_types_to_copy and file_lower.endswith(self.CFF_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.COMTRADE_CFF)
    
                    elif TYPE_OSC.BRESLER in actual_types_to_copy and file_lower.endswith(self.BRS_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.BRESLER)
    
                    elif TYPE_OSC.BLACK_BOX in actual_types_to_copy and file_lower.endswith(self.BLACK_BOX_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.BLACK_BOX)
    
                    elif TYPE_OSC.RES_3 in actual_types_to_copy and file_lower.endswith(self.RES_3_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.RES_3)
    
                    elif TYPE_OSC.OSC in actual_types_to_copy and file_lower.endswith(self.OSC_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.OSC)
    
                    elif TYPE_OSC.NEVA in actual_types_to_copy and self.NEVA_EXTENSION in file_lower and not file_lower.endswith('.xml'):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.NEVA)
    
                    elif TYPE_OSC.EKRA in actual_types_to_copy and file_lower.endswith(self.EKRA_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.EKRA)

                    elif TYPE_OSC.PARMA_TO in actual_types_to_copy and (self.PARMA_TO_EXTENSION in file_lower or self.PARMA_T_ZERO_EXTENSION in file_lower):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.PARMA_TO)

                    elif (TYPE_OSC.PARMA in actual_types_to_copy and
                          not (file_lower.endswith(self.WORD_1_EXTENSION) or file_lower.endswith(self.WORD_2_EXTENSION) or file_lower.endswith(self.WORD_3_EXTENSION) or 
                               ".doc" in file_lower) and
                          (self.PARMA_O_EXTENSION in file_lower or self.PARMA_ZERO_EXTENSION)):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.PARMA)
    
                    elif (file_lower.endswith(self.ARCHIVE_7Z_EXTENSION) or file_lower.endswith(self.ARCHIVE_ZIP_EXTENSION) or
                        file_lower.endswith(self.ARCHIVE_RAR_EXTENSION) ):
                        count_new_files += self._process_archive_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                 preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                 _first_run=_first_run, _path_temp=_path_temp,
                                                                 types_to_copy=actual_types_to_copy,
                                                                 progress_callback=progress_callback, stop_processing_fn=stop_processing_fn, is_write_names=is_write_names)
            
            return count_new_files

    def _process_file(self, file: str, root: str, source_dir: str, dest_dir: str, copied_hashes: dict = {}, 
                     preserve_dir_structure: bool = True, use_hashes: bool = True, _new_copied_hashes: dict = {},
                     new_folder: str = "Новая папка", type_osc: TYPE_OSC = TYPE_OSC.BRESLER) -> int:
        """
        Обрабатывает один файл, копируя его в каталог назначения и обновляя словарь copied_hashes.

        Args:
            file (str): имя обрабатываемого файла.
            root (str): корневой каталог файла.
            dest_dir (str): каталог назначения для скопированных файлов.
            copied_hashes (dict): словарь хешей скопированных файлов.
            preserve_dir_structure (bool): сохранять ли структуру каталогов.
            use_hashes (bool): использовать ли хеши файлов для сравнения.
            new_folder (str): имя новой папки, в которую будет сохранен файл
            type_osc (TYPE_OSC): тип обрабатываемой осциллограммы.

            локальные переменные
            _new_copied_hashes (dict): словарь хешей новых скопированных файлов.
        Returns:
            int: количество скопированных новых файлов.
        """
        match type_osc:
            case TYPE_OSC.COMTRADE_CFG_DAT:
                return self._process_comtrade_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                   preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes)
            case TYPE_OSC.COMTRADE_CFF:
                file = file[:-4] + self.CFF_EXTENSION
                if new_folder == "Новая папка": new_folder = "COMTRADE_CFF"
            case TYPE_OSC.BRESLER:
                file = file[:-4] + self.BRS_EXTENSION
                if new_folder == "Новая папка": new_folder = "BRESLER"
            case TYPE_OSC.EKRA:
                file = file[:-4] + self.EKRA_EXTENSION
                if new_folder == "Новая папка": new_folder = "EKRA"
            case TYPE_OSC.BLACK_BOX:
                file = file[:-3] + self.BLACK_BOX_EXTENSION
                if new_folder == "Новая папка": new_folder = "BLACK_BOX"
            case TYPE_OSC.RES_3:
                file = file[:-4] + self.RES_3_EXTENSION
                if new_folder == "Новая папка": new_folder = "RES_3"
            case TYPE_OSC.OSC:
                file = file[:-4] + self.OSC_EXTENSION
                if new_folder == "Новая папка": new_folder = "OSC"
            case TYPE_OSC.PARMA:
                # Тип файла не меняется, так как у него есть разные варианты
                if new_folder == "Новая папка": new_folder = "PARMA"
            case TYPE_OSC.PARMA_TO:
                # Тип файла не меняется, так как у него есть разные варианты
                if new_folder == "Новая папка": new_folder = "PARMA_TO"
            case TYPE_OSC.NEVA:
                # Тип файла не меняется, так как у него есть разные варианты
                if new_folder == "Новая папка": new_folder = "NEVA"

        file_path = os.path.join(root, file) # Получение полного пути к файлу
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()  # Вычисление хеш-суммы файла dat
            if file_hash not in copied_hashes or not use_hashes:
                dest_subdir = os.path.relpath(root, source_dir)  # Получение относительного пути
                if preserve_dir_structure:
                    dest_path = os.path.join(dest_dir, new_folder, dest_subdir, file)
                else:
                    dest_path = os.path.join(dest_dir, new_folder, file)
                if not os.path.exists(dest_path):
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Создание всех несуществующих каталогов для целевого файла
                    shutil.copy2(file_path, dest_path)

                if use_hashes:  
                    copied_hashes[file_hash] = (file, file_path)  # Добавление хеш-суммы файла в хеш-таблицу
                    _new_copied_hashes[file_hash] = (file, file_path)
                    return 1
        return 0

    def _process_comtrade_file(self, file: str, root: str, source_dir: str, dest_dir: str, copied_hashes: dict = {}, 
                              preserve_dir_structure: bool = True, use_hashes: bool = True, _new_copied_hashes: dict = {}) -> int:
        """
        Обрабатывает один файл, копируя его в каталог назначения и обновляя словарь copied_hashes.

        Args:
            file (str): имя обрабатываемого файла.
            root (str): корневой каталог файла.
            dest_dir (str): каталог назначения для скопированных файлов.
            copied_hashes (dict): словарь хешей скопированных файлов.
            preserve_dir_structure (bool): сохранять ли структуру каталогов.
            use_hashes (bool): использовать ли хеши файлов для сравнения.

            локальные переменные
            _new_copied_hashes (dict): словарь хешей новых скопированных файлов.
        Returns:
            int: количество скопированных новых файлов.
        """
        # Работа аналогична "_process_file"
        cfg_file = file[:-4] + self.CFG_EXTENSION
        cfg_file_path = os.path.join(root, cfg_file)
        dat_file = cfg_file[:-4] + self.DAT_EXTENSION
        dat_file_path = os.path.join(root, dat_file)
        is_exist = os.path.exists(dat_file_path)  
        import logging
        if is_exist:
            try:
                with open(dat_file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
            except Exception as e:
                logging.error(f"[copy_new_oscillograms] Ошибка чтения dat-файла: {e} ({dat_file_path})")
                return 0
            if file_hash not in copied_hashes or not use_hashes:
                dest_subdir = os.path.relpath(root, source_dir)

                if preserve_dir_structure:
                    dest_path = os.path.join(dest_dir, "COMTRADE_CFG_DAT", dest_subdir, cfg_file)
                    dat_dest_path = os.path.join(dest_dir, "COMTRADE_CFG_DAT", dest_subdir, dat_file)
                else:
                    dest_path = os.path.join(dest_dir, "COMTRADE_CFG_DAT", cfg_file)
                    dat_dest_path = os.path.join(dest_dir, "COMTRADE_CFG_DAT", dat_file)

                try:
                    if not os.path.exists(dest_path):
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.copy2(cfg_file_path, dest_path)
                except FileNotFoundError as e:
                    logging.error(f"[copy_new_oscillograms] FileNotFoundError: {e} (cfg: {cfg_file_path} → {dest_path})")
                    return 0
                except Exception as e:
                    logging.error(f"[copy_new_oscillograms] Unexpected error: {e} (cfg: {cfg_file_path} → {dest_path})")
                    return 0

                try:
                    if not os.path.exists(dat_dest_path):
                        os.makedirs(os.path.dirname(dat_dest_path), exist_ok=True)
                        shutil.copy2(dat_file_path, dat_dest_path)
                except FileNotFoundError as e:
                    logging.error(f"[copy_new_oscillograms] FileNotFoundError: {e} (dat: {dat_file_path} → {dat_dest_path})")
                    return 0
                except Exception as e:
                    logging.error(f"[copy_new_oscillograms] Unexpected error: {e} (dat: {dat_file_path} → {dat_dest_path})")
                    return 0

                copied_hashes[file_hash] = (cfg_file, cfg_file_path)
                _new_copied_hashes[file_hash] = (cfg_file, cfg_file_path)
                return 1
        return 0

    def _process_archive_file(self, file: str, root: str, source_dir: str, dest_dir: str, copied_hashes: dict = {}, 
                             preserve_dir_structure: bool = True, use_hashes: bool = True, _new_copied_hashes: dict = {},
                             _first_run: bool = False, _path_temp = None,
                             types_to_copy: list[TYPE_OSC] = None,
                             progress_callback = None, stop_processing_fn = None, is_write_names = None, **kwargs) -> int:
        """
        Обрабатывает один архивный файл, извлекая его, а затем обрабатывая его содержимое.

        Args:
            file (str): имя обрабатываемого архивного файла.
            root (str): корневой каталог архивного файла.
            dest_dir (str): каталог назначения для скопированных файлов.
            copied_hashes (dict): словарь хешей скопированных файлов.
            preserve_dir_structure (bool): сохранять ли структуру каталогов.
            use_hashes (bool): использовать ли хеши файлов для сравнения.
            types_to_copy (list[TYPE_OSC]): список типов осциллограмм для копирования из архива.

            локальные переменные
            _new_copied_hashes (dict): словарь хешей новых скопированных файлов.
        Returns:
            int: количество скопированных новых файлов.
        """
        count_new_files = 0
        file_path = os.path.join(root, file) # Получение полного пути к файлу
        if _path_temp is None:
            _path_temp = os.path.join(dest_dir,'temp')  # Формирование пути для копирования
            source_dir_temp = _path_temp
            if preserve_dir_structure:
                dest_subdir = os.path.relpath(root, source_dir)
                _path_temp = os.path.join(dest_dir,'temp', dest_subdir)
        else:
            _path_temp = os.path.join(_path_temp, 'temp')
            source_dir_temp = _path_temp

        # # Создание каталога, если он не существует
        os.makedirs(_path_temp, exist_ok=True)

        try: # Попытка определить тип и разархивировать
            if file.lower().endswith(self.ARCHIVE_7Z_EXTENSION):
                with py7zr.SevenZipFile(file_path, mode='r') as archive:
                    archive.extractall(_path_temp)
                    dest_dir = os.path.join(dest_dir, "archive_7z")
            elif file.lower().endswith(self.ARCHIVE_ZIP_EXTENSION):
                with zipfile.ZipFile(file_path, 'r') as zf:
                    zf.extractall(_path_temp)
                    dest_dir = os.path.join(dest_dir, "archive_zip")
            elif file.lower().endswith(self.ARCHIVE_RAR_EXTENSION):
                with az.rar.RarArchive(file_path) as archive:
                    archive.extract_to_directory(_path_temp)
                dest_dir = os.path.join(dest_dir, "archive_rar")
        except Exception as e:
            # FIXME: Подумайте о дизайне нормального лога.
            print(f"Произошла ошибка при распаковке файла {_path_temp}: {e}")

        # FIXME: Подумайте о сохранении пути внутри архивов - эта функция пока работает не полностью.
        count_new_files += self.copy_new_oscillograms(source_dir=_path_temp, dest_dir=dest_dir, copied_hashes=copied_hashes, preserve_dir_structure=preserve_dir_structure,
                                                      use_hashes=use_hashes, types_to_copy=types_to_copy,
                                                      _new_copied_hashes=_new_copied_hashes, _first_run = False, _path_temp=_path_temp,
                                                      progress_callback=progress_callback, stop_processing_fn=stop_processing_fn, is_write_names=is_write_names)
        try:
            shutil.rmtree(source_dir_temp) # Удаление временного каталога
        except Exception as e:
            # FIXME: Подумайте о дизайне нормального лога.
            print(f"Ошибка удаления файлов файла {_path_temp}: {e}")

        if _first_run:
            _path_temp = None
        else:
            _path_temp = _path_temp[:-5] # вычитает "/temp"

        return count_new_files
            
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
