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
import comtrade

# TODO: add file logging

# Create an ".exe" file: 
# 1) on the command line, I enter "pip install auto-py-to-exe"
# 2) then on the command line "python -m auto_py_to_exe"

class TYPE_OSC(Enum):
        # TODO: написать полноценные описание типов
        COMTRADE_CFG_DAT = "Comtrade type files consisting of two files .cfg and .dat"
        COMTRADE_CFF = "Comtrade type files consisting of a single .cff file"
        BRESLER = "File type (.brs) typical for the terminal manufacturer NPP Bresler LLC, official link: https://www.bresler.ru /"
        BLACK_BOX = "black_box (.bb)" #??
        RES_3 = "File type (.sg2) typical for the terminal manufacturer Prosoft-Systems LLC, official link : https://prosoftsystems.ru / and a link to the devices https://prosoftsystems.ru/catalog/show/cifrovoj--registrator-jelektricheskih-sobytij-pjes3 "
        EKRA = "File type (.dfr) typical for the terminal manufacturer NPP EKRA LLC, official link: https://ekra.ru /"
        PARMA = "File type (.do) typical for the terminal manufacturer PARMA LLC, official link: https://parma.spb.ru /"
        PARMA_TO = "File type (.to) typical for the manufacturer of terminals LLC PARMA, this type is intended for registration of long-term processes, official link: https://parma.spb.ru /"
        NEVA = "File type (.os1 and similar) typical for the terminal manufacturer NPF ENERGOSOYUZ, official link: https://www.energosoyuz.spb.ru/ru and a link to the devices https://www.energosoyuz.spb.ru/ru/content/registrator-avariynyh-sobytiy-neva-ras "
        OSC = "File format (.osc) - the contact binding has not yet been fully clarified"

class SearchOscillograms():
    """
    Library for primary search of waveforms in folders and primary processing.
    Supports file formats: 'cfg'+'dat', 'cff', 'brs', 'dfr', 'osc', 'os2' and analogues, 'bb', 'sg2', 'do' and analogues, 'to'.
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
        self.PARMA_O_EXTENSION, self.PARMA_ZERO_EXTENSION = '.do', '.d0' # in reality, they are DO, D01 and D02...
        self.PARMA_TO_EXTENSION, self.PARMA_T_ZERO_EXTENSION = '.to', '.t0' # it opens with the TO viewer, probably the T01 files are superfluous...
        self.ARCHIVE_7Z_EXTENSION, self.ARCHIVE_ZIP_EXTENSION, self.ARCHIVE_RAR_EXTENSION = '.7z', '.zip', '.rar'
        self.WORD_1_EXTENSION, self.WORD_2_EXTENSION, self.WORD_3_EXTENSION, self.PDF_EXTENSION = '.doc', '.docx', '.dot', '.pdf'


    def copy_new_oscillograms(self, source_dir: str, dest_dir: str, copied_hashes: dict = {},
                            preserve_dir_structure: bool = True, use_hashes: bool = True,
                            types_to_copy: list[TYPE_OSC] = None,
                            _new_copied_hashes: dict = {}, _first_run: bool = True, _path_temp = None, 
                            progress_callback = None, stop_processing_fn = None, is_write_names = None, **kwargs) -> int:
        """
        Copies oscillogram files from the source directory to the target directory, keeping track of the copied files.
    
        Args:
            source_dir (str): the path to the source directory.
            dest_dir (str): the path to the target directory.
            copied_hashes (dict): A hash table for tracking copied files (by default, an empty dictionary).
            preserve_dir_structure (bool): do I save the directory structure in the target directory?
            use_hashes (bool): do I use hash sums to verify the uniqueness of files?
            types_to_copy (list[TYPE_OSC], optional): A list of oscillogram types (enum TYPE_OSC) to copy.
                                                     If None or empty, all types will be copied. Defaults to None.
            
            local variables
            _new_copied_hashes (dict): Dictionary of hashes of newly copied files.
            _first_run (bool): Flag indicating the first launch.
            _path_temp (str): Temporary path.
            
            External variables
            progress_callback (callback): Called when the progress callback (For external reading by the program)
            stop_processing_fn (callback): Called when the program stops processing 
            is_write_names_fn (bool): Flag indicating whether to write names of processed files.
        Returns:
            Returns the number of saved oscillograms.
            At the same time, new files are created in the target directory:
            - the "_hash_table" file is updated - a hash table for tracking copied files
            - a new file "_new_hash_table" is being created
        """
        count_new_files = 0
        actual_types_to_copy = list(TYPE_OSC) if not types_to_copy else types_to_copy
         
        if _first_run:
            # TODO: rewrite the functions into one so as not to repeat
            if progress_callback:
                progress_callback("Starting the process of counting total files in the source directory...")
            print("We count the total number of files in the source directory...")
            total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
            print(f"Total number of files: {total_files}, starting processing...")
            if progress_callback:
                progress_callback(f"Total number of files: {total_files}, starting processing...")
            with tqdm(total=total_files, desc="Copying files") as pbar:
                for root, dirs, files in os.walk(source_dir):
                    if stop_processing_fn and stop_processing_fn():
                        progress_callback("The process was interrupted by the user.")
                        break
                    for file in files:
                        if stop_processing_fn and stop_processing_fn():
                            progress_callback("The process was interrupted by the user.")
                            break
                        
                        pbar.update(1)
                        file_lower = file.lower()
                        if progress_callback and is_write_names:
                            if is_write_names:
                                progress_callback(f"Processing file: {root} / {file}")
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
    
                        elif TYPE_OSC.OSC in actual_types_to_copy and file_lower.endswith(self.OSC_EXTENSION):  # I have not yet found out who the osc format is typical for
                            count_new_files +=self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                 preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                 type_osc=TYPE_OSC.OSC)
    
                        elif TYPE_OSC.NEVA in actual_types_to_copy and self.NEVA_EXTENSION in file_lower and not file_lower.endswith('.xml'):
                            # TODO: The search needs to be finalized. files do not always have the number 1 at the end (.os1). 
                            # It may be necessary to adjust the search, as it is not optimal right now.
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
                            # TODO: The search needs to be finalized. It may be necessary to adjust the search, as it is not optimal right now.
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
    
            
            print(f"The number of newly copied files: {count_new_files}") 
            if progress_callback:
                progress_callback(f"The number of newly copied files: {count_new_files}")
            # Saving hash_table and new_hash_table JSON files                      
            hash_table_file_path = os.path.join(dest_dir, '_hash_table.json')
            if use_hashes: 
                try:
                    with open(hash_table_file_path, 'w') as file:
                        json.dump(copied_hashes, file) # Saving the file
                    if progress_callback:
                        progress_callback("Successfully saved _hash_table.json")
                except:
                    print("Failed to save hash_table to JSON file")
                    if progress_callback:
                        progress_callback("Failed to save _hash_table.json")
    
                data_now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                new_copied_hashes_file_path = os.path.join(dest_dir, f'new_hash_table_{data_now}.json')
                try:
                    with open(new_copied_hashes_file_path, 'w') as file:
                        json.dump(_new_copied_hashes, file)
                        if progress_callback:
                            progress_callback(f"Successfully saved new hash table as {new_copied_hashes_file_path}")
                except:
                    print("Failed to save new_hash_table to JSON file")
                    if progress_callback:
                        progress_callback("Failed to save new_hash_table JSON file")
            return count_new_files
        else:
            # A copy of the work inside the archives
            # TODO: rewrite the functions into one so as not to repeat
            for root, dirs, files in os.walk(source_dir):
                if stop_processing_fn and stop_processing_fn():
                    progress_callback("The process was interrupted by the user.")
                    break
                for file in files:
                    if stop_processing_fn and stop_processing_fn():
                        progress_callback("The process was interrupted by the user.")
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
                file = file[:-4] + self.CFF_EXTENSION
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
                # The file type does not change, because it has different variants
                if new_folder == "New folder": new_folder = "PARMA"
            case TYPE_OSC.PARMA_TO:
                # The file type does not change, because it has different variants
                if new_folder == "New folder": new_folder = "PARMA_TO"
            case TYPE_OSC.NEVA:
                # The file type does not change, because it has different variants
                if new_folder == "New folder": new_folder = "NEVA"

        file_path = os.path.join(root, file) # Getting the full path to the file
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()  # Calculate the hash sum of the dat file
            if file_hash not in copied_hashes or not use_hashes:
                dest_subdir = os.path.relpath(root, source_dir)  # Getting a relative path
                if preserve_dir_structure:
                    dest_path = os.path.join(dest_dir, new_folder, dest_subdir, file)
                else:
                    dest_path = os.path.join(dest_dir, new_folder, file)
                if not os.path.exists(dest_path):
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Creating all non-existent directories for the target file
                    shutil.copy2(file_path, dest_path)

                if use_hashes:  
                    copied_hashes[file_hash] = (file, file_path)  # Adding the hash sum of the file to the hash table
                    _new_copied_hashes[file_hash] = (file, file_path)
                    return 1
        return 0

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
        # The work is similar with "_process_file"
        cfg_file = file[:-4] + self.CFG_EXTENSION
        cfg_file_path = os.path.join(root, cfg_file)
        dat_file = cfg_file[:-4] + self.DAT_EXTENSION
        dat_file_path = os.path.join(root, dat_file)
        is_exist = os.path.exists(dat_file_path)  
        if is_exist:
            with open(dat_file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash not in copied_hashes or not use_hashes:
                    dest_subdir = os.path.relpath(root, source_dir)

                    if preserve_dir_structure:
                        dest_path = os.path.join(dest_dir, "COMTRADE_CFG_DAT", dest_subdir, cfg_file)
                        dat_dest_path = os.path.join(dest_dir, "COMTRADE_CFG_DAT", dest_subdir, dat_file)
                    else:
                        dest_path = os.path.join(dest_dir, "COMTRADE_CFG_DAT", cfg_file)
                        dat_dest_path = os.path.join(dest_dir, "COMTRADE_CFG_DAT", dat_file)

                    if not os.path.exists(dest_path):
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.copy2(cfg_file_path, dest_path)

                    if not os.path.exists(dat_dest_path):
                        os.makedirs(os.path.dirname(dat_dest_path), exist_ok=True)
                        shutil.copy2(dat_file_path, dat_dest_path)

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
        Processes a single archive file, extracting it and then processing its contents.

        Args:
            file (str): The name of the archive file to process.
            root (str): The root directory of the archive file.
            dest_dir (str): The destination directory for the copied files.
            copied_hashes (dict): The dictionary of copied file hashes.
            preserve_dir_structure (bool): Whether to preserve the directory structure.
            use_hashes (bool): Whether to use file hashes for comparison.
            types_to_copy (list[TYPE_OSC]): A list of oscillogram types to copy from the archive.

            local variables
            _new_copied_hashes (dict): The dictionary of new copied file hashes.
        Returns:
            int: The number of new files copied.
        """
        count_new_files = 0
        file_path = os.path.join(root, file) # Getting the full path to the file
        if _path_temp is None:
            _path_temp = os.path.join(dest_dir,'temp')  # Forming a path for copying
            source_dir_temp = _path_temp
            if preserve_dir_structure:
                dest_subdir = os.path.relpath(root, source_dir)
                _path_temp = os.path.join(dest_dir,'temp', dest_subdir)
        else:
            _path_temp = os.path.join(_path_temp, 'temp')
            source_dir_temp = _path_temp

        # # Creating a directory if it does not exist
        os.makedirs(_path_temp, exist_ok=True)

        try: # Attempt to identify the type and unarchive
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
            # FIXME: Think about the design of a normal log.
            print(f"An error occurred while unzipping the file {_path_temp}: {e}")

        # FIXME: Think about saving the path inside the archives - this feature does not work fully yet.
        count_new_files += self.copy_new_oscillograms(source_dir=_path_temp, dest_dir=dest_dir, copied_hashes=copied_hashes, preserve_dir_structure=preserve_dir_structure,
                                                      use_hashes=use_hashes, types_to_copy=types_to_copy,
                                                      _new_copied_hashes=_new_copied_hashes, _first_run = False, _path_temp=_path_temp,
                                                      progress_callback=progress_callback, stop_processing_fn=stop_processing_fn, is_write_names=is_write_names)
        try:
            shutil.rmtree(source_dir_temp) # Deleting the temporary directory
        except Exception as e:
            # FIXME: Think about the design of a normal log.
            print(f"Ошибка удаления файлов файла {_path_temp}: {e}")

        if _first_run:
            _path_temp = None
        else:
            _path_temp = _path_temp[:-5] # subtracts "/temp"

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
        Copies the waveforms structuring them by terminal numbers.
          
        Args:
            source_dir (str): the source directory.
            dest_dir (str): The directory to save the copied files.
            terminal_list (list): a list of terminal names.
            osc_name_dict (dict): a dictionary for storing terminals and hash names of waveforms.
        Returns:
            None
        """
        osc_terminal_dict = {}  # dictionary for storing waveform names and their terminal accessories
        # A function that forms a reverse list to the osc_name_dict dictionary (from "terminal -> osc name" to "osc name -> terminal")
        for terminal_name in terminal_list:
            for osc_name in terminal_oscillogram_names[terminal_name]:
                osc_terminal_dict[osc_name] = terminal_name
        
        print("We count the total number of files in the source directory...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Total number of files: {total_files}, starting processing...")
        with tqdm(total=total_files, desc="Organize oscillograms") as pbar:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    pbar.update(1)
                    if file.endswith(self.CFG_EXTENSION):
                        file_name = file[:-4]
                        if not is_hashes:
                            dat_file = file[:-4] + self.DAT_EXTENSION
                            dat_path = os.path.join(root, dat_file)
                            with open(dat_path, 'rb') as f:
                                file_name = hashlib.md5(f.read()).hexdigest()

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
    terminals_to_search = [1, 2, 3]
    f.find_terminal_hashes_from_json(input_json_path, terminals_to_search, output_json_path)

