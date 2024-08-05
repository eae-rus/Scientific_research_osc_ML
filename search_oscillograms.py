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
        PARMA_TO = "File type (.dfr) typical for the terminal manufacturer NPP EKRA LLC, official link: https://ekra.ru /"
        PARMA = "File type (.to) typical for the manufacturer of terminals LLC PARMA, this type is intended for registration of long-term processes, official link: https://parma.spb.ru /"
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
                              use_comtrade: bool = True, use_new_comtrade: bool = True, use_brs: bool = True,
                              use_neva: bool = True, use_ekra: bool = True, use_parma: bool = True,
                              use_black_box: bool = True, use_res_3: bool = True, use_osc: bool = True,
                              _new_copied_hashes: dict = {}, _first_run: bool = True, _path_temp = None) -> int:
        """
        Copies oscillogram files from the source directory to the target directory, keeping track of the copied files.
    
        Args:
            source_dir (str): the path to the source directory.
            dest_dir (str): the path to the target directory.
            copied_hashes (dict): A hash table for tracking copied files (by default, an empty dictionary).
            preserve_structure (bool): do I save the directory structure in the target directory?
            use_hashes (bool): do I use hash sums to verify the uniqueness of files?
            copy_comtrade (bool): do I copy Comtrade files (.cfg and .dat)?
            copy_brs (bool): do I copy Bresler files (.brs)?
            
            local variables
            _new_copied_hashes (dict): Dictionary of hashes of newly copied files.
            _first_run (bool): Flag indicating the first launch.
            _path_temp (str): Temporary path.
        Returns:
            Returns the number of saved oscillograms.
            At the same time, new files are created in the target directory:
            - the "_hash_table" file is updated - a hash table for tracking copied files
            - a new file "_new_hash_table" is being created
        """
        count_new_files = 0
         
        if _first_run:
            # TODO: rewrite the functions into one so as not to repeat
            print("We count the total number of files in the source directory...")
            total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
            print(f"Total number of files: {total_files}, starting processing...")
            with tqdm(total=total_files, desc="Copying files") as pbar:
                for root, dirs, files in os.walk(source_dir):
                    for file in files:
                        pbar.update(1)
                        file_lower = file.lower()
                        if use_comtrade and file_lower.endswith(self.CFG_EXTENSION):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.COMTRADE_CFG_DAT)
    
                        elif use_new_comtrade and file_lower.endswith(self.CFF_EXTENSION):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.COMTRADE_CFF)
    
                        elif use_brs and file_lower.endswith(self.BRS_EXTENSION):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.BRESLER)
    
                        elif use_black_box and file_lower.endswith(self.BLACK_BOX_EXTENSION):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.BLACK_BOX)
    
                        elif use_res_3 and file_lower.endswith(self.RES_3_EXTENSION):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.RES_3)
    
                        elif use_osc and file_lower.endswith(self.OSC_EXTENSION):  # I have not yet found out who the osc format is typical for
                            count_new_files +=self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                 preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                 type_osc=TYPE_OSC.OSC)
    
                        elif use_neva and self.NEVA_EXTENSION in file_lower and not file_lower.endswith('.xml'):
                            # TODO: The search needs to be finalized. files do not always have the number 1 at the end (.os1). 
                            # It may be necessary to adjust the search, as it is not optimal right now.
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.NEVA)
    
                        elif use_ekra and file_lower.endswith(self.EKRA_EXTENSION):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.EKRA)
    
                        elif use_parma and (self.PARMA_TO_EXTENSION in file_lower or self.PARMA_T_ZERO_EXTENSION in file_lower):
                            count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                                  preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                                  type_osc=TYPE_OSC.PARMA_TO)
    
                        elif (use_parma and 
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
                                                                          use_comtrade=use_comtrade, use_new_comtrade=use_new_comtrade, use_neva=use_neva, use_ekra=use_ekra,
                                                                          use_brs=use_brs, use_black_box=use_black_box, use_res_3=use_res_3, use_parma=use_parma, use_osc=use_osc)
    
            
            print(f"The number of newly copied files: {count_new_files}") 
            # Saving hash_table and new_hash_table JSON files                      
            hash_table_file_path = os.path.join(dest_dir, '_hash_table.json')
            if use_hashes: 
                try:
                    with open(hash_table_file_path, 'w') as file:
                        json.dump(copied_hashes, file) # Saving the file
                except:
                    print("Failed to save hash_table to JSON file")
    
                data_now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                new_copied_hashes_file_path = os.path.join(dest_dir, f'new_hash_table_{data_now}.json')
                try:
                    with open(new_copied_hashes_file_path, 'w') as file:
                        json.dump(_new_copied_hashes, file)
                except:
                    print("Failed to save new_hash_table to JSON file")
            return count_new_files
        else:
            # A copy of the work inside the archives
            # TODO: rewrite the functions into one so as not to repeat
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_lower = file.lower()
                    if use_comtrade and file_lower.endswith(self.CFG_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.COMTRADE_CFG_DAT)
    
                    elif use_new_comtrade and file_lower.endswith(self.CFF_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.COMTRADE_CFF)
    
                    elif use_brs and file_lower.endswith(self.BRS_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.BRESLER)
    
                    elif use_black_box and file_lower.endswith(self.BLACK_BOX_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.BLACK_BOX)
    
                    elif use_res_3 and file_lower.endswith(self.RES_3_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.RES_3)
    
                    elif use_osc and file_lower.endswith(self.OSC_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.OSC)
    
                    elif use_neva and self.NEVA_EXTENSION in file_lower and not file_lower.endswith('.xml'):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.NEVA)
    
                    elif use_ekra and file_lower.endswith(self.EKRA_EXTENSION):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.EKRA)

                    elif use_parma and (self.PARMA_TO_EXTENSION in file_lower or self.PARMA_T_ZERO_EXTENSION in file_lower):
                        count_new_files += self._process_file(file=file, root=root, source_dir=source_dir, dest_dir=dest_dir, copied_hashes=copied_hashes, 
                                                              preserve_dir_structure=preserve_dir_structure, use_hashes=use_hashes, _new_copied_hashes=_new_copied_hashes,
                                                              type_osc=TYPE_OSC.PARMA_TO)

                    elif (use_parma and 
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
                                                      use_hashes=use_hashes, 
                                                      _new_copied_hashes=_new_copied_hashes, _first_run = False, _path_temp=_path_temp,
                                                      use_comtrade=use_comtrade, use_new_comtrade=use_new_comtrade, use_neva=use_neva, use_ekra=use_ekra,
                                                      use_brs=use_brs, use_black_box=use_black_box, use_res_3=use_res_3, use_parma=use_parma, use_osc=use_osc)
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

    def match_oscillograms_to_terminals(self, source_dir: str, copied_hashes: dict, terminal_oscillogram_names: dict) -> None:
        """
        Searches for oscillogram codes in the hash table and adds them to a new hash list.

        Args:
            dest_dir (str): A directory to save the waveform dictionary.
            copied_hashes (dict): A hash table for tracking copied files.
            terminal_oscillogram_name (dict): a dictionary for storing waveform codes and their hash sums.
        Returns:
            None
        """
        SCM_NAME = '\\ИПМ'
        new_osc_name_dict = {}
        new_osc_name_arr = [] # It is taken out separately to speed up the work of cycles
        for osc_name in terminal_oscillogram_names.keys():
            new_osc_name_arr.append(osc_name)
            new_osc_name_dict[osc_name] = '\\'+osc_name[1:]

        # Go through all the files in the folder
        with tqdm(total=len(copied_hashes), desc="Matching oscillograms") as pbar:
            for key in copied_hashes.keys():  # there are many times more names for hash sums than terminals. Therefore, we go through the terminals.
                pbar.update(1)
                for osc_name in new_osc_name_arr:
                    if osc_name in copied_hashes[key][0]:
                        terminal_oscillogram_names[osc_name].append(key)
                        break # if the name of the waveform is found, then we interrupt the cycle.
                    elif ('ОТГРУЖЕННЫЕ ТЕРМИНАЛЫ И ШКАФЫ' in copied_hashes[key][1] and new_osc_name_dict[osc_name] in copied_hashes[key][1] and 
                          not SCM_NAME in copied_hashes[key][1]):
                        # done separately to make sure that the check was added for a reason.
                        terminal_oscillogram_names[osc_name].append(key)
                        break
                    
                    
        osc_name_dict_file_path = os.path.join(source_dir, '_osc_name_dict.json')
        try:
            with open(osc_name_dict_file_path, 'w') as file:
                json.dump(terminal_oscillogram_names, file)  # Saving the file
        except:
            print("Не удалось сохранить osc_name_dict в JSON файл")

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