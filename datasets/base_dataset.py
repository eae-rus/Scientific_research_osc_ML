from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

from common.utils import get_short_names_ml_signals


class BaseDataset(ABC):
    """Base class for splitting dataframe to train, val and test."""
    def __init__(self, csv_path=None):
        self.df = pd.read_csv(csv_path, index_col=['file_name', 'sample'])
        self.target = self._set_target()
        self.train_mask, self.val_mask, self.test_mask = self._train_test_split()

    @abstractmethod
    def _set_target(self, ):
        """
        This method has to be implemented by all children.
        """
        pass

    @abstractmethod
    def _train_test_split(self, ):
        """
        This method has to be implemented by all children.
        """
        pass

class FDDDataset(BaseDataset):
    def _set_target(self):
        # Get lists of signals for each event
        _, ml_opr_swch, ml_abnorm_evnt, ml_emerg_evnt = get_short_names_ml_signals()
        
        # Masks for each event
        target_opr = self.df.filter(ml_opr_swch).any(axis=1).astype(int)
        target_abnorm = self.df.filter(ml_abnorm_evnt).any(axis=1).astype(int)
        target_emerg = self.df.filter(ml_emerg_evnt).any(axis=1).astype(int)
        
        # Combine into a DataFrame
        target = pd.DataFrame({
            'opr_swch': target_opr,
            'abnorm_evnt': target_abnorm,
            'emerg_evnt': target_emerg
        }, index=self.df.index)
        return target

    def _train_test_split(self):
        # Первые 32 символа из имени файла для группировки
        file_groups = self.df.index.get_level_values('file_name').str[:32]
        unique_files = file_groups.unique()
        
        # Стратификация по emerg_evnt (максимум в группе файлов)
        file_labels = self.target.groupby(file_groups).max()
        strat_col = file_labels['emerg_evnt']  # Только emerg_evnt
        
        # Первое разделение (train_val + test)
        files_train_val, files_test = train_test_split(
            unique_files,
            test_size=0.2,
            stratify=strat_col,  # Стратификация по emerg_evnt
            random_state=42
        )
        
        # Второе разделение (train + val)
        strat_train_val = file_labels.loc[files_train_val, 'emerg_evnt']  # Только emerg_evnt
        files_train, files_val = train_test_split(
            files_train_val,
            test_size=0.25,
            stratify=strat_train_val,  # Стратификация по emerg_evnt
            random_state=42
        )
        
        # Создание масок
        return (
            file_groups.isin(files_train),
            file_groups.isin(files_val),
            file_groups.isin(files_test)
        )
