from datasets.base_dataset import BaseDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from common.utils import get_short_names_ml_signals


class FDDDataset(BaseDataset):
    def __init__(self, csv_path=None):
        super().__init__(csv_path)
        self.df = self.df[['IA', 'IC', 'UA BB', 'UB BB', 'UC BB']]

    def _set_target(self):
        # Get lists of signals for each event
        _, ml_opr_swch, ml_abnorm_evnt, ml_emerg_evnt = get_short_names_ml_signals()
        
        # Masks for each event
        target_opr = self.df.filter(ml_opr_swch).any(axis=1).astype(int)
        target_abnorm = self.df.filter(ml_abnorm_evnt).any(axis=1).astype(int)
        target_emerg = self.df.filter(ml_emerg_evnt).any(axis=1).astype(int)
        
        # Calculate norm_state where all events are 0
        norm_state = ((target_opr + target_abnorm + target_emerg) == 0).astype(int)
        
        # Create target with priority: emerg_evnt (2) > abnorm_evnt (1) > (opr_swch or norm_state) (0)
        conditions = [
            target_emerg.astype(bool),
            target_abnorm.astype(bool),
            (target_opr.astype(bool) | norm_state.astype(bool))
        ]
        choices = [2, 1, 0]
        target = pd.Series(np.select(conditions, choices, default=0), index=self.df.index, name='target')
        
        return target

    def _train_test_split(self):
        # Grouping by first 32 symbols from file name
        file_groups = self.df.index.get_level_values('file_name').str[:32]
        unique_files = file_groups.unique()
        
        # Stratification by emerg_evnt (target ==2)
        file_labels = self.target.groupby(file_groups).max()
        strat_col = (file_labels == 2).astype(int)
        
        # (train_val + test)
        files_train_val, files_test = train_test_split(
            unique_files,
            test_size=0.2,
            stratify=strat_col,
            random_state=42
        )
        
        # (train + val)
        strat_train_val = (file_labels.loc[files_train_val] == 2).astype(int)
        #files_train, files_val = train_test_split(
        #    files_train_val,
        #    test_size=0.0,
        #    stratify=strat_train_val,
        #    random_state=42
        #)
        
        # Create masks
        return (
            file_groups.isin(files_train_val),
            file_groups.isin(files_test)
        )
