from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm


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


class SlidingWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target: pd.Series, window_size: int, stride: int = 1):
        # Convert data and targets to numpy 
        self.data = df.values.astype('float32')
        self.target = target.values.astype('float32') if target is not None else None
        
        # Save multindex for safety checks
        self.index = df.index
        self.window_size = window_size
        
        # Calculate valid windows for each run_id
        self.valid_windows = self._precompute_valid_windows(stride)

    def _precompute_valid_windows(self, stride):
        valid_windows = []
        run_ids = self.index.get_level_values(0).unique()

        for run_id in tqdm(run_ids, desc="Building safe windows"):
            # Get all indices for running run_id
            run_mask = self.index.get_level_values(0) == run_id
            run_indices = np.where(run_mask)[0]
        
            # Check run_id has enough points
            if len(run_indices) < self.window_size:
                continue
                
            # Generate end window indices ONLY in the borders of this run_id
            for end_pos in range(self.window_size, len(run_indices), stride):
                start_pos = end_pos - self.window_size
                start_idx = run_indices[start_pos]
                end_idx = run_indices[end_pos]
                
                # Extra check
                assert self.index[start_idx][0] == self.index[end_idx-1][0], "Window crosses run_id boundary!"
                
                valid_windows.append((start_idx, end_idx))
                
        return valid_windows

    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        start_idx, end_idx = self.valid_windows[idx]
        sample = self.data[start_idx:end_idx]
        target = self.target[end_idx]
        return sample, target
