import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


class PQDDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int = 32,
                 stride: int = 1,
                 target_mode: bool = False
                 ):
        self.df = df
        self.window_size = window_size
        self.stride = stride
        self.target_mode = target_mode
        self.x_cols = ['IA', 'IC', 'UA BB', 'UB BB', 'UC BB']

        self.prepare_target()
        end_indices = []
        osc_ids = df.file_name.unique()
        for osc_id in tqdm(osc_ids, desc="Creating window slices"):
            indices = np.array(df.index[df.file_name == osc_id])
            indices = indices[self.window_size:: stride]
            end_indices.extend(indices)
        self.end_indices = np.array(end_indices)

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, idx):
        index = self.end_indices[idx]
        sample = self.df[self.x_cols].values[index - self.window_size: index]
        if self.target_mode:
            target = self.df.target[index - self.window_size: index].mode()[0]
        else:
            target = self.df.target.values[index]
        return sample.astype("float32"), target

    def prepare_target(self):
        self.df['target'] = 0
        #self.df['target'][self.df['opr_swch'] == 1] = 1
        self.df['target'][self.df['abnorm_evnt'] == 1] = 1
        self.df['target'][self.df['emerg_evnt'] == 1] = 2
