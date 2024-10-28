import pandas as pd
import numpy as np
from pqd_dataset import PQDDataset
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader
from pqd_trainer import train_model, evaluate_model
from models.conv_mlp import GRU
import random
import os


def seed_everything(seed: int = 42):
    """
    This function is used to maintain repeatability
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
seed_everything(seed)

window_size = 32
stride = 1
target_mode = True
epochs=100
batch_size=32
lr = 0.0005

train_df = pd.read_csv('train.csv')
train_dataset = PQDDataset(
                            df=train_df,
                            window_size=window_size,
                            stride=stride,
                            target_mode=target_mode
                            )

test_df = pd.read_csv('test.csv')
test_dataset = PQDDataset(
                            df=test_df,
                            window_size=window_size,
                            stride=stride,
                            target_mode=target_mode
                            )

model = GRU()

print(sum(p.numel() for p in model.parameters()))
train_model(
            model=model,
            dataset=train_dataset,
            run_seed=seed,
            eval_dataset=test_dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weights=None)