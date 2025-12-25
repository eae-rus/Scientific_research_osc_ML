import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
from unittest.mock import MagicMock

# Р”РѕР±Р°РІР»СЏРµРј РєРѕСЂРµРЅСЊ РїСЂРѕРµРєС‚Р° РІ sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ML_model.train_PDR import CustomDataset as CustomDatasetPDR, MultiFileRandomPointSampler
from ML_model.train import CustomDataset as CustomDatasetTrain, FastBalancedBatchSampler
from osc_tools.core.constants import PDRFeatures, Features

class TestMLInfrastructure:
    """РўРµСЃС‚С‹ РґР»СЏ ML РёРЅС„СЂР°СЃС‚СЂСѓРєС‚СѓСЂС‹ (Dataset, Sampler)."""

    def test_custom_dataset_pdr(self):
        """РўРµСЃС‚ CustomDataset РёР· train_PDR.py."""
        # РЎРѕР·РґР°РµРј С„РёРєС‚РёРІРЅС‹Рµ РґР°РЅРЅС‹Рµ
        cols = PDRFeatures.ALL_MODEL_1
        data = pd.DataFrame(np.random.rand(100, len(cols)), columns=cols)
        data['target'] = np.random.rand(100)
        
        indexes = pd.DataFrame(index=[0, 10, 20, 30])
        
        dataset = CustomDatasetPDR(name_target='target', dt=data, indexes=indexes, frame_size=5)
        
        assert len(dataset) == 4
        x, y = dataset[0]
        assert x.shape == (5, len(cols))
        assert isinstance(y, torch.Tensor)

    def test_custom_dataset_train(self):
        """РўРµСЃС‚ CustomDataset РёР· train.py."""
        # РЎРѕР·РґР°РµРј С„РёРєС‚РёРІРЅС‹Рµ РґР°РЅРЅС‹Рµ
        cols = Features.ALL
        data = pd.DataFrame(np.random.rand(100, len(cols)), columns=cols)
        data[Features.TARGET] = np.random.randint(0, 2, (100, 3))
        
        indexes = pd.DataFrame(index=[0, 10, 20, 30])
        
        dataset = CustomDatasetTrain(dt=data, indexes=indexes, frame_size=5)
        
        assert len(dataset) == 4
        x, y = dataset[0]
        assert x.shape == (5, len(cols))
        assert isinstance(y, torch.Tensor)

    def test_multi_file_random_point_sampler(self):
        """РўРµСЃС‚ MultiFileRandomPointSampler РёР· train_PDR.py."""
        all_files = ['file1', 'file2', 'file3']
        file_map = {
            'file1': [0, 1, 2, 3, 4],
            'file2': [0, 1, 2, 3, 4],
            'file3': [0, 1, 2, 3, 4]
        }
        
        sampler = MultiFileRandomPointSampler(
            all_files=all_files,
            file_to_valid_start_indices_map=file_map,
            num_batches_per_epoch=2,
            num_files_per_batch=2,
            num_samples_per_file=2
        )
        
        assert len(sampler) == 2 # num_batches_per_epoch
        
        indices = list(sampler)
        assert len(indices) == 2 # 2 batches
        assert len(indices[0]) == 4 # 2 files * 2 samples

    def test_fast_balanced_batch_sampler(self):
        """РўРµСЃС‚ FastBalancedBatchSampler РёР· train.py."""
        datasets_by_class = {
            'class0': [0, 1, 2, 3, 4, 5],
            'class1': [10, 11, 12, 13, 14, 15]
        }
        
        sampler = FastBalancedBatchSampler(
            datasets_by_class=datasets_by_class,
            batch_size_per_class=2,
            num_batches=3
        )
        
        assert len(sampler) == 3
        
        batches = list(sampler)
        assert len(batches) == 3
        for batch in batches:
            assert len(batch) == 4 # 2 classes * 2 samples
            # РџСЂРѕРІРµСЂСЏРµРј, С‡С‚Рѕ РІ Р±Р°С‚С‡Рµ РµСЃС‚СЊ РёРЅРґРµРєСЃС‹ РёР· РѕР±РѕРёС… РєР»Р°СЃСЃРѕРІ
            c0 = [i for i in batch if i in datasets_by_class['class0']]
            c1 = [i for i in batch if i in datasets_by_class['class1']]
            assert len(c0) == 2
            assert len(c1) == 2
