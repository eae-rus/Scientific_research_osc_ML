from abc import ABC
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

from common.utils import get_available_device
from datasets.base_dataset import SlidingWindowDataset


class FDDEvaluator(ABC):
    """Evaluator for TEP dataset with GBAD model."""

    def __init__(self, config, model, dataset, experiment_dir=None):
        """
        Initialize the evaluator.

        Args:
            config_path: Path to the configuration file
            experiment_dir: Directory to save experiment results
        """
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = get_available_device()
        self.model = model
        self.dataset = dataset
        self.test_loader = None
        self.results_path = None

    def setup(self):
        """Set up the model, dataset and data loaders."""

        # Load parameters
        window_size = self.config.get('data', {}).get('window_size', None)
        batch_size = self.config.get('training', {}).get('batch_size', 128)
        stride = self.config.get('data', {}).get('stride', 10)
        lr = self.config.get('training', {}).get('learning_rate', 0.001)

        # Move model to device
        self.model = self.model.to(self.device)

        # Create dataloader
        test_dataset = SlidingWindowDataset(
            df=self.dataset.df[self.dataset.test_mask],
            target=self.dataset.target[self.dataset.test_mask].astype(int),
            window_size=window_size,
            stride=stride
        )
        batch_size = self.config.get('training', {}).get('batch_size', 32)
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

    def evaluate(self):
        """Evaluate the model on the test dataset."""
        self.model.eval()
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            # Progress bar
            pbar = tqdm(self.test_loader, desc="Evaluating on test set")

            for data, target in pbar:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device).long()

                # Forward pass
                output = self.model(data)

                all_targets.extend(target.cpu().numpy())
                # Get class predictions
                _, preds = torch.max(output, 1)
                all_predictions.extend(preds.detach().cpu().numpy())

        # Convert to numpy arrays for metric calculation
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)

        # Calculate metrics and convert numpy arrays to lists
        metric = f1_score(all_targets, all_predictions, average=None).tolist()

        # Store results with serializable types
        self.results = {
            'metrics': {'f1_scores': metric}
        }

        return self.results