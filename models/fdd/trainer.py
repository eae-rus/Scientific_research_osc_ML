import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from abc import ABC
import time
import os

from common.utils import get_available_device
from datasets.fdd_dataset import FDDDataset
from datasets.base_dataset import SlidingWindowDataset


class FDDTrainer(ABC):
    """FDD trainer."""

    def __init__(self, config, model, dataset, experiment_dir=None):
        """
        Initialize the trainer.

        Args:
            config_path: Path to the configuration file
            experiment_dir: Directory to save experiment results
        """
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = get_available_device()
        self.model = model
        self.dataset = dataset
        self.train_loader = None
        self.val_loader = None
        self.criterion = None
        self.optimizer = None
        self.best_loss = float('inf')
        self.best_checkpoint_path = None

    def train(self, ):
        """Train the model for the specified number of epochs."""
        epochs = self.config.get('training', {}).get('epochs', 10)

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            _, val_loss = self._train_epoch(epoch)
            elapsed_time = time.time() - start_time

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, val_loss)

    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint."""

        self.best_checkpoint_path = os.path.join(self.experiment_dir,
                                                 "checkpoints", "best_model.pth")

        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        # Save checkpoint
        torch.save(checkpoint, self.best_checkpoint_path)

    def setup(self):
        """Set up the model, training parameters, self.dataset and data loaders."""

        # Load parameters
        window_size = self.config.get('data', {}).get('window_size', None)
        batch_size = self.config.get('training', {}).get('batch_size', 128)
        stride = self.config.get('data', {}).get('stride', 1)
        lr = self.config.get('training', {}).get('learning_rate', 0.001)

        # Move model to device
        self.model = self.model.to(self.device)

        # Set up optimizer
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        # Set up loss function
        self.criterion = nn.CrossEntropyLoss()

        # Create dataloaders
        train_dataset = SlidingWindowDataset(
            self.dataset.df[self.dataset.train_mask],
            self.dataset.target[self.dataset.train_mask],
            window_size=window_size,
            stride=stride
        )
        val_dataset = SlidingWindowDataset(
            self.dataset.df[self.dataset.val_mask],
            self.dataset.target[self.dataset.train_mask],
            window_size=window_size,
            stride=stride
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

    def _train_epoch(self, epoch):
        """Train for one epoch."""

        # train
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        self.model.train()
        total_loss = 0

        for data, target in pbar:
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            output = self.model(data)

            # Calculate loss
            loss = self.criterion(output, target)

            # Zero gradients
            self.optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()

        # Calculate average loss and metrics
        avg_loss = total_loss / len(self.train_loader)

        # validation
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Train]")
        self.model.eval()
        total_val_loss = 0

        for data, target in pbar:
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            output = self.model(data)

            # Calculate loss
            loss = self.criterion(output, target)

            # Update metrics
            total_val_loss += loss.item()

        # Calculate average loss and metrics
        avg_val_loss = total_loss / len(self.train_loader)

        return avg_loss, avg_val_loss