import json

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from abc import ABC
import time
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from common.utils import get_available_device
from datasets.base_dataset import SlidingWindowDataset


class FocalLoss(nn.Module):
    """
    Focal Loss для борьбы с несбалансированными классами
    Фокусируется на сложных примерах (особенно полезно для редких аварий)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Веса классов
        self.gamma = gamma  # Фокусирующий параметр
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Вычисляем cross entropy loss
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # Вычисляем probability для правильного класса
        pt = torch.exp(-ce_loss)

        # Применяем focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Focal loss
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing для предотвращения переобучения"""

    def __init__(self, classes, smoothing=0.1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.weight = weight

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        if self.weight is not None:
            true_dist = true_dist * self.weight[target].unsqueeze(1)

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class FDDTrainer(ABC):
    """FDD trainer with advanced loss functions and class balancing."""

    def __init__(self, config, model, dataset, experiment_dir=None):
        """
        Initialize the trainer.

        Args:
            config: Configuration dictionary
            model: PyTorch model
            dataset: FDD dataset
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
        self.scheduler = None
        self.best_loss = float('inf')
        self.best_checkpoint_path = None
        self.patience_counter = 0
        self.patience = config.get('training', {}).get('early_stopping_patience', 10)
        self.train_losses = []
        self.val_losses = []

    def train(self):
        """Train the model for the specified number of epochs."""
        epochs = self.config.get('training', {}).get('epochs', 10)

        print(f"Начинается обучение на {epochs} эпох...")
        print(f"Early stopping patience: {self.patience}")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_loss, val_loss = self._train_epoch(epoch)
            elapsed_time = time.time() - start_time

            # Сохраните историю
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Print epoch results with timing
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f} "
                  f"(время: {elapsed_time:.1f}s)")

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
                print(f" Новая лучшая модель сохранена! Val Loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1

            # Learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f" Learning rate: {current_lr:.6f}")

            # Early stopping check
            if self.patience and self.patience_counter >= self.patience:
                print(f" Early stopping на эпохе {epoch} (patience={self.patience})")
                break

            print("-" * 60)

        print("Обучение завершено!")
        self._save_training_history()

    def _save_training_history(self):
        """Save training history to JSON file."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
            'total_epochs': len(self.train_losses)
        }
        history_path = os.path.join(self.experiment_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f" История обучения сохранена: {history_path}")

    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint."""
        # Create checkpoints directory if it doesn't exist
        checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        # Save checkpoint
        torch.save(checkpoint, self.best_checkpoint_path)

    def setup(self):
        """Set up the model, training parameters, dataset and data loaders."""
        print(" Настройка тренировки...")

        # Load parameters
        window_size = self.config.get('data', {}).get('window_size', 32)
        batch_size = self.config.get('training', {}).get('batch_size', 128)
        stride = self.config.get('data', {}).get('stride', 1)
        val_stride = self.config.get('data', {}).get('val_stride', stride * 5)
        lr = self.config.get('training', {}).get('learning_rate', 0.001)

        # Move model to device
        self.model = self.model.to(self.device)
        print(f" Модель перенесена на устройство: {self.device}")

        # Set up optimizer
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        print(f" Оптимизатор: Adam, LR = {lr}")

        # Set up learning rate scheduler
        use_scheduler = self.config.get('training', {}).get('use_scheduler', False)
        if use_scheduler:
            step_size = self.config.get('training', {}).get('scheduler_step_size', 5)
            gamma = self.config.get('training', {}).get('scheduler_gamma', 0.8)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
            print(f" LR Scheduler: StepLR (step={step_size}, gamma={gamma})")
        else:
            self.scheduler = None

        # ===== CLASS WEIGHTS AND LOSS FUNCTION SETUP =====
        self._setup_loss_function()

        # Create dataloaders
        print(" Создание DataLoader'ов...")
        train_dataset = SlidingWindowDataset(
            self.dataset.df[self.dataset.train_mask],
            self.dataset.target[self.dataset.train_mask].astype(int),
            window_size=window_size,
            stride=stride
        )
        val_dataset = SlidingWindowDataset(
            self.dataset.df[self.dataset.val_mask],
            self.dataset.target[self.dataset.val_mask].astype(int),
            window_size=window_size,
            stride=val_stride
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        print(f" Тренировочный набор: {len(train_dataset)} окон, {len(self.train_loader)} батчей")
        print(f" Валидационный набор: {len(val_dataset)} окон, {len(self.val_loader)} батчей")
        print("=" * 60)

    def _setup_loss_function(self):
        """Setup loss function with class weights."""
        # Get loss function type from config
        loss_type = self.config.get('training', {}).get('loss_function', 'weighted_ce')

        # Analyze class distribution
        train_targets = self.dataset.target[self.dataset.train_mask].values
        unique_classes = np.unique(train_targets)

        print("=" * 60)
        print(" АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ:")
        print("=" * 60)

        # Calculate basic statistics
        total_samples = len(train_targets)
        for cls in unique_classes:
            count = np.sum(train_targets == cls)
            percentage = count / total_samples * 100
            print(f"Класс {cls}: {count:6d} примеров ({percentage:5.1f}%)")

        # Calculate class weights
        if loss_type in ['weighted_ce', 'focal', 'label_smoothing']:
            weight_strategy = self.config.get('training', {}).get('weight_strategy', 'balanced_sqrt')

            if weight_strategy == 'balanced':
                # Standard balanced weights
                class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=train_targets)
            elif weight_strategy == 'balanced_sqrt':
                # Softened balanced weights (recommended)
                class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=train_targets)
                class_weights_array = np.sqrt(class_weights_array)
            elif weight_strategy == 'manual':
                # Manual weights from config
                class_weights_array = np.array(self.config.get('training', {}).get('manual_weights', [1.0, 1.0, 1.0]))
            elif weight_strategy == 'none':
                # No weights
                class_weights_array = np.ones(len(unique_classes))
            else:
                raise ValueError(f"Unknown weight strategy: {weight_strategy}")

            class_weights_tensor = torch.FloatTensor(class_weights_array).to(self.device)

            print("=" * 60)
            print(f"  ВЕСА КЛАССОВ ({weight_strategy}):")
            print("=" * 60)
            for cls in unique_classes:
                weight = class_weights_tensor[cls].cpu().item()
                print(f"Класс {cls}: вес = {weight:.3f}")
            print("=" * 60)
        else:
            class_weights_tensor = None

        # Setup loss function
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss()
            print(" Loss функция: CrossEntropyLoss")
        elif loss_type == 'weighted_ce':
            self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            print(" Loss функция: Weighted CrossEntropyLoss")
        elif loss_type == 'focal':
            gamma = self.config.get('training', {}).get('focal_gamma', 2.0)
            self.criterion = FocalLoss(alpha=class_weights_tensor, gamma=gamma)
            print(f" Loss функция: FocalLoss (gamma={gamma})")
        elif loss_type == 'label_smoothing':
            smoothing = self.config.get('training', {}).get('label_smoothing', 0.1)
            self.criterion = LabelSmoothingLoss(classes=len(unique_classes), smoothing=smoothing,
                                                weight=class_weights_tensor)
            print(f" Loss функция: LabelSmoothingLoss (smoothing={smoothing})")
        elif loss_type == 'combined':
            # Combination of weighted CE and focal loss
            gamma = self.config.get('training', {}).get('focal_gamma', 2.0)
            self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            self.focal_criterion = FocalLoss(alpha=class_weights_tensor, gamma=gamma)
            self.criterion = None  # Will be handled in _train_epoch
            print(f" Loss функция: Combined (CE + FocalLoss, gamma={gamma})")
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

    def _train_epoch(self, epoch):
        """Train for one epoch."""
        # Training phase
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Progress bar for training
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", dynamic_ncols=True)

        for data, target in pbar:
            # Move data to device
            data, target = data.to(self.device), target.to(self.device).long()

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)

            # Calculate loss
            if self.criterion is not None:
                loss = self.criterion(output, target)
            else:
                # Combined loss
                ce_loss = self.ce_criterion(output, target)
                focal_loss = self.focal_criterion(output, target)
                loss = 0.5 * ce_loss + 0.5 * focal_loss

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Update metrics
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'Loss': f'{batch_loss:.4f}'})

        # Calculate average training loss
        avg_train_loss = total_loss / num_batches

        # Validation phase
        self.model.eval()
        total_val_loss = 0
        num_val_batches = 0

        # Progress bar for validation
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", dynamic_ncols=True)

        with torch.no_grad():
            for data, target in pbar:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device).long()

                # Forward pass
                output = self.model(data)

                # Calculate loss
                if self.criterion is not None:
                    loss = self.criterion(output, target)
                else:
                    # Combined loss
                    ce_loss = self.ce_criterion(output, target)
                    focal_loss = self.focal_criterion(output, target)
                    loss = 0.5 * ce_loss + 0.5 * focal_loss

                # Update metrics
                batch_loss = loss.item()
                total_val_loss += batch_loss
                num_val_batches += 1

                # Update progress bar
                pbar.set_postfix({'Val Loss': f'{batch_loss:.4f}'})

        # Calculate average validation loss
        avg_val_loss = total_val_loss / num_val_batches

        return avg_train_loss, avg_val_loss

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load training history if available
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
                self.val_losses = checkpoint['val_losses']

            print(f" Checkpoint загружен: {checkpoint_path}")
            return checkpoint['epoch'], checkpoint['val_loss']
        else:
            raise FileNotFoundError(f"Checkpoint не найден: {checkpoint_path}")

    def get_model_summary(self):
        """Get model summary with parameter count."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }