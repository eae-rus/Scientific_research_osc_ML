import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
from typing import Optional

from osc_tools.ml.config import ExperimentConfig
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.models import (
    SimpleMLP, SimpleCNN, ResNet1D, 
    PDR_MLP_v2, FFT_MLP_COMPLEX_v1
)

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(config.training.save_dir) / config.training.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed
        torch.manual_seed(config.training.seed)
        
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.criterion = self._init_criterion()
        
    def _init_model(self):
        name = self.config.model.name
        params = self.config.model.params
        
        if name == 'SimpleMLP':
            model = SimpleMLP(**params)
        elif name == 'SimpleCNN':
            model = SimpleCNN(**params)
        elif name == 'ResNet1D':
            model = ResNet1D(**params)
        elif name == 'PDR_MLP_v2':
            model = PDR_MLP_v2(**params)
        elif name == 'FFT_MLP_COMPLEX_v1':
            model = FFT_MLP_COMPLEX_v1(**params)
        else:
            raise ValueError(f"Unknown model: {name}")
            
        return model.to(self.device)
    
    def _init_optimizer(self):
        return optim.Adam(
            self.model.parameters(), 
            lr=self.config.training.learning_rate, 
            weight_decay=self.config.training.weight_decay
        )
    
    def _init_criterion(self):
        mode = self.config.data.mode
        if mode == 'classification':
            return nn.CrossEntropyLoss()
        elif mode == 'reconstruction':
            return nn.MSELoss()
        elif mode == 'segmentation':
            return nn.CrossEntropyLoss() # Assuming class indices
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        print(f"Starting training on {self.device}")
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.epochs):
            start_time = time.time()
            
            # Training
            self.model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(x)
                
                # Adjust target shape if needed
                if self.config.data.mode == 'classification':
                    y = y.long()
                    if y.dim() > 1 and y.shape[1] == 1:
                        y = y.squeeze(1)
                
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        outputs = self.model(x)
                        
                        if self.config.data.mode == 'classification':
                            y = y.long()
                            if y.dim() > 1 and y.shape[1] == 1:
                                y = y.squeeze(1)
                                
                        loss = self.criterion(outputs, y)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
            else:
                avg_val_loss = 0.0
                
            # Logging
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.config.training.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_loader and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint('best_model.pt')
                
        self.save_checkpoint('final_model.pt')
        print("Training completed.")

    def save_checkpoint(self, filename):
        path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
