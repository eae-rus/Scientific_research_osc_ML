import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
from typing import Optional
from tqdm import tqdm
from dataclasses import asdict

from osc_tools.ml.config import ExperimentConfig
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.models import (
    SimpleMLP, SimpleCNN, ResNet1D, 
    PDR_MLP_v2, FFT_MLP_COMPLEX_v1,
    SimpleKAN, ConvKAN, PhysicsKAN, AutoEncoder
)
from osc_tools.ml.class_weights import compute_pos_weight_from_loader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score
import numpy as np

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(config.training.save_dir) / config.training.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed
        torch.manual_seed(config.training.seed)
        
        self._save_config()
        
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        # Критерий отложим до наличия train_loader (нужны статистики меток для pos_weight)
        self.criterion = None
        
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
        elif name == 'SimpleKAN':
            model = SimpleKAN(**params)
        elif name == 'ConvKAN':
            model = ConvKAN(**params)
        elif name == 'PhysicsKAN':
            model = PhysicsKAN(**params)
        elif name == 'AutoEncoder':
            model = AutoEncoder(**params)
        else:
            raise ValueError(f"Unknown model: {name}")
            
        return model.to(self.device)
    
    def _init_optimizer(self):
        return optim.Adam(
            self.model.parameters(), 
            lr=self.config.training.learning_rate, 
            weight_decay=self.config.training.weight_decay
        )
    
    def _init_criterion(self, train_loader: Optional[DataLoader] = None):
        """
        Инициализация критерия. Для режима `multilabel` при наличии флага
        `config.training.use_pos_weight=True` вычисляем `pos_weight` из `train_loader`.
        """
        mode = self.config.data.mode
        if mode == 'classification':
            return nn.CrossEntropyLoss()
        elif mode == 'multilabel':
            # Поддержка pos_weight
            use_pw = getattr(self.config.training, 'use_pos_weight', False) if hasattr(self.config, 'training') else False
            if use_pw:
                if train_loader is None:
                    raise ValueError('train_loader должен быть предоставлен для вычисления pos_weight.')
                pw = compute_pos_weight_from_loader(train_loader, device=self.device)
                return nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                return nn.BCEWithLogitsLoss()
        elif mode == 'reconstruction':
            return nn.MSELoss()
        elif mode == 'segmentation':
            return nn.CrossEntropyLoss() # Assuming class indices
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        # Инициализируем критерий здесь, т.к. для pos_weight нужен train_loader
        if self.criterion is None:
            self.criterion = self._init_criterion(train_loader)

        print(f"Запуск обучения на {self.device}")
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        
        for epoch in range(self.config.training.epochs):
            start_time = time.time()
            
            # Training
            self.model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{self.config.training.epochs} [Train]")
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(x)
                
                # Adjust target shape if needed
                if self.config.data.mode == 'classification':
                    y = y.long()
                    if y.dim() > 1 and y.shape[1] == 1:
                        y = y.squeeze(1)
                elif self.config.data.mode == 'multilabel':
                    y = y.float()
                
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            all_preds = []
            all_targets = []
            
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Эпоха {epoch+1}/{self.config.training.epochs} [Val]", leave=False)
                    for x, y in val_pbar:
                        x, y = x.to(self.device), y.to(self.device)
                        outputs = self.model(x)
                        
                        if self.config.data.mode == 'classification':
                            y = y.long()
                            if y.dim() > 1 and y.shape[1] == 1:
                                y = y.squeeze(1)
                            
                            _, predicted = torch.max(outputs.data, 1)
                            val_total += y.size(0)
                            val_correct += (predicted == y).sum().item()
                            
                            all_preds.extend(predicted.cpu().numpy())
                            all_targets.extend(y.cpu().numpy())
                            
                        elif self.config.data.mode == 'multilabel':
                            y = y.float()
                            # сигмоид и порог 0.5
                            probs = torch.sigmoid(outputs)
                            predicted = (probs > 0.5).float()
                            
                            all_preds.extend(predicted.cpu().numpy())
                            all_targets.extend(y.cpu().numpy())
                                
                        loss = self.criterion(outputs, y)
                        val_loss += loss.item()
                        val_pbar.set_postfix({'loss': loss.item()})
                
                avg_val_loss = val_loss / len(val_loader)
                
                # рассчет метрик
                all_preds = np.array(all_preds)
                all_targets = np.array(all_targets)
                
                if self.config.data.mode == 'classification':
                    val_acc = accuracy_score(all_targets, all_preds)
                    val_f1 = f1_score(all_targets, all_preds, average='macro')
                    val_balanced_acc = balanced_accuracy_score(all_targets, all_preds)
                    per_class_f1 = f1_score(all_targets, all_preds, average=None).tolist()
                elif self.config.data.mode == 'multilabel':
                    # Для multilabel, accuracy - это точное совпадение (строгое)
                    # Используйте F1-macro или F1-weighted
                    val_acc = accuracy_score(all_targets, all_preds) # Exact match
                    val_f1 = f1_score(all_targets, all_preds, average='macro')
                    val_balanced_acc = 0.0 # Не определено для multilabel
                    per_class_f1 = f1_score(all_targets, all_preds, average=None).tolist()
                    
            else:
                avg_val_loss = 0.0
                val_acc = 0.0
                val_f1 = 0.0
                val_balanced_acc = 0.0
                per_class_f1 = []
            
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
                
            # Logging
            epoch_time = time.time() - start_time
            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_balanced_acc": val_balanced_acc,
                "per_class_f1": per_class_f1,
                "time": epoch_time
            }
            self._save_metrics(metrics)
            
            print(f"Эпоха {epoch+1}/{self.config.training.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Сохраняем лучшую модель
            if val_loader and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint('best_model.pt')
            
            # Периодический чекпоинт (каждые N эпох)
            if (epoch + 1) % self.config.training.checkpoint_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
                
        self.save_checkpoint('final_model.pt')
        print("Обучение завершено.")
        return history

    def _save_metrics(self, metrics):
        metrics_file = self.save_dir / "metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def _save_config(self):
        config_file = self.save_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(asdict(self.config), f, indent=4)

    def save_checkpoint(self, filename):
        path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
