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
    HierarchicalCNN, HierarchicalConvKAN, HierarchicalMLP,
    HierarchicalResNet, HierarchicalSimpleKAN, HierarchicalPhysicsKAN,
    HybridMLP, HybridCNN, HybridResNet,
    HybridSimpleKAN, HybridConvKAN, HybridPhysicsKAN,
    PDR_MLP_v2, FFT_MLP_COMPLEX_v1,
    SimpleKAN, ConvKAN, PhysicsKAN, PhysicsKANConditional, AutoEncoder
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
        
        self.model = self._init_model()
        self._save_config()
        
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
        elif name == 'HierarchicalCNN':
            model = HierarchicalCNN(**params)
        elif name == 'HierarchicalConvKAN':
            model = HierarchicalConvKAN(**params)
        elif name == 'HierarchicalMLP':
            model = HierarchicalMLP(**params)
        elif name == 'HierarchicalResNet':
            model = HierarchicalResNet(**params)
        elif name == 'HierarchicalSimpleKAN':
            model = HierarchicalSimpleKAN(**params)
        elif name == 'HierarchicalPhysicsKAN':
            model = HierarchicalPhysicsKAN(**params)
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
        elif name == 'PhysicsKANConditional':
            model = PhysicsKANConditional(**params)
        elif name == 'AutoEncoder':
            model = AutoEncoder(**params)
        # Гибридные модели (Exp 2.6.3)
        elif name == 'HybridMLP':
            model = HybridMLP(**params)
        elif name == 'HybridCNN':
            model = HybridCNN(**params)
        elif name == 'HybridResNet':
            model = HybridResNet(**params)
        elif name == 'HybridSimpleKAN':
            model = HybridSimpleKAN(**params)
        elif name == 'HybridConvKAN':
            model = HybridConvKAN(**params)
        elif name == 'HybridPhysicsKAN':
            model = HybridPhysicsKAN(**params)
        else:
            raise ValueError(f"Unknown model: {name}")
            
        # Подсчет параметров
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
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
        elif mode == 'multitask_conditional':
            # Для условных голов: BCE по 4 бинарным классам (Normal, ML_1, ML_2, ML_3)
            use_pw = getattr(self.config.training, 'use_pos_weight', False) if hasattr(self.config, 'training') else False
            if use_pw:
                if train_loader is None:
                    raise ValueError('train_loader должен быть предоставлен для вычисления pos_weight.')
                pos_weight = self._compute_pos_weight_multitask(train_loader)
                self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.bce_loss = nn.BCEWithLogitsLoss()
            return None
        elif mode == 'reconstruction':
            return nn.MSELoss()
        elif mode == 'segmentation':
            return nn.CrossEntropyLoss() # Assuming class indices
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _compute_pos_weight_multitask(self, train_loader: DataLoader) -> torch.Tensor:
        """Вычисляет pos_weight для 4 бинарных таргетов (Normal, ML_1, ML_2, ML_3)."""
        total_pos = torch.zeros(4, device=self.device)
        total_count = 0
        for _, y in train_loader:
            y = y.to(self.device).float()
            total_pos += y[:, :4].sum(dim=0)
            total_count += y.shape[0]
        total_neg = total_count - total_pos
        pos_weight = total_neg / (total_pos + 1e-6)
        return pos_weight

    def _compute_multitask_conditional_loss(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Лосс для режима multitask_conditional (BCE по 4 бинарным таргетам)."""
        y = y.float()
        out_binary = outputs[:, :4]
        return self.bce_loss(out_binary, y[:, :4])

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
            pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{self.config.training.epochs} [Train]", leave=False)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(x)
                
                # Adjust target shape if needed
                if self.config.data.mode == 'classification':
                    y = y.long()
                    if y.dim() > 1 and y.shape[1] == 1:
                        y = y.squeeze(1)
                    loss = self.criterion(outputs, y)
                elif self.config.data.mode == 'multilabel':
                    y = y.float()
                    loss = self.criterion(outputs, y)
                elif self.config.data.mode == 'multitask_conditional':
                    loss = self._compute_multitask_conditional_loss(outputs, y)
                else:
                    raise ValueError(f"Unknown mode: {self.config.data.mode}")
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_loss = 0.0
            
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
                            
                            all_preds.extend(predicted.cpu().numpy())
                            all_targets.extend(y.cpu().numpy())
                            loss = self.criterion(outputs, y)
                            
                        elif self.config.data.mode == 'multilabel':
                            y = y.float()
                            # сигмоид и порог 0.5
                            probs = torch.sigmoid(outputs)
                            predicted = (probs > 0.5).float()
                            
                            all_preds.extend(predicted.cpu().numpy())
                            all_targets.extend(y.cpu().numpy())
                            loss = self.criterion(outputs, y)
                                
                        elif self.config.data.mode == 'multitask_conditional':
                            y = y.float()

                            probs = torch.sigmoid(outputs[:, :4])
                            pred_binary = (probs > 0.5).int().cpu().numpy()

                            # Ограничение: если ML_3=1, то ML_2=0
                            pred_binary[:, 2] = np.where(pred_binary[:, 3] == 1, 0, pred_binary[:, 2])

                            all_preds.extend(pred_binary)
                            all_targets.extend(y[:, :4].cpu().numpy())

                            loss = self._compute_multitask_conditional_loss(outputs, y)
                        else:
                            raise ValueError(f"Unknown mode: {self.config.data.mode}")
                        val_loss += loss.item()
                        val_pbar.set_postfix({'loss': loss.item()})
                
                avg_val_loss = val_loss / len(val_loader)
                
                # рассчет метрик
                all_preds = np.array(all_preds)
                all_targets = np.array(all_targets)
                
                if self.config.data.mode == 'classification':
                    val_acc = accuracy_score(all_targets, all_preds)
                    val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
                    val_balanced_acc = balanced_accuracy_score(all_targets, all_preds)
                    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0).tolist()
                elif self.config.data.mode == 'multilabel':
                    # Для multilabel, accuracy - это точное совпадение (строгое)
                    # Используйте F1-macro или F1-weighted
                    val_acc = accuracy_score(all_targets, all_preds) # Exact match
                    val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
                    val_balanced_acc = 0.0 # Не определено для multilabel
                    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0).tolist()
                elif self.config.data.mode == 'multitask_conditional':
                    # Отдельные метрики по головам (4 бинарных)
                    y_true = np.array(all_targets)
                    y_pred = np.array(all_preds)

                    f1_normal = f1_score(y_true[:, 0], y_pred[:, 0], zero_division=0)
                    f1_ml1 = f1_score(y_true[:, 1], y_pred[:, 1], zero_division=0)
                    f1_ml2 = f1_score(y_true[:, 2], y_pred[:, 2], zero_division=0)
                    f1_ml3 = f1_score(y_true[:, 3], y_pred[:, 3], zero_division=0)

                    acc_normal = accuracy_score(y_true[:, 0], y_pred[:, 0])
                    acc_ml1 = accuracy_score(y_true[:, 1], y_pred[:, 1])
                    acc_ml2 = accuracy_score(y_true[:, 2], y_pred[:, 2])
                    acc_ml3 = accuracy_score(y_true[:, 3], y_pred[:, 3])

                    val_acc = float(np.mean([acc_normal, acc_ml1, acc_ml2, acc_ml3]))
                    val_f1 = float(np.mean([f1_normal, f1_ml1, f1_ml2, f1_ml3]))
                    val_balanced_acc = 0.0
                    per_class_f1 = [float(f1_normal), float(f1_ml1), float(f1_ml2), float(f1_ml3)]
                else:
                    val_acc = 0.0
                    val_f1 = 0.0
                    val_balanced_acc = 0.0
                    per_class_f1 = []
                    
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
                "epoch_time": epoch_time
            }
            self._save_metrics(metrics)
            
            print(f"Эпоха {epoch+1}/{self.config.training.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f}")
            
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
        config_dict = asdict(self.config)
        # Добавляем инфо о параметрах, чтобы не считать потом
        config_dict['model_info'] = {
            'num_params': self.num_params,
            'device': str(self.device)
        }
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=4)

    def save_checkpoint(self, filename):
        path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
