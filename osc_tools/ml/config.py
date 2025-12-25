from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataConfig:
    path: str
    window_size: int
    batch_size: int
    num_workers: int = 0
    features: Optional[List[str]] = None
    target: Optional[str] = None
    mode: str = 'classification' # classification, segmentation, reconstruction

@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    weight_decay: float = 0.0
    device: str = 'cuda'
    save_dir: str = 'experiments'
    experiment_name: str = 'default'
    seed: int = 42

@dataclass
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
