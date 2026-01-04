import torch
import time
import numpy as np
from typing import Tuple, Dict, Any, Optional

class InferenceBenchmark:
    """
    Класс для бенчмаркинга моделей: замер скорости инференса и подсчет параметров.
    """
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)

    def benchmark(
        self, 
        model: torch.nn.Module, 
        input_shape: Tuple[int, ...], 
        num_runs: int = 100, 
        warmup: int = 10,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Запускает бенчмарк модели.

        Args:
            model: Модель (nn.Module).
            input_shape: Форма входного тензора (без batch dimension), например (12, 640).
            num_runs: Количество прогонов для усреднения.
            warmup: Количество прогревочных прогонов.
            batch_size: Размер батча.

        Returns:
            Словарь с метриками (время, FPS, параметры).
        """
        model.to(self.device)
        model.eval()
        
        # Создаем случайный входной тензор
        x = torch.randn(batch_size, *input_shape).to(self.device)
        
        # Подсчет параметров
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Warmup (прогрев)
        try:
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(x)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
        except Exception as e:
            return {"error": f"Benchmark failed during warmup: {e}"}
        
        # Замер времени
        times = []
        try:
            with torch.no_grad():
                for _ in range(num_runs):
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    
                    _ = model(x)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append(end - start)
        except Exception as e:
            return {"error": f"Benchmark failed during run: {e}"}
                
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = batch_size / avg_time if avg_time > 0 else 0.0
        
        return {
            "device": str(self.device),
            "batch_size": batch_size,
            "input_shape": input_shape,
            "avg_time_sec": avg_time,
            "std_time_sec": std_time,
            "fps": fps,
            "latency_ms": avg_time * 1000,
            "num_params": num_params,
            "trainable_params": trainable_params
        }
