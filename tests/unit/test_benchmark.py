import pytest
import torch
import torch.nn as nn
from osc_tools.ml.benchmark import InferenceBenchmark

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x.flatten(1))

def test_benchmark_cpu():
    model = SimpleModel()
    bench = InferenceBenchmark(device='cpu')
    
    # Input shape (Channels, Length) -> (1, 10)
    results = bench.benchmark(model, input_shape=(1, 10), num_runs=10, warmup=5)
    
    assert "avg_time_sec" in results
    assert "num_params" in results
    assert results["num_params"] == (10 * 2 + 2) # Weights + Bias
    assert results["device"] == "cpu"
    assert results["fps"] > 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_benchmark_cuda():
    model = SimpleModel()
    bench = InferenceBenchmark(device='cuda')
    
    results = bench.benchmark(model, input_shape=(1, 10), num_runs=10, warmup=5)
    
    assert "avg_time_sec" in results
    assert results["device"] == "cuda"
