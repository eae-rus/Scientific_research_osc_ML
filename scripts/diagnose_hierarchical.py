"""
Диагностика архитектуры иерархических и гибридных моделей.
Сравниваем количество параметров и время работы.
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from osc_tools.ml.models.kan import SimpleKAN, ConvKAN, PhysicsKAN
from osc_tools.ml.models.baseline import SimpleMLP, SimpleCNN
from osc_tools.ml.models.cnn import ResNet1D
from osc_tools.ml.models.advanced import (
    HierarchicalSimpleKAN, HierarchicalConvKAN, HierarchicalPhysicsKAN,
    HierarchicalCNN, HierarchicalMLP, HierarchicalResNet
)
from osc_tools.ml.models.hybrid import (
    HybridMLP, HybridCNN, HybridResNet, 
    HybridSimpleKAN, HybridConvKAN, HybridPhysicsKAN
)

def count_params(model):
    """Подсчёт количества параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def profile_model(model, x, warmup=2, runs=10):
    """Замер времени и профилирование."""
    import time
    
    # Прогрев
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    
    # Замер
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        times.append(time.perf_counter() - start)
    
    avg_ms = sum(times) / len(times) * 1000
    return avg_ms

def test_model(name, model, x, baseline_params=None, baseline_time=None):
    """Тестируем модель и выводим статистику."""
    try:
        params = count_params(model)
        time_ms = profile_model(model, x)
        
        # Проверка forward pass
        with torch.no_grad():
            out = model(x)
        
        ratio_p = f"{params / baseline_params:.2f}x" if baseline_params else "-"
        ratio_t = f"{time_ms / baseline_time:.2f}x" if baseline_time else "-"
        
        print(f"  {name:30s} | {params:>12,} ({ratio_p:>6s}) | {time_ms:>8.2f}ms ({ratio_t:>6s}) | {tuple(out.shape)}")
        return params, time_ms
    except Exception as e:
        print(f"  {name:30s} | ОШИБКА: {e}")
        return None, None

def main():
    # Типичные параметры для heavy complexity
    in_channels = 40  # phase_polar с гармониками
    seq_len = 10      # snapshot mode
    batch = 64
    num_classes = 4
    
    # Создаём тестовый вход
    x = torch.randn(batch, in_channels, seq_len)
    input_size = in_channels * seq_len  # 400
    
    print("=" * 100)
    print(f"Вход: batch={batch}, channels={in_channels}, seq_len={seq_len}, input_size(flatten)={input_size}")
    print("=" * 100)
    
    # Heavy complexity параметры
    stem_config = {'independent_layers': 3, 'grouped_layers': 3}
    
    print(f"\n{'Модель':30s} | {'Параметры':>20s} | {'Время':>20s} | Output")
    print("-" * 100)
    
    # === Baselines ===
    print(">>> BASELINES (для сравнения)")
    
    simple_kan = SimpleKAN(input_size=input_size, hidden_sizes=[256, 128, 64, 32], output_size=num_classes, grid_size=5, dropout=0.3)
    baseline_params, baseline_time = test_model("SimpleKAN", simple_kan, x)
    
    simple_mlp = SimpleMLP(input_size=input_size, hidden_sizes=[512, 256, 128, 64], output_size=num_classes, dropout=0.4)
    test_model("SimpleMLP", simple_mlp, x, baseline_params, baseline_time)
    
    simple_cnn = SimpleCNN(in_channels=in_channels, num_classes=num_classes, channels=[64, 128, 256, 512], dropout=0.4)
    test_model("SimpleCNN", simple_cnn, x, baseline_params, baseline_time)
    
    conv_kan = ConvKAN(in_channels=in_channels, num_classes=num_classes, channels=[32, 64, 128], grid_size=8, dropout=0.3)
    test_model("ConvKAN", conv_kan, x, baseline_params, baseline_time)
    
    resnet = ResNet1D(in_channels=in_channels, num_classes=num_classes, layers=[3, 4, 6, 3], base_filters=64)
    test_model("ResNet1D", resnet, x, baseline_params, baseline_time)
    
    # === Hierarchical KAN Models ===
    print("\n>>> HIERARCHICAL KAN (должны быть МЕНЬШЕ и БЫСТРЕЕ baselines)")
    
    hier_kan = HierarchicalSimpleKAN(
        in_channels=in_channels, num_classes=num_classes, 
        channels=[256, 128, 64, 32], grid_size=5, dropout=0.3,
        stem_config=stem_config, input_size=input_size
    )
    test_model("HierarchicalSimpleKAN", hier_kan, x, baseline_params, baseline_time)
    
    hier_conv_kan = HierarchicalConvKAN(
        in_channels=in_channels, num_classes=num_classes,
        channels=[32, 64, 128], grid_size=8, dropout=0.3,
        stem_config=stem_config
    )
    test_model("HierarchicalConvKAN", hier_conv_kan, x, baseline_params, baseline_time)
    
    hier_physics = HierarchicalPhysicsKAN(
        in_channels=in_channels, num_classes=num_classes,
        channels=[32, 64, 128], grid_size=8, dropout=0.3,
        stem_config=stem_config, input_size=input_size
    )
    test_model("HierarchicalPhysicsKAN", hier_physics, x, baseline_params, baseline_time)
    
    # === Hierarchical Non-KAN Models ===
    print("\n>>> HIERARCHICAL CNN/MLP (должны быть сопоставимы с baselines)")
    
    hier_cnn = HierarchicalCNN(
        in_channels=in_channels, num_classes=num_classes,
        channels=[64, 128, 256, 512], dropout=0.4,
        stem_config=stem_config
    )
    test_model("HierarchicalCNN", hier_cnn, x, baseline_params, baseline_time)
    
    hier_mlp = HierarchicalMLP(
        in_channels=in_channels, num_classes=num_classes,
        channels=[512, 256, 128, 64], dropout=0.4,
        stem_config=stem_config
    )
    test_model("HierarchicalMLP", hier_mlp, x, baseline_params, baseline_time)
    
    hier_resnet = HierarchicalResNet(
        in_channels=in_channels, num_classes=num_classes,
        layers=[3, 4, 6, 3], base_filters=64,
        stem_config=stem_config
    )
    test_model("HierarchicalResNet", hier_resnet, x, baseline_params, baseline_time)
    
    # === Hybrid Models ===
    print("\n>>> HYBRID MODELS (две ветки: raw + features)")
    
    # Для гибридных моделей: первые 8 каналов = raw, остальные 32 = features
    raw_ch = 8
    feat_ch = in_channels - raw_ch  # 32
    
    hybrid_mlp = HybridMLP(
        in_channels=in_channels, num_classes=num_classes,
        hidden_sizes=[256, 128, 64, 32], dropout=0.4,
        raw_channels=raw_ch, features_channels=feat_ch, seq_len=seq_len
    )
    test_model("HybridMLP", hybrid_mlp, x, baseline_params, baseline_time)
    
    hybrid_cnn = HybridCNN(
        in_channels=in_channels, num_classes=num_classes,
        channels=[32, 64, 128, 256], dropout=0.4,
        raw_channels=raw_ch, features_channels=feat_ch
    )
    test_model("HybridCNN", hybrid_cnn, x, baseline_params, baseline_time)
    
    hybrid_resnet = HybridResNet(
        in_channels=in_channels, num_classes=num_classes,
        layers=[3, 4, 6, 3], base_filters=32,
        raw_channels=raw_ch, features_channels=feat_ch
    )
    test_model("HybridResNet", hybrid_resnet, x, baseline_params, baseline_time)
    
    hybrid_kan = HybridSimpleKAN(
        in_channels=in_channels, num_classes=num_classes,
        hidden_sizes=[128, 64, 32, 16], grid_size=5, dropout=0.3,
        raw_channels=raw_ch, features_channels=feat_ch, seq_len=seq_len
    )
    test_model("HybridSimpleKAN", hybrid_kan, x, baseline_params, baseline_time)
    
    hybrid_conv_kan = HybridConvKAN(
        in_channels=in_channels, num_classes=num_classes,
        channels=[16, 32, 64], grid_size=8, dropout=0.3,
        raw_channels=raw_ch, features_channels=feat_ch
    )
    test_model("HybridConvKAN", hybrid_conv_kan, x, baseline_params, baseline_time)
    
    hybrid_physics = HybridPhysicsKAN(
        in_channels=in_channels, num_classes=num_classes,
        channels=[16, 32, 64], grid_size=8, dropout=0.3,
        raw_channels=raw_ch, features_channels=feat_ch
    )
    test_model("HybridPhysicsKAN", hybrid_physics, x, baseline_params, baseline_time)
    
    print("\n" + "=" * 100)
    print("ВЫВОД: Иерархические модели должны иметь:")
    print("  - Параметры: <= 1.5x от baseline (желательно меньше)")
    print("  - Время: <= 2x от baseline")
    print("=" * 100)
    
if __name__ == "__main__":
    main()
