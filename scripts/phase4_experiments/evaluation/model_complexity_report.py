"""Отчёт сложности модели для сравнения в статье.

Считает для чекпоинта:
- trainable/total params;
- размер чекпоинта на диске;
- latency batch=1;
- throughput для заданного batch;
- peak CUDA memory (если доступна CUDA).

Пример:
    python scripts/phase4_experiments/evaluation/model_complexity_report.py \
        --checkpoint experiments/phase4/.../best_model.pt \
        --batch-size 256
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase4_experiments.evaluate_phase4 import load_model_from_checkpoint


def _resolve_checkpoint(path: str) -> Path:
    ckpt = Path(path)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt}")
    return ckpt


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Считает параметры модели."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': int(total), 'trainable_params': int(trainable)}


def benchmark_model(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    batch_size: int = 256,
    warmup: int = 10,
    repeats: int = 50,
) -> dict[str, float]:
    """Измеряет latency/throughput на синтетическом входе правильной формы."""
    channels = int(config.get('num_input_channels', 220))
    seq_len = int(config.get('max_eval_seq_len', 72))
    num_classes = int(config.get('num_classes', 4))

    x1 = torch.randn(1, channels, seq_len, device=device)
    xb = torch.randn(batch_size, channels, seq_len, device=device)

    model.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x1, mode='classify')
            _ = model(xb, mode='classify')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(x1, mode='classify')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        for _ in range(repeats):
            _ = model(xb, mode='classify')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t3 = time.perf_counter()

    latency_ms_batch1 = (t1 - t0) / repeats * 1000
    latency_ms_batch_n = (t3 - t2) / repeats * 1000
    throughput = batch_size / ((t3 - t2) / repeats)
    peak_vram_mb = 0.0
    if device.type == 'cuda':
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        'input_channels': channels,
        'seq_len': seq_len,
        'num_classes': num_classes,
        'latency_ms_batch1': float(latency_ms_batch1),
        'latency_ms_batchN': float(latency_ms_batch_n),
        'batch_size': int(batch_size),
        'throughput_samples_per_s': float(throughput),
        'peak_vram_mb': float(peak_vram_mb),
    }


def build_report(
    checkpoint: str,
    batch_size: int = 256,
    warmup: int = 10,
    repeats: int = 50,
    output: str | None = None,
) -> dict:
    """Формирует и сохраняет JSON-отчёт сложности."""
    ckpt_path = _resolve_checkpoint(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_model_from_checkpoint(ckpt_path, device)

    params = count_parameters(model)
    bench = benchmark_model(
        model=model,
        config=config,
        device=device,
        batch_size=batch_size,
        warmup=warmup,
        repeats=repeats,
    )

    report = {
        'checkpoint': str(ckpt_path),
        'model_type': config.get('model_type'),
        'd_model': config.get('d_model'),
        'num_layers': config.get('num_layers'),
        'num_heads': config.get('num_heads'),
        'cls_head_type': config.get('cls_head_type'),
        'ffn_type': config.get('ffn_type'),
        'checkpoint_size_mb': ckpt_path.stat().st_size / (1024 ** 2),
        **params,
        **bench,
    }

    if output is None:
        out_dir = PROJECT_ROOT / 'reports' / 'phase4' / 'model_complexity'
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f'{ckpt_path.parent.name}_{ckpt_path.stem}_complexity.json'
    else:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nОтчёт сохранён: {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Model complexity report')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    build_report(
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
        output=args.output,
    )


if __name__ == '__main__':
    main()
