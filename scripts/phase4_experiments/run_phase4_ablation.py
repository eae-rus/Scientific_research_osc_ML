"""
Скрипт абляционного исследования Physical KAN-Transformer.

Два режима:
  --mode inference   : Берёт обученную модель, поочерёдно отключает физ. блоки,
                       прогоняет evaluate → таблица «блок → падение F1».
  --mode train       : Обучает модели с нуля при отключённых блоках.

CLI:
  python scripts/phase4_experiments/run_phase4_ablation.py --mode inference --checkpoint path/to/best_model.pt
  python scripts/phase4_experiments/run_phase4_ablation.py --mode train --checkpoint path/to/ssl.pt

Или запуск через константы в блоке __main__ (ручной запуск).
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch

# Добавляем корень проекта в sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.phase4_experiments.evaluate_phase4 import (
    load_model_from_checkpoint,
    prepare_val_dataloader,
    run_inference,
    compute_full_metrics,
    compute_boundary_metrics,
    run_inference_zone_level,
    PREDICTION_THRESHOLD,
)


# =====================================================================
# Сценарии абляций
# =====================================================================

# Каждый сценарий — dict с kwargs для model.set_ablation()
ABLATION_SCENARIOS = {
    'full_model': {},
    'no_kan': {'disable_kan': True},
    'no_interaction': {'disable_interaction': True},
    'no_phase_shift': {'disable_phase_shift': True},
    'no_kan_no_interaction': {'disable_kan': True, 'disable_interaction': True},
    'no_kan_no_phase_shift': {'disable_kan': True, 'disable_phase_shift': True},
    'all_disabled': {'disable_kan': True, 'disable_interaction': True, 'disable_phase_shift': True},
}


# =====================================================================
# Inference-абляция
# =====================================================================

def run_inference_ablation(
    checkpoint_path: str,
    scenarios: Optional[dict] = None,
    save_report: bool = True,
) -> dict:
    """Прогоняет inference с поочерёдным отключением блоков.

    Args:
        checkpoint_path: путь к fine-tuned чекпоинту
        scenarios: dict {name: ablation_kwargs}, или None = все стандартные
        save_report: сохранять JSON-отчёт

    Returns:
        dict {scenario_name: metrics}
    """
    if scenarios is None:
        scenarios = ABLATION_SCENARIOS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    # Загрузка модели
    model, config = load_model_from_checkpoint(ckpt_path, device)
    target_columns = config.get('target_columns', [])

    # Подготовка данных
    val_loader, target_columns = prepare_val_dataloader(config)

    results = {}
    stride_samples = config.get('downsampling_stride', 16)

    print(f"\n{'='*80}")
    print(f"АБЛЯЦИОННОЕ ИССЛЕДОВАНИЕ (inference)")
    print(f"Чекпоинт: {ckpt_path.name}")
    print(f"Сценариев: {len(scenarios)}")
    print(f"{'='*80}\n")

    for name, ablation_kwargs in scenarios.items():
        print(f"\n--- Сценарий: {name} ---")
        if ablation_kwargs:
            print(f"    Отключено: {ablation_kwargs}")
        else:
            print(f"    Полная модель (baseline)")

        # Применить абляцию
        if hasattr(model, 'set_ablation'):
            # Сначала сбросить всё
            model.set_ablation(
                disable_kan=False,
                disable_interaction=False,
                disable_phase_shift=False,
            )
            # Затем применить сценарий
            if ablation_kwargs:
                model.set_ablation(**ablation_kwargs)
        elif ablation_kwargs:
            print(f"    ⚠ Модель не поддерживает set_ablation, пропуск")
            continue

        # Window-level метрики
        preds, targets, inf_time = run_inference(model, val_loader, device)
        metrics = compute_full_metrics(preds, targets, target_columns)

        # Zone-level boundary метрики
        try:
            zone_preds, zone_targets = run_inference_zone_level(model, val_loader, device)
            boundary = compute_boundary_metrics(
                zone_preds, zone_targets, target_columns,
                stride_samples=stride_samples,
            )
            metrics['boundary'] = boundary
        except Exception as e:
            print(f"    Boundary-метрики не удались: {e}")

        metrics['inference_time'] = inf_time
        results[name] = metrics

        # Краткая печать
        print(f"    Macro-F1: {metrics['macro_f1']:.4f}  |  "
              f"AUC: {metrics.get('macro_roc_auc', 0):.4f}  |  "
              f"Time: {inf_time:.2f}s")

    # Сводная таблица
    _print_ablation_table(results, target_columns)

    # Сохранение
    if save_report:
        report_dir = ckpt_path.parent
        report = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': str(ckpt_path),
            'scenarios': {k: str(v) for k, v in scenarios.items()},
            'results': results,
        }
        report_path = report_dir / 'ablation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nОтчёт сохранён: {report_path}")

    return results


def _print_ablation_table(results: dict, target_columns: list[str]) -> None:
    """Печатает сводную таблицу абляций."""
    print(f"\n{'='*80}")
    print("СВОДНАЯ ТАБЛИЦА АБЛЯЦИЙ")
    print(f"{'='*80}")

    base_f1 = results.get('full_model', {}).get('macro_f1', 0)

    header = f"{'Сценарий':<30s} {'Macro-F1':>9s} {'Δ F1':>8s} {'AUC':>8s}"
    for col in target_columns:
        header += f"  {col[:12]:>12s}"
    print(header)
    print("-" * len(header))

    for name, metrics in results.items():
        f1 = metrics.get('macro_f1', 0)
        delta = f1 - base_f1
        auc = metrics.get('macro_roc_auc', 0)
        row = f"{name:<30s} {f1:9.4f} {delta:+8.4f} {auc:8.4f}"
        for col in target_columns:
            cls_f1 = metrics.get('per_class', {}).get(col, {}).get('f1', 0)
            row += f"  {cls_f1:12.4f}"
        print(row)

    # Boundary-метрики
    print(f"\n{'Сценарий':<30s} {'Delay(зон)':>11s} {'Smear_on':>9s} {'Smear_off':>10s}")
    print("-" * 65)
    for name, metrics in results.items():
        b = metrics.get('boundary', {})
        print(f"{name:<30s} {b.get('mean_delay_zones', 0):11.2f} "
              f"{b.get('mean_smearing_onset', 0):9.2f} "
              f"{b.get('mean_smearing_offset', 0):10.2f}")


# =====================================================================
# Train-абляция (упрощённая обёртка — запускает finetune с отключёнными блоками)
# =====================================================================

def run_train_ablation(
    ssl_checkpoint: str,
    scenarios: Optional[dict] = None,
    epochs: int = 50,
    complexity: str = 'light',
) -> None:
    """Запускает finetune с нуля для каждого сценария абляции.

    Каждый сценарий запускается через CLI-подпроцесс finetune с
    дополнительными аргументами абляции.
    """
    import subprocess

    if scenarios is None:
        scenarios = {
            'full_model': [],
            'no_kan': ['--ablate-kan'],
            'no_interaction': ['--ablate-interaction'],
            'no_phase_shift': ['--ablate-phase-shift'],
        }

    print(f"\n{'='*80}")
    print(f"АБЛЯЦИОННОЕ ОБУЧЕНИЕ")
    print(f"SSL чекпоинт: {ssl_checkpoint}")
    print(f"Эпох: {epochs}, Сложность: {complexity}")
    print(f"Сценариев: {len(scenarios)}")
    print(f"{'='*80}\n")

    for name, extra_args in scenarios.items():
        print(f"\n{'='*40}")
        print(f"Сценарий: {name}")
        print(f"{'='*40}")

        run_name = f"ablation_{name}"
        cmd = [
            sys.executable,
            str(_ROOT / 'scripts' / 'phase4_experiments' / 'run_phase4_finetune.py'),
            '--checkpoint', ssl_checkpoint,
            '--complexity', complexity,
            '--epochs', str(epochs),
        ] + extra_args + [f'--run-name={run_name}']

        print(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  ⚠ Сценарий {name} завершился с ошибкой (code={result.returncode})")


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 4: Ablation Study')
    parser.add_argument('--mode', choices=['inference', 'train'],
                        default='inference', help='Режим абляции')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Путь к чекпоинту (fine-tuned для inference, SSL для train)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Число эпох для train-абляции')
    parser.add_argument('--complexity', choices=['light', 'medium', 'heavy'],
                        default='light', help='Сложность модели')
    parser.add_argument('--scenarios', type=str, nargs='*', default=None,
                        help='Список сценариев (если не указано — все стандартные)')
    return parser.parse_args()


# =====================================================================
# Точка входа
# =====================================================================

if __name__ == '__main__':
    # =================================================================
    # РЕЖИМ РУЧНОГО ЗАПУСКА ЧЕРЕЗ КОНСТАНТЫ
    # Раскомментируйте нужный блок и запустите файл напрямую.
    # =================================================================

    # --- Режим: 'inference' | 'train' ---
    MODE = 'inference'

    # --- Чекпоинт (fine-tuned для inference, SSL для train) ---
    CHECKPOINT = 'experiments/phase4/finetune_PhysicalKANTransformer_20260328_230623/best_model.pt'

    # --- Параметры train-абляции ---
    SSL_CHECKPOINT = 'experiments/phase4/pretrain_PhysicalKANTransformer_20260327_155618/best_model.pt'
    TRAIN_EPOCHS = 50
    TRAIN_COMPLEXITY = 'light'

    # --- Выбор сценариев (None = все стандартные) ---
    # SELECTED_SCENARIOS = None  # все
    SELECTED_SCENARIOS = {
        'full_model': {},
        'no_kan': {'disable_kan': True},
        'no_interaction': {'disable_interaction': True},
        'no_phase_shift': {'disable_phase_shift': True},
        'all_disabled': {'disable_kan': True, 'disable_interaction': True, 'disable_phase_shift': True},
    }

    # --- Запуск ---
    if len(sys.argv) > 1:
        # CLI-режим
        args = parse_args()
        if args.mode == 'inference':
            if args.scenarios:
                # Фильтруем стандартные сценарии
                sc = {k: v for k, v in ABLATION_SCENARIOS.items() if k in args.scenarios}
            else:
                sc = None
            run_inference_ablation(args.checkpoint, scenarios=sc)
        elif args.mode == 'train':
            run_train_ablation(args.checkpoint, epochs=args.epochs, complexity=args.complexity)
    else:
        # Ручной режим
        if MODE == 'inference':
            run_inference_ablation(CHECKPOINT, scenarios=SELECTED_SCENARIOS)
        elif MODE == 'train':
            run_train_ablation(
                SSL_CHECKPOINT,
                epochs=TRAIN_EPOCHS,
                complexity=TRAIN_COMPLEXITY,
            )
