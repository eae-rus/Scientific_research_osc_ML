import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from osc_tools.data_management.dataset_manager import DatasetManager
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.labels import get_target_columns, prepare_labels_for_experiment
from osc_tools.ml.kan_pruning import (
    collect_kan_inputs,
    calculate_kan_importance,
    apply_pruning_mask,
    compute_importance_stats
)

# Переиспользуем проверенные функции из evaluation
from scripts.evaluation.aggregate_reports import (
    _create_model_from_config,
    _load_state_dict_safe,
    _get_eval_batch_size,
    evaluate_model_full
)
from scripts.evaluation.plot_model_marking import (
    _find_experiment_dir,
    _resolve_feature_mode,
    _resolve_sampling_strategy,
    _resolve_num_harmonics,
    _resolve_target_level,
    json_load
)


def _build_dataset(
    config: Dict[str, Any],
    exp_name: str,
    data_dir: Path,
    max_windows: int,
    eval_stride: int
) -> Tuple[OscillogramDataset, List[int], List[str]]:
    """Создание тестового датасета под текущую конфигурацию."""
    window_size = config.get('data', {}).get('window_size', 320)
    sampling_rate = config.get('data', {}).get('sampling_rate', 1600)
    input_size = config.get('model', {}).get('params', {}).get('input_size', 0)
    in_channels = config.get('model', {}).get('params', {}).get('in_channels')

    feature_mode = _resolve_feature_mode(config, exp_name)
    sampling_strategy, downsampling_stride = _resolve_sampling_strategy(
        exp_name, input_size, in_channels, window_size
    )

    num_harmonics, in_channels = _resolve_num_harmonics(
        feature_mode, sampling_strategy, downsampling_stride, window_size, in_channels, input_size
    )

    target_level = _resolve_target_level(exp_name)

    dm = DatasetManager(str(data_dir))
    test_df = dm.load_test_df(precomputed=False)
    test_df = test_df.with_row_index("row_nr")
    test_df = prepare_labels_for_experiment(test_df, target_level)

    target_cols = get_target_columns(target_level, test_df)

    indices = OscillogramDataset.create_indices(
        test_df,
        window_size=window_size,
        mode='test',
        stride=eval_stride
    )

    # max_windows < 0 означает использовать все окна
    if max_windows > 0 and len(indices) > max_windows:
        indices = indices[:max_windows]
    
    print(f"       Датасет: {len(indices)} окон (stride={eval_stride})")

    ds = OscillogramDataset(
        dataframe=test_df,
        indices=indices,
        window_size=window_size,
        mode='classification',
        feature_mode=feature_mode,
        sampling_rate=sampling_rate,
        target_columns=target_cols,
        target_level=target_level if target_level != 'base' else 'base_labels',
        physical_normalization=True,
        norm_coef_path=config.get('data', {}).get('norm_coef_path'),
        num_harmonics=num_harmonics,
        sampling_strategy=sampling_strategy,
        downsampling_stride=downsampling_stride,
        augment=False
    )

    return ds, indices, target_cols


class PhysicsKANAblation(torch.nn.Module):
    """
    Обёртка для оценки вклада мультипликативных/делительных признаков.
    """
    def __init__(self, base_model: torch.nn.Module, scale_raw: float, scale_mult: float, scale_div: float):
        super().__init__()
        self.base = base_model
        self.scale_raw = scale_raw
        self.scale_mult = scale_mult
        self.scale_div = scale_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Повторяем логику PhysicsKAN.forward, но с возможностью отключения частей
        s = self.base.mult(x)
        s = self.base.bn_mult(s) * self.scale_mult

        z = self.base.div(x)
        z = self.base.bn_div(z) * self.scale_div

        x_scaled = x * self.scale_raw
        x_combined = torch.cat([x_scaled, s, z], dim=1)
        return self.base.processing_net(x_combined)


def _evaluate_model(
    model: torch.nn.Module,
    ds: OscillogramDataset,
    device: torch.device,
    batch_size: int,
    mode: str
) -> Dict[str, float]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return evaluate_model_full(model, loader, device, mode=mode)


def _compute_arithmetic_stats(
    model: torch.nn.Module,
    ds: OscillogramDataset,
    device: torch.device,
    batch_size: int,
    max_batches: int = 5
) -> Dict[str, float]:
    """Оценка средней мощности сигналов (|x|, |x*y|, |x/y|)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    sums = {
        'raw_abs': 0.0,
        'mult_abs': 0.0,
        'div_abs': 0.0,
        'count': 0
    }

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            s = model.mult(x)
            z = model.div(x)

            sums['raw_abs'] += float(torch.mean(torch.abs(x)).item())
            sums['mult_abs'] += float(torch.mean(torch.abs(s)).item())
            sums['div_abs'] += float(torch.mean(torch.abs(z)).item())
            sums['count'] += 1

            if i + 1 >= max_batches:
                break

    count = max(1, sums['count'])
    return {
        'raw_abs_mean': sums['raw_abs'] / count,
        'mult_abs_mean': sums['mult_abs'] / count,
        'div_abs_mean': sums['div_abs'] / count
    }


def run_pruning_experiment(
    exp_name: str,
    data_dir: Path,
    output_dir: Path,
    thresholds: List[float],
    max_importance_windows: int,
    max_eval_windows: int,
    eval_stride: int
) -> None:
    exp_dir = _find_experiment_dir(exp_name)
    config = json_load(exp_dir / 'config.json')

    model = _create_model_from_config(config)
    if model is None:
        raise ValueError("Не удалось создать модель из config.json")

    ckpt_path = exp_dir / 'best_model.pt'
    if not ckpt_path.exists():
        ckpt_path = exp_dir / 'final_model.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Не найден чекпоинт модели (best/final) в {exp_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    _load_state_dict_safe(model, checkpoint, exp_dir.name, 'pruning')
    model = model.to(device)
    model.eval()

    # Датасеты для importance/eval (с разным числом окон)
    ds_importance, _, _ = _build_dataset(
        config=config,
        exp_name=exp_name,
        data_dir=data_dir,
        max_windows=max_importance_windows,
        eval_stride=eval_stride
    )

    ds_eval, _, _ = _build_dataset(
        config=config,
        exp_name=exp_name,
        data_dir=data_dir,
        max_windows=max_eval_windows,
        eval_stride=eval_stride
    )

    mode = config.get('data', {}).get('mode', 'multilabel')
    eval_batch_size = _get_eval_batch_size(config, model_name=config.get('model', {}).get('name'))

    # Базовая оценка
    print(f"[1/5] Базовая оценка модели на {len(ds_eval)} окнах...")
    base_metrics = _evaluate_model(model, ds_eval, device, eval_batch_size, mode)
    print(f"       Базовые метрики: Acc={base_metrics.get('acc', 0):.4f}, F1={base_metrics.get('f1', 0):.4f}")

    # Сбор входов и важностей
    print(f"[2/5] Сбор входов KANLinear и расчёт важностей...")
    importance_loader = DataLoader(ds_importance, batch_size=64, shuffle=False, num_workers=0)
    inputs = collect_kan_inputs(model, importance_loader, device, max_batches=5, max_samples=1000)
    importances = calculate_kan_importance(model, inputs, device)
    stats = compute_importance_stats(importances)
    print(f"       Найдено {len(importances)} слоёв KANLinear")

    # Прореживание
    print(f"[3/5] Эксперимент pruning по {len(thresholds)} порогам...")
    pruning_results = []
    for threshold in thresholds:
        pruned_model = copy.deepcopy(model)
        active_edges, total_edges = apply_pruning_mask(pruned_model, importances, threshold)
        metrics = _evaluate_model(pruned_model, ds_eval, device, eval_batch_size, mode)

        keep_ratio = float(active_edges) / float(total_edges) if total_edges > 0 else 0.0
        pruning_results.append({
            'threshold': float(threshold),
            'keep_ratio': keep_ratio,
            'active_edges': int(active_edges),
            'total_edges': int(total_edges),
            'acc': float(metrics.get('acc', 0.0)),
            'f1': float(metrics.get('f1', 0.0))
        })

    # Абляция физических слоёв (если это PhysicsKAN)
    print(f"[4/5] Абляционные эксперименты...")
    ablation_results = {}
    if config.get('model', {}).get('name') == 'PhysicsKAN':
        ablations = {
            'baseline': (1.0, 1.0, 1.0),
            'no_mult': (1.0, 0.0, 1.0),
            'no_div': (1.0, 1.0, 0.0),
            'no_arith': (1.0, 0.0, 0.0),
            'only_arith': (0.0, 1.0, 1.0)
        }

        for name, (s_raw, s_mult, s_div) in ablations.items():
            wrapped = PhysicsKANAblation(model, s_raw, s_mult, s_div).to(device)
            ablation_results[name] = _evaluate_model(wrapped, ds_eval, device, eval_batch_size, mode)

        arith_stats = _compute_arithmetic_stats(model, ds_importance, device, batch_size=64)
    else:
        arith_stats = {}

    # Сохранение отчёта
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'experiment': exp_name,
        'base_metrics': base_metrics,
        'pruning_results': pruning_results,
        'importance_stats': stats,
        'ablation_results': ablation_results,
        'arithmetic_stats': arith_stats
    }

    report_path = output_dir / f"{exp_name}_pruning_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # График pruning curve
    if pruning_results:
        xs = [r['keep_ratio'] for r in pruning_results]
        ys = [r['f1'] for r in pruning_results]

        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, marker='o')
        plt.xlabel('Доля сохранённых связей')
        plt.ylabel('F1 (macro)')
        plt.title('Pruning Curve')
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.tight_layout()
        plt.savefig(output_dir / f"{exp_name}_pruning_curve.png", dpi=150)
        plt.close()

    # График абляций
    if ablation_results:
        labels = list(ablation_results.keys())
        f1_vals = [ablation_results[k].get('f1', 0.0) for k in labels]

        plt.figure(figsize=(8, 4))
        plt.bar(labels, f1_vals)
        plt.ylabel('F1 (macro)')
        plt.title('Вклад физических слоёв (абляция)')
        plt.xticks(rotation=30, ha='right')
        plt.grid(True, axis='y', alpha=0.3, linestyle=':')
        plt.tight_layout()
        plt.savefig(output_dir / f"{exp_name}_ablation_f1.png", dpi=150)
        plt.close()

    # Гистограммы и heatmap важности (первые 3 слоя по среднему значению)
    print(f"[5/5] Генерация графиков важности...")
    if stats:
        top_layers = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)[:3]
        for layer_name, layer_stats in top_layers:
            scores = importances[layer_name].numpy()
            safe_name = layer_name.replace('.', '_')
            
            # Гистограмма
            plt.figure(figsize=(6, 4))
            plt.hist(scores.reshape(-1), bins=40, alpha=0.8, color='steelblue')
            plt.axvline(layer_stats['median'], color='red', linestyle='--', label=f"median={layer_stats['median']:.4f}")
            plt.title(f"Распределение важности: {layer_name}")
            plt.xlabel('Важность |phi(x)|')
            plt.ylabel('Частота')
            plt.legend()
            plt.grid(True, alpha=0.3, linestyle=':')
            plt.tight_layout()
            plt.savefig(output_dir / f"{exp_name}_hist_{safe_name}.png", dpi=150)
            plt.close()
            
            # Heatmap (если размерность не слишком большая)
            if scores.shape[0] <= 64 and scores.shape[1] <= 128:
                plt.figure(figsize=(max(6, scores.shape[1] * 0.15), max(4, scores.shape[0] * 0.2)))
                plt.imshow(scores, aspect='auto', cmap='viridis')
                plt.colorbar(label='Важность |phi(x)|')
                plt.title(f"Heatmap важности: {layer_name}")
                plt.xlabel('Вход')
                plt.ylabel('Выход')
                plt.tight_layout()
                plt.savefig(output_dir / f"{exp_name}_heatmap_{safe_name}.png", dpi=150)
                plt.close()
    
    print(f"\n=== Отчёт сохранён в {report_path} ===")
    print(f"    Графики: {output_dir}")
    if pruning_results:
        best_pr = max(pruning_results, key=lambda r: r['f1'])
        print(f"    Лучший pruning: keep_ratio={best_pr['keep_ratio']:.2%}, F1={best_pr['f1']:.4f}")
    if ablation_results:
        print(f"    Абляции: {list(ablation_results.keys())}")


if __name__ == "__main__":
    # ==========================================================================
    # РУЧНОЙ РЕЖИМ НАСТРОЙКИ (как в других скриптах проекта)
    # ==========================================================================
    MANUAL_RUN = True
    
    if MANUAL_RUN:
        # --- Основные настройки ---
        # Имя эксперимента (папка в experiments/)
        EXP_NAME = "Exp_2.6.1_PhysicsKAN_medium_phase_polar_stride_base_weights_aug"
        
        # Путь к данным
        DATA_DIR = ROOT_DIR / 'data' / 'ml_datasets'
        
        # Папка для отчётов
        OUTPUT_DIR = ROOT_DIR / 'reports' / 'phase2_6' / 'exp_2_6_5'
        
        # --- Пороги для pruning ---
        # Связи с важностью |phi(x)| < threshold будут удалены (маска = 0)
        # Чем выше порог, тем больше связей удаляется
        THRESHOLDS = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1]
        
        # --- Параметры оценки ---
        # Количество окон для расчёта важности (больше = точнее статистика)
        # -1 = использовать все доступные окна
        MAX_IMPORTANCE_WINDOWS = -1  # все окна для точной статистики
        
        # Количество окон для оценки качества (больше = точнее метрики)
        # -1 = использовать все доступные окна
        MAX_EVAL_WINDOWS = -1  # весь тестовый датасет
        
        # Шаг между окнами при создании индексов (1 = каждое окно, 16 = каждое 16-е)
        # Меньший шаг = больше окон, точнее оценка, но дольше
        EVAL_STRIDE = 1  # каждое окно для полного покрытия
    else:
        parser = argparse.ArgumentParser(description="Pruning и абляция PhysicsKAN (Exp 2.6.5)")
        parser.add_argument('--exp', type=str, required=True, help='Имя эксперимента')
        parser.add_argument('--data-dir', type=str, default=None, help='Путь к data/ml_datasets')
        parser.add_argument('--output-dir', type=str, default=None, help='Папка отчёта')
        parser.add_argument('--thresholds', type=str, default='1e-4,1e-3,5e-3,1e-2,5e-2,1e-1,2e-1')
        parser.add_argument('--max-importance-windows', type=int, default=-1)
        parser.add_argument('--max-eval-windows', type=int, default=-1)
        parser.add_argument('--eval-stride', type=int, default=1)
        args = parser.parse_args()
        
        EXP_NAME = args.exp
        DATA_DIR = Path(args.data_dir) if args.data_dir else ROOT_DIR / 'data' / 'ml_datasets'
        OUTPUT_DIR = Path(args.output_dir) if args.output_dir else ROOT_DIR / 'reports' / 'phase2_6' / 'exp_2_6_5'
        THRESHOLDS = [float(t.strip()) for t in args.thresholds.split(',') if t.strip()]
        MAX_IMPORTANCE_WINDOWS = args.max_importance_windows
        MAX_EVAL_WINDOWS = args.max_eval_windows
        EVAL_STRIDE = args.eval_stride
    
    print(f"=== Exp 2.6.5: KAN Pruning & Ablation ===")
    print(f"    Эксперимент: {EXP_NAME}")
    print(f"    Данные: {DATA_DIR}")
    print(f"    Отчёт: {OUTPUT_DIR}")
    print(f"    Пороги: {THRESHOLDS}")
    print(f"    Окна importance: {'все' if MAX_IMPORTANCE_WINDOWS < 0 else MAX_IMPORTANCE_WINDOWS}")
    print(f"    Окна eval: {'все' if MAX_EVAL_WINDOWS < 0 else MAX_EVAL_WINDOWS}")
    print(f"    Stride: {EVAL_STRIDE}")
    print()

    run_pruning_experiment(
        exp_name=EXP_NAME,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        thresholds=THRESHOLDS,
        max_importance_windows=MAX_IMPORTANCE_WINDOWS,
        max_eval_windows=MAX_EVAL_WINDOWS,
        eval_stride=EVAL_STRIDE
    )
