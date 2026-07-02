"""Статистика и иллюстрации для раздела статьи о насыщении ТТ.

Сценарий умеет:
1. построить кривые завершённого обучения;
2. оценить модель на исходном validation holdout по фазе A;
3. отдельно проверить перенос голов A/B/C на фиксированных циклических копиях;
4. построить компактные двухпанельные рисунки симулированных осциллограмм;
5. разметить подготовленный реальный CSV без эталонных меток насыщения.

При запуске без аргументов используется подробный ручной блок в конце файла.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.ct_saturation_dataset import (
    CTSaturationFile,
    compute_current_spectral_features,
    current_feature_count,
    load_ct_saturation_mat,
)
from osc_tools.ml.models.transformer import PhysicalKANTransformer


PHASE_NAMES = ("A", "B", "C")
PHASE_COLORS = ("#D4A000", "#228B22", "#D84315")


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Восстановить Physical KAN-Transformer из checkpoint насыщения ТТ."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    num_features = current_feature_count(config["num_harmonics"], config["sub_periods"])
    model = PhysicalKANTransformer(
        num_input_channels=num_features,
        num_current_pairs=num_features // 2,
        num_classes=3,
        zone_size=1,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=0.0,
        cls_head_type="kan",
        max_seq_len=(config["num_periods"] - 1) * config["stride_fraction"],
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, config


def read_training_log(path: str | Path) -> list[dict]:
    """Прочитать JSONL-журнал обучения."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot_training_curves(rows: Sequence[dict], output_path: Path) -> None:
    """Построить один рисунок с loss и F1 по фазе A."""
    epochs = np.array([row["epoch"] + 1 for row in rows])
    train_loss = np.array([row["train_loss"] for row in rows])
    val_loss = np.array([row["val_loss"] for row in rows])
    train_f1 = np.array([row["train"]["phase_f1"][0] for row in rows])
    val_f1_a = np.array([row["val"]["phase_f1"][0] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(epochs, train_loss, label="Обучение", color="#2563A6")
    axes[0].plot(epochs, val_loss, label="Проверка", color="#D84315")
    axes[0].set(xlabel="Эпоха", ylabel="Функция потерь", title="Сходимость обучения")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25, linestyle=":")
    axes[0].legend()

    axes[1].plot(epochs, train_f1, label="Обучение, фаза A", color="#2563A6")
    axes[1].plot(epochs, val_f1_a, label="Проверка, фаза A", color="#D84315")
    axes[1].set(xlabel="Эпоха", ylabel="F1", title="Качество распознавания")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25, linestyle=":")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _feature_sequence(
    currents: np.ndarray,
    fs_hz: float,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Рассчитать признаки всей осциллограммы и позиции токенов в отсчётах."""
    spp = round(fs_hz / 50.0)
    stride = max(1, spp // config["stride_fraction"])
    context = max(config["sub_periods"]) * spp
    padded = np.pad(currents, ((context, 0), (0, 0)), mode="edge")
    features = compute_current_spectral_features(
        padded,
        samples_per_period=spp,
        stride=stride,
        warmup=context + spp,
        num_harmonics=config["num_harmonics"],
        sub_periods=config["sub_periods"],
    )
    positions = spp + np.arange(len(features), dtype=np.int64) * stride
    valid = positions < len(currents)
    return features[valid], positions[valid], stride


@torch.no_grad()
def infer_oscillogram(
    model: torch.nn.Module,
    currents_normalized: np.ndarray,
    fs_hz: float,
    config: dict,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Разметить всю осциллограмму перекрывающимися окнами признаков."""
    features, positions, stride = _feature_sequence(currents_normalized, fs_hz, config)
    sequence_length = (config["num_periods"] - 1) * config["stride_fraction"]
    step_tokens = config["stride_fraction"]  # одно новое окно на период
    if len(features) < sequence_length:
        pad = sequence_length - len(features)
        features = np.pad(features, ((0, pad), (0, 0)), mode="edge")
        positions = np.pad(positions, (0, pad), mode="edge")
    starts = list(range(0, max(1, len(features) - sequence_length + 1), step_tokens))
    last = max(0, len(features) - sequence_length)
    if starts[-1] != last:
        starts.append(last)

    probability_sum = np.zeros((len(features), 3), dtype=np.float64)
    coverage = np.zeros(len(features), dtype=np.int32)
    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start:batch_start + batch_size]
        batch = np.stack([
            features[start:start + sequence_length].T for start in batch_starts
        ]).astype(np.float32)
        logits = model(torch.from_numpy(batch).to(device), mode="classify")["classify"]
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()
        for local_index, start in enumerate(batch_starts):
            stop = start + sequence_length
            probability_sum[start:stop] += probabilities[local_index]
            coverage[start:stop] += 1
    mask = coverage > 0
    result = np.full((len(features), 3), np.nan, dtype=np.float32)
    result[mask] = (probability_sum[mask] / coverage[mask, None]).astype(np.float32)
    original_length = np.searchsorted(positions, positions[-1], side="right")
    return result[:original_length], positions[:original_length], stride


def token_targets(labels: np.ndarray, positions: np.ndarray, stride: int) -> np.ndarray:
    """Агрегировать поотсчётные метки в те же зоны, что и предсказания."""
    targets = np.zeros((len(positions), 3), dtype=np.uint8)
    for index, start in enumerate(positions):
        stop = min(int(start + stride), len(labels))
        if start < len(labels) and stop > start:
            targets[index] = labels[int(start):stop].max(axis=0)
    return targets


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Рассчитать привычные бинарные метрики без введения PR-AUC."""
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def plot_confusion_matrix(metrics: dict, output_path: Path) -> None:
    """Построить бинарную матрицу ошибок для исходной фазы A."""
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["Нет насыщения", "Насыщение"])
    ax.set_yticks([0, 1], ["Нет насыщения", "Насыщение"])
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истинная метка")
    ax.set_title("Матрица ошибок, фаза A")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_two_panel_marking(
    output_path: Path,
    currents: np.ndarray,
    fs_hz: float,
    probabilities: np.ndarray,
    positions: np.ndarray,
    *,
    title: str,
    threshold: float,
    labels: np.ndarray | None = None,
    normalized_currents: bool = False,
) -> None:
    """Построить рисунок: три тока сверху, три смещённых прогноза снизу."""
    time_ms = np.arange(len(currents)) * 1000.0 / fs_hz
    token_time_ms = positions * 1000.0 / fs_hz
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1.0]})
    for phase, color, signal in zip(PHASE_NAMES, PHASE_COLORS, currents.T):
        axes[0].plot(time_ms, signal, label=f"I{phase}", color=color, linewidth=0.9)
    axes[0].set_ylabel("Ток, о.е." if normalized_currents else "Вторичный ток, А")
    axes[0].legend(ncol=3, loc="upper right")
    axes[0].grid(alpha=0.25, linestyle=":")
    axes[0].set_title(title)

    for phase_index, (phase, color) in enumerate(zip(PHASE_NAMES, PHASE_COLORS)):
        offset = float(phase_index)
        axes[1].plot(token_time_ms, probabilities[:, phase_index] + offset,
                     color=color, linewidth=1.1, label=f"P({phase})")
        axes[1].axhline(offset + threshold, color=color, linestyle="--", linewidth=0.7, alpha=0.65)
        if labels is not None:
            true_mask = labels[:, phase_index].astype(bool)
            if np.any(true_mask):
                axes[1].fill_between(time_ms, offset, offset + 1, where=true_mask,
                                     color=color, alpha=0.12, step="pre")
    axes[1].set_yticks([0.5, 1.5, 2.5], ["Фаза A", "Фаза B", "Фаза C"])
    axes[1].set_ylim(-0.05, 3.05)
    axes[1].set_xlabel("Время, мс")
    axes[1].set_ylabel("Вероятность + уровень")
    axes[1].grid(alpha=0.2, linestyle=":")
    axes[1].legend(ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _parse_split_files(names: Iterable[str], prepared_dir: Path) -> list[CTSaturationFile]:
    result = []
    for name in names:
        parts = name.split("_")
        tau = 0.015 if "0.015" in name else 0.4
        record_id = int(parts[-2])
        result.append(CTSaturationFile(prepared_dir / name, record_id, tau))
    return result


def select_validation_files(
    files: Sequence[CTSaturationFile], max_files: int | None, seed: int,
) -> list[CTSaturationFile]:
    """Выбрать воспроизводимый поднабор с сохранением долей двух tau-групп."""
    if max_files is None or max_files >= len(files):
        return list(files)
    rng = np.random.default_rng(seed)
    groups: dict[float, list[CTSaturationFile]] = defaultdict(list)
    for item in files:
        groups[item.source_tau_s].append(item)
    selected: list[CTSaturationFile] = []
    for group in groups.values():
        count = round(max_files * len(group) / len(files))
        indices = rng.choice(len(group), size=min(count, len(group)), replace=False)
        selected.extend(group[index] for index in indices)
    return sorted(selected, key=lambda item: (item.source_tau_s, item.record_id))[:max_files]


def split_audit(split: dict, output_path: Path) -> dict:
    """Зафиксировать файловое разбиение и риск близких режимов между частями."""
    train = set(split["train"])
    val = set(split["val"])
    report = {
        "train_files": len(train),
        "validation_files": len(val),
        "exact_filename_overlap": len(train & val),
        "split_unit": "oscillogram file name",
        "grouping_by_physical_regime": False,
        "warning": (
            "Файлы не пересекаются, но hash-разбиение не группирует соседние точки "
            "параметрической сетки моделирования. Близкие режимы могут присутствовать "
            "в обеих частях; это ограничение следует указать в статье."
        ),
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def evaluate_simulated(config: dict, model, model_config: dict, device: torch.device) -> dict:
    """Собрать статистику на исходном validation holdout и рисунки примеров."""
    run_dir = Path(config["run_dir"])
    output_dir = Path(config["output_dir"]) / "simulated"
    figures_dir = output_dir / "examples"
    figures_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    split_audit(split, output_dir / "split_audit.json")
    files = _parse_split_files(split["val"], Path(model_config["data_dir"]))
    files = select_validation_files(files, config["max_sim_files"], config["seed"])

    threshold = config["threshold"]
    zone_truth, zone_pred = [], []
    file_truth, file_pred = [], []
    delays_ms: list[float] = []
    by_tau: dict[float, dict[str, list]] = defaultdict(lambda: {"truth": [], "pred": []})
    examples_saved = 0
    examples_by_tau: dict[float, int] = defaultdict(int)
    example_quota_per_tau = max(1, math.ceil(config["sim_example_count"] / 2))
    for item in tqdm(files, desc="Simulated validation", dynamic_ncols=True):
        record = load_ct_saturation_mat(item.path)
        currents_a = np.asarray(record["secondary_a"], dtype=np.float32)
        normalized = currents_a / (model_config["nominal_secondary_a"] * model_config["reserve_factor"])
        probs, positions, stride = infer_oscillogram(
            model, normalized, float(record["fs_hz"]), model_config, device,
            batch_size=config["inference_batch_size"],
        )
        targets = token_targets(np.asarray(record["labels"]), positions, stride)
        valid = ~np.isnan(probs[:, 0])
        truth_a = targets[valid, 0] >= 0.5
        pred_a = probs[valid, 0] >= threshold
        zone_truth.append(truth_a)
        zone_pred.append(pred_a)
        is_positive = bool(np.any(truth_a))
        is_detected = bool(np.any(pred_a))
        file_truth.append(is_positive)
        file_pred.append(is_detected)
        by_tau[item.source_tau_s]["truth"].append(is_positive)
        by_tau[item.source_tau_s]["pred"].append(is_detected)
        if is_positive and is_detected:
            true_index = int(np.flatnonzero(truth_a)[0])
            pred_after = np.flatnonzero(pred_a & (np.arange(len(pred_a)) >= true_index))
            if len(pred_after):
                delays_ms.append(float((positions[pred_after[0]] - positions[true_index]) * 1000 / record["fs_hz"]))
        if (is_positive and examples_saved < config["sim_example_count"]
                and examples_by_tau[item.source_tau_s] < example_quota_per_tau):
            plot_two_panel_marking(
                figures_dir / f"sim_tau_{item.source_tau_s:g}_{item.record_id}.png",
                currents_a, float(record["fs_hz"]), probs, positions,
                title=f"Моделирование: τ={item.source_tau_s:g} с, опыт {item.record_id}",
                threshold=threshold, labels=np.asarray(record["labels"]),
            )
            examples_saved += 1
            examples_by_tau[item.source_tau_s] += 1

    zone = binary_metrics(np.concatenate(zone_truth), np.concatenate(zone_pred))
    event = binary_metrics(np.asarray(file_truth), np.asarray(file_pred))
    tau_metrics = {
        f"tau_{tau:g}": binary_metrics(values["truth"], values["pred"])
        for tau, values in sorted(by_tau.items())
    }
    summary = {
        "evaluated_files": len(files),
        "threshold": threshold,
        "zone_phase_A": zone,
        "event_phase_A": event,
        "event_by_tau": tau_metrics,
        "detection_delay_ms": {
            "count": len(delays_ms),
            "median": float(np.median(delays_ms)) if delays_ms else None,
            "p95": float(np.percentile(delays_ms, 95)) if delays_ms else None,
        },
        "phase_note": (
            "Основная holdout-оценка относится к исходной фазе A. "
            "Головы B/C обучены циклической перестановкой и не включены в эту основную цифру."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_confusion_matrix(zone, output_dir / "confusion_matrix_phase_A.png")
    return summary


def evaluate_phase_transfer(config: dict, model, model_config: dict, device: torch.device) -> dict:
    """Проверить три головы на одинаковых сигналах с фиксированными сдвигами фаз."""
    run_dir = Path(config["run_dir"])
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    files = _parse_split_files(split["val"], Path(model_config["data_dir"]))
    files = select_validation_files(files, config["phase_transfer_max_files"], config["seed"] + 1)
    truth_by_phase = [[] for _ in range(3)]
    pred_by_phase = [[] for _ in range(3)]
    for item in tqdm(files, desc="Phase-transfer audit", dynamic_ncols=True):
        record = load_ct_saturation_mat(item.path)
        base_labels = np.asarray(record["labels"])
        if not np.any(base_labels[:, 0]):
            continue
        for shift in range(3):
            currents = np.roll(np.asarray(record["secondary_a"]), shift, axis=1)
            labels = np.roll(base_labels, shift, axis=1)
            normalized = currents / (model_config["nominal_secondary_a"] * model_config["reserve_factor"])
            probs, positions, stride = infer_oscillogram(
                model, normalized, float(record["fs_hz"]), model_config, device,
                batch_size=config["inference_batch_size"],
            )
            targets = token_targets(labels, positions, stride)
            valid = ~np.isnan(probs[:, shift])
            truth_by_phase[shift].append(targets[valid, shift] >= 0.5)
            pred_by_phase[shift].append(probs[valid, shift] >= config["threshold"])
    result = {}
    for phase_index, phase in enumerate(PHASE_NAMES):
        if truth_by_phase[phase_index]:
            result[phase] = binary_metrics(
                np.concatenate(truth_by_phase[phase_index]),
                np.concatenate(pred_by_phase[phase_index]),
            )
    output_dir = Path(config["output_dir"]) / "simulated"
    (output_dir / "phase_transfer.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def plot_real_dataset(config: dict, model, model_config: dict, device: torch.device) -> None:
    """Построить двухпанельную разметку всех выбранных реальных файлов."""
    output_dir = Path(config["output_dir"]) / "real_marking"
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["sample", "file_name", "IA", "IB", "IC"]
    dataframe = pl.read_csv(
        config["real_csv_path"], columns=columns,
        schema_overrides={"file_name": pl.String, "IA": pl.Float32, "IB": pl.Float32, "IC": pl.Float32},
    )
    groups = dataframe.partition_by("file_name", maintain_order=True)
    if config["max_real_files"] is not None:
        groups = groups[:config["max_real_files"]]
    summary_rows: list[dict] = []
    for group in tqdm(groups, desc="Real oscillograms", dynamic_ncols=True):
        file_name = str(group["file_name"][0])
        currents = group.select(["IA", "IB", "IC"]).to_numpy().astype(np.float32)
        probabilities, positions, _ = infer_oscillogram(
            model, currents, config["real_fs_hz"], model_config, device,
            batch_size=config["inference_batch_size"],
        )
        max_probs = np.nanmax(probabilities, axis=0)
        summary_rows.append({
            "file_name": file_name,
            "max_probability_A": float(max_probs[0]),
            "max_probability_B": float(max_probs[1]),
            "max_probability_C": float(max_probs[2]),
            "detected_A": bool(max_probs[0] >= config["threshold"]),
            "detected_B": bool(max_probs[1] >= config["threshold"]),
            "detected_C": bool(max_probs[2] >= config["threshold"]),
        })
        safe_name = file_name.replace("/", "_").replace("\\", "_")
        plot_two_panel_marking(
            output_dir / f"{safe_name}.png",
            currents, config["real_fs_hz"], probabilities, positions,
            title=f"Реальная осциллограмма: {file_name}",
            threshold=config["threshold"], normalized_currents=True,
        )
    with (output_dir / "real_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["file_name"])
        writer.writeheader()
        writer.writerows(summary_rows)


def write_report(config: dict, simulated: dict | None, transfer: dict | None) -> None:
    """Сохранить краткий Markdown-отчёт, пригодный как источник текста статьи."""
    lines = ["# Насыщение ТТ: отчёт первого цикла", ""]
    if simulated:
        zone = simulated["zone_phase_A"]
        event = simulated["event_phase_A"]
        lines += [
            f"Проверено файлов: {simulated['evaluated_files']}.", "",
            f"Позонно, фаза A: Precision={zone['precision']:.4f}, "
            f"Recall={zone['recall']:.4f}, F1={zone['f1']:.4f}.",
            f"По осциллограммам: Precision={event['precision']:.4f}, "
            f"Recall={event['recall']:.4f}, F1={event['f1']:.4f}.", "",
        ]
    if transfer:
        lines.append("Циклическая проверка фазовых голов:")
        for phase, values in transfer.items():
            lines.append(f"- {phase}: Precision={values['precision']:.4f}, Recall={values['recall']:.4f}, F1={values['f1']:.4f}.")
        lines.append("")
    lines += [
        "Ограничение разбиения: train и validation не пересекаются по именам "
        "осциллограмм, однако соседние точки параметрической сетки не объединялись "
        "в группы и могли попасть в разные части.",
        "",
        "Реальная выборка не имеет эталонной разметки насыщения ТТ; её рисунки "
        "предназначены только для последующего экспертного отбора и качественного анализа.",
    ]
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(config["output_dir"]) / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(config: dict) -> None:
    """Выполнить выбранные части ручного анализа."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model(config["checkpoint_path"], device)
    rows = read_training_log(Path(config["run_dir"]) / "training_log.jsonl")
    plot_training_curves(rows, output_dir / "training_curves.png")
    simulated = evaluate_simulated(config, model, model_config, device) if config["do_simulated"] else None
    transfer = evaluate_phase_transfer(config, model, model_config, device) if config["do_phase_transfer"] else None
    if config["do_real"]:
        plot_real_dataset(config, model, model_config, device)
    write_report(config, simulated, transfer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Анализ модели насыщения ТТ")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-sim-files", type=int)
    parser.add_argument("--max-real-files", type=int)
    parser.add_argument("--real", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"),
        "output_dir": str(Path(args.output_dir) if args.output_dir else run_dir / "article_report"),
        "real_csv_path": str(PROJECT_ROOT / "data" / "ml_datasets" / "labeled_2025_12_03.csv"),
        "do_simulated": True,
        "do_phase_transfer": True,
        "do_real": args.real,
        "max_sim_files": args.max_sim_files,
        "phase_transfer_max_files": 300,
        "max_real_files": args.max_real_files,
        "sim_example_count": 4,
        "real_fs_hz": 1600.0,
        "threshold": 0.5,
        "inference_batch_size": 64,
        "seed": 42,
    }
    run_analysis(config)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        sys.exit(0)

    # =================================================================
    # РУЧНОЙ ЗАПУСК ИЗ IDE
    # =================================================================
    RUN_DIR = PROJECT_ROOT / "experiments" / "phase4" / "ct_saturation" / "run_20260702_023838"
    CHECKPOINT_PATH = RUN_DIR / "latest_checkpoint.pt"
    OUTPUT_DIR = RUN_DIR / "article_report"
    REAL_CSV_PATH = PROJECT_ROOT / "data" / "ml_datasets" / "labeled_2025_12_03.csv"

    # Части анализа. Реальные рисунки можно включить отдельным вторым запуском.
    DO_SIMULATED = False
    DO_PHASE_TRANSFER = False
    DO_REAL = True

    # None означает полный validation; для первого прогона удобно 1000–3000.
    MAX_SIM_FILES = 5000
    PHASE_TRANSFER_MAX_FILES = 300
    SIM_EXAMPLE_COUNT = 4

    # Реальный архив: None — построить все файлы; 20 — проверить контур.
    MAX_REAL_FILES = None
    REAL_FS_HZ = 1600.0

    THRESHOLD = 0.50
    INFERENCE_BATCH_SIZE = 64
    SEED = 42
    # =================================================================

    manual_config = {
        "run_dir": str(RUN_DIR),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "output_dir": str(OUTPUT_DIR),
        "real_csv_path": str(REAL_CSV_PATH),
        "do_simulated": DO_SIMULATED,
        "do_phase_transfer": DO_PHASE_TRANSFER,
        "do_real": DO_REAL,
        "max_sim_files": MAX_SIM_FILES,
        "phase_transfer_max_files": PHASE_TRANSFER_MAX_FILES,
        "max_real_files": MAX_REAL_FILES,
        "sim_example_count": SIM_EXAMPLE_COUNT,
        "real_fs_hz": REAL_FS_HZ,
        "threshold": THRESHOLD,
        "inference_batch_size": INFERENCE_BATCH_SIZE,
        "seed": SEED,
    }
    run_analysis(manual_config)
