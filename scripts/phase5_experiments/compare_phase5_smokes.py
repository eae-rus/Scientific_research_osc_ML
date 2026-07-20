"""Собрать сопоставимый отчёт по завершённым Phase 5 A/B smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(run_dir: Path) -> dict[str, object]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    best = min(records, key=lambda record: record["val_loss"])
    last = records[-1]
    return {
        "run": run_dir.name,
        "feature_version": config["feature_version"],
        "feature_contract": config["feature_passport"]["feature_contract"],
        "temporal_mode": config["temporal_mode"],
        "loss_type": config.get("loss_type", "complex_mse"),
        "epochs": len(records),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "last_open_ee_loss": last.get("val_open_ee_loss"),
        "last_french_rte_loss": last.get("val_french_rte_loss"),
        "last_train_samples_per_second": last.get("train_samples_per_second"),
        "last_peak_cuda_memory_mib": last.get("peak_cuda_memory_mib"),
    }


def build_report(run_dirs: list[Path], output_path: Path) -> list[dict[str, object]]:
    summaries = [summarize(path) for path in run_dirs]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Phase 5 A/B smoke comparison",
        "",
        "Loss сопоставим только как masked complex reconstruction probe; он не заменяет PDR/transfer оценку.",
        "",
        "| Run | Contract | Loss | Mode | Epochs | Best val | Open_EE | French | samples/s | CUDA MiB |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        value = lambda key: "—" if item[key] is None else f"{item[key]:.4f}"
        lines.append(
            f"| {item['run']} | {item['feature_contract']} | {item['loss_type']} | {item['temporal_mode']} | "
            f"{item['epochs']} | {value('best_val_loss')} | {value('last_open_ee_loss')} | "
            f"{value('last_french_rte_loss')} | {value('last_train_samples_per_second')} | "
            f"{value('last_peak_cuda_memory_mib')} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/phase5/pretrain_smoke_comparison.md"))
    args = parser.parse_args()
    print(json.dumps(build_report(args.run_dirs, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
