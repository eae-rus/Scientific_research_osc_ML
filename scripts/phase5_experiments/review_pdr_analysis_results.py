"""Дополнительный воспроизводимый аудит полной Phase 5 PDR-разметки.

Сценарий не читает тяжёлые label shards: он объединяет уже рассчитанные
``record_statistics.csv`` и ``signal_record_statistics.csv`` и формирует
компактные таблицы для инженерного и научного ревью.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "data/phase5/pdr_analysis_v4"
ALGORITHMS = (
    "adaptive_pdr_mir",
    "phase_pdr_basic",
    "pos_seq_pdr_basic",
    "phase_power_pdr_basic",
    "pos_seq_power_pdr_basic",
)
SIGNAL_METRICS = (
    "current_rms",
    "voltage_rms",
    "current_to_voltage_rms",
    "current_phase_unbalance_cv",
    "voltage_phase_unbalance_cv",
    "current_rms_last_over_first",
    "current_crest_p99_over_rms",
    "mean_three_phase_power_proxy",
)


def _quantiles(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"mean": np.nan, "median": np.nan, "p90": np.nan, "p95": np.nan, "p99": np.nan}
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
    }


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    valid = numeric_values.notna() & numeric_weights.notna() & (numeric_weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(numeric_values[valid], weights=numeric_weights[valid]))


def _write_metric_guide(path: Path) -> None:
    path.write_text("""# Как читать статистику PDR

## Единицы усреднения

- `record_mean`: каждая осциллограмма имеет один голос независимо от длины и SPP.
- `duration_weighted`: вклад пропорционален физической длительности записи.
- `point_weighted`: вклад пропорционален числу подходящих точек. Для доли
  FORWARD весом служит число валидных точек конкретного РНМ.
- Доверительные интервалы следует строить по осциллограммам или семействам
  одинаковых SHA-256, а не по соседним точкам одной записи.

## Согласие органов

- `agreement`: обычная доля совпавших решений. Может быть завышена, если почти
  всё время встречается один класс.
- `balanced_agreement_symmetric`: совпадение, в котором REVERSE и FORWARD имеют
  равный вес; дополнительно симметризовано относительно двух органов.
- `cohen_kappa`: совпадение сверх ожидаемого при наблюдаемых долях классов.
  При сильном дисбалансе читать только вместе с матрицей 00/01/10/11.
- `mcc`: корреляция двух бинарных решений от -1 до 1; 1 — полное совпадение,
  0 — отсутствие бинарной связи, -1 — противоположные решения.
- `jaccard_forward/reverse`: пересечение выбранного состояния относительно
  объединения точек, где хотя бы один орган выбрал это состояние.
- `forward_prevalence_left/right`: доли FORWARD каждого органа; объясняют,
  вызвано ли расхождение разными рабочими порогами.

## Временные показатели

- `state_entropy_bits`: разнообразие 0/1 внутри записи; ноль означает одно
  постоянное состояние, максимум 1 бит — близкие доли двух состояний.
- `transition_entropy_bits`: разнообразие переходов 00/01/10/11.
- `lag1_autocorrelation`: сохранение состояния между соседними точками.
- `chatter_returns_le_100ms`: пары быстрых возвратных переключений за 100 мс.
- `max_switches_in_0_5s/1_0s`: локальная плотность переключений, отделяющая
  короткий содержательный эпизод от равномерного шума по всей записи.

## Комбинации пяти органов

`state_pattern` содержит биты в порядке из `algorithm_order`. Например, `10100`
означает FORWARD у первого и третьего органов. Сравнение `point_fraction` с
`mean_record_fraction` показывает влияние длинных осциллограмм и высокого SPP.
""", encoding="utf-8")


def _build_review_figures(
    analysis_dir: Path,
    merged: pd.DataFrame,
    pointwise: pd.DataFrame,
    patterns: pd.DataFrame,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = analysis_dir / "review_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for source, group in merged.groupby("source"):
        current = pd.to_numeric(group["current_rms_physical_pu"], errors="coerce").dropna().sort_values()
        transitions = pd.to_numeric(group["total_transitions"], errors="coerce").dropna().sort_values()
        if len(current):
            axes[0].step(current, np.arange(1, len(current) + 1) / len(current), where="post", label=source)
        if len(transitions):
            axes[1].step(transitions, np.arange(1, len(transitions) + 1) / len(transitions), where="post", label=source)
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].set_xlabel("RMS тока, I/Iном")
    axes[0].set_ylabel("Доля осциллограмм ≤ x")
    axes[0].set_title("ECDF тока: один голос на осциллограмму")
    axes[1].set_xscale("symlog", linthresh=1.0)
    axes[1].set_xlabel("Число переключений всех РНМ")
    axes[1].set_ylabel("Доля осциллограмм ≤ x")
    axes[1].set_title("ECDF временной активности")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    path = figure_dir / "record_ecdf.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    outputs.append(path)

    if not pointwise.empty:
        agreement_metric = (
            "balanced_agreement_symmetric"
            if "balanced_agreement_symmetric" in pointwise.columns else "agreement"
        )
        sources = list(pointwise["source"].drop_duplicates())
        figure, axes = plt.subplots(1, len(sources), figsize=(7 * len(sources), 6), squeeze=False)
        for axis, source in zip(axes[0], sources):
            subset = pointwise[pointwise["source"] == source]
            matrix = pd.DataFrame(np.eye(len(ALGORITHMS)), index=ALGORITHMS, columns=ALGORITHMS)
            for row in subset.itertuples():
                value = getattr(row, agreement_metric)
                matrix.loc[row.left, row.right] = value
                matrix.loc[row.right, row.left] = value
            image = axis.imshow(matrix.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="viridis")
            axis.set_xticks(range(len(ALGORITHMS)), [name.replace("_pdr_basic", "").replace("adaptive_pdr_mir", "adaptive") for name in ALGORITHMS], rotation=35, ha="right")
            axis.set_yticks(range(len(ALGORITHMS)), [name.replace("_pdr_basic", "").replace("adaptive_pdr_mir", "adaptive") for name in ALGORITHMS])
            axis.set_title(f"Balanced agreement — {source}")
            for row_index in range(len(ALGORITHMS)):
                for column_index in range(len(ALGORITHMS)):
                    axis.text(column_index, row_index, f"{matrix.iloc[row_index, column_index]:.2f}", ha="center", va="center", color="white" if matrix.iloc[row_index, column_index] < 0.55 else "black")
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="Симметричное balanced agreement")
        figure.subplots_adjust(bottom=0.22, wspace=0.35)
        path = figure_dir / "balanced_agreement_heatmap.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        outputs.append(path)

    if not patterns.empty:
        for source, group in patterns.groupby("source"):
            subset = group.sort_values("mean_record_fraction", ascending=False).head(12).iloc[::-1]
            y = np.arange(len(subset))
            figure, axis = plt.subplots(figsize=(10, 6))
            axis.barh(y - 0.18, subset["mean_record_fraction"], height=0.34, label="среднее по осциллограммам")
            axis.barh(y + 0.18, subset["point_fraction"], height=0.34, label="по всем точкам")
            axis.set_yticks(y, subset["state_pattern"])
            axis.set_xlabel("Доля")
            axis.set_ylabel("Состояния органов в порядке algorithm_order")
            axis.set_title(f"Частые комбинации пяти РНМ — {source}")
            axis.legend()
            axis.grid(axis="x", alpha=0.25)
            figure.tight_layout()
            path = figure_dir / f"state_patterns__{source}.png"
            figure.savefig(path, dpi=170)
            plt.close(figure)
            outputs.append(path)
    return outputs


def build_review_tables(analysis_dir: Path) -> dict[str, Path]:
    analysis_dir = Path(analysis_dir)
    records = pd.read_csv(analysis_dir / "record_statistics.csv", low_memory=False)
    signals = pd.read_csv(analysis_dir / "signal_record_statistics.csv", low_memory=False)
    pair_records = pd.read_csv(analysis_dir / "pairwise_record_agreement.csv")
    duplicates = pd.read_csv(analysis_dir / "duplicate_groups.csv")
    pointwise_path = analysis_dir / "pairwise_pointwise_agreement.csv"
    patterns_path = analysis_dir / "algorithm_state_patterns.csv"
    temporal_path = analysis_dir / "teacher_temporal_statistics.csv"
    pointwise = pd.read_csv(pointwise_path) if pointwise_path.exists() else pd.DataFrame()
    patterns = (
        pd.read_csv(patterns_path, dtype={"state_pattern": str})
        if patterns_path.exists() else pd.DataFrame()
    )
    temporal = pd.read_csv(temporal_path) if temporal_path.exists() else pd.DataFrame()

    merged = records.merge(
        signals.drop(columns=["file_name", "f_adc", "voltage_basis"], errors="ignore"),
        on=["source", "record_id"],
        how="left",
        validate="one_to_one",
    )
    for column in ("missing_current_group", "missing_voltage_group"):
        if column in merged:
            merged[column] = merged[column].map(
                lambda value: str(value).strip().lower() in {"1", "true", "yes"}
                if isinstance(value, str) else bool(value)
            )
    merged["current_rms_physical_pu"] = merged["current_rms"] * 20.0
    current_edges = [-np.inf, 0.01, 0.05, 0.20, 0.50, 1.0, 2.0, np.inf]
    current_labels = ["<0.01", "0.01-0.05", "0.05-0.20", "0.20-0.50", "0.50-1.00", "1.00-2.00", ">=2.00"]
    merged["current_bin_physical_pu"] = pd.cut(
        merged["current_rms_physical_pu"], current_edges, labels=current_labels, right=False
    ).astype("object")
    merged.loc[merged["current_rms_physical_pu"].isna(), "current_bin_physical_pu"] = "missing"

    sampling_profiles = (
        merged.groupby(["source", "spp", "f_adc"], dropna=False, as_index=False)
        .agg(
            records=("record_id", "size"),
            duration_hours=("duration_sec", lambda value: value.sum() / 3600.0),
            median_duration_sec=("duration_sec", "median"),
            median_current_rms_physical_pu=("current_rms_physical_pu", "median"),
            missing_current_fraction=("missing_current_group", "mean"),
            missing_voltage_fraction=("missing_voltage_group", "mean"),
            mean_disagreement_fraction=("disagreement_fraction", "mean"),
            teacher_mean_coverage=(f"{ALGORITHMS[0]}__coverage_fraction", "mean"),
            teacher_mean_forward=(f"{ALGORITHMS[0]}__forward_fraction", "mean"),
            teacher_median_transitions=(f"{ALGORITHMS[0]}__transitions", "median"),
        )
    )
    sampling_profiles["fraction_of_source_records"] = sampling_profiles["records"] / sampling_profiles.groupby(
        "source"
    )["records"].transform("sum")

    weighting_rows: list[dict[str, object]] = []
    weighting_groups = [("all", merged)] + list(merged.groupby("source"))
    for source, group in weighting_groups:
        metric_specs: list[tuple[str, str]] = [
            ("disagreement_fraction", "n_windows"),
        ]
        for algorithm in ALGORITHMS:
            metric_specs.extend([
                (f"{algorithm}__coverage_fraction", "n_windows"),
                (f"{algorithm}__forward_fraction", f"{algorithm}__valid_windows"),
                (f"{algorithm}__transitions_per_second", "duration_sec"),
            ])
        for metric, point_weight in metric_specs:
            values = pd.to_numeric(group[metric], errors="coerce")
            include = values.notna()
            if metric.endswith("__forward_fraction"):
                include &= pd.to_numeric(group[point_weight], errors="coerce").fillna(0) > 0
            weighting_rows.append({
                "source": source,
                "metric": metric,
                "records_included": int(include.sum()),
                "record_mean": float(values[include].mean()) if include.any() else np.nan,
                "duration_weighted_mean": _weighted_mean(values[include], group.loc[include, "duration_sec"]),
                "point_weighted_mean": _weighted_mean(values[include], group.loc[include, point_weight]),
                "point_weight": point_weight,
            })
    weighting_sensitivity = pd.DataFrame(weighting_rows)

    temporal_profiles = pd.DataFrame()
    chatter_candidates = pd.DataFrame()
    if not temporal.empty:
        temporal_metrics = (
            "transitions", "state_entropy_bits", "transition_entropy_bits",
            "lag1_autocorrelation", "median_run_duration_sec",
            "p05_run_duration_sec", "p95_run_duration_sec",
            "short_runs_le_20ms", "short_runs_le_100ms",
            "chatter_returns_le_100ms", "max_switches_in_0_5s",
            "max_switches_in_1_0s", "first_transition_edge_distance_sec",
            "last_transition_edge_distance_sec",
        )
        temporal_rows: list[dict[str, object]] = []
        temporal_groups = [("all", temporal)] + list(temporal.groupby("source"))
        for source, group in temporal_groups:
            for metric in temporal_metrics:
                values = pd.to_numeric(group[metric], errors="coerce")
                row = {
                    "source": source,
                    "metric": metric,
                    "records_included": int(values.notna().sum()),
                }
                row.update(_quantiles(values))
                temporal_rows.append(row)
        temporal_profiles = pd.DataFrame(temporal_rows)
        chatter_candidates = temporal.merge(
            records[["source", "record_id", "file_name", "split", "input_sha256"]],
            on=["source", "record_id"],
            how="left",
            validate="one_to_one",
        ).sort_values(
            ["chatter_returns_le_100ms", "max_switches_in_0_5s", "transitions"],
            ascending=False,
        )

    current_rows: list[dict[str, object]] = []
    for (source, current_bin), group in merged.groupby(
        ["source", "current_bin_physical_pu"], observed=False, dropna=False
    ):
        row: dict[str, object] = {
            "source": source,
            "current_bin_physical_pu": current_bin,
            "records": len(group),
            "fraction_of_source": len(group) / int((merged["source"] == source).sum()),
            "median_current_rms_physical_pu": group["current_rms_physical_pu"].median(),
            "mean_disagreement_fraction": group["disagreement_fraction"].mean(),
            "switching_record_fraction": (group["total_transitions"] > 0).mean(),
        }
        for algorithm in ALGORITHMS:
            coverage = group[f"{algorithm}__coverage_fraction"] > 0
            row[f"{algorithm}__valid_records"] = int(coverage.sum())
            row[f"{algorithm}__mean_forward_fraction"] = group.loc[
                coverage, f"{algorithm}__forward_fraction"
            ].mean()
            row[f"{algorithm}__any_forward_fraction"] = (
                group.loc[coverage, f"{algorithm}__forward_fraction"] > 0
            ).mean()
        current_rows.append(row)
    current_profiles = pd.DataFrame(current_rows)

    split_rows: list[dict[str, object]] = []
    for (source, split), group in merged.groupby(["source", "split"], dropna=False):
        row = {
            "source": source,
            "split": split,
            "records": len(group),
            "median_duration_sec": group["duration_sec"].median(),
            "median_current_rms_physical_pu": group["current_rms_physical_pu"].median(),
            "mean_disagreement_fraction": group["disagreement_fraction"].mean(),
            "switching_record_fraction": (group["total_transitions"] > 0).mean(),
        }
        for algorithm in ALGORITHMS:
            coverage = group[f"{algorithm}__coverage_fraction"] > 0
            row[f"{algorithm}__mean_forward_fraction"] = group.loc[
                coverage, f"{algorithm}__forward_fraction"
            ].mean()
        split_rows.append(row)
    split_profiles = pd.DataFrame(split_rows)

    pair_rows: list[dict[str, object]] = []
    for (source, left, right), group in pair_records.groupby(["source", "left", "right"]):
        row = {"source": source, "left": left, "right": right, "records": len(group)}
        row.update({f"disagreement_{key}": value for key, value in _quantiles(group["disagreement_fraction"]).items()})
        for threshold in (0.001, 0.01, 0.10, 0.50):
            row[f"records_disagreement_gt_{threshold:g}"] = float(
                (group["disagreement_fraction"] > threshold).mean()
            )
        row.update({f"transition_{key}": value for key, value in _quantiles(group["disagreement_transitions"]).items()})
        pair_rows.append(row)
    pair_profiles = pd.DataFrame(pair_rows)

    duplicate_hashes = set(duplicates.loc[duplicates["records"] > 1, "input_sha256"].astype(str))
    merged["is_duplicate_record"] = merged["input_sha256"].astype(str).isin(duplicate_hashes)
    unique_records = merged.sort_values(["source", "record_id"]).drop_duplicates("input_sha256", keep="first")
    duplicate_rows: list[dict[str, object]] = []
    for source in ("open_ee", "french_rte", "all"):
        full = merged if source == "all" else merged[merged["source"] == source]
        unique = unique_records if source == "all" else unique_records[unique_records["source"] == source]
        row = {
            "source": source,
            "records_full": len(full),
            "records_unique_hash": len(unique),
            "mean_disagreement_full": full["disagreement_fraction"].mean(),
            "mean_disagreement_unique_hash": unique["disagreement_fraction"].mean(),
        }
        for algorithm in ALGORITHMS:
            column = f"{algorithm}__forward_fraction"
            full_valid = full[f"{algorithm}__coverage_fraction"] > 0
            unique_valid = unique[f"{algorithm}__coverage_fraction"] > 0
            row[f"{algorithm}__mean_forward_full"] = full.loc[full_valid, column].mean()
            row[f"{algorithm}__mean_forward_unique_hash"] = unique.loc[unique_valid, column].mean()
        duplicate_rows.append(row)
    duplicate_sensitivity = pd.DataFrame(duplicate_rows)

    quality_rows: list[dict[str, object]] = []
    for source, group in merged.groupby("source"):
        row: dict[str, object] = {
            "source": source,
            "records": len(group),
            "missing_current_records": int(group["missing_current_group"].sum()),
            "missing_voltage_records": int(group["missing_voltage_group"].sum()),
            "nonfinite_interest_score": int((~np.isfinite(group["interest_score"])).sum()),
            "forward_fraction_out_of_range": 0,
            "coverage_fraction_out_of_range": 0,
        }
        for algorithm in ALGORITHMS:
            forward = group[f"{algorithm}__forward_fraction"]
            coverage = group[f"{algorithm}__coverage_fraction"]
            row["forward_fraction_out_of_range"] += int(((forward < 0) | (forward > 1)).sum())
            row["coverage_fraction_out_of_range"] += int(((coverage < 0) | (coverage > 1)).sum())
        quality_rows.append(row)
    quality = pd.DataFrame(quality_rows)

    shift_rows: list[dict[str, object]] = []
    open_signals = merged[merged["source"] == "open_ee"]
    french_signals = merged[merged["source"] == "french_rte"]
    for metric in SIGNAL_METRICS:
        left = pd.to_numeric(open_signals[metric], errors="coerce").dropna().to_numpy()
        right = pd.to_numeric(french_signals[metric], errors="coerce").dropna().to_numpy()
        ks = ks_2samp(left, right, method="asymp")
        mw = mannwhitneyu(left, right, alternative="two-sided", method="asymptotic")
        pooled = np.concatenate([left, right])
        pooled_iqr = float(np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25))
        clip_low, clip_high = np.quantile(pooled, [0.01, 0.99])
        left_winsorized = np.clip(left, clip_low, clip_high)
        right_winsorized = np.clip(right, clip_low, clip_high)
        robust_iqr = float(
            np.quantile(np.concatenate([left_winsorized, right_winsorized]), 0.75)
            - np.quantile(np.concatenate([left_winsorized, right_winsorized]), 0.25)
        )
        shift_rows.append({
            "metric": metric,
            "records_open_ee": len(left),
            "records_french_rte": len(right),
            "median_open_ee": float(np.median(left)),
            "median_french_rte": float(np.median(right)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            # Положительный Cliff delta: случайное Open_EE значение чаще больше.
            "cliffs_delta_open_minus_french": float(2.0 * mw.statistic / (len(left) * len(right)) - 1.0),
            "wasserstein": float(wasserstein_distance(left, right)),
            "wasserstein_over_pooled_iqr": (
                float(wasserstein_distance(left, right) / pooled_iqr) if pooled_iqr > 0 else np.nan
            ),
            "winsorized_01_99_wasserstein_over_iqr": (
                float(wasserstein_distance(left_winsorized, right_winsorized) / robust_iqr)
                if robust_iqr > 0 else np.nan
            ),
        })
    signal_shift = pd.DataFrame(shift_rows)

    rng = np.random.default_rng(20260818)
    bootstrap_rows: list[dict[str, object]] = []
    for source, group in merged.groupby("source"):
        # Один голос на уникальный входной hash; это не позволяет французским
        # точным дубликатам искусственно сужать интервал.
        unique_hash = group.groupby("input_sha256", as_index=False).mean(numeric_only=True)
        for algorithm in ALGORITHMS:
            column = f"{algorithm}__forward_fraction"
            coverage = f"{algorithm}__coverage_fraction"
            values = unique_hash.loc[unique_hash[coverage] > 0, column].dropna().to_numpy()
            bootstrap = np.empty(1000, dtype=np.float64)
            for index in range(len(bootstrap)):
                bootstrap[index] = rng.choice(values, size=len(values), replace=True).mean()
            bootstrap_rows.append({
                "source": source,
                "algorithm_id": algorithm,
                "unique_input_hashes": len(values),
                "mean_record_forward_fraction": float(values.mean()),
                "median_record_forward_fraction": float(np.median(values)),
                "bootstrap95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap95_high": float(np.quantile(bootstrap, 0.975)),
            })
    algorithm_bootstrap = pd.DataFrame(bootstrap_rows)

    # Исходный fallback ``stable_consensus`` означает только отсутствие других
    # флагов и может скрывать постоянное расхождение при изменении coverage после
    # warm-up. Для ревью задаём категории непосредственно через наблюдаемые доли.
    review_category = np.select(
        [
            (merged["total_transitions"] == 0) & (merged["disagreement_fraction"] <= 0.01),
            (merged["total_transitions"] <= 1) & (merged["disagreement_fraction"] >= 0.50),
            (merged["total_transitions"] == 0) & (merged["disagreement_fraction"] > 0.01),
            (merged["total_transitions"] > 0) & (merged["disagreement_fraction"] >= 0.10),
            merged["total_transitions"] > 0,
        ],
        [
            "stable_agreement",
            "persistent_major_disagreement",
            "stable_minor_disagreement",
            "switching_with_disagreement",
            "switching_low_disagreement",
        ],
        default="unclassified",
    )
    merged["review_category"] = review_category
    category_profiles = (
        merged.groupby(["source", "review_category"], as_index=False)
        .agg(
            records=("record_id", "size"),
            median_disagreement=("disagreement_fraction", "median"),
            mean_disagreement=("disagreement_fraction", "mean"),
            median_total_transitions=("total_transitions", "median"),
            median_interest_score=("interest_score", "median"),
        )
    )
    category_profiles["fraction_of_source"] = category_profiles["records"] / category_profiles.groupby(
        "source"
    )["records"].transform("sum")
    persistent_candidates = merged.loc[
        merged["review_category"] == "persistent_major_disagreement",
        [
            "source", "record_id", "file_name", "split", "input_sha256",
            "duration_sec", "disagreement_fraction", "interest_score", "categories",
            "low_coverage", "current_rms_physical_pu",
        ] + [f"{algorithm}__forward_fraction" for algorithm in ALGORITHMS],
    ].sort_values(["disagreement_fraction", "duration_sec"], ascending=[False, False])

    outputs = {
        "current_profiles": analysis_dir / "review_current_profiles.csv",
        "split_profiles": analysis_dir / "review_split_profiles.csv",
        "pair_profiles": analysis_dir / "review_pairwise_record_profiles.csv",
        "duplicate_sensitivity": analysis_dir / "review_duplicate_sensitivity.csv",
        "quality": analysis_dir / "review_quality_checks.csv",
        "category_profiles": analysis_dir / "review_category_profiles.csv",
        "persistent_candidates": analysis_dir / "review_persistent_disagreement_candidates.csv",
        "signal_shift": analysis_dir / "review_signal_source_shift.csv",
        "algorithm_bootstrap": analysis_dir / "review_algorithm_record_bootstrap.csv",
        "sampling_profiles": analysis_dir / "review_sampling_profiles.csv",
        "weighting_sensitivity": analysis_dir / "review_weighting_sensitivity.csv",
        "metric_guide": analysis_dir / "STATISTICAL_METRICS_GUIDE.md",
        "temporal_profiles": analysis_dir / "review_temporal_profiles.csv",
        "chatter_candidates": analysis_dir / "review_chatter_candidates.csv",
    }
    current_profiles.to_csv(outputs["current_profiles"], index=False)
    split_profiles.to_csv(outputs["split_profiles"], index=False)
    pair_profiles.to_csv(outputs["pair_profiles"], index=False)
    duplicate_sensitivity.to_csv(outputs["duplicate_sensitivity"], index=False)
    quality.to_csv(outputs["quality"], index=False)
    category_profiles.to_csv(outputs["category_profiles"], index=False)
    persistent_candidates.to_csv(outputs["persistent_candidates"], index=False)
    signal_shift.to_csv(outputs["signal_shift"], index=False)
    algorithm_bootstrap.to_csv(outputs["algorithm_bootstrap"], index=False)
    sampling_profiles.to_csv(outputs["sampling_profiles"], index=False)
    weighting_sensitivity.to_csv(outputs["weighting_sensitivity"], index=False)
    _write_metric_guide(outputs["metric_guide"])
    temporal_profiles.to_csv(outputs["temporal_profiles"], index=False)
    chatter_candidates.to_csv(outputs["chatter_candidates"], index=False)
    for index, path in enumerate(_build_review_figures(analysis_dir, merged, pointwise, patterns), start=1):
        outputs[f"figure_{index}"] = path
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    args = parser.parse_args()
    for name, path in build_review_tables(args.analysis_dir).items():
        print(f"{name}: {path}")
    return 0


def run_manual() -> None:
    """Ручной запуск после завершения `analyze_pdr_dataset_study.py`."""
    ANALYSIS_DIR = DEFAULT_ANALYSIS_DIR
    for name, path in build_review_tables(ANALYSIS_DIR).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
