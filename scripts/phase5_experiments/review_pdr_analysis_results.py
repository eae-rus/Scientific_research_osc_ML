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
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "data/phase5/pdr_analysis_v6"
ALGORITHMS = (
    "adaptive_pdr_mir",
    "phase_pdr_basic",
    "pos_seq_pdr_basic",
    "phase_power_pdr_basic",
    "pos_seq_power_pdr_basic",
    "pdr_sivokobylenko_2pt",
    "pdr_sivokobylenko_5pt",
    "pdr_bmrz_q_assisted",
    "pdr_bavr072_crosspol",
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

MULTIVARIATE_FEATURES = (
    "duration_sec", "f_network", "spp", "f_adc",
    "current_rms_physical_pu", "voltage_rms", "current_to_voltage_rms",
    "current_phase_unbalance_cv", "voltage_phase_unbalance_cv",
    "current_rms_last_over_first", "current_crest_p99_over_rms",
    "disagreement_fraction", "localized_disagreement",
    "adaptive_pdr_mir__coverage_fraction", "adaptive_pdr_mir__forward_fraction",
    "adaptive_pdr_mir__transitions_per_second",
    "phase_pdr_basic__forward_fraction", "pos_seq_pdr_basic__forward_fraction",
    "phase_power_pdr_basic__forward_fraction",
    "pos_seq_power_pdr_basic__forward_fraction",
    "pdr_sivokobylenko_2pt__forward_fraction",
    "pdr_sivokobylenko_5pt__forward_fraction",
    "pdr_bmrz_q_assisted__forward_fraction",
    "pdr_bavr072_crosspol__forward_fraction",
)

SOURCE_SHIFT_FEATURES = (
    "duration_sec", "current_rms_physical_pu", "voltage_rms",
    "current_to_voltage_rms", "current_phase_unbalance_cv",
    "voltage_phase_unbalance_cv", "current_rms_last_over_first",
    "current_crest_p99_over_rms",
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

## Комбинации органов

`state_pattern` содержит биты в порядке из `algorithm_order`. Например, в v6
`101000000` означает FORWARD у первого и третьего из девяти органов. Сравнение `point_fraction` с
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
            axis.set_title(f"Частые комбинации {len(ALGORITHMS)} РНМ — {source}")
            axis.legend()
            axis.grid(axis="x", alpha=0.25)
            figure.tight_layout()
            path = figure_dir / f"state_patterns__{source}.png"
            figure.savefig(path, dpi=170)
            plt.close(figure)
            outputs.append(path)
    return outputs


def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR внутри одной заранее заданной семьи тестов."""

    values = np.asarray(pvalues, dtype=np.float64)
    result = np.full_like(values, np.nan)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if not finite_values.size:
        return result
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    finite_result = np.empty_like(adjusted)
    finite_result[order] = np.minimum(adjusted, 1.0)
    result[finite_mask] = finite_result
    return result


def _research_matrix(frame: pd.DataFrame, columns: tuple[str, ...]):
    """Median-impute + robust-scale с логарифмом положительных heavy-tail полей."""

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler

    raw = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").copy()
    log_columns = {
        "duration_sec", "spp", "f_adc", "current_rms_physical_pu", "voltage_rms",
        "current_to_voltage_rms", "current_phase_unbalance_cv",
        "voltage_phase_unbalance_cv", "current_rms_last_over_first",
        "current_crest_p99_over_rms", "adaptive_pdr_mir__transitions_per_second",
    }
    for column in columns:
        if column in log_columns:
            raw[column] = np.log10(np.clip(raw[column], 0.0, None) + 1e-6)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    return scaler.fit_transform(imputer.fit_transform(raw)), raw


def _build_multivariate_research(
    analysis_dir: Path,
    eligible: pd.DataFrame,
    temporal: pd.DataFrame,
) -> dict[str, Path]:
    """Продолжение слоёв E/F/H: shift, PCA, устойчивость и аномалии."""

    from itertools import combinations
    from scipy.stats import spearmanr
    from sklearn.calibration import calibration_curve
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        adjusted_rand_score, balanced_accuracy_score, brier_score_loss,
        calinski_harabasz_score, davies_bouldin_score, log_loss,
        roc_auc_score, silhouette_score,
    )
    from sklearn.model_selection import StratifiedKFold

    outputs: dict[str, Path] = {}
    feature_columns = tuple(column for column in MULTIVARIATE_FEATURES if column in eligible)
    matrix, _ = _research_matrix(eligible, feature_columns)

    # PCA — воспроизводимый линейный baseline, а не доказательство кластеров.
    pca = PCA(n_components=min(10, matrix.shape[1]), random_state=20260820)
    coordinates = pca.fit_transform(matrix)
    coordinate_rows = eligible[["source", "record_id", "input_sha256"]].copy()
    for index in range(coordinates.shape[1]):
        coordinate_rows[f"PC{index + 1}"] = coordinates[:, index]
    coordinate_rows["pdr_structurally_eligible"] = True
    pca_path = analysis_dir / "research_pca_coordinates.csv"
    coordinate_rows.to_csv(pca_path, index=False)
    outputs["pca_coordinates"] = pca_path
    variance_path = analysis_dir / "research_pca_explained_variance.csv"
    pd.DataFrame({
        "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    }).to_csv(variance_path, index=False)
    outputs["pca_variance"] = variance_path

    loading_path = analysis_dir / "research_pca_loadings.csv"
    pd.DataFrame(
        pca.components_,
        columns=feature_columns,
        index=[f"PC{index + 1}" for index in range(pca.components_.shape[0])],
    ).rename_axis("component").reset_index().to_csv(loading_path, index=False)
    outputs["pca_loadings"] = loading_path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(20260820)
    sample_indices = rng.choice(len(eligible), size=min(12000, len(eligible)), replace=False)
    figure, axis = plt.subplots(figsize=(8, 6))
    for source in sorted(eligible["source"].unique()):
        mask = eligible.iloc[sample_indices]["source"].to_numpy() == source
        axis.scatter(coordinates[sample_indices[mask], 0], coordinates[sample_indices[mask], 1],
                     s=5, alpha=0.28, label=source, rasterized=True)
    axis.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    axis.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    axis.set_title("PCA применимых PDR-записей (robust-scaled record features)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    pca_figure = analysis_dir / "review_figures" / "research_pca_sources.png"
    pca_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pca_figure, dpi=180)
    plt.close(figure)
    outputs["pca_figure"] = pca_figure

    # Устойчивость KMeans проверяется отдельно внутри источников.
    diagnostics: list[dict[str, object]] = []
    assignments: list[pd.DataFrame] = []
    profiles: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for source, source_frame in eligible.groupby("source"):
        source_matrix, _ = _research_matrix(source_frame, feature_columns)
        evaluation_indices = rng.choice(
            len(source_frame), size=min(5000, len(source_frame)), replace=False
        )
        source_results: list[dict[str, object]] = []
        labels_by_k: dict[int, list[np.ndarray]] = {}
        for k in range(2, 13):
            seed_labels = []
            seed_silhouettes = []
            for seed in range(5):
                labels = KMeans(n_clusters=k, n_init=1, random_state=20260820 + seed).fit_predict(
                    source_matrix
                )
                seed_labels.append(labels)
                seed_silhouettes.append(silhouette_score(
                    source_matrix[evaluation_indices], labels[evaluation_indices]
                ))
            ari_values = [
                adjusted_rand_score(seed_labels[left], seed_labels[right])
                for left, right in combinations(range(len(seed_labels)), 2)
            ]
            row = {
                "source": source,
                "k": k,
                "silhouette_mean": float(np.mean(seed_silhouettes)),
                "silhouette_std": float(np.std(seed_silhouettes)),
                "seed_stability_ari_mean": float(np.mean(ari_values)),
                "seed_stability_ari_min": float(np.min(ari_values)),
                "davies_bouldin": float(davies_bouldin_score(source_matrix, seed_labels[0])),
                "calinski_harabasz": float(calinski_harabasz_score(source_matrix, seed_labels[0])),
            }
            diagnostics.append(row)
            source_results.append(row)
            labels_by_k[k] = seed_labels
        stable = [row for row in source_results if row["seed_stability_ari_mean"] >= 0.80]
        if not stable:
            selections.append({
                "source": source,
                "selection_status": "no_stable_solution",
                "selected_k": pd.NA,
                "silhouette_mean": pd.NA,
                "seed_stability_ari_mean": pd.NA,
                "stability_threshold": 0.80,
            })
            continue
        selected = max(stable, key=lambda row: row["silhouette_mean"])
        selected_k = int(selected["k"])
        selections.append({
            "source": source,
            "selection_status": "stable_solution_selected",
            "selected_k": selected_k,
            "silhouette_mean": selected["silhouette_mean"],
            "seed_stability_ari_mean": selected["seed_stability_ari_mean"],
            "stability_threshold": 0.80,
        })
        selected_labels = labels_by_k[selected_k][0]
        assignment = source_frame[["source", "record_id", "input_sha256"]].copy()
        assignment["selected_k"] = selected_k
        assignment["cluster"] = selected_labels
        assignments.append(assignment)
        for cluster in range(selected_k):
            mask = selected_labels == cluster
            row = {
                "source": source,
                "selected_k": selected_k,
                "cluster": cluster,
                "records": int(mask.sum()),
                "fraction": float(mask.mean()),
                "silhouette_mean": selected["silhouette_mean"],
                "seed_stability_ari_mean": selected["seed_stability_ari_mean"],
            }
            for column in feature_columns:
                row[f"median__{column}"] = pd.to_numeric(
                    source_frame.loc[mask, column], errors="coerce"
                ).median()
            profiles.append(row)
    cluster_diagnostics_path = analysis_dir / "research_cluster_stability.csv"
    pd.DataFrame(diagnostics).to_csv(cluster_diagnostics_path, index=False)
    outputs["cluster_stability"] = cluster_diagnostics_path
    cluster_selection_path = analysis_dir / "research_cluster_selection.csv"
    pd.DataFrame(selections).to_csv(cluster_selection_path, index=False)
    outputs["cluster_selection"] = cluster_selection_path
    cluster_assignment_path = analysis_dir / "research_cluster_assignments.csv"
    pd.concat(assignments, ignore_index=True).to_csv(cluster_assignment_path, index=False)
    outputs["cluster_assignments"] = cluster_assignment_path
    cluster_profile_path = analysis_dir / "research_stable_cluster_profiles.csv"
    pd.DataFrame(profiles).to_csv(cluster_profile_path, index=False)
    outputs["cluster_profiles"] = cluster_profile_path

    # Isolation Forest — только рейтинг кандидатов, не физический класс.
    anomaly_rows: list[pd.DataFrame] = []
    for source, source_frame in eligible.groupby("source"):
        source_matrix, _ = _research_matrix(source_frame, feature_columns)
        model = IsolationForest(
            n_estimators=200, max_samples=min(4096, len(source_frame)),
            contamination="auto", random_state=20260820, n_jobs=-1,
        ).fit(source_matrix)
        result = source_frame[[
            "source", "record_id", "file_name", "input_sha256", "split",
            "current_rms_physical_pu", "voltage_rms", "disagreement_fraction",
            "total_transitions",
        ]].copy()
        result["anomaly_score"] = -model.score_samples(source_matrix)
        result = result.sort_values("anomaly_score", ascending=False)
        result["anomaly_rank_within_source"] = np.arange(1, len(result) + 1)
        anomaly_rows.append(result)
    anomaly_path = analysis_dir / "research_anomaly_candidates.csv"
    pd.concat(anomaly_rows, ignore_index=True).to_csv(anomaly_path, index=False)
    outputs["anomaly_candidates"] = anomaly_path

    # Classifier two-sample test: насколько источник восстанавливается из
    # инженерных признаков. Считаем на уникальных hashes, чтобы не учить копии.
    source_frame = eligible.sort_values(["source", "record_id"]).drop_duplicates(
        "input_sha256", keep="first"
    )
    classifier_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    for feature_set, columns in (
        ("physical_only", SOURCE_SHIFT_FEATURES),
        ("physical_plus_acquisition", SOURCE_SHIFT_FEATURES + ("f_network", "spp", "f_adc")),
    ):
        columns = tuple(column for column in columns if column in source_frame)
        x, _ = _research_matrix(source_frame, columns)
        y = (source_frame["source"].to_numpy() == "french_rte").astype(np.int8)
        predicted = np.zeros(len(y), dtype=np.float64)
        coefficient_rows = []
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260820)
        for fold, (train, test) in enumerate(splitter.split(x, y), start=1):
            model = LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=20260820 + fold
            ).fit(x[train], y[train])
            predicted[test] = model.predict_proba(x[test])[:, 1]
            coefficient_rows.append(model.coef_[0])
        classifier_rows.append({
            "feature_set": feature_set,
            "unique_input_hashes": len(y),
            "french_fraction": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, predicted)),
            "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, predicted >= 0.5)),
            "log_loss": float(log_loss(y, predicted)),
            "brier_score": float(brier_score_loss(y, predicted)),
        })
        prob_true, prob_pred = calibration_curve(y, predicted, n_bins=10, strategy="quantile")
        for bin_index, (observed, predicted_mean) in enumerate(zip(prob_true, prob_pred), start=1):
            calibration_rows.append({
                "feature_set": feature_set, "quantile_bin": bin_index,
                "mean_predicted_french_probability": predicted_mean,
                "observed_french_fraction": observed,
            })
        coefficients = np.asarray(coefficient_rows)
        for column_index, column in enumerate(columns):
            importance_rows.append({
                "feature_set": feature_set,
                "feature": column,
                "mean_standardized_logit_coefficient": float(coefficients[:, column_index].mean()),
                "mean_abs_standardized_logit_coefficient": float(
                    np.abs(coefficients[:, column_index]).mean()
                ),
                "fold_std": float(coefficients[:, column_index].std()),
            })
    classifier_path = analysis_dir / "research_source_classifier.csv"
    pd.DataFrame(classifier_rows).to_csv(classifier_path, index=False)
    outputs["source_classifier"] = classifier_path
    calibration_path = analysis_dir / "research_source_classifier_calibration.csv"
    pd.DataFrame(calibration_rows).to_csv(calibration_path, index=False)
    outputs["source_calibration"] = calibration_path
    importance_path = analysis_dir / "research_source_classifier_coefficients.csv"
    pd.DataFrame(importance_rows).sort_values(
        ["feature_set", "mean_abs_standardized_logit_coefficient"], ascending=[True, False]
    ).to_csv(importance_path, index=False)
    outputs["source_coefficients"] = importance_path

    # Spearman + BH-FDR по независимой единице "уникальная осциллограмма".
    correlation_frame = eligible.merge(
        temporal[["source", "record_id", "state_entropy_bits"]],
        on=["source", "record_id"], how="left", validate="one_to_one",
    ).sort_values(["source", "record_id"]).drop_duplicates("input_sha256", keep="first")
    predictors = tuple(column for column in SOURCE_SHIFT_FEATURES + ("spp", "f_adc") if column in correlation_frame)
    targets = (
        "disagreement_fraction", "adaptive_pdr_mir__transitions_per_second",
        "state_entropy_bits",
    )
    correlation_rows: list[dict[str, object]] = []
    for (source, target), group in [
        ((source, target), group)
        for source, source_group in correlation_frame.groupby("source")
        for target in targets
        for group in [source_group]
    ]:
        family_start = len(correlation_rows)
        pvalues = []
        for predictor in predictors:
            pair = group[[predictor, target]].apply(pd.to_numeric, errors="coerce").dropna()
            rho, pvalue = spearmanr(pair[predictor], pair[target])
            correlation_rows.append({
                "source": source, "target": target, "predictor": predictor,
                "records": len(pair), "spearman_rho": float(rho),
                "pvalue": float(pvalue),
            })
            pvalues.append(float(pvalue))
        adjusted = _bh_adjust(np.asarray(pvalues))
        for offset, value in enumerate(adjusted):
            correlation_rows[family_start + offset]["fdr_bh_qvalue"] = float(value)
    correlation_path = analysis_dir / "research_spearman_fdr.csv"
    pd.DataFrame(correlation_rows).to_csv(correlation_path, index=False)
    outputs["spearman_fdr"] = correlation_path
    return outputs


def _write_data_dictionary(path: Path, frame: pd.DataFrame) -> None:
    """Сформировать проверяемый словарь полей основной record-level таблицы."""

    known = {
        "source": ("Источник датасета", "category"),
        "record_id": ("Идентификатор записи внутри источника", "id"),
        "input_sha256": ("SHA-256 нормализованного входа; ключ точных дублей", "hash"),
        "duration_sec": ("Физическая длительность записи", "s"),
        "n_windows": ("Число causal-точек РНМ в записи", "points"),
        "f_network": ("Номинальная частота сети", "Hz"),
        "f_adc": ("Частота дискретизации", "Hz"),
        "spp": ("Отсчётов на период сети", "samples/period"),
        "current_rms": ("Медиана фазных waveform RMS во внутреннем масштабе", "internal p.u."),
        "current_rms_physical_pu": ("current_rms после восстановления current_reserve=20", "I/Iном"),
        "voltage_rms": ("Медиана фазных waveform RMS во внутреннем масштабе", "internal p.u."),
        "disagreement_fraction": ("Доля общих валидных точек, где не все РНМ совпали", "fraction"),
        "pdr_structurally_eligible": ("Есть минимум 2I и 2U для восстановления", "boolean"),
        "current_bin_physical_pu": ("Диапазон waveform RMS тока в физических номиналах", "I/Iном"),
        "total_transitions": ("Сумма переходов всех рассчитываемых РНМ", "count"),
        "interest_score": ("Эвристический рейтинг динамики/расхождения для навигации", "dimensionless"),
    }
    suffixes = {
        "valid_windows": ("Число точек с решением данного РНМ", "points"),
        "coverage_fraction": ("Доля точек не UNLABELED у данного РНМ", "fraction"),
        "forward_fraction": ("Доля FORWARD среди валидных точек данного РНМ", "fraction"),
        "transitions": ("Число смен 0↔1 данного РНМ", "count"),
        "transitions_per_second": ("Частота смен 0↔1 данного РНМ", "1/s"),
        "short_run_fraction": ("Доля коротких серий решений данного РНМ", "fraction"),
        "near_boundary_fraction": ("Доля валидных точек около нулевого margin", "fraction"),
        "mean_confidence": ("Средняя эвристическая confidence данного РНМ", "fraction"),
    }
    lines = [
        "# Data dictionary record-level PDR-анализа", "",
        "Основная независимая единица — осциллограмма. `internal p.u.` нельзя ",
        "смешивать с физическими номиналами без явно указанного scale profile.", "",
        "| Поле | Тип | Единица | Определение |", "|---|---|---|---|",
    ]
    for column in frame.columns:
        definition, unit = known.get(column, ("Производное или исходное поле record audit", "see source"))
        if "__" in column:
            algorithm, suffix = column.split("__", 1)
            if suffix in suffixes:
                suffix_definition, unit = suffixes[suffix]
                definition = f"{suffix_definition}; algorithm_id={algorithm}"
        lines.append(f"| `{column}` | `{frame[column].dtype}` | {unit} | {definition} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_review_tables(analysis_dir: Path) -> dict[str, Path]:
    analysis_dir = Path(analysis_dir)
    records = pd.read_csv(analysis_dir / "record_statistics.csv", low_memory=False)
    signals = pd.read_csv(analysis_dir / "signal_record_statistics.csv", low_memory=False)
    pair_records = pd.read_csv(analysis_dir / "pairwise_record_agreement.csv")
    duplicates = pd.read_csv(analysis_dir / "duplicate_groups.csv")
    pointwise_path = analysis_dir / "pairwise_pointwise_agreement.csv"
    patterns_path = analysis_dir / "algorithm_state_patterns.csv"
    temporal_path = analysis_dir / "teacher_temporal_statistics.csv"
    eligibility_path = analysis_dir / "record_signal_eligibility.csv"
    pointwise = pd.read_csv(pointwise_path) if pointwise_path.exists() else pd.DataFrame()
    patterns = (
        pd.read_csv(patterns_path, dtype={"state_pattern": str})
        if patterns_path.exists() else pd.DataFrame()
    )
    temporal = pd.read_csv(temporal_path) if temporal_path.exists() else pd.DataFrame()
    eligibility = pd.read_csv(eligibility_path) if eligibility_path.exists() else pd.DataFrame()

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
    if not eligibility.empty:
        eligibility["pdr_structurally_eligible"] = eligibility["pdr_structurally_eligible"].map(
            lambda value: str(value).strip().lower() in {"1", "true", "yes"}
            if isinstance(value, str) else bool(value)
        )
        merged = merged.merge(
            eligibility,
            on=["source", "record_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        # Совместимость со старыми результатами анализа. Для v4 и новее
        # следует сначала выполнить MODE="eligibility".
        merged["pdr_structurally_eligible"] = ~(
            merged["missing_current_group"] | merged["missing_voltage_group"]
        )
    merged["pdr_structurally_eligible"] = merged["pdr_structurally_eligible"].fillna(False)
    merged["current_rms_physical_pu"] = merged["current_rms"] * 20.0
    current_edges = [-np.inf, 0.01, 0.05, 0.20, 0.50, 1.0, 2.0, np.inf]
    current_labels = ["<0.01", "0.01-0.05", "0.05-0.20", "0.20-0.50", "0.50-1.00", "1.00-2.00", ">=2.00"]
    merged["current_bin_physical_pu"] = pd.cut(
        merged["current_rms_physical_pu"], current_edges, labels=current_labels, right=False
    ).astype("object")
    merged.loc[merged["current_rms_physical_pu"].isna(), "current_bin_physical_pu"] = "missing"
    eligible = merged[merged["pdr_structurally_eligible"]].copy()

    population_rows: list[dict[str, object]] = []
    for population, population_frame in (("all_input", merged), ("pdr_eligible_2i2u", eligible)):
        for source, group in [("all", population_frame), *list(population_frame.groupby("source"))]:
            row: dict[str, object] = {
                "population": population,
                "source": source,
                "records": len(group),
                "duration_hours": group["duration_sec"].sum() / 3600.0,
                "windows": group["n_windows"].sum(),
                "mean_disagreement_fraction": group["disagreement_fraction"].mean(),
            }
            for algorithm in ALGORITHMS:
                row[f"{algorithm}__record_mean_coverage"] = group[
                    f"{algorithm}__coverage_fraction"
                ].mean()
            population_rows.append(row)
    population_summary = pd.DataFrame(population_rows)

    sampling_profiles = (
        eligible.groupby(["source", "f_network", "spp", "f_adc"], dropna=False, as_index=False)
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
            teacher_p90_transitions=(f"{ALGORITHMS[0]}__transitions", lambda value: value.quantile(0.90)),
            teacher_p95_transitions=(f"{ALGORITHMS[0]}__transitions", lambda value: value.quantile(0.95)),
        )
    )
    sampling_profiles["fraction_of_source_records"] = sampling_profiles["records"] / sampling_profiles.groupby(
        "source"
    )["records"].transform("sum")

    weighting_rows: list[dict[str, object]] = []
    weighting_groups = [
        (population, source, group)
        for population, frame in (("all_input", merged), ("pdr_eligible_2i2u", eligible))
        for source, group in [("all", frame), *list(frame.groupby("source"))]
    ]
    for population, source, group in weighting_groups:
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
                "population": population,
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
    temporal_shortlist = pd.DataFrame()
    if not temporal.empty:
        temporal = temporal.merge(
            eligible[["source", "record_id"]],
            on=["source", "record_id"],
            how="inner",
            validate="one_to_one",
        )
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
        strata_rows: list[pd.DataFrame] = []
        for source, group in chatter_candidates.groupby("source"):
            transition_p99 = group["transitions"].quantile(0.99)
            burst_p99 = group["max_switches_in_0_5s"].quantile(0.99)
            stratum_masks = {
                "extreme_global_chatter": group["transitions"] >= transition_p99,
                "extreme_local_burst": group["max_switches_in_0_5s"] >= burst_p99,
                "moderate_10_100_transitions": group["transitions"].between(10, 100),
                "rapid_returns": group["chatter_returns_le_100ms"] > 0,
            }
            for stratum, mask in stratum_masks.items():
                selected = group.loc[mask].copy()
                selected["candidate_stratum"] = stratum
                selected["stratum_rank"] = np.arange(1, len(selected) + 1)
                strata_rows.append(selected)
        if strata_rows:
            temporal_shortlist = pd.concat(strata_rows, ignore_index=True)
            temporal_shortlist = temporal_shortlist[
                temporal_shortlist["stratum_rank"] <= 20
            ].sort_values(["source", "candidate_stratum", "stratum_rank"])

    current_rows: list[dict[str, object]] = []
    for (source, current_bin), group in eligible.groupby(
        ["source", "current_bin_physical_pu"], observed=False, dropna=False
    ):
        row: dict[str, object] = {
            "source": source,
            "current_bin_physical_pu": current_bin,
            "records": len(group),
            "fraction_of_source": len(group) / int((eligible["source"] == source).sum()),
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
    for (source, split), group in eligible.groupby(["source", "split"], dropna=False):
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
    pair_records = pair_records.merge(
        eligible[["source", "record_id"]],
        on=["source", "record_id"],
        how="inner",
        validate="many_to_one",
    )
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
    unique_records = eligible.sort_values(["source", "record_id"]).drop_duplicates("input_sha256", keep="first")
    duplicate_rows: list[dict[str, object]] = []
    for source in ("open_ee", "french_rte", "all"):
        full = eligible if source == "all" else eligible[eligible["source"] == source]
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
            "structurally_ineligible_records": int((~group["pdr_structurally_eligible"]).sum()),
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
    open_signals = eligible[eligible["source"] == "open_ee"]
    french_signals = eligible[eligible["source"] == "french_rte"]
    log_metrics = {
        "current_rms", "voltage_rms", "current_to_voltage_rms",
        "current_rms_last_over_first", "current_crest_p99_over_rms",
    }
    shift_specs = [(metric, "raw") for metric in SIGNAL_METRICS] + [
        (metric, "log10_positive") for metric in SIGNAL_METRICS if metric in log_metrics
    ]
    for metric, transform in shift_specs:
        left = pd.to_numeric(open_signals[metric], errors="coerce").dropna().to_numpy()
        right = pd.to_numeric(french_signals[metric], errors="coerce").dropna().to_numpy()
        if transform == "log10_positive":
            left = np.log10(left[left > 0])
            right = np.log10(right[right > 0])
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
            "transform": transform,
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
    for source, group in eligible.groupby("source"):
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
    eligible = merged[merged["pdr_structurally_eligible"]].copy()
    category_profiles = (
        eligible.groupby(["source", "review_category"], as_index=False)
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
    persistent_candidates = eligible.loc[
        eligible["review_category"] == "persistent_major_disagreement",
        [
            "source", "record_id", "file_name", "split", "input_sha256",
            "duration_sec", "disagreement_fraction", "interest_score", "categories",
            "low_coverage", "current_rms_physical_pu",
        ] + [f"{algorithm}__forward_fraction" for algorithm in ALGORITHMS],
    ].sort_values(["disagreement_fraction", "duration_sec"], ascending=[False, False])

    research_outputs = _build_multivariate_research(analysis_dir, eligible, temporal)
    data_dictionary_path = analysis_dir / "PDR_RECORD_DATA_DICTIONARY.md"
    _write_data_dictionary(data_dictionary_path, merged)

    outputs = {
        "population_summary": analysis_dir / "review_population_summary.csv",
        "data_dictionary": data_dictionary_path,
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
        "temporal_shortlist": analysis_dir / "review_temporal_candidate_shortlist.csv",
    }
    outputs.update(research_outputs)
    population_summary.to_csv(outputs["population_summary"], index=False)
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
    temporal_shortlist.to_csv(outputs["temporal_shortlist"], index=False)
    for index, path in enumerate(_build_review_figures(analysis_dir, eligible, pointwise, patterns), start=1):
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
