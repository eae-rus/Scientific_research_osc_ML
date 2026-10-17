"""Дополнительный воспроизводимый аудит полной Phase 5 PDR-разметки.

F5: инженерные временные метрики сохранённых плотных ответов RTDS и
экспертной галереи. Нейросети и Фурье повторно не рассчитываются.
Профиль legacy объединяет готовые CSV прежнего массового аудита.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
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


def build_engineering_review(*, scope: str, output_root: Path, gallery_root: Path,
                             rtds_root: Path, config=None, resume: bool = True,
                             max_records: int | None = None, bootstrap_repeats: int = 2000) -> dict:
    """CPU-аудит плотных сохранённых ответов: без запуска нейросетей/Фурье.

    Возобновление по целой записи; в кэше находятся только метрики и события.
    Экспертная часть — исключительно validation, не train/holdout.
    """
    import json
    import csv
    import hashlib
    from dataclasses import asdict
    from osc_tools.pdr.temporal_metrics import TemporalConfig, temporal_metrics
    from osc_tools.pdr.expert_labels import read_comtrade_1999_ascii, _read_text
    from osc_tools.pdr.study import PDRStudyLabelStore
    from scripts.phase5_experiments.run_pdr_dataset_study import _atomic_write_json
    from scripts.phase5_experiments.progress import ProgressReporter

    if scope not in ("rtds", "expert") or (max_records is not None and max_records < 1) or bootstrap_repeats < 0:
        raise ValueError("scope=rtds/expert; MAX_RECORDS=None или положительное число")
    cfg = config or TemporalConfig()
    def digest(path: Path) -> str:
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    files = (sorted(rtds_root.rglob("*.cfg")) if scope == "rtds" else
             sorted((gallery_root / "validation").rglob("*.json")))
    if not files:
        raise FileNotFoundError("Нет входных записей: проверьте путь/завершение галереи")
    total_available = len(files)
    files = files[:max_records] if max_records else files
    # Пробный запуск не затирает полный отчёт и его прогресс.
    output_root = output_root / scope / ("smoke" if max_records else "full")
    output_root.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(f"Инженерные метрики: {scope}", len(files), unit="запись")
    code = {"metrics": digest(PROJECT_ROOT / "osc_tools/pdr/temporal_metrics.py"), "runner": digest(Path(__file__))}
    manual = {}
    split_records = {}
    split_hash = None
    if scope == "expert":
        split_file = PROJECT_ROOT / "data/phase5/pdr_expert_labels_v1/records.csv"
        split_hash = digest(split_file)
        with split_file.open(encoding="utf-8-sig", newline="") as stream:
            split_records = {(x["source"], int(x["record_id"])): x for x in csv.DictReader(stream) if x["split"] == "validation"}
        available = {(p.parent.name, int(p.stem.rsplit("_",1)[1])) for p in
                     (gallery_root / "validation").rglob("*.json")}
        if available != set(split_records):
            raise ValueError("Галерея validation не соответствует текущему split: обновите галерею")
        for p in (PROJECT_ROOT / "data/phase5/pdr_manual_labels").rglob("*.cfg"):
            manual.setdefault(p.stem, []).append(p)
    stores, manifests, rows, checkpoint_hashes = {}, {}, [], {}
    model_set = None
    try:
        for number, path in enumerate(files, 1):
            meta_path = path.with_suffix(".json") if scope == "rtds" else path
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if scope == "expert":
                if meta.get("split") != "validation" or meta.get("status") not in ("completed", "invalid_quality"):
                    raise ValueError(f"Недопустимая часть/статус: {path}")
                if meta.get("inference_stride") != 1 or meta.get("startup_policy") != "early":
                    raise ValueError("Нужна плотная галерея: stride=1, startup_policy=early")
                if len(manual.get(path.stem, [])) != 1:
                    raise ValueError(f"Неоднозначный/отсутствующий экспертный COMTRADE: {path.stem}")
                cfg_path = manual[path.stem][0]
                source = path.parent.name
                record_id = int(path.stem.rsplit("_", 1)[1])
                if source not in stores:
                    root = PROJECT_ROOT / "data/phase5/pdr_labels_v6" / source
                    manifest = root / "manifest.json"
                    manifests[source] = digest(manifest)
                    stores[source] = {key: PDRStudyLabelStore(root, algorithm_id=key) for key in meta["algorithm_ids"]}
                if manifests[source] != meta["automatic_manifest_sha256"]:
                    raise ValueError(f"Изменилась формульная разметка: обновите галерею {path}")
                if digest(cfg_path) != meta["cfg_sha256"] or digest(cfg_path.with_suffix(".dat")) != meta["dat_sha256"]:
                    raise ValueError(f"Изменена экспертная разметка: обновите галерею {path}")
                for model in meta["models"].values():
                    weight = Path(model["checkpoint"])
                    if str(weight) not in checkpoint_hashes:
                        checkpoint_hashes[str(weight)] = digest(weight)
                    if checkpoint_hashes[str(weight)] != model["checkpoint_sha256"]:
                        raise ValueError(f"Изменены веса: обновите галерею {weight}")
                    if digest(weight.parent / "config.json") != model["config_sha256"]:
                        raise ValueError(f"Изменена конфигурация: обновите галерею {weight}")
                selected = tuple(sorted(meta["models"]))
                if model_set is not None and model_set != selected:
                    raise ValueError("Галерея не закончена: различается состав моделей")
                model_set = selected
                inputs = {"meta": digest(path), "npz": digest(path.with_suffix(".npz")),
                          "cfg": meta["cfg_sha256"], "dat": meta["dat_sha256"], "manifest": manifests[source], "split": split_hash}
            else:
                cfg_path, source = path, "rtds"
                selected = tuple(sorted(set(meta["fingerprint"]["algorithm_params"]) |
                    set(meta["fingerprint"]["models"]) | set(meta["additional_algorithm_ids"])))
                if model_set is not None and model_set != selected:
                    raise ValueError("RTDS не закончен: различается состав алгоритмов")
                model_set = selected
                if meta["fingerprint"].get("stride") != 1 or meta["augmentation_fingerprint"].get("startup") != "early":
                    raise ValueError("RTDS должен быть рассчитан плотно с ранним стартом")
                for suffix in (".cfg", ".dat"):
                    if digest(path.with_suffix(suffix)) != meta.get("output_hashes", {}).get(suffix):
                        raise ValueError(f"Изменена дополненная пара RTDS: {path}")
                reference = Path(meta["expert_reference_path"])
                for suffix, key in ((".cfg", "cfg"), (".dat", "dat")):
                    if digest(reference.with_suffix(suffix)) != meta["augmentation_fingerprint"][key]:
                        raise ValueError(f"Эксперт RTDS изменён: сначала обновите дополненные COMTRADE {path}")
                inputs = {"meta": digest(meta_path), **meta["output_hashes"]}
            request = json.loads(json.dumps({"inputs": inputs, "code": code, "config": asdict(cfg)}))
            cache = output_root / "records" / f"{source}__{path.stem}.json"
            previous = json.loads(cache.read_text(encoding="utf-8")) if resume and cache.exists() else {}
            if previous.get("request") == request:
                part = previous["rows"]
                print(f"[Кэш метрик] {path.stem}", flush=True)
            else:
                rec = read_comtrade_1999_ascii(cfg_path)
                cfg_rows = list(csv.reader(_read_text(cfg_path).splitlines()))
                frequency = float(cfg_rows[2+int(cfg_rows[1][0])][0])
                if not np.isfinite(frequency) or frequency <= 0:
                    raise ValueError(f"Неверная частота сети: {cfg_path}")
                first_fourier = min(rec.n_samples, round(rec.sample_rate_hz/frequency)-1)
                if not np.allclose(rec.timestamps_us, np.arange(rec.n_samples)*1e6/rec.sample_rate_hz, atol=1, rtol=0):
                    raise ValueError(f"Нерегулярная временная сетка: {cfg_path}")
                part = []
                def add(algorithm, section, truth, pred, first, cluster):
                    part.append({"source": source, "record": path.stem, "section": section,
                        "cluster": cluster, "algorithm": algorithm,
                        "metrics": temporal_metrics(truth, pred, rec.sample_rate_hz, config=cfg, startup_samples=first)})
                if scope == "rtds":
                    allowed = set(meta["fingerprint"]["algorithm_params"]) | set(meta["fingerprint"]["models"]) | set(meta["additional_algorithm_ids"])
                    for section in (1,2):
                        prefix = f"S{section}__"
                        truth = np.where(rec.digital[prefix+"expert__VALID"], rec.digital[prefix+"expert__FWD"], -1).astype(np.int8)
                        for algorithm in sorted(allowed):
                            name = prefix+algorithm+"__FWD"
                            pred = np.where(rec.digital[prefix+algorithm+"__VALID"], rec.digital[name], -1).astype(np.int8)
                            add(algorithm, section, truth, pred, first_fourier, path.stem)
                else:
                    truth = np.where(rec.digital["expert__VALID"], rec.digital["expert__FWD"], -1).astype(np.int8)
                    # Канонический хеш исходных входов из импортированного архива:
                    # дополнительные каналы эксперта не создают новые группы.
                    cluster = source+":"+split_records[(source, record_id)]["input_sha256"]
                    with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
                        for name in meta["models"]:
                            samples = archive[name+"__samples"]
                            if not len(samples) or not np.array_equal(samples, np.arange(int(samples[0]), rec.n_samples)):
                                raise ValueError(f"Нет плотного полного ответа: {path}, {name}")
                            probability = archive[name+"__probability_valid"]
                            direction = archive[name+"__direction"]
                            if not np.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all() or not np.isin(direction, (0,1)).all():
                                raise ValueError(f"Невалидный ответ сети: {path}")
                            pred = np.full(rec.n_samples, -1, dtype=np.int8)
                            pred[samples] = np.where(probability >= .5, direction, -1)
                            add("nn_"+name, 0, truth, pred, int(samples[0]), cluster)
                    for name, store in stores[source].items():
                        if not store.has_record(record_id):
                            raise ValueError(f"Нет формульного ответа: {source}, {record_id}, {name}")
                        automatic = store.get_record(record_id)
                        samples = np.asarray(automatic["samples"], dtype=np.int64)
                        if not len(samples) or not np.array_equal(samples, np.arange(int(samples[0]), rec.n_samples)):
                            raise ValueError(f"Формульная сетка не плотная: {source}/{record_id}/{name}")
                        values = np.asarray(automatic["directions"])
                        if not np.isin(values, (-999,0,1)).all():
                            raise ValueError("Неполученные формульные ответы не являются неприменимостью")
                        pred = np.full(rec.n_samples, -1, dtype=np.int8)
                        pred[samples] = np.where(values == -999, -1, values)
                        add(name, 0, truth, pred, max(first_fourier, int(samples[0])), cluster)
                cache.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_json(cache, {"request": request, "provenance":meta, "rows": part})
            rows.extend(part)
            reporter.update(number)
            _atomic_write_json(output_root / "progress.json", {"status": "running", "completed": number, "total": len(files)})
        result = _summarize_engineering(rows, bootstrap_repeats)
        result.update({"scope": scope, "config": asdict(cfg), "records": len(files),
                       "available_records": total_available, "partial": max_records is not None,
                       "reference_status": "author_recheck_pending" if scope == "rtds" else "validation_not_independent_holdout",
                       "per_record_results": "records/*.json"})
        _atomic_write_json(output_root / "summary.json", result)
        _plot_engineering(rows, output_root)
        _atomic_write_json(output_root / "progress.json", {"status": "complete", "completed": len(files), "total": len(files)})
        reporter.finish()
        print(f"[Готово] {output_root / 'summary.json'}", flush=True)
        return result
    except Exception as exc:
        _atomic_write_json(output_root / "progress.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        for group in stores.values():
            for store in group.values():
                store.close()


def _summarize_engineering(rows: list[dict], bootstrap_repeats: int) -> dict:
    """Равный вес записи; секции RTDS вместе. Интервалы — по целым группам."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["algorithm"])].append(r)
    result = {}
    for (source, algorithm), group in groups.items():
        errors = {}
        for name in group[0]["metrics"]["regions"]["all"]:
            items = [r["metrics"]["regions"]["all"][name] for r in group]
            by_record = defaultdict(list)
            for r, item in zip(group, items):
                if item["fraction"] is not None:
                    by_record[r["cluster"]].append(item["fraction"])
            values = np.array([np.mean(x) for x in by_record.values()])
            support = sum(x["support_ms"] for x in items)
            error = sum(x["error_ms"] for x in items)
            ci = None
            if len(values) > 1 and bootstrap_repeats:
                rng = np.random.default_rng(42)
                means = np.mean(rng.choice(values, (bootstrap_repeats, len(values))), axis=1)
                ci = np.quantile(means, [.025,.975]).tolist()
            errors[name] = {"error_ms": error, "support_ms": support,
                "time_weighted_fraction": error/support if support else None,
                "supported_groups": len(values), "total_groups": len({r['cluster'] for r in group}),
                "group_mean": float(values.mean()) if len(values) else None,
                "group_median": float(np.median(values)) if len(values) else None,
                "group_p90": float(np.quantile(values,.9)) if len(values) else None,
                "group_p95": float(np.quantile(values,.95)) if len(values) else None,
                "group_max": float(values.max()) if len(values) else None, "group_mean_ci95": ci}
        events = defaultdict(list)
        for r in group:
            for event in r["metrics"]["event_matching"]:
                events[(event["from"], event["to"], event["window_ms"], event["hold_ms"])].append(event)
        event_summary = []
        for (a,b,w,h), chunks in events.items():
            records = [x for chunk in chunks for x in chunk["events"]]
            delays = [x["delay_ms"] for x in records if x["status"] == "matched"]
            deadlines = {t: {k:sum(c["by_deadline"].get(t,{}).get(k,0) for c in chunks)
                             for k in ("eligible","responded")}
                         for t in chunks[0]["by_deadline"]}
            event_summary.append({"from":a,"to":b,"window_ms":w,"hold_ms":h,
                "reference_events": len(records),
                "statuses": {status:sum(x["status"]==status for x in records)
                    for status in ("matched","missed","censored","short_reference","next_reference_event")},
                "unmatched_predicted_events":sum(c["unmatched_predicted_events"] for c in chunks),
                "early_events":sum(x<0 for x in delays), "late_events":sum(x>0 for x in delays),
                "signed_delay_median_ms":float(np.median(delays)) if delays else None,
                "late_delay_p90_ms":float(np.quantile([x for x in delays if x>=0],.9)) if any(x>=0 for x in delays) else None,
                "by_deadline":deadlines})
        worst = defaultdict(list)
        for r in group:
            worst[r["cluster"]].append(r["metrics"]["regions"]["all"]["all"]["max_episode_ms"])
        durations = np.array([max(v) for v in worst.values()])
        result.setdefault(source,{})[algorithm] = {"record_sections": len(group), "errors": errors,
            "event_summary":event_summary,
            "worst_episode_by_group_ms": {"median":float(np.median(durations)), "p90":float(np.quantile(durations,.9)),
                "p95":float(np.quantile(durations,.95)), "max":float(durations.max()),
                "groups_over_threshold":{str(t):int((durations>t+1e-9).sum()) for t in group[0]["metrics"]["config"]["thresholds_ms"]}},
            "by_section": {str(s): {"records":sum(r["section"]==s for r in group),
                "mean_error_fraction":float(np.mean([r["metrics"]["regions"]["all"]["all"]["fraction"] for r in group if r["section"]==s]))}
                for s in sorted({r["section"] for r in group}) if s}}
    # Парные разности общего ошибочного времени на одинаковых группах;
    # отрицательная разность B-A означает меньшее ошибочное время у B.
    paired = {}
    from itertools import combinations
    for source in result:
        maps = {}
        for algorithm in result[source]:
            values = defaultdict(list)
            for r in groups[(source,algorithm)]:
                values[r["cluster"]].append(r["metrics"]["regions"]["all"]["all"]["fraction"])
            maps[algorithm] = {k:float(np.mean(v)) for k,v in values.items()}
        comparisons = []
        for a,b in combinations(sorted(maps),2):
            keys = sorted(maps[a].keys() & maps[b].keys())
            diff = np.array([maps[b][k]-maps[a][k] for k in keys])
            ci = None
            if len(diff)>1 and bootstrap_repeats:
                rng = np.random.default_rng(42)
                ci = np.quantile(rng.choice(diff,(bootstrap_repeats,len(diff))).mean(axis=1),[.025,.975]).tolist()
            comparisons.append({"a":a,"b":b,"groups":len(diff),"mean_difference_b_minus_a":float(diff.mean()) if len(diff) else None,
                "ci95":ci,"improved_b":int((diff < -1e-12).sum()),"worse_b":int((diff > 1e-12).sum()),
                "unchanged":int((np.abs(diff)<=1e-12).sum())})
        paired[source] = comparisons
    return {"schema": 1, "by_source": result, "paired_error_comparisons":paired, "bootstrap_repeats": bootstrap_repeats,
            "warning": "Интервалы условны для имеющихся групп и весов; редкий класс может быть представлен несколькими опытами. RTDS не случайная эксплуатационная выборка."}


def _plot_engineering(rows: list[dict], output_root: Path) -> None:
    """Диагностические рисунки; статья и её рисунок 11 не перезаписываются."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for source in sorted({r["source"] for r in rows}):
        # В обзорных рисунках только последние экспертные сети и формулы.
        # Все weak/best остаются в JSON и парных сравнениях.
        selected = [r for r in rows if r["source"] == source
                    and not r["algorithm"].startswith("nn_weak_") and not r["algorithm"].endswith("_best")]
        names = sorted({r["algorithm"] for r in selected})
        fig, axes = plt.subplots(1,2, figsize=(15,6))
        for name in names:
            group = [r["metrics"] for r in selected if r["algorithm"] == name]
            clusters = {}
            for row in selected:
                if row["algorithm"] == name:
                    value = row["metrics"]["regions"]["all"]["all"]["max_episode_ms"]
                    clusters[row["cluster"]] = max(value,clusters.get(row["cluster"],0))
            lengths = np.array(list(clusters.values()))
            thresholds = np.r_[0,5,20,50,100,200,500,1000]
            axes[0].plot(thresholds, [100*np.mean(lengths>t) for t in thresholds], label=name)
            observed = [m for m in group if m["stability"]["stable_switches_per_second"] is not None]
            axes[1].scatter([100*m["regions"]["all"]["all"]["fraction"] for m in observed],
                            [m["stability"]["stable_switches_per_second"] for m in observed], s=9, alpha=.4, label=name)
        axes[0].set(xlabel="Длительность непрерывной ошибки, мс", ylabel="Доля исходных групп с более длинной ошибкой, %", xscale="symlog")
        axes[1].set(xlabel="Ошибочное время, % записи", ylabel="Переключений/с вне переходной зоны и старта", yscale="symlog")
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle(source+": ошибки и неустойчивость (диагностический обзор)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6)
        fig.subplots_adjust(bottom=.32, top=.90)
        fig.savefig(output_root / f"{source}_engineering.png", dpi=160)
        plt.close(fig)
        summary = _summarize_engineering(selected, 0)["by_source"][source]
        keys = ("false_forward","false_reverse","unnecessary_refusal","missed_invalid")
        matrix = np.array([[summary[name]["errors"][key]["group_mean"]
                            if summary[name]["errors"][key]["group_mean"] is not None else np.nan
                            for key in keys] for name in names])*100
        fig, ax = plt.subplots(figsize=(12, max(5, .45*len(names))))
        cmap = plt.get_cmap("YlOrRd").copy(); cmap.set_bad("lightgray")
        im = ax.imshow(matrix, vmin=0, vmax=100, cmap=cmap, aspect="auto")
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_xticks(range(4), ("Ложное прямое\n/ время REV", "Ложное обратное\n/ время FWD",
                                "Лишний отказ\n/ применимое время", "Пропуск Н\n/ неприменимое время"), fontsize=8)
        for i,name in enumerate(names):
            for j,key in enumerate(keys):
                item = summary[name]["errors"][key]
                label = "нет эталона" if np.isnan(matrix[i,j]) else f"{matrix[i,j]:.2f}%\nn={item['supported_groups']}"
                ax.text(j,i,label,ha="center",va="center",fontsize=7)
        ax.set_title(source+": условные ошибки, равный вес исходных групп; n — поддержка")
        fig.colorbar(im, ax=ax, label="Ошибка, % времени соответствующего класса")
        fig.tight_layout(); fig.savefig(output_root/f"{source}_conditional_errors.png", dpi=160); plt.close(fig)
        fig, axes = plt.subplots(2,2, figsize=(13,10))
        configuration = selected[0]["metrics"]["config"]
        window = max(configuration["match_windows_ms"])
        hold = 5. if 5. in configuration["hold_ms"] else configuration["hold_ms"][0]
        families = (((0,1),), ((1,0),), ((0,-1),(1,-1)), ((-1,0),(-1,1)))
        titles = ("REVERSE → FORWARD", "FORWARD → REVERSE", "Появление неприменимости", "Восстановление применимости")
        for ax, family, title in zip(axes.flat,families,titles):
            for name in names:
                chunks = [e for e in summary[name]["event_summary"] if (e["from"],e["to"]) in family
                          and e["window_ms"] == window and e["hold_ms"] == hold]
                deadlines = [t for t in configuration["thresholds_ms"] if t<=window]
                fractions = []
                for t in deadlines:
                    eligible = sum(c["by_deadline"].get(str(t),{}).get("eligible",0) for c in chunks)
                    responded = sum(c["by_deadline"].get(str(t),{}).get("responded",0) for c in chunks)
                    fractions.append(100*responded/eligible if eligible else np.nan)
                if np.isfinite(fractions).any():
                    ax.plot(deadlines, fractions, marker=".", label=name)
            ax.set(title=title, xlabel="Начало устойчивого ответа не позже, мс", ylabel="Доля наблюдаемых событий, %", ylim=(0,105))
            ax.grid(alpha=.25)
        fig.suptitle(f"{source}: удержание {hold:g} мс, окно ±{window:g} мс; опережения включены, пропуски не скрыты")
        handles = {}
        for ax in axes.flat:
            for handle,label in zip(*ax.get_legend_handles_labels()):
                handles[label]=handle
        fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, fontsize=6)
        fig.subplots_adjust(bottom=.24,hspace=.35,top=.91)
        fig.savefig(output_root/f"{source}_event_response.png",dpi=160); plt.close(fig)


def run_manual() -> None:
    """F5: инженерные метрики плотной галереи/RTDS или прежний CSV-аудит."""
    # F5 без аргументов: только новые метрики, без инференса и изменения статьи.
    ACTION = "engineering"  # engineering: плотные ответы; legacy: прежний массовый статистический аудит.
    SCOPES = ("rtds", "expert")  # Можно оставить один источник; expert — только validation.
    OUTPUT_ROOT = PROJECT_ROOT / "data/phase5/pdr_engineering_review"
    GALLERY_ROOT = PROJECT_ROOT / "data/phase5/pdr_expert_gallery"
    RTDS_ROOT = PROJECT_ROOT / "data/phase5/pdr_rtds_review_h123"
    MAX_RECORDS = None  # None: весь набор; 1: проба в отдельной подпапке smoke.
    RESUME = True  # Повторно использовать метрики при неизменности входов, кода и параметров.
    BOOTSTRAP_REPEATS = 2000  # Повторы по целым группам записей; 0 отключает интервалы.
    if ACTION == "engineering":
        from osc_tools.pdr.temporal_metrics import TemporalConfig
        config = TemporalConfig(
            transition_ms=5.,  # Граница эксперта и следующие 5 мс: отдельная область проверки.
            thresholds_ms=(5.,20.,50.,100.),  # Диагностические длительности, не норматив БАВР.
            match_windows_ms=(100.,50.),  # Симметричные окна сопоставления событий, мс.
            hold_ms=(0.,5.,20.),  # Удержание ответа для оценки; НЕ выдержка в модели.
            sliding_ms=100.,  # Ошибочное время в наиболее неблагоприятном окне, мс.
        )
        for scope in SCOPES:
            build_engineering_review(scope=scope, output_root=OUTPUT_ROOT, gallery_root=GALLERY_ROOT,
                rtds_root=RTDS_ROOT, config=config, resume=RESUME, max_records=MAX_RECORDS,
                bootstrap_repeats=BOOTSTRAP_REPEATS)
        return
    if ACTION != "legacy":
        raise ValueError("ACTION: engineering или legacy")
    ANALYSIS_DIR = DEFAULT_ANALYSIS_DIR  # Уже готовые таблицы основного анализа; здесь появится расширенная статистика.
    for name, path in build_review_tables(ANALYSIS_DIR).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
