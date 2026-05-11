"""
Тесты для агрегации результатов по fold/seed (mean ± std).

Покрывает функцию aggregate_by_fold_seed из aggregate_reports.py.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.evaluation.aggregate_reports import aggregate_by_fold_seed


class TestAggregateByFoldSeed:
    """Тесты для aggregate_by_fold_seed."""

    @pytest.fixture
    def sample_df_with_folds(self):
        """DataFrame с данными из нескольких fold/seed для одной конфигурации."""
        rows = []
        # SimpleMLP_light_Raw_Dense — 3 fold × 2 seed = 6 строк
        for fold in [1, 2, 3]:
            for seed in [0, 1]:
                rows.append({
                    'ExpID': '2.7',
                    'Experiment': f'Exp_2.7_f{fold}_s{seed}_SimpleMLP_light_raw_none',
                    'Model': 'MLP',
                    'Complexity': 'Light',
                    'Features': 'Raw',
                    'Sampling': 'Dense',
                    'TargetLevel': 'base',
                    'Val F1': 0.4 + fold * 0.01 + seed * 0.005,
                    'Val Loss': 1.5 - fold * 0.1,
                    'Val Acc': 0.2 + fold * 0.02,
                    'Params': 1000,
                    'Epochs': 30,
                    'Fold': fold,
                    'SeedIdx': seed,
                })
        # SimpleCNN_medium_PhasePolar_Stride — 2 строки
        for fold in [1, 2]:
            rows.append({
                'ExpID': '2.7',
                'Experiment': f'Exp_2.7_f{fold}_s0_SimpleCNN_medium_phase_polar_stride',
                'Model': 'CNN',
                'Complexity': 'Medium',
                'Features': 'PhasePolar',
                'Sampling': 'Stride',
                'TargetLevel': 'base',
                'Val F1': 0.5 + fold * 0.02,
                'Val Loss': 1.2 - fold * 0.05,
                'Val Acc': 0.3,
                'Params': 5000,
                'Epochs': 30,
                'Fold': fold,
                'SeedIdx': 0,
            })
        return pd.DataFrame(rows)

    @pytest.fixture
    def sample_df_no_folds(self):
        """DataFrame без fold/seed (старый формат Фазы 2.6)."""
        return pd.DataFrame([{
            'ExpID': '2.6.1',
            'Experiment': 'Exp_2.6.1_SimpleMLP_medium_phase_polar_stride_base_weights_aug',
            'Model': 'MLP',
            'Complexity': 'Medium',
            'Val F1': 0.65,
            'Val Loss': 0.8,
        }])

    def test_aggregation_produces_mean_std(self, sample_df_with_folds):
        """Агрегация вычисляет mean и std для числовых метрик."""
        result = aggregate_by_fold_seed(sample_df_with_folds)
        assert result is not None
        assert len(result) == 2  # 2 уникальных конфигурации (MLP + CNN)

        # Проверяем наличие столбцов mean и std
        assert 'Val F1' in result.columns
        assert 'Val F1 std' in result.columns
        assert 'Val Loss' in result.columns
        assert 'Val Loss std' in result.columns
        assert 'N_runs' in result.columns

    def test_aggregation_correct_counts(self, sample_df_with_folds):
        """N_runs соответствует количеству строк в группе."""
        result = aggregate_by_fold_seed(sample_df_with_folds)
        
        mlp_row = result[result['Model'] == 'MLP'].iloc[0]
        cnn_row = result[result['Model'] == 'CNN'].iloc[0]
        
        assert mlp_row['N_runs'] == 6  # 3 fold × 2 seed
        assert cnn_row['N_runs'] == 2  # 2 fold × 1 seed

    def test_aggregation_mean_is_correct(self, sample_df_with_folds):
        """Среднее значение вычисляется корректно."""
        result = aggregate_by_fold_seed(sample_df_with_folds)
        mlp_row = result[result['Model'] == 'MLP'].iloc[0]
        
        # Ручной расчёт среднего Val F1 для MLP:
        # fold=1,s=0: 0.410, fold=1,s=1: 0.415
        # fold=2,s=0: 0.420, fold=2,s=1: 0.425
        # fold=3,s=0: 0.430, fold=3,s=1: 0.435
        expected_mean = np.mean([0.410, 0.415, 0.420, 0.425, 0.430, 0.435])
        assert abs(mlp_row['Val F1'] - expected_mean) < 1e-3

    def test_returns_none_without_fold_column(self, sample_df_no_folds):
        """Возвращает None если нет столбца Fold."""
        result = aggregate_by_fold_seed(sample_df_no_folds)
        assert result is None

    def test_returns_none_for_empty_folds(self):
        """Возвращает None если все значения Fold — NaN."""
        df = pd.DataFrame([{
            'ExpID': '2.7', 'Model': 'MLP', 'Val F1': 0.5,
            'Fold': None, 'SeedIdx': None,
        }])
        result = aggregate_by_fold_seed(df)
        assert result is None

    def test_sorted_by_val_f1(self, sample_df_with_folds):
        """Результат отсортирован по Val F1 (убывание)."""
        result = aggregate_by_fold_seed(sample_df_with_folds)
        f1_values = result['Val F1'].tolist()
        assert f1_values == sorted(f1_values, reverse=True)

    def test_group_columns_preserved(self, sample_df_with_folds):
        """Группировочные колонки сохранены в результате."""
        result = aggregate_by_fold_seed(sample_df_with_folds)
        for col in ['ExpID', 'Model', 'Complexity', 'Features', 'Sampling']:
            assert col in result.columns
