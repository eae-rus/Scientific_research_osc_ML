"""
Тесты для парсинга имён экспериментов (config_resolvers).

Покрывает:
- Формат Фазы 2.7 (Exp_2.7_f{N}_s{M}_...)
- Формат Фазы 2.6 (Exp_2.6.1_...)
- Обратную совместимость со старыми форматами
- Парсинг fold/seed
- Парсинг feature_mode, sampling, target_level
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.evaluation._core.config_resolvers import (
    parse_experiment_info,
    extract_base_exp_id,
)


class TestParseExperimentInfo:
    """Тесты для parse_experiment_info."""

    # --- Формат Фазы 2.7 ---
    
    def test_phase27_basic(self):
        """Базовый формат Exp_2.7_f1_s0_SimpleMLP_light_raw_none."""
        info = parse_experiment_info("Exp_2.7_f1_s0_SimpleMLP_light_raw_none")
        assert info["exp_id"] == "2.7"
        assert info["model_family"] == "MLP"
        assert info["complexity"] == "Light"
        assert info["feature_mode"] == "Raw"
        assert info["sampling"] == "Dense"
        assert info["fold"] == 1
        assert info["seed_idx"] == 0

    def test_phase27_phase_polar_stride(self):
        """Формат с phase_polar и stride."""
        info = parse_experiment_info("Exp_2.7_f3_s2_ConvKAN_heavy_phase_polar_stride")
        assert info["exp_id"] == "2.7"
        assert info["model_family"] == "ConvKAN"
        assert info["complexity"] == "Heavy"
        assert info["feature_mode"] == "PhasePolar"
        assert info["sampling"] == "Stride"
        assert info["fold"] == 3
        assert info["seed_idx"] == 2

    def test_phase27_rPhysicsKAN(self):
        """rPhysicsKAN парсится корректно (не путается с PhysicsKAN)."""
        info = parse_experiment_info("Exp_2.7_f1_s0_rPhysicsKAN_medium_phase_polar_stride")
        assert info["model_family"] == "rPhysicsKAN"
        assert info["complexity"] == "Medium"

    def test_phase27_cPhysicsKAN(self):
        """cPhysicsKAN парсится корректно."""
        info = parse_experiment_info("Exp_2.7_f2_s1_cPhysicsKAN_light_phase_polar_none")
        assert info["model_family"] == "cPhysicsKAN"

    def test_phase27_ozz_suffix(self):
        """OZZ суффикс корректно определяет target_level."""
        info = parse_experiment_info("Exp_2.7_f1_s0_SimpleMLP_heavy_phase_polar_stride_ozz")
        assert info["target_level"] == "ozz"
        assert info["fold"] == 1

    def test_phase27_snapshot(self):
        """Snapshot как sampling strategy."""
        info = parse_experiment_info("Exp_2.7_f5_s4_SimpleCNN_medium_raw_snapshot")
        assert info["sampling"] == "Snapshot"
        assert info["feature_mode"] == "Raw"
        assert info["fold"] == 5
        assert info["seed_idx"] == 4

    def test_phase27_phase_complex(self):
        """phase_complex распознаётся."""
        info = parse_experiment_info("Exp_2.7_f1_s0_SimpleMLP_light_phase_complex_stride")
        assert info["feature_mode"] == "PhaseComplex"

    def test_phase27_symmetric_polar(self):
        """symmetric_polar распознаётся (не путается с symmetric)."""
        info = parse_experiment_info("Exp_2.7_f1_s0_SimpleMLP_light_symmetric_polar_none")
        assert info["feature_mode"] == "SymmetricPolar"

    def test_phase27_alpha_beta(self):
        """alpha_beta распознаётся."""
        info = parse_experiment_info("Exp_2.7_f1_s0_SimpleMLP_light_alpha_beta_stride")
        assert info["feature_mode"] == "AlphaBeta"

    def test_phase27_resnet(self):
        """ResNet1D → ResNet."""
        info = parse_experiment_info("Exp_2.7_f1_s0_ResNet1D_heavy_raw_none")
        assert info["model_family"] == "ResNet"

    # --- Обратная совместимость (Фаза 2.6) ---

    def test_phase26_format(self):
        """Старый формат без fold/seed."""
        info = parse_experiment_info("Exp_2.6.1_PhysicsKAN_medium_phase_polar_stride_base_weights_aug")
        assert info["exp_id"] == "2.6.1"
        assert info["model_family"] == "PhysicsKAN"
        assert info["complexity"] == "Medium"
        assert info["feature_mode"] == "PhasePolar"
        assert info["sampling"] == "Stride"
        assert info["fold"] is None
        assert info["seed_idx"] is None
        assert info["is_aug"] == "Yes"
        assert info["balancing"] == "Weights"

    def test_phase26_hierarchical(self):
        """Иерархическая модель из Фазы 2.6."""
        info = parse_experiment_info("Exp_2.6.4_HierarchicalCNN_medium_phase_polar_stride_full_by_levels")
        assert info["arch_type"] == "Hierarchical"
        assert "Hier" in info["model_family"]

    def test_phase26_ozz_by_id(self):
        """OZZ по ID эксперимента 2.6.11."""
        info = parse_experiment_info("Exp_2.6.11_SimpleMLP_heavy_phase_polar_stride_base_weights_aug")
        assert info["target_level"] == "ozz"

    # --- Граничные случаи ---

    def test_unknown_format(self):
        """Неизвестный формат не крашит парсер."""
        info = parse_experiment_info("some_random_folder_name")
        assert info["exp_id"] == "Unknown"
        assert info["model_family"] == "Unknown"
        assert info["fold"] is None

    def test_fold_seed_always_present(self):
        """Поля fold и seed_idx всегда присутствуют в результате."""
        info = parse_experiment_info("anything")
        assert "fold" in info
        assert "seed_idx" in info


class TestExtractBaseExpId:
    """Тесты для extract_base_exp_id."""

    def test_two_level_id(self):
        assert extract_base_exp_id("Exp_2.7_f1_s0_SimpleMLP") == "2.7"

    def test_three_level_id(self):
        assert extract_base_exp_id("Exp_2.6.1_PhysicsKAN") == "2.6.1"

    def test_four_level_id(self):
        assert extract_base_exp_id("Exp_2.5.1.0_SimpleMLP") == "2.5.1.0"

    def test_no_id(self):
        assert extract_base_exp_id("random_text") == "Unknown"
