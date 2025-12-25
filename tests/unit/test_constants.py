"""
Unit-тесты для osc_tools.core.constants.

Тестирует константы, перечисления и классы-контейнеры признаков.
Это чистые unit-тесты без файловых зависимостей.
"""

import pytest
import os
import sys

# Make project root importable when tests are executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from osc_tools.core.constants import TYPE_OSC, Features, PDRFeatures


class TestTypeOSCEnum:
    """Тесты для перечисления TYPE_OSC."""
    
    def test_type_osc_members_exist(self):
        """Проверяет, что все типы осциллограмм определены."""
        expected_types = [
            "COMTRADE_CFG_DAT",
            "COMTRADE_CFF",
            "BRESLER",
            "BLACK_BOX",
            "RES_3",
            "EKRA",
            "PARMA",
            "PARMA_TO",
            "NEVA",
            "OSC",
        ]
        for type_name in expected_types:
            assert hasattr(TYPE_OSC, type_name), \
                f"TYPE_OSC должна содержать {type_name}"
    
    def test_type_osc_enum_values_are_strings(self):
        """Проверяет, что все значения enum — строки."""
        for member in TYPE_OSC:
            assert isinstance(member.value, str), \
                f"{member.name}.value должна быть строкой, а не {type(member.value)}"
    
    def test_type_osc_comtrade_cfg_dat(self):
        """Проверяет описание COMTRADE_CFG_DAT."""
        assert "Comtrade" in TYPE_OSC.COMTRADE_CFG_DAT.value
        assert ".cfg" in TYPE_OSC.COMTRADE_CFG_DAT.value.lower()
        assert ".dat" in TYPE_OSC.COMTRADE_CFG_DAT.value.lower()


class TestFeaturesClass:
    """Тесты для класса Features (индексы и имена признаков)."""
    
    def test_current_indices_exist(self):
        """Проверяет, что определены все токовые индексы."""
        expected = {"IA": 0, "IB": 1, "IC": 2, "IN": -1}
        assert Features.CURRENT_INDICES == expected
    
    def test_voltage_indices_consistency(self):
        """Проверяет консистентность индексов напряжения."""
        # BB (с нейтралью относительно земли)
        assert "UA BB" in Features.VOLTAGE_PHAZE_BB_INDICES
        assert "UB BB" in Features.VOLTAGE_PHAZE_BB_INDICES
        assert "UC BB" in Features.VOLTAGE_PHAZE_BB_INDICES
        
        # CL (между фазами — линейные напряжения)
        assert "UAB CL" in Features.VOLTAGE_LINE_CL_INDICES
        assert "UBC CL" in Features.VOLTAGE_LINE_CL_INDICES
        assert "UCA CL" in Features.VOLTAGE_LINE_CL_INDICES
    
    def test_current_list(self):
        """Проверяет список токов."""
        assert Features.CURRENT == ["IA", "IB", "IC"]
        assert len(Features.CURRENT) == 3
    
    def test_voltage_phaze_lists(self):
        """Проверяет списки фазных напряжений."""
        assert len(Features.VOLTAGE_PHAZE_BB) == 3
        assert len(Features.VOLTAGE_PHAZE_CL) == 3
        assert "UA BB" in Features.VOLTAGE_PHAZE_BB
        assert "UA CL" in Features.VOLTAGE_PHAZE_CL
    
    def test_all_combines_currents_and_voltages(self):
        """Проверяет, что ALL содержит токи и напряжения."""
        assert set(Features.CURRENT).issubset(set(Features.ALL))
        assert "UA BB" in Features.ALL
        assert "IA" in Features.ALL
    
    def test_target_features_exist(self):
        """Проверяет целевые признаки для обучения."""
        expected_targets = ["opr_swch", "abnorm_evnt", "emerg_evnt"]
        assert Features.TARGET == expected_targets
    
    def test_no_duplicates_in_all(self):
        """Проверяет, что в ALL нет дубликатов."""
        assert len(Features.ALL) == len(set(Features.ALL)), \
            "В Features.ALL найдены дубликаты"


class TestPDRFeaturesClass:
    """Тесты для класса PDRFeatures (симметричные составляющие)."""
    
    def test_current_1_features(self):
        """Проверяет признаки прямой последовательности тока."""
        assert "I_pos_seq_mag" in PDRFeatures.CURRENT_1
        assert "I_pos_seq_angle" in PDRFeatures.CURRENT_1
    
    def test_current_2_features(self):
        """Проверяет признаки обратной последовательности тока."""
        assert "I_neg_seq_mag" in PDRFeatures.CURRENT_2
        assert "I_neg_seq_angle" in PDRFeatures.CURRENT_2
    
    def test_voltage_features(self):
        """Проверяет признаки напряжения."""
        assert "V_pos_seq_mag" in PDRFeatures.VOLTAGE_1
        assert "V_neg_seq_mag" in PDRFeatures.VOLTAGE_2
    
    def test_power_features(self):
        """Проверяет признаки мощности."""
        assert "P_pos_seq" in PDRFeatures.POWER_1
        assert "Q_pos_seq" in PDRFeatures.POWER_1
        assert "P_neg_seq" in PDRFeatures.POWER_2
    
    def test_impedance_features(self):
        """Проверяет признаки импеданса."""
        assert "Z_pos_seq_mag" in PDRFeatures.IMPEDANCE
        assert "Z_pos_seq_angle" in PDRFeatures.IMPEDANCE
    
    def test_all_model_1_features(self):
        """Проверяет, что модель 1 содержит токи и напряжение."""
        assert len(PDRFeatures.ALL_MODEL_1) > 0
        # Должны быть токи и напряжение прямой последовательности
        assert any("I_" in f for f in PDRFeatures.ALL_MODEL_1)
        assert any("V_" in f for f in PDRFeatures.ALL_MODEL_1)
    
    def test_target_features(self):
        """Проверяет целевые признаки."""
        assert "rPDR PS" in PDRFeatures.TARGET_TRAIN
        assert "iPDR PS" in PDRFeatures.TARGET_TEST


@pytest.mark.unit
class TestConstantsIntegration:
    """Интеграционные проверки между классами констант."""
    
    def test_features_and_pdr_features_no_overlap(self):
        """
        Проверяет, что Features и PDRFeatures не дублируют названия
        (Features — исходные признаки, PDRFeatures — производные).
        """
        # Это проверка архитектуры — признаки разных уровней
        features_names = set(Features.ALL)
        pdr_names = set(PDRFeatures.ALL_MODEL_1)
        
        # Не должно быть совпадения (это разные наборы)
        overlap = features_names & pdr_names
        assert len(overlap) == 0, \
            f"Features и PDRFeatures имеют общие названия: {overlap}"
