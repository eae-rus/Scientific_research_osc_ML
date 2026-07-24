"""Интеграционный тест пайплайна псевдоразметки РНМ."""

import math
import numpy as np
import pytest

from osc_tools.ml.phase5_contracts import TimebaseContract, ChannelProvenance
from osc_tools.pdr.public_algorithms import PhasePDRAlgorithm
from osc_tools.pdr.labeler import PDRDatasetLabeler
from osc_tools.pdr.base import PDRDirection


def test_pdr_labeler_single_record():
    """Проверка псевдоразметки одной симулированной осциллограммы."""
    fs = 6400.0
    fn = 50.0
    spp = int(fs / fn)  # 128
    timebase = TimebaseContract.create(fs, fn, window_periods=10.0, stride_fraction=8)

    # Создаём 20 периодов сигнала
    n_periods = 20
    n_samples = n_periods * spp
    t = np.arange(n_samples) / fs

    signals = np.zeros((8, n_samples), dtype=np.float32)
    # Напряжения A, B, C (номинал 1.0)
    signals[4] = np.sin(2 * np.pi * fn * t)  # UA
    signals[5] = np.sin(2 * np.pi * fn * t - 2 * np.pi / 3)  # UB
    signals[6] = np.sin(2 * np.pi * fn * t + 2 * np.pi / 3)  # UC

    # Токи A, B, C (ток отстает от U на 45 град -> прямое направление)
    phi_rad = math.radians(-45)
    signals[0] = 0.5 * np.sin(2 * np.pi * fn * t + phi_rad)  # IA
    signals[1] = 0.5 * np.sin(2 * np.pi * fn * t - 2 * np.pi / 3 + phi_rad)  # IB
    signals[2] = 0.5 * np.sin(2 * np.pi * fn * t + 2 * np.pi / 3 + phi_rad)  # IC

    provenance = np.array([
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MISSING,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MISSING,
    ], dtype=np.uint8)

    teacher = PhasePDRAlgorithm(phi_mch_deg=45.0)
    labeler = PDRDatasetLabeler(teacher_algorithm=teacher)

    res = labeler.label_single_record(
        record_id="test_rec_01",
        signals=signals,
        provenance=provenance,
        timebase=timebase,
    )

    assert len(res.directions) > 0
    # Проверка зоны разогрева (первые 10 периодов имеют UNLABELED / -999)
    warmup_indices = np.where(res.warmup_mask)[0]
    valid_indices = np.where(~res.warmup_mask)[0]

    assert len(warmup_indices) > 0
    assert len(valid_indices) > 0

    # В зоне разогрева метки должны быть UNLABELED (-999)
    assert (res.directions[warmup_indices] == int(PDRDirection.UNLABELED)).all()

    # После зоны разогрева метки должны быть FORWARD (1) для нашего тестового сигнала
    assert (res.directions[valid_indices] == int(PDRDirection.FORWARD)).all()
