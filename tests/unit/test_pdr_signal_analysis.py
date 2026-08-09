"""Unit-тесты для модуля анализа полноты сигналов pdr/signal_analysis.py."""

import pytest
import numpy as np

from osc_tools.ml.phase5_contracts import ChannelProvenance, CHANNEL_ORDER
from osc_tools.pdr.signal_analysis import derive_missing_currents, check_pdr_signal_sufficiency


def test_derive_missing_ib():
    """Проверка векторного расчета missing IB из IA и IC."""
    n_samples = 100
    signals = np.zeros((8, n_samples), dtype=np.float32)
    # IA: синусоида 1.0 A
    t = np.linspace(0, 2 * np.pi, n_samples)
    signals[0] = np.sin(t)  # IA
    signals[1] = np.nan     # IB отсутствует
    signals[2] = np.sin(t - 4 * np.pi / 3)  # IC (-120 град)

    provenance = np.array([
        ChannelProvenance.MEASURED,
        ChannelProvenance.MISSING,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MISSING,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MISSING,
    ], dtype=np.uint8)

    new_signals, new_prov = derive_missing_currents(signals, provenance)

    idx_ib = CHANNEL_ORDER.index("IB")
    assert new_prov[idx_ib] == ChannelProvenance.DERIVED
    # Проверка вычисленного сигнала IB = -(IA + IC)
    expected_ib = -(signals[0] + signals[2])
    np.testing.assert_allclose(new_signals[idx_ib], expected_ib, atol=1e-5)


def test_check_signal_sufficiency():
    """Проверка аудита полноты сигналов."""
    provenance = np.array([
        ChannelProvenance.MEASURED,
        ChannelProvenance.DERIVED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MISSING,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MISSING,
    ], dtype=np.uint8)

    res = check_pdr_signal_sufficiency(provenance, voltage_basis="phase")

    assert res.can_run_phase_pdr is True
    assert res.can_run_pos_seq_pdr is True
    assert "IB" in res.derived_channels
    assert "IN" in res.missing_channels


def test_line_basis_without_two_voltage_channels_is_insufficient():
    provenance = np.full(8, int(ChannelProvenance.MISSING), dtype=np.uint8)
    provenance[CHANNEL_ORDER.index("IA")] = int(ChannelProvenance.MEASURED)
    provenance[CHANNEL_ORDER.index("IB")] = int(ChannelProvenance.MEASURED)

    res = check_pdr_signal_sufficiency(provenance, voltage_basis="line")

    assert res.can_run_phase_pdr is False
    assert res.can_run_pos_seq_pdr is False
