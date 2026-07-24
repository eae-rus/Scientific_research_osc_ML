"""Заглушки и каркасы публичных алгоритмов сторонних производителей БАВР (PDR).

Служат точками расширения при добавлении алгоритмов SEL, Siemens, ABB, Schneider, ЭКРА и т.д.
"""

from __future__ import annotations

from osc_tools.pdr.base import PDRAlgorithm, PDRInputData, PDROutput, PDRDirection


class ManufacturerPowerPDRStub(PDRAlgorithm):
    """Публичная заглушка-напоминание для алгоритмов РНМ по мощности других производителей БАВР."""

    algorithm_id = "bavr_manufacturer_power_stub"
    name = "BAVR Manufacturer Power PDR (Stub / Research Task)"
    is_public = True
    tunable_parameters = {"power_threshold_pu": 0.01}

    def compute(self, input_data: PDRInputData) -> PDROutput:
        return PDROutput(
            direction=PDRDirection.REVERSE,
            is_tripped=False,
            margin=0.0,
            confidence=0.0,
            diagnostics={"status": "stub_pending_research_implementation"},
        )


class ManufacturerCurrentPDRStub(PDRAlgorithm):
    """Публичная заглушка-напоминание для алгоритмов РНМ с токовыми адаптивностями других производителей."""

    algorithm_id = "bavr_manufacturer_current_stub"
    name = "BAVR Manufacturer Current PDR (Stub / Research Task)"
    is_public = True
    tunable_parameters = {"i_threshold_pu": 0.02}

    def compute(self, input_data: PDRInputData) -> PDROutput:
        return PDROutput(
            direction=PDRDirection.REVERSE,
            is_tripped=False,
            margin=0.0,
            confidence=0.0,
            diagnostics={"status": "stub_pending_research_implementation"},
        )
