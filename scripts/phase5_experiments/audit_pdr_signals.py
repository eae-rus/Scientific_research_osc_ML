"""Скрипт аудита полноты сигналов и возможности вычисления I_B для РНМ в Phase 5.

Проверяет пригодность сигналов в Open_EE и French/RTE датасетах для запуска
фазных и прямой последовательности алгоритмов РНМ.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

from osc_tools.ml.phase5_sources import OpenEEShardedSource, FrenchRTESource
from osc_tools.pdr.signal_analysis import check_pdr_signal_sufficiency, derive_missing_currents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pdr_signal_audit")


def audit_open_ee(manifest_path: Path) -> dict:
    """Аудит датасета Open_EE."""
    if not manifest_path.exists():
        logger.warning(f"Манифест Open_EE не найден: {manifest_path}")
        return {"status": "missing"}

    source = OpenEEShardedSource(manifest_path)
    total = len(source)
    logger.info(f"Запуск аудита Open_EE ({total} записей)...")

    can_phase = 0
    can_pos_seq = 0
    ib_derived = 0
    missing_phase_u = 0

    for i in range(total):
        provenance = source.get_provenance(i)
        meta = source.get_metadata(i)
        v_basis = str(meta.get("voltage_basis", "phase"))

        # Проверка $I_B$
        dummy_signals = np.zeros((8, 100), dtype=np.float32)
        _, new_prov = derive_missing_currents(dummy_signals, provenance)

        res = check_pdr_signal_sufficiency(new_prov, voltage_basis=v_basis)
        if res.can_run_phase_pdr:
            can_phase += 1
        if res.can_run_pos_seq_pdr:
            can_pos_seq += 1
        if "IB" in res.derived_channels:
            ib_derived += 1
        if "UA" in res.missing_channels or "UB" in res.missing_channels or "UC" in res.missing_channels:
            missing_phase_u += 1

    source.close()

    report = {
        "dataset": "open_ee",
        "total_records": total,
        "can_run_phase_pdr": can_phase,
        "can_run_pos_seq_pdr": can_pos_seq,
        "ib_derived_count": ib_derived,
        "missing_phase_u_count": missing_phase_u,
        "phase_pdr_coverage_pct": round(100.0 * can_phase / max(1, total), 2),
        "pos_seq_pdr_coverage_pct": round(100.0 * can_pos_seq / max(1, total), 2),
    }
    logger.info(f"Open_EE аудит завершён: {report['phase_pdr_coverage_pct']}% пригодно для фазного РНМ")
    return report


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent.parent
    open_ee_manifest = root_dir / "data" / "phase5" / "open_ee_shards" / "manifest.json"

    import numpy as np
    global np

    report_open_ee = audit_open_ee(open_ee_manifest)

    reports_dir = root_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_file = reports_dir / "pdr_signal_audit.json"

    out_file.write_text(json.dumps({"open_ee": report_open_ee}, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Отчёт по аудиту сохранён в {out_file}")


if __name__ == "__main__":
    main()
