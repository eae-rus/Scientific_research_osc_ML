"""Convert curated COMTRADE pairs into indexed, mmap-readable signal arrays."""

import json
from pathlib import Path
import re
import sys
from uuid import uuid4

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase5_experiments.audit_real_ozz_labels import LABEL
from osc_tools.core.comtrade_custom import Comtrade

RAW = ROOT / "raw_data/real_ozz"
NOMINALS = ROOT / "data/norm_coef_all_v1.4.csv"
SMOKE = True
PHASE_COLUMN = re.compile(r"(?:I \| Bus|U \| (?:BusBar|CableLine))-(\d+) \| phase: [ABCN]")


def sections(frame, negative):
    if negative:
        return {m[1]: np.zeros(frame.height, dtype=bool) for column in frame.columns
                if (m := PHASE_COLUMN.fullmatch(column))}
    labels = {}
    for column in frame.columns:
        match = LABEL.fullmatch(column.strip())
        if match:
            mask = frame[column].fill_null(0).to_numpy() > 0
            bus = match[1]
            labels[bus] = labels.get(bus, False) | mask
    return {bus: mask for bus, mask in labels.items() if mask.any()}


def normalize_section(frame, bus, nominal):
    raw = np.full((frame.height, 8), np.nan, dtype=np.float32)
    columns = {}
    for i, phase in enumerate("ABCN"):
        columns[i] = (f"I | Bus-{bus} | phase: {phase}", "Ip" if i < 3 else "Iz", 20 if i < 3 else 5)

    def base(field):
        value = nominal.get(f"{bus}{field}_base")
        return float(value) if value else np.nan

    def usable(column):
        return column in frame.columns and np.isfinite(frame[column].cast(pl.Float32).to_numpy()).any()

    candidates = [("BusBar", "Ub"), ("CableLine", "Uc")]
    candidates = [(location, field) for location, field in candidates
                  if np.isfinite(base(field)) and base(field) > 0]
    if candidates:
        location, field = max(candidates, key=lambda c: sum(
            usable(f"U | {c[0]}-{bus} | phase: {phase}") for phase in "ABC"))
        for i, phase in enumerate("ABCN", 4):
            columns[i] = (f"U | {location}-{bus} | phase: {phase}", field, 3)

    used = {}
    for i, (column, field, factor) in columns.items():
        scale = base(field)
        if usable(column) and np.isfinite(scale) and scale > 0:
            raw[:, i] = frame[column].cast(pl.Float32).to_numpy() / (scale * factor)
            used[str(i)] = column
    return raw, used


def prepare(inputs, output, *, smoke):
    jobs = [(cfg, negative) for directory, negative in inputs
            for cfg in sorted(directory.glob("*.cfg"))]
    if not jobs:
        raise ValueError("No COMTRADE inputs")
    if len({p.stem.casefold() for p, _ in jobs}) != len(jobs):
        raise ValueError("Duplicate original IDs across input folders")
    nominals = {r["name"]: r for r in pl.read_csv(NOMINALS, infer_schema=False).to_dicts()}
    generation = output / uuid4().hex
    generation.mkdir(parents=True)
    records, skipped = [], []
    for cfg, negative in jobs:
        try:
            nominal = nominals.get(cfg.stem, {})
            if "YES" not in str(nominal.get("norm")):
                raise ValueError("No approved normalization")
            recording = Comtrade()
            recording.load(str(cfg))
            frame = recording.to_dataframe()
            rates = {float(rate) for rate, _ in recording.cfg.sample_rates}
            frequency = float(recording.cfg.frequency)
            fs = next(iter(rates), 0.)
            if frequency <= 0 or len(rates) != 1 or fs < 8 * frequency or not np.allclose(
                np.diff(recording.time), 1 / fs, rtol=.02, atol=1e-7,
            ):
                raise ValueError("Unsupported sample grid")
            selected = sections(frame, negative)
            if not selected:
                raise ValueError("No eligible section")
            for bus, label in selected.items():
                raw, columns = normalize_section(frame, bus, nominal)
                if not np.isfinite(raw[:, [0, 1, 2, 4, 5, 6]]).any():
                    skipped.append(dict(id=cfg.stem, bus=bus, reason="No normalized phases"))
                    continue
                stem = generation / f"{cfg.stem}_{bus}"
                signal_path = stem.with_suffix(".signals.npy")
                target_path = stem.with_suffix(".target.npy")
                np.save(signal_path, raw, allow_pickle=False)
                np.save(target_path, label.astype(np.uint8), allow_pickle=False)
                records.append(dict(
                    id=cfg.stem, bus=bus, kind="hard_negative" if negative else "annotated",
                    fs=fs, f_network=frequency, samples=len(raw), positive_samples=int(label.sum()),
                    channels=columns, signals=signal_path.relative_to(output).as_posix(),
                    target=target_path.relative_to(output).as_posix(),
                ))
        except (ValueError, KeyError, IndexError, OSError) as exc:
            skipped.append(dict(id=cfg.stem, reason=str(exc)))
    manifest = dict(version=1, smoke=smoke, records=records, skipped=skipped)
    temporary = generation / "manifest.pending"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output / "manifest.json")
    print(f"Prepared {len(records)} sections, skipped {len(skipped)} entries: {output}")
    return manifest


if __name__ == "__main__":
    if SMOKE:
        from scripts.phase5_experiments.audit_real_ozz_labels import INPUT

        prepare([(INPUT, False), (RAW / "smoke/hard_negative", True)],
                ROOT / "data/phase5/real_ozz_smoke", smoke=True)
    else:
        prepare([(RAW / "annotated", False), (RAW / "hard_negative", True)],
                ROOT / "data/phase5/real_ozz_prepared", smoke=False)
