"""Grouped partition and lazy loading of prepared real OZZ sections."""

import hashlib
import json
from pathlib import Path

import numpy as np


def partition_real(manifest_path, seed):
    path = Path(manifest_path).resolve()
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if manifest.get("version") != 1 or manifest.get("smoke"):
        raise ValueError("Prepare the production OZZ dataset with SMOKE=False")
    splits = {name: [] for name in ("train", "validation", "test")}
    assignments = {}
    for kind, group in (("annotated", 5), ("hard_negative", 6)):
        rows = [r for r in manifest["records"] if r["kind"] == kind]
        ids = sorted({r["id"].casefold() for r in rows})
        if len(ids) < 3:
            raise ValueError(f"{kind}: need at least three original records for disjoint splits")
        ordered = sorted(ids, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
        n_train = min(len(ids)-2, max(1, round(.4*len(ids))))
        n_val = min(len(ids)-n_train-1, max(1, round(.1*len(ids))))
        for index, key in enumerate(ordered):
            if key in assignments:
                raise ValueError(f"Conflicting source categories for {key}")
            assignments[key] = "train" if index < n_train else "validation" if index < n_train+n_val else "test"
        for row in rows:
            record = dict(row, kind="prepared_ozz", group=group,
                          name=f"{row['id']}_Bus {row['bus']}", prepared_root=str(path.parent))
            splits[assignments[row["id"].casefold()]].append(record)
    return splits, set(assignments), hashlib.sha256(payload).hexdigest()


def load_section(record):
    root = Path(record["prepared_root"])
    raw = np.load(root / record["signals"], mmap_mode="r", allow_pickle=False)
    label = np.load(root / record["target"], mmap_mode="r", allow_pickle=False)
    if raw.shape != (len(label), 8) or len(label) != record["samples"]:
        raise ValueError(f"Invalid prepared arrays for {record['name']}")
    return raw, label
