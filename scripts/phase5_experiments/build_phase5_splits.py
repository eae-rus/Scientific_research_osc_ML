"""Создать immutable research_strict split manifest без чтения сигналов."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.phase5_splits import assign_keys, write_split_manifest


def build(output_path: Path, seed: int = 42) -> dict[str, object]:
    open_manifest = json.loads(
        (PROJECT_ROOT / "data/phase5/open_ee_shards/manifest.json").read_text(encoding="utf-8")
    )
    open_keys = [str(record["source_csv"]) for record in open_manifest["records"]]
    french = np.load(
        PROJECT_ROOT / "data/phase5/french_rte/DATA_S.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    french_keys = [f"record_{index:05d}" for index in range(int(french.shape[0]))]
    manifest: dict[str, object] = {
        "version": 1,
        "protocol": "research_strict",
        "seed": seed,
        "validation_fraction": 0.1,
        "holdout_fraction": 0.1,
        "sources": {
            "open_ee": {
                "grouping": "source_csv",
                "splits": assign_keys(open_keys, "open_ee", seed=seed),
            },
            "french_rte": {
                "grouping": "record",
                "splits": assign_keys(french_keys, "french_rte", seed=seed),
            },
        },
    }
    return write_split_manifest(output_path, manifest)


def run_manual() -> None:
    OUTPUT_PATH = PROJECT_ROOT / "data/phase5/research_strict_splits.json"
    SEED = 42
    result = build(OUTPUT_PATH, SEED)
    print(json.dumps({
        "path": str(OUTPUT_PATH),
        "sha256": result["sha256"],
        "sizes": {
            source: {split: len(indices) for split, indices in value["splits"].items()}
            for source, value in result["sources"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_manual()
