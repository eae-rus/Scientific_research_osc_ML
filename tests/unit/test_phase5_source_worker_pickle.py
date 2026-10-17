"""Регрессии Windows DataLoader spawn для ленивых Phase 5 sources."""

import json
import pickle
from pathlib import Path

import numpy as np

from osc_tools.ml.phase5_sources import FrenchRTESource, OpenEEShardedSource


def test_open_ee_source_drops_open_npz_cache_before_pickle(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    np.savez(
        tmp_path / "shard.npz",
        signals=np.arange(32, dtype=np.float32).reshape(4, 8),
        offsets=np.asarray([0, 4], dtype=np.int64),
        provenance=np.ones((1, 8), dtype=np.uint8),
    )
    (manifest_dir / "manifest.json").write_text(
        json.dumps({
            "kind": "open_ee_sharded",
            "records": [{"shard_path": "shard.npz", "local_index": 0}],
        }),
        encoding="utf-8",
    )
    source = OpenEEShardedSource(manifest_dir / "manifest.json")
    expected = source.load_signal(0)
    assert source._cache

    restored = pickle.loads(pickle.dumps(source))

    assert not restored._cache
    assert np.array_equal(restored.load_signal(0), expected)
    source.close()
    restored.close()


def test_french_source_reopens_memmap_in_worker(tmp_path: Path) -> None:
    path = tmp_path / "french.npy"
    np.save(path, np.arange(24, dtype=np.float32).reshape(1, 6, 4))
    source = FrenchRTESource(path)
    expected = source.load_signal(0)

    payload = pickle.dumps(source)
    restored = pickle.loads(payload)

    assert len(payload) < 10_000
    assert isinstance(restored.data, np.memmap)
    assert np.array_equal(restored.load_signal(0), expected)
