"""Research-strict split invariants."""

from osc_tools.ml.phase5_splits import assign_keys, split_manifest_hash


def test_group_keys_never_cross_splits() -> None:
    keys = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]
    splits = assign_keys(keys, "source", validation_fraction=0.2, holdout_fraction=0.2, seed=3)
    membership = {}
    for split, indices in splits.items():
        for index in indices:
            membership.setdefault(keys[index], set()).add(split)
    assert all(len(value) == 1 for value in membership.values())
    assert sorted(sum(splits.values(), [])) == list(range(len(keys)))


def test_manifest_hash_ignores_existing_hash_field() -> None:
    manifest = {"version": 1, "sources": {"x": [1, 2]}}
    digest = split_manifest_hash(manifest)
    assert split_manifest_hash(manifest | {"sha256": digest}) == digest
