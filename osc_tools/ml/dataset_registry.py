"""Версионированный registry источников Phase 5."""

from __future__ import annotations

import json
from pathlib import Path

from .phase5_sources import DatasetSource, FrenchRTESource, OpenEEShardedSource


def load_registry(path: Path) -> dict[str, object]:
    """Загрузить registry и проверить обязательные поля источников."""

    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("version") != 1 or not isinstance(registry.get("sources"), dict):
        raise ValueError("Неподдерживаемый datasets_registry")
    return registry


def create_source(registry_path: Path, name: str) -> DatasetSource:
    """Создать reader только для подготовленного random-access источника."""

    source = load_registry(registry_path)["sources"].get(name)
    if source is None:
        raise KeyError(f"Источник {name!r} не зарегистрирован")
    root = Path(registry_path).resolve().parents[2]
    kind = source["kind"]
    if kind == "open_ee_sharded":
        return OpenEEShardedSource(root / source["path"] / "manifest.json")
    if kind == "french_rte_prepared":
        prepared = source.get("prepared_path")
        if not prepared:
            raise ValueError("French source не готов: prepared_path не задан")
        return FrenchRTESource(root / prepared, source.get("current_nominal_a"))
    raise ValueError(f"Неизвестный kind источника: {kind}")
