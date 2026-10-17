"""Паспорт feature contract в Phase 5 checkpoints/configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

from .phase5_contracts import CHANNEL_ORDER
from .spectral_features import FeatureSchema


@dataclass(frozen=True)
class FeatureContractPassport:
    feature_contract: str
    feature_names: tuple[str, ...]
    channel_order: tuple[str, ...]
    temporal_mode: str
    cyclic_angle_encoding: bool
    use_provenance_embedding: bool
    schema_sha256: str

    @classmethod
    def create(
        cls,
        schema: FeatureSchema,
        temporal_mode: str,
        cyclic_angle_encoding: bool,
        use_provenance_embedding: bool,
    ) -> "FeatureContractPassport":
        payload = {
            "feature_contract": schema.contract_name,
            "feature_names": list(schema.names),
            "channel_order": list(CHANNEL_ORDER),
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return cls(
            schema.contract_name,
            schema.names,
            CHANNEL_ORDER,
            temporal_mode,
            cyclic_angle_encoding,
            use_provenance_embedding,
            digest,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def assert_compatible(self, stored: Mapping[str, object]) -> None:
        """Дать понятную ошибку до попытки загрузить несовместимый state_dict."""

        expected = self.to_dict()
        mismatches = [
            key for key in expected
            if _normalized(stored.get(key)) != _normalized(expected[key])
        ]
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={stored.get(key)!r}, expected={expected[key]!r}"
                for key in mismatches
                if key != "feature_names"
            )
            if "feature_names" in mismatches:
                details += (", " if details else "") + "feature_names: различается порядок/состав"
            raise ValueError(f"Несовместимый Phase 5 feature contract ({details})")


def _normalized(value: object) -> object:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return value
