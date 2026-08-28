"""Canonical, secret-free fingerprints for recipes and source slices."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping
from uuid import UUID

from .domain import DomainValidationError


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise DomainValidationError("canonical timestamps must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise DomainValidationError("canonical decimal values must be finite")
        return format(value, "f")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DomainValidationError("canonical float values must be finite")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DomainValidationError("canonical mapping keys must be strings")
            normalized[key] = _normalize(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise DomainValidationError(
        f"unsupported canonical value type: {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    """Serialize supported values deterministically for hashing and audits."""
    return json.dumps(
        _normalize(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_fingerprint(value: Any) -> str:
    """Return the lowercase SHA-256 digest of canonical UTF-8 JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
