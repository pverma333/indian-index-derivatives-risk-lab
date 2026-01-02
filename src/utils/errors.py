from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DataIntegrityError(ValueError):
    """
    Raised when a DataFrame violates an invariant required by the engine.
    Keeps a small, human-readable summary plus optional structured details.
    """
    message: str
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        if not self.details:
            return self.message
        return f"{self.message} | details={self.details}"


def _sample_indices(mask, max_n: int = 10) -> list[int]:
    try:
        return list(mask[mask].index[:max_n])
    except Exception:
        return []
