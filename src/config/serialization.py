from __future__ import annotations

import dataclasses
from datetime import date, datetime
from typing import Any, Dict, Mapping, Sequence


def to_jsonable(obj: Any) -> Any:
    """
    Convert common Python objects (dataclasses, dates, mappings, sequences) into JSON-serializable forms.
    Deterministic: no time-based behavior, no lossy float formatting.
    """
    if dataclasses.is_dataclass(obj):
        return to_jsonable(dataclasses.asdict(obj))

    if isinstance(obj, (date, datetime)):
        return obj.isoformat()

    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    # primitives / None
    return obj
