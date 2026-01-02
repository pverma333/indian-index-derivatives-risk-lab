from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class DataIntegrityError(ValueError):
    rule: str
    details: str
    violations_preview: Optional[pd.DataFrame] = None

    def __str__(self) -> str:
        msg = f"[{self.rule}] {self.details}"
        if self.violations_preview is not None and not self.violations_preview.empty:
            msg += "\nFirst 20 violations:\n" + self.violations_preview.to_string(index=False)
        return msg
