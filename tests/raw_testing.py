from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union



class ConfigError(ValueError):
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


raise ConfigError(
    "Strategy params validation failed",
    errors=["qty_lots must be >= 1", "liquidity_mode invalid"]
)

