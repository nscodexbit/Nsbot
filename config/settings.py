from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Settings:
    raw: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path = "config/default.yaml") -> "Settings":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(raw=yaml.safe_load(f))

    def section(self, name: str) -> Dict[str, Any]:
        return self.raw.get(name, {})
