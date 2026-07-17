from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InMemoryCache:
    values: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any | None:
        return self.values.get(key)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

