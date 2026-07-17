from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mapi.data.validation import normalize_ohlcv


@dataclass
class CsvPriceDataProvider:
    root: str | Path | None = None
    symbol_paths: dict[str, str | Path] = field(default_factory=dict)

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        path = self._resolve_path(symbol)
        frame = pd.read_csv(path)
        return normalize_ohlcv(frame)

    def _resolve_path(self, symbol: str) -> Path:
        if symbol in self.symbol_paths:
            return Path(self.symbol_paths[symbol])
        if self.root is None:
            raise ValueError(f"No CSV path configured for symbol {symbol}")
        root = Path(self.root)
        candidates = [
            root / f"{symbol}.csv",
            root / f"{symbol.lower()}.csv",
            root / f"{symbol.upper()}.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No CSV file found for symbol {symbol} under {root}")

