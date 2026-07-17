from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mapi.data.validation import normalize_ohlcv


@dataclass
class MockPriceDataProvider:
    rows: int = 260
    seed: int = 42

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + sum(ord(ch) for ch in symbol))
        timestamps = pd.date_range("2025-01-01", periods=self.rows, freq="B", tz="UTC")
        returns = rng.normal(0.0005, 0.018, self.rows)
        close = 100.0 * np.exp(np.cumsum(returns))
        open_ = close * (1.0 + rng.normal(0.0, 0.003, self.rows))
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.012, self.rows))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.012, self.rows))
        volume = rng.lognormal(13.0, 0.35, self.rows).astype(int)
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        return normalize_ohlcv(frame)

