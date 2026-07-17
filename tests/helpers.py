from __future__ import annotations

import numpy as np
import pandas as pd

from mapi.config import MapiConfig
from mapi.models import HorizonConfig


def make_ohlcv(rows: int = 180, seed: int = 42, start_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-02", periods=rows, freq="B", tz="UTC")
    daily_returns = rng.normal(0.0004, 0.012, size=rows)
    close = start_price * np.exp(np.cumsum(daily_returns))
    open_price = np.r_[start_price, close[:-1]] * (1.0 + rng.normal(0.0, 0.002, rows))
    intraday_range = rng.uniform(0.003, 0.018, rows)
    high = np.maximum(open_price, close) * (1.0 + intraday_range)
    low = np.minimum(open_price, close) * (1.0 - intraday_range)
    volume = rng.integers(750_000, 2_500_000, size=rows).astype(float)
    if rows >= 30:
        volume[-3] *= 5.0
        close[-3] = close[-4] * 1.001
        high[-3] = max(open_price[-3], close[-3]) * 1.01
        low[-3] = min(open_price[-3], close[-3]) * 0.99
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def small_config(include_cross_asset: bool = True) -> MapiConfig:
    config = MapiConfig()
    config.horizons = {
        "short_term": HorizonConfig(
            "short_term", return_window=3, rolling_window=30, min_periods=10
        )
    }
    if not include_cross_asset:
        config.enabled_components = [
            "price_volume_divergence",
            "momentum_disagreement",
            "volatility_anomaly",
        ]
    config.redundancy_window = 24
    return config


def mutate_after(frame: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    changed = frame.copy()
    rows = changed.index >= cutoff
    changed.loc[rows, ["open", "high", "low", "close"]] *= 3.5
    changed.loc[rows, "volume"] *= 7.0
    return changed
