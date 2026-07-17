from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-12


def safe_divide(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    return numerator / (pd.Series(denominator, index=numerator.index) + EPSILON)


def rolling_zscore(
    series: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    min_periods = min_periods or max(3, window // 4)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / (std.replace(0.0, np.nan) + EPSILON)


def historical_zscore(
    series: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    min_periods = min_periods or max(3, window // 4)
    mean = series.rolling(window, min_periods=min_periods).mean().shift(1)
    std = series.rolling(window, min_periods=min_periods).std(ddof=0).shift(1)
    return (series - mean) / (std.replace(0.0, np.nan) + EPSILON)


def rolling_percentile_rank(
    series: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    min_periods = min_periods or max(3, window // 4)

    def rank_last(values: np.ndarray) -> float:
        clean = values[np.isfinite(values)]
        if len(clean) == 0:
            return np.nan
        current = clean[-1]
        return float(np.mean(clean <= current))

    return series.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def robust_unit_score_from_z(zscore: pd.Series, cap: float = 3.0) -> pd.Series:
    return (zscore.abs() / cap).clip(lower=0.0, upper=1.0).fillna(0.0)


def signed_unit_from_z(zscore: pd.Series, scale: float = 2.0) -> pd.Series:
    return pd.Series(np.tanh(zscore.fillna(0.0) / scale), index=zscore.index).clip(-1.0, 1.0)


def clipped01(series: pd.Series | float) -> pd.Series | float:
    if isinstance(series, pd.Series):
        return series.clip(lower=0.0, upper=1.0).fillna(0.0)
    if not np.isfinite(series):
        return 0.0
    return float(min(1.0, max(0.0, series)))


def close_location_value(frame: pd.DataFrame) -> pd.Series:
    span = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    return (((frame["close"] - frame["low"]) / (span + EPSILON)) - 0.5).clip(-0.5, 0.5)


def true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)

