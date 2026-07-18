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


def prior_percentile_rank(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
    tie_tolerance: float = 1e-12,
) -> pd.Series:
    """Rank the current value against prior-only history using mid-rank ties."""

    min_periods = min_periods or max(3, window // 4)
    values = series.astype(float).to_numpy()
    output = np.full(len(values), np.nan, dtype=float)
    for index, current in enumerate(values):
        if not np.isfinite(current):
            continue
        history = values[max(0, index - window) : index]
        history = history[np.isfinite(history)]
        if len(history) < min_periods:
            continue
        ties = np.isclose(history, current, rtol=0.0, atol=tie_tolerance)
        less = history < (current - tie_tolerance)
        output[index] = (float(less.sum()) + 0.5 * float(ties.sum())) / len(history)
    return pd.Series(output, index=series.index, name="historical_extremeness")


def event_novelty(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
    recurrence_tolerance: float = 0.05,
    tie_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Separate prior-only extremeness from recurrence-adjusted event novelty."""

    min_periods = min_periods or max(3, window // 4)
    extremeness = prior_percentile_rank(
        series,
        window=window,
        min_periods=min_periods,
        tie_tolerance=tie_tolerance,
    )
    values = series.astype(float).to_numpy()
    recurrence = np.full(len(values), np.nan, dtype=float)
    for index, current in enumerate(values):
        if not np.isfinite(current):
            continue
        history = values[max(0, index - window) : index]
        history = history[np.isfinite(history)]
        if len(history) < min_periods:
            continue
        recurrence[index] = float(
            np.mean(np.abs(history - current) <= recurrence_tolerance)
        )
    recurrence_rate = pd.Series(
        recurrence, index=series.index, name="recurrence_rate"
    )
    novelty = (extremeness * (1.0 - recurrence_rate)).clip(0.0, 1.0)
    novelty.name = "novelty"
    return pd.concat([extremeness, recurrence_rate, novelty], axis=1)


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
