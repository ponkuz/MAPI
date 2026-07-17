from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"OHLCV data is missing required columns: {missing}")
    output = frame.copy()
    if "timestamp" in output.index.names:
        output = output.reset_index(drop=True)
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output[["open", "high", "low", "close", "volume"]] = output[
        ["open", "high", "low", "close", "volume"]
    ].replace([np.inf, -np.inf], np.nan)
    output = output.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    output = output.set_index("timestamp", drop=False)
    return output


def validate_ohlcv(frame: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        warnings.append(f"Missing required columns: {missing}")
        return warnings
    if frame[REQUIRED_OHLCV_COLUMNS].isna().any().any():
        warnings.append("OHLCV data contains missing values")
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        warnings.append("OHLCV data contains non-finite numeric values")
    if (frame["volume"] < 0).any():
        warnings.append("Volume contains negative values")
    if (frame["volume"] == 0).any():
        warnings.append("Volume contains zero values; volume-based evidence is incomplete")
    if ((frame["high"] < frame["low"]) | (frame["high"] < frame["close"]) | (frame["low"] > frame["close"])).any():
        warnings.append("Some OHLC rows have inconsistent high/low/close values")
    if not frame["timestamp"].is_monotonic_increasing:
        warnings.append("Timestamps are not monotonic increasing before normalization")
    if frame["timestamp"].duplicated().any():
        warnings.append("Duplicate timestamps will be collapsed during normalization")
    close = pd.to_numeric(frame["close"], errors="coerce")
    close_ratio = close / close.shift(1)
    if ((close_ratio >= 1.8) | (close_ratio <= 1.0 / 1.8)).any():
        warnings.append(
            "Split-like price discontinuity detected; MAPI requires split- and dividend-adjusted OHLCV"
        )
    return warnings


def ohlcv_quality_score(
    frame: pd.DataFrame,
    rolling_window: int,
    penalize_split_like: bool = True,
) -> pd.Series:
    required_present = frame[["open", "high", "low", "close", "volume"]].notna().all(axis=1)
    positive_prices = (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
    positive_volume = frame["volume"] > 0
    consistent_range = (
        (frame["high"] >= frame["low"])
        & (frame["high"] >= frame[["open", "close"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close"]].min(axis=1))
    )
    close_ratio = frame["close"] / frame["close"].shift(1)
    no_split_like_jump = close_ratio.between(1.0 / 1.8, 1.8).fillna(True)
    if not penalize_split_like:
        no_split_like_jump = pd.Series(True, index=frame.index)
    row_quality = (
        required_present.astype(float)
        + positive_prices.astype(float)
        + positive_volume.astype(float)
        + consistent_range.astype(float)
        + no_split_like_jump.astype(float)
    ) / 5.0
    coverage = required_present.rolling(rolling_window, min_periods=1).mean()
    return (row_quality * 0.75 + coverage * 0.25).clip(0.0, 1.0)
