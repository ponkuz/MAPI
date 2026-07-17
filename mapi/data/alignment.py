from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlignmentResult:
    frame: pd.DataFrame
    valid: pd.Series
    staleness_seconds: pd.Series
    frequency_compatible: bool
    target_frequency_seconds: float | None
    source_frequency_seconds: float | None


def align_point_in_time(
    target_frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    max_staleness_bars: float = 1.5,
    frequency_tolerance: float = 1.5,
) -> AlignmentResult:
    """Backward-match source bars without crossing a UTC trading date."""

    target_index = pd.DatetimeIndex(pd.to_datetime(target_frame.index, utc=True))
    source_index = pd.DatetimeIndex(pd.to_datetime(source_frame.index, utc=True))
    target_frequency = _median_interval_seconds(target_index)
    source_frequency = _median_interval_seconds(source_index)
    compatible = bool(
        target_frequency is not None
        and source_frequency is not None
        and source_frequency <= target_frequency * frequency_tolerance
    )

    left = pd.DataFrame({"_target_timestamp": target_index}).sort_values(
        "_target_timestamp"
    )
    source_values = source_frame.copy()
    if "timestamp" in source_values.columns:
        source_values = source_values.drop(columns=["timestamp"])
    source_values = source_values.reset_index(drop=True)
    source_values.insert(0, "_source_timestamp", source_index)
    source_values = source_values.sort_values("_source_timestamp")
    merged = pd.merge_asof(
        left,
        source_values,
        left_on="_target_timestamp",
        right_on="_source_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    merged.index = target_index
    target_timestamp = pd.Series(merged["_target_timestamp"].array, index=target_index)
    source_timestamp = pd.Series(merged["_source_timestamp"].array, index=target_index)
    staleness = (target_timestamp - source_timestamp).dt.total_seconds()
    same_session = target_timestamp.dt.date == source_timestamp.dt.date
    if target_frequency is None:
        within_staleness = staleness == 0.0
    else:
        within_staleness = staleness <= target_frequency * max_staleness_bars
    valid = (
        source_timestamp.notna()
        & same_session
        & within_staleness.fillna(False)
        & compatible
    )
    data_columns = [
        column
        for column in merged.columns
        if column not in {"_target_timestamp", "_source_timestamp"}
    ]
    aligned = merged[data_columns].copy()
    aligned.loc[~valid, :] = np.nan
    aligned["matched_timestamp"] = source_timestamp
    return AlignmentResult(
        frame=aligned,
        valid=valid.astype(bool),
        staleness_seconds=staleness,
        frequency_compatible=compatible,
        target_frequency_seconds=target_frequency,
        source_frequency_seconds=source_frequency,
    )


def _median_interval_seconds(index: pd.DatetimeIndex) -> float | None:
    if len(index) < 2:
        return None
    differences = index.to_series().diff().dt.total_seconds()
    positive = differences[differences > 0.0]
    return float(positive.median()) if not positive.empty else None
