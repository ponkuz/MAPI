from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mapi.models import HorizonConfig


@dataclass(frozen=True)
class FrequencyValidation:
    median_interval_seconds: float | None
    observed_frequency: str
    compatible: bool
    warning: str | None


def validate_horizon_frequency(
    index: pd.Index, horizon: HorizonConfig
) -> FrequencyValidation:
    timestamps = pd.Series(pd.to_datetime(index, utc=True)).sort_values()
    gaps = timestamps.diff().dt.total_seconds()
    gaps = gaps[(gaps > 0.0) & np.isfinite(gaps)]
    if gaps.empty:
        return FrequencyValidation(None, "unknown", False, "Input frequency is unknown")
    median_seconds = float(gaps.median())
    observed = "intraday" if median_seconds < 20.0 * 60.0 * 60.0 else "daily"
    expected = horizon.expected_frequency
    compatible = expected == "any" or expected == observed
    warning = None
    if not compatible:
        warning = (
            f"Horizon '{horizon.name}' expects {expected} bars but the median input "
            f"interval is {median_seconds:g} seconds ({observed})"
        )
    return FrequencyValidation(median_seconds, observed, compatible, warning)
