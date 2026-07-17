from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def compute_redundancy_penalties(
    component_frames: dict[str, pd.DataFrame],
    window: int,
    threshold: float,
    minimum_penalty: float,
    min_periods: int = 8,
) -> dict[str, pd.Series]:
    if not component_frames:
        return {}
    first_index = next(iter(component_frames.values())).index
    penalties = {
        name: pd.Series(1.0, index=first_index, dtype=float)
        for name in component_frames
    }
    for left_name, right_name in itertools.combinations(component_frames, 2):
        left = component_frames[left_name]["anomaly_strength"].fillna(0.0)
        right = component_frames[right_name]["anomaly_strength"].fillna(0.0)
        corr = (
            left.rolling(window, min_periods=max(min_periods, window // 4))
            .corr(right)
            .shift(1)
            .abs()
        )
        overlap = corr > threshold
        reduction = pd.Series(np.where(overlap.fillna(False), 0.75, 1.0), index=left.index)
        penalties[left_name] = (penalties[left_name] * reduction).clip(
            lower=minimum_penalty, upper=1.0
        )
        penalties[right_name] = (penalties[right_name] * reduction).clip(
            lower=minimum_penalty, upper=1.0
        )
    return penalties
