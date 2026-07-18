from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrequencyMatch:
    threshold: float
    target_frequency: float
    fitted_frequency: float


def chronological_masks(
    index: pd.Index, fit_fraction: float = 0.70
) -> tuple[pd.Series, pd.Series]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between zero and one")
    split = max(1, min(len(index) - 1, int(len(index) * fit_fraction)))
    fit = pd.Series(False, index=index)
    fit.iloc[:split] = True
    return fit, ~fit


def candidate_frequency(
    signals: pd.DataFrame,
    score_column: str,
    threshold: float,
    min_direction: float,
    mask: pd.Series,
) -> float:
    eligible = (
        (signals[score_column].fillna(0.0) >= threshold)
        & (signals["mapi_direction"].fillna(0.0).abs() >= min_direction)
        & mask.reindex(signals.index, fill_value=False)
    )
    denominator = int(mask.reindex(signals.index, fill_value=False).sum())
    return float(eligible.sum() / denominator) if denominator else 0.0


def fit_frequency_matched_threshold(
    signals: pd.DataFrame,
    score_column: str,
    target_frequency: float,
    min_direction: float,
    fit_mask: pd.Series,
) -> FrequencyMatch:
    if not 0.0 <= target_frequency <= 1.0:
        raise ValueError("target_frequency must be in [0, 1]")
    scores = signals[score_column].astype(float).replace([np.inf, -np.inf], np.nan)
    eligible = (
        fit_mask.reindex(signals.index, fill_value=False)
        & (signals["mapi_direction"].fillna(0.0).abs() >= min_direction)
        & scores.notna()
    )
    values = scores[eligible]
    if values.empty:
        return FrequencyMatch(
            float(np.nextafter(100.0, np.inf)), target_frequency, 0.0
        )
    thresholds = np.unique(values.to_numpy(dtype=float))
    thresholds = np.append(thresholds, np.nextafter(thresholds.max(), np.inf))
    candidates = [
        (
            abs(
                candidate_frequency(
                    signals, score_column, float(value), min_direction, fit_mask
                )
                - target_frequency
            ),
            -float(value),
            float(value),
        )
        for value in thresholds
    ]
    threshold = min(candidates)[2]
    fitted = candidate_frequency(
        signals, score_column, threshold, min_direction, fit_mask
    )
    return FrequencyMatch(threshold, target_frequency, fitted)


def test_only_signals(signals: pd.DataFrame, test_mask: pd.Series) -> pd.DataFrame:
    output = signals.copy()
    outside = ~test_mask.reindex(output.index, fill_value=False)
    output.loc[outside, "mapi_direction"] = 0.0
    return output
