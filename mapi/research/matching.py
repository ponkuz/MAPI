from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrequencyMatch:
    threshold: float
    target_frequency: float
    fitted_frequency: float
    tie_fraction: float = 1.0


@dataclass(frozen=True)
class EligibilityExclusions:
    low_confidence: int
    low_quality: int
    frequency_mismatch: int


def chronological_masks(
    index: pd.Index, fit_fraction: float = 0.70
) -> tuple[pd.Series, pd.Series]:
    if len(index) < 2:
        raise ValueError("At least two timestamps are required for a chronological split")
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between zero and one")
    split = max(1, min(len(index) - 1, int(len(index) * fit_fraction)))
    fit = pd.Series(False, index=index)
    fit.iloc[:split] = True
    return fit, ~fit


def partition_metadata(
    index: pd.Index,
    fit_mask: pd.Series,
    test_mask: pd.Series,
    requested_fit_fraction: float,
) -> dict[str, object]:
    fit = fit_mask.reindex(index, fill_value=False).astype(bool)
    test = test_mask.reindex(index, fill_value=False).astype(bool)
    if bool((fit & test).any()) or not bool((fit | test).all()):
        raise ValueError("fit_mask and test_mask must be disjoint and cover the index")

    def bounds(mask: pd.Series) -> tuple[object | None, object | None]:
        selected = index[mask.to_numpy()]
        return (selected[0], selected[-1]) if len(selected) else (None, None)

    fit_start, fit_end = bounds(fit)
    test_start, test_end = bounds(test)
    total = len(index)
    return {
        "requested_fit_fraction": requested_fit_fraction,
        "fit_fraction": float(fit.sum() / total) if total else 0.0,
        "test_fraction": float(test.sum() / total) if total else 0.0,
        "fit_count": int(fit.sum()),
        "test_count": int(test.sum()),
        "fit_start": fit_start,
        "fit_end": fit_end,
        "test_start": test_start,
        "test_end": test_end,
    }


def evidence_eligibility_mask(
    signals: pd.DataFrame,
    mask: pd.Series,
    min_confidence: float = 0.0,
    min_data_quality: float = 0.0,
    require_frequency_compatible: bool = False,
) -> pd.Series:
    evaluation = mask.reindex(signals.index, fill_value=False).astype(bool)
    confidence = _column(signals, "mapi_confidence", 1.0).astype(float).fillna(0.0)
    quality = _column(signals, "data_quality_score", 1.0).astype(float).fillna(0.0)
    frequency = _column(
        signals, "horizon_frequency_compatible", True
    ).fillna(False).astype(bool)
    eligible = evaluation & (confidence >= min_confidence) & (quality >= min_data_quality)
    if require_frequency_compatible:
        eligible &= frequency
    return eligible


def candidate_mask(
    signals: pd.DataFrame,
    score_column: str,
    threshold: float,
    min_direction: float,
    mask: pd.Series,
    min_confidence: float = 0.0,
    min_data_quality: float = 0.0,
    require_frequency_compatible: bool = False,
    direction_column: str = "mapi_direction",
) -> pd.Series:
    if score_column not in signals:
        raise ValueError(f"Score column is unavailable: {score_column}")
    if direction_column not in signals:
        raise ValueError(f"Direction column is unavailable: {direction_column}")
    score = signals[score_column].astype(float).replace([np.inf, -np.inf], np.nan)
    direction = signals[direction_column].astype(float).fillna(0.0)
    return (
        evidence_eligibility_mask(
            signals,
            mask,
            min_confidence,
            min_data_quality,
            require_frequency_compatible,
        )
        & score.notna()
        & (score >= threshold)
        & (direction.abs() >= min_direction)
    )


def candidate_exclusions(
    signals: pd.DataFrame,
    score_column: str,
    threshold: float,
    min_direction: float,
    mask: pd.Series,
    min_confidence: float,
    min_data_quality: float,
    require_frequency_compatible: bool,
    direction_column: str = "mapi_direction",
    selection_mask: pd.Series | None = None,
) -> EligibilityExclusions:
    evaluation = mask.reindex(signals.index, fill_value=False).astype(bool)
    direction = signals[direction_column].astype(float).fillna(0.0)
    if selection_mask is None:
        score = signals[score_column].astype(float).replace([np.inf, -np.inf], np.nan)
        raw = evaluation & score.notna() & (score >= threshold)
    else:
        raw = evaluation & selection_mask.reindex(signals.index, fill_value=False)
    raw &= direction.abs() >= min_direction

    confidence = _column(signals, "mapi_confidence", 1.0).astype(float).fillna(0.0)
    quality = _column(signals, "data_quality_score", 1.0).astype(float).fillna(0.0)
    frequency = _column(
        signals, "horizon_frequency_compatible", True
    ).fillna(False).astype(bool)
    low_confidence = raw & (confidence < min_confidence)
    after_confidence = raw & ~low_confidence
    low_quality = after_confidence & (quality < min_data_quality)
    after_quality = after_confidence & ~low_quality
    frequency_mismatch = (
        after_quality & ~frequency
        if require_frequency_compatible
        else pd.Series(False, index=signals.index)
    )
    return EligibilityExclusions(
        low_confidence=int(low_confidence.sum()),
        low_quality=int(low_quality.sum()),
        frequency_mismatch=int(frequency_mismatch.sum()),
    )


def candidate_frequency(
    signals: pd.DataFrame,
    score_column: str,
    threshold: float,
    min_direction: float,
    mask: pd.Series,
    min_confidence: float = 0.0,
    min_data_quality: float = 0.0,
    require_frequency_compatible: bool = False,
    direction_column: str = "mapi_direction",
    frequency_match: FrequencyMatch | None = None,
) -> float:
    selected = (
        apply_frequency_match(
            signals,
            score_column,
            frequency_match,
            min_direction,
            mask,
            min_confidence,
            min_data_quality,
            require_frequency_compatible,
            direction_column,
        )
        if frequency_match is not None
        else candidate_mask(
            signals,
            score_column,
            threshold,
            min_direction,
            mask,
            min_confidence,
            min_data_quality,
            require_frequency_compatible,
            direction_column,
        )
    )
    denominator = int(mask.reindex(signals.index, fill_value=False).sum())
    return float(selected.sum() / denominator) if denominator else 0.0


def fit_frequency_matched_threshold(
    signals: pd.DataFrame,
    score_column: str,
    target_frequency: float,
    min_direction: float,
    fit_mask: pd.Series,
    min_confidence: float = 0.0,
    min_data_quality: float = 0.0,
    require_frequency_compatible: bool = False,
    direction_column: str = "mapi_direction",
) -> FrequencyMatch:
    if not 0.0 <= target_frequency <= 1.0:
        raise ValueError("target_frequency must be in [0, 1]")
    scores = signals[score_column].astype(float).replace([np.inf, -np.inf], np.nan)
    direction = signals[direction_column].astype(float).fillna(0.0)
    eligible = evidence_eligibility_mask(
        signals,
        fit_mask,
        min_confidence,
        min_data_quality,
        require_frequency_compatible,
    ) & (direction.abs() >= min_direction) & scores.notna()
    values = scores[eligible]
    denominator = int(fit_mask.reindex(signals.index, fill_value=False).sum())
    desired = min(len(values), int(round(target_frequency * denominator)))
    if target_frequency > 0.0 and desired == 0 and len(values) > 0:
        desired = 1
    if values.empty or desired == 0:
        maximum = float(values.max()) if not values.empty else 100.0
        return FrequencyMatch(
            float(np.nextafter(maximum, np.inf)), target_frequency, 0.0, 0.0
        )

    ordered = values.sort_values(ascending=False, kind="mergesort")
    threshold = float(ordered.iloc[desired - 1])
    strict_count = int((values > threshold).sum())
    tie_count = int((values == threshold).sum())
    ties_needed = max(0, desired - strict_count)
    tie_fraction = float(ties_needed / tie_count) if tie_count else 0.0
    fitted = float(desired / denominator) if denominator else 0.0
    return FrequencyMatch(threshold, target_frequency, fitted, tie_fraction)


def apply_frequency_match(
    signals: pd.DataFrame,
    score_column: str,
    match: FrequencyMatch,
    min_direction: float,
    mask: pd.Series,
    min_confidence: float = 0.0,
    min_data_quality: float = 0.0,
    require_frequency_compatible: bool = False,
    direction_column: str = "mapi_direction",
) -> pd.Series:
    score = signals[score_column].astype(float).replace([np.inf, -np.inf], np.nan)
    direction = signals[direction_column].astype(float).fillna(0.0)
    eligible = evidence_eligibility_mask(
        signals,
        mask,
        min_confidence,
        min_data_quality,
        require_frequency_compatible,
    ) & (direction.abs() >= min_direction) & score.notna()
    selected = eligible & (score > match.threshold)
    ties = eligible & (score == match.threshold)
    tie_count = int(ties.sum())
    ties_to_select = int(round(match.tie_fraction * tie_count))
    if match.tie_fraction > 0.0 and tie_count > 0:
        ties_to_select = max(1, ties_to_select)
    if ties_to_select > 0:
        tie_positions = np.flatnonzero(ties.to_numpy())[:ties_to_select]
        selected.iloc[tie_positions] = True
    return selected


def test_only_signals(signals: pd.DataFrame, test_mask: pd.Series) -> pd.DataFrame:
    output = signals.copy()
    outside = ~test_mask.reindex(output.index, fill_value=False)
    for column in ("mapi_direction", "mapi_forecast_direction"):
        if column in output:
            output.loc[outside, column] = 0.0
    return output


def _column(signals: pd.DataFrame, name: str, default: object) -> pd.Series:
    if name in signals:
        return signals[name]
    return pd.Series(default, index=signals.index)
