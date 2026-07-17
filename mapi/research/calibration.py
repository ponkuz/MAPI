from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ConfidenceCalibrationModel:
    bins: tuple[float, ...]
    success_rates: tuple[float, ...]
    fitted_start: pd.Timestamp
    fitted_end: pd.Timestamp

    def transform(self, confidence: pd.Series) -> pd.Series:
        output = pd.Series(0.0, index=confidence.index, dtype=float)
        clipped = confidence.clip(0.0, 1.0)
        for index, (left, right) in enumerate(zip(self.bins[:-1], self.bins[1:])):
            in_bucket = (clipped >= left) & (clipped < right)
            output.loc[in_bucket] = self.success_rates[index]
        return output


def confidence_calibration_table(
    signals: pd.DataFrame,
    realized_success: pd.Series,
    bins: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.01),
) -> pd.DataFrame:
    """Evaluate fixed confidence buckets; it does not fit a mapping."""
    frame = pd.DataFrame(
        {
            "confidence": signals["mapi_confidence"].clip(0.0, 1.0),
            "success": realized_success.astype(float),
        }
    ).dropna()
    rows: list[dict[str, object]] = []
    for left, right in zip(bins[:-1], bins[1:]):
        bucket = frame[(frame["confidence"] >= left) & (frame["confidence"] < right)]
        rows.append(
            {
                "confidence_bucket": f"{left:.2f}-{right:.2f}",
                "sample_count": int(len(bucket)),
                "mean_confidence": float(bucket["confidence"].mean()) if len(bucket) else 0.0,
                "realized_success_rate": float(bucket["success"].mean()) if len(bucket) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def fit_confidence_calibrator(
    train_signals: pd.DataFrame,
    train_success: pd.Series,
    bins: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.01),
    dataset_role: str = "train",
) -> ConfidenceCalibrationModel:
    """Fit only on chronological train or validation data, never test data."""

    if dataset_role not in {"train", "validation"}:
        raise ValueError("Calibration may only be fit on train or validation data")
    table = confidence_calibration_table(train_signals, train_success, bins)
    rates = tuple(float(value) for value in table["realized_success_rate"])
    if train_signals.empty:
        raise ValueError("Calibration data cannot be empty")
    ordered_index = pd.DatetimeIndex(pd.to_datetime(train_signals.index, utc=True))
    return ConfidenceCalibrationModel(
        bins=bins,
        success_rates=rates,
        fitted_start=ordered_index.min(),
        fitted_end=ordered_index.max(),
    )
