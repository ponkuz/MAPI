from __future__ import annotations

import numpy as np
import pandas as pd

from mapi.components.base import ComponentContext, finalize_component_frame
from mapi.config import MapiConfig
from mapi.models import HorizonConfig
from mapi.normalization import event_novelty, historical_zscore, signed_unit_from_z


class MomentumDisagreement:
    name = "momentum_disagreement"
    family = "momentum"

    def calculate(
        self,
        price_frame: pd.DataFrame,
        context: ComponentContext,
        horizon: HorizonConfig,
        config: MapiConfig,
    ) -> pd.DataFrame:
        close = price_frame["close"]
        short_window = max(1, horizon.return_window)
        medium_window = max(short_window + 1, horizon.return_window * 3)
        short_return = close.pct_change(short_window)
        medium_return = close.pct_change(medium_window)
        short_z = historical_zscore(short_return, horizon.rolling_window, horizon.min_periods)
        medium_z = historical_zscore(medium_return, horizon.rolling_window, horizon.min_periods)
        short_signal = signed_unit_from_z(short_z, scale=2.0)
        medium_signal = signed_unit_from_z(medium_z, scale=2.0)
        timescale_disagreement = ((short_signal - medium_signal).abs() / 2.0).clip(0.0, 1.0)

        momentum = close.diff(short_window)
        previous_high, momentum_at_previous_high = _prior_extreme_momentum(
            close,
            momentum,
            horizon.rolling_window,
            horizon.min_periods,
            "high",
        )
        previous_low, momentum_at_previous_low = _prior_extreme_momentum(
            close,
            momentum,
            horizon.rolling_window,
            horizon.min_periods,
            "low",
        )
        new_high_nonconfirm = (
            (close > previous_high) & (momentum < momentum_at_previous_high)
        ).astype(float)
        new_low_nonconfirm = (
            (close < previous_low) & (momentum > momentum_at_previous_low)
        ).astype(float)
        nonconfirmation = pd.concat(
            [new_high_nonconfirm * 0.75, new_low_nonconfirm * 0.75], axis=1
        ).max(axis=1)
        strength = pd.concat([timescale_disagreement, nonconfirmation], axis=1).max(axis=1)
        observed_pressure = (
            (short_signal * 0.65) + (medium_signal * 0.35)
        ).clip(-1.0, 1.0)
        timescale_evidence = timescale_disagreement > 0.0
        timescale_forecast = (
            short_signal * timescale_disagreement
        ).where(timescale_evidence, 0.0).clip(-1.0, 1.0)
        forecast_direction = pd.Series(
            np.select(
                [new_high_nonconfirm > 0.0, new_low_nonconfirm > 0.0],
                [-nonconfirmation, nonconfirmation],
                default=timescale_forecast,
            ),
            index=price_frame.index,
            dtype=float,
        ).clip(-1.0, 1.0)
        directional_evidence_strength = pd.Series(
            np.select(
                [
                    new_high_nonconfirm > 0.0,
                    new_low_nonconfirm > 0.0,
                    timescale_evidence,
                ],
                [1.0, 1.0, 1.0],
                default=0.0,
            ),
            index=price_frame.index,
            dtype=float,
        )
        direction_semantics = np.select(
            [
                new_high_nonconfirm > 0.0,
                new_low_nonconfirm > 0.0,
                timescale_evidence,
            ],
            [
                "reversal_hypothesis_bearish_new_high_nonconfirmation",
                "reversal_hypothesis_bullish_new_low_nonconfirmation",
                "continuation_hypothesis_short_term_momentum_dominance",
            ],
            default="direction_neutral_momentum_alignment",
        )
        coverage = close.rolling(horizon.rolling_window, min_periods=1).count() / float(
            horizon.rolling_window
        )
        confidence = coverage.clip(0.0, 1.0) * 0.85
        novelty = event_novelty(
            strength, horizon.rolling_window, horizon.min_periods
        )
        labels = np.select(
            [
                new_high_nonconfirm > 0.0,
                new_low_nonconfirm > 0.0,
                (short_signal > 0.2) & (medium_signal < -0.2),
                (short_signal < -0.2) & (medium_signal > 0.2),
            ],
            [
                "Price made a new high while momentum failed to confirm",
                "Price made a new low while selling momentum weakened",
                "Short-term momentum improved while medium-term momentum deteriorated",
                "Short-term momentum deteriorated while medium-term momentum improved",
            ],
            default="Momentum timescales are broadly aligned",
        )
        metrics = [
            {
                "short_return": float(short_return.iloc[i])
                if pd.notna(short_return.iloc[i])
                else None,
                "medium_return": float(medium_return.iloc[i])
                if pd.notna(medium_return.iloc[i])
                else None,
                "short_z": float(short_z.iloc[i]) if pd.notna(short_z.iloc[i]) else None,
                "medium_z": float(medium_z.iloc[i]) if pd.notna(medium_z.iloc[i]) else None,
                "previous_extreme_price": float(previous_high.iloc[i])
                if new_high_nonconfirm.iloc[i] > 0.0
                else float(previous_low.iloc[i])
                if new_low_nonconfirm.iloc[i] > 0.0
                else None,
                "momentum_at_previous_extreme": float(momentum_at_previous_high.iloc[i])
                if new_high_nonconfirm.iloc[i] > 0.0
                else float(momentum_at_previous_low.iloc[i])
                if new_low_nonconfirm.iloc[i] > 0.0
                else None,
            }
            for i in range(len(price_frame))
        ]
        frame = pd.DataFrame(
            {
                "anomaly_strength": strength,
                "direction": forecast_direction,
                "forecast_direction": forecast_direction,
                "observed_pressure": observed_pressure,
                "directional_evidence_strength": directional_evidence_strength,
                "direction_semantics": direction_semantics,
                "direction_contract_warning": None,
                "confidence": confidence,
                "novelty": novelty["novelty"].fillna(0.0),
                "historical_extremeness": novelty["historical_extremeness"].fillna(0.0),
                "recurrence_rate": novelty["recurrence_rate"].fillna(0.0),
                "reason": labels,
                "metrics": metrics,
            },
            index=price_frame.index,
        )
        return finalize_component_frame(frame, price_frame.index, "Momentum disagreement")


def _prior_extreme_momentum(
    close: pd.Series,
    momentum: pd.Series,
    window: int,
    min_periods: int,
    extreme: str,
) -> tuple[pd.Series, pd.Series]:
    extreme_price = pd.Series(np.nan, index=close.index, dtype=float)
    extreme_momentum = pd.Series(np.nan, index=close.index, dtype=float)
    for position in range(len(close)):
        history = close.iloc[max(0, position - window) : position].dropna()
        if len(history) < min_periods:
            continue
        timestamp = history.idxmax() if extreme == "high" else history.idxmin()
        extreme_price.iloc[position] = float(close.loc[timestamp])
        value = momentum.loc[timestamp]
        if pd.notna(value):
            extreme_momentum.iloc[position] = float(value)
    return extreme_price, extreme_momentum
