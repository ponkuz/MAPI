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

        previous_high = close.shift(1).rolling(
            horizon.rolling_window, min_periods=horizon.min_periods
        ).max()
        previous_low = close.shift(1).rolling(
            horizon.rolling_window, min_periods=horizon.min_periods
        ).min()
        momentum = close.diff(short_window)
        prior_momentum_high = momentum.shift(1).rolling(
            horizon.rolling_window, min_periods=horizon.min_periods
        ).max()
        prior_momentum_low = momentum.shift(1).rolling(
            horizon.rolling_window, min_periods=horizon.min_periods
        ).min()
        new_high_nonconfirm = ((close > previous_high) & (momentum < prior_momentum_high)).astype(float)
        new_low_nonconfirm = ((close < previous_low) & (momentum > prior_momentum_low)).astype(float)
        nonconfirmation = pd.concat(
            [new_high_nonconfirm * 0.75, new_low_nonconfirm * 0.75], axis=1
        ).max(axis=1)
        strength = pd.concat([timescale_disagreement, nonconfirmation], axis=1).max(axis=1)
        direction = ((short_signal * 0.65) + (medium_signal * 0.35)).clip(-1.0, 1.0)
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
            }
            for i in range(len(price_frame))
        ]
        frame = pd.DataFrame(
            {
                "anomaly_strength": strength,
                "direction": direction,
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
