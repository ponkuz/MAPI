from __future__ import annotations

import numpy as np
import pandas as pd

from mapi.components.base import ComponentContext, finalize_component_frame
from mapi.config import MapiConfig
from mapi.models import HorizonConfig
from mapi.normalization import (
    close_location_value,
    historical_zscore,
    event_novelty,
    robust_unit_score_from_z,
    signed_unit_from_z,
)


class PriceVolumeDivergence:
    name = "price_volume_divergence"
    family = "price_volume"

    def calculate(
        self,
        price_frame: pd.DataFrame,
        context: ComponentContext,
        horizon: HorizonConfig,
        config: MapiConfig,
    ) -> pd.DataFrame:
        close = price_frame["close"]
        volume = price_frame["volume"].astype(float)
        returns = close.pct_change(horizon.return_window)
        price_z = historical_zscore(returns, horizon.rolling_window, horizon.min_periods)
        volume_z = historical_zscore(
            np.log1p(volume), horizon.rolling_window, horizon.min_periods
        )
        relative_volume = volume / (
            volume.rolling(horizon.rolling_window, min_periods=horizon.min_periods)
            .median()
            .shift(1)
            .replace(0.0, np.nan)
        )
        previous_high = close.shift(1).rolling(
            horizon.rolling_window, min_periods=horizon.min_periods
        ).max()
        breakout_on_weak_volume = ((close > previous_high) & (volume_z < -0.35)).astype(float)
        high_volume_flat_price = (
            (volume_z.clip(lower=0.0) / 3.0).clip(0.0, 1.0)
            * (1.0 - (price_z.abs() / 1.2).clip(0.0, 1.0))
        )
        large_move_low_volume = (
            (price_z.abs() / 3.0).clip(0.0, 1.0)
            * ((-volume_z).clip(lower=0.0) / 2.0).clip(0.0, 1.0)
        )
        sign_mismatch = (
            (np.sign(price_z.fillna(0.0)) != np.sign(volume_z.fillna(0.0)))
            & (price_z.abs() > 0.8)
            & (volume_z.abs() > 0.8)
        ).astype(float) * ((price_z.abs() + volume_z.abs()) / 6.0).clip(0.0, 1.0)
        strength = pd.concat(
            [
                breakout_on_weak_volume,
                high_volume_flat_price,
                large_move_low_volume,
                sign_mismatch,
            ],
            axis=1,
        ).max(axis=1)
        close_location = close_location_value(price_frame)
        direction = (
            signed_unit_from_z(price_z, scale=2.0) * 0.75 + close_location * 0.5
        ).clip(-1.0, 1.0)
        coverage = volume.where(volume > 0.0).rolling(
            horizon.rolling_window, min_periods=1
        ).count() / float(horizon.rolling_window)
        confidence = coverage.clip(0.0, 1.0) * 0.95
        novelty = event_novelty(
            strength, horizon.rolling_window, horizon.min_periods
        )

        labels = np.select(
            [
                breakout_on_weak_volume > 0.0,
                high_volume_flat_price > 0.45,
                large_move_low_volume > 0.35,
                sign_mismatch > 0.0,
            ],
            [
                "Price broke a prior high while volume was weaker than its recent baseline",
                "Volume expanded unusually while price movement remained muted",
                "A large price move appeared on unusually light volume",
                "Price and volume moved in statistically unusual opposite directions",
            ],
            default="Price and volume behavior is close to its recent baseline",
        )
        metrics = [
            {
                "return": float(returns.iloc[i]) if pd.notna(returns.iloc[i]) else None,
                "price_z": float(price_z.iloc[i]) if pd.notna(price_z.iloc[i]) else None,
                "volume_z": float(volume_z.iloc[i]) if pd.notna(volume_z.iloc[i]) else None,
                "relative_volume": float(relative_volume.iloc[i])
                if pd.notna(relative_volume.iloc[i])
                else None,
            }
            for i in range(len(price_frame))
        ]
        frame = pd.DataFrame(
            {
                "anomaly_strength": strength.clip(0.0, 1.0),
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
        return finalize_component_frame(frame, price_frame.index, labels[0] if len(labels) else "")
