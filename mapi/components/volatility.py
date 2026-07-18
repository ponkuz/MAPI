from __future__ import annotations

import numpy as np
import pandas as pd

from mapi.components.base import ComponentContext, finalize_component_frame
from mapi.config import MapiConfig
from mapi.models import HorizonConfig
from mapi.normalization import (
    historical_zscore,
    event_novelty,
    rolling_percentile_rank,
    signed_unit_from_z,
    true_range,
)


class VolatilityAnomaly:
    name = "volatility_anomaly"
    family = "volatility"

    def calculate(
        self,
        price_frame: pd.DataFrame,
        context: ComponentContext,
        horizon: HorizonConfig,
        config: MapiConfig,
    ) -> pd.DataFrame:
        close = price_frame["close"]
        returns = close.pct_change()
        realized_vol = returns.rolling(
            horizon.rolling_window, min_periods=horizon.min_periods
        ).std(ddof=0)
        vol_rank = rolling_percentile_rank(
            realized_vol, horizon.rolling_window, horizon.min_periods
        )
        atr = true_range(price_frame).rolling(
            max(2, horizon.return_window), min_periods=max(2, min(horizon.return_window, 5))
        ).mean()
        atr_z = historical_zscore(atr / close, horizon.rolling_window, horizon.min_periods)
        volume_z = historical_zscore(
            np.log1p(price_frame["volume"].astype(float)),
            horizon.rolling_window,
            horizon.min_periods,
        )
        return_z = historical_zscore(
            close.pct_change(horizon.return_window),
            horizon.rolling_window,
            horizon.min_periods,
        )
        compression_with_volume = (
            (1.0 - vol_rank.fillna(0.5)).clip(0.0, 1.0)
            * (volume_z.clip(lower=0.0) / 3.0).clip(0.0, 1.0)
        )
        expansion_without_trend = (
            (atr_z.clip(lower=0.0) / 3.0).clip(0.0, 1.0)
            * (1.0 - (return_z.abs() / 1.5).clip(0.0, 1.0))
        )
        gap = (price_frame["open"] / close.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
        gap_z = historical_zscore(gap.abs(), horizon.rolling_window, horizon.min_periods)
        abnormal_gap = (gap_z.clip(lower=0.0) / 3.0).clip(0.0, 1.0)
        strength = pd.concat(
            [compression_with_volume, expansion_without_trend, abnormal_gap], axis=1
        ).max(axis=1)
        observed_pressure = signed_unit_from_z(return_z, scale=2.5)
        forecast_direction = pd.Series(0.0, index=price_frame.index)
        direction_semantics = np.select(
            [
                compression_with_volume > 0.35,
                expansion_without_trend > 0.35,
                abnormal_gap > 0.35,
            ],
            [
                "direction_neutral_volatility_compression",
                "direction_neutral_volatility_expansion_without_trend",
                "direction_neutral_abnormal_gap",
            ],
            default="direction_neutral_volatility_baseline",
        )
        confidence = (
            price_frame[["open", "high", "low", "close"]]
            .notna()
            .all(axis=1)
            .rolling(horizon.rolling_window, min_periods=1)
            .mean()
            * 0.9
        )
        novelty = event_novelty(
            strength, horizon.rolling_window, horizon.min_periods
        )
        labels = np.select(
            [
                compression_with_volume > 0.35,
                expansion_without_trend > 0.35,
                abnormal_gap > 0.35,
            ],
            [
                "Volatility compressed while volume increased",
                "Volatility expanded without a proportional directional price move",
                "The opening gap was unusually large versus recent history",
            ],
            default="Volatility behavior is close to its recent baseline",
        )
        metrics = [
            {
                "realized_volatility": float(realized_vol.iloc[i])
                if pd.notna(realized_vol.iloc[i])
                else None,
                "volatility_rank": float(vol_rank.iloc[i]) if pd.notna(vol_rank.iloc[i]) else None,
                "atr_z": float(atr_z.iloc[i]) if pd.notna(atr_z.iloc[i]) else None,
                "gap_z": float(gap_z.iloc[i]) if pd.notna(gap_z.iloc[i]) else None,
            }
            for i in range(len(price_frame))
        ]
        frame = pd.DataFrame(
            {
                "anomaly_strength": strength,
                "direction": forecast_direction,
                "forecast_direction": forecast_direction,
                "observed_pressure": observed_pressure,
                "directional_evidence_strength": 0.0,
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
        return finalize_component_frame(frame, price_frame.index, "Volatility anomaly")
