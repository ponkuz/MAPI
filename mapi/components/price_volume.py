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
        close_location = close_location_value(price_frame)
        directional_flow = close_location * np.log1p(relative_volume.clip(lower=0.0))
        flow_z = historical_zscore(
            directional_flow, horizon.rolling_window, horizon.min_periods
        )
        directional_flow_mismatch = (
            (np.sign(price_z.fillna(0.0)) != np.sign(flow_z.fillna(0.0)))
            & (price_z.abs() > 0.8)
            & (flow_z.abs() > 0.8)
        ).astype(float) * ((price_z.abs() + flow_z.abs()) / 6.0).clip(0.0, 1.0)
        strength = pd.concat(
            [
                breakout_on_weak_volume,
                high_volume_flat_price,
                large_move_low_volume,
                directional_flow_mismatch,
            ],
            axis=1,
        ).max(axis=1)
        observed_pressure = (
            signed_unit_from_z(price_z, scale=2.0) * 0.6
            + signed_unit_from_z(flow_z, scale=2.0) * 0.4
        ).clip(-1.0, 1.0)
        (
            forecast_direction,
            directional_evidence_strength,
            direction_semantics,
            anomaly_subtype,
        ) = _price_volume_direction_contract(
            breakout_on_weak_volume=breakout_on_weak_volume,
            high_volume_flat_price=high_volume_flat_price,
            large_move_low_volume=large_move_low_volume,
            directional_flow_mismatch=directional_flow_mismatch,
            price_z=price_z,
            flow_z=flow_z,
        )
        coverage = volume.where(volume > 0.0).rolling(
            horizon.rolling_window, min_periods=1
        ).count() / float(horizon.rolling_window)
        confidence = coverage.clip(0.0, 1.0) * 0.95
        novelty = event_novelty(
            strength, horizon.rolling_window, horizon.min_periods
        )

        labels = np.select(
            [
                anomaly_subtype == "breakout_on_weak_volume",
                anomaly_subtype == "high_volume_flat_price",
                anomaly_subtype == "large_move_low_volume",
                anomaly_subtype == "directional_flow_mismatch",
            ],
            [
                "Price broke a prior high while volume was weaker than its recent baseline",
                "Volume expanded unusually while price movement remained muted",
                "A large price move appeared on unusually light volume",
                "Price movement and directional volume flow disagreed",
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
                "directional_flow_z": float(flow_z.iloc[i])
                if pd.notna(flow_z.iloc[i])
                else None,
                "directional_flow_mismatch": float(
                    directional_flow_mismatch.iloc[i]
                ),
                "anomaly_subtype": str(anomaly_subtype.iloc[i]),
            }
            for i in range(len(price_frame))
        ]
        frame = pd.DataFrame(
            {
                "anomaly_strength": strength.clip(0.0, 1.0),
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
        return finalize_component_frame(frame, price_frame.index, labels[0] if len(labels) else "")


def _price_volume_direction_contract(
    breakout_on_weak_volume: pd.Series,
    high_volume_flat_price: pd.Series,
    large_move_low_volume: pd.Series,
    directional_flow_mismatch: pd.Series,
    price_z: pd.Series,
    flow_z: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    subtype_scores = pd.DataFrame(
        {
            "breakout_on_weak_volume": breakout_on_weak_volume,
            "high_volume_flat_price": high_volume_flat_price,
            "large_move_low_volume": large_move_low_volume,
            "directional_flow_mismatch": directional_flow_mismatch,
        }
    ).fillna(0.0)
    dominant_strength = subtype_scores.max(axis=1)
    anomaly_subtype = subtype_scores.idxmax(axis=1).where(
        dominant_strength > 0.0, "baseline"
    )
    breakout = anomaly_subtype == "breakout_on_weak_volume"
    flat_price = anomaly_subtype == "high_volume_flat_price"
    low_volume_move = anomaly_subtype == "large_move_low_volume"
    flow_mismatch = anomaly_subtype == "directional_flow_mismatch"

    price_pressure = signed_unit_from_z(price_z, scale=2.0)
    flow_pressure = signed_unit_from_z(flow_z, scale=2.0)
    forecast_direction = pd.Series(0.0, index=subtype_scores.index, dtype=float)
    forecast_direction.loc[breakout] = -dominant_strength.loc[breakout]
    forecast_direction.loc[low_volume_move] = (
        -price_pressure.loc[low_volume_move] * dominant_strength.loc[low_volume_move]
    )
    forecast_direction.loc[flow_mismatch] = (
        flow_pressure.loc[flow_mismatch] * dominant_strength.loc[flow_mismatch]
    )
    forecast_direction = forecast_direction.clip(-1.0, 1.0)

    directional_evidence_strength = (
        breakout | low_volume_move | flow_mismatch
    ).astype(float)
    direction_semantics = pd.Series(
        np.select(
            [
                breakout,
                flat_price,
                low_volume_move & (price_z < 0.0),
                low_volume_move & (price_z >= 0.0),
                flow_mismatch & (flow_z > 0.0),
                flow_mismatch & (flow_z <= 0.0),
            ],
            [
                "reversal_hypothesis_bearish_weak_volume_breakout",
                "direction_neutral_high_volume_flat_price",
                "reversal_hypothesis_bullish_large_down_move_low_volume",
                "reversal_hypothesis_bearish_large_up_move_low_volume",
                "reversal_hypothesis_bullish_directional_flow_against_price",
                "reversal_hypothesis_bearish_directional_flow_against_price",
            ],
            default="direction_neutral_price_volume_baseline",
        ),
        index=subtype_scores.index,
        dtype=object,
    )
    return (
        forecast_direction,
        directional_evidence_strength,
        direction_semantics,
        anomaly_subtype.astype(str),
    )
