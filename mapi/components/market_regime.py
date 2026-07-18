from __future__ import annotations

import numpy as np
import pandas as pd

from mapi.components.base import (
    ComponentContext,
    empty_component_frame,
    finalize_component_frame,
)
from mapi.config import MapiConfig
from mapi.data.alignment import align_point_in_time
from mapi.models import HorizonConfig
from mapi.normalization import event_novelty, historical_zscore, signed_unit_from_z


class MarketRegimeDivergence:
    name = "market_regime_divergence"
    family = "market_context"

    def calculate(
        self,
        price_frame: pd.DataFrame,
        context: ComponentContext,
        horizon: HorizonConfig,
        config: MapiConfig,
    ) -> pd.DataFrame:
        benchmark = context.benchmark_frame
        if benchmark is None:
            return empty_component_frame(
                price_frame.index,
                "Benchmark data is unavailable; market-regime divergence is inactive",
            )
        alignment = align_point_in_time(
            price_frame,
            benchmark,
            max_staleness_bars=config.cross_asset_max_staleness_bars,
            frequency_tolerance=config.cross_asset_frequency_tolerance,
        )
        benchmark_aligned = alignment.frame
        stock_return = price_frame["close"].pct_change(horizon.return_window)
        benchmark_return = benchmark_aligned["close"].pct_change(
            horizon.return_window, fill_method=None
        )
        spread = stock_return - benchmark_return
        spread_z = historical_zscore(spread, horizon.rolling_window, horizon.min_periods)
        benchmark_z = historical_zscore(
            benchmark_return, horizon.rolling_window, horizon.min_periods
        )
        regime_force = (benchmark_z.abs() / 2.5).clip(0.0, 1.0)
        opposite_direction = (
            np.sign(stock_return.fillna(0.0)) != np.sign(benchmark_return.fillna(0.0))
        ).astype(float)
        strength = pd.concat(
            [(spread_z.abs() / 3.0).clip(0.0, 1.0), opposite_direction * regime_force],
            axis=1,
        ).max(axis=1)
        observed_pressure = signed_unit_from_z(spread_z, scale=2.0)
        directional_evidence = spread_z.abs() > 1.2
        forecast_direction = observed_pressure.where(directional_evidence, 0.0)
        direction_semantics = np.where(
            directional_evidence,
            "continuation_hypothesis_broad_market_relative_strength",
            "direction_neutral_broad_market_divergence",
        )
        confidence = (
            alignment.valid.rolling(horizon.rolling_window, min_periods=1).mean()
            * alignment.valid.astype(float)
            * (spread_z.notna() | benchmark_z.notna()).astype(float)
            * 0.85
        )
        novelty = event_novelty(
            strength, horizon.rolling_window, horizon.min_periods
        )
        labels = pd.Series(
            np.select(
                [
                    (spread_z > 1.2) & (benchmark_return < 0),
                    (spread_z < -1.2) & (benchmark_return > 0),
                    spread_z > 1.2,
                    spread_z < -1.2,
                ],
                [
                    "Stock strength appeared while the broad market weakened",
                    "Stock weakness appeared while the broad market strengthened",
                    "Stock returns outpaced the broad market by an abnormal margin",
                    "Stock returns lagged the broad market by an abnormal margin",
                ],
                default="Stock behavior is close to the broad-market regime",
            ),
            index=price_frame.index,
            dtype=object,
        )
        labels.loc[~alignment.valid] = (
            "Benchmark data is stale, from another session, or frequency-incompatible"
        )
        metrics = [
            {
                "stock_return": float(stock_return.iloc[i])
                if pd.notna(stock_return.iloc[i])
                else None,
                "benchmark_return": float(benchmark_return.iloc[i])
                if pd.notna(benchmark_return.iloc[i])
                else None,
                "spread_z": float(spread_z.iloc[i]) if pd.notna(spread_z.iloc[i]) else None,
                "matched_timestamp": benchmark_aligned["matched_timestamp"].iloc[i],
                "staleness_seconds": float(alignment.staleness_seconds.iloc[i])
                if pd.notna(alignment.staleness_seconds.iloc[i])
                else None,
                "frequency_compatible": alignment.frequency_compatible,
            }
            for i in range(len(price_frame))
        ]
        frame = pd.DataFrame(
            {
                "anomaly_strength": strength,
                "direction": forecast_direction,
                "forecast_direction": forecast_direction,
                "observed_pressure": observed_pressure,
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
        return finalize_component_frame(frame, price_frame.index, "Market-regime divergence")
