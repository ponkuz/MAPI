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


class StockSectorDivergence:
    name = "stock_sector_divergence"
    family = "cross_asset"

    def calculate(
        self,
        price_frame: pd.DataFrame,
        context: ComponentContext,
        horizon: HorizonConfig,
        config: MapiConfig,
    ) -> pd.DataFrame:
        if context.sector_frame is None:
            return empty_component_frame(
                price_frame.index,
                "Sector benchmark data is unavailable; stock-sector divergence is inactive",
            )
        alignment = align_point_in_time(
            price_frame,
            context.sector_frame,
            max_staleness_bars=config.cross_asset_max_staleness_bars,
            frequency_tolerance=config.cross_asset_frequency_tolerance,
        )
        sector = alignment.frame
        stock_return = price_frame["close"].pct_change(horizon.return_window)
        sector_return = sector["close"].pct_change(
            horizon.return_window, fill_method=None
        )

        rolling_cov = stock_return.rolling(
            horizon.rolling_window, min_periods=horizon.min_periods
        ).cov(sector_return, ddof=0)
        rolling_var = sector_return.rolling(
            horizon.rolling_window, min_periods=horizon.min_periods
        ).var(ddof=0)
        beta = (rolling_cov / rolling_var.replace(0.0, np.nan)).shift(1)
        stock_mean = stock_return.rolling(
            horizon.rolling_window, min_periods=horizon.min_periods
        ).mean().shift(1)
        sector_mean = sector_return.rolling(
            horizon.rolling_window, min_periods=horizon.min_periods
        ).mean().shift(1)
        alpha = stock_mean - beta * sector_mean
        expected_return = alpha + beta * sector_return
        residual = stock_return - expected_return
        residual_z = historical_zscore(residual, horizon.rolling_window, horizon.min_periods)
        residual_strength = (residual_z.abs() / 3.0).clip(0.0, 1.0)
        observed_pressure = signed_unit_from_z(residual_z, scale=2.0)
        corr = stock_return.rolling(
            horizon.rolling_window, min_periods=horizon.min_periods
        ).corr(sector_return)
        historical_corr_baseline = corr.abs().rolling(
            horizon.rolling_window, min_periods=horizon.min_periods
        ).median().shift(1)
        corr_breakdown = (historical_corr_baseline - corr.abs()).clip(0.0, 1.0)
        correlation_strength = corr_breakdown * 0.65
        strength = pd.concat([residual_strength, correlation_strength], axis=1).max(axis=1)
        residual_forecast = (
            (residual_z.abs() > 1.2) & (residual_strength >= correlation_strength)
        )
        forecast_direction = observed_pressure.where(residual_forecast, 0.0)
        direction_semantics = np.select(
            [
                (correlation_strength > residual_strength) & (corr_breakdown > 0.0),
                residual_forecast,
            ],
            [
                "direction_neutral_correlation_breakdown",
                "continuation_hypothesis_beta_adjusted_residual",
            ],
            default="direction_neutral_insufficient_residual_evidence",
        )
        confidence = (
            alignment.valid.rolling(horizon.rolling_window, min_periods=1).mean()
            * beta.notna().astype(float).replace(0.0, 0.25)
            * alignment.valid.astype(float)
            * (residual_z.notna() | corr.notna()).astype(float)
        ).clip(0.0, 0.9)
        novelty = event_novelty(
            strength, horizon.rolling_window, horizon.min_periods
        )
        labels = np.select(
            [
                residual_z > 1.2,
                residual_z < -1.2,
                corr_breakdown > 0.6,
            ],
            [
                "Stock returns are outperforming the sector after beta adjustment",
                "Stock returns are underperforming the sector after beta adjustment",
                "The stock-sector correlation has weakened versus its recent history",
            ],
            default="Stock-sector behavior is close to its beta-adjusted expectation",
        )
        labels = pd.Series(labels, index=price_frame.index, dtype=object)
        labels.loc[~alignment.valid] = (
            "Sector data is stale, from another session, or frequency-incompatible"
        )
        metrics = [
            {
                "stock_return": float(stock_return.iloc[i])
                if pd.notna(stock_return.iloc[i])
                else None,
                "sector_return": float(sector_return.iloc[i])
                if pd.notna(sector_return.iloc[i])
                else None,
                "expected_return": float(expected_return.iloc[i])
                if pd.notna(expected_return.iloc[i])
                else None,
                "residual_z": float(residual_z.iloc[i])
                if pd.notna(residual_z.iloc[i])
                else None,
                "rolling_correlation": float(corr.iloc[i]) if pd.notna(corr.iloc[i]) else None,
                "historical_correlation_baseline": float(historical_corr_baseline.iloc[i])
                if pd.notna(historical_corr_baseline.iloc[i])
                else None,
                "correlation_decline": float(corr_breakdown.iloc[i])
                if pd.notna(corr_breakdown.iloc[i])
                else None,
                "matched_timestamp": alignment.frame["matched_timestamp"].iloc[i],
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
                "directional_evidence_strength": residual_forecast.astype(float),
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
        return finalize_component_frame(frame, price_frame.index, "Stock-sector divergence")
