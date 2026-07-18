from __future__ import annotations

import pandas as pd

from mapi.config import MapiConfig
from mapi.research.backtest import run_backtest
from mapi.research.matching import (
    candidate_frequency,
    chronological_masks,
    fit_frequency_matched_threshold,
    test_only_signals,
)
from mapi.scoring import calculate_mapi


def run_ablation(
    symbol: str,
    price_frame: pd.DataFrame,
    sector_frame: pd.DataFrame | None,
    benchmark_frame: pd.DataFrame | None,
    config: MapiConfig,
    horizon_name: str = "short_term",
    backtest_horizon_bars: int | None = None,
    score_column: str | None = None,
    score_threshold: float = 60.0,
    min_direction: float = 0.10,
    fit_fraction: float = 0.70,
) -> pd.DataFrame:
    config.validate()
    if horizon_name not in config.horizons:
        raise ValueError(f"Unknown horizon: {horizon_name}")
    holding_bars = (
        config.horizons[horizon_name].return_window
        if backtest_horizon_bars is None
        else backtest_horizon_bars
    )
    if holding_bars <= 0:
        raise ValueError("backtest_horizon_bars must be positive")
    score_column_used = score_column or config.research_score_column
    rows: list[dict[str, object]] = []
    full = calculate_mapi(symbol, price_frame, sector_frame, benchmark_frame, config)[
        horizon_name
    ]
    fit_mask, test_mask = chronological_masks(full.index, fit_fraction)
    target_frequency = candidate_frequency(
        full, score_column_used, score_threshold, min_direction, fit_mask
    )
    full_metrics = run_backtest(
        test_only_signals(full, test_mask),
        price_frame,
        horizon_bars=holding_bars,
        name="full_mapi",
        score_threshold=score_threshold,
        min_direction=min_direction,
        score_column=score_column_used,
        evaluation_mask=test_mask,
        transaction_cost_bps=config.transaction_cost_bps,
        spread_bps=config.spread_bps,
        slippage_bps=config.slippage_bps,
        large_move_threshold=config.large_move_threshold,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.deterministic_seed,
    )
    rows.append(
        {
            "variant": "full_mapi",
            "removed_component": None,
            **full_metrics.to_dict(),
            "fitted_threshold": score_threshold,
            "target_fit_frequency": target_frequency,
            "candidate_fit_frequency": target_frequency,
            "candidate_test_frequency": candidate_frequency(
                full, score_column_used, score_threshold, min_direction, test_mask
            ),
            "selected_event_count": full_metrics.sample_count,
            "sample_count_change_vs_full": 0,
        }
    )
    for component_name in config.enabled_components:
        disabled = config.with_disabled_component(component_name)
        ablated = calculate_mapi(
            symbol, price_frame, sector_frame, benchmark_frame, disabled
        )[horizon_name]
        match = fit_frequency_matched_threshold(
            ablated,
            score_column_used,
            target_frequency,
            min_direction,
            fit_mask,
        )
        metrics = run_backtest(
            test_only_signals(ablated, test_mask),
            price_frame,
            horizon_bars=holding_bars,
            name=f"without_{component_name}",
            score_threshold=match.threshold,
            min_direction=min_direction,
            score_column=score_column_used,
            evaluation_mask=test_mask,
            transaction_cost_bps=config.transaction_cost_bps,
            spread_bps=config.spread_bps,
            slippage_bps=config.slippage_bps,
            large_move_threshold=config.large_move_threshold,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.deterministic_seed,
        )
        row = {
            "variant": f"without_{component_name}",
            "removed_component": component_name,
            **metrics.to_dict(),
            "fitted_threshold": match.threshold,
            "target_fit_frequency": target_frequency,
            "candidate_fit_frequency": match.fitted_frequency,
            "candidate_test_frequency": candidate_frequency(
                ablated,
                score_column_used,
                match.threshold,
                min_direction,
                test_mask,
            ),
            "selected_event_count": metrics.sample_count,
            "sample_count_change_vs_full": metrics.sample_count - full_metrics.sample_count,
            "delta_mean_return_vs_full": metrics.mean_return - full_metrics.mean_return,
            "delta_event_return_mean_to_std_vs_full": (
                metrics.event_return_mean_to_std
                - full_metrics.event_return_mean_to_std
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)
