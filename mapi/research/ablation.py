from __future__ import annotations

import pandas as pd

from mapi.config import MapiConfig
from mapi.research.backtest import run_backtest
from mapi.scoring import calculate_mapi


def run_ablation(
    symbol: str,
    price_frame: pd.DataFrame,
    sector_frame: pd.DataFrame | None,
    benchmark_frame: pd.DataFrame | None,
    config: MapiConfig,
    horizon_name: str = "short_term",
    backtest_horizon_bars: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    full = calculate_mapi(symbol, price_frame, sector_frame, benchmark_frame, config)[
        horizon_name
    ]
    full_metrics = run_backtest(
        full,
        price_frame,
        horizon_bars=backtest_horizon_bars,
        name="full_mapi",
        transaction_cost_bps=config.transaction_cost_bps,
        spread_bps=config.spread_bps,
        slippage_bps=config.slippage_bps,
    )
    rows.append({"variant": "full_mapi", "removed_component": None, **full_metrics.to_dict()})
    for component_name in config.enabled_components:
        disabled = config.with_disabled_component(component_name)
        ablated = calculate_mapi(
            symbol, price_frame, sector_frame, benchmark_frame, disabled
        )[horizon_name]
        metrics = run_backtest(
            ablated,
            price_frame,
            horizon_bars=backtest_horizon_bars,
            name=f"without_{component_name}",
            transaction_cost_bps=config.transaction_cost_bps,
            spread_bps=config.spread_bps,
            slippage_bps=config.slippage_bps,
        )
        row = {
            "variant": f"without_{component_name}",
            "removed_component": component_name,
            **metrics.to_dict(),
            "delta_mean_return_vs_full": metrics.mean_return - full_metrics.mean_return,
            "delta_sharpe_vs_full": metrics.sharpe_ratio - full_metrics.sharpe_ratio,
        }
        rows.append(row)
    return pd.DataFrame(rows)
