from __future__ import annotations

import numpy as np
import pandas as pd

from mapi.config import MapiConfig
from mapi.data.alignment import align_point_in_time
from mapi.models import HorizonConfig
from mapi.normalization import rolling_percentile_rank


def detect_market_regime(
    price_frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None,
    horizon: HorizonConfig,
    config: MapiConfig | None = None,
) -> pd.DataFrame:
    config = config or MapiConfig()
    close = price_frame["close"].astype(float)
    source = pd.Series("stock_fallback", index=price_frame.index, dtype=object)
    source_confidence = pd.Series(0.6, index=price_frame.index, dtype=float)
    current_valid = close.notna()
    if benchmark_frame is not None:
        alignment = align_point_in_time(
            price_frame,
            benchmark_frame,
            max_staleness_bars=config.cross_asset_max_staleness_bars,
            frequency_tolerance=config.cross_asset_frequency_tolerance,
        )
        close = alignment.frame["close"].where(alignment.valid)
        current_valid = alignment.valid & close.notna()
        source[:] = "benchmark"
        source.loc[~current_valid] = "benchmark_stale"
        source_confidence[:] = 1.0
        source_confidence.loc[~current_valid] = 0.0
    trend_window = max(3, horizon.return_window * 2)
    trend_return = close.pct_change(trend_window, fill_method=None)
    one_bar_return = close.pct_change(fill_method=None)
    realized_vol = one_bar_return.rolling(
        horizon.rolling_window, min_periods=horizon.min_periods
    ).std(ddof=0)
    vol_rank = rolling_percentile_rank(
        realized_vol, horizon.rolling_window, horizon.min_periods
    ).fillna(0.5)

    history_ready = current_valid & trend_return.notna() & realized_vol.notna()
    values: list[str] = []
    for ready, trend, vol in zip(history_ready, trend_return, vol_rank):
        if not ready:
            values.append("unknown")
            continue
        if trend < -0.08 and vol > 0.8:
            values.append("panic")
        elif trend < -0.02 and vol > 0.6:
            values.append("risk_off")
        elif trend > 0.02 and vol > 0.65:
            values.append("risk_on_high_volatility")
        elif trend > 0.02:
            values.append("risk_on_low_volatility")
        elif abs(trend) <= 0.01 and vol < 0.35:
            values.append("range_low_volatility")
        elif np.isfinite(vol) and vol > 0.75:
            values.append("high_volatility_range")
        else:
            values.append("trend_or_transition")
    confidence = source_confidence.where(history_ready, 0.0)
    return pd.DataFrame(
        {
            "regime": values,
            "regime_source": source,
            "regime_confidence": confidence,
        },
        index=price_frame.index,
    )
