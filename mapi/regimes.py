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
) -> pd.Series:
    config = config or MapiConfig()
    close = price_frame["close"]
    if benchmark_frame is not None:
        alignment = align_point_in_time(
            price_frame,
            benchmark_frame,
            max_staleness_bars=config.cross_asset_max_staleness_bars,
            frequency_tolerance=config.cross_asset_frequency_tolerance,
        )
        close = alignment.frame["close"].where(alignment.valid, close)
    trend_window = max(3, horizon.return_window * 2)
    trend_return = close.pct_change(trend_window)
    one_bar_return = close.pct_change()
    realized_vol = one_bar_return.rolling(
        horizon.rolling_window, min_periods=horizon.min_periods
    ).std(ddof=0)
    vol_rank = rolling_percentile_rank(
        realized_vol, horizon.rolling_window, horizon.min_periods
    ).fillna(0.5)

    values: list[str] = []
    for trend, vol in zip(trend_return.fillna(0.0), vol_rank):
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
    return pd.Series(values, index=price_frame.index, name="mapi_regime")
