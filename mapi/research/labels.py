from __future__ import annotations

import numpy as np
import pandas as pd

from mapi.data.validation import normalize_ohlcv


def forward_return(
    price_frame: pd.DataFrame,
    horizon_bars: int,
    signal_delay_bars: int = 1,
) -> pd.Series:
    prices = normalize_ohlcv(price_frame) if "timestamp" in price_frame.columns else price_frame
    close = prices["close"]
    entry = close.shift(-signal_delay_bars)
    exit_ = close.shift(-(signal_delay_bars + horizon_bars))
    return (exit_ / entry - 1.0).replace([np.inf, -np.inf], np.nan)


def forward_path_metrics(
    price_frame: pd.DataFrame,
    horizon_bars: int,
    signal_delay_bars: int = 1,
    movement_threshold: float = 0.02,
) -> pd.DataFrame:
    """Build research-only forward labels after the signal timestamp."""

    prices = normalize_ohlcv(price_frame) if "timestamp" in price_frame.columns else price_frame
    close = prices["close"].reset_index(drop=True)
    intrabar_mfe: list[float] = []
    intrabar_mae: list[float] = []
    realized: list[float] = []
    absolute_return: list[float] = []
    future_volatility: list[float] = []
    breakout: list[float] = []
    reversal: list[float] = []
    time_to_move: list[float] = []
    for i in range(len(close)):
        entry_idx = i + signal_delay_bars
        exit_idx = entry_idx + horizon_bars
        if exit_idx >= len(close) or entry_idx >= len(close):
            intrabar_mfe.append(np.nan)
            intrabar_mae.append(np.nan)
            realized.append(np.nan)
            absolute_return.append(np.nan)
            future_volatility.append(np.nan)
            breakout.append(np.nan)
            reversal.append(np.nan)
            time_to_move.append(np.nan)
            continue
        entry = close.iloc[entry_idx]
        path = close.iloc[entry_idx : exit_idx + 1]
        if entry == 0 or pd.isna(entry):
            intrabar_mfe.append(np.nan)
            intrabar_mae.append(np.nan)
            realized.append(np.nan)
            absolute_return.append(np.nan)
            future_volatility.append(np.nan)
            breakout.append(np.nan)
            reversal.append(np.nan)
            time_to_move.append(np.nan)
            continue
        path_return = path / entry - 1.0
        realized_value = float(path_return.iloc[-1])
        future_high = prices["high"].iloc[entry_idx + 1 : exit_idx + 1]
        future_low = prices["low"].iloc[entry_idx + 1 : exit_idx + 1]
        intrabar_mfe.append(float((future_high / entry - 1.0).max()))
        intrabar_mae.append(float((future_low / entry - 1.0).min()))
        realized.append(realized_value)
        absolute_return.append(abs(realized_value))
        future_volatility.append(float(path.pct_change().dropna().std(ddof=0)))

        history_high = prices["high"].iloc[max(0, i - 19) : i + 1]
        history_low = prices["low"].iloc[max(0, i - 19) : i + 1]
        if len(history_high) >= 5:
            broke_range = (
                future_high.max() > history_high.max()
                or future_low.min() < history_low.min()
            )
            breakout.append(float(broke_range))
        else:
            breakout.append(np.nan)

        if i >= horizon_bars and close.iloc[i - horizon_bars] != 0:
            trailing_return = close.iloc[i] / close.iloc[i - horizon_bars] - 1.0
            reversed_trend = (
                abs(trailing_return) >= movement_threshold / 2.0
                and abs(realized_value) >= movement_threshold / 2.0
                and np.sign(trailing_return) != np.sign(realized_value)
            )
            reversal.append(float(reversed_trend))
        else:
            reversal.append(np.nan)

        reached = np.flatnonzero(
            (future_high / entry - 1.0).to_numpy() >= movement_threshold
        )
        reached_down = np.flatnonzero(
            (future_low / entry - 1.0).to_numpy() <= -movement_threshold
        )
        candidates = [
            int(values[0] + 1) for values in (reached, reached_down) if len(values) > 0
        ]
        time_to_move.append(float(min(candidates)) if candidates else np.nan)
    return pd.DataFrame(
        {
            "forward_return": realized,
            "absolute_return": absolute_return,
            "future_volatility": future_volatility,
            "intrabar_breakout": breakout,
            "reversal": reversal,
            "time_to_move": time_to_move,
            "intrabar_mfe": intrabar_mfe,
            "intrabar_mae": intrabar_mae,
            # Compatibility aliases retain values, not the former close-only semantics.
            "breakout": breakout,
            "mfe": intrabar_mfe,
            "mae": intrabar_mae,
        },
        index=prices.index,
    )
