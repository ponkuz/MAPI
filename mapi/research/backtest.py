from __future__ import annotations

import numpy as np
import pandas as pd

from mapi.data.validation import normalize_ohlcv
from mapi.models import BacktestMetrics
from mapi.research.labels import forward_path_metrics


def run_event_study(
    signals: pd.DataFrame,
    price_frame: pd.DataFrame,
    horizon_bars: int,
    name: str = "mapi",
    score_threshold: float = 60.0,
    min_direction: float = 0.10,
    signal_delay_bars: int = 1,
    transaction_cost_bps: float = 10.0,
    spread_bps: float = 2.0,
    slippage_bps: float = 5.0,
    allow_overlapping: bool = False,
    score_column: str | None = None,
) -> BacktestMetrics:
    """Evaluate signal events; this is not a capital-aware portfolio simulation."""

    prices = normalize_ohlcv(price_frame) if "timestamp" in price_frame.columns else price_frame
    aligned = signals.reindex(prices.index)
    selected_score_column = score_column or (
        "mapi_actionability_score"
        if "mapi_actionability_score" in aligned.columns
        else "mapi_score"
    )
    score = aligned[selected_score_column].fillna(0.0)
    direction = aligned["mapi_direction"].fillna(0.0).clip(-1.0, 1.0)
    candidate_side = pd.Series(
        np.where(
        (score >= score_threshold) & (direction.abs() >= min_direction),
        np.sign(direction),
        0.0,
        ),
        index=prices.index,
        dtype=float,
    )
    labels = forward_path_metrics(prices, horizon_bars, signal_delay_bars)
    valid_candidate = (candidate_side != 0.0) & labels["forward_return"].notna()
    selected = valid_candidate.copy()
    excluded_overlap = 0
    if not allow_overlapping:
        selected[:] = False
        next_allowed_position = 0
        for position in np.flatnonzero(valid_candidate.to_numpy()):
            if position < next_allowed_position:
                excluded_overlap += 1
                continue
            selected.iloc[position] = True
            next_allowed_position = position + horizon_bars + 1
    side = candidate_side.where(selected, 0.0)
    gross_return = side * labels["forward_return"]
    round_trip_cost = (transaction_cost_bps + spread_bps + slippage_bps) / 10000.0
    net_return = gross_return - np.where(side != 0.0, round_trip_cost, 0.0)
    trades = net_return[(side != 0.0) & net_return.notna()]
    warnings = [
        "Event-study statistics are not realizable portfolio metrics; capital, sizing, borrow, and capacity are not modeled."
    ]
    if allow_overlapping and int(valid_candidate.sum()) > 1:
        warnings.append(
            "Forward-return observations may overlap; Sharpe, drawdown, and profit factor are descriptive event-sequence statistics."
        )
    if trades.empty:
        return BacktestMetrics(
            name=name,
            horizon_bars=horizon_bars,
            sample_count=0,
            mean_return=0.0,
            median_return=0.0,
            hit_rate=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            maximum_drawdown=0.0,
            profit_factor=0.0,
            precision=0.0,
            recall=0.0,
            information_coefficient=0.0,
            mean_mfe=0.0,
            mean_mae=0.0,
            mean_absolute_return=0.0,
            mean_future_volatility=0.0,
            breakout_rate=0.0,
            reversal_rate=0.0,
            mean_time_to_move=0.0,
            non_overlapping=not allow_overlapping,
            overlapping_candidates_excluded=excluded_overlap,
            warnings=warnings,
        )

    sample_count = int(trades.count())
    mean_return = float(trades.mean())
    median_return = float(trades.median())
    hit_rate = float((trades > 0).mean())
    std = float(trades.std(ddof=0))
    annualizer = np.sqrt(max(1.0, 252.0 / max(1, horizon_bars)))
    sharpe = float(mean_return / std * annualizer) if std > 0 else 0.0
    downside = trades[trades < 0]
    downside_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    sortino = float(mean_return / downside_std * annualizer) if downside_std > 0 else 0.0
    equity = (1.0 + trades.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    gains = float(trades[trades > 0].sum())
    losses = float(trades[trades < 0].sum())
    profit_factor = gains / abs(losses) if losses < 0 else (np.inf if gains > 0 else 0.0)
    realized = labels["forward_return"].reindex(trades.index)
    trade_side = side.reindex(trades.index)
    correct_direction = np.sign(realized) == np.sign(trade_side)
    precision = float(correct_direction.mean()) if len(correct_direction) else 0.0
    opportunity_cutoff = labels["forward_return"].abs().median()
    opportunities = labels["forward_return"].abs() >= opportunity_cutoff
    correct_opportunities = (
        opportunities
        & (side != 0.0)
        & (np.sign(labels["forward_return"]) == np.sign(side))
    )
    recall = float(correct_opportunities.sum() / max(1, opportunities.sum()))
    prediction = (score * direction).where(selected).replace([np.inf, -np.inf], np.nan)
    ic_frame = pd.concat([prediction, labels["forward_return"]], axis=1).dropna()
    has_ic_variation = (
        len(ic_frame) >= 3
        and float(ic_frame.iloc[:, 0].std(ddof=0)) > 0.0
        and float(ic_frame.iloc[:, 1].std(ddof=0)) > 0.0
    )
    information_coefficient = (
        float(ic_frame.iloc[:, 0].corr(ic_frame.iloc[:, 1]))
        if has_ic_variation
        else 0.0
    )
    trade_labels = labels.reindex(trades.index)
    directional_mfe = pd.Series(
        np.where(trade_side > 0, trade_labels["mfe"], -trade_labels["mae"]),
        index=trades.index,
    )
    directional_mae = pd.Series(
        np.where(trade_side > 0, trade_labels["mae"], -trade_labels["mfe"]),
        index=trades.index,
    )
    return BacktestMetrics(
        name=name,
        horizon_bars=horizon_bars,
        sample_count=sample_count,
        mean_return=mean_return,
        median_return=median_return,
        hit_rate=hit_rate,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        maximum_drawdown=max_drawdown,
        profit_factor=float(profit_factor),
        precision=precision,
        recall=recall,
        information_coefficient=information_coefficient,
        mean_mfe=float(directional_mfe.mean()) if not directional_mfe.empty else 0.0,
        mean_mae=float(directional_mae.mean()) if not directional_mae.empty else 0.0,
        mean_absolute_return=float(trade_labels["absolute_return"].mean()),
        mean_future_volatility=float(trade_labels["future_volatility"].mean()),
        breakout_rate=float(trade_labels["breakout"].mean()),
        reversal_rate=float(trade_labels["reversal"].mean()),
        mean_time_to_move=float(trade_labels["time_to_move"].mean()),
        non_overlapping=not allow_overlapping,
        overlapping_candidates_excluded=excluded_overlap,
        warnings=warnings,
    )


def run_backtest(
    signals: pd.DataFrame,
    price_frame: pd.DataFrame,
    horizon_bars: int,
    name: str = "mapi",
    score_threshold: float = 60.0,
    min_direction: float = 0.10,
    signal_delay_bars: int = 1,
    transaction_cost_bps: float = 10.0,
    spread_bps: float = 2.0,
    slippage_bps: float = 5.0,
    allow_overlapping: bool = False,
    score_column: str | None = None,
) -> BacktestMetrics:
    """Compatibility wrapper for the event-study engine."""

    return run_event_study(
        signals=signals,
        price_frame=price_frame,
        horizon_bars=horizon_bars,
        name=name,
        score_threshold=score_threshold,
        min_direction=min_direction,
        signal_delay_bars=signal_delay_bars,
        transaction_cost_bps=transaction_cost_bps,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        allow_overlapping=allow_overlapping,
        score_column=score_column,
    )


def compare_score_buckets(
    signals: pd.DataFrame,
    price_frame: pd.DataFrame,
    horizon_bars: int,
    buckets: tuple[int, ...] = (0, 20, 40, 60, 80, 101),
    score_column: str = "mapi_score",
    **kwargs: object,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    score = signals[score_column].fillna(0.0)
    for left, right in zip(buckets[:-1], buckets[1:]):
        bucket_signals = signals.copy()
        in_bucket = (score >= left) & (score < right)
        bucket_signals.loc[~in_bucket, score_column] = 0.0
        bucket_signals.loc[~in_bucket, "mapi_direction"] = 0.0
        metrics = run_backtest(
            bucket_signals,
            price_frame,
            horizon_bars=horizon_bars,
            name=f"score_{left}_{right - 1}",
            score_threshold=max(float(left), 1e-9),
            score_column=score_column,
            **kwargs,
        )
        row = metrics.to_dict()
        row["bucket"] = f"{left}-{right - 1}"
        rows.append(row)
    return pd.DataFrame(rows)
