from __future__ import annotations

import numpy as np
import pandas as pd

from mapi.data.alignment import align_point_in_time
from mapi.data.validation import normalize_ohlcv
from mapi.normalization import historical_zscore, rolling_percentile_rank
from mapi.research.backtest import run_event_study
from mapi.research.matching import (
    candidate_frequency,
    chronological_masks,
    fit_frequency_matched_threshold,
    test_only_signals,
)


def generate_baselines(
    price_frame: pd.DataFrame,
    seed: int = 42,
    signal_frequency: float = 0.10,
    sector_frame: pd.DataFrame | None = None,
    mapi_signals: pd.DataFrame | None = None,
    score_column: str = "mapi_score",
) -> dict[str, pd.DataFrame]:
    prices = normalize_ohlcv(price_frame) if "timestamp" in price_frame.columns else price_frame
    close = prices["close"]
    returns = close.pct_change()
    baselines: dict[str, pd.DataFrame] = {}
    baselines["random_same_frequency"] = _random_signal(prices.index, seed, signal_frequency)
    baselines["previous_day_return"] = _frame(
        prices.index,
        score=rolling_percentile_rank(returns.abs(), 60, 20).fillna(0.0) * 100.0,
        direction=np.sign(returns.shift(1).fillna(0.0)),
    )
    baselines["pure_volume_zscore"] = _frame(
        prices.index,
        score=(historical_zscore(np.log1p(prices["volume"].astype(float)), 60, 20).abs() / 3.0)
        .clip(0.0, 1.0)
        .fillna(0.0)
        * 100.0,
        direction=np.sign(returns.fillna(0.0)),
    )
    baselines["moving_average_crossover"] = _moving_average_crossover(prices)
    baselines["rsi_reversal"] = _rsi_reversal(prices)
    baselines["macd_direction"] = _macd_direction(prices)
    baselines["always_long_fixed_horizon"] = _frame(
        prices.index,
        score=pd.Series(100.0, index=prices.index),
        direction=pd.Series(1.0, index=prices.index),
    )
    if sector_frame is not None:
        baselines["sector_relative_strength"] = _sector_relative_strength(
            prices, sector_frame
        )
    if mapi_signals is not None:
        baselines["equal_weight_components"] = _equal_weight_components(mapi_signals)
        baselines["shuffled_mapi_scores"] = _shuffled_mapi(
            mapi_signals, seed, score_column
        )
        baselines["isolated_stock_sector_residual"] = _isolated_component(
            mapi_signals, "stock_sector_divergence"
        )
        baselines["isolated_volatility_anomaly"] = _isolated_component(
            mapi_signals, "volatility_anomaly"
        )
    return baselines


def compare_baselines(
    price_frame: pd.DataFrame,
    horizon_bars: int,
    sector_frame: pd.DataFrame | None = None,
    mapi_signals: pd.DataFrame | None = None,
    seed: int = 42,
    signal_frequency: float = 0.10,
    reference_score_column: str = "mapi_score",
    reference_score_threshold: float = 60.0,
    min_direction: float = 0.10,
    fit_fraction: float = 0.70,
    **event_study_kwargs: object,
) -> pd.DataFrame:
    """Evaluate each negative control under identical event-study assumptions."""

    prices = normalize_ohlcv(price_frame) if "timestamp" in price_frame.columns else price_frame
    fit_mask, test_mask = chronological_masks(prices.index, fit_fraction)
    target_frequency = signal_frequency
    if mapi_signals is not None:
        target_frequency = candidate_frequency(
            mapi_signals.reindex(prices.index),
            reference_score_column,
            reference_score_threshold,
            min_direction,
            fit_mask,
        )
    rows: list[dict[str, object]] = []
    for name, signals in generate_baselines(
        price_frame,
        seed=seed,
        signal_frequency=target_frequency,
        sector_frame=sector_frame,
        mapi_signals=mapi_signals,
        score_column=reference_score_column,
    ).items():
        match = fit_frequency_matched_threshold(
            signals,
            "baseline_score",
            target_frequency,
            min_direction,
            fit_mask,
        )
        evaluation_signals = test_only_signals(signals, test_mask)
        metrics = run_event_study(
            evaluation_signals,
            prices,
            horizon_bars=horizon_bars,
            name=name,
            score_threshold=match.threshold,
            min_direction=min_direction,
            score_column="baseline_score",
            evaluation_mask=test_mask,
            **event_study_kwargs,
        )
        row = metrics.to_dict()
        row.update(
            {
                "fitted_threshold": match.threshold,
                "target_fit_frequency": target_frequency,
                "candidate_fit_frequency": match.fitted_frequency,
                "candidate_test_frequency": candidate_frequency(
                    signals,
                    "baseline_score",
                    match.threshold,
                    min_direction,
                    test_mask,
                ),
                "selected_event_count": metrics.sample_count,
                "excluded_overlap_count": metrics.overlapping_candidates_excluded,
                "reference_score_column": reference_score_column,
            }
        )
        rows.append(row)
    rows.append(
        _full_period_buy_and_hold(
            prices,
            test_mask,
            event_study_kwargs,
            reference_score_column,
        )
    )
    return pd.DataFrame(rows)


def random_control_distribution(
    price_frame: pd.DataFrame,
    horizon_bars: int,
    seeds: tuple[int, ...] = tuple(range(10)),
    signal_frequency: float = 0.10,
    fit_fraction: float = 0.70,
    min_direction: float = 0.10,
    mapi_signals: pd.DataFrame | None = None,
    reference_score_column: str = "mapi_score",
    reference_score_threshold: float = 60.0,
    **event_study_kwargs: object,
) -> pd.DataFrame:
    """Return multiple deterministic random-control outcomes, one per seed."""

    prices = normalize_ohlcv(price_frame) if "timestamp" in price_frame.columns else price_frame
    fit_mask, test_mask = chronological_masks(prices.index, fit_fraction)
    target_frequency = signal_frequency
    if mapi_signals is not None:
        target_frequency = candidate_frequency(
            mapi_signals.reindex(prices.index),
            reference_score_column,
            reference_score_threshold,
            min_direction,
            fit_mask,
        )
    rows: list[dict[str, object]] = []
    for seed in seeds:
        signals = _random_signal(prices.index, seed, target_frequency)
        match = fit_frequency_matched_threshold(
            signals,
            "baseline_score",
            target_frequency,
            min_direction,
            fit_mask,
        )
        metrics = run_event_study(
            test_only_signals(signals, test_mask),
            prices,
            horizon_bars=horizon_bars,
            name=f"random_seed_{seed}",
            score_threshold=match.threshold,
            min_direction=min_direction,
            score_column="baseline_score",
            evaluation_mask=test_mask,
            **event_study_kwargs,
        )
        rows.append(
            {
                "seed": seed,
                **metrics.to_dict(),
                "fitted_threshold": match.threshold,
                "target_fit_frequency": target_frequency,
                "candidate_fit_frequency": match.fitted_frequency,
                "candidate_test_frequency": candidate_frequency(
                    signals,
                    "baseline_score",
                    match.threshold,
                    min_direction,
                    test_mask,
                ),
                "selected_event_count": metrics.sample_count,
                "excluded_overlap_count": metrics.overlapping_candidates_excluded,
                "reference_score_column": reference_score_column,
            }
        )
    return pd.DataFrame(rows)


def _frame(index: pd.Index, score: pd.Series, direction: pd.Series | np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "baseline_score": pd.Series(score, index=index).fillna(0.0).clip(0.0, 100.0),
            "mapi_direction": pd.Series(direction, index=index).fillna(0.0).clip(-1.0, 1.0),
            "mapi_confidence": 1.0,
        },
        index=index,
    )


def _random_signal(index: pd.Index, seed: int, signal_frequency: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    active = rng.random(len(index)) < signal_frequency
    direction = rng.choice([-1.0, 1.0], size=len(index))
    score = np.where(active, 100.0, 0.0)
    return _frame(index, pd.Series(score, index=index), pd.Series(direction, index=index))


def _moving_average_crossover(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"]
    short = close.rolling(20, min_periods=10).mean()
    long = close.rolling(50, min_periods=25).mean()
    spread = (short / long - 1.0).replace([np.inf, -np.inf], np.nan)
    score = (historical_zscore(spread.abs(), 80, 25).abs() / 3.0).clip(0.0, 1.0) * 100.0
    return _frame(prices.index, score.fillna(0.0), np.sign(spread.fillna(0.0)))


def _rsi_reversal(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"]
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    direction = pd.Series(0.0, index=prices.index)
    direction[rsi < 30.0] = 1.0
    direction[rsi > 70.0] = -1.0
    score = ((rsi - 50.0).abs() * 2.0).clip(0.0, 100.0)
    return _frame(prices.index, score.fillna(0.0), direction)


def _macd_direction(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"]
    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    spread = macd - signal
    score = (historical_zscore(spread.abs(), 80, 25).abs() / 3.0).clip(0.0, 1.0) * 100.0
    return _frame(prices.index, score.fillna(0.0), np.sign(spread.fillna(0.0)))


def _sector_relative_strength(
    prices: pd.DataFrame, sector_frame: pd.DataFrame
) -> pd.DataFrame:
    sector = (
        normalize_ohlcv(sector_frame)
        if "timestamp" in sector_frame.columns
        else sector_frame
    )
    alignment = align_point_in_time(prices, sector)
    relative_return = prices["close"].pct_change(5) - alignment.frame["close"].pct_change(
        5, fill_method=None
    )
    relative_z = historical_zscore(relative_return, 60, 20)
    return _frame(
        prices.index,
        (relative_z.abs() / 3.0).clip(0.0, 1.0).fillna(0.0) * 100.0,
        np.sign(relative_z.fillna(0.0)),
    )


def _equal_weight_components(signals: pd.DataFrame) -> pd.DataFrame:
    scores: list[float] = []
    directions: list[float] = []
    confidences: list[float] = []
    for signal in signals["signal"]:
        active = [item for item in signal.anomaly_components if item.confidence > 0.0]
        if not active:
            scores.append(0.0)
            directions.append(0.0)
            confidences.append(0.0)
            continue
        component_scores = [item.anomaly_strength * item.novelty for item in active]
        scores.append(100.0 * float(np.mean(component_scores)))
        directions.append(float(np.mean([item.direction for item in active])))
        confidences.append(float(np.mean([item.confidence for item in active])))
    return pd.DataFrame(
        {
            "baseline_score": scores,
            "mapi_direction": directions,
            "mapi_confidence": confidences,
        },
        index=signals.index,
    )


def _shuffled_mapi(
    signals: pd.DataFrame, seed: int, score_column: str
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    source_score = signals[score_column]
    positions = rng.permutation(len(signals))
    return pd.DataFrame(
        {
            "baseline_score": source_score.to_numpy()[positions],
            "mapi_direction": signals["mapi_direction"].to_numpy()[positions],
            "mapi_confidence": signals["mapi_confidence"].to_numpy()[positions],
        },
        index=signals.index,
    )


def _isolated_component(signals: pd.DataFrame, component_name: str) -> pd.DataFrame:
    scores: list[float] = []
    directions: list[float] = []
    confidences: list[float] = []
    for signal in signals["signal"]:
        component = next(
            (item for item in signal.anomaly_components if item.name == component_name),
            None,
        )
        if component is None or component.confidence <= 0.0:
            scores.append(0.0)
            directions.append(0.0)
            confidences.append(0.0)
            continue
        scores.append(100.0 * component.anomaly_strength * component.novelty)
        directions.append(component.direction)
        confidences.append(component.confidence)
    return pd.DataFrame(
        {
            "baseline_score": scores,
            "mapi_direction": directions,
            "mapi_confidence": confidences,
        },
        index=signals.index,
    )


def _full_period_buy_and_hold(
    prices: pd.DataFrame,
    test_mask: pd.Series,
    event_study_kwargs: dict[str, object],
    reference_score_column: str,
) -> dict[str, object]:
    test_prices = prices.loc[test_mask]
    total_return = 0.0
    if len(test_prices) >= 2 and float(test_prices["close"].iloc[0]) > 0.0:
        total_return = float(
            test_prices["close"].iloc[-1] / test_prices["close"].iloc[0] - 1.0
        )
        costs = sum(
            float(event_study_kwargs.get(name, 0.0))
            for name in ("transaction_cost_bps", "spread_bps", "slippage_bps")
        ) / 10000.0
        total_return -= costs
    return {
        "name": "full_period_buy_and_hold",
        "analysis_type": "full_period_benchmark",
        "sample_count": 1 if len(test_prices) >= 2 else 0,
        "selected_event_count": 1 if len(test_prices) >= 2 else 0,
        "total_return": total_return,
        "mean_return": total_return,
        "fitted_threshold": None,
        "target_fit_frequency": None,
        "candidate_fit_frequency": None,
        "candidate_test_frequency": 1.0 if len(test_prices) >= 2 else 0.0,
        "excluded_overlap_count": 0,
        "reference_score_column": reference_score_column,
        "warnings": [
            "Full-period buy-and-hold is a capital-path benchmark, not an event study."
        ],
    }
