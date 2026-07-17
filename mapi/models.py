from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd


def clamp(value: float, low: float, high: float) -> float:
    if not isfinite(value):
        return low
    return max(low, min(high, value))


def json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if not isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


@dataclass(frozen=True)
class HorizonConfig:
    name: str
    return_window: int
    rolling_window: int
    min_periods: int


@dataclass
class ComponentSignal:
    name: str
    family: str
    anomaly_strength: float
    direction: float
    confidence: float
    weight: float
    novelty: float
    redundancy_penalty: float
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    weight_factors: dict[str, float] = field(default_factory=dict)

    @property
    def effective_score(self) -> float:
        return (
            clamp(self.anomaly_strength, 0.0, 1.0)
            * clamp(self.confidence, 0.0, 1.0)
            * clamp(self.weight, 0.0, 1.0)
            * clamp(self.novelty, 0.0, 1.0)
            * clamp(self.redundancy_penalty, 0.0, 1.0)
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "name": self.name,
                "family": self.family,
                "anomaly_strength": clamp(self.anomaly_strength, 0.0, 1.0),
                "direction": clamp(self.direction, -1.0, 1.0),
                "confidence": clamp(self.confidence, 0.0, 1.0),
                "weight": clamp(self.weight, 0.0, 1.0),
                "novelty": clamp(self.novelty, 0.0, 1.0),
                "redundancy_penalty": clamp(self.redundancy_penalty, 0.0, 1.0),
                "effective_score": self.effective_score,
                "reason": self.reason,
                "metrics": self.metrics,
                "weight_factors": self.weight_factors,
            }
        )


@dataclass
class MapiSignal:
    symbol: str
    timestamp: Any
    horizon: str
    mapi_score: float
    mapi_raw_score: float
    mapi_actionability_score: float
    mapi_direction: float
    mapi_confidence: float
    mapi_regime: str
    anomaly_components: list[ComponentSignal]
    dominant_anomalies: list[str]
    data_quality_score: float
    ohlcv_quality_score: float
    evidence_coverage_score: float
    signal_version: str
    anomaly_state: str
    anomaly_first_detected_at: Any | None
    anomaly_age_bars: int
    anomaly_trend: str
    confirmation_count: int
    already_realized_score: float
    machine_reasons: list[dict[str, Any]]
    human_summary: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "symbol": self.symbol,
                "timestamp": self.timestamp,
                "horizon": self.horizon,
                "mapi_score": round(clamp(self.mapi_score, 0.0, 100.0), 4),
                "mapi_raw_score": round(clamp(self.mapi_raw_score, 0.0, 100.0), 4),
                "mapi_actionability_score": round(
                    clamp(self.mapi_actionability_score, 0.0, 100.0), 4
                ),
                "mapi_direction": round(clamp(self.mapi_direction, -1.0, 1.0), 4),
                "mapi_confidence": round(clamp(self.mapi_confidence, 0.0, 1.0), 4),
                "mapi_regime": self.mapi_regime,
                "anomaly_components": [
                    component.to_dict() for component in self.anomaly_components
                ],
                "dominant_anomalies": self.dominant_anomalies,
                "data_quality_score": round(clamp(self.data_quality_score, 0.0, 1.0), 4),
                "ohlcv_quality_score": round(
                    clamp(self.ohlcv_quality_score, 0.0, 1.0), 4
                ),
                "evidence_coverage_score": round(
                    clamp(self.evidence_coverage_score, 0.0, 1.0), 4
                ),
                "signal_version": self.signal_version,
                "anomaly_state": self.anomaly_state,
                "anomaly_first_detected_at": self.anomaly_first_detected_at,
                "anomaly_age_bars": self.anomaly_age_bars,
                "anomaly_trend": self.anomaly_trend,
                "confirmation_count": self.confirmation_count,
                "already_realized_score": round(
                    clamp(self.already_realized_score, 0.0, 1.0), 4
                ),
                "machine_reasons": self.machine_reasons,
                "human_summary": self.human_summary,
            }
        )


@dataclass
class BacktestMetrics:
    name: str
    horizon_bars: int
    sample_count: int
    mean_return: float
    median_return: float
    hit_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    maximum_drawdown: float
    profit_factor: float
    precision: float
    recall: float
    information_coefficient: float
    mean_mfe: float
    mean_mae: float
    mean_absolute_return: float
    mean_future_volatility: float
    breakout_rate: float
    reversal_rate: float
    mean_time_to_move: float
    analysis_type: str = "event_study"
    non_overlapping: bool = True
    overlapping_candidates_excluded: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload.update(
            {
                "event_return_sharpe": self.sharpe_ratio,
                "event_sequence_drawdown": self.maximum_drawdown,
                "event_profit_factor": self.profit_factor,
            }
        )
        return json_safe(payload)
