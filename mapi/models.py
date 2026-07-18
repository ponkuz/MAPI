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
    expected_frequency: str = "any"


@dataclass
class ComponentSignal:
    name: str
    family: str
    anomaly_strength: float
    direction: float
    observed_pressure: float
    directional_evidence_strength: float
    direction_semantics: str
    confidence: float
    weight: float
    novelty: float
    historical_extremeness: float
    recurrence_rate: float
    redundancy_penalty: float
    direction_contract_warning: str | None = None
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    weight_factors: dict[str, float] = field(default_factory=dict)

    @property
    def intensity_effective_score(self) -> float:
        return (
            clamp(self.anomaly_strength, 0.0, 1.0)
            * clamp(self.confidence, 0.0, 1.0)
            * clamp(self.weight, 0.0, 1.0)
            * clamp(self.redundancy_penalty, 0.0, 1.0)
        )

    @property
    def alert_effective_score(self) -> float:
        return self.intensity_effective_score * clamp(self.novelty, 0.0, 1.0)

    @property
    def effective_score(self) -> float:
        """Compatibility alias for the recurrence-adjusted alert contribution."""

        return self.alert_effective_score

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "name": self.name,
                "family": self.family,
                "anomaly_strength": clamp(self.anomaly_strength, 0.0, 1.0),
                "direction": clamp(self.direction, -1.0, 1.0),
                "forecast_direction": clamp(self.direction, -1.0, 1.0),
                "observed_pressure": clamp(self.observed_pressure, -1.0, 1.0),
                "directional_evidence_strength": clamp(
                    self.directional_evidence_strength, 0.0, 1.0
                ),
                "direction_semantics": self.direction_semantics,
                "direction_contract_warning": self.direction_contract_warning,
                "confidence": clamp(self.confidence, 0.0, 1.0),
                "weight": clamp(self.weight, 0.0, 1.0),
                "novelty": clamp(self.novelty, 0.0, 1.0),
                "historical_extremeness": clamp(
                    self.historical_extremeness, 0.0, 1.0
                ),
                "recurrence_rate": clamp(self.recurrence_rate, 0.0, 1.0),
                "redundancy_penalty": clamp(self.redundancy_penalty, 0.0, 1.0),
                "intensity_effective_score": self.intensity_effective_score,
                "alert_effective_score": self.alert_effective_score,
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
    mapi_intensity_score: float
    mapi_novelty_score: float
    mapi_alert_score: float
    mapi_actionability_score: float
    mapi_direction: float
    mapi_forecast_direction: float
    mapi_observed_pressure: float
    mapi_direction_semantics: str
    mapi_confidence: float
    mapi_regime: str
    regime_source: str
    regime_confidence: float
    anomaly_components: list[ComponentSignal]
    dominant_anomalies: list[str]
    dominant_intensity_anomalies: list[str]
    dominant_alert_anomalies: list[str]
    data_quality_score: float
    ohlcv_quality_score: float
    evidence_coverage_score: float
    signal_version: str
    implementation_version: str
    config_fingerprint: str
    algorithm_revision: str
    data_contract_version: str
    anomaly_state: str
    anomaly_first_detected_at: Any | None
    anomaly_age_bars: int
    anomaly_trend: str
    confirmation_count: int
    recent_move_extremeness: float
    directional_move_since_detection: float
    directional_realization_score: float
    already_realized_score: float
    input_interval_seconds: float | None
    horizon_frequency_compatible: bool
    horizon_warning: str | None
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
                "mapi_intensity_score": round(
                    clamp(self.mapi_intensity_score, 0.0, 100.0), 4
                ),
                "mapi_novelty_score": round(
                    clamp(self.mapi_novelty_score, 0.0, 100.0), 4
                ),
                "mapi_alert_score": round(
                    clamp(self.mapi_alert_score, 0.0, 100.0), 4
                ),
                "mapi_actionability_score": round(
                    clamp(self.mapi_actionability_score, 0.0, 100.0), 4
                ),
                "mapi_direction": round(clamp(self.mapi_direction, -1.0, 1.0), 4),
                "mapi_forecast_direction": round(
                    clamp(self.mapi_forecast_direction, -1.0, 1.0), 4
                ),
                "mapi_observed_pressure": round(
                    clamp(self.mapi_observed_pressure, -1.0, 1.0), 4
                ),
                "mapi_direction_semantics": self.mapi_direction_semantics,
                "mapi_confidence": round(clamp(self.mapi_confidence, 0.0, 1.0), 4),
                "mapi_regime": self.mapi_regime,
                "regime_source": self.regime_source,
                "regime_confidence": round(
                    clamp(self.regime_confidence, 0.0, 1.0), 4
                ),
                "anomaly_components": [
                    component.to_dict() for component in self.anomaly_components
                ],
                "dominant_anomalies": self.dominant_anomalies,
                "dominant_intensity_anomalies": self.dominant_intensity_anomalies,
                "dominant_alert_anomalies": self.dominant_alert_anomalies,
                "data_quality_score": round(clamp(self.data_quality_score, 0.0, 1.0), 4),
                "ohlcv_quality_score": round(
                    clamp(self.ohlcv_quality_score, 0.0, 1.0), 4
                ),
                "evidence_coverage_score": round(
                    clamp(self.evidence_coverage_score, 0.0, 1.0), 4
                ),
                "signal_version": self.signal_version,
                "implementation_version": self.implementation_version,
                "config_fingerprint": self.config_fingerprint,
                "algorithm_revision": self.algorithm_revision,
                "data_contract_version": self.data_contract_version,
                "anomaly_state": self.anomaly_state,
                "anomaly_first_detected_at": self.anomaly_first_detected_at,
                "anomaly_age_bars": self.anomaly_age_bars,
                "anomaly_trend": self.anomaly_trend,
                "confirmation_count": self.confirmation_count,
                "recent_move_extremeness": round(
                    clamp(self.recent_move_extremeness, 0.0, 1.0), 4
                ),
                "directional_move_since_detection": round(
                    self.directional_move_since_detection, 6
                ),
                "directional_realization_score": round(
                    clamp(self.directional_realization_score, 0.0, 1.0), 4
                ),
                "already_realized_score": round(
                    clamp(self.already_realized_score, 0.0, 1.0), 4
                ),
                "input_interval_seconds": self.input_interval_seconds,
                "horizon_frequency_compatible": self.horizon_frequency_compatible,
                "horizon_warning": self.horizon_warning,
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
    gross_directional_accuracy: float
    net_profitable_event_rate: float
    large_move_capture_rate: float
    event_return_mean_to_std: float
    event_return_mean_to_downside_std: float
    mean_return_ci_lower: float
    mean_return_ci_upper: float
    bootstrap_samples: int
    maximum_drawdown: float
    profit_factor: float
    active_event_ic: float
    mean_intrabar_mfe: float
    mean_intrabar_mae: float
    mean_absolute_return: float
    mean_future_volatility: float
    intrabar_breakout_rate: float
    reversal_rate: float
    mean_time_to_move: float
    analysis_type: str = "event_study"
    non_overlapping: bool = True
    overlapping_candidates_excluded: int = 0
    score_column_used: str = "mapi_score"
    direction_column_used: str = "mapi_forecast_direction"
    excluded_low_confidence_count: int = 0
    excluded_low_quality_count: int = 0
    excluded_frequency_mismatch_count: int = 0
    evaluation_count: int = 0
    evaluation_start: Any | None = None
    evaluation_end: Any | None = None
    selected_event_timestamps: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.net_profitable_event_rate

    @property
    def sharpe_ratio(self) -> float:
        return self.event_return_mean_to_std

    @property
    def sortino_ratio(self) -> float:
        return self.event_return_mean_to_downside_std

    @property
    def precision(self) -> float:
        return self.gross_directional_accuracy

    @property
    def recall(self) -> float:
        return self.large_move_capture_rate

    @property
    def mean_mfe(self) -> float:
        return self.mean_intrabar_mfe

    @property
    def mean_mae(self) -> float:
        return self.mean_intrabar_mae

    @property
    def breakout_rate(self) -> float:
        return self.intrabar_breakout_rate

    @property
    def information_coefficient(self) -> float:
        """Compatibility alias; the statistic uses selected events only."""

        return self.active_event_ic

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload.update(
            {
                "hit_rate": self.hit_rate,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "precision": self.precision,
                "recall": self.recall,
                "mean_mfe": self.mean_mfe,
                "mean_mae": self.mean_mae,
                "breakout_rate": self.breakout_rate,
                "information_coefficient": self.information_coefficient,
                "event_return_sharpe": self.event_return_mean_to_std,
                "event_sequence_drawdown": self.maximum_drawdown,
                "event_profit_factor": self.profit_factor,
            }
        )
        compatibility_warning = (
            "Legacy hit_rate/precision/recall/Sharpe/Sortino/MFE/MAE/breakout and "
            "information_coefficient fields are compatibility aliases; use the explicitly "
            "named event metrics, including active_event_ic."
        )
        payload["warnings"] = [*self.warnings, compatibility_warning]
        return json_safe(payload)
