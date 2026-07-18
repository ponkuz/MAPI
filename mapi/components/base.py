from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from mapi.config import MapiConfig
from mapi.models import HorizonConfig


@dataclass
class ComponentContext:
    symbol: str
    sector_frame: pd.DataFrame | None = None
    benchmark_frame: pd.DataFrame | None = None
    regime: pd.Series | None = None


class AnomalyComponent(Protocol):
    name: str
    family: str

    def calculate(
        self,
        price_frame: pd.DataFrame,
        context: ComponentContext,
        horizon: HorizonConfig,
        config: MapiConfig,
    ) -> pd.DataFrame:
        ...


def empty_component_frame(index: pd.Index, reason: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anomaly_strength": 0.0,
            "direction": 0.0,
            "forecast_direction": 0.0,
            "observed_pressure": 0.0,
            "directional_evidence_strength": 0.0,
            "direction_semantics": "direction_neutral_unavailable",
            "direction_contract_warning": None,
            "confidence": 0.0,
            "novelty": 0.0,
            "historical_extremeness": 0.0,
            "recurrence_rate": 0.0,
            "reason": reason,
            "metrics": [{} for _ in range(len(index))],
        },
        index=index,
    )


def finalize_component_frame(
    frame: pd.DataFrame,
    index: pd.Index,
    default_reason: str,
) -> pd.DataFrame:
    output = frame.reindex(index)
    fallback_fields: list[str] = []
    if "forecast_direction" not in output:
        output["forecast_direction"] = output.get("direction", 0.0)
        fallback_fields.append("forecast_direction")
    if "observed_pressure" not in output:
        output["observed_pressure"] = output.get("direction", 0.0)
        fallback_fields.append("observed_pressure")
    if "directional_evidence_strength" not in output:
        output["directional_evidence_strength"] = (
            output["forecast_direction"].astype(float).abs() > 0.0
        ).astype(float)
        fallback_fields.append("directional_evidence_strength")
    output["direction"] = output["forecast_direction"]
    for column, default in {
        "anomaly_strength": 0.0,
        "direction": 0.0,
        "forecast_direction": 0.0,
        "observed_pressure": 0.0,
        "directional_evidence_strength": 0.0,
        "confidence": 0.0,
        "novelty": 0.0,
        "historical_extremeness": 0.0,
        "recurrence_rate": 0.0,
    }.items():
        if column not in output:
            output[column] = default
        output[column] = output[column].astype(float).clip(
            lower=-1.0
            if column in {"direction", "forecast_direction", "observed_pressure"}
            else 0.0,
            upper=1.0,
        )
        output[column] = output[column].fillna(0.0)
    if "direction_semantics" not in output:
        output["direction_semantics"] = "deprecated_implicit_direction_fallback"
        fallback_fields.append("direction_semantics")
    output["direction_semantics"] = output["direction_semantics"].fillna(
        "deprecated_implicit_direction_fallback"
    ).astype(str)
    if "direction_contract_warning" not in output:
        warning = (
            "Legacy component direction fallback populated: "
            + ", ".join(fallback_fields)
            if fallback_fields
            else None
        )
        output["direction_contract_warning"] = warning
    if "reason" not in output:
        output["reason"] = default_reason
    output["reason"] = output["reason"].fillna(default_reason).astype(str)
    if "metrics" not in output:
        output["metrics"] = [{} for _ in range(len(output))]
    output["metrics"] = output["metrics"].apply(lambda value: value if isinstance(value, dict) else {})
    return output
