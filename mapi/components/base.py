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
            "direction_semantics": "hypothesized_forward_direction",
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
    if "forecast_direction" not in output:
        output["forecast_direction"] = output.get("direction", 0.0)
    if "observed_pressure" not in output:
        output["observed_pressure"] = output.get("direction", 0.0)
    output["direction"] = output["forecast_direction"]
    for column, default in {
        "anomaly_strength": 0.0,
        "direction": 0.0,
        "forecast_direction": 0.0,
        "observed_pressure": 0.0,
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
        output["direction_semantics"] = "hypothesized_forward_direction"
    output["direction_semantics"] = output["direction_semantics"].fillna(
        "hypothesized_forward_direction"
    ).astype(str)
    if "reason" not in output:
        output["reason"] = default_reason
    output["reason"] = output["reason"].fillna(default_reason).astype(str)
    if "metrics" not in output:
        output["metrics"] = [{} for _ in range(len(output))]
    output["metrics"] = output["metrics"].apply(lambda value: value if isinstance(value, dict) else {})
    return output
