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
    for column, default in {
        "anomaly_strength": 0.0,
        "direction": 0.0,
        "confidence": 0.0,
        "novelty": 0.0,
        "historical_extremeness": 0.0,
        "recurrence_rate": 0.0,
    }.items():
        if column not in output:
            output[column] = default
        output[column] = output[column].astype(float).clip(
            lower=-1.0 if column == "direction" else 0.0,
            upper=1.0,
        )
        if column != "direction":
            output[column] = output[column].fillna(0.0)
        else:
            output[column] = output[column].fillna(0.0)
    if "reason" not in output:
        output["reason"] = default_reason
    output["reason"] = output["reason"].fillna(default_reason).astype(str)
    if "metrics" not in output:
        output["metrics"] = [{} for _ in range(len(output))]
    output["metrics"] = output["metrics"].apply(lambda value: value if isinstance(value, dict) else {})
    return output
