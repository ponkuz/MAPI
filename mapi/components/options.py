from __future__ import annotations

import pandas as pd

from mapi.components.base import ComponentContext, empty_component_frame
from mapi.config import MapiConfig
from mapi.models import HorizonConfig


class OptionsAnomaly:
    name = "options_anomaly"
    family = "options"

    def calculate(
        self,
        price_frame: pd.DataFrame,
        context: ComponentContext,
        horizon: HorizonConfig,
        config: MapiConfig,
    ) -> pd.DataFrame:
        return empty_component_frame(
            price_frame.index,
            "Options data is not supplied in Phase 1; options anomaly is inactive",
        )

