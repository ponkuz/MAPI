from __future__ import annotations

import pandas as pd

from mapi.components.base import ComponentContext, empty_component_frame
from mapi.config import MapiConfig
from mapi.models import HorizonConfig


class FundamentalExpectationsDivergence:
    name = "fundamental_expectations_divergence"
    family = "fundamentals"

    def calculate(
        self,
        price_frame: pd.DataFrame,
        context: ComponentContext,
        horizon: HorizonConfig,
        config: MapiConfig,
    ) -> pd.DataFrame:
        return empty_component_frame(
            price_frame.index,
            "Fundamental point-in-time data is not supplied in Phase 1; fundamentals are inactive",
        )

