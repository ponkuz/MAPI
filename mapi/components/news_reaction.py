from __future__ import annotations

import pandas as pd

from mapi.components.base import ComponentContext, empty_component_frame
from mapi.config import MapiConfig
from mapi.models import HorizonConfig


class NewsReactionDivergence:
    name = "news_reaction_divergence"
    family = "news"

    def calculate(
        self,
        price_frame: pd.DataFrame,
        context: ComponentContext,
        horizon: HorizonConfig,
        config: MapiConfig,
    ) -> pd.DataFrame:
        return empty_component_frame(
            price_frame.index,
            "News data is not supplied in Phase 1; news reaction divergence is inactive",
        )

