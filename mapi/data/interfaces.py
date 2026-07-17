from __future__ import annotations

from typing import Protocol

import pandas as pd


class PriceDataProvider(Protocol):
    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        ...


class FundamentalDataProvider(Protocol):
    def get_fundamentals(self, symbol: str) -> pd.DataFrame | None:
        ...


class NewsDataProvider(Protocol):
    def get_news(self, symbol: str) -> pd.DataFrame | None:
        ...


class OptionsDataProvider(Protocol):
    def get_options(self, symbol: str) -> pd.DataFrame | None:
        ...


class SentimentDataProvider(Protocol):
    def get_sentiment(self, symbol: str) -> pd.DataFrame | None:
        ...


class MarketContextProvider(Protocol):
    def get_benchmark(self, symbol: str | None = None) -> pd.DataFrame | None:
        ...

    def get_sector(self, symbol: str) -> pd.DataFrame | None:
        ...

