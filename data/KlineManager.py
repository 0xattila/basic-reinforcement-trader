from datetime import datetime
from typing import Callable

import pandas as pd
from pandas import DataFrame

import config

from connections import Database

from data import Timeframe


class KlineManager:
    def __init__(self, data: list, timeframe: Timeframe) -> None:
        df_raw = DataFrame(data)
        df_raw.drop("_id", axis=1, inplace=True)
        df_raw["date"] = df_raw["date"].apply(
            lambda x: datetime.fromtimestamp(x / 1000)
        )

        df_raw.sort_values(["pair", "date"], inplace=True)
        df_raw.reset_index(drop=True, inplace=True)

        self.pairs = df_raw["pair"].unique().tolist()
        self.btc_index = self.pairs.index("BTCUSDT")

        min_dates = []
        max_dates = []

        for pair in self.pairs:
            df_pair = df_raw.loc[df_raw["pair"] == pair]
            min_dates.append(df_pair["date"].min())
            max_dates.append(df_pair["date"].max())

        start = max(min_dates)
        end = min(max_dates)
        date_index = pd.date_range(
            start, end, freq=timeframe.pandas_symbol, name="date"
        )

        df_index = DataFrame(index=date_index)
        df = DataFrame()

        for pair in self.pairs:
            df_pair = df_raw.loc[df_raw["pair"] == pair].copy()
            df_pair.set_index("date", drop=True, inplace=True)

            df_pair = df_index.join(df_pair, how="left")

            df = pd.concat([df, df_pair])

        self.__df = df

    @classmethod
    def from_database(cls, timeframe: Timeframe):
        """Create straight from the database."""
        with Database() as db:
            cursor = db[f"klines_{timeframe.symbol}"].find(
                {"pair": {"$in": config.data["pairs"]}}
            )
            return cls(list(cursor), timeframe)

    @property
    def df(self) -> DataFrame:
        return self.__df.copy()

    def get_pair(self, pair: str) -> DataFrame:
        return self.__df.loc[self.__df["pair"] == pair].copy()

    def get_column(self, column: str) -> DataFrame:
        df = DataFrame()

        for pair in self.pairs:
            df[pair] = self.get_pair(pair)[column]

        return df

    def add_indicator(self, fn: Callable, name: str, dropna: bool = False) -> None:
        for pair in self.pairs:
            df = self.get_pair(pair)
            self.__df.loc[self.__df["pair"] == pair, name] = fn(df)

        if dropna:
            self.__df.dropna(inplace=True)
