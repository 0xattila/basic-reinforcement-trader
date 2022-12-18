from datetime import datetime, timedelta

from data.exceptions import TimeframeException


class Timeframe:
    def __init__(self, value: int, unit: str) -> None:
        assert unit in ["m", "h"], f"wrong unit: {unit}"

        self.__value = value
        self.__unit = unit

    @classmethod
    def from_string(cls, timeframe: str):
        """Use this method to parse string like '5m'"""
        value = int(timeframe[:-1])
        unit = timeframe[-1]
        return cls(value, unit)

    @property
    def symbol(self) -> str:
        """Symbol to use at data downloading."""
        return f"{self.__value}{self.__unit}"

    @property
    def pandas_symbol(self) -> str:
        """Symbol for pandas date_range."""
        unit = self.__unit

        if unit == "m":
            unit = "T"

        return f"{self.__value}{unit}"

    @property
    def delta(self) -> timedelta:
        """Timeframe's delta"""
        deltas = {
            "m": timedelta(minutes=self.__value),
            "h": timedelta(hours=self.__value),
        }
        return deltas[self.__unit]

    def __repr__(self) -> str:
        return self.symbol

    def ceil_date(self, date: datetime) -> datetime:
        """Ceil date by timeframe."""
        date = date.replace(microsecond=0, second=0)

        if self.__unit == "m":
            delta = self.__value - date.minute % self.__value
            return date + timedelta(minutes=delta)

        if self.__unit == "h":
            date = date.replace(minute=0)
            delta = self.__value - date.hour % self.__value
            return date + timedelta(hours=delta)

        raise TimeframeException("unable to ceil date")

    def increase_date(self, date: datetime) -> datetime:
        """Increase date by timeframe's delta."""
        return date + self.delta
