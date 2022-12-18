import logging
from datetime import datetime
from typing import Optional

import pymongo

from pandas import DataFrame

from connections import Database, Exchange

from data import Timeframe
from data.exceptions import KlinesException

logger = logging.getLogger(__name__)


def _get_max_date(pair: str, timeframe: Timeframe) -> Optional[datetime]:
    """If some data already exists then start downloading from the latest record."""
    with Database() as db:
        cursor = (
            db[f"klines_{timeframe.symbol}"]
            .find({"pair": pair})
            .sort("date", pymongo.DESCENDING)
            .limit(1)
        )

        if line := next(cursor, None):
            logger.info("%s has records", pair)
            return datetime.fromtimestamp(int(line["date"] / 1000))

    return None


def _insert_data(data: list, pair: str, timeframe: Timeframe) -> None:
    """Format and insert downloaded data into database."""
    df = DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
    df["pair"] = pair

    with Database() as db:
        db[f"klines_{timeframe.symbol}"].insert_many(df.to_dict("records"))


def download(
    pairs: list[str], since: datetime, until: Optional[datetime], timeframe: Timeframe
) -> None:
    """Download pairs from exchange and insert them into the database."""
    since = timeframe.ceil_date(since)

    if until is None:
        until = timeframe.ceil_date(datetime.now())
    else:
        until = timeframe.ceil_date(until)

    since -= timeframe.delta
    until -= timeframe.delta

    until_ts = int(until.timestamp() * 1000)

    if since >= until:
        raise KlinesException("since can't be greater or equal to until")

    for pair in pairs:
        pair_since = since

        if max_date := _get_max_date(pair, timeframe):
            pair_since = max_date + timeframe.delta  # since is inclusive, must increase

        logging.info("Download %s since %s until %s", pair, pair_since, until)

        while pair_since < until:
            since_ts = int(pair_since.timestamp() * 1000)

            with Exchange() as exchange:
                data = exchange.fetch_ohlcv(pair, timeframe.symbol, since=since_ts)

            data = [x for x in data if x[0] <= until_ts]

            pair_since = datetime.fromtimestamp(max(x[0] for x in data) / 1000)
            pair_since += timeframe.delta

            _insert_data(data, pair, timeframe)
