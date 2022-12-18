import logging
from argparse import ArgumentParser
import dateutil.parser as dateparser

import config

from data import klines, Timeframe

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
)


parser = ArgumentParser(
    prog="Data manager module",
    description="This module manages ohlcv data.",
)

subparsers = parser.add_subparsers(dest="command")

download_parser = subparsers.add_parser(
    "download", description="It downloads data from exchange."
)
download_parser.add_argument(
    "-s", "--since", type=str, help="ISO format date like '2020-01-01T00:00:00'"
)

download_parser.add_argument(
    "-u",
    "--until",
    type=str,
    help="(Optional) ISO format date like '2020-01-01T00:00:00'",
)

download_parser.add_argument("-t", "--timeframe", type=str, help="Like '30m' or '1h'")


args = parser.parse_args()

if args.command == "download":
    since = dateparser.parse(args.since)
    timeframe = Timeframe.from_string(args.timeframe)

    until = None
    if args.until is not None:
        until = dateparser.parse(args.until)

    klines.download(config.data["pairs"], since, until, timeframe)
