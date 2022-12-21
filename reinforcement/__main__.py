import logging
from argparse import ArgumentParser

from data import Timeframe

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
)


parser = ArgumentParser(
    prog="reinforcement",
    description="this module learns to properly interact with the environment",
)

subparsers = parser.add_subparsers(dest="command")

train_parser = subparsers.add_parser("train", description="train the model")
train_parser.add_argument("-e", "--epochs", type=int, help="number of peochs")
train_parser.add_argument("-t", "--timeframe", type=str, help="Like '30m' or '1h'")

backtest_parser = subparsers.add_parser("backtest", description="backtest the model")
backtest_parser.add_argument("-t", "--timeframe", type=str, help="Like '30m' or '1h'")

args = parser.parse_args()

if args.command == "train":
    from reinforcement.operators import TrainOperator

    tf = Timeframe.from_string(args.timeframe)

    operator = TrainOperator(tf)
    operator.run(epochs=args.epochs)

if args.command == "backtest":
    from reinforcement.operators import BacktestOperator

    tf = Timeframe.from_string(args.timeframe)

    operator = BacktestOperator(tf)
    operator.run()
