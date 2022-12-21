import logging

from numpy import ndarray
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

logger = logging.getLogger(__name__)


def plot_train_epochs(train_rewards: ndarray, test_rewards: ndarray) -> None:
    logger.info("Plot train results")

    train_epochs = train_rewards.sum(axis=tuple(range(1, train_rewards.ndim)))
    test_epochs = test_rewards.sum(axis=tuple(range(1, train_rewards.ndim)))

    plt.figure(figsize=(12.8, 7.2), dpi=200)

    plt.plot(train_epochs, label="train epoch rewards")
    plt.plot(test_epochs, label="test epoch rewards")

    plt.title("Train epochs")
    plt.legend()

    plt.savefig("storage/train_epochs.jpg")


def plot_backtest(rewards: ndarray, returns: ndarray, pairs: list[str]) -> None:
    logger.info("Plot backtest results")

    n_pairs = len(pairs)

    fig, ax = plt.subplots(n_pairs, 1, figsize=(12.8, 7.2 * n_pairs), dpi=200)

    for i in range(n_pairs):
        ax[i].plot(returns[:, i].cumsum(), label="cummulative returns")
        ax[i].plot(rewards[1:, i].cumsum(), label="cummulative rewards")
        ax[i].set_title("%s results" % pairs[i])
        ax[i].legend()

    fig.savefig("storage/backtest.jpg")
