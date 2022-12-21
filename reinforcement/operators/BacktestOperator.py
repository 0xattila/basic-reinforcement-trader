import logging
import time
from datetime import timedelta

from reinforcement import plot
from reinforcement.operators import Operator

logger = logging.getLogger(__name__)


class BacktestOperator(Operator):
    def run(self, **_) -> None:
        logger.info("Run backtest")

        self.agent.dqn.load_weights()
        self.agent.eval()

        t0 = time.perf_counter()
        test_rewards = self._run_episode(self.test_env)

        logger.info(
            "Backtest rewards: %.4f, duration: %s",
            test_rewards.sum(),
            timedelta(seconds=time.perf_counter() - t0),
        )

        plot.plot_backtest(
            test_rewards.numpy(), self.test_env.rewards.numpy(), self.klines.pairs
        )
