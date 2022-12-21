import logging
import time
from datetime import timedelta

import torchinfo

from reinforcement.operators import Operator

logger = logging.getLogger(__name__)


class TrainOperator(Operator):
    def run(self, **kwargs) -> None:
        logger.info("Run training")

        epochs = kwargs["epochs"]

        print("Model summary:")
        torchinfo.summary(self.agent.dqn)

        for epoch in range(epochs):
            t0 = time.perf_counter()
            logger.info("Epoch %d/%d", epoch + 1, epochs)

            self.agent.train()
            train_rewards = self._run_episode(self.train_env)

            self.agent.eval()
            test_rewards = self._run_episode(self.test_env)

            logger.info(
                "Train rewards: %.4f, test rewards: %.4f, duration: %s",
                train_rewards.sum(),
                test_rewards.sum(),
                timedelta(seconds=time.perf_counter() - t0),
            )

            self.agent.dqn.save_weights()
