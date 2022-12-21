import logging
import time
from datetime import timedelta

import torch
import torchinfo

from reinforcement import plot
from reinforcement.operators import Operator

logger = logging.getLogger(__name__)


class TrainOperator(Operator):
    def run(self, **kwargs) -> None:
        logger.info("Run training")

        epochs = kwargs["epochs"]

        print("Model summary:")
        torchinfo.summary(self.agent.dqn)

        train_rewards = torch.zeros(
            (epochs, self.train_env.n_states - 1, self.train_env.n_pairs),
            dtype=torch.float32,
        )
        test_rewards = torch.zeros(
            (epochs, self.test_env.n_states - 1, self.test_env.n_pairs),
            dtype=torch.float32,
        )

        for epoch in range(epochs):
            t0 = time.perf_counter()
            logger.info("Epoch %d/%d", epoch + 1, epochs)

            self.agent.train()
            train_rewards[epoch] = self._run_episode(self.train_env)

            self.agent.eval()
            test_rewards[epoch] = self._run_episode(self.test_env)

            logger.info(
                "Train rewards: %.4f, test rewards: %.4f, duration: %s",
                train_rewards[epoch].sum(),
                test_rewards[epoch].sum(),
                timedelta(seconds=time.perf_counter() - t0),
            )

            self.agent.dqn.save_weights()

        plot.plot_train_epochs(train_rewards.numpy(), test_rewards.numpy())
