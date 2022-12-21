from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split

import torch
from torch import Tensor
from torch.nn import Module

from data import KlineManager, Timeframe

from reinforcement import Environment, Agent


class Operator(ABC):
    """The operator is responsibe to manage the interactions between the agent and the environment."""

    def __init__(self, timeframe: Timeframe) -> None:
        seq_len = 10

        self.klines = KlineManager.from_database(timeframe)
        self.klines.add_indicator(
            lambda x: x["close"].pct_change(), name="return", dropna=True
        )

        data = self.klines.get_column("return").to_numpy()
        x_train, x_test, y_train, y_test = self.__build_datasets(data, seq_len)

        self.train_env = Environment(
            states=torch.from_numpy(x_train),
            rewards=torch.from_numpy(y_train),
            btc_index=self.klines.btc_index,
        )
        self.test_env = Environment(
            states=torch.from_numpy(x_test),
            rewards=torch.from_numpy(y_test),
            btc_index=self.klines.btc_index,
        )

        self.agent = Agent(self.train_env.n_pairs, seq_len, self.train_env.n_actions)
        self.agent.dqn.apply(self.__init_weights)

    @staticmethod
    def __build_datasets(data: ndarray, T: int) -> list:
        """Build time series datasets."""
        D = data.shape[0] - T
        X = np.zeros((D, data.shape[1], T), dtype=np.float32)
        Y = np.zeros((D, data.shape[1]), dtype=np.float32)

        for i in range(D):
            X[i] = data[i : i + T].T
            Y[i] = data[i + T]

        return train_test_split(X, Y, test_size=0.2, shuffle=False)

    @staticmethod
    def __init_weights(x: Module) -> None:
        """Init weights for better performance."""
        with torch.no_grad():
            if hasattr(x, "weight"):
                torch.nn.init.xavier_uniform_(x.weight.data)

            if hasattr(x, "bias") and not isinstance(x.bias, bool):
                x.bias.fill_(1e-2)

    def _run_episode(self, env: Environment) -> Tensor:
        """Run 1 eposide until the environment is done."""
        states = env.reset()
        done = False

        all_rewards = torch.empty((0, env.n_pairs), dtype=torch.float32)

        i = 0
        while not done:
            i += 1
            print(f"{i}/{env.n_states}", end="\r")
            actions = self.agent.actions(states)
            next_states, rewards, done = env.step(actions)

            all_rewards = torch.concat((all_rewards, rewards.reshape(1, -1)), dim=0)

            if self.agent.is_train:
                self.agent.memory.store(states, next_states, actions, rewards, done)
                self.agent.replay(batch_size=32)

            states = next_states

        print()

        return all_rewards

    @abstractmethod
    def run(self, **kwargs) -> None:
        pass
