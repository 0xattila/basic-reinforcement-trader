import logging

import numpy as np

import torch
from torch import FloatTensor
from torch.nn import Linear, Tanh, Sequential, Module


logger = logging.getLogger(__name__)


class DeepQLearningNetwork(Module):
    """Neural net for reinforcement learning."""

    FILEPATH = "storage/dqn.pt"

    def __init__(self, n_inputs: int, n_outputs: int, n_layers: int) -> None:
        super().__init__()

        dims = np.linspace(n_inputs, n_outputs, n_layers + 1).astype(np.int64)

        fc = []

        for i in range(n_layers):
            fc.append(Linear(dims[i], dims[i + 1]))
            fc.append(Tanh())

        self.fc = Sequential(*fc)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = x.flatten(-2)  # type: ignore
        x = self.fc(x)

        return x

    def save_weights(self) -> None:
        logger.info("Save weights to %s", self.FILEPATH)
        torch.save(self.state_dict(), self.FILEPATH)

    def load_weights(self) -> None:
        logger.info("Load weights from %s", self.FILEPATH)
        state_dict = torch.load(self.FILEPATH)
        self.load_state_dict(state_dict)
