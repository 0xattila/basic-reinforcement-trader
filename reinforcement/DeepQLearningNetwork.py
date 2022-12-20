import logging

import torch
import torch.nn.functional as F
from torch import FloatTensor
from torch.nn import Conv1d, Linear, Tanh, Sequential, Module


logger = logging.getLogger(__name__)


class DeepQLearningNetwork(Module):
    FILEPATH = "storage/dqn.pt"

    def __init__(self, n_pair: int, n_output: int, n_cnn: int, cnn_delta: int) -> None:
        super().__init__()

        cnn = []

        in_ch = n_pair
        out_ch = in_ch + cnn_delta
        stride = 2

        for i in range(n_cnn):
            cnn.append(Conv1d(in_ch, out_ch, kernel_size=3, stride=stride))
            cnn.append(Tanh())

            if i + 1 >= n_cnn:
                continue

            in_ch = out_ch
            out_ch += cnn_delta

        self.cnn = Sequential(*cnn)

        latest_cnn_weights = cnn[-2].weight.data
        fc_in = latest_cnn_weights.shape[0] * (latest_cnn_weights.shape[2] // stride)

        self.fc = Sequential(
            Linear(fc_in, 16),
            Tanh(),
            Linear(16, n_output),
        )

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.cnn(x)

        x = x.flatten(-2)  # type: ignore
        x = F.dropout(x, 0.05)  # type: ignore
        x = self.fc(x)

        return x

    def save_weights(self) -> None:
        logger.info("Save weights to %s", self.FILEPATH)
        torch.save(self.state_dict(), self.FILEPATH)

    def load_weights(self) -> None:
        logger.info("Load weights from %s", self.FILEPATH)
        state_dict = torch.load(self.FILEPATH)
        self.load_state_dict(state_dict)
