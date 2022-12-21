import numpy as np

import torch
from torch import FloatTensor, Tensor

from reinforcement.device import device
from reinforcement.exceptions import ReplayBufferException


class ReplayBuffer:
    """
    The replay buffer is responsible for storing limited amount of data.
    It avoids memory overflow by reusing indicies with a pointer.
    """

    def __init__(self, series_size: int, n_pair, size: int = 500) -> None:
        self.state_buf = torch.zeros((size, n_pair, series_size), dtype=torch.float32)
        self.next_state_buf = torch.zeros(
            (size, n_pair, series_size), dtype=torch.float32
        )
        self.act_buf = torch.zeros((size, n_pair), dtype=torch.int64)
        self.rew_buf = torch.zeros((size, n_pair), dtype=torch.float32)
        self.done_buf = torch.zeros((size), dtype=torch.bool)

        self.ptr, self.size = 0, 0
        self.max_size = size

    def store(
        self,
        states: Tensor,
        next_states: FloatTensor,
        actions: Tensor,
        rewards: FloatTensor,
        done: Tensor,
    ) -> None:
        """Store data."""
        self.state_buf[self.ptr] = states
        self.next_state_buf[self.ptr] = next_states
        self.act_buf[self.ptr] = actions
        self.rew_buf[self.ptr] = rewards
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> tuple:
        """Get randomly selected data from the replay buffer."""
        if self.size < batch_size:
            raise ReplayBufferException("buffer is not ready")

        idxs = np.random.choice(self.size, batch_size, replace=False)

        return (
            self.state_buf[idxs].to(device),
            self.next_state_buf[idxs].to(device),
            self.act_buf[idxs].to(device),
            self.rew_buf[idxs].to(device),
            self.done_buf[idxs].to(device),
        )
