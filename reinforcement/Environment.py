import torch
from torch import Tensor, FloatTensor

from reinforcement.exceptions import EnvironmentException


class Environment:
    """The environment contains all the elements that the agent interacts with by actions and rewards."""

    def __init__(self, states: Tensor, rewards: Tensor, btc_index: int) -> None:
        self.action_vec = (0, 1, 2)  # HOLD, BUY, SELL

        self.states = states
        self.rewards = rewards
        self.btc_index = btc_index

        self.reset()

    @property
    def n_states(self) -> int:
        return self.states.shape[0]

    @property
    def n_pairs(self) -> int:
        return self.states.shape[1]

    @property
    def n_actions(self) -> int:
        return len(self.action_vec)

    @property
    def is_done(self) -> bool:
        return self.ptr + 1 >= self.n_states

    def reset(self) -> Tensor:
        """Reset environment and return with the first state."""
        self.ptr = 0
        self.pos = torch.zeros(self.n_pairs, dtype=torch.bool)

        return self.states[self.ptr]

    def step(self, actions: Tensor) -> tuple:
        """One step in the environment."""
        if self.is_done:
            raise EnvironmentException("no more steps left")

        self.ptr += 1

        for i, a in enumerate(actions):
            if not self.pos[i] and a == 1:
                self.pos[i] = True

            if self.pos[i] and a == 2:
                self.pos[i] = False

        next_states = self.states[self.ptr]
        rewards = self.rewards[self.ptr] * self.pos
        done = self.is_done

        return next_states, rewards, done
