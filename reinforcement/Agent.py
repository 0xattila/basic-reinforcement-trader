import numpy as np

import torch
from torch import FloatTensor, Tensor
from torch.nn import MSELoss
from torch.optim import Adam

from reinforcement import DeepQLearningNetwork, ReplayBuffer
from reinforcement.device import device


class Agent:
    """The agent learns to interact with the enviroment, and gives back the best actions for the given state."""

    def __init__(self, n_pairs: int, seq_len: int, n_actions: int) -> None:
        self.gamma = 0.8
        self.epsilon = 0.9
        self.epsilon_decay = 0.01
        self.epsilon_min = 0.1

        self.is_train = True

        self.n_pairs = n_pairs
        self.n_actions = n_actions
        self.n_outputs = n_pairs * n_actions

        self.dqn = DeepQLearningNetwork(
            n_inputs=n_pairs * seq_len, n_outputs=self.n_outputs, n_layers=3
        ).to(device)
        self.optim = Adam(self.dqn.parameters(), lr=1e-5)
        self.loss_fn = MSELoss().to(device)

        self.memory = ReplayBuffer(seq_len, n_pairs)

        default_mask = []
        for i in range(n_actions):
            mask = [False] * n_actions
            mask[i] = True
            default_mask.append(mask)

        self.default_mask = torch.tensor(default_mask, dtype=torch.bool)

    def train(self) -> None:
        """Agent train mode. It includes epsilon's random exploration."""
        self.is_train = True

    def eval(self) -> None:
        """Agent evaluation mode. It excludes epsilon's random exploration."""
        self.is_train = False

    def _act_epsilon(self) -> Tensor:
        """Get random actions to force the agent to explore the environment."""
        return torch.randn(self.n_outputs, dtype=torch.float32)

    def _act_dqn(self, states: Tensor) -> Tensor:
        """Act by the neural net."""
        self.dqn.eval()
        with torch.no_grad():
            states = torch.unsqueeze(states, dim=0)
            return self.dqn(states.to(device)).to("cpu")

    def actions(self, states: Tensor) -> Tensor:
        """Get actions for the given state."""
        if self.is_train and np.random.rand() < self.epsilon:
            probabilities = self._act_epsilon()
        else:
            probabilities = self._act_dqn(states)

        return probabilities.reshape(self.n_pairs, self.n_actions).argmax(dim=1)

    def replay(self, batch_size: int) -> None:
        """Replay steps from the memory and learn."""
        if batch_size > self.memory.size:
            return

        states, next_states, actions, rewards, dones = self.memory.sample_batch(
            batch_size
        )

        self.dqn.eval()
        with torch.no_grad():
            out_shape = (batch_size, self.n_pairs, self.n_actions)
            targets = self.dqn(states).reshape(out_shape)
            next_output = self.dqn(next_states).reshape(out_shape)

        use_next = ~dones.reshape(batch_size, 1)
        subtargets = rewards + use_next * self.gamma * next_output.amax(dim=2)

        mask = self.default_mask[actions]
        targets[mask] = subtargets.flatten()

        self._train(states, targets.reshape(batch_size, -1))

        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def _train(self, states: FloatTensor, target: FloatTensor) -> None:
        """Simply train 1 batch."""
        self.dqn.train()
        self.optim.zero_grad()

        output = self.dqn(states)

        loss = self.loss_fn(output, target)
        loss.backward()

        self.optim.step()
