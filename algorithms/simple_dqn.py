from collections import deque
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baseline_models import BaseLearningModel

__all__ = ["Network", "DQN"]


class Network(nn.Module):
    def __init__(self, in_size: int, out_size: int, num_hidden: int, widths: Sequence[int]):
        super().__init__()
        assert len(widths) == (num_hidden + 1), "DQN widths and number of layers mismatch!"
        self.input_layer = nn.Linear(in_size, widths[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(widths[idx], widths[idx + 1]) for idx in range(num_hidden)
        )
        self.out_layer = nn.Linear(widths[-1], out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        return self.out_layer(x)


class DQN(BaseLearningModel):
    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        device: str = "cpu",
        eps_init: float = 0.99,
        eps_decay: float = 0.998,
        buffer_size: int = 256,
        batch_size: int = 16,
        lr: float = 0.003,
        num_epochs: int = 1,
        num_hidden: int = 2,
        widths: Sequence[int] = (32, 64, 32),
    ):
        super().__init__()
        self.device = device
        self.action_space_size = action_space_size
        self.epsilon = eps_init
        self.eps_decay = eps_decay
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.q_network = Network(state_size, action_space_size, num_hidden, widths).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.loss = list()

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        self.last_state = state
        self.last_action = action
        return action

    def push(self, reward: float) -> None:
        # All interactions are single-step, so we only store the last state, action, and reward
        self.memory.append((self.last_state, self.last_action, reward))
        del self.last_state, self.last_action

    def learn(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        step_loss = list()
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards = zip(*batch)
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

            current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)
            target_q_values = rewards_tensor

            loss = self.loss_fn(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            step_loss.append(loss.item())
        self.loss.append(sum(step_loss) / len(step_loss))
        self.decay_epsilon()

    def decay_epsilon(self) -> None:
        self.epsilon *= self.eps_decay
