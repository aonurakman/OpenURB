"""
Independent Q-Learning (IQL) implementation.

This is a classic (Double) DQN with:
- A value network and a target network.
- TD targets with optional Double DQN target action selection.
- Replay buffer + gradient clipping.

Usage:
- Single-step tasks (OpenURB): call `act(state)` then `push(reward)` (default `done=True`).
- Multi-step tasks: call `push_transition(state, action, reward, next_state, done)` each step.
"""

from collections import deque
import random
from typing import Optional, Sequence

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
    """Classic (Double) DQN with target network."""

    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        device: str = "cpu",
        eps_init: float = 0.99,
        eps_decay: float = 0.998,
        eps_min: float = 0.05,
        buffer_size: int = 256,
        batch_size: int = 16,
        lr: float = 0.003,
        num_epochs: int = 1,
        num_hidden: int = 2,
        widths: Sequence[int] = (32, 64, 32),
        gamma: float = 0.99,
        target_update_every: int = 100,
        double_dqn: bool = True,
        tau: float = 1.0,
        max_grad_norm: float = 10.0,
    ):
        super().__init__()
        self.device = device
        self.action_space_size = int(action_space_size)
        self.epsilon = float(eps_init)
        self.eps_decay = float(eps_decay)
        self.eps_min = float(eps_min)
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.gamma = float(gamma)
        self.target_update_every = max(1, int(target_update_every))
        self.double_dqn = bool(double_dqn)
        self.tau = float(tau)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
        self._learn_steps = 0
        self._state_size = int(state_size)

        self.value_network = Network(state_size, self.action_space_size, num_hidden, widths).to(self.device)
        self.target_network = Network(state_size, self.action_space_size, num_hidden, widths).to(self.device)
        self.target_network.load_state_dict(self.value_network.state_dict())
        self.target_network.eval()

        # Backwards-compatible alias.
        self.q_network = self.value_network

        self.optimizer = optim.Adam(self.value_network.parameters(), lr=float(lr))
        self.loss_fn = nn.SmoothL1Loss()
        self.loss = []

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            action = int(np.random.choice(self.action_space_size))
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.value_network(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        self.last_state = np.asarray(state, dtype=np.float32)
        self.last_action = action
        return action

    def push(self, reward: float, next_state: Optional[np.ndarray] = None, done: bool = True) -> None:
        state = getattr(self, "last_state", None)
        action = getattr(self, "last_action", None)
        if state is None or action is None:
            raise RuntimeError("push() called before act(); use push_transition(...) instead.")

        if (next_state is None) and (not done):
            raise ValueError("next_state is required when done=False.")
        if next_state is None:
            next_state = np.zeros(self._state_size, dtype=np.float32)

        self.push_transition(state, action, reward, next_state, done)
        del self.last_state, self.last_action

    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((np.asarray(state), int(action), float(reward), np.asarray(next_state), bool(done)))

    def _update_target_network(self) -> None:
        if self.tau >= 1.0:
            self.target_network.load_state_dict(self.value_network.state_dict())
            return
        with torch.no_grad():
            for target_param, value_param in zip(self.target_network.parameters(), self.value_network.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * value_param.data)

    def learn(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        step_losses = []
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
            actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
            rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states_tensor = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
            dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

            current_q_values = self.value_network(states_tensor).gather(1, actions_tensor)
            with torch.no_grad():
                if self.double_dqn:
                    next_actions = torch.argmax(self.value_network(next_states_tensor), dim=1, keepdim=True)
                    next_q_values = self.target_network(next_states_tensor).gather(1, next_actions)
                else:
                    next_q_values = self.target_network(next_states_tensor).max(dim=1, keepdim=True).values
                target_q_values = rewards_tensor + (1.0 - dones_tensor) * self.gamma * next_q_values

            loss = self.loss_fn(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            self._learn_steps += 1
            if self._learn_steps % self.target_update_every == 0:
                self._update_target_network()

            step_losses.append(float(loss.item()))

        self.loss.append(float(sum(step_losses) / len(step_losses)))
        self.decay_epsilon()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
