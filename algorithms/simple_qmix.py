"""
Simplified QMIX implementation for single-step episodes with parameter sharing.

The agent network produces per-agent Q-values, while the mixing network combines
active agent values into a global Q_total conditioned on the global state.
"""

from collections import deque
import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from baseline_models import BaseLearningModel

__all__ = ["AgentNetwork", "MixingNetwork", "QMIX"]


class AgentNetwork(nn.Module):
    """Shared agent-level Q-network used by all machine agents."""
    def __init__(self, in_size: int, out_size: int, num_hidden: int, widths: Sequence[int]):
        super().__init__()
        assert len(widths) == (num_hidden + 1), "QMIX widths and number of layers mismatch!"
        self.input_layer = nn.Linear(in_size, widths[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(widths[idx], widths[idx + 1]) for idx in range(num_hidden)
        )
        self.out_layer = nn.Linear(widths[-1], out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Maps a single agent observation to Q-values for all actions.
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        return self.out_layer(x)


class MixingNetwork(nn.Module):
    """QMIX mixing network with hypernetworks enforcing monotonicity."""
    def __init__(self, num_agents: int, state_dim: int, mixing_embed_dim: int, hypernet_embed: int):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, num_agents * mixing_embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, mixing_embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        # agent_qs: [batch, num_agents], states: [batch, state_dim]
        batch_size = agent_qs.shape[0]
        # Hypernetworks enforce monotonic mixing through positive weights.
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(batch_size, self.num_agents, self.mixing_embed_dim)
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)

        # First mixing layer combines per-agent Qs into a hidden embedding.
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.elu(hidden)

        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)

        # Second mixing layer produces a scalar joint Q-value.
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size)


class QMIX(BaseLearningModel):
    """Single-step QMIX learner with shared agent network and centralized mixing."""
    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        num_agents: int,
        global_state_size: int,
        device: str = "cpu",
        eps_init: float = 0.99,
        eps_decay: float = 0.998,
        eps_min: float = 0.05,
        buffer_size: int = 256,
        batch_size: int = 16,
        lr: float = 0.0005,
        num_epochs: int = 1,
        num_hidden: int = 2,
        widths: Sequence[int] = (64, 64, 64),
        mixing_embed_dim: int = 32,
        hypernet_embed: int = 64,
        max_grad_norm: float = 10.0,
    ):
        super().__init__()
        self.device = device
        self.action_space_size = action_space_size
        self.num_agents = num_agents
        self.global_state_size = global_state_size
        self.epsilon = eps_init
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm

        # Shared per-agent Q-network and centralized mixing network.
        self.agent_net = AgentNetwork(state_size, action_space_size, num_hidden, widths).to(self.device)
        self.mixing_net = MixingNetwork(num_agents, global_state_size, mixing_embed_dim, hypernet_embed).to(self.device)

        self.optimizer = optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixing_net.parameters()),
            lr=lr,
        )
        self.loss = []
        self.memory = deque(maxlen=buffer_size)

    def _random_action(self, action_mask: Optional[np.ndarray]) -> int:
        # Uniform sampling with optional action masking support.
        if action_mask is None:
            return int(np.random.randint(self.action_space_size))
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            return int(np.random.randint(self.action_space_size))
        return int(np.random.choice(valid_actions))

    def act(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        # Epsilon-greedy policy over agent Q-values, with masking if available.
        if np.random.rand() < self.epsilon:
            return self._random_action(action_mask)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.agent_net(state_tensor).squeeze(0)

        if action_mask is not None:
            # Mask invalid actions by assigning a large negative value.
            mask = torch.tensor(action_mask, dtype=torch.bool, device=q_values.device)
            if mask.shape[0] == q_values.shape[0] and mask.any():
                q_values = q_values.masked_fill(~mask, -1e9)
            elif not mask.any():
                return self._random_action(action_mask)

        return int(torch.argmax(q_values).item())

    def store_episode(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        active_mask: np.ndarray,
        global_state: np.ndarray,
    ) -> None:
        # Store single-step joint transition for centralized training.
        self.memory.append(
            {
                "obs": np.asarray(observations, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.int64),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "active_mask": np.asarray(active_mask, dtype=np.float32),
                "state": np.asarray(global_state, dtype=np.float32),
            }
        )

    def learn(self) -> None:
        # Learn from a batch of single-step joint transitions.
        if len(self.memory) < self.batch_size:
            return

        step_loss = []
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            # Stack joint data into tensors: [batch, num_agents, ...]
            obs = torch.tensor(np.stack([item["obs"] for item in batch]), device=self.device)
            actions = torch.tensor(np.stack([item["actions"] for item in batch]), device=self.device)
            rewards = torch.tensor(np.stack([item["rewards"] for item in batch]), device=self.device)
            active_mask = torch.tensor(np.stack([item["active_mask"] for item in batch]), device=self.device)
            states = torch.tensor(np.stack([item["state"] for item in batch]), device=self.device)

            batch_size, num_agents, obs_dim = obs.shape
            # Compute per-agent Q-values with the shared agent network.
            q_values = self.agent_net(obs.view(-1, obs_dim)).view(
                batch_size, num_agents, self.action_space_size
            )

            # Ignore actions for inactive agents to avoid invalid indexing.
            safe_actions = actions.clone()
            safe_actions[active_mask == 0] = 0
            chosen_q = torch.gather(q_values, 2, safe_actions.unsqueeze(-1)).squeeze(-1)
            # Zero-out inactive agent contributions in the joint value.
            chosen_q = chosen_q * active_mask

            # Mix active agent Q-values into a global joint Q_total.
            q_tot = self.mixing_net(chosen_q, states)
            # Team reward is the sum of rewards for active agents.
            team_rewards = (rewards * active_mask).sum(dim=1)

            # Single-step target: no bootstrapping in this environment.
            loss = F.mse_loss(q_tot, team_rewards)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.agent_net.parameters()) + list(self.mixing_net.parameters()),
                max_norm=self.max_grad_norm,
            )
            self.optimizer.step()
            step_loss.append(loss.item())

        self.loss.append(sum(step_loss) / len(step_loss))
        self.decay_epsilon()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def set_eval_mode(self) -> None:
        self.agent_net.eval()
        self.mixing_net.eval()

    def set_train_mode(self) -> None:
        self.agent_net.train()
        self.mixing_net.train()
