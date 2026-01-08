"""
Simplified QMIX implementation for *single-step* episodes with optional parameter sharing.

The agent networks produce per-agent Q-values, while the mixing network combines
active agent values into a global Q_total conditioned on the global state.

This implementation intentionally targets the "one joint transition per episode" use case:
- Each stored sample contains a full joint action (one action per active agent).
- The learning target is a Monte Carlo team return for that joint action (no bootstrapping).

If you need temporal credit assignment across multiple environment steps, you'll want a
sequence-based QMIX variant (RNN agent nets + TD targets).
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
    """Agent-level Q-network used by machine agents."""
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
    """Single-step QMIX learner with optional parameter sharing and centralized mixing.

    Key ideas:
    - Each agent has a local Q-network Q_i(o_i, a_i).
    - A monotonic mixing network produces a joint action-value:
        Q_tot(s, u) = MIX_s(Q_1, ..., Q_n)
      where MIX_s enforces dQ_tot/dQ_i >= 0 via positive mixing weights.
    - With monotonicity, the greedy joint action can be obtained by greedy per-agent actions.

    Notes for this repo:
    - We store exactly one joint transition per episode and fit Q_tot to the observed team return.
    - `active_mask` is used to zero-out inactive agents (e.g., humans) so the fixed-size mixer can
      handle dynamic populations.
    """
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
        share_parameters: bool = False,
    ):
        super().__init__()
        self.device = device
        self.action_space_size = action_space_size
        self.num_agents = num_agents
        self.global_state_size = global_state_size
        self.share_parameters = share_parameters
        self.epsilon = eps_init
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm

        # Per-agent Q-network(s) and centralized mixing network.
        if self.share_parameters:
            self.agent_net = AgentNetwork(state_size, action_space_size, num_hidden, widths).to(self.device)
            self.agent_nets = None
        else:
            self.agent_net = None
            self.agent_nets = nn.ModuleList(
                [
                    AgentNetwork(state_size, action_space_size, num_hidden, widths).to(self.device)
                    for _ in range(num_agents)
                ]
            )
        self.mixing_net = MixingNetwork(num_agents, global_state_size, mixing_embed_dim, hypernet_embed).to(self.device)

        self.optimizer = optim.Adam(
            list(self._agent_parameters()) + list(self.mixing_net.parameters()),
            lr=lr,
        )
        self.loss = []
        self.memory = deque(maxlen=buffer_size)

    def _agent_parameters(self):
        if self.share_parameters:
            return self.agent_net.parameters()
        return self.agent_nets.parameters()

    def _select_agent_net(self, agent_index: Optional[int]) -> AgentNetwork:
        if self.share_parameters:
            return self.agent_net
        if agent_index is None:
            raise ValueError("agent_index must be provided when share_parameters is False.")
        if agent_index < 0 or agent_index >= self.num_agents:
            raise ValueError(f"agent_index {agent_index} is out of range for {self.num_agents} agents.")
        return self.agent_nets[agent_index]

    def _random_action(self, action_mask: Optional[np.ndarray]) -> int:
        # Uniform sampling with optional action masking support.
        if action_mask is None:
            return int(np.random.randint(self.action_space_size))
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            return int(np.random.randint(self.action_space_size))
        return int(np.random.choice(valid_actions))

    def act(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        agent_index: Optional[int] = None,
    ) -> int:
        # Epsilon-greedy policy over the current agent's Q-values.
        # If an action mask is provided, invalid actions are excluded from sampling and argmax.
        if np.random.rand() < self.epsilon:
            return self._random_action(action_mask)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self._select_agent_net(agent_index)(state_tensor).squeeze(0)

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
        # Store one joint transition (one step) for centralized training.
        #
        # Shapes:
        # - observations:  [num_agents, obs_dim]
        # - actions:       [num_agents]
        # - rewards:       [num_agents] (per-agent rewards from the environment)
        # - active_mask:   [num_agents] (1 for active agents this episode; else 0)
        # - global_state:  [state_dim]  (state vector used by the mixing hypernetworks)
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
        #
        # Target:
        # - This learner uses a Monte Carlo target for the joint action:
        #     y = team_return
        #   so there is no TD bootstrap or target network in this simplified variant.
        if len(self.memory) < self.batch_size:
            return

        step_loss = []
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            # Stack joint data into tensors:
            # - obs:         [B, N, obs_dim]
            # - actions:     [B, N]
            # - rewards:     [B, N]
            # - active_mask: [B, N]
            # - states:      [B, state_dim]
            obs = torch.tensor(np.stack([item["obs"] for item in batch]), device=self.device)
            actions = torch.tensor(np.stack([item["actions"] for item in batch]), device=self.device)
            rewards = torch.tensor(np.stack([item["rewards"] for item in batch]), device=self.device)
            active_mask = torch.tensor(np.stack([item["active_mask"] for item in batch]), device=self.device)
            states = torch.tensor(np.stack([item["state"] for item in batch]), device=self.device)

            batch_size, num_agents, obs_dim = obs.shape
            # Compute per-agent Q-values with shared or per-agent networks.
            if self.share_parameters:
                q_values = self.agent_net(obs.view(-1, obs_dim)).view(
                    batch_size, num_agents, self.action_space_size
                )
            else:
                if num_agents != self.num_agents:
                    raise ValueError(
                        f"Observed {num_agents} agents, expected {self.num_agents} for QMIX."
                    )
                q_values = torch.stack(
                    [self.agent_nets[idx](obs[:, idx, :]) for idx in range(num_agents)],
                    dim=1,
                )

            # Ignore actions for inactive agents to avoid invalid indexing in gather().
            safe_actions = actions.clone()
            safe_actions[active_mask == 0] = 0
            chosen_q = torch.gather(q_values, 2, safe_actions.unsqueeze(-1)).squeeze(-1)
            # Zero-out inactive agent contributions in the joint value.
            chosen_q = chosen_q * active_mask

            # Mix active agent Q-values into a global joint action-value Q_tot(s, u).
            q_tot = self.mixing_net(chosen_q, states)
            # Team return:
            # - Use the mean reward across active agents to keep target scale stable when the
            #   number of active agents changes (dynamic populations).
            # - This differs from the sum only by a per-episode scalar factor when the number
            #   of active agents is fixed.
            active_counts = active_mask.sum(dim=1).clamp(min=1.0)
            team_rewards = (rewards * active_mask).sum(dim=1) / active_counts

            # Single-step Monte Carlo target: no bootstrapping.
            loss = F.mse_loss(q_tot, team_rewards)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self._agent_parameters()) + list(self.mixing_net.parameters()),
                max_norm=self.max_grad_norm,
            )
            self.optimizer.step()
            step_loss.append(loss.item())

        self.loss.append(sum(step_loss) / len(step_loss))
        self.decay_epsilon()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def set_eval_mode(self) -> None:
        if self.share_parameters:
            self.agent_net.eval()
        else:
            self.agent_nets.eval()
        self.mixing_net.eval()

    def set_train_mode(self) -> None:
        if self.share_parameters:
            self.agent_net.train()
        else:
            self.agent_nets.train()
        self.mixing_net.train()
