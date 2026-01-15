"""
QMIX (Value Decomposition Networks) implementation.

This follows the standard QMIX setup:
- Per-agent action-value functions `Q_i(o_i, a_i)` (here: GRU-based to support partial observability).
- A monotonic mixing network `Q_tot(s, Q_1..Q_n)` trained with a TD loss.
- Target networks (agent + mixer) for stable TD targets.
- Optional Double Q-learning for target action selection.

Practical notes:
- Single-step tasks are handled as length-1 sequences (no special-casing).
- Multi-step tasks require calling `reset_episode()` on environment reset so GRU state is cleared.

API notes (matches existing OpenURB scripts):
- `act(obs, action_mask=None, agent_index=...)`
- `store_episode(...)` for single-step episodes (OpenURB default)
- `store_transition(..., done=...)` for multi-step rollouts
- `learn()`, `set_eval_mode()`, `set_train_mode()`
"""

from __future__ import annotations

from collections import deque
import copy
import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from baseline_models import BaseLearningModel

__all__ = ["AgentRNN", "MixingNetwork", "QMIX"]


class AgentRNN(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rnn_hidden_dim: int,
        num_hidden: int,
        widths: Sequence[int],
    ):
        super().__init__()
        assert len(widths) == (num_hidden + 1), "QMIX widths and number of layers mismatch!"
        self.input_layer = nn.Linear(obs_dim, widths[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(widths[idx], widths[idx + 1]) for idx in range(num_hidden)
        )
        self.rnn = nn.GRU(input_size=widths[-1], hidden_size=rnn_hidden_dim, batch_first=True)
        self.out = nn.Linear(rnn_hidden_dim, action_dim)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return x

    def forward(self, obs_seq: torch.Tensor, h0: Optional[torch.Tensor] = None):
        # obs_seq: [B, T, obs_dim], h0: [1, B, rnn_hidden_dim]
        b, t, d = obs_seq.shape
        x = self._encode(obs_seq.reshape(b * t, d)).reshape(b, t, -1)
        out, hn = self.rnn(x, h0)
        q = self.out(out)  # [B, T, A]
        return q, hn


class MixingNetwork(nn.Module):
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        mixing_embed_dim: int,
        hypernet_embed: int,
        weight_clip: Optional[float] = None,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        self.weight_clip = float(weight_clip) if weight_clip is not None else None

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
        # agent_qs: [B, N], states: [B, state_dim] -> [B]
        batch_size = agent_qs.shape[0]
        w1 = F.softplus(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        if self.weight_clip is not None:
            w1 = torch.clamp(w1, max=self.weight_clip)
        w1 = w1.view(batch_size, self.num_agents, self.mixing_embed_dim)
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)

        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.elu(hidden)

        w2 = F.softplus(self.hyper_w2(states))
        if self.weight_clip is not None:
            w2 = torch.clamp(w2, max=self.weight_clip)
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size)


class QMIX(BaseLearningModel):
    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        num_agents: int,
        global_state_size: int,
        device: str = "cpu",
        eps_init: float = 1.0,
        eps_decay: float = 0.999,
        eps_min: float = 0.05,
        buffer_size: int = 2048,
        batch_size: int = 32,
        lr: float = 3e-4,
        num_epochs: int = 1,
        num_hidden: int = 2,
        widths: Sequence[int] = (128, 128, 128),
        rnn_hidden_dim: int = 64,
        mixing_embed_dim: int = 32,
        hypernet_embed: int = 64,
        max_grad_norm: float = 10.0,
        gamma: float = 0.99,
        target_update_every: int = 200,
        double_q: bool = True,
        tau: float = 1.0,
        share_parameters: bool = True,
        mixing_weight_clip: Optional[float] = None,
        q_tot_clip: Optional[float] = None,
        use_huber_loss: bool = True,
    ):
        super().__init__()
        self.device = device
        self.obs_size = int(state_size)
        self.action_space_size = int(action_space_size)
        self.num_agents = int(num_agents)
        self.global_state_size = int(global_state_size)
        self.share_parameters = bool(share_parameters)
        self.epsilon = float(eps_init)
        self.eps_decay = float(eps_decay)
        self.eps_min = float(eps_min)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.max_grad_norm = float(max_grad_norm)
        self.gamma = float(gamma)
        self.target_update_every = max(1, int(target_update_every))
        self.double_q = bool(double_q)
        self.tau = float(tau)
        self.q_tot_clip = float(q_tot_clip) if q_tot_clip is not None else None
        self.use_huber_loss = bool(use_huber_loss)
        self._learn_steps = 0

        if self.share_parameters:
            self.agent_net = AgentRNN(
                self.obs_size, self.action_space_size, rnn_hidden_dim, num_hidden, widths
            ).to(self.device)
            self.target_agent_net = copy.deepcopy(self.agent_net).to(self.device)
            self.agent_nets = None
            self.target_agent_nets = None
        else:
            self.agent_net = None
            self.target_agent_net = None
            self.agent_nets = nn.ModuleList(
                [
                    AgentRNN(self.obs_size, self.action_space_size, rnn_hidden_dim, num_hidden, widths).to(self.device)
                    for _ in range(self.num_agents)
                ]
            )
            self.target_agent_nets = copy.deepcopy(self.agent_nets).to(self.device)

        self.mixing_net = MixingNetwork(
            self.num_agents,
            self.global_state_size,
            mixing_embed_dim,
            hypernet_embed,
            weight_clip=mixing_weight_clip,
        ).to(self.device)
        self.target_mixing_net = copy.deepcopy(self.mixing_net).to(self.device)

        self.target_mixing_net.eval()
        if self.share_parameters:
            self.target_agent_net.eval()
        else:
            self.target_agent_nets.eval()

        self.optimizer = optim.Adam(
            list(self._agent_parameters()) + list(self.mixing_net.parameters()),
            lr=lr,
        )

        self.loss = []
        self.memory = deque(maxlen=int(buffer_size))
        self._episode_steps = []

        # Inference-time hidden state (per-agent). Must be reset on env.reset().
        self._inference_hidden = {}

    def _agent_parameters(self):
        if self.share_parameters:
            return self.agent_net.parameters()
        return self.agent_nets.parameters()

    def reset_episode(self) -> None:
        self._inference_hidden = {}

    def _get_h0(self, agent_index: int, hidden_dim: int) -> torch.Tensor:
        h = self._inference_hidden.get(agent_index)
        if h is None:
            h = torch.zeros(1, 1, hidden_dim, device=self.device)
        return h

    def _set_h(self, agent_index: int, h: torch.Tensor) -> None:
        self._inference_hidden[agent_index] = h.detach()

    def _random_action(self, action_mask: Optional[np.ndarray]) -> int:
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
        if agent_index is None:
            agent_index = 0

        obs_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, 1, -1)
        if self.share_parameters:
            hidden_dim = self.agent_net.rnn.hidden_size
            h0 = self._get_h0(agent_index, hidden_dim)
            q_seq, hn = self.agent_net(obs_t, h0)
        else:
            if agent_index < 0 or agent_index >= self.num_agents:
                raise ValueError(f"agent_index {agent_index} is out of range for {self.num_agents} agents.")
            net = self.agent_nets[agent_index]
            hidden_dim = net.rnn.hidden_size
            h0 = self._get_h0(agent_index, hidden_dim)
            q_seq, hn = net(obs_t, h0)
        self._set_h(agent_index, hn)
        q_values = q_seq.squeeze(0).squeeze(0)  # [A]

        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=q_values.device)
            if mask.shape[0] == q_values.shape[0] and mask.any():
                q_values = q_values.masked_fill(~mask, -1e9)
            elif not mask.any():
                return self._random_action(action_mask)

        if np.random.rand() < self.epsilon:
            return self._random_action(action_mask)
        return int(torch.argmax(q_values).item())

    def store_transition(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        active_mask: np.ndarray,
        global_state: np.ndarray,
        next_observations: np.ndarray,
        next_active_mask: np.ndarray,
        next_global_state: np.ndarray,
        done: bool,
        action_masks: Optional[np.ndarray] = None,
        next_action_masks: Optional[np.ndarray] = None,
    ) -> None:
        self._episode_steps.append(
            {
                "obs": np.asarray(observations, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.int64),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "active_mask": np.asarray(active_mask, dtype=np.float32),
                "state": np.asarray(global_state, dtype=np.float32),
                "next_obs": np.asarray(next_observations, dtype=np.float32),
                "next_active_mask": np.asarray(next_active_mask, dtype=np.float32),
                "next_state": np.asarray(next_global_state, dtype=np.float32),
                "done": bool(done),
                "action_masks": None if action_masks is None else np.asarray(action_masks, dtype=np.int8),
                "next_action_masks": None
                if next_action_masks is None
                else np.asarray(next_action_masks, dtype=np.int8),
            }
        )

        if done:
            episode = self._finalize_episode(self._episode_steps)
            self.memory.append(episode)
            self._episode_steps = []

    def store_episode(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        active_mask: np.ndarray,
        global_state: np.ndarray,
    ) -> None:
        zeros_obs = np.zeros_like(np.asarray(observations, dtype=np.float32))
        zeros_state = np.zeros_like(np.asarray(global_state, dtype=np.float32))
        self.store_transition(
            observations=observations,
            actions=actions,
            rewards=rewards,
            active_mask=active_mask,
            global_state=global_state,
            next_observations=zeros_obs,
            next_active_mask=active_mask,
            next_global_state=zeros_state,
            done=True,
        )

    def _finalize_episode(self, steps: list[dict]) -> dict:
        obs = np.stack([s["obs"] for s in steps], axis=0)
        actions = np.stack([s["actions"] for s in steps], axis=0)
        rewards = np.stack([s["rewards"] for s in steps], axis=0)
        active_mask = np.stack([s["active_mask"] for s in steps], axis=0)
        state = np.stack([s["state"] for s in steps], axis=0)
        next_obs = np.stack([s["next_obs"] for s in steps], axis=0)
        next_active_mask = np.stack([s["next_active_mask"] for s in steps], axis=0)
        next_state = np.stack([s["next_state"] for s in steps], axis=0)
        done = np.asarray([s["done"] for s in steps], dtype=np.float32)

        action_masks = None
        next_action_masks = None
        if all(s.get("action_masks") is not None for s in steps):
            action_masks = np.stack([s["action_masks"] for s in steps], axis=0)
        if all(s.get("next_action_masks") is not None for s in steps):
            next_action_masks = np.stack([s["next_action_masks"] for s in steps], axis=0)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "active_mask": active_mask,
            "state": state,
            "next_obs": next_obs,
            "next_active_mask": next_active_mask,
            "next_state": next_state,
            "done": done,
            "action_masks": action_masks,
            "next_action_masks": next_action_masks,
            "T": obs.shape[0],
        }

    def _masked_argmax(self, q_values: torch.Tensor, action_masks: Optional[torch.Tensor]) -> torch.Tensor:
        # q_values: [B, T, N, A], action_masks: [B, T, N, A] or None -> [B, T, N]
        if action_masks is None:
            return torch.argmax(q_values, dim=-1)
        mask = action_masks.to(dtype=torch.bool, device=q_values.device)
        if mask.shape != q_values.shape:
            return torch.argmax(q_values, dim=-1)
        masked = q_values.masked_fill(~mask, -1e9)
        no_valid = ~mask.any(dim=-1)
        argmax = torch.argmax(masked, dim=-1)
        return torch.where(no_valid, torch.zeros_like(argmax), argmax)

    def _update_targets(self) -> None:
        if self.tau >= 1.0:
            if self.share_parameters:
                self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            else:
                self.target_agent_nets.load_state_dict(self.agent_nets.state_dict())
            self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
            return

        with torch.no_grad():
            if self.share_parameters:
                for target_p, online_p in zip(self.target_agent_net.parameters(), self.agent_net.parameters()):
                    target_p.data.mul_(1.0 - self.tau).add_(self.tau * online_p.data)
            else:
                for target_p, online_p in zip(self.target_agent_nets.parameters(), self.agent_nets.parameters()):
                    target_p.data.mul_(1.0 - self.tau).add_(self.tau * online_p.data)
            for target_p, online_p in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
                target_p.data.mul_(1.0 - self.tau).add_(self.tau * online_p.data)

    def _agent_q_values(self, obs: torch.Tensor, net, share: bool) -> torch.Tensor:
        # obs: [B, T, N, obs_dim] -> [B, T, N, A]
        b, t, n, d = obs.shape
        if share:
            obs_bn = obs.permute(0, 2, 1, 3).reshape(b * n, t, d)
            q_bn, _ = net(obs_bn, None)
            q = q_bn.reshape(b, n, t, -1).permute(0, 2, 1, 3)
            return q
        qs = []
        for idx in range(n):
            q_i, _ = net[idx](obs[:, :, idx, :], None)
            qs.append(q_i.unsqueeze(2))
        return torch.cat(qs, dim=2)

    def learn(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        step_losses = []
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            max_t = max(int(ep["T"]) for ep in batch)

            def pad_time(x, pad_value=0.0):
                t = x.shape[0]
                if t == max_t:
                    return x
                pad_shape = (max_t - t,) + x.shape[1:]
                pad = np.full(pad_shape, pad_value, dtype=x.dtype)
                return np.concatenate([x, pad], axis=0)

            obs = torch.tensor(np.stack([pad_time(ep["obs"]) for ep in batch]), device=self.device)
            actions = torch.tensor(np.stack([pad_time(ep["actions"]) for ep in batch]), device=self.device)
            rewards = torch.tensor(np.stack([pad_time(ep["rewards"]) for ep in batch]), device=self.device)
            active_mask = torch.tensor(np.stack([pad_time(ep["active_mask"]) for ep in batch]), device=self.device)
            states = torch.tensor(np.stack([pad_time(ep["state"]) for ep in batch]), device=self.device)

            next_obs = torch.tensor(np.stack([pad_time(ep["next_obs"]) for ep in batch]), device=self.device)
            next_active_mask = torch.tensor(
                np.stack([pad_time(ep["next_active_mask"]) for ep in batch]), device=self.device
            )
            next_states = torch.tensor(np.stack([pad_time(ep["next_state"]) for ep in batch]), device=self.device)
            dones = torch.tensor(
                np.stack([pad_time(ep["done"].reshape(-1, 1)) for ep in batch]),
                device=self.device,
                dtype=torch.float32,
            ).squeeze(-1)

            lengths = torch.tensor([int(ep["T"]) for ep in batch], device=self.device, dtype=torch.int64)
            time_mask = (
                torch.arange(max_t, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            ).to(dtype=torch.float32)

            action_masks = None
            next_action_masks = None
            if all(ep.get("action_masks") is not None for ep in batch):
                action_masks = torch.tensor(np.stack([pad_time(ep["action_masks"]) for ep in batch]), device=self.device)
            if all(ep.get("next_action_masks") is not None for ep in batch):
                next_action_masks = torch.tensor(
                    np.stack([pad_time(ep["next_action_masks"]) for ep in batch]), device=self.device
                )

            if self.share_parameters:
                q_all = self._agent_q_values(obs, self.agent_net, share=True)
                next_q_online = self._agent_q_values(next_obs, self.agent_net, share=True)
                with torch.no_grad():
                    next_q_target = self._agent_q_values(next_obs, self.target_agent_net, share=True)
            else:
                q_all = self._agent_q_values(obs, self.agent_nets, share=False)
                next_q_online = self._agent_q_values(next_obs, self.agent_nets, share=False)
                with torch.no_grad():
                    next_q_target = self._agent_q_values(next_obs, self.target_agent_nets, share=False)

            safe_actions = actions.clone()
            safe_actions[active_mask == 0] = 0
            chosen_q = torch.gather(q_all, 3, safe_actions.unsqueeze(-1)).squeeze(-1)
            chosen_q = chosen_q * active_mask

            b, t, n = chosen_q.shape
            q_tot = self.mixing_net(chosen_q.reshape(b * t, n), states.reshape(b * t, -1)).reshape(b, t)
            if self.q_tot_clip is not None:
                q_tot = torch.clamp(q_tot, -self.q_tot_clip, self.q_tot_clip)

            active_counts = active_mask.sum(dim=2).clamp(min=1.0)
            team_rewards = (rewards * active_mask).sum(dim=2) / active_counts

            with torch.no_grad():
                if self.double_q:
                    next_actions = self._masked_argmax(next_q_online, next_action_masks)
                else:
                    next_actions = self._masked_argmax(next_q_target, next_action_masks)
                safe_next_actions = next_actions.clone()
                safe_next_actions[next_active_mask == 0] = 0
                next_chosen_q = torch.gather(next_q_target, 3, safe_next_actions.unsqueeze(-1)).squeeze(-1)
                next_chosen_q = next_chosen_q * next_active_mask

                q_tot_next = self.target_mixing_net(
                    next_chosen_q.reshape(b * t, n), next_states.reshape(b * t, -1)
                ).reshape(b, t)
                if self.q_tot_clip is not None:
                    q_tot_next = torch.clamp(q_tot_next, -self.q_tot_clip, self.q_tot_clip)

                targets = team_rewards + (1.0 - dones) * self.gamma * q_tot_next
                if self.q_tot_clip is not None:
                    targets = torch.clamp(targets, -self.q_tot_clip, self.q_tot_clip)

            if self.use_huber_loss:
                td = F.smooth_l1_loss(q_tot, targets, reduction="none")
            else:
                td = F.mse_loss(q_tot, targets, reduction="none")
            loss = (td * time_mask).sum() / time_mask.sum().clamp(min=1.0)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self._agent_parameters()) + list(self.mixing_net.parameters()),
                max_norm=self.max_grad_norm,
            )
            self.optimizer.step()

            self._learn_steps += 1
            if self._learn_steps % self.target_update_every == 0:
                self._update_targets()

            step_losses.append(float(loss.item()))

        self.loss.append(float(sum(step_losses) / len(step_losses)))
        self.decay_epsilon()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def set_eval_mode(self) -> None:
        if self.share_parameters:
            self.agent_net.eval()
            self.target_agent_net.eval()
        else:
            self.agent_nets.eval()
            self.target_agent_nets.eval()
        self.mixing_net.eval()
        self.target_mixing_net.eval()

    def set_train_mode(self) -> None:
        if self.share_parameters:
            self.agent_net.train()
        else:
            self.agent_nets.train()
        self.mixing_net.train()
