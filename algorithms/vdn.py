"""
VDN (Value Decomposition Networks) implementation.

VDN (Sunehag et al., 2018) is one of the simplest cooperative MARL value-factorization methods:
- Each agent learns its own action-value function Q_i(o_i, a_i).
- A fixed mixing function combines individual values into a joint team value:
      Q_tot = sum_i Q_i
  (We optionally normalize by the number of *active* agents to keep value scales stable when the
   team size changes over time, which is important for OpenURB's open-environment setting.)
- Centralized training uses a TD loss on Q_tot with target networks.
- Decentralized execution selects actions using each agent's own Q-values.

Design choices (kept consistent with other OpenURB algorithms):
- Recurrent per-agent networks (MLP encoder + GRU) to support partial observability by building a
  learned latent "belief-like" state from observation history.
- Boltzmann (softmax) exploration with a decaying temperature schedule.
- Episode-based replay buffer so recurrent training has temporal context.

API notes (mirrors `algorithms/qmix.py` so scripts can stay almost identical):
- `act(obs, action_mask=None, agent_index=...)`
- `store_episode(...)` for OpenURB's single-step "day" episodes
- `store_transition(..., done=...)` for multi-step rollouts in generic recurrent MARL environments
- `learn()`, `set_eval_mode()`, `set_train_mode()`, `reset_episode()`
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

__all__ = ["AgentRNN", "VDN"]


class AgentRNN(nn.Module):
    """Per-agent recurrent Q-network: MLP encoder + GRU + linear action-value head."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rnn_hidden_dim: int,
        num_hidden: int,
        widths: Sequence[int],
    ):
        super().__init__()
        assert len(widths) == (num_hidden + 1), "VDN widths and number of layers mismatch!"

        self.input_layer = nn.Linear(int(obs_dim), int(widths[0]))
        self.hidden_layers = nn.ModuleList(
            nn.Linear(int(widths[idx]), int(widths[idx + 1])) for idx in range(int(num_hidden))
        )
        self.rnn = nn.GRU(input_size=int(widths[-1]), hidden_size=int(rnn_hidden_dim), batch_first=True)
        self.out = nn.Linear(int(rnn_hidden_dim), int(action_dim))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # Apply feature extraction per timestep before the GRU.
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return x

    def forward(self, obs_seq: torch.Tensor, h0: Optional[torch.Tensor] = None):
        # obs_seq: [B, T, obs_dim], h0: [1, B, rnn_hidden_dim] or None
        b, t, d = obs_seq.shape
        # Flatten time to apply the MLP on all timesteps, then restore to sequence form for the GRU.
        x = self._encode(obs_seq.reshape(b * t, d)).reshape(b, t, -1)
        out, hn = self.rnn(x, h0)
        q = self.out(out)  # [B, T, A]
        return q, hn


class VDN(BaseLearningModel):
    """Value Decomposition Networks (VDN) with recurrent per-agent Q-functions.

    This follows the standard DQN-style centralized training:
    - Online networks produce per-agent Q_i values.
    - A fixed additive mixer computes Q_tot.
    - Target networks provide stable TD targets.
    """

    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        num_agents: int,
        device: str = "cpu",
        temp_init: float = 1.0,
        temp_decay: float = 0.999,
        temp_min: float = 0.05,
        buffer_size: int = 2048,
        batch_size: int = 32,
        lr: float = 3e-4,
        num_epochs: int = 1,
        num_hidden: int = 2,
        widths: Sequence[int] = (128, 128, 128),
        rnn_hidden_dim: int = 64,
        max_grad_norm: float = 10.0,
        gamma: float = 0.99,
        target_update_every: int = 200,
        double_q: bool = True,
        tau: float = 1.0,
        share_parameters: bool = True,
        q_tot_clip: Optional[float] = None,
        use_huber_loss: bool = True,
        normalize_by_active: bool = True,
    ):
        super().__init__()
        self.device = device
        self.obs_size = int(state_size)
        self.action_space_size = int(action_space_size)
        self.num_agents = int(num_agents)

        # Parameter sharing:
        # - If agents are homogeneous, sharing reduces parameters and improves sample efficiency.
        # - If agents are heterogeneous, separate networks may be beneficial.
        self.share_parameters = bool(share_parameters)

        # Boltzmann (softmax) exploration schedule (decentralized execution).
        self.temperature = float(temp_init)
        self.temp_decay = float(temp_decay)
        self.temp_min = float(temp_min)

        # Replay / optimization settings.
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.max_grad_norm = float(max_grad_norm)

        # TD learning hyperparameters.
        self.gamma = float(gamma)
        self.target_update_every = max(1, int(target_update_every))
        self.double_q = bool(double_q)
        self.tau = float(tau)
        self.q_tot_clip = float(q_tot_clip) if q_tot_clip is not None else None
        self.use_huber_loss = bool(use_huber_loss)

        # OpenURB uses an "active mask" because the AV team is variable-sized.
        # `normalize_by_active=True` makes Q_tot an average over active agents, matching the
        # reward aggregation used in `algorithms/qmix.py` and keeping the scale stable when the
        # number of AVs changes.
        self.normalize_by_active = bool(normalize_by_active)

        self._learn_steps = 0

        if self.share_parameters:
            # One shared agent network reused for all agents.
            self.agent_net = AgentRNN(
                self.obs_size,
                self.action_space_size,
                rnn_hidden_dim=int(rnn_hidden_dim),
                num_hidden=int(num_hidden),
                widths=widths,
            ).to(self.device)
            self.target_agent_net = copy.deepcopy(self.agent_net).to(self.device)
            self.agent_nets = None
            self.target_agent_nets = None
            self.target_agent_net.eval()
        else:
            # Separate networks per agent (more expressive, more parameters).
            self.agent_net = None
            self.target_agent_net = None
            self.agent_nets = nn.ModuleList(
                [
                    AgentRNN(
                        self.obs_size,
                        self.action_space_size,
                        rnn_hidden_dim=int(rnn_hidden_dim),
                        num_hidden=int(num_hidden),
                        widths=widths,
                    ).to(self.device)
                    for _ in range(self.num_agents)
                ]
            )
            self.target_agent_nets = copy.deepcopy(self.agent_nets).to(self.device)
            self.target_agent_nets.eval()

        # One optimizer over the trainable online network(s).
        self.optimizer = optim.Adam(list(self._agent_parameters()), lr=float(lr))

        # Replay buffer stores *episodes* (sequences) so RNN training can preserve temporal context.
        self.memory = deque(maxlen=int(buffer_size))
        self._episode_steps = []

        # Track training losses and inference-time hidden state per agent.
        self.loss: list[float] = []
        self._inference_hidden: dict[int, torch.Tensor] = {}

    def _agent_parameters(self):
        if self.share_parameters:
            return self.agent_net.parameters()
        return self.agent_nets.parameters()

    def reset_episode(self) -> None:
        # Call this once after env.reset(). It clears hidden state for all agents.
        self._inference_hidden = {}

    def _get_h0(self, agent_index: int, hidden_dim: int) -> torch.Tensor:
        # If we haven't seen this agent yet in the episode, start from an all-zero hidden state.
        h = self._inference_hidden.get(agent_index)
        if h is None:
            h = torch.zeros(1, 1, int(hidden_dim), device=self.device)
        return h

    def _set_h(self, agent_index: int, h: torch.Tensor) -> None:
        # Detach so inference-time hidden state doesn't keep an autograd graph alive.
        self._inference_hidden[agent_index] = h.detach()

    def _random_action(self, action_mask: Optional[np.ndarray]) -> int:
        # Uniform random action, optionally restricted by an action mask.
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
        # Decentralized execution: each agent selects its action from its own Q-values.
        if agent_index is None:
            agent_index = 0

        # Run the agent's recurrent Q-network for a single timestep.
        # We treat the current observation as a length-1 sequence: [B=1, T=1, obs_dim].
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

        # Update hidden state so the next call to act() can use the accumulated history.
        self._set_h(agent_index, hn)
        q_values = q_seq.squeeze(0).squeeze(0)  # [A]

        if action_mask is not None:
            # Mask invalid actions by setting their Q-values to a very negative number.
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=q_values.device)
            if mask.shape[0] == q_values.shape[0] and mask.any():
                q_values = q_values.masked_fill(~mask, -1e9)
            elif not mask.any():
                return self._random_action(action_mask)

        return self._boltzmann_action(q_values)

    def _boltzmann_action(self, q_values: torch.Tensor) -> int:
        # q_values: [A] (already masked if an action_mask was provided)
        temp = float(self.temperature)
        if temp <= 0.0:
            return int(torch.argmax(q_values).item())
        logits = q_values / temp
        logits = logits - torch.max(logits)
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def _ensure_state(self, state: Optional[np.ndarray]) -> np.ndarray:
        # VDN does not use a centralized global state, but we keep the same replay structure
        # as QMIX for compatibility with existing scripts (and because external tasks may pass it).
        if state is None:
            return np.zeros(1, dtype=np.float32)
        return np.asarray(state, dtype=np.float32)

    def store_transition(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        active_mask: np.ndarray,
        global_state: Optional[np.ndarray],
        next_observations: np.ndarray,
        next_active_mask: np.ndarray,
        next_global_state: Optional[np.ndarray],
        done: bool,
        action_masks: Optional[np.ndarray] = None,
        next_action_masks: Optional[np.ndarray] = None,
    ) -> None:
        # Store one *joint* transition for all agents at a timestep.
        #
        # Shapes (multi-step):
        # - observations:      [N, obs_dim]
        # - actions:           [N]
        # - rewards:           [N]   (per-agent rewards; aggregated to a team reward in learn())
        # - active_mask:       [N]   (1 if agent is active at this step)
        # - global_state:      unused for VDN; stored only for replay compatibility
        self._episode_steps.append(
            {
                "obs": np.asarray(observations, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.int64),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "active_mask": np.asarray(active_mask, dtype=np.float32),
                "state": self._ensure_state(global_state),
                "next_obs": np.asarray(next_observations, dtype=np.float32),
                "next_active_mask": np.asarray(next_active_mask, dtype=np.float32),
                "next_state": self._ensure_state(next_global_state),
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
        global_state: Optional[np.ndarray] = None,
    ) -> None:
        # Convenience wrapper for OpenURB's single-step "day" episodes.
        zeros_obs = np.zeros_like(np.asarray(observations, dtype=np.float32))
        state = self._ensure_state(global_state)
        zeros_state = np.zeros_like(state)
        self.store_transition(
            observations=observations,
            actions=actions,
            rewards=rewards,
            active_mask=active_mask,
            global_state=state,
            next_observations=zeros_obs,
            next_active_mask=active_mask,
            next_global_state=zeros_state,
            done=True,
        )

    def _finalize_episode(self, steps: list[dict]) -> dict:
        # Convert a python list of step dicts into a single dict of stacked arrays.
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
        # Target network update:
        # - Hard update (tau >= 1.0): copy weights exactly.
        # - Soft update (tau in (0,1)): Polyak averaging for smoother changes.
        if self.tau >= 1.0:
            if self.share_parameters:
                self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            else:
                self.target_agent_nets.load_state_dict(self.agent_nets.state_dict())
            return

        with torch.no_grad():
            if self.share_parameters:
                for target_p, online_p in zip(self.target_agent_net.parameters(), self.agent_net.parameters()):
                    target_p.data.mul_(1.0 - self.tau).add_(self.tau * online_p.data)
            else:
                for target_p, online_p in zip(self.target_agent_nets.parameters(), self.agent_nets.parameters()):
                    target_p.data.mul_(1.0 - self.tau).add_(self.tau * online_p.data)

    def _agent_q_values(self, obs: torch.Tensor, net, share: bool) -> torch.Tensor:
        # obs: [B, T, N, obs_dim] -> [B, T, N, A]
        b, t, n, d = obs.shape
        if share:
            # Parameter sharing: run one shared RNN over a batch of size (B*N).
            obs_bn = obs.permute(0, 2, 1, 3).reshape(b * n, t, d)
            q_bn, _ = net(obs_bn, None)
            return q_bn.reshape(b, n, t, -1).permute(0, 2, 1, 3)

        qs = []
        for idx in range(n):
            q_i, _ = net[idx](obs[:, :, idx, :], None)
            qs.append(q_i.unsqueeze(2))
        return torch.cat(qs, dim=2)

    def _mix_q_tot(self, chosen_q: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        # VDN mixing: sum_i Q_i (optionally normalized by number of active agents).
        # chosen_q: [B, T, N], active_mask: [B, T, N] -> q_tot: [B, T]
        masked_q = chosen_q * active_mask
        q_sum = masked_q.sum(dim=2)
        if not self.normalize_by_active:
            return q_sum
        counts = active_mask.sum(dim=2).clamp(min=1.0)
        return q_sum / counts

    def learn(self) -> None:
        # Centralized training step:
        # 1) sample episodes from replay
        # 2) compute per-agent Q-values (online + target)
        # 3) aggregate into Q_tot using the VDN sum mixer
        # 4) compute TD targets with target networks (optionally Double-Q)
        # 5) backprop through the agent networks
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

            next_obs = torch.tensor(np.stack([pad_time(ep["next_obs"]) for ep in batch]), device=self.device)
            next_active_mask = torch.tensor(
                np.stack([pad_time(ep["next_active_mask"]) for ep in batch]), device=self.device
            )
            dones = torch.tensor(
                np.stack([pad_time(ep["done"].reshape(-1, 1)) for ep in batch]),
                device=self.device,
                dtype=torch.float32,
            ).squeeze(-1)

            # Mask padded timesteps so loss does not depend on padding artifacts.
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

            # Per-agent Q-values for current and next observations.
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

            # Select Q-values for the executed actions.
            safe_actions = actions.clone()
            safe_actions[active_mask == 0] = 0
            chosen_q = torch.gather(q_all, 3, safe_actions.unsqueeze(-1)).squeeze(-1)
            chosen_q = chosen_q * active_mask

            q_tot = self._mix_q_tot(chosen_q, active_mask)
            if self.q_tot_clip is not None:
                q_tot = torch.clamp(q_tot, -self.q_tot_clip, self.q_tot_clip)

            # Aggregate per-agent rewards into a team reward.
            # We use mean over active agents so reward scale doesn't change with team size.
            active_counts = active_mask.sum(dim=2).clamp(min=1.0)
            team_rewards = (rewards * active_mask).sum(dim=2) / active_counts

            with torch.no_grad():
                # Next action selection:
                # - Double-Q: choose action using the online network, evaluate with the target network.
                # - Standard DQN: choose action using the target network directly.
                if self.double_q:
                    next_actions = self._masked_argmax(next_q_online, next_action_masks)
                else:
                    next_actions = self._masked_argmax(next_q_target, next_action_masks)
                safe_next_actions = next_actions.clone()
                safe_next_actions[next_active_mask == 0] = 0
                next_chosen_q = torch.gather(next_q_target, 3, safe_next_actions.unsqueeze(-1)).squeeze(-1)
                next_chosen_q = next_chosen_q * next_active_mask

                q_tot_next = self._mix_q_tot(next_chosen_q, next_active_mask)
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
            nn.utils.clip_grad_norm_(list(self._agent_parameters()), max_norm=self.max_grad_norm)
            self.optimizer.step()

            self._learn_steps += 1
            if self._learn_steps % self.target_update_every == 0:
                self._update_targets()

            step_losses.append(float(loss.item()))

        self.loss.append(float(sum(step_losses) / len(step_losses)))
        self.decay_temperature()

    def decay_temperature(self) -> None:
        # Keep temperature above temp_min so agents keep sampling alternative actions.
        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)

    def set_eval_mode(self) -> None:
        # Switch networks to eval mode (disables dropout, etc.).
        if self.share_parameters:
            self.agent_net.eval()
            self.target_agent_net.eval()
        else:
            self.agent_nets.eval()
            self.target_agent_nets.eval()

    def set_train_mode(self) -> None:
        # Switch trainable networks back to train mode (targets can stay in eval mode).
        if self.share_parameters:
            self.agent_net.train()
        else:
            self.agent_nets.train()
