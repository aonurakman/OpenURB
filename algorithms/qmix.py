"""
QMIX (Value Decomposition Networks) implementation.

This follows the standard QMIX setup:
- Per-agent action-value functions `Q_i(o_i, a_i)` (here: GRU-based to support partial observability).
- A monotonic mixing network `Q_tot(s, Q_1..Q_n)` trained with a TD loss.
- Target networks (agent + mixer) for stable TD targets.
- Optional Double Q-learning for target action selection.
-
- Exploration uses a Boltzmann (softmax) policy over per-agent Q-values, controlled by a
  temperature parameter (higher = more exploration, lower = more greedy).

Practical notes:
- Single-step tasks are handled as length-1 sequences (no special-casing).
- Multi-step tasks require calling `reset_episode()` at episode start so GRU state is cleared.

Recurrent state (`reset_episode`)
- `act(...)` maintains an internal GRU hidden state for *inference-time* action selection.
- Call `reset_episode()` once after every environment `reset()` (i.e., at the beginning of each episode/trajectory).
- For OpenURB scripts (single-step per agent/day), this means calling `reset_episode()` once per day; the GRU is
  applied for a single step so it effectively behaves like a feedforward policy.
- For multi-step tasks (e.g. `external_tasks/`), this means calling `reset_episode()` once per episode; the GRU then
  carries information across steps within the episode.

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


def _build_mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(in_dim)
    for hidden_dim in hidden_sizes:
        hidden_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, int(out_dim)))
    return nn.Sequential(*layers)


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
        # Per-agent Q-network used in QMIX.
        # We use an MLP encoder + GRU so each agent can build a latent state from its observation history
        # (useful in partially observable settings).
        self.input_layer = nn.Linear(obs_dim, widths[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(widths[idx], widths[idx + 1]) for idx in range(num_hidden)
        )
        self.rnn = nn.GRU(input_size=widths[-1], hidden_size=rnn_hidden_dim, batch_first=True)
        # Output layer produces Q-values for all discrete actions.
        self.out = nn.Linear(rnn_hidden_dim, action_dim)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction per timestep (shared across the sequence).
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return x

    def forward(self, obs_seq: torch.Tensor, h0: Optional[torch.Tensor] = None):
        # obs_seq: [B, T, obs_dim], h0: [1, B, rnn_hidden_dim]
        b, t, d = obs_seq.shape
        # Apply the MLP to all timesteps, then feed the sequence into the GRU.
        x = self._encode(obs_seq.reshape(b * t, d)).reshape(b, t, -1)
        # GRU output: a hidden state per timestep (a learned summary of history).
        out, hn = self.rnn(x, h0)
        # Map hidden states to Q-values per action for each timestep.
        q = self.out(out)  # [B, T, A]
        return q, hn


class MixingNetwork(nn.Module):
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        mixing_embed_dim: int,
        hypernet_hidden_sizes: Sequence[int],
        weight_clip: Optional[float] = None,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        self.weight_clip = float(weight_clip) if weight_clip is not None else None

        hidden_sizes = tuple(int(x) for x in hypernet_hidden_sizes)

        # QMIX mixer uses "hypernetworks" that take the global state s and output the
        # mixing weights/biases. The key constraint from the QMIX paper is *monotonicity*:
        # dQ_tot / dQ_i >= 0. We enforce this by making the mixing weights non-negative.
        self.hyper_w1 = _build_mlp(state_dim, hidden_sizes, num_agents * mixing_embed_dim)
        self.hyper_b1 = _build_mlp(state_dim, hidden_sizes, mixing_embed_dim)

        self.hyper_w2 = _build_mlp(state_dim, hidden_sizes, mixing_embed_dim)
        self.hyper_b2 = _build_mlp(state_dim, hidden_sizes, 1)

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        # agent_qs: [B, N], states: [B, state_dim] -> [B]
        batch_size = agent_qs.shape[0]
        # First-layer mixing weights are generated from state. softplus() makes them >= 0,
        # which is the monotonicity requirement in QMIX.
        w1 = F.softplus(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        if self.weight_clip is not None:
            # Optional stability trick: cap the magnitude of mixing weights.
            w1 = torch.clamp(w1, max=self.weight_clip)
        w1 = w1.view(batch_size, self.num_agents, self.mixing_embed_dim)
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)

        # Combine individual agent Q-values into a hidden mixing representation.
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.elu(hidden)

        # Second-layer mixing weights and final bias.
        w2 = F.softplus(self.hyper_w2(states))
        if self.weight_clip is not None:
            w2 = torch.clamp(w2, max=self.weight_clip)
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)

        # Final mixed Q-value for the team.
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
        mixing_embed_dim: int = 32,
        hypernet_embed: int = 64,
        mixing_num_hidden: Optional[int] = None,
        mixing_widths: Optional[Sequence[int]] = None,
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
        # Parameter sharing is a common MARL choice:
        # - If agents are homogeneous, sharing reduces parameters and improves sample efficiency.
        # - If agents are heterogeneous, you may want per-agent networks (share_parameters=False).
        self.share_parameters = bool(share_parameters)
        # Boltzmann (softmax) exploration for decentralized execution:
        # sample actions from softmax(Q_i(o_i,·)/temperature), annealing temperature over time.
        self.temperature = float(temp_init)
        self.temp_decay = float(temp_decay)
        self.temp_min = float(temp_min)
        # Batch size counts "episodes" stored in replay (variable-length sequences).
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        # Clip gradients to reduce exploding gradients (especially important with RNNs and TD targets).
        self.max_grad_norm = float(max_grad_norm)
        # TD learning hyperparameters.
        self.gamma = float(gamma)
        self.target_update_every = max(1, int(target_update_every))
        self.double_q = bool(double_q)
        self.tau = float(tau)
        # Optional value clipping for extra stability.
        self.q_tot_clip = float(q_tot_clip) if q_tot_clip is not None else None
        self.use_huber_loss = bool(use_huber_loss)
        self._learn_steps = 0

        if mixing_widths is not None:
            mixing_widths = tuple(int(x) for x in mixing_widths)
            if mixing_num_hidden is not None and len(mixing_widths) != int(mixing_num_hidden):
                raise AssertionError("QMIX mixing_widths and mixing_num_hidden mismatch!")
            hypernet_hidden_sizes = mixing_widths
        else:
            hypernet_hidden_sizes = (int(hypernet_embed),)

        if self.share_parameters:
            # One shared agent network is reused for all agents.
            self.agent_net = AgentRNN(
                self.obs_size, self.action_space_size, rnn_hidden_dim, num_hidden, widths
            ).to(self.device)
            # Target agent network is a lagged copy used for TD targets.
            self.target_agent_net = copy.deepcopy(self.agent_net).to(self.device)
            self.agent_nets = None
            self.target_agent_nets = None
        else:
            # Separate networks per agent (more expressive, more parameters).
            self.agent_net = None
            self.target_agent_net = None
            self.agent_nets = nn.ModuleList(
                [
                    AgentRNN(self.obs_size, self.action_space_size, rnn_hidden_dim, num_hidden, widths).to(self.device)
                    for _ in range(self.num_agents)
                ]
            )
            self.target_agent_nets = copy.deepcopy(self.agent_nets).to(self.device)

        # Mixer takes individual agent Q-values and the global state, and outputs a joint Q_tot.
        self.mixing_net = MixingNetwork(
            self.num_agents,
            self.global_state_size,
            mixing_embed_dim,
            hypernet_hidden_sizes,
            weight_clip=mixing_weight_clip,
        ).to(self.device)
        # Target mixer is lagged, same reason as target agent networks.
        self.target_mixing_net = copy.deepcopy(self.mixing_net).to(self.device)

        # Target networks are used only for inference of TD targets, so keep them in eval mode.
        self.target_mixing_net.eval()
        if self.share_parameters:
            self.target_agent_net.eval()
        else:
            self.target_agent_nets.eval()

        # One optimizer over all trainable parameters (agent networks + mixing network).
        self.optimizer = optim.Adam(
            list(self._agent_parameters()) + list(self.mixing_net.parameters()),
            lr=lr,
        )

        self.loss = []
        # Replay buffer stores *episodes* (sequences of joint transitions).
        self.memory = deque(maxlen=int(buffer_size))
        self._episode_steps = []

        # Inference-time GRU hidden state (per agent). This is what "makes it recurrent"
        # during action selection. It must be reset at episode start.
        self._inference_hidden = {}

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
            h = torch.zeros(1, 1, hidden_dim, device=self.device)
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
        # Store one *joint* transition for all agents at a timestep.
        #
        # Shapes (multi-step):
        # - observations:      [N, obs_dim]
        # - actions:           [N]
        # - rewards:           [N]   (per-agent rewards; we aggregate to a team reward in learn())
        # - active_mask:       [N]   (1 if agent is "active" / part of the learning team at this step)
        # - global_state:      [state_dim] (centralized state for the mixer)
        #
        # We append to `_episode_steps` until `done=True`, then we push a whole episode
        # into replay. This is the standard way to train recurrent networks from replay.
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
            # Episode boundary: stack lists into arrays and store as a single replay entry.
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
        # Convenience wrapper for OpenURB's single-step "day" episodes.
        # We turn a single step into a 1-step episode by creating dummy next_obs/next_state
        # and setting done=True. The learner treats it as a length-1 sequence.
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
        # Convert a python list of step dicts into a single dict of stacked arrays.
        # This makes sampling/padding in `learn()` much faster.
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
        # Set invalid actions to a very low value, so argmax never selects them.
        masked = q_values.masked_fill(~mask, -1e9)
        # If an agent has *no* valid actions, fall back to action 0 (should be rare).
        no_valid = ~mask.any(dim=-1)
        argmax = torch.argmax(masked, dim=-1)
        return torch.where(no_valid, torch.zeros_like(argmax), argmax)

    def _update_targets(self) -> None:
        # Target network update:
        # - Hard update (tau >= 1.0): copy weights exactly (typical DQN-style target net).
        # - Soft update (tau in (0,1)): Polyak averaging for smoother changes.
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
            # Parameter sharing case: run one shared RNN over a batch of size (B*N).
            # We reshape so each agent's sequence is treated independently by the GRU.
            obs_bn = obs.permute(0, 2, 1, 3).reshape(b * n, t, d)
            q_bn, _ = net(obs_bn, None)
            q = q_bn.reshape(b, n, t, -1).permute(0, 2, 1, 3)
            return q
        # Separate-network case: run each agent's network on its own slice.
        qs = []
        for idx in range(n):
            q_i, _ = net[idx](obs[:, :, idx, :], None)
            qs.append(q_i.unsqueeze(2))
        return torch.cat(qs, dim=2)

    def learn(self) -> None:
        # Centralized training step:
        # 1) sample joint episodes from replay
        # 2) compute per-agent Q-values (online + target)
        # 3) mix them into Q_tot via the mixer
        # 4) compute TD targets using target networks (optionally Double-Q)
        # 5) backprop through agent nets + mixer
        if len(self.memory) < self.batch_size:
            return

        step_losses = []
        for _ in range(self.num_epochs):
            # Sample episodes, then pad them to the same length so we can batch them.
            batch = random.sample(self.memory, self.batch_size)
            max_t = max(int(ep["T"]) for ep in batch)

            def pad_time(x, pad_value=0.0):
                t = x.shape[0]
                if t == max_t:
                    return x
                pad_shape = (max_t - t,) + x.shape[1:]
                pad = np.full(pad_shape, pad_value, dtype=x.dtype)
                return np.concatenate([x, pad], axis=0)

            # Convert the replay batch into tensors.
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

            # `time_mask` indicates which timesteps are real vs padding.
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

            # Per-agent Q-values for current states (online net),
            # and for next states (online + target nets) for Double-Q targets.
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

            # Select Q-values for the executed actions. We also zero out inactive agents
            # so they do not contribute to Q_tot or the loss.
            safe_actions = actions.clone()
            safe_actions[active_mask == 0] = 0
            chosen_q = torch.gather(q_all, 3, safe_actions.unsqueeze(-1)).squeeze(-1)
            chosen_q = chosen_q * active_mask

            # Mix chosen per-agent Q-values into a single joint Q_tot.
            b, t, n = chosen_q.shape
            q_tot = self.mixing_net(chosen_q.reshape(b * t, n), states.reshape(b * t, -1)).reshape(b, t)
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

                # Mix next-state per-agent values using the *target* mixing network.
                q_tot_next = self.target_mixing_net(
                    next_chosen_q.reshape(b * t, n), next_states.reshape(b * t, -1)
                ).reshape(b, t)
                if self.q_tot_clip is not None:
                    q_tot_next = torch.clamp(q_tot_next, -self.q_tot_clip, self.q_tot_clip)

                # TD target: r_team + gamma * (1-done) * Q_tot_next
                targets = team_rewards + (1.0 - dones) * self.gamma * q_tot_next
                if self.q_tot_clip is not None:
                    targets = torch.clamp(targets, -self.q_tot_clip, self.q_tot_clip)

            # TD loss over Q_tot, masked to ignore padded timesteps.
            if self.use_huber_loss:
                td = F.smooth_l1_loss(q_tot, targets, reduction="none")
            else:
                td = F.mse_loss(q_tot, targets, reduction="none")
            loss = (td * time_mask).sum() / time_mask.sum().clamp(min=1.0)

            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients across both the agent and mixer networks.
            nn.utils.clip_grad_norm_(
                list(self._agent_parameters()) + list(self.mixing_net.parameters()),
                max_norm=self.max_grad_norm,
            )
            self.optimizer.step()

            # Update target networks periodically (DQN-style stability).
            self._learn_steps += 1
            if self._learn_steps % self.target_update_every == 0:
                self._update_targets()

            step_losses.append(float(loss.item()))

        self.loss.append(float(sum(step_losses) / len(step_losses)))
        # Decay temperature after learning updates.
        self.decay_temperature()

    def decay_temperature(self) -> None:
        # Keep temperature above temp_min so agents keep sampling alternative actions.
        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)

    def set_eval_mode(self) -> None:
        # Switch networks to eval mode (disables dropout, uses running stats for batchnorm, etc.).
        # This is typically used for evaluation/testing when you want deterministic behavior.
        if self.share_parameters:
            self.agent_net.eval()
            self.target_agent_net.eval()
        else:
            self.agent_nets.eval()
            self.target_agent_nets.eval()
        self.mixing_net.eval()
        self.target_mixing_net.eval()

    def set_train_mode(self) -> None:
        # Switch trainable networks back to train mode.
        # Note: target networks can stay in eval mode since we don't backprop through them.
        if self.share_parameters:
            self.agent_net.train()
        else:
            self.agent_nets.train()
        self.mixing_net.train()
