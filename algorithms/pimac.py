"""
PI-MAC (pimac) implementation for OpenURB.

This file implements a MAPPO-style learner with a token-teacher critic:
- Decentralized actor policy pi(a|o) with a shared recurrent actor.
- Centralized value critic trained through a set teacher that builds K coordination
  tokens from active-agent observations via cross-attention.
- Actor-side context distillation from teacher per-agent context using
  heteroscedastic Gaussian NLL.
- Uncertainty-gated FiLM conditioning inside the actor policy head.
- PPO clipped objective + GAE(lambda) + entropy regularization.

Execution remains decentralized (`act` uses only one agent's observation).
Training uses centralized set context only inside critic/teacher paths.
"""

from __future__ import annotations

from collections import deque
import copy
import math
import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baseline_models import BaseLearningModel

__all__ = [
    "PIMACActorRNN",
    "SetTokenTeacher",
    "SetValueCritic",
    "PIMACBase",
    "PIMACAEC",
    "PIMACParallel",
    "PIMAC",
]


def _build_mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int) -> nn.Sequential:
    """Build a simple ReLU MLP."""
    layers: list[nn.Module] = []
    last = int(in_dim)
    for h in hidden_sizes:
        layers.append(nn.Linear(last, int(h)))
        layers.append(nn.ReLU())
        last = int(h)
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


def _sorted_agent_ids(keys) -> list:
    """Stable key ordering helper used by parallel/AEC convenience APIs."""
    return sorted(list(keys), key=lambda x: str(x))


class PIMACActorRNN(nn.Module):
    """
    Per-agent recurrent actor: local observation history -> action logits.

    The actor additionally predicts context mean/log-variance for distillation and
    uses uncertainty-gated FiLM to modulate policy features.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_hidden: int,
        widths: Sequence[int],
        rnn_hidden_dim: int,
        ctx_dim: int,
        ctx_logvar_min: float = -6.0,
        ctx_logvar_max: float = 4.0,
    ):
        super().__init__()
        assert len(widths) == (int(num_hidden) + 1), "PI-MAC actor widths and layer count mismatch."

        self.input_layer = nn.Linear(int(obs_dim), int(widths[0]))
        self.hidden_layers = nn.ModuleList(
            nn.Linear(int(widths[idx]), int(widths[idx + 1])) for idx in range(int(num_hidden))
        )
        self.rnn = nn.GRU(input_size=int(widths[-1]), hidden_size=int(rnn_hidden_dim), batch_first=True)

        hidden_dim = int(rnn_hidden_dim)
        self.ctx_mu_head = nn.Linear(hidden_dim, int(ctx_dim))
        self.ctx_logvar_head = nn.Linear(hidden_dim, int(ctx_dim))
        self.film_head = nn.Linear(int(ctx_dim), 2 * hidden_dim)

        # Scalar uncertainty gate: g = sigmoid(w * (-mean(logvar)) + b)
        self.gate_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.gate_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.policy_head = nn.Linear(hidden_dim, int(action_dim))

        self.ctx_logvar_min = float(ctx_logvar_min)
        self.ctx_logvar_max = float(ctx_logvar_max)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return x

    def forward(self, obs_seq: torch.Tensor, h0: Optional[torch.Tensor] = None, return_aux: bool = False):
        """
        Args:
            obs_seq: [B, T, obs_dim]
            h0: optional GRU hidden [1, B, H]
            return_aux: include distillation/FiLM outputs when True.

        Returns:
            logits: [B, T, action_dim]
            hn: [1, B, H]
            aux (optional): dict with
                - ctx_mu: [B, T, ctx_dim]
                - ctx_logvar: [B, T, ctx_dim]
                - gate: [B, T, 1]
        """
        b, t, d = obs_seq.shape
        x = self._encode(obs_seq.reshape(b * t, d)).reshape(b, t, -1)
        h, hn = self.rnn(x, h0)

        ctx_mu = self.ctx_mu_head(h)
        ctx_logvar = torch.clamp(self.ctx_logvar_head(h), min=self.ctx_logvar_min, max=self.ctx_logvar_max)

        gamma_beta = self.film_head(ctx_mu)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        uncertainty = ctx_logvar.mean(dim=-1, keepdim=True)
        gate = torch.sigmoid(self.gate_weight * (-uncertainty) + self.gate_bias)

        h_mod = h * (1.0 + gate * gamma) + gate * beta
        logits = self.policy_head(h_mod)

        if not return_aux:
            return logits, hn

        return logits, hn, {"ctx_mu": ctx_mu, "ctx_logvar": ctx_logvar, "gate": gate}


class SetTokenTeacher(nn.Module):
    """
    Token-based set teacher with masked cross-attention.

    Inputs:
        obs: [B, T, N, obs_dim]
        active_mask: [B, T, N] in {0,1}

    Outputs:
        tokens: [B, T, K, D]
        ctx_teacher: [B, T, N, D]
    """

    def __init__(
        self,
        obs_dim: int,
        tok_dim: int,
        num_tokens: int,
        teacher_hidden_sizes: Sequence[int],
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.tok_dim = int(tok_dim)
        self.num_tokens = int(num_tokens)

        self.agent_mlp = _build_mlp(self.obs_dim, tuple(teacher_hidden_sizes), self.tok_dim)
        self.token_queries = nn.Parameter(torch.randn(self.num_tokens, self.tok_dim) * 0.02)

    @staticmethod
    def _masked_cross_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Mask-safe scaled dot-product attention.

        query: [B, T, Q, D]
        key: [B, T, M, D]
        value: [B, T, M, D]
        key_mask: optional [B, T, M]
        """
        scale = math.sqrt(float(query.shape[-1]))
        scores = torch.einsum("btqd,btmd->btqm", query, key) / max(scale, 1e-8)

        if key_mask is None:
            weights = torch.softmax(scores, dim=-1)
            return torch.einsum("btqm,btmd->btqd", weights, value)

        mask_bool = key_mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(~mask_bool.unsqueeze(2), -1e9)

        # Numerical stabilization for all-masked rows:
        # - softmax(-1e9, ..., -1e9) is well-defined but produces an arbitrary simplex point.
        # - multiplying by the boolean mask removes those entries.
        # - dividing by the masked sum (clamped) keeps rows finite and sums to 1.0 only when
        #   at least one key was valid; otherwise the row stays all zeros.
        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask_bool.unsqueeze(2).to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.einsum("btqm,btmd->btqd", weights, value)

    def forward(self, obs: torch.Tensor, active_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, n, d = obs.shape

        emb = self.agent_mlp(obs.reshape(b * t * n, d)).reshape(b, t, n, self.tok_dim)
        q_tokens = self.token_queries.view(1, 1, self.num_tokens, self.tok_dim).expand(b, t, -1, -1)

        tokens = self._masked_cross_attention(q_tokens, emb, emb, key_mask=active_mask)
        ctx_teacher = self._masked_cross_attention(emb, tokens, tokens, key_mask=None)

        # Ensure inactive/padded agents do not contribute to downstream losses/metrics.
        ctx_teacher = ctx_teacher * active_mask.unsqueeze(-1)
        return tokens, ctx_teacher


class SetValueCritic(nn.Module):
    """Centralized value critic that consumes token teacher context."""

    def __init__(
        self,
        obs_dim: int,
        tok_dim: int,
        num_tokens: int,
        teacher_hidden_sizes: Sequence[int],
        critic_hidden_sizes: Sequence[int],
    ):
        super().__init__()
        self.teacher = SetTokenTeacher(
            obs_dim=int(obs_dim),
            tok_dim=int(tok_dim),
            num_tokens=int(num_tokens),
            teacher_hidden_sizes=tuple(teacher_hidden_sizes),
        )
        self.value_mlp = _build_mlp(int(tok_dim), tuple(critic_hidden_sizes), out_dim=1)

    def forward(
        self,
        obs: torch.Tensor,
        active_mask: torch.Tensor,
        return_details: bool = False,
    ):
        tokens, ctx_teacher = self.teacher(obs, active_mask)
        team_ctx = tokens.mean(dim=2)

        b, t, c = team_ctx.shape
        values = self.value_mlp(team_ctx.reshape(b * t, c)).reshape(b, t)

        if not return_details:
            return values
        return values, tokens, ctx_teacher


class PIMACBase(BaseLearningModel):
    """
    Shared PI-MAC core.

    This base class contains only algorithmic/common logic:
    - actor/critic/teacher construction
    - replay formatting and roster-aware finalization
    - GAE target computation
    - PPO/value/distillation updates and diagnostics
    - train/eval mode and optimizer state

    Interaction styles (AEC vs parallel env APIs) are implemented by subclasses.
    """

    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        num_agents: int,
        device: str = "cpu",
        buffer_size: int = 2048,
        batch_size: int = 32,
        lr: float = 3e-4,
        num_epochs: int = 4,
        num_hidden: int = 2,
        widths: Sequence[int] = (128, 128, 128),
        rnn_hidden_dim: int = 64,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: Optional[float] = 1.0,
        critic_hidden_sizes: Sequence[int] = (128, 128),
        deterministic: bool = False,
        num_tokens: int = 4,
        tok_dim: int = 128,
        teacher_hidden_sizes: Sequence[int] = (128, 128),
        teacher_drop_prob: float = 0.1,
        teacher_ema_tau: float = 0.01,
        distill_weight: float = 0.1,
        ctx_logvar_min: float = -6.0,
        ctx_logvar_max: float = 4.0,
    ):
        super().__init__()

        self.device = device
        self.obs_size = int(state_size)
        self.action_space_size = int(action_space_size)
        self.num_agents = int(num_agents)

        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.normalize_advantage = bool(normalize_advantage)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

        self.ctx_dim = int(tok_dim)
        self.num_tokens = int(num_tokens)
        self.teacher_hidden_sizes = tuple(teacher_hidden_sizes)
        self.teacher_drop_prob = float(teacher_drop_prob)
        self.teacher_drop_prob = max(0.0, min(1.0, self.teacher_drop_prob))
        self.teacher_ema_tau = float(max(0.0, min(1.0, teacher_ema_tau)))
        self.distill_weight = float(distill_weight)
        self.ctx_logvar_min = float(ctx_logvar_min)
        self.ctx_logvar_max = float(ctx_logvar_max)

        self.deterministic = bool(deterministic)

        self.actor_net = PIMACActorRNN(
            obs_dim=self.obs_size,
            action_dim=self.action_space_size,
            num_hidden=int(num_hidden),
            widths=tuple(widths),
            rnn_hidden_dim=int(rnn_hidden_dim),
            ctx_dim=self.ctx_dim,
            ctx_logvar_min=self.ctx_logvar_min,
            ctx_logvar_max=self.ctx_logvar_max,
        ).to(self.device)

        self.critic = SetValueCritic(
            obs_dim=self.obs_size,
            tok_dim=self.ctx_dim,
            num_tokens=self.num_tokens,
            teacher_hidden_sizes=self.teacher_hidden_sizes,
            critic_hidden_sizes=tuple(critic_hidden_sizes),
        ).to(self.device)

        # EMA target teacher is used only for distillation targets + diagnostics.
        self.target_teacher = copy.deepcopy(self.critic.teacher).to(self.device)
        for param in self.target_teacher.parameters():
            param.requires_grad_(False)
        self.target_teacher.eval()

        actor_params = list(self._actor_parameters())
        critic_params = list(self.critic.parameters())
        self.optimizer = optim.Adam(actor_params + critic_params, lr=float(lr))

        # Replay stores complete trajectories, but updates are strictly on-policy.
        self.memory = deque(maxlen=int(buffer_size))
        self._episode_steps: list[dict] = []
        self._aec_cycle = None

        self.loss: list[float] = []
        self.last_losses: dict[str, float] = {}
        self.loss_history: list[dict[str, float]] = []

        # Per-agent hidden state used only during decentralized inference.
        self._inference_hidden: dict[object, torch.Tensor] = {}

    def _actor_parameters(self):
        return self.actor_net.parameters()

    def reset_episode(self) -> None:
        """
        Clear recurrent inference state for a new environment episode.
        """
        self._inference_hidden = {}
        self._aec_cycle = None

    def _get_h0(self, agent_key: object, hidden_dim: int) -> torch.Tensor:
        h = self._inference_hidden.get(agent_key)
        if h is None:
            h = torch.zeros(1, 1, int(hidden_dim), device=self.device)
        return h

    def _set_h(self, agent_key: object, h: torch.Tensor) -> None:
        self._inference_hidden[agent_key] = h.detach()

    def _act_single(
        self,
        state: np.ndarray,
        actor_key: object,
    ) -> int:
        """
        Shared decentralized action path for one agent stream.

        Args:
            state: local observation.
            actor_key: recurrent-state key.
        """
        obs_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, 1, -1)
        hidden_dim = self.actor_net.rnn.hidden_size
        h0 = self._get_h0(actor_key, hidden_dim)
        logits_seq, hn = self.actor_net(obs_t, h0)

        self._set_h(actor_key, hn)
        logits = logits_seq.squeeze(0).squeeze(0)

        if self.deterministic:
            return int(torch.argmax(logits).item())

        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    @staticmethod
    def _ensure_state(state: Optional[np.ndarray]) -> np.ndarray:
        if state is None:
            return np.zeros(1, dtype=np.float32)
        return np.asarray(state, dtype=np.float32)

    def _coerce_step(
        self,
        observations,
        actions,
        rewards,
        active_mask,
        global_state,
        next_observations,
        next_active_mask,
        next_global_state,
        done,
        agent_ids: Optional[Sequence[object]],
    ) -> dict:
        obs_is_dict = isinstance(observations, dict)
        if obs_is_dict:
            if agent_ids is None:
                agent_ids = _sorted_agent_ids(observations.keys())
            obs_arr = np.stack([np.asarray(observations[aid], dtype=np.float32) for aid in agent_ids], axis=0)
        else:
            obs_arr = np.asarray(observations, dtype=np.float32)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr.reshape(1, -1)
            if agent_ids is None:
                agent_ids = list(range(obs_arr.shape[0]))

        if isinstance(actions, dict):
            actions_arr = np.asarray([actions.get(aid, 0) for aid in agent_ids], dtype=np.int64)
        else:
            actions_arr = np.asarray(actions, dtype=np.int64)
            if actions_arr.ndim == 0:
                actions_arr = actions_arr.reshape(1)

        if isinstance(rewards, dict):
            rewards_arr = np.asarray([rewards.get(aid, 0.0) for aid in agent_ids], dtype=np.float32)
        else:
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            if rewards_arr.ndim == 0:
                rewards_arr = rewards_arr.reshape(1)

        if active_mask is None:
            active_arr = np.ones(len(agent_ids), dtype=np.float32)
        elif isinstance(active_mask, dict):
            active_arr = np.asarray([active_mask.get(aid, 0.0) for aid in agent_ids], dtype=np.float32)
        else:
            active_arr = np.asarray(active_mask, dtype=np.float32)
            if active_arr.ndim == 0:
                active_arr = active_arr.reshape(1)

        next_obs_is_dict = isinstance(next_observations, dict)
        if next_obs_is_dict:
            next_agent_ids = _sorted_agent_ids(next_observations.keys())
            if next_agent_ids:
                next_obs_arr = np.stack(
                    [np.asarray(next_observations[aid], dtype=np.float32) for aid in next_agent_ids], axis=0
                )
            else:
                # Terminal parallel steps may provide an empty next-observation dict.
                # Keep a valid 2D tensor shape to preserve downstream roster/padding logic.
                next_obs_arr = np.zeros((0, obs_arr.shape[-1]), dtype=np.float32)
        else:
            next_obs_arr = np.asarray(next_observations, dtype=np.float32)
            if next_obs_arr.ndim == 1:
                next_obs_arr = next_obs_arr.reshape(1, -1)
            next_agent_ids = list(range(next_obs_arr.shape[0]))

        if next_active_mask is None:
            next_active_arr = np.ones(len(next_agent_ids), dtype=np.float32)
        elif isinstance(next_active_mask, dict):
            next_active_arr = np.asarray(
                [next_active_mask.get(aid, 0.0) for aid in next_agent_ids], dtype=np.float32
            )
        else:
            next_active_arr = np.asarray(next_active_mask, dtype=np.float32)
            if next_active_arr.ndim == 0:
                next_active_arr = next_active_arr.reshape(1)

        return {
            "agent_ids": list(agent_ids),
            "obs": obs_arr,
            "actions": actions_arr,
            "rewards": rewards_arr,
            "active_mask": active_arr,
            "state": self._ensure_state(global_state),
            "next_obs": next_obs_arr,
            "next_active_mask": next_active_arr,
            "next_state": self._ensure_state(next_global_state),
            "done": bool(done),
            "next_agent_ids": list(next_agent_ids),
        }

    def store_transition(
        self,
        observations,
        actions,
        rewards,
        active_mask,
        global_state: Optional[np.ndarray],
        next_observations,
        next_active_mask,
        next_global_state: Optional[np.ndarray],
        done: bool,
        agent_ids: Optional[Sequence[object]] = None,
    ) -> None:
        self._episode_steps.append(
            self._coerce_step(
                observations,
                actions,
                rewards,
                active_mask,
                global_state,
                next_observations,
                next_active_mask,
                next_global_state,
                done,
                agent_ids,
            )
        )

        if done:
            episode = self._finalize_episode(self._episode_steps)
            self.memory.append(episode)
            self._episode_steps = []

    def _actor_forward(self, obs: torch.Tensor, return_aux: bool = False):
        """obs: [B, T, N, obs_dim] -> logits: [B, T, N, A], aux tensors if requested."""
        b, t, n, d = obs.shape

        # Shared-actor execution: flatten [B, N] into one stream axis so the actor
        # processes each agent trajectory independently with the same parameters.
        obs_bn = obs.permute(0, 2, 1, 3).reshape(b * n, t, d)
        if return_aux:
            logits_bn, _, aux_bn = self.actor_net(obs_bn, None, return_aux=True)
            logits = logits_bn.reshape(b, n, t, -1).permute(0, 2, 1, 3)
            aux = {
                "ctx_mu": aux_bn["ctx_mu"].reshape(b, n, t, -1).permute(0, 2, 1, 3),
                "ctx_logvar": aux_bn["ctx_logvar"].reshape(b, n, t, -1).permute(0, 2, 1, 3),
                "gate": aux_bn["gate"].reshape(b, n, t, -1).permute(0, 2, 1, 3),
            }
            return logits, aux

        logits_bn, _ = self.actor_net(obs_bn, None)
        logits = logits_bn.reshape(b, n, t, -1).permute(0, 2, 1, 3)
        return logits, None

    @staticmethod
    def _team_rewards(rewards: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        numer = (rewards * active_mask).sum(axis=1)
        denom = active_mask.sum(axis=1)
        out = np.zeros(rewards.shape[0], dtype=np.float32)
        valid = denom > 0
        out[valid] = numer[valid] / denom[valid]
        return out

    def _compute_old_log_probs(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(0)
        active_t = torch.as_tensor(active_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self._actor_forward(obs_t, return_aux=False)
            dist = torch.distributions.Categorical(logits=logits)
            old_log_probs = dist.log_prob(actions_t) * active_t

        return old_log_probs.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def _compute_values(self, obs: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        active_t = torch.as_tensor(active_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            values = self.critic(obs_t, active_t).squeeze(0)
        return values.cpu().numpy().astype(np.float32, copy=False)

    def _compute_gae(
        self,
        team_rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        dones: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generalized Advantage Estimation on team rewards.

        The recursion is performed backward in time and respects episode terminals via
        `(1 - done_t)` in both TD residual and advantage carry-over terms.
        """
        advantages = np.zeros_like(team_rewards, dtype=np.float32)
        last_adv = 0.0

        for t in range(team_rewards.shape[0] - 1, -1, -1):
            nonterminal = 1.0 - float(dones[t])
            delta = team_rewards[t] + self.gamma * nonterminal * next_values[t] - values[t]
            last_adv = delta + self.gamma * self.gae_lambda * nonterminal * last_adv
            advantages[t] = last_adv

        returns = advantages + values
        return advantages.astype(np.float32, copy=False), returns.astype(np.float32, copy=False)

    def _finalize_episode(self, steps: list[dict]) -> dict:
        """
        Convert variable-roster raw steps into one roster-aligned episode record.

        Roster flow:
        1. Build the union of agent ids observed in current and next observations.
        2. Allocate dense `[T, N, ...]` arrays using that roster as the slot map.
        3. Fill active entries and leave absent agents at zero.
        4. Compute on-policy targets (`old_log_probs`, value estimates, GAE, returns).
        """
        roster = []
        roster_index = {}

        def add_agents(agent_ids):
            for aid in agent_ids:
                if aid not in roster_index:
                    roster_index[aid] = len(roster)
                    roster.append(aid)

        for step in steps:
            add_agents(step["agent_ids"])
            add_agents(step["next_agent_ids"])

        t = len(steps)
        n = len(roster)
        obs_dim = steps[0]["obs"].shape[-1]

        obs = np.zeros((t, n, obs_dim), dtype=np.float32)
        actions = np.zeros((t, n), dtype=np.int64)
        rewards = np.zeros((t, n), dtype=np.float32)
        active_mask = np.zeros((t, n), dtype=np.float32)
        next_obs = np.zeros((t, n, obs_dim), dtype=np.float32)
        next_active_mask = np.zeros((t, n), dtype=np.float32)
        # `state` and `next_state` are retained for API compatibility with callers that
        # persist global state snapshots, even though PI-MAC PPO losses do not consume them.
        state = np.stack([s["state"] for s in steps], axis=0)
        next_state = np.stack([s["next_state"] for s in steps], axis=0)
        done = np.asarray([s["done"] for s in steps], dtype=np.float32)

        for ti, step in enumerate(steps):
            for idx, aid in enumerate(step["agent_ids"]):
                slot = roster_index[aid]
                obs[ti, slot] = step["obs"][idx]
                actions[ti, slot] = step["actions"][idx]
                rewards[ti, slot] = step["rewards"][idx]
                active_mask[ti, slot] = step["active_mask"][idx]

            for idx, aid in enumerate(step["next_agent_ids"]):
                slot = roster_index[aid]
                next_obs[ti, slot] = step["next_obs"][idx]
                next_active_mask[ti, slot] = step["next_active_mask"][idx]

        # Behavior-policy statistics are frozen at finalization time; PPO reuses these as
        # `old_log_probs` and bootstrap values during subsequent updates.
        old_log_probs = self._compute_old_log_probs(obs, actions, active_mask)
        values = self._compute_values(obs, active_mask)
        next_values = self._compute_values(next_obs, next_active_mask)
        team_rewards = self._team_rewards(rewards, active_mask)
        advantages, returns = self._compute_gae(team_rewards, values, next_values, done)

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
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "returns": returns,
            "values": values,
            "team_rewards": team_rewards,
            "T": int(obs.shape[0]),
            "N": int(obs.shape[1]),
        }

    def _sample_teacher_mask(self, active_mask: torch.Tensor, drop_prob: float) -> torch.Tensor:
        """
        Drop active agents for teacher/critic inputs while keeping at least one active
        entry per timestep whenever the original mask has active agents.
        """
        if drop_prob <= 0.0:
            return active_mask

        keep = (torch.rand_like(active_mask) > float(drop_prob)).to(dtype=active_mask.dtype)
        dropped = active_mask * keep

        original_active = active_mask > 0.0
        originally_non_empty = original_active.any(dim=-1)
        became_empty = (dropped.sum(dim=-1) <= 0.0) & originally_non_empty

        if became_empty.any():
            bt_indices = torch.nonzero(became_empty, as_tuple=False)
            for bt in bt_indices:
                b_idx = int(bt[0].item())
                t_idx = int(bt[1].item())
                active_ids = torch.nonzero(original_active[b_idx, t_idx], as_tuple=False).squeeze(-1)
                if active_ids.numel() == 0:
                    continue
                choice = int(torch.randint(active_ids.numel(), (1,), device=active_mask.device).item())
                keep_idx = int(active_ids[choice].item())
                dropped[b_idx, t_idx, keep_idx] = 1.0

        return dropped

    def _teacher_for_targets(self) -> SetTokenTeacher:
        """Select distillation target source (EMA target teacher or online teacher)."""
        if self.teacher_ema_tau > 0.0:
            return self.target_teacher
        return self.critic.teacher

    def _update_target_teacher(self) -> None:
        """
        Exponential moving average (EMA) update of target teacher parameters.

        Buffers are updated with EMA for floating tensors and copied exactly for non-floating
        buffers so module state stays consistent.
        """
        if self.teacher_ema_tau <= 0.0:
            return

        tau = float(self.teacher_ema_tau)
        with torch.no_grad():
            for tgt_param, src_param in zip(self.target_teacher.parameters(), self.critic.teacher.parameters()):
                tgt_param.mul_(1.0 - tau).add_(src_param, alpha=tau)

            for tgt_buf, src_buf in zip(self.target_teacher.buffers(), self.critic.teacher.buffers()):
                if torch.is_floating_point(tgt_buf):
                    tgt_buf.mul_(1.0 - tau).add_(src_buf, alpha=tau)
                else:
                    tgt_buf.copy_(src_buf)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over valid entries only; avoids NaNs when masks are fully zero."""
        return (values * mask).sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def _pad_time(x: np.ndarray, max_t: int, pad_value: float = 0.0) -> np.ndarray:
        """Pad along the time axis only: [T, ...] -> [max_t, ...]."""
        t = x.shape[0]
        if t == max_t:
            return x
        pad_shape = (max_t - t,) + x.shape[1:]
        pad = np.full(pad_shape, pad_value, dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    @staticmethod
    def _pad_time_agents(x: np.ndarray, max_t: int, max_n: int, pad_value: float = 0.0) -> np.ndarray:
        """
        Pad `[T, N, ...]` tensors to a shared minibatch grid.

        This is the key roster/padding bridge from variable-size episodes to one dense
        tensor batch. Padding is always zero so masks can cleanly remove fake entries.
        """
        t, n = x.shape[0], x.shape[1]
        if t == max_t and n == max_n:
            return x
        pad_width = [(0, max_t - t), (0, max_n - n)] + [(0, 0)] * (x.ndim - 2)
        return np.pad(x, pad_width, mode="constant", constant_values=pad_value)

    def _build_minibatch_tensors(self, batch: list[dict]) -> dict[str, torch.Tensor | None]:
        """
        Convert sampled finalized episodes into padded torch tensors.

        Returned masks have distinct roles:
        - `time_mask`: real timesteps vs right-padding.
        - `combined_mask`: active agents at real timesteps (policy/distillation support).
        - `valid_time`: real timesteps with at least one active agent (value support).
        """
        max_t = max(int(ep["T"]) for ep in batch)
        max_n = max(int(ep["N"]) for ep in batch)

        obs = torch.as_tensor(
            np.stack([self._pad_time_agents(ep["obs"], max_t, max_n) for ep in batch]),
            device=self.device,
            dtype=torch.float32,
        )
        actions = torch.as_tensor(
            np.stack([self._pad_time_agents(ep["actions"], max_t, max_n, pad_value=0) for ep in batch]),
            device=self.device,
            dtype=torch.int64,
        )
        active_mask = torch.as_tensor(
            np.stack([self._pad_time_agents(ep["active_mask"], max_t, max_n, pad_value=0.0) for ep in batch]),
            device=self.device,
            dtype=torch.float32,
        )
        old_log_probs = torch.as_tensor(
            np.stack([self._pad_time_agents(ep["old_log_probs"], max_t, max_n, pad_value=0.0) for ep in batch]),
            device=self.device,
            dtype=torch.float32,
        )
        advantages = torch.as_tensor(
            np.stack([self._pad_time(ep["advantages"], max_t, pad_value=0.0) for ep in batch]),
            device=self.device,
            dtype=torch.float32,
        )
        returns = torch.as_tensor(
            np.stack([self._pad_time(ep["returns"], max_t, pad_value=0.0) for ep in batch]),
            device=self.device,
            dtype=torch.float32,
        )

        lengths = torch.as_tensor([int(ep["T"]) for ep in batch], device=self.device, dtype=torch.int64)
        time_mask = (torch.arange(max_t, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)).to(
            dtype=torch.float32
        )
        combined_mask = active_mask * time_mask.unsqueeze(-1)
        valid_time = time_mask * (active_mask.sum(dim=2) > 0.0).to(dtype=torch.float32)

        return {
            "obs": obs,
            "actions": actions,
            "active_mask": active_mask,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "returns": returns,
            "time_mask": time_mask,
            "combined_mask": combined_mask,
            "valid_time": valid_time,
        }

    def _normalize_advantages(self, advantages: torch.Tensor, valid_time: torch.Tensor) -> torch.Tensor:
        """
        Normalize team advantages over valid timesteps only.

        We normalize on `[B, T]` values before broadcasting to agents, preserving the
        original team-level PPO signal semantics.
        """
        if not self.normalize_advantage:
            return advantages

        flat_adv = advantages[valid_time.bool()]
        if flat_adv.numel() == 0:
            return advantages

        adv_mean = flat_adv.mean()
        adv_std = flat_adv.std(unbiased=False)
        return (advantages - adv_mean) / (adv_std + 1e-8)

    def _compute_policy_terms(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute PPO policy and entropy terms on active decisions only."""
        logits, actor_aux = self._actor_forward(obs, return_aux=True)

        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        adv_expanded = advantages.unsqueeze(-1).expand_as(new_log_probs)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_expanded

        decision_denom = combined_mask.sum().clamp(min=1.0)
        policy_loss = -(torch.min(surr1, surr2) * combined_mask).sum() / decision_denom
        entropy_bonus = (entropy * combined_mask).sum() / decision_denom

        return {
            "actor_aux": actor_aux,
            "new_log_probs": new_log_probs,
            "ratio": ratio,
            "policy_loss": policy_loss,
            "entropy_bonus": entropy_bonus,
            "decision_denom": decision_denom,
        }

    def _compute_value_and_distillation_terms(
        self,
        obs: torch.Tensor,
        active_mask: torch.Tensor,
        returns: torch.Tensor,
        valid_time: torch.Tensor,
        combined_mask: torch.Tensor,
        decision_denom: torch.Tensor,
        actor_aux: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | SetTokenTeacher]:
        """
        Compute critic/value and heteroscedastic distillation losses.

        Distillation uses full active masks for teacher targets; agent-drop is used only
        on critic input as robustness augmentation.
        """
        critic_mask = self._sample_teacher_mask(active_mask, self.teacher_drop_prob)
        values, _, _ = self.critic(obs, critic_mask, return_details=True)
        value_loss = (((returns - values) ** 2) * valid_time).sum() / valid_time.sum().clamp(min=1.0)

        teacher_for_targets = self._teacher_for_targets()
        with torch.no_grad():
            _, ctx_tgt = teacher_for_targets(obs, active_mask)

        ctx_mu = actor_aux["ctx_mu"]
        ctx_logvar = actor_aux["ctx_logvar"]
        gate = actor_aux["gate"].squeeze(-1)

        # Heteroscedastic Gaussian NLL:
        #   0.5 * exp(-logvar) * (mu - target)^2 + 0.5 * logvar
        # summed over context channels and averaged over valid agent-decisions.
        distill_nll = 0.5 * torch.exp(-ctx_logvar) * ((ctx_mu - ctx_tgt) ** 2) + 0.5 * ctx_logvar
        distill_per_agent = distill_nll.sum(dim=-1)
        distill_loss = (distill_per_agent * combined_mask).sum() / decision_denom

        distill_mse_per_agent = ((ctx_mu - ctx_tgt) ** 2).mean(dim=-1)
        distill_mse = (distill_mse_per_agent * combined_mask).sum() / decision_denom

        return {
            "values": values,
            "value_loss": value_loss,
            "distill_loss": distill_loss,
            "distill_mse": distill_mse,
            "ctx_logvar": ctx_logvar,
            "gate": gate,
            "teacher_for_targets": teacher_for_targets,
        }

    def _compute_update_diagnostics(
        self,
        obs: torch.Tensor,
        active_mask: torch.Tensor,
        time_mask: torch.Tensor,
        combined_mask: torch.Tensor,
        valid_time: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        ratio: torch.Tensor,
        ctx_logvar: torch.Tensor,
        gate: torch.Tensor,
        teacher_for_targets: SetTokenTeacher,
        decision_denom: torch.Tensor,
    ) -> dict[str, torch.Tensor | float]:
        """
        Compute the same diagnostics used by existing training scripts.

        Includes:
        - PPO stability signals (KL/clip fraction/explained variance)
        - teacher/token interpretability metrics
        - context consistency diagnostic under agent-drop masking
        - uncertainty/gate statistics
        """
        with torch.no_grad():
            log_ratio = new_log_probs - old_log_probs
            approx_kl = ((-log_ratio) * combined_mask).sum() / decision_denom
            clip_frac = (((ratio - 1.0).abs() > self.clip_eps).to(dtype=torch.float32) * combined_mask).sum()
            clip_frac = clip_frac / decision_denom

            vt = valid_time.bool()
            if vt.any():
                returns_valid = returns[vt]
                values_valid = values.detach()[vt]
                var_returns = torch.var(returns_valid, unbiased=False)
                if float(var_returns.item()) > 1e-8:
                    explained_variance = 1.0 - (
                        torch.var(returns_valid - values_valid, unbiased=False) / (var_returns + 1e-8)
                    )
                else:
                    explained_variance = torch.tensor(0.0, device=self.device)
            else:
                explained_variance = torch.tensor(0.0, device=self.device)

            tokens_full, ctx_full_online = self.critic.teacher(obs, active_mask)
            token_norm = torch.linalg.norm(tokens_full, dim=-1)
            token_mask = valid_time.unsqueeze(-1).expand_as(token_norm)
            token_norm_mean = self._masked_mean(token_norm, token_mask)

            teacher_ctx_norm = torch.linalg.norm(ctx_full_online, dim=-1)
            teacher_ctx_norm_mean = self._masked_mean(teacher_ctx_norm, combined_mask)

            # Consistency metric: compare teacher context under full vs dropped masks.
            # This is diagnostic-only (no gradient path, no extra loss term).
            diag_drop_mask = self._sample_teacher_mask(active_mask, self.teacher_drop_prob)
            _, ctx_full_diag = teacher_for_targets(obs, active_mask)
            _, ctx_drop_diag = teacher_for_targets(obs, diag_drop_mask)
            diag_mask = diag_drop_mask * time_mask.unsqueeze(-1)
            ctx_l2 = torch.sqrt(((ctx_full_diag - ctx_drop_diag) ** 2).sum(dim=-1) + 1e-12)
            ctx_consistency_l2 = self._masked_mean(ctx_l2, diag_mask)

            logvar_mean_per_agent = ctx_logvar.mean(dim=-1)
            selected_logvar = logvar_mean_per_agent[combined_mask.bool()]
            if selected_logvar.numel() > 0:
                ctx_logvar_mean = selected_logvar.mean()
                ctx_logvar_std = selected_logvar.std(unbiased=False)
            else:
                ctx_logvar_mean = torch.tensor(0.0, device=self.device)
                ctx_logvar_std = torch.tensor(0.0, device=self.device)

            selected_gate = gate[combined_mask.bool()]
            if selected_gate.numel() > 0:
                gate_mean = selected_gate.mean()
                gate_std = selected_gate.std(unbiased=False)
            else:
                gate_mean = torch.tensor(0.0, device=self.device)
                gate_std = torch.tensor(0.0, device=self.device)

            active_decisions = float(combined_mask.sum().item())
            active_agents_per_step = active_mask.sum(dim=-1)
            active_agents_mean = self._masked_mean(active_agents_per_step, valid_time)

        return {
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "explained_variance": explained_variance,
            "token_norm_mean": token_norm_mean,
            "teacher_ctx_norm_mean": teacher_ctx_norm_mean,
            "ctx_consistency_l2": ctx_consistency_l2,
            "ctx_logvar_mean": ctx_logvar_mean,
            "ctx_logvar_std": ctx_logvar_std,
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "active_decisions": active_decisions,
            "active_agents_mean": active_agents_mean,
        }

    def learn(self) -> None:
        """
        Run one PI-MAC update phase.

        Each epoch samples on-policy finalized episodes, pads to a dense batch, applies
        PPO + value + distillation objectives, logs diagnostics, then clears memory.
        """
        if len(self.memory) < self.batch_size:
            return

        epoch_losses = []
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            minibatch = self._build_minibatch_tensors(batch)
            obs = minibatch["obs"]
            actions = minibatch["actions"]
            active_mask = minibatch["active_mask"]
            old_log_probs = minibatch["old_log_probs"]
            advantages = self._normalize_advantages(minibatch["advantages"], minibatch["valid_time"])
            returns = minibatch["returns"]
            time_mask = minibatch["time_mask"]
            combined_mask = minibatch["combined_mask"]
            valid_time = minibatch["valid_time"]

            policy_terms = self._compute_policy_terms(
                obs=obs,
                actions=actions,
                old_log_probs=old_log_probs,
                advantages=advantages,
                combined_mask=combined_mask,
            )
            value_distill_terms = self._compute_value_and_distillation_terms(
                obs=obs,
                active_mask=active_mask,
                returns=returns,
                valid_time=valid_time,
                combined_mask=combined_mask,
                decision_denom=policy_terms["decision_denom"],
                actor_aux=policy_terms["actor_aux"],
            )

            policy_loss = policy_terms["policy_loss"]
            entropy_bonus = policy_terms["entropy_bonus"]
            ratio = policy_terms["ratio"]
            new_log_probs = policy_terms["new_log_probs"]
            decision_denom = policy_terms["decision_denom"]
            value_loss = value_distill_terms["value_loss"]
            distill_loss = value_distill_terms["distill_loss"]
            distill_mse = value_distill_terms["distill_mse"]
            values = value_distill_terms["values"]
            ctx_logvar = value_distill_terms["ctx_logvar"]
            gate = value_distill_terms["gate"]
            teacher_for_targets = value_distill_terms["teacher_for_targets"]

            ppo_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus
            loss = ppo_loss + self.distill_weight * distill_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    list(self._actor_parameters()) + list(self.critic.parameters()),
                    max_norm=self.max_grad_norm,
                )
            self.optimizer.step()
            self._update_target_teacher()

            diagnostics = self._compute_update_diagnostics(
                obs=obs,
                active_mask=active_mask,
                time_mask=time_mask,
                combined_mask=combined_mask,
                valid_time=valid_time,
                returns=returns,
                values=values,
                old_log_probs=old_log_probs,
                new_log_probs=new_log_probs,
                ratio=ratio,
                ctx_logvar=ctx_logvar,
                gate=gate,
                teacher_for_targets=teacher_for_targets,
                decision_denom=decision_denom,
            )

            total = float(loss.detach().item())
            epoch_losses.append(total)
            self.last_losses = {
                "policy_loss": float(policy_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "entropy": float(entropy_bonus.detach().item()),
                "total_loss": total,
                "approx_kl": float(diagnostics["approx_kl"].detach().item()),
                "clip_frac": float(diagnostics["clip_frac"].detach().item()),
                "explained_variance": float(diagnostics["explained_variance"].detach().item()),
                "distill_loss": float(distill_loss.detach().item()),
                "distill_mse": float(distill_mse.detach().item()),
                "ctx_logvar_mean": float(diagnostics["ctx_logvar_mean"].detach().item()),
                "ctx_logvar_std": float(diagnostics["ctx_logvar_std"].detach().item()),
                "gate_mean": float(diagnostics["gate_mean"].detach().item()),
                "gate_std": float(diagnostics["gate_std"].detach().item()),
                "token_norm_mean": float(diagnostics["token_norm_mean"].detach().item()),
                "teacher_ctx_norm_mean": float(diagnostics["teacher_ctx_norm_mean"].detach().item()),
                "ctx_consistency_l2": float(diagnostics["ctx_consistency_l2"].detach().item()),
                "active_decisions": float(diagnostics["active_decisions"]),
                "active_agents_mean": float(diagnostics["active_agents_mean"].detach().item()),
            }
            self.loss_history.append(self.last_losses.copy())

        if epoch_losses:
            self.loss.append(float(sum(epoch_losses) / len(epoch_losses)))

        # PPO is on-policy; old trajectories are stale after an update.
        self.memory.clear()

    def set_eval_mode(self) -> None:
        self.actor_net.eval()
        self.critic.eval()
        self.target_teacher.eval()

    def set_train_mode(self) -> None:
        self.actor_net.train()
        self.critic.train()
        self.target_teacher.eval()


class PIMACAEC(PIMACBase):
    """
    PI-MAC interface for AEC/OpenURB style environments.

    This class exposes single-agent action selection and AEC cycle helpers while
    delegating all optimization/math to `PIMACBase`.
    """

    def act(
        self,
        state: np.ndarray,
        agent_index: Optional[object] = None,
    ) -> int:
        """
        Select one action for one currently acting agent.
        """
        if agent_index is None:
            agent_index = 0

        return self._act_single(
            state=state,
            actor_key=agent_index,
        )

    def store_episode(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        active_mask: np.ndarray,
        global_state: Optional[np.ndarray] = None,
    ) -> None:
        """
        Convenience helper for OpenURB one-step daily transitions.

        A one-step episode is represented by providing the same active mask for
        current/next slots and a zero next-observation placeholder.
        """
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

    def aec_begin_cycle(self, agent_ids_current: Sequence[object]) -> None:
        """Start one AEC joint cycle before recording per-agent turns."""
        self._aec_cycle = {
            "agent_ids": list(agent_ids_current),
            "obs": {},
            "actions": {},
            "rewards": {},
            "next_obs": {},
            "dones": {},
        }

    def aec_record(
        self,
        agent_id: object,
        obs: np.ndarray,
        action: Optional[int],
        reward: float,
        next_obs: Optional[np.ndarray],
        done: bool,
    ) -> None:
        """Record one agent turn inside the active AEC cycle."""
        if self._aec_cycle is None:
            raise RuntimeError("Call aec_begin_cycle before aec_record.")
        self._aec_cycle["obs"][agent_id] = obs
        self._aec_cycle["actions"][agent_id] = 0 if action is None else int(action)
        self._aec_cycle["rewards"][agent_id] = float(reward)
        if next_obs is None:
            next_obs = np.zeros_like(np.asarray(obs, dtype=np.float32))
        self._aec_cycle["next_obs"][agent_id] = next_obs
        self._aec_cycle["dones"][agent_id] = bool(done)

    def aec_end_cycle(
        self,
        global_state: Optional[np.ndarray] = None,
        next_global_state: Optional[np.ndarray] = None,
        done_all: bool = False,
    ) -> None:
        """Finalize and store a joint transition for the current AEC cycle."""
        if self._aec_cycle is None:
            raise RuntimeError("Call aec_begin_cycle before aec_end_cycle.")
        done = bool(done_all) or all(self._aec_cycle["dones"].values())
        self.store_transition(
            observations=self._aec_cycle["obs"],
            actions=self._aec_cycle["actions"],
            rewards=self._aec_cycle["rewards"],
            active_mask=None,
            global_state=global_state,
            next_observations=self._aec_cycle["next_obs"],
            next_active_mask=None,
            next_global_state=next_global_state,
            done=done,
            agent_ids=self._aec_cycle["agent_ids"],
        )
        self._aec_cycle = None


class PIMACParallel(PIMACBase):
    """
    PI-MAC interface for parallel multi-agent environments.
    """

    def act(
        self,
        state: np.ndarray,
        agent_index: Optional[object] = None,
    ) -> int:
        """
        Single-agent action helper for parallel setups.

        `agent_index` should be the environment agent id used as recurrent-state key.
        Prefer `act_parallel` for regular use.
        """
        if agent_index is None:
            raise ValueError("PIMACParallel.act requires agent_index. Prefer act_parallel for parallel environments.")
        return self._act_single(state=state, actor_key=agent_index)

    def act_parallel(self, obs_dict: dict) -> dict:
        """
        Convert parallel observations into one action per currently present agent.
        """
        actions = {}
        for agent_id in _sorted_agent_ids(obs_dict.keys()):
            actions[agent_id] = self._act_single(
                state=obs_dict[agent_id],
                actor_key=agent_id,
            )
        return actions

    @staticmethod
    def _resolve_parallel_done(done_dict) -> bool:
        """
        Resolve parallel `done` containers.

        Supported formats:
        - dict with `"__all__"` key
        - dict without `"__all__"`: interpreted as all(per_agent_done)
        - scalar bool-like value
        """
        if isinstance(done_dict, dict):
            if "__all__" in done_dict:
                return bool(done_dict.get("__all__", False))
            if not done_dict:
                return False
            return all(bool(flag) for flag in done_dict.values())
        return bool(done_dict)

    def store_parallel_step(
        self,
        obs_dict: dict,
        action_dict: dict,
        reward_dict: dict,
        next_obs_dict: dict,
        done_dict,
        active_mask_dict: Optional[dict] = None,
        next_active_mask_dict: Optional[dict] = None,
        global_state: Optional[np.ndarray] = None,
        next_global_state: Optional[np.ndarray] = None,
    ) -> None:
        """
        Store one parallel-env joint transition.

        `active_mask_dict` and `next_active_mask_dict` are optional; when omitted,
        active entries default to 1.0 for all ids present in the corresponding
        observation dict.
        """
        done = self._resolve_parallel_done(done_dict)
        self.store_transition(
            observations=obs_dict,
            actions=action_dict,
            rewards=reward_dict,
            active_mask=active_mask_dict,
            global_state=global_state,
            next_observations=next_obs_dict,
            next_active_mask=next_active_mask_dict,
            next_global_state=next_global_state,
            done=done,
        )


# Minimal compatibility alias for existing call sites that still import `PIMAC`.
PIMAC = PIMACAEC
