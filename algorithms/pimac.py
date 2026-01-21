"""
PI-MAC (pimac) implementation for OpenURB.

High-level idea (as implemented here):
- Base learner is VDN-style value factorization with recurrent per-agent Q networks.
- Each agent predicts a distilled context from its local observation history and uses FiLM conditioning.
- A permutation-invariant set teacher produces global tokens and per-agent teacher context for distillation.
- Training is CTDE; execution is decentralized (local observation only).
- Supports variable team sizes via padding and masks.
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
import torch.nn.functional as F
import torch.optim as optim

from baseline_models import BaseLearningModel

__all__ = ["PIMACAgentRNN", "SetTokenTeacher", "PIMAC"]


def _build_mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = int(in_dim)
    for h in hidden_sizes:
        layers.append(nn.Linear(last, int(h)))
        layers.append(nn.ReLU())
        last = int(h)
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


def _sorted_agent_ids(keys) -> list:
    return sorted(list(keys), key=lambda x: str(x))


class PIMACAgentRNN(nn.Module):
    """Per-agent recurrent Q-network with distilled context FiLM conditioning."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rnn_hidden_dim: int,
        num_hidden: int,
        widths: Sequence[int],
        ctx_dim: int,
        obs_index_dim: int = 3,
    ):
        super().__init__()
        assert len(widths) == (int(num_hidden) + 1), "PI-MAC widths and number of layers mismatch!"

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.ctx_dim = int(ctx_dim)

        self.input_layer = nn.Linear(int(obs_dim), int(widths[0]))
        self.hidden_layers = nn.ModuleList(
            nn.Linear(int(widths[idx]), int(widths[idx + 1])) for idx in range(int(num_hidden))
        )
        self.rnn = nn.GRU(input_size=int(widths[-1]), hidden_size=int(rnn_hidden_dim), batch_first=True)

        # Distilled context heads from recurrent state.
        self.ctx_head = nn.Linear(int(rnn_hidden_dim), int(ctx_dim))
        self.logvar_head = nn.Linear(int(rnn_hidden_dim), int(ctx_dim))
        self.film = nn.Linear(int(ctx_dim), int(2 * rnn_hidden_dim))

        # Residual conditioning from observation indices (static role/task features).
        effective_index_dim = min(int(obs_index_dim), int(obs_dim))
        self.obs_index_dim = max(0, int(effective_index_dim))
        self.index_proj = None
        if self.obs_index_dim > 0:
            self.index_proj = nn.Linear(int(self.obs_index_dim), int(rnn_hidden_dim))

        self.out = nn.Linear(int(rnn_hidden_dim), int(action_dim))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return x

    def forward(self, obs_seq: torch.Tensor, h0: Optional[torch.Tensor] = None):
        # obs_seq: [B, T, obs_dim]
        b, t, d = obs_seq.shape
        x = self._encode(obs_seq.reshape(b * t, d)).reshape(b, t, -1)
        out, hn = self.rnn(x, h0)  # out: [B, T, H]

        ctx_pred = self.ctx_head(out)  # [B, T, ctx_dim]
        logvar_pred = self.logvar_head(out)  # [B, T, ctx_dim]

        h = out
        if self.index_proj is not None:
            idx = obs_seq[:, :, -self.obs_index_dim :]  # [B, T, K]
            h = h + self.index_proj(idx)

        gamma_beta = self.film(ctx_pred)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gate = torch.sigmoid(-logvar_pred.mean(dim=-1, keepdim=True))
        h_film = (1.0 + gate * gamma) * h + gate * beta

        q = self.out(h_film)  # [B, T, A]
        return q, hn, ctx_pred, logvar_pred


class SetTokenTeacher(nn.Module):
    """Permutation-invariant set teacher producing global tokens and per-agent context."""

    def __init__(
        self,
        in_dim: int,
        num_tokens: int,
        tok_dim: int,
        ctx_dim: int,
        action_dim: Optional[int] = None,
        emb_dim: int = 128,
        hidden_sizes: Sequence[int] = (128, 128),
        attn_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_tokens = int(num_tokens)
        self.tok_dim = int(tok_dim)
        self.ctx_dim = int(ctx_dim)
        self.emb_dim = int(emb_dim)
        self.attn_dim = int(attn_dim) if attn_dim is not None else int(tok_dim)

        self.agent_mlp = _build_mlp(self.in_dim, hidden_sizes, self.emb_dim)
        self.token_mlp = _build_mlp(self.emb_dim, hidden_sizes, self.num_tokens * self.tok_dim)

        self.q_proj = nn.Linear(self.emb_dim, self.attn_dim)
        self.k_proj = nn.Linear(self.tok_dim, self.attn_dim)
        self.v_proj = nn.Linear(self.tok_dim, self.attn_dim)
        self.out_proj = nn.Linear(self.attn_dim, self.ctx_dim)

        self.action_dim = int(action_dim) if action_dim is not None else None
        self.action_hist_head = None
        if self.action_dim is not None:
            self.action_hist_head = _build_mlp(self.tok_dim, hidden_sizes, self.action_dim)

    def _masked_mean(self, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1)
        denom = weights.sum(dim=2).clamp(min=1.0)
        return (u * weights).sum(dim=2) / denom

    def forward(self, x: torch.Tensor, active_mask: torch.Tensor, pool_mask: torch.Tensor):
        # x: [B, T, N, D]
        b, t, n, d = x.shape
        u = self.agent_mlp(x.reshape(b * t * n, d)).reshape(b, t, n, -1)

        u_queries = u * active_mask.unsqueeze(-1)
        pooled = self._masked_mean(u, pool_mask)

        tok = self.token_mlp(pooled.reshape(b * t, -1)).reshape(b, t, self.num_tokens, self.tok_dim)

        q = self.q_proj(u_queries)  # [B, T, N, A]
        k = self.k_proj(tok)  # [B, T, K, A]
        v = self.v_proj(tok)  # [B, T, K, A]

        scores = (q.unsqueeze(3) * k.unsqueeze(2)).sum(dim=-1) / math.sqrt(self.attn_dim)
        attn = F.softmax(scores, dim=-1)
        ctx = (attn.unsqueeze(-1) * v.unsqueeze(2)).sum(dim=3)
        ctx = self.out_proj(ctx)
        return tok, ctx

    def predict_action_hist(self, tokens: torch.Tensor) -> Optional[torch.Tensor]:
        if self.action_hist_head is None:
            return None
        b, t, _, _ = tokens.shape
        tok_mean = tokens.mean(dim=2)
        logits = self.action_hist_head(tok_mean.reshape(b * t, -1)).reshape(b, t, self.action_dim)
        return logits


class PIMAC(BaseLearningModel):
    """PI-MAC: VDN + set-based teacher distillation for scalable team composition context."""

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
        teacher_lr: float = 3e-4,
        num_epochs: int = 1,
        num_hidden: int = 2,
        widths: Sequence[int] = (128, 128, 128),
        rnn_hidden_dim: int = 64,
        ctx_dim: int = 16,
        num_tokens: int = 8,
        tok_dim: Optional[int] = None,
        teacher_emb_dim: int = 128,
        teacher_hidden_sizes: Sequence[int] = (128, 128),
        teacher_attn_dim: Optional[int] = None,
        teacher_drop_prob: float = 0.5,
        distill_weight: float = 1.0,
        teacher_aux_weight: float = 1.0,
        token_smooth_weight: float = 0.0,
        teacher_use_actions: bool = True,
        obs_index_dim: int = 3,
        max_grad_norm: float = 10.0,
        gamma: float = 0.99,
        target_update_every: int = 200,
        double_q: bool = True,
        tau: float = 1.0,
        share_parameters: bool = True,
        q_tot_clip: Optional[float] = None,
        use_huber_loss: bool = True,
        normalize_by_active: bool = True,
        # Sub-team augmentation.
        subteam_samples: int = 0,
        subteam_keep_prob: float = 0.75,
        subteam_td_weight: float = 0.5,
    ):
        super().__init__()
        self.device = device
        self.obs_size = int(state_size)
        self.action_space_size = int(action_space_size)
        self.num_agents = int(num_agents)

        self.share_parameters = bool(share_parameters)

        self.temperature = float(temp_init)
        self.temp_decay = float(temp_decay)
        self.temp_min = float(temp_min)

        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.max_grad_norm = float(max_grad_norm)

        self.gamma = float(gamma)
        self.target_update_every = max(1, int(target_update_every))
        self.double_q = bool(double_q)
        self.tau = float(tau)
        self.q_tot_clip = float(q_tot_clip) if q_tot_clip is not None else None
        self.use_huber_loss = bool(use_huber_loss)
        self.normalize_by_active = bool(normalize_by_active)

        self.subteam_samples = max(0, int(subteam_samples))
        self.subteam_keep_prob = float(subteam_keep_prob)
        self.subteam_td_weight = float(subteam_td_weight)

        self.ctx_dim = int(ctx_dim)
        self.num_tokens = int(num_tokens)
        self.tok_dim = int(tok_dim) if tok_dim is not None else int(ctx_dim)
        self.teacher_drop_prob = float(teacher_drop_prob)
        self.distill_weight = float(distill_weight)
        self.teacher_aux_weight = float(teacher_aux_weight)
        self.token_smooth_weight = float(token_smooth_weight)
        self.teacher_use_actions = bool(teacher_use_actions)

        self._learn_steps = 0

        if self.share_parameters:
            self.agent_net = PIMACAgentRNN(
                self.obs_size,
                self.action_space_size,
                rnn_hidden_dim=int(rnn_hidden_dim),
                num_hidden=int(num_hidden),
                widths=widths,
                ctx_dim=int(ctx_dim),
                obs_index_dim=int(obs_index_dim),
            ).to(self.device)
            self.target_agent_net = copy.deepcopy(self.agent_net).to(self.device)
            self.target_agent_net.eval()
            self.agent_nets = None
            self.target_agent_nets = None
        else:
            self.agent_net = None
            self.target_agent_net = None
            self.agent_nets = nn.ModuleList(
                [
                    PIMACAgentRNN(
                        self.obs_size,
                        self.action_space_size,
                        rnn_hidden_dim=int(rnn_hidden_dim),
                        num_hidden=int(num_hidden),
                        widths=widths,
                        ctx_dim=int(ctx_dim),
                        obs_index_dim=int(obs_index_dim),
                    ).to(self.device)
                    for _ in range(self.num_agents)
                ]
            )
            self.target_agent_nets = copy.deepcopy(self.agent_nets).to(self.device)
            self.target_agent_nets.eval()

        teacher_in_dim = self.obs_size + (self.action_space_size if self.teacher_use_actions else 0)
        self.teacher = SetTokenTeacher(
            in_dim=teacher_in_dim,
            num_tokens=self.num_tokens,
            tok_dim=self.tok_dim,
            ctx_dim=self.ctx_dim,
            action_dim=self.action_space_size,
            emb_dim=int(teacher_emb_dim),
            hidden_sizes=teacher_hidden_sizes,
            attn_dim=teacher_attn_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(list(self._agent_parameters()), lr=float(lr))
        self.teacher_optimizer = optim.Adam(self.teacher.parameters(), lr=float(teacher_lr))

        self.memory = deque(maxlen=int(buffer_size))
        self._episode_steps = []
        self._aec_cycle = None

        self.loss: list[float] = []
        self.last_losses: dict[str, float] = {}
        self._inference_hidden: dict[object, torch.Tensor] = {}

    def _agent_parameters(self):
        if self.share_parameters:
            return self.agent_net.parameters()
        return self.agent_nets.parameters()

    def reset_episode(self) -> None:
        self._inference_hidden = {}
        self._aec_cycle = None

    def _get_h0(self, agent_key: object, hidden_dim: int) -> torch.Tensor:
        h = self._inference_hidden.get(agent_key)
        if h is None:
            h = torch.zeros(1, 1, int(hidden_dim), device=self.device)
        return h

    def _set_h(self, agent_key: object, h: torch.Tensor) -> None:
        self._inference_hidden[agent_key] = h.detach()

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
        agent_index: Optional[object] = None,
    ) -> int:
        if agent_index is None:
            agent_index = 0

        obs_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, 1, -1)
        if self.share_parameters:
            hidden_dim = self.agent_net.rnn.hidden_size
            h0 = self._get_h0(agent_index, hidden_dim)
            q_seq, hn, _, _ = self.agent_net(obs_t, h0)
        else:
            if not isinstance(agent_index, int):
                raise ValueError("agent_index must be an int when share_parameters=False.")
            if agent_index < 0 or agent_index >= self.num_agents:
                raise ValueError(f"agent_index {agent_index} is out of range for {self.num_agents} agents.")
            net = self.agent_nets[agent_index]
            hidden_dim = net.rnn.hidden_size
            h0 = self._get_h0(agent_index, hidden_dim)
            q_seq, hn, _, _ = net(obs_t, h0)

        self._set_h(agent_index, hn)
        q_values = q_seq.squeeze(0).squeeze(0)

        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=q_values.device)
            if mask.shape[0] == q_values.shape[0] and mask.any():
                q_values = q_values.masked_fill(~mask, -1e9)
            elif not mask.any():
                return self._random_action(action_mask)

        return self._boltzmann_action(q_values)

    # Parallel usage example:
    # actions = pimac.act_parallel(obs_dict, action_mask_dict)
    # pimac.store_parallel_step(obs_dict, action_dict, reward_dict, next_obs_dict, done_dict)
    def act_parallel(self, obs_dict: dict, action_mask_dict: Optional[dict] = None) -> dict:
        """Parallel API helper: obs_dict -> action_dict."""
        actions = {}
        for agent_id in _sorted_agent_ids(obs_dict.keys()):
            action_mask = None if action_mask_dict is None else action_mask_dict.get(agent_id)
            actions[agent_id] = self.act(obs_dict[agent_id], action_mask=action_mask, agent_index=agent_id)
        return actions

    def store_parallel_step(
        self,
        obs_dict: dict,
        action_dict: dict,
        reward_dict: dict,
        next_obs_dict: dict,
        done_dict: dict,
        action_mask_dict: Optional[dict] = None,
        next_action_mask_dict: Optional[dict] = None,
        global_state: Optional[np.ndarray] = None,
        next_global_state: Optional[np.ndarray] = None,
        ) -> None:
        """Parallel API helper: store a joint transition once per env step."""
        done = bool(done_dict.get("__all__", False)) if isinstance(done_dict, dict) else bool(done_dict)
        self.store_transition(
            observations=obs_dict,
            actions=action_dict,
            rewards=reward_dict,
            active_mask=None,
            global_state=global_state,
            next_observations=next_obs_dict,
            next_active_mask=None,
            next_global_state=next_global_state,
            done=done,
            action_masks=action_mask_dict,
            next_action_masks=next_action_mask_dict,
        )

    # AEC usage example:
    # pimac.aec_begin_cycle(env.possible_agents)
    # for agent_id in env.agent_iter(): pimac.aec_record(...)
    # pimac.aec_end_cycle(done_all=done)
    def aec_begin_cycle(self, agent_ids_current: Sequence[object]) -> None:
        """AEC helper: begin a new joint step cycle before per-agent turns."""
        self._aec_cycle = {
            "agent_ids": list(agent_ids_current),
            "obs": {},
            "actions": {},
            "rewards": {},
            "next_obs": {},
            "dones": {},
            "action_masks": {},
            "next_action_masks": {},
        }

    def aec_record(
        self,
        agent_id: object,
        obs: np.ndarray,
        action: Optional[int],
        reward: float,
        next_obs: Optional[np.ndarray],
        done: bool,
        action_mask: Optional[np.ndarray] = None,
        next_action_mask: Optional[np.ndarray] = None,
    ) -> None:
        """AEC helper: record per-agent data within the current joint cycle."""
        if self._aec_cycle is None:
            raise RuntimeError("Call aec_begin_cycle before aec_record.")
        self._aec_cycle["obs"][agent_id] = obs
        self._aec_cycle["actions"][agent_id] = 0 if action is None else int(action)
        self._aec_cycle["rewards"][agent_id] = float(reward)
        if next_obs is None:
            next_obs = np.zeros_like(np.asarray(obs, dtype=np.float32))
        self._aec_cycle["next_obs"][agent_id] = next_obs
        self._aec_cycle["dones"][agent_id] = bool(done)
        if action_mask is not None:
            self._aec_cycle["action_masks"][agent_id] = action_mask
        if next_action_mask is not None:
            self._aec_cycle["next_action_masks"][agent_id] = next_action_mask

    def aec_end_cycle(
        self,
        global_state: Optional[np.ndarray] = None,
        next_global_state: Optional[np.ndarray] = None,
        done_all: bool = False,
    ) -> None:
        """AEC helper: finalize the joint transition after all agents acted in the cycle."""
        if self._aec_cycle is None:
            raise RuntimeError("Call aec_begin_cycle before aec_end_cycle.")
        done = bool(done_all) or all(self._aec_cycle["dones"].values())
        action_masks = self._aec_cycle["action_masks"] or None
        next_action_masks = self._aec_cycle["next_action_masks"] or None
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
            action_masks=action_masks,
            next_action_masks=next_action_masks,
            agent_ids=self._aec_cycle["agent_ids"],
        )
        self._aec_cycle = None

    def _boltzmann_action(self, q_values: torch.Tensor) -> int:
        temp = float(self.temperature)
        if temp <= 0.0:
            return int(torch.argmax(q_values).item())
        logits = q_values / temp
        logits = logits - torch.max(logits)
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def _ensure_state(self, state: Optional[np.ndarray]) -> np.ndarray:
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
        action_masks,
        next_action_masks,
        agent_ids: Optional[Sequence[object]],
    ) -> dict:
        obs_is_dict = isinstance(observations, dict)
        if obs_is_dict:
            if agent_ids is None:
                agent_ids = _sorted_agent_ids(observations.keys())
            obs_arr = np.stack(
                [np.asarray(observations[aid], dtype=np.float32) for aid in agent_ids], axis=0
            )
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
            next_obs_arr = np.stack(
                [np.asarray(next_observations[aid], dtype=np.float32) for aid in next_agent_ids], axis=0
            )
        else:
            next_obs_arr = np.asarray(next_observations, dtype=np.float32)
            if next_obs_arr.ndim == 1:
                next_obs_arr = next_obs_arr.reshape(1, -1)
            next_agent_ids = list(range(next_obs_arr.shape[0]))

        if next_active_mask is None:
            next_active_arr = np.ones(len(next_agent_ids), dtype=np.float32)
        elif isinstance(next_active_mask, dict):
            next_active_arr = np.asarray([next_active_mask.get(aid, 0.0) for aid in next_agent_ids], dtype=np.float32)
        else:
            next_active_arr = np.asarray(next_active_mask, dtype=np.float32)
            if next_active_arr.ndim == 0:
                next_active_arr = next_active_arr.reshape(1)

        if action_masks is None:
            action_masks_arr = None
        elif isinstance(action_masks, dict):
            action_masks_arr = np.stack(
                [
                    np.asarray(
                        action_masks.get(aid, np.zeros(self.action_space_size, dtype=np.int8)), dtype=np.int8
                    )
                    for aid in agent_ids
                ],
                axis=0,
            )
        else:
            action_masks_arr = np.asarray(action_masks, dtype=np.int8)

        if next_action_masks is None:
            next_action_masks_arr = None
        elif isinstance(next_action_masks, dict):
            next_action_masks_arr = np.stack(
                [
                    np.asarray(
                        next_action_masks.get(aid, np.zeros(self.action_space_size, dtype=np.int8)), dtype=np.int8
                    )
                    for aid in next_agent_ids
                ],
                axis=0,
            )
        else:
            next_action_masks_arr = np.asarray(next_action_masks, dtype=np.int8)

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
            "action_masks": action_masks_arr,
            "next_action_masks": next_action_masks_arr,
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
        action_masks: Optional[np.ndarray] = None,
        next_action_masks: Optional[np.ndarray] = None,
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
                action_masks,
                next_action_masks,
                agent_ids,
            )
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
        state = np.stack([s["state"] for s in steps], axis=0)
        next_state = np.stack([s["next_state"] for s in steps], axis=0)
        done = np.asarray([s["done"] for s in steps], dtype=np.float32)

        action_masks = None
        next_action_masks = None
        if all(s.get("action_masks") is not None for s in steps):
            action_dim = steps[0]["action_masks"].shape[-1]
            action_masks = np.zeros((t, n, action_dim), dtype=np.int8)
        if all(s.get("next_action_masks") is not None for s in steps):
            action_dim = steps[0]["next_action_masks"].shape[-1]
            next_action_masks = np.zeros((t, n, action_dim), dtype=np.int8)

        for ti, step in enumerate(steps):
            for idx, aid in enumerate(step["agent_ids"]):
                slot = roster_index[aid]
                obs[ti, slot] = step["obs"][idx]
                actions[ti, slot] = step["actions"][idx]
                rewards[ti, slot] = step["rewards"][idx]
                active_mask[ti, slot] = step["active_mask"][idx]
                if action_masks is not None:
                    action_masks[ti, slot] = step["action_masks"][idx]
            for idx, aid in enumerate(step["next_agent_ids"]):
                slot = roster_index[aid]
                next_obs[ti, slot] = step["next_obs"][idx]
                next_active_mask[ti, slot] = step["next_active_mask"][idx]
                if next_action_masks is not None:
                    next_action_masks[ti, slot] = step["next_action_masks"][idx]

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
            "N": obs.shape[1],
        }

    def _masked_argmax(self, q_values: torch.Tensor, action_masks: Optional[torch.Tensor]) -> torch.Tensor:
        if action_masks is None:
            return torch.argmax(q_values, dim=-1)
        mask = action_masks.to(dtype=torch.bool, device=q_values.device)
        if mask.shape != q_values.shape:
            return torch.argmax(q_values, dim=-1)
        masked = q_values.masked_fill(~mask, -1e9)
        no_valid = ~mask.any(dim=-1)
        argmax = torch.argmax(masked, dim=-1)
        return torch.where(no_valid, torch.zeros_like(argmax), argmax)

    def _mix_q_tot(self, chosen_q: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        masked_q = chosen_q * active_mask
        q_sum = masked_q.sum(dim=2)
        if not self.normalize_by_active:
            return q_sum
        counts = active_mask.sum(dim=2).clamp(min=1.0)
        return q_sum / counts

    def _update_targets(self) -> None:
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
            obs_bn = obs.permute(0, 2, 1, 3).reshape(b * n, t, d)
            q_bn, _, _, _ = net(obs_bn, None)
            return q_bn.reshape(b, n, t, -1).permute(0, 2, 1, 3)

        qs = []
        for idx in range(n):
            q_i, _, _, _ = net[idx](obs[:, :, idx, :], None)
            qs.append(q_i.unsqueeze(2))
        return torch.cat(qs, dim=2)

    def _agent_forward(self, obs: torch.Tensor, net, share: bool):
        # Returns Q-values + predicted context and log-variance.
        b, t, n, d = obs.shape
        if share:
            obs_bn = obs.permute(0, 2, 1, 3).reshape(b * n, t, d)
            q_bn, _, ctx_bn, logvar_bn = net(obs_bn, None)
            q = q_bn.reshape(b, n, t, -1).permute(0, 2, 1, 3)
            ctx = ctx_bn.reshape(b, n, t, -1).permute(0, 2, 1, 3)
            logvar = logvar_bn.reshape(b, n, t, -1).permute(0, 2, 1, 3)
            return q, ctx, logvar

        qs, ctxs, logvars = [], [], []
        for idx in range(n):
            q_i, _, ctx_i, logvar_i = net[idx](obs[:, :, idx, :], None)
            qs.append(q_i.unsqueeze(2))
            ctxs.append(ctx_i.unsqueeze(2))
            logvars.append(logvar_i.unsqueeze(2))
        return torch.cat(qs, dim=2), torch.cat(ctxs, dim=2), torch.cat(logvars, dim=2)

    def _sample_subteam_mask(self, active_mask: torch.Tensor) -> torch.Tensor:
        # active_mask: [B, T, N] in {0,1} -> sub_mask: [B, T, N] in {0,1}
        if self.subteam_keep_prob >= 1.0:
            return torch.ones_like(active_mask)
        keep = (torch.rand_like(active_mask) < self.subteam_keep_prob).to(dtype=active_mask.dtype)
        sub = active_mask * keep

        # Ensure at least one active agent per (B,T) when there exists any active agent.
        b, t, _ = active_mask.shape
        active_counts = active_mask.sum(dim=2)
        sub_counts = sub.sum(dim=2)
        need_fix = (active_counts > 0.0) & (sub_counts == 0.0)
        if not need_fix.any():
            return sub

        for bi, ti in need_fix.nonzero(as_tuple=False):
            candidates = torch.nonzero(active_mask[bi, ti] > 0.0, as_tuple=False).squeeze(-1)
            chosen = candidates[torch.randint(0, candidates.shape[0], (1,), device=active_mask.device)]
            sub[bi, ti, chosen] = 1.0
        return sub

    def _teacher_pool_mask(self, active_mask: torch.Tensor) -> torch.Tensor:
        if self.teacher_drop_prob <= 0.0:
            return active_mask
        keep = (torch.rand_like(active_mask) > self.teacher_drop_prob).to(dtype=active_mask.dtype)
        pool_mask = active_mask * keep
        need_full = (pool_mask.sum(dim=2) == 0.0) & (active_mask.sum(dim=2) > 0.0)
        if need_full.any():
            pool_mask = torch.where(need_full.unsqueeze(-1), active_mask, pool_mask)
        return pool_mask

    def _distill_loss(
        self,
        ctx_pred: torch.Tensor,
        logvar_pred: torch.Tensor,
        teacher_ctx: torch.Tensor,
        active_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        logvar = torch.clamp(logvar_pred, min=-10.0, max=10.0)
        err = (ctx_pred - teacher_ctx.detach()).pow(2)
        per = (torch.exp(-logvar) * err + logvar).mean(dim=-1)
        weights = active_mask * time_mask.unsqueeze(-1)
        denom = weights.sum().clamp(min=1.0)
        return (per * weights).sum() / denom

    def _action_histogram(self, actions: torch.Tensor, active_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        one_hot = F.one_hot(actions.clamp(min=0), num_classes=self.action_space_size).to(dtype=torch.float32)
        counts = (one_hot * active_mask.unsqueeze(-1)).sum(dim=2)
        active_counts = active_mask.sum(dim=2)
        denom = active_counts.clamp(min=1.0)
        hist = counts / denom.unsqueeze(-1)
        return hist, active_counts

    def learn(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        step_losses = []
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            max_t = max(int(ep["T"]) for ep in batch)
            max_n = max(int(ep["N"]) for ep in batch)

            if not self.share_parameters and max_n > self.num_agents:
                raise ValueError("max_n exceeds the number of agent networks; enable share_parameters.")

            def pad_time(x, pad_value=0.0):
                t = x.shape[0]
                if t == max_t:
                    return x
                pad_shape = (max_t - t,) + x.shape[1:]
                pad = np.full(pad_shape, pad_value, dtype=x.dtype)
                return np.concatenate([x, pad], axis=0)

            def pad_time_agents(x, pad_value=0.0):
                t, n = x.shape[0], x.shape[1]
                if t == max_t and n == max_n:
                    return x
                pad_width = [(0, max_t - t), (0, max_n - n)] + [(0, 0)] * (x.ndim - 2)
                return np.pad(x, pad_width, mode="constant", constant_values=pad_value)

            obs = torch.tensor(
                np.stack([pad_time_agents(ep["obs"]) for ep in batch]), device=self.device, dtype=torch.float32
            )
            actions = torch.tensor(
                np.stack([pad_time_agents(ep["actions"], pad_value=0) for ep in batch]),
                device=self.device,
                dtype=torch.int64,
            )
            rewards = torch.tensor(
                np.stack([pad_time_agents(ep["rewards"], pad_value=0.0) for ep in batch]),
                device=self.device,
                dtype=torch.float32,
            )
            active_mask = torch.tensor(
                np.stack([pad_time_agents(ep["active_mask"], pad_value=0.0) for ep in batch]),
                device=self.device,
                dtype=torch.float32,
            )

            next_obs = torch.tensor(
                np.stack([pad_time_agents(ep["next_obs"]) for ep in batch]), device=self.device, dtype=torch.float32
            )
            next_active_mask = torch.tensor(
                np.stack([pad_time_agents(ep["next_active_mask"], pad_value=0.0) for ep in batch]),
                device=self.device,
                dtype=torch.float32,
            )
            dones = torch.tensor(
                np.stack([pad_time(ep["done"].reshape(-1, 1)) for ep in batch]),
                device=self.device,
                dtype=torch.float32,
            ).squeeze(-1)

            lengths = torch.tensor([int(ep["T"]) for ep in batch], device=self.device, dtype=torch.int64)
            time_mask = (
                torch.arange(max_t, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            ).to(dtype=torch.float32)

            agent_pad_mask = torch.tensor(
                np.stack(
                    [
                        np.concatenate(
                            [np.ones(ep["N"], dtype=np.float32), np.zeros(max_n - ep["N"], dtype=np.float32)]
                        )
                        for ep in batch
                    ]
                ),
                device=self.device,
                dtype=torch.float32,
            )

            action_masks = None
            next_action_masks = None
            if all(ep.get("action_masks") is not None for ep in batch):
                action_masks = torch.tensor(
                    np.stack([pad_time_agents(ep["action_masks"], pad_value=0) for ep in batch]),
                    device=self.device,
                    dtype=torch.int8,
                )
            if all(ep.get("next_action_masks") is not None for ep in batch):
                next_action_masks = torch.tensor(
                    np.stack([pad_time_agents(ep["next_action_masks"], pad_value=0) for ep in batch]),
                    device=self.device,
                    dtype=torch.int8,
                )

            combined_mask = active_mask * agent_pad_mask.unsqueeze(1)
            next_combined_mask = next_active_mask * agent_pad_mask.unsqueeze(1)

            if self.share_parameters:
                q_all, ctx_pred, logvar_pred = self._agent_forward(obs, self.agent_net, share=True)
                next_q_online = self._agent_q_values(next_obs, self.agent_net, share=True)
                with torch.no_grad():
                    next_q_target = self._agent_q_values(next_obs, self.target_agent_net, share=True)
            else:
                q_all, ctx_pred, logvar_pred = self._agent_forward(obs, self.agent_nets, share=False)
                next_q_online = self._agent_q_values(next_obs, self.agent_nets, share=False)
                with torch.no_grad():
                    next_q_target = self._agent_q_values(next_obs, self.target_agent_nets, share=False)

            safe_actions = actions.clone()
            safe_actions[combined_mask == 0] = 0
            chosen_q = torch.gather(q_all, 3, safe_actions.unsqueeze(-1)).squeeze(-1)
            chosen_q = chosen_q * combined_mask

            q_tot = self._mix_q_tot(chosen_q, combined_mask)
            if self.q_tot_clip is not None:
                q_tot = torch.clamp(q_tot, -self.q_tot_clip, self.q_tot_clip)

            active_counts = combined_mask.sum(dim=2).clamp(min=1.0)
            team_rewards = (rewards * combined_mask).sum(dim=2) / active_counts

            with torch.no_grad():
                if self.double_q:
                    next_actions = self._masked_argmax(next_q_online, next_action_masks)
                else:
                    next_actions = self._masked_argmax(next_q_target, next_action_masks)
                safe_next_actions = next_actions.clone()
                safe_next_actions[next_combined_mask == 0] = 0
                next_chosen_q = torch.gather(next_q_target, 3, safe_next_actions.unsqueeze(-1)).squeeze(-1)
                next_chosen_q = next_chosen_q * next_combined_mask

                q_tot_next = self._mix_q_tot(next_chosen_q, next_combined_mask)
                if self.q_tot_clip is not None:
                    q_tot_next = torch.clamp(q_tot_next, -self.q_tot_clip, self.q_tot_clip)

                targets = team_rewards + (1.0 - dones) * self.gamma * q_tot_next
                if self.q_tot_clip is not None:
                    targets = torch.clamp(targets, -self.q_tot_clip, self.q_tot_clip)

            if self.use_huber_loss:
                td = F.smooth_l1_loss(q_tot, targets, reduction="none")
            else:
                td = F.mse_loss(q_tot, targets, reduction="none")
            td_loss = (td * time_mask).sum() / time_mask.sum().clamp(min=1.0)

            subteam_td_loss = torch.tensor(0.0, device=self.device)
            if self.subteam_samples > 0 and self.subteam_td_weight > 0.0:
                acc = 0.0
                for _k in range(self.subteam_samples):
                    sub_mask = self._sample_subteam_mask(combined_mask)
                    sub_active = sub_mask
                    sub_counts = sub_active.sum(dim=2).clamp(min=1.0)
                    sub_rewards = (rewards * sub_active).sum(dim=2) / sub_counts

                    chosen_q_sub = chosen_q * sub_mask
                    q_tot_sub = self._mix_q_tot(chosen_q_sub, sub_active)
                    if self.q_tot_clip is not None:
                        q_tot_sub = torch.clamp(q_tot_sub, -self.q_tot_clip, self.q_tot_clip)

                    with torch.no_grad():
                        sub_next_active = next_combined_mask * sub_mask
                        next_chosen_q_sub = next_chosen_q * sub_mask
                        q_tot_next_sub = self._mix_q_tot(next_chosen_q_sub, sub_next_active)
                        if self.q_tot_clip is not None:
                            q_tot_next_sub = torch.clamp(q_tot_next_sub, -self.q_tot_clip, self.q_tot_clip)
                        targets_sub = sub_rewards + (1.0 - dones) * self.gamma * q_tot_next_sub
                        if self.q_tot_clip is not None:
                            targets_sub = torch.clamp(targets_sub, -self.q_tot_clip, self.q_tot_clip)

                    if self.use_huber_loss:
                        td_sub = F.smooth_l1_loss(q_tot_sub, targets_sub, reduction="none")
                    else:
                        td_sub = F.mse_loss(q_tot_sub, targets_sub, reduction="none")
                    acc = acc + (td_sub * time_mask).sum() / time_mask.sum().clamp(min=1.0)
                subteam_td_loss = acc / float(self.subteam_samples)

            distill_loss = torch.tensor(0.0, device=self.device)
            teacher_aux_loss = torch.tensor(0.0, device=self.device)
            teacher_smooth = torch.tensor(0.0, device=self.device)

            need_teacher = (
                self.distill_weight > 0.0
                or self.teacher_aux_weight > 0.0
                or self.token_smooth_weight > 0.0
            )
            if need_teacher:
                teacher_in = obs
                if self.teacher_use_actions:
                    action_one_hot = F.one_hot(safe_actions, num_classes=self.action_space_size).to(dtype=torch.float32)
                    teacher_in = torch.cat([teacher_in, action_one_hot], dim=-1)

                pool_mask = self._teacher_pool_mask(combined_mask)
                tokens, teacher_ctx = self.teacher(teacher_in, active_mask=combined_mask, pool_mask=pool_mask)

                # Teacher outputs are detached for distillation; execution uses local ctx_pred only.
                distill_loss = self._distill_loss(ctx_pred, logvar_pred, teacher_ctx, combined_mask, time_mask)

                if self.teacher_aux_weight > 0.0:
                    hist_logits = self.teacher.predict_action_hist(tokens)
                    if hist_logits is not None:
                        pred_hist = F.softmax(hist_logits, dim=-1)
                        target_hist, active_counts = self._action_histogram(actions, combined_mask)
                        mse = (pred_hist - target_hist).pow(2).mean(dim=-1)
                        hist_weights = time_mask * (active_counts > 0.0).to(dtype=time_mask.dtype)
                        teacher_aux_loss = (mse * hist_weights).sum() / hist_weights.sum().clamp(min=1.0)

                if self.token_smooth_weight > 0.0 and tokens.shape[1] > 1:
                    diff = tokens[:, 1:] - tokens[:, :-1]
                    smooth = diff.pow(2).mean(dim=(2, 3))
                    smooth_mask = time_mask[:, 1:] * time_mask[:, :-1]
                    teacher_smooth = (smooth * smooth_mask).sum() / smooth_mask.sum().clamp(min=1.0)

            agent_loss = td_loss
            if self.subteam_td_weight > 0.0 and self.subteam_samples > 0:
                agent_loss = agent_loss + self.subteam_td_weight * subteam_td_loss
            if self.distill_weight > 0.0:
                agent_loss = agent_loss + self.distill_weight * distill_loss

            teacher_loss = (
                self.teacher_aux_weight * teacher_aux_loss + self.token_smooth_weight * teacher_smooth
            )

            self.optimizer.zero_grad()
            agent_loss.backward()
            nn.utils.clip_grad_norm_(list(self._agent_parameters()), max_norm=self.max_grad_norm)
            self.optimizer.step()

            if need_teacher and teacher_loss.item() > 0.0:
                self.teacher_optimizer.zero_grad()
                teacher_loss.backward()
                nn.utils.clip_grad_norm_(list(self.teacher.parameters()), max_norm=self.max_grad_norm)
                self.teacher_optimizer.step()

            self._learn_steps += 1
            if self._learn_steps % self.target_update_every == 0:
                self._update_targets()

            total = float((agent_loss.detach() + teacher_loss.detach()).item())
            step_losses.append(total)
            self.last_losses = {
                "td": float(td_loss.detach().item()),
                "subteam_td": float(subteam_td_loss.detach().item()),
                "distill": float(distill_loss.detach().item()),
                "teacher_aux": float(teacher_aux_loss.detach().item()),
                "teacher_smooth": float(teacher_smooth.detach().item()),
                "agent_loss": float(agent_loss.detach().item()),
                "teacher_loss": float(teacher_loss.detach().item()),
            }

        self.loss.append(float(sum(step_losses) / len(step_losses)))
        self.decay_temperature()

    def decay_temperature(self) -> None:
        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)

    def set_eval_mode(self) -> None:
        if self.share_parameters:
            self.agent_net.eval()
            self.target_agent_net.eval()
        else:
            self.agent_nets.eval()
            self.target_agent_nets.eval()
        self.teacher.eval()

    def set_train_mode(self) -> None:
        if self.share_parameters:
            self.agent_net.train()
        else:
            self.agent_nets.train()
        self.teacher.train()
