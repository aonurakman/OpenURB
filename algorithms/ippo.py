"""
IPPO (Independent PPO) implementation.

This is a standard clipped PPO actor-critic:
- Policy + value function (here: a shared GRU trunk to support partial observability).
- GAE(λ) for advantage estimation.
- Clipped surrogate objective with entropy bonus and value loss.

Compatibility with existing OpenURB scripts:
- `act(state)` stores last transition context.
- `push(reward)` finalizes the last stored transition (done=True by default).
- `learn()` performs PPO updates and clears the on-policy buffer.
- `policy_net` attribute exists (for `.eval()` in testing phase).
- `deterministic` flag controls greedy vs sampling actions.
"""

from __future__ import annotations

from collections import deque
import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from baseline_models import BaseLearningModel

__all__ = ["ActorCriticRNN", "PPO"]


class ActorCriticRNN(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_hidden: int,
        widths: Sequence[int],
        rnn_hidden_dim: int,
    ):
        super().__init__()
        assert len(widths) == (num_hidden + 1), "PPO widths and number of layers mismatch!"
        self.input_layer = nn.Linear(obs_dim, widths[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(widths[idx], widths[idx + 1]) for idx in range(num_hidden)
        )
        self.rnn = nn.GRU(input_size=widths[-1], hidden_size=rnn_hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(rnn_hidden_dim, action_dim)
        self.value_head = nn.Linear(rnn_hidden_dim, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return x

    def forward(self, obs_seq: torch.Tensor, h0: Optional[torch.Tensor] = None):
        # obs_seq: [B, T, obs_dim]
        b, t, d = obs_seq.shape
        x = self._encode(obs_seq.reshape(b * t, d)).reshape(b, t, -1)
        out, hn = self.rnn(x, h0)
        logits = self.policy_head(out)  # [B, T, A]
        values = self.value_head(out).squeeze(-1)  # [B, T]
        return logits, values, hn


class PPO(BaseLearningModel):
    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        device: str = "cpu",
        batch_size: int = 16,
        lr: float = 3e-4,
        num_epochs: int = 4,
        num_hidden: int = 2,
        widths: Sequence[int] = (64, 64, 64),
        rnn_hidden_dim: int = 64,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        entropy_coef: float = 0.1,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        buffer_size: int = 2048,
    ):
        super().__init__()
        self.device = device
        self.action_space_size = int(action_space_size)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.clip_eps = float(clip_eps)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.normalize_advantage = bool(normalize_advantage)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

        self.policy_net = ActorCriticRNN(
            obs_dim=int(state_size),
            action_dim=self.action_space_size,
            num_hidden=int(num_hidden),
            widths=tuple(widths),
            rnn_hidden_dim=int(rnn_hidden_dim),
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=float(lr))

        self.loss = []
        self.deterministic = False

        # Store completed episodes for PPO updates.
        self.memory = deque(maxlen=int(buffer_size))
        self._episode_steps = []
        self._inference_hidden: Optional[torch.Tensor] = None

    def reset_episode(self) -> None:
        self._inference_hidden = None

    def _random_action(self, action_mask: Optional[np.ndarray]) -> int:
        if action_mask is None:
            return int(np.random.randint(self.action_space_size))
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            return int(np.random.randint(self.action_space_size))
        return int(np.random.choice(valid_actions))

    def act(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        state_np = np.asarray(state, dtype=np.float32)
        obs_t = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).view(1, 1, -1)

        logits, values, hn = self.policy_net(obs_t, self._inference_hidden)
        self._inference_hidden = hn.detach()

        logits_t = logits.squeeze(0).squeeze(0)  # [A]
        value_t = values.squeeze(0).squeeze(0)  # []

        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits_t.device)
            if mask.shape[0] == logits_t.shape[0] and mask.any():
                logits_t = logits_t.masked_fill(~mask, -1e9)
            elif not mask.any():
                action = self._random_action(action_mask)
                self.last_state = state_np
                self.last_action = int(action)
                self.last_log_prob = float("nan")
                self.last_value = float(value_t.item())
                return int(action)

        dist = torch.distributions.Categorical(logits=logits_t)
        if self.deterministic:
            action = int(torch.argmax(logits_t).item())
        else:
            action = int(dist.sample().item())
        log_prob = float(dist.log_prob(torch.tensor(action, device=logits_t.device)).item())

        self.last_state = state_np
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_value = float(value_t.item())
        return action

    def push(self, reward: float, done: bool = True) -> None:
        state = getattr(self, "last_state", None)
        action = getattr(self, "last_action", None)
        log_prob = getattr(self, "last_log_prob", None)
        value = getattr(self, "last_value", None)
        if state is None or action is None or log_prob is None or value is None:
            raise RuntimeError("push() called before act(); use act() first or implement a push_transition().")

        self._episode_steps.append(
            {
                "obs": np.asarray(state, dtype=np.float32),
                "action": int(action),
                "reward": float(reward),
                "done": bool(done),
                "log_prob": float(log_prob),
                "value": float(value),
            }
        )
        del self.last_state, self.last_action, self.last_log_prob, self.last_value

        if done:
            episode = self._finalize_episode(self._episode_steps)
            self.memory.append(episode)
            self._episode_steps = []

    def _finalize_episode(self, steps: list[dict]) -> dict:
        obs = np.stack([s["obs"] for s in steps], axis=0).astype(np.float32, copy=False)
        actions = np.asarray([s["action"] for s in steps], dtype=np.int64)
        rewards = np.asarray([s["reward"] for s in steps], dtype=np.float32)
        dones = np.asarray([s["done"] for s in steps], dtype=np.float32)
        old_log_probs = np.asarray([s["log_prob"] for s in steps], dtype=np.float32)
        values = np.asarray([s["value"] for s in steps], dtype=np.float32)

        # GAE(λ) advantages; terminal bootstrap assumed 0.
        adv = np.zeros_like(rewards, dtype=np.float32)
        last_adv = 0.0
        next_value = 0.0
        for t in range(rewards.shape[0] - 1, -1, -1):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * nonterminal * next_value - values[t]
            last_adv = delta + self.gamma * self.gae_lambda * nonterminal * last_adv
            adv[t] = last_adv
            next_value = values[t]
        returns = adv + values

        return {
            "obs": obs,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "advantages": adv,
            "returns": returns,
            "T": int(obs.shape[0]),
        }

    def learn(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        losses = []
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

            obs = torch.as_tensor(np.stack([pad_time(ep["obs"]) for ep in batch]), device=self.device)
            actions = torch.as_tensor(np.stack([pad_time(ep["actions"]) for ep in batch]), device=self.device)
            old_log_probs = torch.as_tensor(
                np.stack([pad_time(ep["old_log_probs"]) for ep in batch]), device=self.device
            )
            advantages = torch.as_tensor(
                np.stack([pad_time(ep["advantages"]) for ep in batch]), device=self.device
            )
            returns = torch.as_tensor(np.stack([pad_time(ep["returns"]) for ep in batch]), device=self.device)

            lengths = torch.tensor([int(ep["T"]) for ep in batch], device=self.device, dtype=torch.int64)
            time_mask = (
                torch.arange(max_t, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            ).to(dtype=torch.float32)

            if self.normalize_advantage:
                flat_adv = advantages[time_mask.bool()]
                advantages = (advantages - flat_adv.mean()) / (flat_adv.std() + 1e-8)

            logits, values, _ = self.policy_net(obs, None)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions.long())
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -(torch.min(surr1, surr2) * time_mask).sum() / time_mask.sum().clamp(min=1.0)

            value_loss = (((returns - values) ** 2) * time_mask).sum() / time_mask.sum().clamp(min=1.0)
            entropy_bonus = (entropy * time_mask).sum() / time_mask.sum().clamp(min=1.0)

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            losses.append(float(loss.item()))

        self.loss.append(float(sum(losses) / len(losses)))
        self.memory.clear()
