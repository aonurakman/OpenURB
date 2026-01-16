"""
IPPO (Independent PPO) implementation.

This is a standard clipped PPO actor-critic:
- Policy + value function (here: a shared GRU trunk to support partial observability).
- GAE(λ) for advantage estimation.
- Clipped surrogate objective with entropy bonus and value loss.

Recurrent state (`reset_episode`)
- `act(...)` maintains an internal GRU hidden state for *inference-time* action selection.
- Call `reset_episode()` once after every environment `reset()` (i.e., at the beginning of each episode/trajectory).
- For OpenURB scripts (single-step per agent/day), this means calling `reset_episode()` once per day; the GRU is
  applied for a single step so it effectively behaves like a feedforward policy.
- For multi-step tasks (e.g. `external_tasks/`), this means calling `reset_episode()` once per episode; the GRU then
  carries information across steps within the episode.

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
        # We first encode observations with an MLP (feature extraction),
        # then run a GRU to accumulate information over time (memory / latent state),
        # then branch into:
        # - a policy head (logits over discrete actions),
        # - a value head (V(s_t), used for advantage estimation).
        self.input_layer = nn.Linear(obs_dim, widths[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(widths[idx], widths[idx + 1]) for idx in range(num_hidden)
        )
        self.rnn = nn.GRU(input_size=widths[-1], hidden_size=rnn_hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(rnn_hidden_dim, action_dim)
        self.value_head = nn.Linear(rnn_hidden_dim, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the MLP per timestep. We keep it separate so we can easily reuse it
        # in both training (batched sequences) and action selection (one-step sequence).
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return x

    def forward(self, obs_seq: torch.Tensor, h0: Optional[torch.Tensor] = None):
        # obs_seq: [B, T, obs_dim]
        b, t, d = obs_seq.shape
        # Flatten time into the batch so the MLP can process all timesteps in one go.
        x = self._encode(obs_seq.reshape(b * t, d)).reshape(b, t, -1)
        # The GRU output at each timestep is a learned summary of past observations.
        out, hn = self.rnn(x, h0)
        # Policy logits and value estimates for each timestep.
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
        # PPO is an on-policy algorithm: updates are done using trajectories
        # collected by the *current* policy. `batch_size` here counts episodes.
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        # PPO clipping parameter (Schulman et al., 2017): constrains how much
        # the policy is allowed to change per update.
        self.clip_eps = float(clip_eps)
        # Discount and GAE parameters (GAE = Generalized Advantage Estimation).
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        # Advantage normalization reduces variance and makes PPO updates more stable.
        self.normalize_advantage = bool(normalize_advantage)
        # Entropy bonus encourages exploration; value_coef balances actor vs critic loss.
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        # Gradient clipping helps with occasional large policy gradients (especially with RNNs).
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

        # Recurrent actor-critic network (shared trunk + two heads).
        self.policy_net = ActorCriticRNN(
            obs_dim=int(state_size),
            action_dim=self.action_space_size,
            num_hidden=int(num_hidden),
            widths=tuple(widths),
            rnn_hidden_dim=int(rnn_hidden_dim),
        ).to(self.device)
        # One optimizer for both policy and value parameters (standard PPO implementation).
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=float(lr))

        self.loss = []
        self.deterministic = False

        # Store completed episodes for PPO updates (on-policy buffer).
        # Each entry holds arrays for obs/actions/log_probs/advantages/returns.
        self.memory = deque(maxlen=int(buffer_size))
        self._episode_steps = []
        # Inference-time GRU hidden state (maintained across timesteps within an episode).
        self._inference_hidden: Optional[torch.Tensor] = None

    def reset_episode(self) -> None:
        # Call this once after env.reset(). It clears the GRU hidden state so
        # the policy doesn't "remember" information across episodes.
        self._inference_hidden = None

    def _random_action(self, action_mask: Optional[np.ndarray]) -> int:
        # Helper used only when an action mask removes all valid actions.
        # (Some environments provide action masks; OpenURB route-choice does not.)
        if action_mask is None:
            return int(np.random.randint(self.action_space_size))
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            return int(np.random.randint(self.action_space_size))
        return int(np.random.choice(valid_actions))

    def act(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        # Convert to a 1-step sequence so we can reuse the same GRU forward path as training.
        state_np = np.asarray(state, dtype=np.float32)
        obs_t = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).view(1, 1, -1)

        # Forward pass updates the GRU hidden state; we keep it for the next timestep.
        logits, values, hn = self.policy_net(obs_t, self._inference_hidden)
        self._inference_hidden = hn.detach()

        logits_t = logits.squeeze(0).squeeze(0)  # [A]
        value_t = values.squeeze(0).squeeze(0)  # []

        if action_mask is not None:
            # Action masks are handled by setting invalid logits to a very negative number.
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits_t.device)
            if mask.shape[0] == logits_t.shape[0] and mask.any():
                logits_t = logits_t.masked_fill(~mask, -1e9)
            elif not mask.any():
                # If the mask says "no valid actions", fall back to a random action.
                # We also store NaN log_prob, because PPO cannot compute a valid log-prob in that case.
                action = self._random_action(action_mask)
                self.last_state = state_np
                self.last_action = int(action)
                self.last_log_prob = float("nan")
                self.last_value = float(value_t.item())
                return int(action)

        # Categorical policy over discrete actions. (For continuous actions you'd use Normal, etc.)
        dist = torch.distributions.Categorical(logits=logits_t)
        if self.deterministic:
            # Greedy action for evaluation (argmax over logits).
            action = int(torch.argmax(logits_t).item())
        else:
            # Sample action for exploration (stochastic policy).
            action = int(dist.sample().item())
        # Store log-prob for PPO's importance sampling ratio.
        log_prob = float(dist.log_prob(torch.tensor(action, device=logits_t.device)).item())

        # Cache the transition context so `push(reward, done)` can write it into the on-policy buffer.
        self.last_state = state_np
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_value = float(value_t.item())
        return action

    def push(self, reward: float, done: bool = True) -> None:
        # PPO collects trajectories first, then updates. This function records one timestep.
        # It pairs the most recent `act()` call with the observed reward/done flag.
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
            # End of episode: compute advantages/returns and store it for learning.
            episode = self._finalize_episode(self._episode_steps)
            self.memory.append(episode)
            self._episode_steps = []

    def _finalize_episode(self, steps: list[dict]) -> dict:
        # Convert a list of step dicts into fixed arrays.
        # We keep everything per-timestep so PPO can compute a loss over the whole trajectory.
        obs = np.stack([s["obs"] for s in steps], axis=0).astype(np.float32, copy=False)
        actions = np.asarray([s["action"] for s in steps], dtype=np.int64)
        rewards = np.asarray([s["reward"] for s in steps], dtype=np.float32)
        dones = np.asarray([s["done"] for s in steps], dtype=np.float32)
        old_log_probs = np.asarray([s["log_prob"] for s in steps], dtype=np.float32)
        values = np.asarray([s["value"] for s in steps], dtype=np.float32)

        # Compute GAE(λ) advantages (Schulman et al., 2016).
        # Intuition: advantage estimates "how much better than expected" the taken action was.
        # We assume terminal bootstrap value is 0 (standard for episodic tasks).
        adv = np.zeros_like(rewards, dtype=np.float32)
        last_adv = 0.0
        next_value = 0.0
        for t in range(rewards.shape[0] - 1, -1, -1):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * nonterminal * next_value - values[t]
            last_adv = delta + self.gamma * self.gae_lambda * nonterminal * last_adv
            adv[t] = last_adv
            next_value = values[t]
        # Returns are the value targets for the critic.
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
        # Perform PPO updates using the stored on-policy episodes.
        if len(self.memory) < self.batch_size:
            return

        losses = []
        for _ in range(self.num_epochs):
            # PPO often reuses the same data for a few epochs ("multiple passes over data").
            batch = random.sample(self.memory, self.batch_size)
            max_t = max(int(ep["T"]) for ep in batch)

            def pad_time(x, pad_value=0.0):
                t = x.shape[0]
                if t == max_t:
                    return x
                pad_shape = (max_t - t,) + x.shape[1:]
                pad = np.full(pad_shape, pad_value, dtype=x.dtype)
                return np.concatenate([x, pad], axis=0)

            # Pad variable-length episodes so we can batch them.
            obs = torch.as_tensor(np.stack([pad_time(ep["obs"]) for ep in batch]), device=self.device)
            actions = torch.as_tensor(np.stack([pad_time(ep["actions"]) for ep in batch]), device=self.device)
            old_log_probs = torch.as_tensor(
                np.stack([pad_time(ep["old_log_probs"]) for ep in batch]), device=self.device
            )
            advantages = torch.as_tensor(
                np.stack([pad_time(ep["advantages"]) for ep in batch]), device=self.device
            )
            returns = torch.as_tensor(np.stack([pad_time(ep["returns"]) for ep in batch]), device=self.device)

            # Mask out padded timesteps when computing losses.
            lengths = torch.tensor([int(ep["T"]) for ep in batch], device=self.device, dtype=torch.int64)
            time_mask = (
                torch.arange(max_t, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            ).to(dtype=torch.float32)

            if self.normalize_advantage:
                # Normalize over only real timesteps (ignoring padding).
                flat_adv = advantages[time_mask.bool()]
                advantages = (advantages - flat_adv.mean()) / (flat_adv.std() + 1e-8)

            # Compute new policy and value predictions for the batch.
            logits, values, _ = self.policy_net(obs, None)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions.long())
            entropy = dist.entropy()

            # PPO uses an importance sampling ratio between new and old policies:
            # r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            # Clipped objective: takes the pessimistic (min) of unclipped and clipped advantage terms.
            policy_loss = -(torch.min(surr1, surr2) * time_mask).sum() / time_mask.sum().clamp(min=1.0)

            # Critic loss: MSE between predicted V and computed returns.
            value_loss = (((returns - values) ** 2) * time_mask).sum() / time_mask.sum().clamp(min=1.0)
            # Entropy bonus: encourages higher-entropy (more exploratory) policies.
            entropy_bonus = (entropy * time_mask).sum() / time_mask.sum().clamp(min=1.0)

            # Full PPO loss: actor + value + entropy regularization.
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            losses.append(float(loss.item()))

        self.loss.append(float(sum(losses) / len(losses)))
        # On-policy buffer is cleared after an update: old data is "stale" for PPO.
        self.memory.clear()
