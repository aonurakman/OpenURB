"""
Independent Q-Learning (IQL) implementation.

This is a classic (Double) DQN with:
- A value network and a target network.
- TD targets with optional Double DQN target action selection.
- Replay buffer + gradient clipping.

Usage:
- Single-step tasks (OpenURB): call `act(state)` then `push(reward)` (default `done=True`).
- Multi-step tasks: call `push_transition(state, action, reward, next_state, done)` each step.

Recurrent state (`reset_episode`)
- The Q-network is a DRQN-style model (MLP encoder + GRU + Q head).
- `act(...)` maintains an internal GRU hidden state for *inference-time* action selection.
- Call `reset_episode()` once after every environment `reset()` (i.e., at the beginning of each episode/trajectory).
- For OpenURB scripts (single-step per agent/day), this means calling `reset_episode()` once per day; the GRU is
  applied for a single step so it effectively behaves like a feedforward policy.
- For multi-step tasks (e.g. `external_tasks/`), this means calling `reset_episode()` once per episode; the GRU then
  carries information across steps within the episode.
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

__all__ = ["RecurrentNetwork", "DQN"]


class RecurrentNetwork(nn.Module):
    """DRQN-style Q-network: MLP encoder + GRU + linear head."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_hidden: int,
        widths: Sequence[int],
        rnn_hidden_dim: int,
    ):
        super().__init__()
        assert len(widths) == (num_hidden + 1), "DQN widths and number of layers mismatch!"
        self.input_layer = nn.Linear(in_size, widths[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(widths[idx], widths[idx + 1]) for idx in range(num_hidden)
        )
        self.rnn = nn.GRU(input_size=widths[-1], hidden_size=int(rnn_hidden_dim), batch_first=True)
        self.out_layer = nn.Linear(int(rnn_hidden_dim), out_size)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # Encode per-timestep observations with an MLP before sending them into the GRU.
        # This is a common DRQN pattern: MLP handles per-step feature extraction,
        # GRU handles "memory" across steps.
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return x

    def forward(self, obs_seq: torch.Tensor, h0: Optional[torch.Tensor] = None):
        # obs_seq: [B, T, obs_dim]
        b, t, d = obs_seq.shape
        # Flatten the time dimension so we can apply the MLP to all steps at once,
        # then restore [B, T, feat] for the GRU.
        x = self._encode(obs_seq.reshape(b * t, d)).reshape(b, t, -1)
        # The GRU turns the sequence into a sequence of hidden states; the hidden state
        # is the learned summary of the history ("belief-like" latent state).
        out, hn = self.rnn(x, h0)
        # Per timestep, predict Q-values for all actions.
        q = self.out_layer(out)  # [B, T, A]
        return q, hn


class DQN(BaseLearningModel):
    """Independent Q-Learning (DQN) with a recurrent Q-network.

    This uses a DRQN-style network (MLP encoder + GRU) so the agent can build a learned
    latent state from observation history under partial observability.

    For single-step tasks (e.g., OpenURB route choice), this behaves like a feedforward
    policy because each episode has length 1 and the GRU state is reset between episodes.
    """

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
        rnn_hidden_dim: int = 0,
        seq_len: int = 8,
        gamma: float = 0.99,
        target_update_every: int = 100,
        double_dqn: bool = True,
        tau: float = 1.0,
        max_grad_norm: float = 10.0,
    ):
        super().__init__()
        self.device = device
        self.action_space_size = int(action_space_size)
        # Epsilon-greedy exploration schedule (classic DQN/IQL):
        # with prob epsilon, take a random action; otherwise take argmax_a Q(s,a).
        self.epsilon = float(eps_init)
        self.eps_decay = float(eps_decay)
        self.eps_min = float(eps_min)
        # Recurrent Q-network settings. We keep the GRU always-on (like IPPO/QMIX).
        # If rnn_hidden_dim is 0, default to the last MLP width (a reasonable baseline).
        self.rnn_hidden_dim = int(rnn_hidden_dim) if int(rnn_hidden_dim) > 0 else int(widths[-1])
        # Sequence length used for truncated backprop-through-time (BPTT) updates.
        # Longer sequences capture longer temporal dependencies but cost more compute/memory.
        self.seq_len = max(1, int(seq_len))
        # Replay buffer stores *episodes* (lists of transitions) so we can train a recurrent Q-network.
        # This is the usual DRQN approach; storing individual transitions would break temporal context.
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        # Discount factor for TD learning.
        self.gamma = float(gamma)
        # Target network update schedule and Double-DQN toggle (stability tricks for DQN).
        self.target_update_every = max(1, int(target_update_every))
        self.double_dqn = bool(double_dqn)
        # Soft-update coefficient for the target network (tau=1.0 -> hard copy).
        self.tau = float(tau)
        # Gradient clipping is a simple stabilizer for noisy TD targets / recurrent nets.
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
        # Count of learner updates (used to trigger target network updates).
        self._learn_steps = 0
        self._state_size = int(state_size)
        # Inference-time hidden state. This is what "makes it recurrent" at action selection time.
        # We reset it on env.reset() via reset_episode().
        self._inference_hidden: Optional[torch.Tensor] = None
        # Buffer transitions until the episode ends; then we push the whole list into replay.
        self._episode_steps = []

        # Online (value) network and target network. Same architecture; target is lagged.
        self.value_network = RecurrentNetwork(
            state_size, self.action_space_size, num_hidden, widths, rnn_hidden_dim=self.rnn_hidden_dim
        ).to(self.device)
        self.target_network = RecurrentNetwork(
            state_size, self.action_space_size, num_hidden, widths, rnn_hidden_dim=self.rnn_hidden_dim
        ).to(self.device)
        self.target_network.load_state_dict(self.value_network.state_dict())
        self.target_network.eval()

        # Backwards-compatible alias.
        self.q_network = self.value_network

        # Optimize only the online network; the target network is updated via (soft/hard) copies.
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=float(lr))
        # Huber loss is a common choice for TD learning (less sensitive to outliers than MSE).
        # Use reduction="none" so we can mask out padded timesteps for variable-length sequences.
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.loss = []

    def reset_episode(self) -> None:
        # Important: call this once at the start of every episode (right after env.reset()).
        # Without this, the GRU hidden state would leak information across episodes.
        self._inference_hidden = None
        self._episode_steps = []

    def act(self, state: np.ndarray) -> int:
        # Epsilon-greedy action selection. This is the "I" in IQL: each agent explores independently.
        if np.random.rand() < self.epsilon:
            # Explore: random action (ignores Q-network).
            action = int(np.random.choice(self.action_space_size))
        else:
            # Exploit: run the recurrent Q-network for a single step, updating hidden state.
            # We pass a sequence of length 1: [B=1, T=1, obs_dim].
            with torch.no_grad():
                obs_seq = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, 1, -1)
                q_seq, hn = self.value_network(obs_seq, self._inference_hidden)
                self._inference_hidden = hn
                q_values = q_seq[:, -1, :]
            # Greedy action w.r.t. predicted Q-values.
            action = int(torch.argmax(q_values, dim=-1).item())
        self.last_state = np.asarray(state, dtype=np.float32)
        self.last_action = action
        return action

    def push(self, reward: float, next_state: Optional[np.ndarray] = None, done: bool = True) -> None:
        # Convenience API used in OpenURB (single-step): act(state) then push(reward).
        # For multi-step environments, prefer push_transition(...) each step.
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
        # Store one environment step. We buffer inside _episode_steps and only commit the episode
        # to replay when `done=True`. This gives the learner temporal context for DRQN training.
        self._episode_steps.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )
        if done:
            self.memory.append(self._episode_steps)
            self._episode_steps = []

    def _update_target_network(self) -> None:
        # Target network update:
        # - Hard update: copy weights exactly every N learner steps (tau >= 1.0).
        # - Soft update: Polyak averaging (tau in (0,1)), smoother but more frequent.
        if self.tau >= 1.0:
            self.target_network.load_state_dict(self.value_network.state_dict())
            return
        with torch.no_grad():
            for target_param, value_param in zip(self.target_network.parameters(), self.value_network.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * value_param.data)

    def _learn_rnn(self) -> None:
        # Recurrent DQN update (DRQN-style):
        # We sample short chunks from stored episodes and train with TD targets.
        if len(self.memory) < self.batch_size:
            return

        step_losses = []
        for _ in range(self.num_epochs):
            # Sample full episodes, then cut a random contiguous chunk from each.
            # This is truncated BPTT: cheaper than training on full episodes, still lets the GRU learn memory.
            episodes = random.sample(self.memory, self.batch_size)

            # Sample a random chunk from each episode for truncated BPTT.
            chunks = []
            max_t = 1
            for ep in episodes:
                if not ep:
                    continue
                start = 0 if len(ep) <= 1 else random.randint(0, len(ep) - 1)
                end = min(len(ep), start + self.seq_len)
                chunk = ep[start:end]
                chunks.append(chunk)
                max_t = max(max_t, len(chunk))

            if not chunks:
                return

            # Build padded batch tensors so we can do one vectorized forward pass.
            # `time_mask` marks which timesteps are real vs padding (used to mask the loss).
            b = len(chunks)
            obs = np.zeros((b, max_t, self._state_size), dtype=np.float32)
            next_obs = np.zeros((b, max_t, self._state_size), dtype=np.float32)
            actions = np.zeros((b, max_t), dtype=np.int64)
            rewards = np.zeros((b, max_t), dtype=np.float32)
            dones = np.ones((b, max_t), dtype=np.float32)
            time_mask = np.zeros((b, max_t), dtype=np.float32)

            for i, chunk in enumerate(chunks):
                for t, (s, a, r, ns, d) in enumerate(chunk):
                    obs[i, t] = s
                    next_obs[i, t] = ns
                    actions[i, t] = int(a)
                    rewards[i, t] = float(r)
                    dones[i, t] = 1.0 if bool(d) else 0.0
                    time_mask[i, t] = 1.0

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            mask_t = torch.as_tensor(time_mask, dtype=torch.float32, device=self.device)

            # Online Q-values for all actions at every timestep: Q(s_t, ·).
            q_seq, _ = self.value_network(obs_t, None)  # [B, T, A]
            # Pick the Q-value of the action actually taken in the data: Q(s_t, a_t).
            chosen_q = torch.gather(q_seq, dim=2, index=actions_t.unsqueeze(-1)).squeeze(-1)  # [B, T]

            with torch.no_grad():
                # Next-state Q-values are used to build TD targets.
                # Double DQN uses the online net to *select* the action and the target net to *evaluate* it.
                next_q_online, _ = self.value_network(next_obs_t, None)
                next_q_target, _ = self.target_network(next_obs_t, None)
                if self.double_dqn:
                    next_actions = torch.argmax(next_q_online, dim=2, keepdim=True)  # [B, T, 1]
                    next_q = torch.gather(next_q_target, dim=2, index=next_actions).squeeze(-1)  # [B, T]
                else:
                    next_q = torch.max(next_q_target, dim=2).values

                # One-step TD target: r + gamma * (1-done) * max_a' Q_target(s',a').
                targets = rewards_t + (1.0 - dones_t) * self.gamma * next_q

            # TD error. We compute it for all timesteps, then mask out padding.
            td = self.loss_fn(chosen_q, targets)
            loss = (td * mask_t).sum() / mask_t.sum().clamp(min=1.0)

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            # Periodically update target network to stabilize learning.
            self._learn_steps += 1
            if self._learn_steps % self.target_update_every == 0:
                self._update_target_network()

            step_losses.append(float(loss.item()))

        self.loss.append(float(sum(step_losses) / len(step_losses)))
        # DQN typically decays epsilon after updates (not after environment steps).
        self.decay_epsilon()

    def learn(self) -> None:
        self._learn_rnn()

    def decay_epsilon(self) -> None:
        # Keep exploration above eps_min so the agent doesn't get stuck too early.
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
