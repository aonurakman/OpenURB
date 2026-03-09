"""
PIMACV0 implementation for OpenURB.

PIMACV0 is MAPPO with one architectural change:
- the centralized critic uses a Deep-Sets encoder (`phi` + masked pooling + `rho`)
  instead of fixed-width observation concatenation.

Everything else follows canonical MAPPO semantics used in this repository:
- shared decentralized recurrent actor,
- centralized critic for team values,
- team-level GAE(lambda),
- PPO clipping + entropy bonus,
- on-policy replay cleared after updates.
"""

from __future__ import annotations

from collections import deque
import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baseline_models import BaseLearningModel

__all__ = [
    "PIMACV0ActorRNN",
    "SetValueCritic",
    "PIMACV0Base",
    "PIMACV0AEC",
    "PIMACV0Parallel",
    "PIMACV0",
]


def _build_mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int) -> nn.Sequential:
    """Build a plain ReLU MLP used by the centralized value critic."""
    layers: list[nn.Module] = []
    current_dim = int(in_dim)
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(current_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        current_dim = int(hidden_dim)
    layers.append(nn.Linear(current_dim, int(out_dim)))
    return nn.Sequential(*layers)


def _sorted_agent_ids(keys) -> list:
    """Return deterministic ordering for mixed-type agent-id dictionaries."""
    return sorted(list(keys), key=lambda agent_identifier: str(agent_identifier))


class PIMACV0ActorRNN(nn.Module):
    """
    Shared recurrent actor used by all agents.

    The actor maps one agent's local observation sequence to categorical action logits.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_hidden: int,
        widths: Sequence[int],
        rnn_hidden_dim: int,
    ):
        super().__init__()
        assert len(widths) == (int(num_hidden) + 1), "PIMACV0 actor widths and layer count mismatch."

        self.input_layer = nn.Linear(int(obs_dim), int(widths[0]))
        self.hidden_layers = nn.ModuleList(
            nn.Linear(int(widths[layer_index]), int(widths[layer_index + 1]))
            for layer_index in range(int(num_hidden))
        )
        self.rnn = nn.GRU(input_size=int(widths[-1]), hidden_size=int(rnn_hidden_dim), batch_first=True)
        self.policy_head = nn.Linear(int(rnn_hidden_dim), int(action_dim))

    def _encode(self, input_features: torch.Tensor) -> torch.Tensor:
        """Encode flattened observations with the actor MLP stack."""
        encoded_features = torch.relu(self.input_layer(input_features))
        for hidden_layer in self.hidden_layers:
            encoded_features = torch.relu(hidden_layer(encoded_features))
        return encoded_features

    def forward(self, obs_seq: torch.Tensor, h0: Optional[torch.Tensor] = None):
        """
        Args:
            obs_seq: Observation sequence with shape [B, T, obs_dim].
            h0: Optional GRU initial state [1, B, H].

        Returns:
            logits: [B, T, action_dim]
            hn: [1, B, H]
        """
        batch_size, num_timesteps, observation_dim = obs_seq.shape
        encoded_sequence = self._encode(obs_seq.reshape(batch_size * num_timesteps, observation_dim)).reshape(
            batch_size,
            num_timesteps,
            -1,
        )
        recurrent_features, next_hidden_state = self.rnn(encoded_sequence, h0)
        action_logits = self.policy_head(recurrent_features)
        return action_logits, next_hidden_state


class SetValueCritic(nn.Module):
    """
    Centralized team-value critic with Deep-Sets pooling.

    The critic is permutation- and size-invariant with respect to agent order:
    1. `phi` embeds each agent observation independently.
    2. Embeddings are masked and pooled with a masked mean over agents.
    3. Optional team-size feature is concatenated.
    4. `rho` maps pooled context to scalar team value.

    This removes any architectural dependency on a configured maximum number of agents.
    """

    def __init__(
        self,
        obs_dim: int,
        set_embed_dim: int,
        set_encoder_hidden_sizes: Sequence[int],
        critic_hidden_sizes: Sequence[int],
        include_team_size_feature: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.set_embed_dim = int(set_embed_dim)
        self.include_team_size_feature = bool(include_team_size_feature)

        # Shared element-wise encoder `phi`: obs_i -> embedding_i.
        self.phi = _build_mlp(
            in_dim=self.obs_dim,
            hidden_sizes=tuple(set_encoder_hidden_sizes),
            out_dim=self.set_embed_dim,
        )
        pooled_dim = self.set_embed_dim + (1 if self.include_team_size_feature else 0)
        # Set-level head `rho`: pooled_embedding -> scalar value.
        self.rho = _build_mlp(
            in_dim=pooled_dim,
            hidden_sizes=tuple(critic_hidden_sizes),
            out_dim=1,
        )

    def forward(self, obs: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, T, N, obs_dim]
            active_mask: [B, T, N]

        Returns:
            team_values: [B, T]
        """
        batch_size, num_timesteps, num_agents, observation_dim = obs.shape
        if observation_dim != self.obs_dim:
            raise ValueError(
                f"Observation dimension mismatch: expected {self.obs_dim}, got {observation_dim}."
            )

        active_mask = active_mask.to(dtype=obs.dtype)
        embedded_agents = self.phi(obs.reshape(batch_size * num_timesteps * num_agents, observation_dim)).reshape(
            batch_size,
            num_timesteps,
            num_agents,
            self.set_embed_dim,
        )

        # Inactive/padded slots are zeroed before pooling.
        masked_embeddings = embedded_agents * active_mask.unsqueeze(-1)
        active_count = active_mask.sum(dim=2, keepdim=True)
        pooled_embedding = masked_embeddings.sum(dim=2) / active_count.clamp(min=1.0)

        if self.include_team_size_feature:
            # Keep explicit team-size signal while preserving invariance to padded inactive slots.
            # We use the active-agent count itself, which depends only on the mask.
            critic_input = torch.cat([pooled_embedding, active_count], dim=-1)
        else:
            critic_input = pooled_embedding

        values = self.rho(critic_input.reshape(batch_size * num_timesteps, -1)).reshape(batch_size, num_timesteps)
        return values


class PIMACV0Base(BaseLearningModel):
    """
    Shared PIMACV0 core used by both AEC and parallel environment wrappers.

    Responsibilities:
    - Shared actor/critic construction.
    - Robust replay coercion for dict/array transition payloads.
    - Episode finalization with roster alignment for variable team sizes.
    - Team-level GAE, PPO updates, diagnostics, and mode switching.
    """

    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        num_agents: Optional[int] = None,
        device: str = "cpu",
        buffer_size: int = 2048,
        batch_size: int = 32,
        lr: float = 3e-4,
        num_epochs: int = 4,
        num_hidden: int = 2,
        widths: Sequence[int] = (64, 64, 64),
        rnn_hidden_dim: int = 64,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: Optional[float] = 1.0,
        critic_hidden_sizes: Sequence[int] = (128, 128),
        set_embed_dim: int = 128,
        set_encoder_hidden_sizes: Sequence[int] = (128, 128),
        include_team_size_feature: bool = True,
        deterministic: bool = False,
    ):
        super().__init__()

        self.device = device
        self.obs_size = int(state_size)
        self.action_space_size = int(action_space_size)
        # Kept for drop-in compatibility with existing call sites; not used by critic width.
        self.num_agents = int(num_agents) if num_agents is not None else None

        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.normalize_advantage = bool(normalize_advantage)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

        self.deterministic = bool(deterministic)

        self.actor_net = PIMACV0ActorRNN(
            obs_dim=self.obs_size,
            action_dim=self.action_space_size,
            num_hidden=int(num_hidden),
            widths=tuple(widths),
            rnn_hidden_dim=int(rnn_hidden_dim),
        ).to(self.device)

        self.critic = SetValueCritic(
            obs_dim=self.obs_size,
            set_embed_dim=int(set_embed_dim),
            set_encoder_hidden_sizes=tuple(set_encoder_hidden_sizes),
            critic_hidden_sizes=tuple(critic_hidden_sizes),
            include_team_size_feature=bool(include_team_size_feature),
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self._actor_parameters()) + list(self.critic.parameters()),
            lr=float(lr),
        )

        # On-policy replay over finalized episodes.
        self.memory = deque(maxlen=int(buffer_size))
        self._episode_steps: list[dict] = []
        self._aec_cycle = None

        # Script-facing diagnostics and scalar loss history.
        self.loss: list[float] = []
        self.last_losses: dict[str, float] = {}
        self.loss_history: list[dict[str, float]] = []

        # Inference-only recurrent states keyed by environment agent ids.
        self._inference_hidden: dict[object, torch.Tensor] = {}

    def _actor_parameters(self):
        """Return actor parameters for optimizer and gradient clipping."""
        return self.actor_net.parameters()

    def reset_episode(self) -> None:
        """Clear inference recurrent state and pending AEC cycle buffers."""
        self._inference_hidden = {}
        self._aec_cycle = None

    def _get_h0(self, agent_key: object, hidden_dim: int) -> torch.Tensor:
        """Fetch or initialize the recurrent hidden state for one actor stream."""
        previous_hidden_state = self._inference_hidden.get(agent_key)
        if previous_hidden_state is None:
            previous_hidden_state = torch.zeros(1, 1, int(hidden_dim), device=self.device)
        return previous_hidden_state

    def _set_h(self, agent_key: object, hidden_state: torch.Tensor) -> None:
        """Persist detached recurrent state after one forward pass."""
        self._inference_hidden[agent_key] = hidden_state.detach()

    def _act_single(self, state: np.ndarray, actor_key: object) -> int:
        """Shared single-agent action path used by both wrapper APIs."""
        observation_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, 1, -1)
        hidden_dim = self.actor_net.rnn.hidden_size
        initial_hidden_state = self._get_h0(actor_key, hidden_dim)
        action_logits_sequence, updated_hidden_state = self.actor_net(observation_tensor, initial_hidden_state)
        self._set_h(actor_key, updated_hidden_state)

        action_logits = action_logits_sequence.squeeze(0).squeeze(0)
        if self.deterministic:
            return int(torch.argmax(action_logits).item())
        categorical_distribution = torch.distributions.Categorical(logits=action_logits)
        return int(categorical_distribution.sample().item())

    @staticmethod
    def _ensure_state(state: Optional[np.ndarray]) -> np.ndarray:
        """Normalize optional global-state payloads to dense float32 arrays."""
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
        """
        Normalize one transition into the internal replay format.

        The coercion layer keeps dict-based and array-based callers compatible
        with one shared episode-finalization path.
        """
        observations_are_mapping = isinstance(observations, dict)
        if observations_are_mapping:
            if agent_ids is None:
                agent_ids = _sorted_agent_ids(observations.keys())
            observation_array = np.stack(
                [np.asarray(observations[agent_id], dtype=np.float32) for agent_id in agent_ids],
                axis=0,
            )
        else:
            observation_array = np.asarray(observations, dtype=np.float32)
            if observation_array.ndim == 1:
                observation_array = observation_array.reshape(1, -1)
            if agent_ids is None:
                agent_ids = list(range(observation_array.shape[0]))

        if isinstance(actions, dict):
            action_array = np.asarray([actions.get(agent_id, 0) for agent_id in agent_ids], dtype=np.int64)
        else:
            action_array = np.asarray(actions, dtype=np.int64)
            if action_array.ndim == 0:
                action_array = action_array.reshape(1)

        if isinstance(rewards, dict):
            reward_array = np.asarray([rewards.get(agent_id, 0.0) for agent_id in agent_ids], dtype=np.float32)
        else:
            reward_array = np.asarray(rewards, dtype=np.float32)
            if reward_array.ndim == 0:
                reward_array = reward_array.reshape(1)

        if active_mask is None:
            active_mask_array = np.ones(len(agent_ids), dtype=np.float32)
        elif isinstance(active_mask, dict):
            active_mask_array = np.asarray([active_mask.get(agent_id, 0.0) for agent_id in agent_ids], dtype=np.float32)
        else:
            active_mask_array = np.asarray(active_mask, dtype=np.float32)
            if active_mask_array.ndim == 0:
                active_mask_array = active_mask_array.reshape(1)

        next_observations_are_mapping = isinstance(next_observations, dict)
        if next_observations_are_mapping:
            next_agent_ids = _sorted_agent_ids(next_observations.keys())
            if next_agent_ids:
                next_observation_array = np.stack(
                    [np.asarray(next_observations[agent_id], dtype=np.float32) for agent_id in next_agent_ids],
                    axis=0,
                )
            else:
                # Parallel terminal steps can report empty next-observation dictionaries.
                next_observation_array = np.zeros((0, observation_array.shape[-1]), dtype=np.float32)
        else:
            next_observation_array = np.asarray(next_observations, dtype=np.float32)
            if next_observation_array.ndim == 1:
                next_observation_array = next_observation_array.reshape(1, -1)
            next_agent_ids = list(range(next_observation_array.shape[0]))

        if next_active_mask is None:
            next_active_mask_array = np.ones(len(next_agent_ids), dtype=np.float32)
        elif isinstance(next_active_mask, dict):
            next_active_mask_array = np.asarray(
                [next_active_mask.get(agent_id, 0.0) for agent_id in next_agent_ids],
                dtype=np.float32,
            )
        else:
            next_active_mask_array = np.asarray(next_active_mask, dtype=np.float32)
            if next_active_mask_array.ndim == 0:
                next_active_mask_array = next_active_mask_array.reshape(1)

        return {
            "agent_ids": list(agent_ids),
            "obs": observation_array,
            "actions": action_array,
            "rewards": reward_array,
            "active_mask": active_mask_array,
            "state": self._ensure_state(global_state),
            "next_obs": next_observation_array,
            "next_active_mask": next_active_mask_array,
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
        """Append one transition and finalize the episode when `done=True`."""
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
            finalized_episode = self._finalize_episode(self._episode_steps)
            self.memory.append(finalized_episode)
            self._episode_steps = []

    def _actor_forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run shared actor over a padded minibatch tensor.

        Args:
            obs: [B, T, N, obs_dim]

        Returns:
            logits: [B, T, N, action_dim]
        """
        batch_size, num_timesteps, num_agents, observation_dim = obs.shape
        flattened_agent_sequences = obs.permute(0, 2, 1, 3).reshape(
            batch_size * num_agents,
            num_timesteps,
            observation_dim,
        )
        flattened_logits, _ = self.actor_net(flattened_agent_sequences, None)
        action_logits = flattened_logits.reshape(batch_size, num_agents, num_timesteps, -1).permute(0, 2, 1, 3)
        return action_logits

    @staticmethod
    def _team_rewards(rewards: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        """Compute mean active-agent reward per timestep."""
        active_reward_sum = (rewards * active_mask).sum(axis=1)
        active_agent_count = active_mask.sum(axis=1)
        timestep_team_rewards = np.zeros(rewards.shape[0], dtype=np.float32)
        valid_timesteps = active_agent_count > 0
        timestep_team_rewards[valid_timesteps] = (
            active_reward_sum[valid_timesteps] / active_agent_count[valid_timesteps]
        )
        return timestep_team_rewards

    def _compute_old_log_probs(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute behavior-policy log-probs stored with finalized episodes."""
        observation_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(0)
        active_mask_tensor = torch.as_tensor(active_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_logits = self._actor_forward(observation_tensor)
            categorical_distribution = torch.distributions.Categorical(logits=action_logits)
            old_log_probabilities = categorical_distribution.log_prob(action_tensor) * active_mask_tensor

        return old_log_probabilities.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def _compute_values(self, obs: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        """Evaluate centralized values for one finalized episode."""
        observation_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        active_mask_tensor = torch.as_tensor(active_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            predicted_values = self.critic(observation_tensor, active_mask_tensor).squeeze(0)
        return predicted_values.cpu().numpy().astype(np.float32, copy=False)

    def _compute_gae(
        self,
        team_rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        dones: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute team-level GAE(lambda) advantages and returns."""
        team_advantages = np.zeros_like(team_rewards, dtype=np.float32)
        running_advantage = 0.0

        for timestep_index in range(team_rewards.shape[0] - 1, -1, -1):
            non_terminal_factor = 1.0 - float(dones[timestep_index])
            temporal_difference_residual = (
                team_rewards[timestep_index]
                + self.gamma * non_terminal_factor * next_values[timestep_index]
                - values[timestep_index]
            )
            running_advantage = (
                temporal_difference_residual
                + self.gamma * self.gae_lambda * non_terminal_factor * running_advantage
            )
            team_advantages[timestep_index] = running_advantage

        team_returns = team_advantages + values
        return (
            team_advantages.astype(np.float32, copy=False),
            team_returns.astype(np.float32, copy=False),
        )

    def _finalize_episode(self, steps: list[dict]) -> dict:
        """
        Convert variable-roster step payloads into one roster-aligned episode record.

        The final episode contains dense [T, N, ...] arrays plus PPO fields
        (`old_log_probs`, `advantages`, `returns`).
        """
        episode_roster: list[object] = []
        roster_slot_by_agent_id: dict[object, int] = {}

        def register_agents(agent_ids_for_step: Sequence[object]) -> None:
            for agent_id in agent_ids_for_step:
                if agent_id not in roster_slot_by_agent_id:
                    roster_slot_by_agent_id[agent_id] = len(episode_roster)
                    episode_roster.append(agent_id)

        for step_record in steps:
            register_agents(step_record["agent_ids"])
            register_agents(step_record["next_agent_ids"])

        num_timesteps = len(steps)
        num_roster_agents = len(episode_roster)
        observation_dim = steps[0]["obs"].shape[-1]

        observations = np.zeros((num_timesteps, num_roster_agents, observation_dim), dtype=np.float32)
        actions = np.zeros((num_timesteps, num_roster_agents), dtype=np.int64)
        rewards = np.zeros((num_timesteps, num_roster_agents), dtype=np.float32)
        active_mask = np.zeros((num_timesteps, num_roster_agents), dtype=np.float32)
        next_observations = np.zeros((num_timesteps, num_roster_agents, observation_dim), dtype=np.float32)
        next_active_mask = np.zeros((num_timesteps, num_roster_agents), dtype=np.float32)
        global_states = np.stack([step_record["state"] for step_record in steps], axis=0)
        next_global_states = np.stack([step_record["next_state"] for step_record in steps], axis=0)
        done_flags = np.asarray([step_record["done"] for step_record in steps], dtype=np.float32)

        for timestep_index, step_record in enumerate(steps):
            for local_agent_index, agent_id in enumerate(step_record["agent_ids"]):
                roster_slot_index = roster_slot_by_agent_id[agent_id]
                observations[timestep_index, roster_slot_index] = step_record["obs"][local_agent_index]
                actions[timestep_index, roster_slot_index] = step_record["actions"][local_agent_index]
                rewards[timestep_index, roster_slot_index] = step_record["rewards"][local_agent_index]
                active_mask[timestep_index, roster_slot_index] = step_record["active_mask"][local_agent_index]

            for local_next_agent_index, agent_id in enumerate(step_record["next_agent_ids"]):
                roster_slot_index = roster_slot_by_agent_id[agent_id]
                next_observations[timestep_index, roster_slot_index] = step_record["next_obs"][local_next_agent_index]
                next_active_mask[timestep_index, roster_slot_index] = step_record["next_active_mask"][
                    local_next_agent_index
                ]

        old_log_probabilities = self._compute_old_log_probs(observations, actions, active_mask)
        team_values = self._compute_values(observations, active_mask)
        next_team_values = self._compute_values(next_observations, next_active_mask)
        team_rewards = self._team_rewards(rewards, active_mask)
        team_advantages, team_returns = self._compute_gae(team_rewards, team_values, next_team_values, done_flags)

        return {
            "obs": observations,
            "actions": actions,
            "rewards": rewards,
            "active_mask": active_mask,
            "state": global_states,
            "next_obs": next_observations,
            "next_active_mask": next_active_mask,
            "next_state": next_global_states,
            "done": done_flags,
            "old_log_probs": old_log_probabilities,
            "advantages": team_advantages,
            "returns": team_returns,
            "values": team_values,
            "team_rewards": team_rewards,
            "T": int(observations.shape[0]),
            "N": int(observations.shape[1]),
        }

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked mean with finite fallback on empty supports."""
        return (values * mask).sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def _pad_time(time_series_array: np.ndarray, max_num_timesteps: int, pad_value: float = 0.0) -> np.ndarray:
        """Right-pad the time axis of an array to a shared batch horizon."""
        current_num_timesteps = time_series_array.shape[0]
        if current_num_timesteps == max_num_timesteps:
            return time_series_array
        padding_shape = (max_num_timesteps - current_num_timesteps,) + time_series_array.shape[1:]
        right_padding = np.full(padding_shape, pad_value, dtype=time_series_array.dtype)
        return np.concatenate([time_series_array, right_padding], axis=0)

    @staticmethod
    def _pad_time_agents(
        time_agent_array: np.ndarray,
        max_num_timesteps: int,
        max_num_agents: int,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        """Pad [T, N, ...] tensors to one dense [max_T, max_N, ...] grid."""
        current_num_timesteps, current_num_agents = time_agent_array.shape[0], time_agent_array.shape[1]
        if current_num_timesteps == max_num_timesteps and current_num_agents == max_num_agents:
            return time_agent_array
        pad_width = [
            (0, max_num_timesteps - current_num_timesteps),
            (0, max_num_agents - current_num_agents),
        ] + [(0, 0)] * (time_agent_array.ndim - 2)
        return np.pad(time_agent_array, pad_width, mode="constant", constant_values=pad_value)

    def _build_minibatch_tensors(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Convert sampled episodes to padded training tensors.

        Padding is dynamic and local to the sampled minibatch:
        we pad to the largest team size present in this batch only.

        This padding exists solely for dense tensor batching; the critic itself is
        size-invariant and does not depend on a configured max-agent architecture.
        """
        max_num_timesteps = max(int(episode["T"]) for episode in batch)
        max_num_agents = max(int(episode["N"]) for episode in batch)

        observations = torch.as_tensor(
            np.stack([self._pad_time_agents(episode["obs"], max_num_timesteps, max_num_agents) for episode in batch]),
            device=self.device,
            dtype=torch.float32,
        )
        actions = torch.as_tensor(
            np.stack(
                [self._pad_time_agents(episode["actions"], max_num_timesteps, max_num_agents, pad_value=0) for episode in batch]
            ),
            device=self.device,
            dtype=torch.int64,
        )
        active_mask = torch.as_tensor(
            np.stack(
                [self._pad_time_agents(episode["active_mask"], max_num_timesteps, max_num_agents, pad_value=0.0) for episode in batch]
            ),
            device=self.device,
            dtype=torch.float32,
        )
        old_log_probs = torch.as_tensor(
            np.stack(
                [self._pad_time_agents(episode["old_log_probs"], max_num_timesteps, max_num_agents, pad_value=0.0) for episode in batch]
            ),
            device=self.device,
            dtype=torch.float32,
        )
        advantages = torch.as_tensor(
            np.stack([self._pad_time(episode["advantages"], max_num_timesteps, pad_value=0.0) for episode in batch]),
            device=self.device,
            dtype=torch.float32,
        )
        returns = torch.as_tensor(
            np.stack([self._pad_time(episode["returns"], max_num_timesteps, pad_value=0.0) for episode in batch]),
            device=self.device,
            dtype=torch.float32,
        )

        episode_lengths = torch.as_tensor([int(episode["T"]) for episode in batch], device=self.device, dtype=torch.int64)
        time_mask = (torch.arange(max_num_timesteps, device=self.device).unsqueeze(0) < episode_lengths.unsqueeze(1)).to(
            dtype=torch.float32
        )
        combined_mask = active_mask * time_mask.unsqueeze(-1)
        valid_time = time_mask * (active_mask.sum(dim=2) > 0.0).to(dtype=torch.float32)

        return {
            "obs": observations,
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
        """Normalize team-level advantages over valid timesteps only."""
        if not self.normalize_advantage:
            return advantages

        valid_advantages = advantages[valid_time.bool()]
        if valid_advantages.numel() == 0:
            return advantages

        advantage_mean = valid_advantages.mean()
        advantage_std = valid_advantages.std(unbiased=False)
        return (advantages - advantage_mean) / (advantage_std + 1e-8)

    def _compute_policy_terms(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute clipped PPO policy objective and entropy bonus."""
        action_logits = self._actor_forward(obs)
        categorical_distribution = torch.distributions.Categorical(logits=action_logits)
        new_log_probabilities = categorical_distribution.log_prob(actions)
        action_entropy = categorical_distribution.entropy()

        broadcast_advantages = advantages.unsqueeze(-1).expand_as(new_log_probabilities)
        importance_ratio = torch.exp(new_log_probabilities - old_log_probs)
        unclipped_surrogate = importance_ratio * broadcast_advantages
        clipped_surrogate = torch.clamp(importance_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * broadcast_advantages

        num_active_decisions = combined_mask.sum().clamp(min=1.0)
        policy_loss = -(torch.min(unclipped_surrogate, clipped_surrogate) * combined_mask).sum() / num_active_decisions
        entropy_bonus = (action_entropy * combined_mask).sum() / num_active_decisions

        return {
            "new_log_probs": new_log_probabilities,
            "ratio": importance_ratio,
            "policy_loss": policy_loss,
            "entropy_bonus": entropy_bonus,
            "decision_denom": num_active_decisions,
        }

    def _compute_value_terms(
        self,
        obs: torch.Tensor,
        active_mask: torch.Tensor,
        returns: torch.Tensor,
        valid_time: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute centralized value loss over valid timesteps."""
        predicted_values = self.critic(obs, active_mask)
        value_loss = (((returns - predicted_values) ** 2) * valid_time).sum() / valid_time.sum().clamp(min=1.0)
        return {
            "values": predicted_values,
            "value_loss": value_loss,
        }

    def _compute_update_diagnostics(
        self,
        active_mask: torch.Tensor,
        valid_time: torch.Tensor,
        combined_mask: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        ratio: torch.Tensor,
        decision_denom: torch.Tensor,
    ) -> dict[str, torch.Tensor | float]:
        """Compute PPO training diagnostics logged to `last_losses` and `loss_history`."""
        with torch.no_grad():
            log_probability_ratio = new_log_probs - old_log_probs
            approx_kl = ((-log_probability_ratio) * combined_mask).sum() / decision_denom
            clip_fraction = (((ratio - 1.0).abs() > self.clip_eps).to(dtype=torch.float32) * combined_mask).sum() / decision_denom

            valid_time_boolean_mask = valid_time.bool()
            if valid_time_boolean_mask.any():
                valid_returns = returns[valid_time_boolean_mask]
                valid_predicted_values = values.detach()[valid_time_boolean_mask]
                returns_variance = torch.var(valid_returns, unbiased=False)
                if float(returns_variance.item()) > 1e-8:
                    explained_variance = 1.0 - (
                        torch.var(valid_returns - valid_predicted_values, unbiased=False) / (returns_variance + 1e-8)
                    )
                else:
                    explained_variance = torch.tensor(0.0, device=self.device)
            else:
                explained_variance = torch.tensor(0.0, device=self.device)

            active_decision_count = float(combined_mask.sum().item())
            active_agents_per_timestep = active_mask.sum(dim=-1)
            active_agents_mean = self._masked_mean(active_agents_per_timestep, valid_time)

        return {
            "approx_kl": approx_kl,
            "clip_frac": clip_fraction,
            "explained_variance": explained_variance,
            "active_decisions": active_decision_count,
            "active_agents_mean": active_agents_mean,
        }

    def learn(self) -> None:
        """
        Run one PIMACV0 update phase.

        Each epoch samples on-policy finalized episodes, computes PPO + value losses,
        performs one optimizer step, and logs diagnostics.
        """
        if len(self.memory) < self.batch_size:
            return

        mean_epoch_losses = []
        for _ in range(self.num_epochs):
            sampled_episodes = random.sample(self.memory, self.batch_size)
            minibatch_tensors = self._build_minibatch_tensors(sampled_episodes)

            observations = minibatch_tensors["obs"]
            actions = minibatch_tensors["actions"]
            active_mask = minibatch_tensors["active_mask"]
            old_log_probabilities = minibatch_tensors["old_log_probs"]
            normalized_advantages = self._normalize_advantages(
                minibatch_tensors["advantages"],
                minibatch_tensors["valid_time"],
            )
            returns = minibatch_tensors["returns"]
            combined_mask = minibatch_tensors["combined_mask"]
            valid_time = minibatch_tensors["valid_time"]

            policy_terms = self._compute_policy_terms(
                obs=observations,
                actions=actions,
                old_log_probs=old_log_probabilities,
                advantages=normalized_advantages,
                combined_mask=combined_mask,
            )
            value_terms = self._compute_value_terms(
                obs=observations,
                active_mask=active_mask,
                returns=returns,
                valid_time=valid_time,
            )

            policy_loss = policy_terms["policy_loss"]
            entropy_bonus = policy_terms["entropy_bonus"]
            importance_ratio = policy_terms["ratio"]
            new_log_probabilities = policy_terms["new_log_probs"]
            decision_denom = policy_terms["decision_denom"]
            value_loss = value_terms["value_loss"]
            predicted_values = value_terms["values"]

            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    list(self._actor_parameters()) + list(self.critic.parameters()),
                    max_norm=self.max_grad_norm,
                )
            self.optimizer.step()

            diagnostics = self._compute_update_diagnostics(
                active_mask=active_mask,
                valid_time=valid_time,
                combined_mask=combined_mask,
                returns=returns,
                values=predicted_values,
                old_log_probs=old_log_probabilities,
                new_log_probs=new_log_probabilities,
                ratio=importance_ratio,
                decision_denom=decision_denom,
            )

            total_loss_scalar = float(total_loss.detach().item())
            mean_epoch_losses.append(total_loss_scalar)
            self.last_losses = {
                "policy_loss": float(policy_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "entropy": float(entropy_bonus.detach().item()),
                "total_loss": total_loss_scalar,
                "approx_kl": float(diagnostics["approx_kl"].detach().item()),
                "clip_frac": float(diagnostics["clip_frac"].detach().item()),
                "explained_variance": float(diagnostics["explained_variance"].detach().item()),
                "active_decisions": float(diagnostics["active_decisions"]),
                "active_agents_mean": float(diagnostics["active_agents_mean"].detach().item()),
            }
            self.loss_history.append(self.last_losses.copy())

        if mean_epoch_losses:
            self.loss.append(float(sum(mean_epoch_losses) / len(mean_epoch_losses)))

        # PPO is on-policy: once updates are done, replay must be cleared.
        self.memory.clear()

    def set_eval_mode(self) -> None:
        """Switch actor and critic modules to evaluation mode."""
        self.actor_net.eval()
        self.critic.eval()

    def set_train_mode(self) -> None:
        """Switch actor and critic modules to training mode."""
        self.actor_net.train()
        self.critic.train()


class PIMACV0AEC(PIMACV0Base):
    """PIMACV0 wrapper for AEC/OpenURB-style environments."""

    def act(
        self,
        state: np.ndarray,
        agent_index: Optional[object] = None,
    ) -> int:
        """Select an action for the currently acting AEC agent."""
        if agent_index is None:
            agent_index = 0
        return self._act_single(state=state, actor_key=agent_index)

    def store_episode(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        active_mask: np.ndarray,
        global_state: Optional[np.ndarray] = None,
    ) -> None:
        """
        Convenience helper for one-step OpenURB daily transitions.

        We represent the next observation with zeros and reuse the same active mask,
        matching existing OpenURB single-step script conventions.
        """
        zero_next_observations = np.zeros_like(np.asarray(observations, dtype=np.float32))
        current_global_state = self._ensure_state(global_state)
        zero_next_global_state = np.zeros_like(current_global_state)
        self.store_transition(
            observations=observations,
            actions=actions,
            rewards=rewards,
            active_mask=active_mask,
            global_state=current_global_state,
            next_observations=zero_next_observations,
            next_active_mask=active_mask,
            next_global_state=zero_next_global_state,
            done=True,
        )

    def aec_begin_cycle(self, agent_ids_current: Sequence[object]) -> None:
        """Initialize one AEC cycle accumulator."""
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
        """Record one per-agent turn inside an active AEC cycle."""
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
        """Finalize and store one joint transition for the active AEC cycle."""
        if self._aec_cycle is None:
            raise RuntimeError("Call aec_begin_cycle before aec_end_cycle.")
        episode_done = bool(done_all) or all(self._aec_cycle["dones"].values())
        self.store_transition(
            observations=self._aec_cycle["obs"],
            actions=self._aec_cycle["actions"],
            rewards=self._aec_cycle["rewards"],
            active_mask=None,
            global_state=global_state,
            next_observations=self._aec_cycle["next_obs"],
            next_active_mask=None,
            next_global_state=next_global_state,
            done=episode_done,
            agent_ids=self._aec_cycle["agent_ids"],
        )
        self._aec_cycle = None


class PIMACV0Parallel(PIMACV0Base):
    """PIMACV0 wrapper for parallel multi-agent environments."""

    def act(
        self,
        state: np.ndarray,
        agent_index: Optional[object] = None,
    ) -> int:
        """Single-agent helper for parallel APIs; prefer `act_parallel`."""
        if agent_index is None:
            raise ValueError("PIMACV0Parallel.act requires agent_index. Prefer act_parallel for parallel environments.")
        return self._act_single(state=state, actor_key=agent_index)

    def act_parallel(self, obs_dict: dict) -> dict:
        """Compute one action per currently present parallel-agent observation."""
        actions_by_agent_id = {}
        for agent_id in _sorted_agent_ids(obs_dict.keys()):
            actions_by_agent_id[agent_id] = self._act_single(
                state=obs_dict[agent_id],
                actor_key=agent_id,
            )
        return actions_by_agent_id

    @staticmethod
    def _resolve_parallel_done(done_dict) -> bool:
        """Normalize heterogeneous parallel done containers to one terminal boolean."""
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
        """Store one parallel-env joint transition."""
        episode_done = self._resolve_parallel_done(done_dict)
        self.store_transition(
            observations=obs_dict,
            actions=action_dict,
            rewards=reward_dict,
            active_mask=active_mask_dict,
            global_state=global_state,
            next_observations=next_obs_dict,
            next_active_mask=next_active_mask_dict,
            next_global_state=next_global_state,
            done=episode_done,
        )


# Compatibility alias for call sites that expect `PIMACV0` directly.
PIMACV0 = PIMACV0AEC
