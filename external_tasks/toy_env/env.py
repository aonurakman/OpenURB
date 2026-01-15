from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from gymnasium import spaces


@dataclass
class CoopLineWorldConfig:
    n_agents: int = 3
    goal: int = 8
    max_steps: int = 25


class CoopLineWorldParallelEnv:
    """A tiny cooperative multi-step environment that is easy to learn.

    Agents live on a 1D line [0..goal]. Each step, each agent chooses:
      0: stay, 1: move right, 2: move left

    Reward is shared and shaped by progress toward the goal:
      r_t = (prev_mean_dist - new_mean_dist) / goal
    plus a bonus when all agents reach the goal:
      +1.0 and terminate.

    The optimal policy is simple: move right until reaching the goal.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: CoopLineWorldConfig, render_mode: Optional[str] = None):
        self.config = config
        self.render_mode = render_mode
        self.possible_agents = [f"agent_{i}" for i in range(self.config.n_agents)]
        self._obs_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self._act_space = spaces.Discrete(3)
        self._rng = np.random.default_rng()
        self._t = 0
        self._positions = np.zeros(self.config.n_agents, dtype=np.int64)

    def observation_space(self, agent_id: str):
        return self._obs_space

    def action_space(self, agent_id: str):
        return self._act_space

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        # Start in a small random region to avoid a degenerate single start-state.
        self._positions = self._rng.integers(0, max(1, self.config.goal // 2 + 1), size=self.config.n_agents)
        obs = {agent_id: self._obs_for_idx(i) for i, agent_id in enumerate(self.possible_agents)}
        infos = {agent_id: {} for agent_id in self.possible_agents}
        return obs, infos

    def state(self) -> np.ndarray:
        pos = self._positions.astype(np.float32) / float(self.config.goal)
        t = np.asarray([self._t / float(self.config.max_steps)], dtype=np.float32)
        return np.concatenate([pos, t], axis=0)

    def _obs_for_idx(self, idx: int) -> np.ndarray:
        pos = float(self._positions[idx])
        pos_norm = pos / float(self.config.goal)
        dist_norm = float(self.config.goal - self._positions[idx]) / float(self.config.goal)
        t_norm = float(self._t) / float(self.config.max_steps)
        return np.asarray([pos_norm, dist_norm, t_norm], dtype=np.float32)

    def step(self, actions: dict):
        prev_mean_dist = float(np.mean(self.config.goal - self._positions))

        for idx, agent_id in enumerate(self.possible_agents):
            a = int(actions.get(agent_id, 0))
            if a == 1:
                self._positions[idx] = min(self.config.goal, self._positions[idx] + 1)
            elif a == 2:
                self._positions[idx] = max(0, self._positions[idx] - 1)

        self._t += 1
        new_mean_dist = float(np.mean(self.config.goal - self._positions))
        reward = (prev_mean_dist - new_mean_dist) / float(self.config.goal)

        terminated = bool(np.all(self._positions >= self.config.goal))
        truncated = self._t >= self.config.max_steps
        if terminated:
            reward += 1.0

        obs = {agent_id: self._obs_for_idx(i) for i, agent_id in enumerate(self.possible_agents)}
        rewards = {agent_id: float(reward) for agent_id in self.possible_agents}
        terminations = {agent_id: terminated for agent_id in self.possible_agents}
        truncations = {agent_id: truncated for agent_id in self.possible_agents}
        infos = {agent_id: {"positions": self._positions.copy()} for agent_id in self.possible_agents}
        return obs, rewards, terminations, truncations, infos

    def render(self):
        # Simple RGB visualization: a horizontal bar with colored agent markers.
        h, w = 64, 256
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        y_mid = h // 2

        # Draw line.
        img[y_mid - 2 : y_mid + 2, 10 : w - 10, :] = 230

        # Draw goal marker.
        goal_x = 10 + int((w - 20) * (self.config.goal / float(self.config.goal)))
        img[:, goal_x - 2 : goal_x + 2, :] = np.array([0, 200, 0], dtype=np.uint8)

        colors = [
            np.array([220, 20, 60], dtype=np.uint8),
            np.array([30, 144, 255], dtype=np.uint8),
            np.array([255, 165, 0], dtype=np.uint8),
            np.array([138, 43, 226], dtype=np.uint8),
            np.array([0, 128, 128], dtype=np.uint8),
        ]
        for idx in range(self.config.n_agents):
            x = 10 + int((w - 20) * (self._positions[idx] / float(self.config.goal)))
            c = colors[idx % len(colors)]
            img[y_mid - 8 : y_mid + 8, x - 3 : x + 3, :] = c

        return img

    def close(self):
        return None
