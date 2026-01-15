import os
import sys

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from external_tasks.common import make_run_dir, plot_curves, save_gif
from external_tasks.toy_env.env import CoopLineWorldConfig, CoopLineWorldParallelEnv


def make_env(seed: int, render_mode=None):
    env = CoopLineWorldParallelEnv(CoopLineWorldConfig(n_agents=3, goal=8, max_steps=25), render_mode=render_mode)
    env.reset(seed=seed)
    return env


def main():
    seed = 42
    episodes = 100
    gif_episodes = 5
    frame_skip = 1
    max_frames = 400

    out_dir = make_run_dir("coop_line_world", "random_policy")
    env = make_env(seed=seed, render_mode="rgb_array")
    agent_ids = list(env.possible_agents)
    n_agents = len(agent_ids)

    frames = []
    episode_rewards = []
    episode_losses = []
    eval_rewards = []
    step_idx = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        term = {aid: False for aid in agent_ids}
        trunc = {aid: False for aid in agent_ids}
        total = 0.0

        while True:
            actions = {
                aid: int(env.action_space(aid).sample())
                for aid in agent_ids
                if not (term.get(aid, False) or trunc.get(aid, False))
            }
            obs, rewards, terminations, truncations, _ = env.step(actions)
            for aid in agent_ids:
                total += float(rewards.get(aid, 0.0))

            if ep < gif_episodes and (step_idx % max(1, int(frame_skip))) == 0 and len(frames) < max_frames:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            step_idx += 1

            term = {k: bool(v) for k, v in terminations.items()}
            trunc = {k: bool(v) for k, v in truncations.items()}
            if all(term.get(a, False) or trunc.get(a, False) for a in agent_ids):
                break

        episode_rewards.append(total / n_agents)
        episode_losses.append(0.0)

    env.close()

    plot_curves(os.path.join(out_dir, "learning_curves.png"), episode_rewards, episode_losses, eval_rewards, window=20)
    np.save(os.path.join(out_dir, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))
    save_gif(frames, os.path.join(out_dir, "policy_rollout.gif"))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
