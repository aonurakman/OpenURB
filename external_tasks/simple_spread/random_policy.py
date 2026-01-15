import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np

from external_tasks.common import make_run_dir, plot_curves, save_gif


def make_env(seed: int, render_mode=None):
    try:
        from pettingzoo.mpe import simple_spread_v3
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "PettingZoo MPE is required. Install (e.g. `pip install pettingzoo[mpe] pygame`)."
        ) from exc

    env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=False, render_mode=render_mode)
    env.reset(seed=seed)
    return env


def main():
    seed = 42
    episodes = 100
    frame_skip = 2
    gif_episodes = 5
    max_frames = 500

    out_dir = make_run_dir("simple_spread_v3", "random_policy")

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
        term = {agent_id: False for agent_id in agent_ids}
        trunc = {agent_id: False for agent_id in agent_ids}
        total = 0.0

        while True:
            actions = {}
            for agent_id in agent_ids:
                if term[agent_id] or trunc[agent_id]:
                    continue
                actions[agent_id] = env.action_space(agent_id).sample()

            obs, rewards, terminations, truncations, _ = env.step(actions)
            for agent_id in agent_ids:
                total += float(rewards.get(agent_id, 0.0))

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
