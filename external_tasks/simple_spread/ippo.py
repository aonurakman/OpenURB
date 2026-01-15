import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch

from algorithms.ippo import PPO
from external_tasks.common import make_run_dir, plot_curves, save_gif, set_global_seeds


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
    episodes = 12000
    update_every_episodes = 8
    eval_every_episodes = 200
    eval_episodes = 10
    min_improve = 1e-3
    frame_skip = 2
    max_frames = 500

    ppo_kwargs = dict(
        batch_size=8,
        lr=3e-4,
        num_epochs=4,
        num_hidden=2,
        widths=(64, 64, 64),
        rnn_hidden_dim=64,
        clip_eps=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantage=True,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=1.0,
        buffer_size=2048,
    )

    set_global_seeds(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = make_env(seed=seed, render_mode=None)
    agent_ids = list(env.possible_agents)
    n_agents = len(agent_ids)

    obs_size = env.observation_space(agent_ids[0]).shape[0]
    action_space_size = env.action_space(agent_ids[0]).n
    models = {aid: PPO(obs_size, action_space_size, device=device, **ppo_kwargs) for aid in agent_ids}

    out_dir = make_run_dir("simple_spread_v3", "ippo")
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pt")

    episode_rewards = []
    episode_losses = []
    eval_rewards = []
    best_eval = -float("inf")

    def run_eval(start_seed: int) -> float:
        det_backup = {aid: models[aid].deterministic for aid in agent_ids}
        for aid in agent_ids:
            models[aid].deterministic = True
            models[aid].policy_net.eval()

        returns = []
        for k in range(eval_episodes):
            obs, _ = env.reset(seed=start_seed + k)
            for aid in agent_ids:
                models[aid].reset_episode()
            term = {aid: False for aid in agent_ids}
            trunc = {aid: False for aid in agent_ids}
            total = 0.0

            while True:
                actions = {}
                for aid in agent_ids:
                    if term[aid] or trunc[aid]:
                        continue
                    actions[aid] = models[aid].act(obs[aid])
                obs, rewards, terminations, truncations, _ = env.step(actions)
                for aid in agent_ids:
                    total += float(rewards.get(aid, 0.0))
                term = {k: bool(v) for k, v in terminations.items()}
                trunc = {k: bool(v) for k, v in truncations.items()}
                if all(term.get(a, False) or trunc.get(a, False) for a in agent_ids):
                    break

            returns.append(total / n_agents)

        for aid in agent_ids:
            models[aid].deterministic = det_backup[aid]
            models[aid].policy_net.train()
        return float(np.mean(returns))

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        for aid in agent_ids:
            models[aid].reset_episode()

        term = {aid: False for aid in agent_ids}
        trunc = {aid: False for aid in agent_ids}
        ep_reward_sum = 0.0

        while True:
            actions = {}
            for aid in agent_ids:
                if term[aid] or trunc[aid]:
                    continue
                actions[aid] = models[aid].act(obs[aid])

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            done = all(bool(terminations.get(a, False)) or bool(truncations.get(a, False)) for a in agent_ids)

            for aid in agent_ids:
                r = float(rewards.get(aid, 0.0))
                models[aid].push(r, done=done)
                ep_reward_sum += r

            obs = next_obs
            term = {k: bool(v) for k, v in terminations.items()}
            trunc = {k: bool(v) for k, v in truncations.items()}
            if done:
                break

        # PPO updates are on-policy; update after collecting whole episodes.
        ep_losses = []
        if update_every_episodes and ((ep + 1) % update_every_episodes == 0):
            for aid in agent_ids:
                before = len(models[aid].loss)
                models[aid].learn()
                after = len(models[aid].loss)
                if after > before:
                    ep_losses.append(models[aid].loss[-1])

        episode_rewards.append(ep_reward_sum / n_agents)
        episode_losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if eval_every_episodes and ((ep + 1) % eval_every_episodes == 0):
            current_eval = run_eval(seed + 10_000 + ep)
            eval_rewards.append(current_eval)
            if current_eval > (best_eval + min_improve):
                best_eval = current_eval
                torch.save(
                    {
                        "env_name": "simple_spread_v3",
                        "seed": seed,
                        "ppo_kwargs": ppo_kwargs,
                        "policy_state_dicts": {aid: models[aid].policy_net.state_dict() for aid in agent_ids},
                    },
                    best_ckpt_path,
                )

        if (ep + 1) % 200 == 0:
            plot_curves(
                os.path.join(out_dir, "learning_curves.png"),
                episode_rewards,
                episode_losses,
                eval_rewards,
                window=50,
            )
            print(f"[{ep+1}/{episodes}] train_reward={episode_rewards[-1]:.3f} loss={episode_losses[-1]:.5f}")

    plot_curves(os.path.join(out_dir, "learning_curves.png"), episode_rewards, episode_losses, eval_rewards, window=50)
    np.save(os.path.join(out_dir, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))
    np.save(os.path.join(out_dir, "episode_losses.npy"), np.asarray(episode_losses, dtype=np.float32))
    np.save(os.path.join(out_dir, "eval_rewards.npy"), np.asarray(eval_rewards, dtype=np.float32))
    env.close()
    print(f"Saved results to {out_dir}")

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        for aid in agent_ids:
            models[aid].policy_net.load_state_dict(ckpt["policy_state_dicts"][aid])

    # Save a GIF of the best policy (deterministic).
    for aid in agent_ids:
        models[aid].deterministic = True
        models[aid].policy_net.eval()

    env_viz = make_env(seed=seed + 200_000, render_mode="rgb_array")
    frames = []
    step_idx = 0
    for ep in range(5):
        obs, _ = env_viz.reset(seed=seed + 200_000 + ep)
        for aid in agent_ids:
            models[aid].reset_episode()
        term = {aid: False for aid in agent_ids}
        trunc = {aid: False for aid in agent_ids}
        while True:
            actions = {}
            for aid in agent_ids:
                if term[aid] or trunc[aid]:
                    continue
                actions[aid] = models[aid].act(obs[aid])
            obs, _, terminations, truncations, _ = env_viz.step(actions)
            if (step_idx % max(1, int(frame_skip))) == 0 and len(frames) < max_frames:
                frame = env_viz.render()
                if frame is not None:
                    frames.append(frame)
            step_idx += 1
            term = {k: bool(v) for k, v in terminations.items()}
            trunc = {k: bool(v) for k, v in truncations.items()}
            if all(term.get(a, False) or trunc.get(a, False) for a in agent_ids):
                break
            if len(frames) >= max_frames:
                break
        if len(frames) >= max_frames:
            break
    env_viz.close()
    save_gif(frames, os.path.join(out_dir, "policy_rollout.gif"))
    print(f"Saved policy rollout GIF to {os.path.join(out_dir, 'policy_rollout.gif')}")


if __name__ == "__main__":
    main()
