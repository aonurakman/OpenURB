import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch

from algorithms.ippo import PPO
from external_tasks.common import make_run_dir, plot_curves, save_gif, set_global_seeds

MIN_AGENTS = 3
MAX_AGENTS = 10


def make_env(seed: int, n_agents: int, render_mode=None):
    try:
        from pettingzoo.mpe import simple_spread_v3
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "PettingZoo MPE is required. Install (e.g. `pip install pettingzoo[mpe] pygame`)."
        ) from exc

    env = simple_spread_v3.parallel_env(N=n_agents, max_cycles=25, continuous_actions=False, render_mode=render_mode)
    env.reset(seed=seed)
    return env


def episode_agent_count(episode_idx: int) -> int:
    span = MAX_AGENTS - MIN_AGENTS + 1
    return MIN_AGENTS + (episode_idx % span)


def eval_agent_counts() -> list[int]:
    mid = (MIN_AGENTS + MAX_AGENTS) // 2
    counts = [MIN_AGENTS, mid, MAX_AGENTS]
    unique = []
    for count in counts:
        if count not in unique:
            unique.append(count)
    if len(unique) < 3:
        for count in range(MIN_AGENTS, MAX_AGENTS + 1):
            if count not in unique:
                unique.append(count)
            if len(unique) == 3:
                break
    return unique


def pad_obs(obs: np.ndarray, target_dim: int) -> np.ndarray:
    obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs_arr.shape[0] >= target_dim:
        return obs_arr[:target_dim]
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[: obs_arr.shape[0]] = obs_arr
    return padded


def ensure_models(
    agent_ids: list[str],
    models: dict,
    obs_size: int,
    action_space_size: int,
    device: torch.device,
    ppo_kwargs: dict,
) -> None:
    base_id = next(iter(models), None)
    for aid in agent_ids:
        if aid in models:
            continue
        model = PPO(obs_size, action_space_size, device=device, **ppo_kwargs)
        if base_id is not None:
            model.policy_net.load_state_dict(models[base_id].policy_net.state_dict())
            model.deterministic = models[base_id].deterministic
        models[aid] = model
        if base_id is None:
            base_id = aid


def close_env_cache(env_cache: dict) -> None:
    for env in env_cache.values():
        env.close()


def main():
    seed = 42
    episodes = 12000
    update_every_episodes = 8
    eval_every_episodes = 200
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

    ref_env = make_env(seed=seed, n_agents=MAX_AGENTS, render_mode=None)
    all_agent_ids = list(ref_env.possible_agents)
    obs_size = ref_env.observation_space(all_agent_ids[0]).shape[0]
    action_space_size = ref_env.action_space(all_agent_ids[0]).n
    ref_env.close()

    models: dict[str, PPO] = {}
    ensure_models([all_agent_ids[0]], models, obs_size, action_space_size, device, ppo_kwargs)

    out_dir = make_run_dir("simple_spread_dynamic_v3", "ippo")
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pt")

    episode_rewards = []
    episode_losses = []
    eval_rewards = []
    best_eval = -float("inf")

    env_cache = {}

    def get_env(n_agents: int, render_mode=None):
        key = (n_agents, render_mode)
        env = env_cache.get(key)
        if env is None:
            env = make_env(seed=seed, n_agents=n_agents, render_mode=render_mode)
            env_cache[key] = env
        return env

    eval_counts = eval_agent_counts()

    def run_eval(start_seed: int) -> float:
        ensure_models(all_agent_ids, models, obs_size, action_space_size, device, ppo_kwargs)
        det_backup = {aid: models[aid].deterministic for aid in models}
        for aid in models:
            models[aid].deterministic = True
            models[aid].policy_net.eval()

        returns = []
        for idx, n_agents in enumerate(eval_counts):
            env = get_env(n_agents, render_mode=None)
            obs, _ = env.reset(seed=start_seed + idx)
            agent_ids = list(env.possible_agents)
            ensure_models(agent_ids, models, obs_size, action_space_size, device, ppo_kwargs)
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
                    actions[aid] = models[aid].act(pad_obs(obs[aid], obs_size))
                obs, rewards, terminations, truncations, _ = env.step(actions)
                for aid in agent_ids:
                    total += float(rewards.get(aid, 0.0))
                term = {k: bool(v) for k, v in terminations.items()}
                trunc = {k: bool(v) for k, v in truncations.items()}
                if all(term.get(a, False) or trunc.get(a, False) for a in agent_ids):
                    break

            returns.append(total / len(agent_ids))

        for aid in models:
            models[aid].deterministic = det_backup[aid]
            models[aid].policy_net.train()
        return float(np.mean(returns))

    for ep in range(episodes):
        n_agents = episode_agent_count(ep)
        env = get_env(n_agents, render_mode=None)
        obs, _ = env.reset(seed=seed + ep)
        agent_ids = list(env.possible_agents)
        ensure_models(agent_ids, models, obs_size, action_space_size, device, ppo_kwargs)
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
                actions[aid] = models[aid].act(pad_obs(obs[aid], obs_size))

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
            for aid in models:
                before = len(models[aid].loss)
                models[aid].learn()
                after = len(models[aid].loss)
                if after > before:
                    ep_losses.append(models[aid].loss[-1])

        episode_rewards.append(ep_reward_sum / len(agent_ids))
        episode_losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if eval_every_episodes and ((ep + 1) % eval_every_episodes == 0):
            current_eval = run_eval(seed + 10_000 + ep)
            eval_rewards.append(current_eval)
            if current_eval > (best_eval + min_improve):
                best_eval = current_eval
                ensure_models(all_agent_ids, models, obs_size, action_space_size, device, ppo_kwargs)
                torch.save(
                    {
                        "env_name": "simple_spread_dynamic_v3",
                        "seed": seed,
                        "ppo_kwargs": ppo_kwargs,
                        "policy_state_dicts": {aid: models[aid].policy_net.state_dict() for aid in all_agent_ids},
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
    print(f"Saved results to {out_dir}")

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        ensure_models(all_agent_ids, models, obs_size, action_space_size, device, ppo_kwargs)
        for aid in all_agent_ids:
            models[aid].policy_net.load_state_dict(ckpt["policy_state_dicts"][aid])

    # Save a GIF of the best policy (deterministic).
    for aid in models:
        models[aid].deterministic = True
        models[aid].policy_net.eval()

    frames = []
    step_idx = 0
    rollout_counts = eval_counts
    rollout_episodes = max(5, len(rollout_counts))
    for ep in range(rollout_episodes):
        n_agents = rollout_counts[ep % len(rollout_counts)]
        env_viz = get_env(n_agents, render_mode="rgb_array")
        obs, _ = env_viz.reset(seed=seed + 200_000 + ep)
        agent_ids = list(env_viz.possible_agents)
        ensure_models(agent_ids, models, obs_size, action_space_size, device, ppo_kwargs)
        for aid in agent_ids:
            models[aid].reset_episode()
        term = {aid: False for aid in agent_ids}
        trunc = {aid: False for aid in agent_ids}
        while True:
            actions = {}
            for aid in agent_ids:
                if term[aid] or trunc[aid]:
                    continue
                actions[aid] = models[aid].act(pad_obs(obs[aid], obs_size))
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

    close_env_cache(env_cache)
    save_gif(frames, os.path.join(out_dir, "policy_rollout.gif"))
    print(f"Saved policy rollout GIF to {os.path.join(out_dir, 'policy_rollout.gif')}")


if __name__ == "__main__":
    main()
