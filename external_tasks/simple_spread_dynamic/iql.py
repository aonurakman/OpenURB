import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch

from algorithms.iql import DQN
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
    dqn_kwargs: dict,
) -> None:
    base_id = next(iter(models), None)
    for aid in agent_ids:
        if aid in models:
            continue
        model = DQN(obs_size, action_space_size, device=device, **dqn_kwargs)
        if base_id is not None:
            model.value_network.load_state_dict(models[base_id].value_network.state_dict())
            model.target_network.load_state_dict(models[base_id].target_network.state_dict())
            model.temperature = models[base_id].temperature
        model.target_network.eval()
        models[aid] = model
        if base_id is None:
            base_id = aid


def close_env_cache(env_cache: dict) -> None:
    for env in env_cache.values():
        env.close()


def main():
    seed = 42
    episodes = 15000
    learning_starts = 2000
    learn_every_steps = 4
    eval_every_episodes = 200
    min_improve = 1e-3
    frame_skip = 2
    max_frames = 500

    set_global_seeds(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ref_env = make_env(seed=seed, n_agents=MAX_AGENTS, render_mode=None)
    all_agent_ids = list(ref_env.possible_agents)
    obs_size = ref_env.observation_space(all_agent_ids[0]).shape[0]
    action_space_size = ref_env.action_space(all_agent_ids[0]).n
    ref_env.close()

    dqn_kwargs = dict(
        temp_init=1.0,
        temp_decay=0.999,
        temp_min=0.05,
        buffer_size=100_000,
        batch_size=128,
        lr=5e-4,
        num_epochs=1,
        num_hidden=2,
        widths=(128, 128, 128),
        gamma=0.99,
        target_update_every=200,
        double_dqn=True,
        tau=1.0,
        max_grad_norm=10.0,
    )

    models: dict[str, DQN] = {}
    ensure_models([all_agent_ids[0]], models, obs_size, action_space_size, device, dqn_kwargs)

    out_dir = make_run_dir("simple_spread_dynamic_v3", "iql")
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pt")

    episode_rewards = []
    episode_losses = []
    eval_rewards = []
    global_step = 0
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
        ensure_models(all_agent_ids, models, obs_size, action_space_size, device, dqn_kwargs)
        temp_backup = {aid: models[aid].temperature for aid in models}
        for aid in models:
            models[aid].temperature = 0.0
            models[aid].value_network.eval()
            models[aid].target_network.eval()

        returns = []
        for idx, n_agents in enumerate(eval_counts):
            env = get_env(n_agents, render_mode=None)
            obs, _ = env.reset(seed=start_seed + idx)
            agent_ids = list(env.possible_agents)
            ensure_models(agent_ids, models, obs_size, action_space_size, device, dqn_kwargs)
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
                next_obs, rewards, terminations, truncations, _ = env.step(actions)
                for aid in agent_ids:
                    total += float(rewards.get(aid, 0.0))
                obs = next_obs
                term = {k: bool(v) for k, v in terminations.items()}
                trunc = {k: bool(v) for k, v in truncations.items()}
                if all(term.get(a, False) or trunc.get(a, False) for a in agent_ids):
                    break

            returns.append(total / len(agent_ids))

        for aid in models:
            models[aid].temperature = temp_backup[aid]
            models[aid].value_network.train()
        return float(np.mean(returns))

    for ep in range(episodes):
        n_agents = episode_agent_count(ep)
        env = get_env(n_agents, render_mode=None)
        obs, _ = env.reset(seed=seed + ep)
        agent_ids = list(env.possible_agents)
        ensure_models(agent_ids, models, obs_size, action_space_size, device, dqn_kwargs)
        for aid in agent_ids:
            models[aid].reset_episode()
        term = {aid: False for aid in agent_ids}
        trunc = {aid: False for aid in agent_ids}

        ep_reward_sum = 0.0
        ep_losses = []

        while True:
            actions = {}
            prev_obs = {}
            prev_actions = {}
            for aid in agent_ids:
                if term[aid] or trunc[aid]:
                    continue
                padded_obs = pad_obs(obs[aid], obs_size)
                a = models[aid].act(padded_obs)
                actions[aid] = a
                prev_obs[aid] = padded_obs
                prev_actions[aid] = a

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            global_step += 1

            for aid in agent_ids:
                r = float(rewards.get(aid, 0.0))
                done = bool(terminations.get(aid, False) or truncations.get(aid, False))
                ns_raw = next_obs.get(aid, None)
                ns = pad_obs(ns_raw, obs_size) if ns_raw is not None else np.zeros(obs_size, dtype=np.float32)
                if aid in prev_actions:
                    models[aid].push_transition(prev_obs[aid], prev_actions[aid], r, ns, done)
                ep_reward_sum += r

            if (global_step >= learning_starts) and ((global_step % learn_every_steps) == 0):
                for aid in models:
                    before = len(models[aid].loss)
                    models[aid].learn()
                    after = len(models[aid].loss)
                    if after > before:
                        ep_losses.append(models[aid].loss[-1])

            obs = next_obs
            term = {k: bool(v) for k, v in terminations.items()}
            trunc = {k: bool(v) for k, v in truncations.items()}
            if all(term.get(a, False) or trunc.get(a, False) for a in agent_ids):
                break

        episode_rewards.append(ep_reward_sum / len(agent_ids))
        episode_losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if eval_every_episodes and ((ep + 1) % eval_every_episodes == 0):
            current_eval = run_eval(seed + 10_000 + ep)
            eval_rewards.append(current_eval)
            if current_eval > (best_eval + min_improve):
                best_eval = current_eval
                ensure_models(all_agent_ids, models, obs_size, action_space_size, device, dqn_kwargs)
                torch.save(
                    {
                        "env_name": "simple_spread_dynamic_v3",
                        "seed": seed,
                        "dqn_kwargs": dqn_kwargs,
                        "value_state_dicts": {aid: models[aid].value_network.state_dict() for aid in all_agent_ids},
                        "target_state_dicts": {aid: models[aid].target_network.state_dict() for aid in all_agent_ids},
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
            print(
                f"[{ep+1}/{episodes}] train_reward={episode_rewards[-1]:.3f} "
                f"loss={episode_losses[-1]:.5f} temp~{np.mean([m.temperature for m in models.values()]):.3f}"
            )

    plot_curves(os.path.join(out_dir, "learning_curves.png"), episode_rewards, episode_losses, eval_rewards, window=50)
    np.save(os.path.join(out_dir, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))
    np.save(os.path.join(out_dir, "episode_losses.npy"), np.asarray(episode_losses, dtype=np.float32))
    np.save(os.path.join(out_dir, "eval_rewards.npy"), np.asarray(eval_rewards, dtype=np.float32))
    print(f"Saved results to {out_dir}")

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        ensure_models(all_agent_ids, models, obs_size, action_space_size, device, dqn_kwargs)
        for aid in all_agent_ids:
            models[aid].value_network.load_state_dict(ckpt["value_state_dicts"][aid])
            models[aid].target_network.load_state_dict(ckpt["target_state_dicts"][aid])

    # Save a GIF of the best policy (temperature=0).
    for aid in models:
        models[aid].temperature = 0.0
        models[aid].value_network.eval()

    frames = []
    step_idx = 0
    rollout_counts = eval_counts
    rollout_episodes = max(5, len(rollout_counts))
    for ep in range(rollout_episodes):
        n_agents = rollout_counts[ep % len(rollout_counts)]
        env_viz = get_env(n_agents, render_mode="rgb_array")
        obs, _ = env_viz.reset(seed=seed + 200_000 + ep)
        agent_ids = list(env_viz.possible_agents)
        ensure_models(agent_ids, models, obs_size, action_space_size, device, dqn_kwargs)
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
