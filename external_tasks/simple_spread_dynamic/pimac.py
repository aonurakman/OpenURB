import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch

from algorithms.pimac import PIMAC
from external_tasks.common import make_run_dir, moving_average, save_gif, set_global_seeds

import matplotlib.pyplot as plt

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


def pad_vector(vec: np.ndarray, target_dim: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] >= target_dim:
        return arr[:target_dim]
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[: arr.shape[0]] = arr
    return padded


def close_env_cache(env_cache: dict) -> None:
    for env in env_cache.values():
        env.close()


def _build_loss_series(loss_history: list[dict[str, float]]) -> dict[str, np.ndarray]:
    if not loss_history:
        return {}
    preferred = [
        "td",
        "subteam_td",
        "distill",
        "teacher_smooth",
        "context_gate_reg",
        "agent_loss",
        "teacher_loss",
    ]
    keys = []
    existing = {k for entry in loss_history for k in entry.keys()}
    for key in preferred:
        if key in existing:
            keys.append(key)
    for key in sorted(existing - set(keys)):
        keys.append(key)
    return {key: np.asarray([entry.get(key, 0.0) for entry in loss_history], dtype=np.float32) for key in keys}


def _should_log_scale(series: dict[str, np.ndarray], threshold: float = 100.0) -> bool:
    if not series:
        return False
    values = np.concatenate([v for v in series.values() if v.size])
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 2:
        return False
    return float(values.max() / values.min()) >= threshold


def plot_pimac_curves(
    save_path: str,
    episode_rewards: list[float],
    eval_steps: list[int],
    eval_means: list[float],
    eval_stds: list[float],
    loss_history: list[dict[str, float]],
    window: int = 50,
) -> None:
    rewards_np = np.asarray(episode_rewards, dtype=np.float32)
    eval_steps_np = np.asarray(eval_steps, dtype=np.int64)
    eval_means_np = np.asarray(eval_means, dtype=np.float32)
    eval_stds_np = np.asarray(eval_stds, dtype=np.float32)
    loss_series = _build_loss_series(loss_history)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    if rewards_np.size:
        episodes = np.arange(1, rewards_np.size + 1)
        axes[0].plot(episodes, rewards_np, color="tab:blue", alpha=0.35, label="train return")
        if rewards_np.size >= window:
            smooth = moving_average(rewards_np, window)
            smooth_x = np.arange(window, window + smooth.size)
        else:
            smooth = rewards_np
            smooth_x = episodes
        axes[0].plot(smooth_x, smooth, color="tab:orange", linewidth=2, label=f"train {window}-ep moving avg")
    axes[0].set_ylabel("Train return")
    axes[0].grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend()

    if eval_means_np.size:
        axes[1].plot(eval_steps_np, eval_means_np, marker="o", color="tab:orange", label="eval mean")
        if eval_stds_np.size:
            axes[1].fill_between(
                eval_steps_np,
                eval_means_np - eval_stds_np,
                eval_means_np + eval_stds_np,
                color="tab:orange",
                alpha=0.2,
                label="eval std",
            )
    axes[1].set_ylabel("Eval return")
    axes[1].grid(True, alpha=0.3)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend()

    log_scale = _should_log_scale(loss_series)
    if loss_series:
        eps = 1e-8
        for key, values in loss_series.items():
            if not values.size:
                continue
            plot_values = np.maximum(values, eps) if log_scale else values
            axes[2].plot(np.arange(1, values.size + 1), plot_values, label=key, linewidth=1.2, alpha=0.85)
        if log_scale:
            axes[2].set_yscale("log")
            axes[2].set_ylabel("Loss (log scale)")
        else:
            axes[2].set_ylabel("Loss")
        handles, labels = axes[2].get_legend_handles_labels()
        if handles:
            axes[2].legend(ncol=2, fontsize=8)
    else:
        axes[2].set_ylabel("Loss")
        axes[2].text(0.5, 0.5, "No loss history yet", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_xlabel("Learner update")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    seed = 42
    episodes = 15000
    learning_starts = 2000
    learn_every_steps = 4
    eval_every_episodes = 200
    min_improve = 1e-3
    obs_clip = 10.0
    frame_skip = 2
    max_frames = 500

    set_global_seeds(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ref_env = make_env(seed=seed, n_agents=MAX_AGENTS, render_mode=None)
    ref_agent_ids = list(ref_env.possible_agents)
    obs_size = ref_env.observation_space(ref_agent_ids[0]).shape[0]
    action_space_size = ref_env.action_space(ref_agent_ids[0]).n
    try:
        ref_state = np.asarray(ref_env.state(), dtype=np.float32).reshape(-1)
        global_state_size = int(ref_state.shape[0])
    except Exception:
        global_state_size = int(MAX_AGENTS * obs_size)
    ref_env.close()

    pimac_kwargs = dict(
        temp_init=1.0,
        temp_decay=0.999,
        temp_min=0.05,
        buffer_size=200_000,
        batch_size=64,
        lr=1e-4,
        teacher_lr=1e-4,
        num_epochs=1,
        num_hidden=2,
        widths=(128, 128, 128),
        rnn_hidden_dim=64,
        ctx_dim=16,
        num_tokens=8,
        tok_dim=16,
        teacher_emb_dim=64,
        teacher_hidden_sizes=(64, 64),
        teacher_attn_dim=None,
        teacher_drop_prob=0.5,
        distill_weight=0.5,
        token_smooth_weight=0.0,
        obs_index_dim=0,
        obs_skip=True,
        context_gate_reg=0.001,
        subteam_samples=0,
        subteam_keep_prob=0.75,
        subteam_td_weight=0.25,
        max_grad_norm=5.0,
        gamma=0.99,
        target_update_every=200,
        double_q=True,
        tau=1.0,
        share_parameters=True,
        q_tot_clip=None,
        use_huber_loss=True,
        normalize_by_active=True,
    )
    pimac = PIMAC(obs_size, action_space_size, num_agents=MAX_AGENTS, device=device, **pimac_kwargs)

    out_dir = make_run_dir("simple_spread_dynamic_v3", "pimac")
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pt")

    episode_rewards = []
    episode_losses = []
    eval_steps = []
    eval_means = []
    eval_stds = []
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

    def run_eval(start_seed: int) -> tuple[float, float]:
        temp_backup = pimac.temperature
        pimac.temperature = 0.0
        pimac.set_eval_mode()

        returns = []
        for idx, n_agents in enumerate(eval_counts):
            env = get_env(n_agents, render_mode=None)
            obs, _ = env.reset(seed=start_seed + idx)
            agent_ids = list(env.possible_agents)
            pimac.reset_episode()
            term = {aid: False for aid in agent_ids}
            trunc = {aid: False for aid in agent_ids}
            total = 0.0
            while True:
                actions = {}
                for agent_index, aid in enumerate(agent_ids):
                    if term[aid] or trunc[aid]:
                        continue
                    o = obs[aid]
                    if obs_clip is not None:
                        o = np.clip(o, -obs_clip, obs_clip).astype(np.float32, copy=False)
                    actions[aid] = pimac.act(pad_vector(o, obs_size), agent_index=agent_index)
                obs, rewards, terminations, truncations, _ = env.step(actions)
                for aid in agent_ids:
                    total += float(rewards.get(aid, 0.0))
                term = {k: bool(v) for k, v in terminations.items()}
                trunc = {k: bool(v) for k, v in truncations.items()}
                if all(term.get(a, False) or trunc.get(a, False) for a in agent_ids):
                    break
            returns.append(total / len(agent_ids))

        pimac.temperature = temp_backup
        pimac.set_train_mode()
        return float(np.mean(returns)), float(np.std(returns))

    for ep in range(episodes):
        n_agents = episode_agent_count(ep)
        env = get_env(n_agents, render_mode=None)
        obs, _ = env.reset(seed=seed + ep)
        agent_ids = list(env.possible_agents)
        pimac.reset_episode()
        term = {aid: False for aid in agent_ids}
        trunc = {aid: False for aid in agent_ids}

        ep_reward_sum = 0.0
        loss_start = len(pimac.loss)

        while True:
            try:
                raw_state_before = np.asarray(env.state(), dtype=np.float32).reshape(-1)
            except Exception:
                raw_state_before = None

            actions = {}
            obs_batch = np.zeros((MAX_AGENTS, obs_size), dtype=np.float32)
            actions_batch = np.zeros(MAX_AGENTS, dtype=np.int64)
            active_mask = np.zeros(MAX_AGENTS, dtype=np.float32)
            for idx, aid in enumerate(agent_ids):
                if term[aid] or trunc[aid]:
                    continue
                o = obs[aid]
                if obs_clip is not None:
                    o = np.clip(o, -obs_clip, obs_clip).astype(np.float32, copy=False)
                padded_obs = pad_vector(o, obs_size)
                a = pimac.act(padded_obs, agent_index=idx)
                actions[aid] = a
                obs_batch[idx] = padded_obs
                actions_batch[idx] = int(a)
                active_mask[idx] = 1.0

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            global_step += 1

            try:
                raw_state_after = np.asarray(env.state(), dtype=np.float32).reshape(-1)
            except Exception:
                raw_state_after = None

            next_obs_batch = np.zeros((MAX_AGENTS, obs_size), dtype=np.float32)
            next_active_mask = np.zeros(MAX_AGENTS, dtype=np.float32)
            rewards_batch = np.zeros(MAX_AGENTS, dtype=np.float32)
            for idx, aid in enumerate(agent_ids):
                r = float(rewards.get(aid, 0.0))
                rewards_batch[idx] = r
                ep_reward_sum += r
                if not (bool(terminations.get(aid, False)) or bool(truncations.get(aid, False))):
                    no = next_obs[aid]
                    if obs_clip is not None:
                        no = np.clip(no, -obs_clip, obs_clip).astype(np.float32, copy=False)
                    next_obs_batch[idx] = pad_vector(no, obs_size)
                    next_active_mask[idx] = 1.0

            done = all(bool(terminations.get(a, False)) or bool(truncations.get(a, False)) for a in agent_ids)
            state_before = pad_vector(
                raw_state_before if raw_state_before is not None else obs_batch.reshape(-1),
                global_state_size,
            )
            state_after = pad_vector(
                raw_state_after if raw_state_after is not None else next_obs_batch.reshape(-1),
                global_state_size,
            )

            pimac.store_transition(
                observations=obs_batch,
                actions=actions_batch,
                rewards=rewards_batch,
                active_mask=active_mask,
                global_state=state_before,
                next_observations=next_obs_batch,
                next_active_mask=next_active_mask,
                next_global_state=state_after,
                done=done,
            )

            if (global_step >= learning_starts) and ((global_step % learn_every_steps) == 0):
                pimac.learn()

            obs = next_obs
            term = {k: bool(v) for k, v in terminations.items()}
            trunc = {k: bool(v) for k, v in truncations.items()}
            if done:
                break

        new_losses = pimac.loss[loss_start:]
        ep_loss = float(np.mean(new_losses)) if new_losses else 0.0
        episode_rewards.append(ep_reward_sum / len(agent_ids))
        episode_losses.append(ep_loss)

        if eval_every_episodes and ((ep + 1) % eval_every_episodes == 0):
            eval_mean, eval_std = run_eval(seed + 10_000 + ep)
            eval_steps.append(ep + 1)
            eval_means.append(eval_mean)
            eval_stds.append(eval_std)
            if eval_mean > (best_eval + min_improve):
                best_eval = eval_mean
                torch.save(
                    {
                        "env_name": "simple_spread_dynamic_v3",
                        "seed": seed,
                        "pimac_kwargs": pimac_kwargs,
                        "agent_state_dict": pimac.agent_net.state_dict(),
                        "teacher_state_dict": pimac.teacher.state_dict(),
                    },
                    best_ckpt_path,
                )

        if (ep + 1) % 200 == 0:
            plot_pimac_curves(
                os.path.join(out_dir, "learning_curves.png"),
                episode_rewards,
                eval_steps,
                eval_means,
                eval_stds,
                pimac.loss_history,
                window=50,
            )
            print(
                f"[{ep+1}/{episodes}] train_reward={episode_rewards[-1]:.3f} "
                f"loss={episode_losses[-1]:.5f} temp={pimac.temperature:.3f}"
            )

    plot_pimac_curves(
        os.path.join(out_dir, "learning_curves.png"),
        episode_rewards,
        eval_steps,
        eval_means,
        eval_stds,
        pimac.loss_history,
        window=50,
    )
    np.save(os.path.join(out_dir, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))
    np.save(os.path.join(out_dir, "episode_losses.npy"), np.asarray(episode_losses, dtype=np.float32))
    np.save(os.path.join(out_dir, "eval_rewards.npy"), np.asarray(eval_means, dtype=np.float32))
    np.save(os.path.join(out_dir, "eval_stds.npy"), np.asarray(eval_stds, dtype=np.float32))
    np.save(os.path.join(out_dir, "eval_steps.npy"), np.asarray(eval_steps, dtype=np.int64))
    print(f"Saved results to {out_dir}")

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        pimac.agent_net.load_state_dict(ckpt["agent_state_dict"])
        pimac.target_agent_net.load_state_dict(ckpt["agent_state_dict"])
        pimac.teacher.load_state_dict(ckpt["teacher_state_dict"])

    pimac.temperature = 0.0
    pimac.set_eval_mode()

    frames = []
    step_idx = 0
    rollout_counts = eval_counts
    rollout_episodes = max(5, len(rollout_counts))
    for ep in range(rollout_episodes):
        n_agents = rollout_counts[ep % len(rollout_counts)]
        env_viz = get_env(n_agents, render_mode="rgb_array")
        obs, _ = env_viz.reset(seed=seed + 200_000 + ep)
        agent_ids = list(env_viz.possible_agents)
        pimac.reset_episode()
        term = {aid: False for aid in agent_ids}
        trunc = {aid: False for aid in agent_ids}
        while True:
            actions = {}
            for idx, aid in enumerate(agent_ids):
                if term[aid] or trunc[aid]:
                    continue
                o = obs[aid]
                if obs_clip is not None:
                    o = np.clip(o, -obs_clip, obs_clip).astype(np.float32, copy=False)
                actions[aid] = pimac.act(pad_vector(o, obs_size), agent_index=idx)
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
