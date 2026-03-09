import os
import sys
import json

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch

from algorithms.pimac_v2 import PIMACV2Parallel
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
    episodes = 15000
    learning_starts = 2000
    eval_every_episodes = 200
    eval_episodes = 10
    min_improve = 1e-3
    obs_clip = 10.0
    frame_skip = 2
    max_frames = 500

    set_global_seeds(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = make_env(seed=seed, render_mode=None)
    agent_ids = list(env.possible_agents)
    n_agents = len(agent_ids)

    obs_size = env.observation_space(agent_ids[0]).shape[0]
    action_space_size = env.action_space(agent_ids[0]).n

    pimac_v2_kwargs = dict(
        buffer_size=200_000,
        batch_size=64,
        lr=1e-4,
        num_epochs=4,
        num_hidden=2,
        widths=(128, 128, 128),
        rnn_hidden_dim=64,
        clip_eps=0.2,
        gae_lambda=0.95,
        normalize_advantage=True,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=5.0,
        gamma=0.99,
        set_embed_dim=64,
        set_encoder_hidden_sizes=(64, 64),
        critic_hidden_sizes=(64, 64),
        include_team_size_feature=True,
        num_tokens=4,
        distill_weight=0.1,
        teacher_ema_tau=0.01,
        ctx_logvar_min=-6.0,
        ctx_logvar_max=4.0,
    )
    pimac_v2 = PIMACV2Parallel(obs_size, action_space_size, device=device, **pimac_v2_kwargs)

    out_dir = make_run_dir("simple_spread_v3", "pimac_v2")
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pt")

    episode_rewards = []
    episode_losses = []
    eval_rewards = []
    global_step = 0
    best_eval = -float("inf")

    def get_state_fallback(obs_batch: np.ndarray) -> np.ndarray:
        return obs_batch.reshape(-1).astype(np.float32, copy=False)

    def run_eval(start_seed: int) -> float:
        det_backup = pimac_v2.deterministic
        pimac_v2.set_eval_mode()
        pimac_v2.deterministic = True

        returns = []
        for k in range(eval_episodes):
            obs, _ = env.reset(seed=start_seed + k)
            pimac_v2.reset_episode()
            term = {aid: False for aid in agent_ids}
            trunc = {aid: False for aid in agent_ids}
            total = 0.0
            while True:
                obs_dict = {}
                for aid in agent_ids:
                    if term[aid] or trunc[aid]:
                        continue
                    o = obs[aid]
                    if obs_clip is not None:
                        o = np.clip(o, -obs_clip, obs_clip).astype(np.float32, copy=False)
                    obs_dict[aid] = o
                actions = pimac_v2.act_parallel(obs_dict)
                obs, rewards, terminations, truncations, _ = env.step(actions)
                for aid in agent_ids:
                    total += float(rewards.get(aid, 0.0))
                term = {k: bool(v) for k, v in terminations.items()}
                trunc = {k: bool(v) for k, v in truncations.items()}
                if all(term.get(a, False) or trunc.get(a, False) for a in agent_ids):
                    break
            returns.append(total / n_agents)

        pimac_v2.deterministic = det_backup
        pimac_v2.set_train_mode()
        return float(np.mean(returns))

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        pimac_v2.reset_episode()
        term = {aid: False for aid in agent_ids}
        trunc = {aid: False for aid in agent_ids}

        ep_reward_sum = 0.0
        loss_start = len(pimac_v2.loss)

        while True:
            try:
                state_before = np.asarray(env.state(), dtype=np.float32).reshape(-1)
            except Exception:
                state_before = None

            obs_dict = {}
            obs_batch = np.zeros((n_agents, obs_size), dtype=np.float32)
            actions_batch = np.zeros(n_agents, dtype=np.int64)
            active_mask = np.zeros(n_agents, dtype=np.float32)
            # Build activity directly from live observations instead of assuming all agents
            # remain synchronized; this stays correct if a subset terminates earlier.
            for idx, aid in enumerate(agent_ids):
                if term.get(aid, False) or trunc.get(aid, False):
                    continue
                if aid not in obs:
                    continue
                o = obs[aid]
                if obs_clip is not None:
                    o = np.clip(o, -obs_clip, obs_clip).astype(np.float32, copy=False)
                obs_dict[aid] = o
                obs_batch[idx] = o
                active_mask[idx] = 1.0

            actions = pimac_v2.act_parallel(obs_dict)
            for idx, aid in enumerate(agent_ids):
                if aid in actions:
                    actions_batch[idx] = int(actions[aid])

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            global_step += 1

            try:
                state_after = np.asarray(env.state(), dtype=np.float32).reshape(-1)
            except Exception:
                state_after = None

            next_obs_batch = np.zeros((n_agents, obs_size), dtype=np.float32)
            next_active_mask = np.zeros(n_agents, dtype=np.float32)
            rewards_batch = np.zeros(n_agents, dtype=np.float32)
            for idx, aid in enumerate(agent_ids):
                r = float(rewards.get(aid, 0.0))
                rewards_batch[idx] = r
                ep_reward_sum += r
                if bool(terminations.get(aid, False)) or bool(truncations.get(aid, False)):
                    continue
                if aid not in next_obs:
                    continue
                no = next_obs[aid]
                if obs_clip is not None:
                    no = np.clip(no, -obs_clip, obs_clip).astype(np.float32, copy=False)
                next_obs_batch[idx] = no
                next_active_mask[idx] = 1.0

            done = all(
                bool(terminations.get(aid, term.get(aid, False)))
                or bool(truncations.get(aid, trunc.get(aid, False)))
                for aid in agent_ids
            )
            s = state_before if state_before is not None else get_state_fallback(obs_batch)
            ns = state_after if state_after is not None else get_state_fallback(next_obs_batch)

            pimac_v2.store_transition(
                observations=obs_batch,
                actions=actions_batch,
                rewards=rewards_batch,
                active_mask=active_mask,
                global_state=s,
                next_observations=next_obs_batch,
                next_active_mask=next_active_mask,
                next_global_state=ns,
                done=done,
            )

            obs = next_obs
            # Preserve prior terminal flags when a parallel env omits some keys.
            term = {aid: bool(terminations.get(aid, term.get(aid, False))) for aid in agent_ids}
            trunc = {aid: bool(truncations.get(aid, trunc.get(aid, False))) for aid in agent_ids}
            if done:
                break

        if (global_step >= learning_starts) and (len(pimac_v2.memory) >= pimac_v2.batch_size):
            pimac_v2.learn()

        new_losses = pimac_v2.loss[loss_start:]
        ep_loss = float(np.mean(new_losses)) if new_losses else 0.0
        episode_rewards.append(ep_reward_sum / n_agents)
        episode_losses.append(ep_loss)

        if eval_every_episodes and ((ep + 1) % eval_every_episodes == 0):
            current_eval = run_eval(seed + 10_000 + ep)
            eval_rewards.append(current_eval)
            if current_eval > (best_eval + min_improve):
                best_eval = current_eval
                torch.save(
                    {
                        "env_name": "simple_spread_v3",
                        "seed": seed,
                        "pimac_v2_kwargs": pimac_v2_kwargs,
                        "actor_state_dict": pimac_v2.actor_net.state_dict(),
                        "critic_state_dict": pimac_v2.critic.state_dict(),
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
                f"loss={episode_losses[-1]:.5f}"
            )

    plot_curves(os.path.join(out_dir, "learning_curves.png"), episode_rewards, episode_losses, eval_rewards, window=50)
    np.save(os.path.join(out_dir, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))
    np.save(os.path.join(out_dir, "episode_losses.npy"), np.asarray(episode_losses, dtype=np.float32))
    np.save(os.path.join(out_dir, "eval_rewards.npy"), np.asarray(eval_rewards, dtype=np.float32))
    with open(os.path.join(out_dir, "pimac_v2_loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(pimac_v2.loss_history, f, indent=2)
    env.close()
    print(f"Saved results to {out_dir}")

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        pimac_v2.actor_net.load_state_dict(ckpt["actor_state_dict"])
        pimac_v2.critic.load_state_dict(ckpt["critic_state_dict"])

    pimac_v2.set_eval_mode()
    pimac_v2.deterministic = True

    env_viz = make_env(seed=seed + 200_000, render_mode="rgb_array")
    frames = []
    step_idx = 0
    for ep in range(10):
        obs, _ = env_viz.reset(seed=seed + 200_000 + ep)
        pimac_v2.reset_episode()
        term = {aid: False for aid in agent_ids}
        trunc = {aid: False for aid in agent_ids}
        while True:
            obs_dict = {}
            for aid in agent_ids:
                if term[aid] or trunc[aid]:
                    continue
                o = obs[aid]
                if obs_clip is not None:
                    o = np.clip(o, -obs_clip, obs_clip).astype(np.float32, copy=False)
                obs_dict[aid] = o
            actions = pimac_v2.act_parallel(obs_dict)
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
