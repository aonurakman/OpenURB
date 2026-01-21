import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch

from algorithms.pimac import PIMAC
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
    learn_every_steps = 4
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
        teacher_aux_weight=1.0,
        token_smooth_weight=0.0,
        teacher_use_actions=True,
        obs_index_dim=0,
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
    pimac = PIMAC(obs_size, action_space_size, num_agents=n_agents, device=device, **pimac_kwargs)

    out_dir = make_run_dir("simple_spread_v3", "pimac")
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pt")

    episode_rewards = []
    episode_losses = []
    eval_rewards = []
    global_step = 0
    best_eval = -float("inf")

    def get_state_fallback(obs_batch: np.ndarray) -> np.ndarray:
        return obs_batch.reshape(-1).astype(np.float32, copy=False)

    def run_eval(start_seed: int) -> float:
        temp_backup = pimac.temperature
        pimac.temperature = 0.0
        pimac.set_eval_mode()

        returns = []
        for k in range(eval_episodes):
            obs, _ = env.reset(seed=start_seed + k)
            pimac.reset_episode()
            term = {aid: False for aid in agent_ids}
            trunc = {aid: False for aid in agent_ids}
            total = 0.0
            while True:
                actions = {}
                for idx, aid in enumerate(agent_ids):
                    if term[aid] or trunc[aid]:
                        continue
                    o = obs[aid]
                    if obs_clip is not None:
                        o = np.clip(o, -obs_clip, obs_clip).astype(np.float32, copy=False)
                    actions[aid] = pimac.act(o, agent_index=idx)
                obs, rewards, terminations, truncations, _ = env.step(actions)
                for aid in agent_ids:
                    total += float(rewards.get(aid, 0.0))
                term = {k: bool(v) for k, v in terminations.items()}
                trunc = {k: bool(v) for k, v in truncations.items()}
                if all(term.get(a, False) or trunc.get(a, False) for a in agent_ids):
                    break
            returns.append(total / n_agents)

        pimac.temperature = temp_backup
        pimac.set_train_mode()
        return float(np.mean(returns))

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        pimac.reset_episode()
        term = {aid: False for aid in agent_ids}
        trunc = {aid: False for aid in agent_ids}

        ep_reward_sum = 0.0
        loss_start = len(pimac.loss)

        while True:
            try:
                state_before = np.asarray(env.state(), dtype=np.float32).reshape(-1)
            except Exception:
                state_before = None

            actions = {}
            obs_batch = np.zeros((n_agents, obs_size), dtype=np.float32)
            actions_batch = np.zeros(n_agents, dtype=np.int64)
            active_mask = np.ones(n_agents, dtype=np.float32)
            for idx, aid in enumerate(agent_ids):
                if term[aid] or trunc[aid]:
                    continue
                o = obs[aid]
                if obs_clip is not None:
                    o = np.clip(o, -obs_clip, obs_clip).astype(np.float32, copy=False)
                a = pimac.act(o, agent_index=idx)
                actions[aid] = a
                obs_batch[idx] = o
                actions_batch[idx] = int(a)

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            global_step += 1

            try:
                state_after = np.asarray(env.state(), dtype=np.float32).reshape(-1)
            except Exception:
                state_after = None

            next_obs_batch = np.zeros((n_agents, obs_size), dtype=np.float32)
            next_active_mask = np.ones(n_agents, dtype=np.float32)
            rewards_batch = np.zeros(n_agents, dtype=np.float32)
            for idx, aid in enumerate(agent_ids):
                r = float(rewards.get(aid, 0.0))
                rewards_batch[idx] = r
                ep_reward_sum += r
                if not (bool(terminations.get(aid, False)) or bool(truncations.get(aid, False))):
                    no = next_obs[aid]
                    if obs_clip is not None:
                        no = np.clip(no, -obs_clip, obs_clip).astype(np.float32, copy=False)
                    next_obs_batch[idx] = no

            done = all(bool(terminations.get(a, False)) or bool(truncations.get(a, False)) for a in agent_ids)
            s = state_before if state_before is not None else get_state_fallback(obs_batch)
            ns = state_after if state_after is not None else get_state_fallback(next_obs_batch)

            pimac.store_transition(
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

            if (global_step >= learning_starts) and ((global_step % learn_every_steps) == 0):
                pimac.learn()

            obs = next_obs
            term = {k: bool(v) for k, v in terminations.items()}
            trunc = {k: bool(v) for k, v in truncations.items()}
            if done:
                break

        new_losses = pimac.loss[loss_start:]
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
                        "pimac_kwargs": pimac_kwargs,
                        "agent_state_dict": (
                            pimac.agent_net.state_dict() if pimac.share_parameters else pimac.agent_nets.state_dict()
                        ),
                        "teacher_state_dict": pimac.teacher.state_dict(),
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
                f"loss={episode_losses[-1]:.5f} temp={pimac.temperature:.3f}"
            )

    plot_curves(os.path.join(out_dir, "learning_curves.png"), episode_rewards, episode_losses, eval_rewards, window=50)
    np.save(os.path.join(out_dir, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))
    np.save(os.path.join(out_dir, "episode_losses.npy"), np.asarray(episode_losses, dtype=np.float32))
    np.save(os.path.join(out_dir, "eval_rewards.npy"), np.asarray(eval_rewards, dtype=np.float32))
    env.close()
    print(f"Saved results to {out_dir}")

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        if pimac.share_parameters:
            pimac.agent_net.load_state_dict(ckpt["agent_state_dict"])
            pimac.target_agent_net.load_state_dict(ckpt["agent_state_dict"])
        else:
            pimac.agent_nets.load_state_dict(ckpt["agent_state_dict"])
            pimac.target_agent_nets.load_state_dict(ckpt["agent_state_dict"])
        pimac.teacher.load_state_dict(ckpt["teacher_state_dict"])

    pimac.temperature = 0.0
    pimac.set_eval_mode()

    env_viz = make_env(seed=seed + 200_000, render_mode="rgb_array")
    frames = []
    step_idx = 0
    for ep in range(10):
        obs, _ = env_viz.reset(seed=seed + 200_000 + ep)
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
                actions[aid] = pimac.act(o, agent_index=idx)
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
