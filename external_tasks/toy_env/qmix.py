import os
import sys

import numpy as np
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from algorithms.qmix import QMIX
from external_tasks.common import make_run_dir, plot_curves, save_gif, set_global_seeds
from external_tasks.toy_env.env import CoopLineWorldConfig, CoopLineWorldParallelEnv


def make_env(seed: int, render_mode=None):
    env = CoopLineWorldParallelEnv(CoopLineWorldConfig(n_agents=3, goal=8, max_steps=25), render_mode=render_mode)
    env.reset(seed=seed)
    return env


def main():
    seed = 42
    episodes = 4000
    learning_starts = 400
    learn_every_steps = 4
    eval_every_episodes = 200
    eval_episodes = 20
    min_improve = 1e-3
    frame_skip = 1
    max_frames = 400

    set_global_seeds(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = make_env(seed=seed, render_mode=None)
    agent_ids = list(env.possible_agents)
    n_agents = len(agent_ids)

    obs_size = env.observation_space(agent_ids[0]).shape[0]
    action_space_size = env.action_space(agent_ids[0]).n
    global_state_size = int(np.asarray(env.state(), dtype=np.float32).reshape(-1).shape[0])

    qmix_kwargs = dict(
        temp_init=1.0,
        temp_decay=0.999,
        temp_min=0.05,
        buffer_size=100_000,
        batch_size=128,
        lr=3e-4,
        num_epochs=1,
        num_hidden=2,
        widths=(64, 64, 64),
        rnn_hidden_dim=64,
        mixing_embed_dim=64,
        hypernet_embed=128,
        max_grad_norm=10.0,
        gamma=0.99,
        target_update_every=200,
        double_q=True,
        tau=1.0,
        share_parameters=False,
        mixing_weight_clip=None,
        q_tot_clip=None,
        use_huber_loss=True,
    )
    qmix = QMIX(obs_size, action_space_size, num_agents=n_agents, global_state_size=global_state_size, device=device, **qmix_kwargs)

    out_dir = make_run_dir("coop_line_world", "qmix")
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pt")

    episode_rewards = []
    episode_losses = []
    eval_rewards = []
    global_step = 0
    best_eval = -float("inf")

    def run_eval(start_seed: int) -> float:
        temp_backup = qmix.temperature
        qmix.temperature = 0.0
        qmix.set_eval_mode()

        returns = []
        for k in range(eval_episodes):
            obs, _ = env.reset(seed=start_seed + k)
            qmix.reset_episode()
            term = {aid: False for aid in agent_ids}
            trunc = {aid: False for aid in agent_ids}
            total = 0.0
            while True:
                actions = {}
                for idx, aid in enumerate(agent_ids):
                    if term[aid] or trunc[aid]:
                        continue
                    actions[aid] = qmix.act(obs[aid], agent_index=idx)
                obs, rewards, terminations, truncations, _ = env.step(actions)
                for aid in agent_ids:
                    total += float(rewards.get(aid, 0.0))
                term = {k: bool(v) for k, v in terminations.items()}
                trunc = {k: bool(v) for k, v in truncations.items()}
                if all(term.get(a, False) or trunc.get(a, False) for a in agent_ids):
                    break
            returns.append(total / n_agents)

        qmix.temperature = temp_backup
        qmix.set_train_mode()
        return float(np.mean(returns))

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        qmix.reset_episode()
        term = {aid: False for aid in agent_ids}
        trunc = {aid: False for aid in agent_ids}

        ep_reward_sum = 0.0
        loss_start = len(qmix.loss)

        while True:
            state_before = np.asarray(env.state(), dtype=np.float32).reshape(-1)

            obs_batch = np.zeros((n_agents, obs_size), dtype=np.float32)
            actions_batch = np.zeros(n_agents, dtype=np.int64)
            active_mask = np.ones(n_agents, dtype=np.float32)

            actions = {}
            for idx, aid in enumerate(agent_ids):
                if term[aid] or trunc[aid]:
                    active_mask[idx] = 0.0
                    continue
                o = obs[aid]
                obs_batch[idx] = o
                a = qmix.act(o, agent_index=idx)
                actions_batch[idx] = int(a)
                actions[aid] = int(a)

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            global_step += 1
            state_after = np.asarray(env.state(), dtype=np.float32).reshape(-1)

            next_obs_batch = np.zeros((n_agents, obs_size), dtype=np.float32)
            next_active_mask = np.ones(n_agents, dtype=np.float32)
            rewards_batch = np.zeros(n_agents, dtype=np.float32)

            for idx, aid in enumerate(agent_ids):
                rewards_batch[idx] = float(rewards.get(aid, 0.0))
                ep_reward_sum += float(rewards_batch[idx])
                if bool(terminations.get(aid, False)) or bool(truncations.get(aid, False)):
                    next_active_mask[idx] = 0.0
                else:
                    next_obs_batch[idx] = next_obs[aid]

            done = all(bool(terminations.get(a, False)) or bool(truncations.get(a, False)) for a in agent_ids)

            qmix.store_transition(
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
                qmix.learn()

            obs = next_obs
            term = {k: bool(v) for k, v in terminations.items()}
            trunc = {k: bool(v) for k, v in truncations.items()}
            if done:
                break

        new_losses = qmix.loss[loss_start:]
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
                        "env_name": "coop_line_world",
                        "seed": seed,
                        "qmix_kwargs": qmix_kwargs,
                        "agent_state_dict": (
                            qmix.agent_net.state_dict() if qmix.share_parameters else qmix.agent_nets.state_dict()
                        ),
                        "mixing_state_dict": qmix.mixing_net.state_dict(),
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
                f"loss={episode_losses[-1]:.5f} temp={qmix.temperature:.3f}"
            )

    plot_curves(os.path.join(out_dir, "learning_curves.png"), episode_rewards, episode_losses, eval_rewards, window=50)
    np.save(os.path.join(out_dir, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))
    np.save(os.path.join(out_dir, "episode_losses.npy"), np.asarray(episode_losses, dtype=np.float32))
    np.save(os.path.join(out_dir, "eval_rewards.npy"), np.asarray(eval_rewards, dtype=np.float32))
    env.close()
    print(f"Saved results to {out_dir}")

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        qmix.mixing_net.load_state_dict(ckpt["mixing_state_dict"])
        qmix.target_mixing_net.load_state_dict(ckpt["mixing_state_dict"])
        if qmix.share_parameters:
            qmix.agent_net.load_state_dict(ckpt["agent_state_dict"])
            qmix.target_agent_net.load_state_dict(ckpt["agent_state_dict"])
        else:
            qmix.agent_nets.load_state_dict(ckpt["agent_state_dict"])
            qmix.target_agent_nets.load_state_dict(ckpt["agent_state_dict"])

    qmix.temperature = 0.0
    qmix.set_eval_mode()

    env_viz = make_env(seed=seed + 200_000, render_mode="rgb_array")
    frames = []
    step_idx = 0
    for ep in range(10):
        obs, _ = env_viz.reset(seed=seed + 200_000 + ep)
        qmix.reset_episode()
        term = {aid: False for aid in agent_ids}
        trunc = {aid: False for aid in agent_ids}
        while True:
            actions = {}
            for idx, aid in enumerate(agent_ids):
                if term[aid] or trunc[aid]:
                    continue
                actions[aid] = qmix.act(obs[aid], agent_index=idx)
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
