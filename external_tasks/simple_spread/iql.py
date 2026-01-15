import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch

from algorithms.iql import DQN
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
    frame_skip = 2
    max_frames = 500

    set_global_seeds(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = make_env(seed=seed, render_mode=None)
    agent_ids = list(env.possible_agents)
    n_agents = len(agent_ids)

    obs_size = env.observation_space(agent_ids[0]).shape[0]
    action_space_size = env.action_space(agent_ids[0]).n

    dqn_kwargs = dict(
        eps_init=1.0,
        eps_decay=0.999,
        eps_min=0.05,
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
    models = {agent_id: DQN(obs_size, action_space_size, device=device, **dqn_kwargs) for agent_id in agent_ids}

    out_dir = make_run_dir("simple_spread_v3", "iql")
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pt")

    episode_rewards = []
    episode_losses = []
    eval_rewards = []
    global_step = 0
    best_eval = -float("inf")

    def run_eval(start_seed: int) -> float:
        eps_backup = {aid: models[aid].epsilon for aid in agent_ids}
        for aid in agent_ids:
            models[aid].epsilon = 0.0
            models[aid].value_network.eval()
            models[aid].target_network.eval()

        returns = []
        for k in range(eval_episodes):
            obs, _ = env.reset(seed=start_seed + k)
            term = {aid: False for aid in agent_ids}
            trunc = {aid: False for aid in agent_ids}
            total = 0.0

            while True:
                actions = {}
                for aid in agent_ids:
                    if term[aid] or trunc[aid]:
                        continue
                    actions[aid] = models[aid].act(obs[aid])
                next_obs, rewards, terminations, truncations, _ = env.step(actions)
                for aid in agent_ids:
                    total += float(rewards.get(aid, 0.0))
                obs = next_obs
                term = {k: bool(v) for k, v in terminations.items()}
                trunc = {k: bool(v) for k, v in truncations.items()}
                if all(term.get(a, False) or trunc.get(a, False) for a in agent_ids):
                    break

            returns.append(total / n_agents)

        for aid in agent_ids:
            models[aid].epsilon = eps_backup[aid]
            models[aid].value_network.train()
        return float(np.mean(returns))

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
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
                a = models[aid].act(obs[aid])
                actions[aid] = a
                prev_obs[aid] = obs[aid]
                prev_actions[aid] = a

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            global_step += 1

            for aid in agent_ids:
                r = float(rewards.get(aid, 0.0))
                done = bool(terminations.get(aid, False) or truncations.get(aid, False))
                ns = next_obs.get(aid, np.zeros(obs_size, dtype=np.float32))
                if aid in prev_actions:
                    models[aid].push_transition(prev_obs[aid], prev_actions[aid], r, ns, done)
                ep_reward_sum += r

            if (global_step >= learning_starts) and ((global_step % learn_every_steps) == 0):
                for aid in agent_ids:
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
                        "dqn_kwargs": dqn_kwargs,
                        "value_state_dicts": {aid: models[aid].value_network.state_dict() for aid in agent_ids},
                        "target_state_dicts": {aid: models[aid].target_network.state_dict() for aid in agent_ids},
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
                f"loss={episode_losses[-1]:.5f} eps~{np.mean([m.epsilon for m in models.values()]):.3f}"
            )

    plot_curves(os.path.join(out_dir, "learning_curves.png"), episode_rewards, episode_losses, eval_rewards, window=50)
    np.save(os.path.join(out_dir, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))
    np.save(os.path.join(out_dir, "episode_losses.npy"), np.asarray(episode_losses, dtype=np.float32))
    np.save(os.path.join(out_dir, "eval_rewards.npy"), np.asarray(eval_rewards, dtype=np.float32))
    env.close()
    print(f"Saved results to {out_dir}")

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        for aid in agent_ids:
            models[aid].value_network.load_state_dict(ckpt["value_state_dicts"][aid])
            models[aid].target_network.load_state_dict(ckpt["target_state_dicts"][aid])

    # Save a GIF of the best policy (epsilon=0).
    for aid in agent_ids:
        models[aid].epsilon = 0.0
        models[aid].value_network.eval()

    env_viz = make_env(seed=seed + 200_000, render_mode="rgb_array")
    frames = []
    step_idx = 0
    for ep in range(5):
        obs, _ = env_viz.reset(seed=seed + 200_000 + ep)
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
