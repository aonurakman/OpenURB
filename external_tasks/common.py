from __future__ import annotations

import os
import time
from typing import Optional, Sequence

import numpy as np

# Headless / deterministic matplotlib config (must be set before importing matplotlib).
# We intentionally redirect Matplotlib's cache to a repo-local folder so that running these
# scripts doesn't touch the user's global cache, and works cleanly in sandboxes/CI.
_EXTERNAL_TASKS_DIR = os.path.abspath(os.path.dirname(__file__))
_MPLCONFIGDIR = os.path.join(_EXTERNAL_TASKS_DIR, ".mplconfig")
os.makedirs(_MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPLCONFIGDIR)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


def set_global_seeds(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(env_name: str, algo_name: str) -> str:
    out_dir = os.path.join(_EXTERNAL_TASKS_DIR, "runs", env_name, algo_name, str(int(time.time())))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if x.size < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_curves(
    save_path: str,
    rewards: Sequence[float],
    losses: Sequence[float],
    eval_rewards: Sequence[float],
    window: int = 50,
) -> None:
    rewards_np = np.asarray(rewards, dtype=np.float32)
    losses_np = np.asarray(losses, dtype=np.float32)
    eval_np = np.asarray(eval_rewards, dtype=np.float32)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(rewards_np, alpha=0.35, label="train episode mean reward")
    axes[0].plot(moving_average(rewards_np, window), label=f"train {window}-ep moving avg")
    if eval_np.size:
        eval_x = np.linspace(0, len(rewards_np), num=eval_np.size, endpoint=False)
        axes[0].plot(eval_x, eval_np, marker="o", linewidth=1.5, label="eval reward")
    axes[0].set_ylabel("Reward (higher is better)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(losses_np, alpha=0.35, label="train episode mean loss")
    axes[1].plot(moving_average(losses_np, window), label=f"train {window}-ep moving avg")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_gif(
    frames: list,
    path: str,
    resize_to: Optional[tuple[int, int]] = (350, 350),
    duration_ms: int = 80,
) -> None:
    if not frames:
        return
    images = []
    for frame in frames:
        img = Image.fromarray(frame)
        if resize_to is not None:
            img = img.resize(resize_to, Image.BILINEAR)
        images.append(img)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
