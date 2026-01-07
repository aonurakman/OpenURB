#!/usr/bin/env python3

"""
Utility to reproduce recorded experiments defined in results/<exp_id>/exp_config.json.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"


def load_configuration(exp_id: str) -> dict:
    # Every finished run leaves behind exp_config.json under results/<id>/.
    # We load it so we can clone the original setup.
    exp_dir = RESULTS_ROOT / exp_id
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment '{exp_id}' not found in {RESULTS_ROOT}.")

    config_path = exp_dir / "exp_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No exp_config.json found for experiment '{exp_id}'.")

    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def remove_flags(tokens: Iterable[str], flags: Iterable[str]) -> List[str]:
    # Drop any flags that we plan to override, regardless of whether they use
    # the "--flag value" or "--flag=value" format.
    skip_flags = set(flags)
    cleaned: List[str] = []
    tokens = list(tokens)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        matched_flag = None
        for flag in skip_flags:
            if token == flag or token.startswith(f"{flag}="):
                matched_flag = flag
                break
        if matched_flag:
            if "=" not in token and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        cleaned.append(token)
        i += 1
    return cleaned


def next_experiment_id(base_id: str, override_seeds: bool) -> str:
    # Naming rules:
    #   * same seeds -> "<id>_repeated", adding "_N" when needed
    #   * new seeds  -> "<id>_v2" onwards
    if override_seeds:
        index = 2
        candidate = f"{base_id}_v{index}"
        while (RESULTS_ROOT / candidate).exists():
            index += 1
            candidate = f"{base_id}_v{index}"
        return candidate

    candidate = f"{base_id}_repeated"
    if not (RESULTS_ROOT / candidate).exists():
        return candidate

    index = 2
    while (RESULTS_ROOT / f"{candidate}_{index}").exists():
        index += 1
    return f"{candidate}_{index}"


def resolve_script_path(config: dict) -> Path:
    # The config stores the original script path; make sure it still exists.
    # If it's relative, treat it as repo-root relative.
    script_value = config.get("script")
    if not script_value:
        raise KeyError("Experiment configuration does not specify the 'script' path.")
    script_path = Path(script_value)
    if not script_path.is_absolute():
        script_path = (REPO_ROOT / script_path).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Recorded script path does not exist: {script_path}")
    return script_path


def build_command(config: dict, new_id: str, env_seed: int | None, torch_seed: int | None) -> List[str]:
    # Recreate the CLI call, but with the fresh experiment id and chosen seeds.
    script_path = resolve_script_path(config)
    base_command = [sys.executable, str(script_path)]

    if config.get("command"):
        tokens = shlex.split(config["command"])
        if tokens:
            tokens = tokens[1:]
        tokens = remove_flags(tokens, ("--id", "--env-seed", "--torch-seed"))
    else:
        tokens = []
        for flag, key in (
            ("--net", "network"),
            ("--alg-conf", "alg_config"),
            ("--task-conf", "task_config"),
            ("--env-conf", "env_config"),
        ):
            value = config.get(key)
            if value is not None:
                tokens.extend([flag, str(value)])

    command = base_command + ["--id", new_id] + tokens
    if env_seed is not None:
        command.extend(["--env-seed", str(env_seed)])
    if torch_seed is not None:
        command.extend(["--torch-seed", str(torch_seed)])
    return command


def main() -> None:
    # Wire it all together: parse args, pick the new ID, and fire off the job.
    parser = argparse.ArgumentParser(
        description="Reproduce a recorded experiment using its saved exp_config.json."
    )
    parser.add_argument("--id", required=True, help="Existing experiment identifier in results/.")
    parser.add_argument("--torch-seed", type=int, help="Override the recorded torch seed.")
    parser.add_argument("--env-seed", type=int, help="Override the recorded environment seed.")
    args = parser.parse_args()

    config = load_configuration(args.id)

    recorded_env_seed = config.get("env_seed")
    recorded_torch_seed = config.get("torch_seed")

    env_seed = args.env_seed if args.env_seed is not None else recorded_env_seed
    torch_seed = args.torch_seed if args.torch_seed is not None else recorded_torch_seed

    override_seeds = args.env_seed is not None or args.torch_seed is not None
    new_exp_id = next_experiment_id(args.id, override_seeds)

    command = build_command(config, new_exp_id, env_seed, torch_seed)

    print(f"Reproducing experiment '{args.id}' as '{new_exp_id}'.")
    print("Executing:", " ".join(command))

    result = subprocess.run(command, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
