<!--
This file is generated from docs/README.template.md. Run: venv/bin/python tools/build_readme.py
-->
##### Forked from [URB](https://urbenchmark.com)

<p align="center">
  <img src="docs/openurb.png" align="center" width="30%"/>
</p>

# OpenURB
[![Leaderboard](https://img.shields.io/badge/Leaderboard-OpenURB-0aa4f6?style=for-the-badge&logo=github)](https://coexistence-project.github.io/OpenURB/)

OpenURB is a benchmark for learning route-choice policies in mixed-autonomy traffic under changing AV population sizes. It uses RouteRL's SUMO-based `TrafficEnvironment` and keeps a stable set of benchmark methods, experiment scripts, reusable configs, and KPI tooling.

The benchmark is method-agnostic. Its purpose is to make experiments comparable across algorithms under the same route-choice task definitions, switching rules, and evaluation pipeline.

## Local agent guidance

If you are a coding agent working in this repo, read `AGENTS.md` and the local notes under `.agents/` before editing anything.

## Workflow

Each experiment script follows the same high-level routine:
- load algorithm, environment, and task configs from `config/`,
- build the RouteRL traffic environment on one of the networks under `networks/`,
- run the human-driver learning phase expected by RouteRL,
- train the AV policy with the selected method,
- apply open or conditioned switching from the task config,
- run evaluation episodes,
- save outputs under `results/<exp_id>/`,
- compute KPIs with `analysis/metrics.py`.

## Setup

### Prerequisites

Install SUMO separately by following the instructions at [SUMO installation](https://sumo.dlr.de/docs/Installing/index.html).

### Clone the repository

```bash
git clone https://github.com/COeXISTENCE-PROJECT/OpenURB.git
cd OpenURB
```

### Create an environment and install dependencies

```bash
python -m venv venv
venv/bin/pip install --force-reinstall --no-cache-dir -r requirements.txt
```

## Networks and demand patterns

Urban Routing Benchmark can be executed on a wide set of networks coupled with demand patterns.

We provide:
- [RouteRL](https://github.com/COeXISTENCE-PROJECT/RouteRL) networks.
- Some [RESCO](https://github.com/Pi-Star-Lab/RESCO) networks where routing is possible.
- A set of 25 small cuts from Ile-de-France based on the synthetic agent-based model.

You can run OpenURB on your custom OSM-derived or hand-made SUMO network with a demand pattern
from a custom source. For compatibility, consult the
[RouteRL documentation](https://coexistence-project.github.io/RouteRL/).

## Running experiments

### Learning methods

Use one of the OpenURB experiment scripts:

```bash
venv/bin/python scripts/<script_name>.py [--id <exp_id>] --alg-conf <alg_conf> --env-conf <env_conf> --task-conf <task_conf> --net <net_name> --env-seed <env_seed> --torch-seed <torch_seed>
```

where:
- `<script_name>` is one of `open_iql`, `cond_open_iql`, `open_ippo`, `cond_open_ippo`, `open_qmix`, `cond_open_qmix`, `open_vdn`, `cond_open_vdn`, `open_mappo`, `cond_open_mappo`, `open_pimac`, or `cond_open_pimac`.
- `<alg_conf>` is the JSON config name from `config/algo_config/<algorithm>/` without the `.json` suffix.
- `<env_conf>` is the JSON config name from `config/env_config/` without the suffix. It is optional and defaults to `config1`.
- `<task_conf>` is the JSON config name from `config/task_config/` without the suffix.
- `<net_name>` is one of the network folders under `networks/`.
- `<env_seed>` and `<torch_seed>` are optional reproducibility seeds; both default to `42`.

If `--id` is omitted, the scripts generate an experiment id from the algorithm, network, configs, and seeds. Conditional scripts prepend `c_`.

Example:

```bash
venv/bin/python scripts/open_mappo.py --alg-conf config1 --task-conf config1 --net saint_arnoult --env-seed 42 --torch-seed 0
```

### Baselines

For non-learning baselines, use:

```bash
venv/bin/python scripts/open_baselines.py [--id <exp_id>] --alg-conf <alg_conf> --env-conf <env_conf> --task-conf <task_conf> --net <net_name> --env-seed <env_seed> --model <model_name>
```

The conditioned baseline script is `scripts/cond_open_baselines.py`.

## Scripts

OpenURB keeps one script pair per benchmark method:
- `open_iql.py` / `cond_open_iql.py`
- `open_ippo.py` / `cond_open_ippo.py`
- `open_qmix.py` / `cond_open_qmix.py`
- `open_vdn.py` / `cond_open_vdn.py`
- `open_mappo.py` / `cond_open_mappo.py`
- `open_pimac.py` / `cond_open_pimac.py`

The `open_` scripts use the switching schedule defined directly in the task config.
The `cond_` scripts scale switching probabilities by the travel-time ratio between the groups.

Baseline scripts are `open_baselines.py` and `cond_open_baselines.py`.

Every script:
- loads algorithm, environment, and task configs,
- runs the SUMO/RouteRL experiment,
- writes outputs under `results/<exp_id>/`,
- and calls `analysis/metrics.py` at the end.

MAPPO scripts additionally write `results/<exp_id>/mappo_loss_history.json`.
PIMAC scripts additionally write `results/<exp_id>/pimac_loss_history.json`.

### Weights & Biases logging

All experiment scripts can log to Weights & Biases.

1. Create `wandb_config.json` in the repo root (gitignored):

```json
{
  "api_key": "YOUR_WANDB_API_KEY",
  "project": "openurb",
  "entity": "your_team_or_user"
}
```

2. Run any script with the usual arguments, for example:

```bash
venv/bin/python scripts/open_iql.py --alg-conf config1 --task-conf config1 --net saint_arnoult --wandb-config wandb_config.json
```

If you do not want W&B logging, pass `--no-wandb`.

## Baseline models

Baseline policies can be run with `scripts/open_baselines.py` and `scripts/cond_open_baselines.py`.

Available models:
- `aon`: deterministically chooses the shortest free-flow route.
- `random`: samples routes uniformly at random.
- `gawron`: provided by RouteRL and used as the human-driver learning model.

## Results

Provided experiment scripts store experiment results in this directory, under the folder name determined by the **experiment identifier**.

The structure of this result data is demonstrated with some sample results provided in this directory. In summary, this data includes:
- Experiment configuration values (`exp_config.json`).
- Demand and route generation data. (XML and CSV files.)
- Tracked loss values for the used algorithm. (`losses/`)
- Episode-level data logs. (`episodes/`)
- Simulation statistics yielded by SUMO. (`SUMO_output/`)
- Experiment data visualizations from RouteRL. (`plots/`)
- Calculated URB KPIs, (`metrics/`)
- Population switch logs (if applicable). (`shifts.csv`)
- Runtime resource utilization statistics. (`runtime.json`)

## Tools

Helpers for managing experiment runs under `results/`.

### `rename.py`
Rename an experiment directory and update text references to the old id inside it.

```bash
venv/bin/python tools/rename.py <old_id> [new_id]
```

### `reproduce.py`
Replay an experiment from `results/<exp_id>/exp_config.json`, optionally with new seeds.

```bash
venv/bin/python tools/reproduce.py --id <existing_exp_id> [--env-seed <seed>] [--torch-seed <seed>]
```

### `run_todo.py`
Run commands from a todo file in batches and capture stdout/stderr under the corresponding result directory.

```bash
venv/bin/python tools/run_todo.py todo.txt --jobs 3
```

## Analysis

`analysis/metrics.py` converts saved experiment outputs into KPI tables and plots.

### Usage

Process one experiment:

```bash
venv/bin/python analysis/metrics.py --id <exp_id>
```

Process every experiment under `results/`:

```bash
venv/bin/python analysis/metrics.py --all
```

Useful options:
- `--results-folder <path>`: use a non-default results directory.
- `--no-skip`: recompute metrics even if they already exist.
- `--jobs <n>`: use multiple workers with `--all`.
- `--verbose True`: print more progress information.

Example:

```bash
venv/bin/python analysis/metrics.py --all --no-skip --jobs 4
```

### Reported indicators

The core outcome is travel time. Metrics are reported for the full population and separately for HDVs and AVs across the training, dynamic-switching, and test phases.

Derived indicators include:
- AV advantage,
- effects of switching to or remaining in the HDV population,
- switch cost,
- dynamic recovery,
- dynamic volatility,
- dynamic instability,
- dynamic time excess,
- cumulative training cost,
- and switch statistics from `shifts.csv` when present.

### Units
- Time: minutes
- Distance: kilometers
- Speed: kilometers per hour
