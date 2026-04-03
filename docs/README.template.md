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

<!-- INCLUDE: ../networks/readme.md -->

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

<!-- INCLUDE: ../scripts/readme.md -->

<!-- INCLUDE: ../baseline_models/readme.md -->

<!-- INCLUDE: ../results/readme.md -->

<!-- INCLUDE: ../tools/readme.md -->

<!-- INCLUDE: ../analysis/readme.md -->
