<!--
This file is generated from docs/README.template.md. Run: python tools/build_readme.py
-->
##### Forked from [URB](https://urbenchmark.com)

<p align="center">
  <img src="docs/openurb.png" align="center" width="30%"/>
</p>

# OpenURB
[![Leaderboard](https://img.shields.io/badge/Leaderboard-OpenURB-0aa4f6?style=for-the-badge&logo=github)](https://coexistence-project.github.io/OpenURB/)

There is a microscopic traffic network in which two agent populations, human-driven vehicles (HDVs) and autonomous vehicles (AVs), make **single-step** route-choice decisions. At each departure time, a driver observes current traffic conditions, selects one of *n* routes from its origin to its destination, receives a travel-time-based reward, and the episode ends for that driver. HDVs aim to optimize individual travel time, while AVs **collectively** try to minimize group travel time.

Unlike static formulations, population composition is **dynamic**: humans may switch to AVs, or AV owners may revert to HDVs with some predefined probabilities. This creates a **non-stationary**, **variable-sized** multi-agent system where each action’s impact and optimality depend on the evolving mix of agents. Therefore, a good solution must handle the appearance of new cooperators and the elimination of known teammates, successfully scaling at **open environments**.

The task is to develop and evaluate RL methods that enable AV agents to learn robust, high-performing routing policies **under these switching dynamics,** while accounting for large group sizes and interactions with human drivers. The ultimate objective of this research is to **(i)** find the problem setting where the current approaches come short, **(ii)** quantify their shortcomings with relevant (potentially novel) dynamic cooperation performance metrics, and **(iii)** contribute with methodological extensions.

### Keywords

`MARL`, `route choice`, `open environment`, `ad-hoc teamwork`, `few-shot cooperation`

## Agent Guidance (Local)

If you are a coding agent working in this repo, read `AGENTS.md` and the docs under `.agents/` before making changes. These files are gitignored and live locally.

## 🔗 Workflow

`OpenURB` (similar to `URB`):
* Runs an experiment script using the `TrafficEnvironment` from `RouteRL`,
* With a RL algorithm or a baseline method (from `baseline_models/`),
* Opens algorithm, environment and task configuration files from `config/`,
* Loads the network and demand from `networks`
* Executes a typical `RouteRL` routine of
   * first learning of human drivers,
   * which then 'mutate` to CAVs,
   * are trained to optimize routing policies with the implemented algorithm,
   * THEN possibly simulating dynamic switches between humans and AVs.
* When the training is finished, it uses raw results to compute a wide-set of KPIs.

---

## 📦 Setup

#### Prerequisites

Make sure you have SUMO installed in your system. This procedure should be carried out separately, by following the instructions provided [here](https://sumo.dlr.de/docs/Installing/index.html).

#### Cloning repository

Clone the **OpenURB** repository from GitHub by

```bash
git clone https://github.com/COeXISTENCE-PROJECT/OpenURB.git
```

#### Creating environment

- **Option 1** (Recommended): Create a virtual environment with `venv`:

```bash
python -m venv .venv
```

and then install dependencies by:

```bash
cd URB
pip install --force-reinstall --no-cache-dir -r requirements.txt
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

## 🔬 Running experiments

#### Usage of **OpenURB** for Reinforcement Learning algorithms

To run the open/cond-open RL experiments, use:

```bash
python scripts/<script_name>.py [--id <exp_id>] --alg-conf <hyperparam_id> --env-conf <env_conf_id> --task-conf <task_id> --net <net_name> --env-seed <env_seed> --torch-seed <torch_seed>
```

where

- ```<script_name>``` is the script you wish to run, available scripts are ```open_iql```, ```cond_open_iql```, ```open_ippo```, ```cond_open_ippo```, ```open_qmix```, ```cond_open_qmix```, ```open_vdn```, ```cond_open_vdn```, ```open_mappo```, ```cond_open_mappo```, ```open_pimac_v0```, ```cond_open_pimac_v0```, ```open_pimac_v1```, ```cond_open_pimac_v1```, ```open_pimac_v2```, ```cond_open_pimac_v2```, ```open_pimac_v3```, and ```cond_open_pimac_v3```,
- ```<exp_id>``` is an optional experiment identifier, for instance ```random_ing```,
- ```<hyperparam_id>``` is the hyperparameterization identifier, it must correspond to a `.json` filename (without extension) in [`config/algo_config`](config/algo_config/). Provided scripts automatically select the algorithm-specific subfolder in this directory.
- ```<env_conf_id>``` is the environment configuration identifier. It must correspond to a `.json` filename (without extension) in [`config/env_config`](config/env_config/). It is used to parameterize environment-specific processes, such as path generation, disk operations, etc. It is **optional** and by default is set to `config1`.
- ```<task_id>``` is the task configuration identifier. It must correspond to a `.json` filename (without extension) in [`config/task_config`](config/task_config/). For this repo, use the provided task configs (e.g., `config1`–`config5`) which already encode dynamic switching parameters.
- ```<net_name>``` is the name of the network you wish to use. Must be one of the folder names in ```networks/``` i.e. ```ing_small```, ```ingolstadt_custom```, ```nangis```, ```nemours```, ```provins``` or ```saint_arnoult```,
- ```<env_seed>``` is reproducibility random seed for the traffic environment, it is **optional** and by default is set to 42,
- ```<torch_seed>``` is reproducibility random seed for PyTorch, it is **optional** and by default is set to 42.

If `--id` is omitted, scripts auto-generate an experiment ID of the form
`<alg-name>_<net>_a<alg_config>_e<env_config>_t<task_config>_<env_seed>_<torch_seed>`.
Conditional scripts prepend `c_` to the ID. For baselines, `<alg-name>` is the selected model and the torch seed is omitted.
If the generated ID already exists under `results/`, `_repeated` is appended.

For example, the following command runs an experiment using:
- IQL algorithm, hyperparameterized by `config/algo_config/iql/config1.json`,
- The task specified in `config/task_config/config1.json`,
- The environment parameterization specified in `config/env_config/config1.json` (by default),
- Experiment identifier `deneme`, which will be used as the folder name in `results/` to save the experiment data,
- Saint Arnoult network and demand, from `networks/saint_arnoult`,
- Environment (also used for `random` and `numpy`) and PyTorch seeds 42 and 0, respectively.

```bash
python scripts/open_iql.py --id deneme --alg-conf config1 --task-conf config1 --net saint_arnoult --env-seed 42 --torch-seed 0
```

Example for QMIX:

```bash
python scripts/open_qmix.py --id deneme_qmix --alg-conf config1 --task-conf config1 --net saint_arnoult --env-seed 42 --torch-seed 0
```

Example for VDN:

```bash
python scripts/open_vdn.py --id deneme_vdn --alg-conf config1 --task-conf config1 --net saint_arnoult --env-seed 42 --torch-seed 0
```

> All experiment scripts in this repo expect task configs from `config/task_config/`; those files define the dynamic switching parameters.

#### Usage **URB** for baselines

Similarly as for RL algorithms, you have to provide command, but there is one additional flag ```model``` for ```scripts/open_baselines.py```, and ```scripts/cond_open_baselines.py```, instead of ```torch-seed```, then you have command of form:

```bash
python scripts/open_baselines.py [--id <exp_id>] --alg-conf <hyperparam_id> --env-conf <env_conf_id> --task-conf <task_id> --net <net_name> --env-seed <env_seed> --model <model_name>
```

For a list of available baseline models, see the **Baseline models** section below.
The open baseline scripts mirror the dynamic switching behavior: use task configs from `config/task_config/`, and `cond_open_baselines.py` conditions switches on group travel times.

For example:

```bash
python scripts/open_baselines.py --id ing_aon --alg-conf config1 --task-conf config2 --net ingolstadt_custom --model aon
```

## Scripts

We provide training scripts for open vs. conditional switching variants:
- `open_ippo.py` runs a simplified IPPO/PPO setup with open (predefined) switching.
- `open_iql.py` runs an IQL setup with open switching.
- `open_qmix.py` runs a QMIX setup with open switching.
- `open_vdn.py` runs a VDN (Value Decomposition Networks) setup with open switching.
- `open_mappo.py` runs canonical MAPPO (CTDE) with open switching.
- `open_pimac_v0.py` runs MAPPO-style PPO with a Deep-Sets centralized critic (set-encoder baseline).
- `open_pimac_v1.py` runs PIMAC_v1 (PIMAC_v0 + token cross-attention teacher and student-context distillation).
- `open_pimac_v2.py` runs PIMAC_v2 (PIMAC_v1 + uncertainty-gated FiLM policy conditioning).
- `open_pimac_v3.py` runs PIMAC_v3 (PIMAC_v2 + context-conditioned low-rank hypernetwork policy-head residuals).
- `cond_` prepend signifies dynamic switching probabilities based on group travel time ratio.

Baseline scripts are `open_baselines.py` and `cond_open_baselines.py` (see `baseline_models/readme.md`
for available models). The open variants run dynamic switching (conditional in the `cond_` version)
and require task configs from `config/task_config/`.

All scripts automatically run `analysis/metrics.py` at the end of an experiment to generate KPI outputs
in the experiment's `results/<exp_id>/metrics/` folder.

MAPPO scripts additionally persist per-update optimization diagnostics to
`results/<exp_id>/mappo_loss_history.json` (while keeping `losses.csv` and mean-loss plots).

PIMAC_v0 scripts additionally persist per-update optimization diagnostics to
`results/<exp_id>/pimac_v0_loss_history.json` (while keeping `losses.csv` and mean-loss plots).

PIMAC_v1 scripts additionally persist per-update optimization diagnostics to
`results/<exp_id>/pimac_v1_loss_history.json` (while keeping `losses.csv` and mean-loss plots).

PIMAC_v2 scripts additionally persist per-update optimization diagnostics to
`results/<exp_id>/pimac_v2_loss_history.json` (while keeping `losses.csv` and mean-loss plots).

PIMAC_v3 scripts additionally persist per-update optimization diagnostics to
`results/<exp_id>/pimac_v3_loss_history.json` (while keeping `losses.csv` and mean-loss plots).

### Weights & Biases logging

All experiment scripts support Weights & Biases logging and will stream per-episode mean rewards and
travel times (overall + by agent kind) as episode CSVs are written to disk.

1. Create `wandb_config.json` in the repo root (gitignored) with your W&B settings:

```json
{
  "api_key": "YOUR_WANDB_API_KEY",
  "project": "openurb",
  "entity": "your_team_or_user"
}
```

   Alternatively, set the `WANDB_API_KEY` environment variable and omit `api_key` from the file.

2. Run:

```bash
python scripts/open_iql.py [--id <exp_id>] --alg-conf <hyperparam_id> --task-conf <task_id> --net <net_name> [--env-conf <env_conf_id>] [--env-seed <env_seed>] [--torch-seed <torch_seed>] [--wandb-config <path>]
```

If `--id` is omitted, scripts auto-generate one and print it before starting.

## external_tasks (sanity checks)

These scripts are **not unit tests**. They run short MARL training loops to sanity-check that the
algorithms in `algorithms/` can learn on small **multi-step** environments.

Outputs are written under `external_tasks/runs/<env>/<algo>/<timestamp>/`:
- `learning_curves.png`
- `episode_rewards.npy`, `episode_losses.npy`, `eval_rewards.npy`
- `mappo_loss_history.json` (for MAPPO runs; per-update diagnostics)
- `pimac_v0_loss_history.json` (for PIMAC_v0 runs; set-critic MAPPO diagnostics)
- `pimac_v1_loss_history.json` (for PIMAC_v1 runs; token teacher-student diagnostics)
- `pimac_v2_loss_history.json` (for PIMAC_v2 runs; FiLM-gated token teacher-student diagnostics)
- `pimac_v3_loss_history.json` (for PIMAC_v3 runs; FiLM + hypernetwork token teacher-student diagnostics)
- `best_checkpoint.pt` (best eval reward)
- `policy_rollout.gif` (rollout of the best checkpoint; always headless)

PIMAC_v* sanity scripts use MAPPO-style on-policy optimization with a token/set-based centralized critic
and progressively stronger decentralized policy conditioning.

### Environments

- `simple_spread_v3` (PettingZoo MPE): `external_tasks/simple_spread/`
- `coop_line_world` (tiny built-in toy env): `external_tasks/toy_env/`

### Dependencies (for PettingZoo)

```bash
pip install "pettingzoo[mpe]" pygame pillow
```

### Run

```bash
python external_tasks/simple_spread/random_policy.py
python external_tasks/simple_spread/iql.py
python external_tasks/simple_spread/ippo.py
python external_tasks/simple_spread/qmix.py
python external_tasks/simple_spread/vdn.py
python external_tasks/simple_spread/mappo.py
python external_tasks/simple_spread/pimac_v0.py
python external_tasks/simple_spread/pimac_v1.py
python external_tasks/simple_spread/pimac_v2.py
python external_tasks/simple_spread/pimac_v3.py

python external_tasks/toy_env/random_policy.py
python external_tasks/toy_env/iql.py
python external_tasks/toy_env/ippo.py
python external_tasks/toy_env/qmix.py
python external_tasks/toy_env/vdn.py
python external_tasks/toy_env/mappo.py
python external_tasks/toy_env/pimac_v0.py
python external_tasks/toy_env/pimac_v1.py
python external_tasks/toy_env/pimac_v2.py
python external_tasks/toy_env/pimac_v3.py

python external_tasks/simple_spread_dynamic/random_policy.py
python external_tasks/simple_spread_dynamic/iql.py
python external_tasks/simple_spread_dynamic/ippo.py
python external_tasks/simple_spread_dynamic/qmix.py
python external_tasks/simple_spread_dynamic/vdn.py
python external_tasks/simple_spread_dynamic/mappo.py
python external_tasks/simple_spread_dynamic/pimac_v0.py
python external_tasks/simple_spread_dynamic/pimac_v1.py
python external_tasks/simple_spread_dynamic/pimac_v2.py
python external_tasks/simple_spread_dynamic/pimac_v3.py
```

## Baseline models

Baseline algorithms can be run with `scripts/open_baselines.py` and `scripts/cond_open_baselines.py`.
Available model options:

- **Baselines included in OpenURB**
  - `aon` deterministically picks the shortest free-flow route regardless of congestion.
  - `random` selects routes uniformly at random.
- **Additionally available from RouteRL**
  - `gawron` (base human learning model) follows Gawron (1998) and iteratively shifts cost
    expectations toward received rewards.

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

Helpers for managing and reproducing experiment results under `results/`.

### rename.py

Rename an experiment directory and update any text references to the old id inside it.

Usage:

```bash
python tools/rename.py <old_id> [new_id]
```

If `new_id` is omitted, a random 5-character alphanumeric id is generated.

### reproduce.py

Every run saves its configuration under `results/<exp_id>/exp_config.json`. Use this
script to replay an experiment with the recorded parameters or with new seeds.

Usage:

```bash
python tools/reproduce.py --id <existing_exp_id> [--env-seed <seed>] [--torch-seed <seed>]
```

The helper stores outputs alongside the original results: plain repeats become
`<id>_repeated`, while seed overrides yield `<id>_v2`, `<id>_v3`, and so on.

### run_todo.py

Run the commands in `todo.txt` in parallel batches and save each command's stdout/stderr
to `results/<exp_id>/stdout.log` (an `--id` is auto-added when missing).

Usage:

```bash
python tools/run_todo.py todo.txt --jobs 3
```

## 📊 Calculating Metrics and indicators  

Each experiment outputs set of raw records, which are then processed with the script in this folder for a set of performance indicators which we report and several additional metrics that track the quality of the solution and its impact to the system.

#### Usage

To use the analysis script, you have to provide in the command line the following command:

```bash
python analysis/metrics.py --id <exp_id> --verbose <verbose> --results-folder <results-folder> --skip-clearing <skip-clearing> --skip-collecting <skip-collecting>
```

that will collect the results from the experiment with identifier ```<exp_id>``` and save them in the
folder ```<exp_id>/metrics/```. The ```--verbose``` flag is optional and if set to ```True``` will
print additional information about the analysis process. Flag ```--results-folder``` is optional
and, if set, will use the folder ```<results-folder>``` instead of the default one ```results/```.
The flags ```--skip-clearing``` and ```--skip-collecting``` are optional and if set to ```True```
will skip clearing and collecting the results from the experiment, respectively. Those operations
have to be done only once, so if you are running the analysis script multiple times, you can skip
them.
To process every experiment under `results/`, use `python analysis/metrics.py --all` (skips runs that already have metrics unless you add `--no-skip`). For simple parallelism, add `--jobs <n>`. Use `--id` for a single experiment; it cannot be combined with `--all` (and `--jobs` is ignored unless `--all` is set).

Examples:

```bash
# Single experiment
python analysis/metrics.py --id open_aon_1 --verbose True
```

```bash
# All experiments, skip those with existing metrics
python analysis/metrics.py --all
```

```bash
# All experiments, force recompute, 4 parallel workers
python analysis/metrics.py --all --no-skip --jobs 4
```

#### Reported indicators
---

The core metric is the travel time $t$, which is both the core term of the utility for human drivers (rational utility maximizers) and of the CAVs reward.
We report the average travel time for the system $\hat{t}$, human drivers $\hat{t}_{HDV}$, and autonomous vehicles $\hat{t}_{CAV}$. We record each during training, dynamic switching, and testing, plus a 50-day pre-mutation baseline ($\hat{t}^{train}, \hat{t}^{dyn}, \hat{t}^{test}, \hat{t}^{pre}$), with start/end windows for training and dynamic phases.

From these, we introduce:

- CAV advantage as $\hat{t}^{test}_{HDV} / \hat{t}^{test}_{CAV}$,
- Effect of changing to CAV as ${\hat{t}^{pre}_{HDV}}/{\hat{t}^{test}_{CAV}}$, and
- Effect of remaining HDV as ${\hat{t}^{pre}_{HDV}}/{\hat{t}^{test}_{HDV}}$.

To understand causes of travel time shifts, we track _Average speed_ and _Average mileage_ (from SUMO), plus dynamic-phase stability metrics:

- _Switch cost_ for all agents and per group: $\hat{t}^{dyn} - \hat{t}^{train}_{end}$, $\hat{t}^{dyn}_{HDV} - \hat{t}^{train}_{HDV,end}$, $\hat{t}^{dyn}_{CAV} - \hat{t}^{train}_{CAV,end}$.
- _Dynamic recovery_: $\hat{t}^{dyn}_{start} - \hat{t}^{dyn}_{end}$.
- _Dynamic volatility_: $\sigma(\hat{t}^{dyn})$ and group-specific versions.
- _Dynamic instability_: per-episode action-change rates for HDVs/CAVs.
- _Dynamic time excess_: average per-episode sum of time lost during switching.

We measure the _Cost of training_, expressed as the average of $\sum_{\tau \in train}(t^\tau_a - \hat{t}^{pre}_a)$ over all agents $a$, i.e. the cumulated disturbance CAVs cause during training. We define $c_{CAV}$ and $c_{HDV}$ accordingly. We call an experiment _won_ by CAVs if their policy was on average faster than human drivers' behaviour. A final _winrate_ is the share of training episodes where CAVs are faster.

Finally, we report switch statistics from `shifts.csv` (when available): total switches by direction, switches per event/agent, unique switch churn, and machine-ratio summary (start/end/avg/min/max/std).

#### Units of measurement
---

All the metrics are expressed in the following units:
- Time: minutes
- Distance: kilometers
- Speed: kilometers per hour
