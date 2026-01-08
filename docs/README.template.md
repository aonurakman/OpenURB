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

<!-- INCLUDE: ../networks/readme.md -->

## 🔬 Running experiments

#### Usage of **OpenURB** for Reinforcement Learning algorithms

To run the open/cond-open RL experiments, use:

```bash
python scripts/<script_name>.py --id <exp_id> --alg-conf <hyperparam_id> --env-conf <env_conf_id> --task-conf <task_id> --net <net_name> --env-seed <env_seed> --torch-seed <torch_seed>
```

where

- ```<script_name>``` is the script you wish to run, available scripts are ```open_iql```, ```cond_open_iql```, ```open_ippo```, ```cond_open_ippo```, and ```open_qmix```,
- ```<exp_id>``` is your own experiment identifier, for instance ```random_ing```,
- ```<hyperparam_id>``` is the hyperparameterization identifier, it must correspond to a `.json` filename (without extension) in [`config/algo_config`](config/algo_config/). Provided scripts automatically select the algorithm-specific subfolder in this directory.
- ```<env_conf_id>``` is the environment configuration identifier. It must correspond to a `.json` filename (without extension) in [`config/env_config`](config/env_config/). It is used to parameterize environment-specific processes, such as path generation, disk operations, etc. It is **optional** and by default is set to `config1`.
- ```<task_id>``` is the task configuration identifier. It must correspond to a `.json` filename (without extension) in [`config/task_config`](config/task_config/). For this repo, use configs with `dynamic` in the name.
- ```<net_name>``` is the name of the network you wish to use. Must be one of the folder names in ```networks/``` i.e. ```ing_small```, ```ingolstadt_custom```, ```nangis```, ```nemours```, ```provins``` or ```saint_arnoult```,
- ```<env_seed>``` is reproducibility random seed for the traffic environment, it is **optional** and by default is set to 42,
- ```<torch_seed>``` is reproducibility random seed for PyTorch, it is **optional** and by default is set to 42.

For example, the following command runs an experiment using:
- IQL algorithm, hyperparameterized by `config/algo_config/iql/config1.json`,
- The task specified in `config/task_config/dynamic1.json`,
- The environment parameterization specified in `config/env_config/config1.json` (by default),
- Experiment identifier `deneme`, which will be used as the folder name in `results/` to save the experiment data,
- Saint Arnoult network and demand, from `networks/saint_arnoult`,
- Environment (also used for `random` and `numpy`) and PyTorch seeds 42 and 0, respectively.

```bash
python scripts/open_iql.py --id deneme --alg-conf config1 --task-conf dynamic1 --net saint_arnoult --env-seed 42 --torch-seed 0
```

Example for QMIX:

```bash
python scripts/open_qmix.py --id deneme_qmix --alg-conf config1 --task-conf dynamic1 --net saint_arnoult --env-seed 42 --torch-seed 0
```

> All experiment scripts in this repo expect task configs with `dynamic` in the name.

#### Usage **URB** for baselines

Similarly as for RL algorithms, you have to provide command, but there is one additional flag ```model``` for ```scripts/open_baselines.py```, and ```scripts/cond_open_baselines.py```, instead of ```torch-seed```, then you have command of form:

```bash
python scripts/open_baselines.py --id <exp_id> --alg-conf <hyperparam_id> --env-conf <env_conf_id> --task-conf <task_id> --net <net_name> --env-seed <env_seed> --model <model_name>
```

For a list of available baseline models, see the **Baseline models** section below.
The open baseline scripts mirror the dynamic switching behavior: use task configs with `dynamic` in the name, and `cond_open_baselines.py` conditions switches on group travel times.

For example:

```bash
python scripts/open_baselines.py --id ing_aon --alg-conf config1 --task-conf dynamic2 --net ingolstadt_custom --model aon
```

<!-- INCLUDE: ../scripts/readme.md -->

<!-- INCLUDE: ../baseline_models/readme.md -->

<!-- INCLUDE: ../results/readme.md -->

<!-- INCLUDE: ../tools/readme.md -->

<!-- INCLUDE: ../analysis/readme.md -->
