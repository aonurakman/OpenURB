### (MA)RL algorithms and baselines.

We deliver here scripts for the experiment runs. The current training scripts are in-house and cover open vs. conditional switching variants:
* ```open_ippo.py``` runs a simplified IPPO/PPO setup with open (predefined) switching between human and AV agents,
* ```cond_open_ippo.py``` is the IPPO variant with switching conditioned on group travel times,
* ```open_iql.py``` runs an IQL setup with open switching,
* ```cond_open_iql.py``` is the conditional-switching version of the IQL setup.

We currently provide two learning families (IPPO and IQL). You can tune them via `config/algo_config/` and extend this folder with your own scripts.

Apart from RL algorithms, we provide baseline algorithms to compare with, can be used with ```open_baselines.py``` and ```cond_open_baselines.py```.
The open variants run dynamic switching (conditional in the `cond_` version) and require task configs with `dynamic` in the name.
Model options consist:
* **Baselines included in URB**
    * ```aon``` model which deterministically picks the shortest free-flow route regardless of the congestion,
    * ```random``` model which is fully undeterministic,
* **Additionally, available from `RouteRL`**
    * ```gawron``` model is based on: `Gawron (1998) <https://kups.ub.uni-koeln.de/9257/>`, the model iteratively shifts the cost expectations towards the received reward.

### Optional: Weights & Biases logging

All experiment scripts accept `--wandb-config` and `--no-wandb` flags. When enabled, they log per-episode mean rewards and travel times (overall + by agent kind) as episode CSVs are written to disk. Use `--no-wandb` to disable logging or if `wandb` is not installed. The default config file is `wandb_config.json` in the repo root (see the root README for details).
