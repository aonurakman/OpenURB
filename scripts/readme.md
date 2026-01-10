## Scripts

We provide training scripts for open vs. conditional switching variants:
- `open_ippo.py` runs a simplified IPPO/PPO setup with open (predefined) switching.
- `cond_open_ippo.py` is the IPPO variant with switching conditioned on group travel times.
- `open_iql.py` runs an IQL setup with open switching.
- `cond_open_iql.py` is the conditional-switching version of the IQL setup.
- `open_qmix.py` runs a QMIX setup with open switching.

Baseline scripts are `open_baselines.py` and `cond_open_baselines.py` (see `baseline_models/readme.md`
for available models). The open variants run dynamic switching (conditional in the `cond_` version)
and require task configs with `dynamic` in the name.

All scripts automatically run `analysis/metrics.py` at the end of an experiment to generate KPI outputs
in the experiment's `results/<exp_id>/metrics/` folder.

### Optional: Weights & Biases logging

All experiment scripts support optional Weights & Biases logging and will stream per-episode mean
rewards and travel times (overall + by agent kind) as episode CSVs are written to disk. Use
`--no-wandb` to disable logging if you do not want W&B integration or if `wandb` is not installed.

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
python scripts/open_iql.py --id <exp_id> --alg-conf <hyperparam_id> --task-conf <task_id> --net <net_name> [--env-conf <env_conf_id>] [--env-seed <env_seed>] [--torch-seed <torch_seed>] [--wandb-config <path>] [--no-wandb]
```

Use `--no-wandb` to disable logging while keeping the original disk outputs.
