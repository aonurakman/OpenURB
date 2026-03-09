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
