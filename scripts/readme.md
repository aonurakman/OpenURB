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
