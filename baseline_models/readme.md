## Baseline models

Baseline policies can be run with `scripts/open_baselines.py` and `scripts/cond_open_baselines.py`.

Available models:
- `aon`: deterministically chooses the shortest free-flow route.
- `random`: samples routes uniformly at random.
- `gawron`: provided by RouteRL and used as the human-driver learning model.
