## Baseline models

Baseline algorithms can be run with `scripts/open_baselines.py` and `scripts/cond_open_baselines.py`.
Available model options:

- **Baselines included in OpenURB**
  - `aon` deterministically picks the shortest free-flow route regardless of congestion.
  - `random` selects routes uniformly at random.
- **Additionally available from RouteRL**
  - `gawron` (base human learning model) follows Gawron (1998) and iteratively shifts cost
    expectations toward received rewards.
