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
