## 🔁 Reproducing experiments

Every run saves its configuration under `results/<exp_id>/exp_config.json`. Use `reproduce/repeat_exp.py` to replay an experiment with the recorded parameters or with new seeds.

#### Usage

```bash
python reproduce/repeat_exp.py --id <existing_exp_id> [--env-seed <seed>] [--torch-seed <seed>]
```

The helper stores outputs alongside the original results: plain repeats become `<id>_repeated`, while seed overrides yield `<id>_v2`, `<id>_v3`, and so on.