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
