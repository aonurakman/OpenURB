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

### run_todo.py

Run the commands in `todo.txt` in parallel batches and save each command's stdout/stderr
to `results/<exp_id>/stdout.log` (an `--id` is auto-added when missing).

Usage:

```bash
python tools/run_todo.py todo.txt --jobs 3
```
