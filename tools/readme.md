## Tools

Helpers for managing experiment runs under `results/`.

### `rename.py`
Rename an experiment directory and update text references to the old id inside it.

```bash
venv/bin/python tools/rename.py <old_id> [new_id]
```

### `reproduce.py`
Replay an experiment from `results/<exp_id>/exp_config.json`, optionally with new seeds.

```bash
venv/bin/python tools/reproduce.py --id <existing_exp_id> [--env-seed <seed>] [--torch-seed <seed>]
```

### `run_todo.py`
Run commands from a todo file in batches and capture stdout/stderr under the corresponding result directory.

```bash
venv/bin/python tools/run_todo.py todo.txt --jobs 3
```
