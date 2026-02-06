## external_tasks (sanity checks)

These scripts are **not unit tests**. They run short MARL training loops to sanity-check that the
algorithms in `algorithms/` can learn on small **multi-step** environments.

Outputs are written under `external_tasks/runs/<env>/<algo>/<timestamp>/`:
- `learning_curves.png`
- `episode_rewards.npy`, `episode_losses.npy`, `eval_rewards.npy`
- `pimac_loss_history.json` (for PI-MAC runs; rich per-update diagnostics)
- `best_checkpoint.pt` (best eval reward)
- `policy_rollout.gif` (rollout of the best checkpoint; always headless)

PI-MAC sanity scripts use the v-next MAPPO-style actor plus token teacher-critic, heteroscedastic distillation,
and uncertainty-gated FiLM from `algorithms/pimac.py`.

### Environments

- `simple_spread_v3` (PettingZoo MPE): `external_tasks/simple_spread/`
- `coop_line_world` (tiny built-in toy env): `external_tasks/toy_env/`

### Dependencies (for PettingZoo)

```bash
pip install "pettingzoo[mpe]" pygame pillow
```

### Run

```bash
python external_tasks/simple_spread/random_policy.py
python external_tasks/simple_spread/iql.py
python external_tasks/simple_spread/ippo.py
python external_tasks/simple_spread/qmix.py
python external_tasks/simple_spread/vdn.py
python external_tasks/simple_spread/pimac.py

python external_tasks/toy_env/random_policy.py
python external_tasks/toy_env/iql.py
python external_tasks/toy_env/ippo.py
python external_tasks/toy_env/qmix.py
python external_tasks/toy_env/vdn.py
python external_tasks/toy_env/pimac.py
```
