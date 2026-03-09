## external_tasks (sanity checks)

These scripts are **not unit tests**. They run short MARL training loops to sanity-check that the
algorithms in `algorithms/` can learn on small **multi-step** environments.

Outputs are written under `external_tasks/runs/<env>/<algo>/<timestamp>/`:
- `learning_curves.png`
- `episode_rewards.npy`, `episode_losses.npy`, `eval_rewards.npy`
- `mappo_loss_history.json` (for MAPPO runs; per-update diagnostics)
- `pimac_v0_loss_history.json` (for PIMAC_v0 runs; set-critic MAPPO diagnostics)
- `pimac_v1_loss_history.json` (for PIMAC_v1 runs; token teacher-student diagnostics)
- `pimac_v2_loss_history.json` (for PIMAC_v2 runs; FiLM-gated token teacher-student diagnostics)
- `pimac_v3_loss_history.json` (for PIMAC_v3 runs; FiLM + hypernetwork token teacher-student diagnostics)
- `best_checkpoint.pt` (best eval reward)
- `policy_rollout.gif` (rollout of the best checkpoint; always headless)

PIMAC_v* sanity scripts use MAPPO-style on-policy optimization with a token/set-based centralized critic
and progressively stronger decentralized policy conditioning.

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
python external_tasks/simple_spread/mappo.py
python external_tasks/simple_spread/pimac_v0.py
python external_tasks/simple_spread/pimac_v1.py
python external_tasks/simple_spread/pimac_v2.py
python external_tasks/simple_spread/pimac_v3.py

python external_tasks/toy_env/random_policy.py
python external_tasks/toy_env/iql.py
python external_tasks/toy_env/ippo.py
python external_tasks/toy_env/qmix.py
python external_tasks/toy_env/vdn.py
python external_tasks/toy_env/mappo.py
python external_tasks/toy_env/pimac_v0.py
python external_tasks/toy_env/pimac_v1.py
python external_tasks/toy_env/pimac_v2.py
python external_tasks/toy_env/pimac_v3.py

python external_tasks/simple_spread_dynamic/random_policy.py
python external_tasks/simple_spread_dynamic/iql.py
python external_tasks/simple_spread_dynamic/ippo.py
python external_tasks/simple_spread_dynamic/qmix.py
python external_tasks/simple_spread_dynamic/vdn.py
python external_tasks/simple_spread_dynamic/mappo.py
python external_tasks/simple_spread_dynamic/pimac_v0.py
python external_tasks/simple_spread_dynamic/pimac_v1.py
python external_tasks/simple_spread_dynamic/pimac_v2.py
python external_tasks/simple_spread_dynamic/pimac_v3.py
```
