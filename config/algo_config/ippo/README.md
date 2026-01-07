# IPPO hyperparameters

All configs in this folder share the same keys and are consumed by `scripts/ippo.py`
and `algorithms/simple_ppo.py`.

- `training_eps`: number of training episodes in the AV learning phase.
- `batch_size`: number of transitions sampled per PPO update.
- `update_every`: run learning every N episodes.
- `lr`: Adam learning rate for the policy network.
- `num_epochs`: gradient steps per update (over resampled minibatches).
- `num_hidden`: number of hidden layers in the policy network.
- `widths`: layer widths; length must be `num_hidden + 1`.
- `clip_eps`: PPO ratio clipping range.
- `normalize_advantage`: normalize rewards/advantages within a batch.
- `entropy_coef`: entropy bonus weight to encourage exploration.
