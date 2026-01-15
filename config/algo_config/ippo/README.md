# IPPO hyperparameters

All configs in this folder share the same keys and are consumed by `scripts/open_ippo.py`
and `algorithms/ippo.py`.

- `training_eps`: number of training episodes in the AV learning phase.
- `batch_size`: number of episodes sampled per PPO update.
- `update_every`: run learning every N episodes.
- `lr`: Adam learning rate for the policy network.
- `num_epochs`: gradient steps per update (over resampled minibatches).
- `num_hidden`: number of hidden layers in the policy network.
- `widths`: layer widths; length must be `num_hidden + 1`.
- `rnn_hidden_dim`: GRU hidden size for the policy/value network.
- `clip_eps`: PPO ratio clipping range.
- `gamma`: discount factor.
- `gae_lambda`: GAE(λ) parameter for advantage estimation.
- `normalize_advantage`: normalize advantages within a batch.
- `entropy_coef`: entropy bonus weight to encourage exploration.
- `value_coef`: value loss weight in the combined actor-critic objective.
- `max_grad_norm`: gradient clipping threshold.
- `buffer_size`: maximum number of stored completed episodes between updates.
