# PIMAC_v0 hyperparameters

All configs in this folder share the same keys and are consumed by
`scripts/open_pimac_v0.py` / `scripts/cond_open_pimac_v0.py` together with
`algorithms/pimac_v0.py`.

- `training_eps`: number of training episodes in the AV learning phase.
- `batch_size`: number of episodes sampled per PPO update.
- `update_every`: run learning every N episodes.
- `lr`: Adam learning rate for actor and critic.
- `num_epochs`: gradient steps per update (over resampled minibatches).
- `num_hidden`: number of hidden layers in the actor observation encoder.
- `widths`: actor layer widths; length must be `num_hidden + 1`.
- `rnn_hidden_dim`: GRU hidden size for the recurrent actor policy.
- `set_embed_dim`: output width of the per-agent set encoder `phi`.
- `set_encoder_hidden_sizes`: hidden-layer widths for the set encoder `phi`.
- `critic_hidden_sizes`: hidden-layer widths for the set critic head `rho`.
- `include_team_size_feature`: if `true`, concatenate active-agent ratio before `rho`.
- `clip_eps`: PPO ratio clipping range.
- `gamma`: discount factor.
- `gae_lambda`: GAE(lambda) parameter for advantage estimation.
- `normalize_advantage`: normalize team-level advantages within a batch.
- `entropy_coef`: entropy bonus weight to encourage exploration.
- `value_coef`: value loss weight in the combined actor-critic objective.
- `max_grad_norm`: gradient clipping threshold.
- `buffer_size`: maximum number of stored completed episodes between updates.
