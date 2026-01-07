# QMIX hyperparameters

All configs in this folder share the same keys and are consumed by `scripts/open_qmix.py`
and `algorithms/simple_qmix.py`.

- `training_eps`: number of training episodes in the AV learning phase.
- `eps_init`: initial epsilon for epsilon-greedy action selection.
- `eps_decay`: multiplicative epsilon decay applied after each learning step.
- `eps_min`: lower bound for epsilon during decay.
- `buffer_size`: replay buffer capacity (single-step joint transitions).
- `batch_size`: number of joint transitions sampled per update.
- `lr`: Adam learning rate for the agent and mixing networks.
- `num_epochs`: gradient steps per update (over resampled minibatches).
- `num_hidden`: number of hidden layers in the agent Q-network.
- `widths`: layer widths; length must be `num_hidden + 1`.
- `mixing_embed_dim`: hidden size for the QMIX mixing network.
- `hypernet_embed`: hidden size for the hypernetworks that generate mixing weights.
- `max_grad_norm`: gradient clipping threshold applied to both networks.
- `update_every`: run learning every N episodes.
