# IQL hyperparameters

All configs in this folder share the same keys and are consumed by `scripts/iql.py`
and `algorithms/simple_dqn.py`.

- `training_eps`: number of training episodes in the AV learning phase.
- `eps_init`: initial epsilon for epsilon-greedy action selection.
- `eps_decay`: multiplicative epsilon decay applied after each learning step.
- `buffer_size`: replay buffer capacity.
- `batch_size`: number of transitions sampled per DQN update.
- `lr`: Adam learning rate for the Q-network.
- `num_epochs`: gradient steps per update (over resampled minibatches).
- `num_hidden`: number of hidden layers in the Q-network.
- `widths`: layer widths; length must be `num_hidden + 1`.
- `update_every`: run learning every N episodes.
