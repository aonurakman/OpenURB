# IQL hyperparameters

All configs in this folder share the same keys and are consumed by `scripts/open_iql.py`
and `algorithms/iql.py`.

- `training_eps`: number of training episodes in the AV learning phase.
- `temp_init`: initial temperature for Boltzmann (softmax) action selection.
- `temp_decay`: multiplicative temperature decay applied after each learning step.
- `temp_min`: lower bound for temperature during decay (0 = greedy).
- `buffer_size`: replay buffer capacity.
- `batch_size`: number of transitions sampled per DQN update.
- `lr`: Adam learning rate for the Q-network.
- `num_epochs`: gradient steps per update (over resampled minibatches).
- `num_hidden`: number of hidden layers in the Q-network.
- `widths`: layer widths; length must be `num_hidden + 1`.
- `update_every`: run learning every N episodes.
- `rnn_hidden_dim`: GRU hidden size for the recurrent Q-network (set to `0` to use `widths[-1]`).
- `seq_len`: sequence length for recurrent replay updates.
- `gamma`: discount factor for TD targets.
- `target_update_every`: number of learner updates between target-network updates.
- `double_dqn`: whether to use Double DQN targets.
- `tau`: target update rate; `1.0` means hard updates every `target_update_every`.
- `max_grad_norm`: gradient clipping threshold.
