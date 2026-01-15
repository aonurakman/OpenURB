# QMIX hyperparameters

All configs in this folder share the same keys and are consumed by `scripts/open_qmix.py`
and `algorithms/qmix.py`.

- `training_eps`: number of training episodes in the AV learning phase.
- `share_parameters`: whether to share a single agent Q-network across all agents.
- `eps_init`: initial epsilon for epsilon-greedy action selection.
- `eps_decay`: multiplicative epsilon decay applied after each learning step.
- `eps_min`: lower bound for epsilon during decay.
- `buffer_size`: replay buffer capacity (episodes; single-step is stored as length-1).
- `batch_size`: number of joint transitions sampled per update.
- `lr`: Adam learning rate for the agent and mixing networks.
- `num_epochs`: gradient steps per update (over resampled minibatches).
- `num_hidden`: number of hidden layers in the agent Q-network.
- `widths`: layer widths; length must be `num_hidden + 1`.
- `rnn_hidden_dim`: GRU hidden size for the agent networks.
- `mixing_embed_dim`: hidden size for the QMIX mixing network.
- `hypernet_embed`: hidden size for the hypernetworks that generate mixing weights.
- `max_grad_norm`: gradient clipping threshold applied to both networks.
- `update_every`: run learning every N episodes.
- `gamma`: discount factor for TD targets (single-step episodes effectively use `done=True`).
- `target_update_every`: number of learner updates between target-network updates.
- `double_q`: whether to use Double Q-learning for action selection in targets.
- `tau`: target update rate; `1.0` means hard updates every `target_update_every`.
- `mixing_weight_clip`: optional cap for mixing weights (set `null` to disable).
- `q_tot_clip`: optional clip for `Q_tot` and targets (set `null` to disable).
- `use_huber_loss`: use SmoothL1 (Huber) loss instead of MSE.
