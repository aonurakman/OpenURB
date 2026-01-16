import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from algorithms.qmix import AgentRNN, QMIX


def test_network_width_mismatch_raises():
    with pytest.raises(AssertionError):
        AgentRNN(obs_dim=3, action_dim=2, rnn_hidden_dim=8, num_hidden=1, widths=(8, 8, 4))


def test_act_respects_action_mask_when_exploiting():
    np.random.seed(0)
    torch.manual_seed(0)

    qmix = QMIX(
        state_size=3,
        action_space_size=4,
        num_agents=1,
        global_state_size=5,
        eps_init=0.0,
        eps_decay=1.0,
        eps_min=0.0,
        buffer_size=8,
        batch_size=2,
        lr=0.01,
        num_epochs=1,
        num_hidden=1,
        widths=(8, 8),
        rnn_hidden_dim=8,
    )
    qmix.reset_episode()

    # Force the agent network to output deterministic Q-values: [0, 1, 2, 3].
    with torch.no_grad():
        qmix.agent_net.input_layer.weight.zero_()
        qmix.agent_net.input_layer.bias.zero_()
        for layer in qmix.agent_net.hidden_layers:
            layer.weight.zero_()
            layer.bias.zero_()
        for p in qmix.agent_net.rnn.parameters():
            p.zero_()
        qmix.agent_net.out.weight.zero_()
        qmix.agent_net.out.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))

    obs = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    # Mask out the best action (3) so the greedy choice is 2.
    action_mask = np.array([1, 1, 1, 0], dtype=np.int8)
    action = qmix.act(obs, action_mask=action_mask, agent_index=0)
    assert action == 2


def test_learn_updates_parameters_and_decays_epsilon():
    np.random.seed(1)
    torch.manual_seed(1)

    qmix = QMIX(
        state_size=3,
        action_space_size=2,
        num_agents=2,
        global_state_size=6,
        eps_init=0.9,
        eps_decay=0.5,
        eps_min=0.0,
        buffer_size=16,
        batch_size=2,
        lr=0.01,
        num_epochs=1,
        num_hidden=1,
        widths=(8, 8),
        rnn_hidden_dim=8,
        mixing_embed_dim=4,
        hypernet_embed=8,
        target_update_every=10,
        double_q=True,
        tau=1.0,
        share_parameters=True,
    )

    # Make Q-values start near zero so there is a clear TD error signal.
    with torch.no_grad():
        qmix.agent_net.input_layer.weight.zero_()
        qmix.agent_net.input_layer.bias.zero_()
        for layer in qmix.agent_net.hidden_layers:
            layer.weight.zero_()
            layer.bias.zero_()
        for p in qmix.agent_net.rnn.parameters():
            p.zero_()
        qmix.agent_net.out.weight.zero_()
        qmix.agent_net.out.bias.zero_()

    obs_batch = np.zeros((2, 3), dtype=np.float32)
    actions_batch = np.array([0, 1], dtype=np.int64)
    rewards_batch = np.array([1.0, 1.0], dtype=np.float32)
    active_mask = np.array([1.0, 1.0], dtype=np.float32)
    # Use a non-zero state so hypernet weights receive gradients (zero inputs -> zero weight grads).
    global_state = np.ones(6, dtype=np.float32)

    # Fill replay with enough single-step episodes.
    for _ in range(2):
        qmix.store_episode(obs_batch, actions_batch, rewards_batch, active_mask, global_state)

    epsilon_before = qmix.epsilon
    mix_weight_before = qmix.mixing_net.hyper_b2[0].weight.detach().clone()
    qmix.learn()

    assert len(qmix.loss) == 1
    assert qmix.epsilon == pytest.approx(epsilon_before * qmix.eps_decay)
    assert not torch.allclose(qmix.mixing_net.hyper_b2[0].weight, mix_weight_before)
