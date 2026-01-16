import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from algorithms.iql import DQN, RecurrentNetwork


def test_network_width_mismatch_raises():
    """Ensure misconfigured width sequences are rejected."""
    with pytest.raises(AssertionError):
        RecurrentNetwork(in_size=3, out_size=2, num_hidden=1, widths=(8, 8, 4), rnn_hidden_dim=8)


def test_act_returns_greedy_action_when_temperature_zero():
    """When temperature is 0 the policy should act greedily."""
    np.random.seed(0)
    torch.manual_seed(0)
    dqn = DQN(
        state_size=2,
        action_space_size=3,
        temp_init=0.0,
        temp_decay=1.0,
        temp_min=0.0,
        batch_size=1,
        buffer_size=8,
        num_hidden=1,
        widths=(4, 4),
    )
    with torch.no_grad():
        layers = [dqn.q_network.input_layer, *dqn.q_network.hidden_layers]
        for layer in layers:
            layer.weight.zero_()
            layer.bias.zero_()
        for param in dqn.q_network.rnn.parameters():
            param.zero_()
        dqn.q_network.out_layer.weight.zero_()
        dqn.q_network.out_layer.bias[:] = torch.tensor([0.0, 1.0, 2.0])
    state = np.array([0.1, -0.2], dtype=np.float32)
    action = dqn.act(state)
    assert action == 2
    assert np.array_equal(dqn.last_state, state.astype(np.float32, copy=False))
    assert dqn.last_action == action


def test_learn_updates_temperature_and_records_loss():
    """Learning with enough samples should update weights and decay temperature."""
    np.random.seed(1)
    torch.manual_seed(1)
    dqn = DQN(
        state_size=3,
        action_space_size=2,
        temp_init=0.9,
        temp_decay=0.8,
        temp_min=0.0,
        batch_size=2,
        buffer_size=16,
        num_epochs=1,
        num_hidden=2,
        widths=(6, 6, 2),
    )
    with torch.no_grad():
        layers = [dqn.q_network.input_layer, *dqn.q_network.hidden_layers]
        for layer in layers:
            layer.weight.zero_()
            layer.bias.zero_()
        for param in dqn.q_network.rnn.parameters():
            param.zero_()
        dqn.q_network.out_layer.weight.zero_()
        dqn.q_network.out_layer.bias[:] = torch.tensor([0.0, 0.5])

    states = [
        np.array([0.5, -0.1, 0.2], dtype=np.float32),
        np.array([-0.3, 0.4, 0.1], dtype=np.float32),
    ]
    rewards = [1.0, 0.4]
    for state, reward in zip(states, rewards):
        dqn.act(state)
        # Store plain Python data to avoid torch tensor creation warnings.
        dqn.last_state = dqn.last_state.tolist()
        dqn.push(reward)

    temp_before = dqn.temperature
    bias_before = dqn.q_network.out_layer.bias.detach().clone()
    dqn.learn()

    assert len(dqn.loss) == 1
    assert dqn.temperature == pytest.approx(temp_before * dqn.temp_decay)
    assert not torch.allclose(dqn.q_network.out_layer.bias, bias_before)


def test_learn_returns_early_when_memory_small():
    """Learning should short-circuit when the replay buffer lacks samples."""
    dqn = DQN(
        state_size=2,
        action_space_size=2,
        temp_init=0.5,
        temp_decay=0.9,
        batch_size=3,
        buffer_size=4,
        num_hidden=2,
        widths=(5, 5, 2),
    )
    temp_before = dqn.temperature
    dqn.learn()
    assert dqn.loss == []
    assert dqn.temperature == temp_before


def test_loss_fn_is_unreduced_for_padding_mask():
    """The TD loss must be unreduced so padded timesteps can be masked out."""
    dqn = DQN(
        state_size=2,
        action_space_size=2,
        batch_size=2,
        buffer_size=8,
        num_hidden=1,
        widths=(8, 8),
    )
    assert getattr(dqn.loss_fn, "reduction", None) == "none"
