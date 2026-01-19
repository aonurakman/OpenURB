import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from algorithms.vdn import AgentRNN, VDN


def test_network_width_mismatch_raises():
    with pytest.raises(AssertionError):
        AgentRNN(obs_dim=3, action_dim=2, rnn_hidden_dim=8, num_hidden=1, widths=(8, 8, 4))


def test_act_respects_action_mask_when_exploiting():
    np.random.seed(0)
    torch.manual_seed(0)

    vdn = VDN(
        state_size=3,
        action_space_size=4,
        num_agents=1,
        temp_init=0.0,
        temp_decay=1.0,
        temp_min=0.0,
        buffer_size=8,
        batch_size=2,
        lr=0.01,
        num_epochs=1,
        num_hidden=1,
        widths=(8, 8),
        rnn_hidden_dim=8,
        share_parameters=True,
        normalize_by_active=True,
    )
    vdn.reset_episode()

    # Force deterministic Q-values: [0, 1, 2, 3].
    with torch.no_grad():
        vdn.agent_net.input_layer.weight.zero_()
        vdn.agent_net.input_layer.bias.zero_()
        for layer in vdn.agent_net.hidden_layers:
            layer.weight.zero_()
            layer.bias.zero_()
        for p in vdn.agent_net.rnn.parameters():
            p.zero_()
        vdn.agent_net.out.weight.zero_()
        vdn.agent_net.out.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))

    obs = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    action_mask = np.array([1, 1, 1, 0], dtype=np.int8)
    action = vdn.act(obs, action_mask=action_mask, agent_index=0)
    assert action == 2


def test_vdn_mixing_is_linear_and_gradients_match_mask():
    torch.manual_seed(0)

    vdn = VDN(state_size=2, action_space_size=2, num_agents=4, normalize_by_active=True)
    chosen_q = torch.randn(2, 3, 4, requires_grad=True)
    active_mask = torch.tensor(
        [
            [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]],
            [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]],
        ],
        dtype=torch.float32,
    )

    q_tot = vdn._mix_q_tot(chosen_q, active_mask)
    q_tot.sum().backward()

    expected = active_mask / active_mask.sum(dim=2, keepdim=True).clamp(min=1.0)
    assert chosen_q.grad is not None
    assert torch.allclose(chosen_q.grad, expected, atol=1e-6)


def test_learn_updates_parameters_and_decays_temperature():
    np.random.seed(1)
    torch.manual_seed(1)

    vdn = VDN(
        state_size=3,
        action_space_size=2,
        num_agents=2,
        temp_init=0.9,
        temp_decay=0.5,
        temp_min=0.0,
        buffer_size=16,
        batch_size=2,
        lr=0.01,
        num_epochs=1,
        num_hidden=1,
        widths=(8, 8),
        rnn_hidden_dim=8,
        target_update_every=10,
        double_q=True,
        tau=1.0,
        share_parameters=True,
        normalize_by_active=True,
    )

    # Start with Q-values near zero so there is a clear TD error signal.
    with torch.no_grad():
        vdn.agent_net.input_layer.weight.zero_()
        vdn.agent_net.input_layer.bias.zero_()
        for layer in vdn.agent_net.hidden_layers:
            layer.weight.zero_()
            layer.bias.zero_()
        for p in vdn.agent_net.rnn.parameters():
            p.zero_()
        vdn.agent_net.out.weight.zero_()
        vdn.agent_net.out.bias.zero_()

    obs_batch = np.zeros((2, 3), dtype=np.float32)
    actions_batch = np.array([0, 1], dtype=np.int64)
    rewards_batch = np.array([1.0, 1.0], dtype=np.float32)
    active_mask = np.array([1.0, 1.0], dtype=np.float32)

    # Fill replay with enough single-step episodes.
    for _ in range(2):
        vdn.store_episode(obs_batch, actions_batch, rewards_batch, active_mask)

    temp_before = vdn.temperature
    bias_before = vdn.agent_net.out.bias.detach().clone()
    vdn.learn()

    assert len(vdn.loss) == 1
    assert vdn.temperature == pytest.approx(temp_before * vdn.temp_decay)
    assert not torch.allclose(vdn.agent_net.out.bias, bias_before)

