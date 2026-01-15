import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import numpy as np
import pytest
import torch

from algorithms.ippo import ActorCriticRNN, PPO


def test_network_width_mismatch_raises():
    """Guard against width layouts that do not align with hidden layers."""
    with pytest.raises(AssertionError):
        ActorCriticRNN(obs_dim=2, action_dim=2, num_hidden=1, widths=[8, 4, 2], rnn_hidden_dim=8)


def test_act_switches_between_stochastic_and_deterministic_modes():
    """Confirm the agent can toggle between exploratory and greedy actions."""
    ppo = PPO(
        state_size=2,
        action_space_size=2,
        batch_size=2,
        lr=0.01,
        num_epochs=1,
        num_hidden=2,
        widths=[6, 6, 2],
        rnn_hidden_dim=8,
    )
    with torch.no_grad():
        layers = [ppo.policy_net.input_layer, *ppo.policy_net.hidden_layers]
        for layer in layers:
            layer.weight.zero_()
            layer.bias.zero_()
        for _, param in ppo.policy_net.rnn.named_parameters():
            param.zero_()
        ppo.policy_net.policy_head.weight.zero_()
        ppo.policy_net.policy_head.bias[:] = torch.tensor([0.3, -0.1])
        ppo.policy_net.value_head.weight.zero_()
        ppo.policy_net.value_head.bias.zero_()

    state = np.array([0.2, -0.1], dtype=np.float32)
    torch.manual_seed(0)
    ppo.deterministic = False
    stochastic_action = ppo.act(state)
    assert stochastic_action in {0, 1}
    ppo.push(0.5)

    ppo.deterministic = True
    deterministic_action = ppo.act(state)
    assert deterministic_action == 0


def test_learn_updates_policy_parameters_and_clears_memory():
    """Learning should adjust network weights and empty the on-policy buffer."""
    ppo = PPO(
        state_size=2,
        action_space_size=2,
        batch_size=2,
        lr=0.01,
        num_epochs=1,
        num_hidden=2,
        widths=[6, 6, 2],
        rnn_hidden_dim=8,
        normalize_advantage=False,
    )
    with torch.no_grad():
        layers = [ppo.policy_net.input_layer, *ppo.policy_net.hidden_layers]
        for layer in layers:
            layer.weight.zero_()
            layer.bias.zero_()
        for _, param in ppo.policy_net.rnn.named_parameters():
            param.zero_()
        ppo.policy_net.policy_head.weight.zero_()
        ppo.policy_net.policy_head.bias[:] = torch.tensor([0.2, -0.2])
        ppo.policy_net.value_head.weight.zero_()
        ppo.policy_net.value_head.bias.zero_()

    states = [
        np.array([0.1, 0.3], dtype=np.float32),
        np.array([-0.2, 0.4], dtype=np.float32),
    ]
    rewards = [1.0, 0.1]

    ppo.deterministic = True
    for state, reward in zip(states, rewards):
        ppo.act(state)
        # Convert to list so Torch avoids wrapping nested numpy arrays.
        ppo.last_state = ppo.last_state.tolist()
        ppo.push(reward)

    params_before = [param.detach().clone() for param in ppo.policy_net.parameters()]
    ppo.learn()

    assert len(ppo.loss) == 1
    assert len(ppo.memory) == 0
    assert any(
        not torch.allclose(param, before)
        for param, before in zip(ppo.policy_net.parameters(), params_before)
    )
