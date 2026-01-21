import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from algorithms.pimac import PIMACAgentRNN, PIMAC, SetTokenTeacher


def test_network_width_mismatch_raises():
    with pytest.raises(AssertionError):
        PIMACAgentRNN(
            obs_dim=3,
            action_dim=2,
            rnn_hidden_dim=8,
            num_hidden=1,
            widths=(8, 8, 4),
            ctx_dim=4,
        )


def test_teacher_tokens_permutation_invariant():
    torch.manual_seed(0)

    teacher = SetTokenTeacher(in_dim=4, num_tokens=4, tok_dim=8, ctx_dim=6, action_dim=5)
    teacher.eval()

    obs = torch.randn(1, 2, 5, 4)
    active_mask = torch.tensor([[[1, 1, 0, 1, 1], [1, 0, 1, 1, 1]]], dtype=torch.float32)

    tokens_a, _ = teacher(obs, active_mask=active_mask, pool_mask=active_mask)

    perm = torch.tensor([2, 0, 4, 1, 3])
    obs_perm = obs[:, :, perm, :]
    mask_perm = active_mask[:, :, perm]

    tokens_b, _ = teacher(obs_perm, active_mask=mask_perm, pool_mask=mask_perm)
    assert torch.allclose(tokens_a, tokens_b, atol=1e-6)


def test_act_respects_action_mask_when_exploiting():
    np.random.seed(0)
    torch.manual_seed(0)

    pimac = PIMAC(
        state_size=3,
        action_space_size=4,
        num_agents=1,
        temp_init=0.0,
        temp_decay=1.0,
        temp_min=0.0,
        buffer_size=8,
        batch_size=2,
        lr=0.01,
        teacher_lr=0.01,
        num_epochs=1,
        num_hidden=1,
        widths=(8, 8),
        rnn_hidden_dim=8,
        ctx_dim=4,
        num_tokens=4,
        tok_dim=4,
        share_parameters=True,
        normalize_by_active=True,
        distill_weight=0.0,
        teacher_aux_weight=0.0,
        token_smooth_weight=0.0,
    )
    pimac.reset_episode()

    with torch.no_grad():
        pimac.agent_net.input_layer.weight.zero_()
        pimac.agent_net.input_layer.bias.zero_()
        for layer in pimac.agent_net.hidden_layers:
            layer.weight.zero_()
            layer.bias.zero_()
        for p in pimac.agent_net.rnn.parameters():
            p.zero_()
        pimac.agent_net.ctx_head.weight.zero_()
        pimac.agent_net.ctx_head.bias.zero_()
        pimac.agent_net.logvar_head.weight.zero_()
        pimac.agent_net.logvar_head.bias.zero_()
        pimac.agent_net.film.weight.zero_()
        pimac.agent_net.film.bias.zero_()
        if pimac.agent_net.index_proj is not None:
            pimac.agent_net.index_proj.weight.zero_()
            pimac.agent_net.index_proj.bias.zero_()
        pimac.agent_net.out.weight.zero_()
        pimac.agent_net.out.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))

    obs = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    action_mask = np.array([1, 1, 1, 0], dtype=np.int8)
    action = pimac.act(obs, action_mask=action_mask, agent_index=0)
    assert action == 2


def test_vdn_mixing_is_linear_and_gradients_match_mask():
    torch.manual_seed(0)

    pimac = PIMAC(state_size=2, action_space_size=2, num_agents=4, normalize_by_active=True)
    chosen_q = torch.randn(2, 3, 4, requires_grad=True)
    active_mask = torch.tensor(
        [
            [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]],
            [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]],
        ],
        dtype=torch.float32,
    )

    q_tot = pimac._mix_q_tot(chosen_q, active_mask)
    q_tot.sum().backward()

    expected = active_mask / active_mask.sum(dim=2, keepdim=True).clamp(min=1.0)
    assert chosen_q.grad is not None
    assert torch.allclose(chosen_q.grad, expected, atol=1e-6)


def test_learn_updates_parameters_and_decays_temperature():
    np.random.seed(1)
    torch.manual_seed(1)

    pimac = PIMAC(
        state_size=3,
        action_space_size=2,
        num_agents=2,
        temp_init=0.9,
        temp_decay=0.5,
        temp_min=0.0,
        buffer_size=16,
        batch_size=2,
        lr=0.01,
        teacher_lr=0.01,
        num_epochs=1,
        num_hidden=1,
        widths=(8, 8),
        rnn_hidden_dim=8,
        ctx_dim=4,
        num_tokens=4,
        tok_dim=4,
        target_update_every=10,
        double_q=True,
        tau=1.0,
        share_parameters=True,
        normalize_by_active=True,
        distill_weight=0.0,
        teacher_aux_weight=0.0,
        token_smooth_weight=0.0,
    )

    with torch.no_grad():
        pimac.agent_net.input_layer.weight.zero_()
        pimac.agent_net.input_layer.bias.zero_()
        for layer in pimac.agent_net.hidden_layers:
            layer.weight.zero_()
            layer.bias.zero_()
        for p in pimac.agent_net.rnn.parameters():
            p.zero_()
        pimac.agent_net.ctx_head.weight.zero_()
        pimac.agent_net.ctx_head.bias.zero_()
        pimac.agent_net.logvar_head.weight.zero_()
        pimac.agent_net.logvar_head.bias.zero_()
        pimac.agent_net.film.weight.zero_()
        pimac.agent_net.film.bias.zero_()
        if pimac.agent_net.index_proj is not None:
            pimac.agent_net.index_proj.weight.zero_()
            pimac.agent_net.index_proj.bias.zero_()
        pimac.agent_net.out.weight.zero_()
        pimac.agent_net.out.bias.zero_()

    obs_batch = np.zeros((2, 3), dtype=np.float32)
    actions_batch = np.array([0, 1], dtype=np.int64)
    rewards_batch = np.array([1.0, 1.0], dtype=np.float32)
    active_mask = np.array([1.0, 1.0], dtype=np.float32)

    for _ in range(2):
        pimac.store_episode(obs_batch, actions_batch, rewards_batch, active_mask)

    temp_before = pimac.temperature
    bias_before = pimac.agent_net.out.bias.detach().clone()
    pimac.learn()

    assert len(pimac.loss) == 1
    assert pimac.temperature == pytest.approx(temp_before * pimac.temp_decay)
    assert not torch.allclose(pimac.agent_net.out.bias, bias_before)


def test_variable_n_batching_smoke():
    np.random.seed(2)
    torch.manual_seed(2)

    pimac = PIMAC(
        state_size=4,
        action_space_size=3,
        num_agents=7,
        buffer_size=4,
        batch_size=2,
        num_epochs=1,
        lr=1e-3,
        teacher_lr=1e-3,
        num_hidden=1,
        widths=(16, 16),
        rnn_hidden_dim=16,
        ctx_dim=8,
        num_tokens=4,
        tok_dim=8,
        share_parameters=True,
    )

    obs_a = np.random.randn(3, 4).astype(np.float32)
    actions_a = np.array([0, 1, 2], dtype=np.int64)
    rewards_a = np.ones(3, dtype=np.float32)
    active_a = np.ones(3, dtype=np.float32)

    obs_b = np.random.randn(7, 4).astype(np.float32)
    actions_b = np.array([2, 1, 0, 1, 2, 0, 1], dtype=np.int64)
    rewards_b = np.ones(7, dtype=np.float32)
    active_b = np.ones(7, dtype=np.float32)

    pimac.store_episode(obs_a, actions_a, rewards_a, active_a)
    pimac.store_episode(obs_b, actions_b, rewards_b, active_b)

    pimac.learn()
    assert len(pimac.loss) == 1


def test_parallel_api_smoke():
    np.random.seed(3)
    torch.manual_seed(3)

    pimac = PIMAC(state_size=3, action_space_size=4, num_agents=5, share_parameters=True)
    obs_dict = {f"agent_{i}": np.random.randn(3).astype(np.float32) for i in range(5)}
    action_dict = pimac.act_parallel(obs_dict)

    assert set(action_dict.keys()) == set(obs_dict.keys())


def test_aec_cycle_storage_smoke():
    np.random.seed(4)
    torch.manual_seed(4)

    pimac = PIMAC(state_size=3, action_space_size=4, num_agents=3, buffer_size=8, batch_size=1)
    agent_ids = ["a", "b", "c"]
    pimac.aec_begin_cycle(agent_ids)

    for aid in agent_ids:
        obs = np.random.randn(3).astype(np.float32)
        pimac.aec_record(aid, obs=obs, action=1, reward=1.0, next_obs=obs, done=False)

    pimac.aec_end_cycle(done_all=True)
    assert len(pimac.memory) == 1
