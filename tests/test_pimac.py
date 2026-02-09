import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from algorithms.pimac import PIMAC, PIMACAEC, PIMACParallel, PIMACActorRNN, SetTokenTeacher


def test_network_width_mismatch_raises():
    with pytest.raises(AssertionError):
        PIMACActorRNN(
            obs_dim=3,
            action_dim=2,
            rnn_hidden_dim=8,
            num_hidden=1,
            widths=(8, 8, 4),
            ctx_dim=8,
        )


def test_set_token_teacher_permutation_padding_and_context_equivariance():
    torch.manual_seed(0)

    teacher = SetTokenTeacher(obs_dim=4, tok_dim=8, num_tokens=3, teacher_hidden_sizes=(16, 16))
    teacher.eval()

    obs = torch.randn(1, 2, 4, 4)
    active_mask = torch.tensor([[[1, 1, 0, 1], [1, 0, 1, 1]]], dtype=torch.float32)

    tokens_a, ctx_a = teacher(obs, active_mask)

    perm = torch.tensor([2, 0, 3, 1])
    inv_perm = torch.argsort(perm)
    obs_perm = obs[:, :, perm, :]
    mask_perm = active_mask[:, :, perm]
    tokens_b, ctx_b = teacher(obs_perm, mask_perm)

    padded_obs = torch.cat([obs, torch.randn(1, 2, 2, 4)], dim=2)
    padded_mask = torch.cat([active_mask, torch.zeros(1, 2, 2)], dim=2)
    tokens_c, ctx_c = teacher(padded_obs, padded_mask)

    # Token extraction should be invariant to agent reordering and inactive padding.
    assert torch.allclose(tokens_a, tokens_b, atol=1e-6)
    assert torch.allclose(tokens_a, tokens_c, atol=1e-6)

    # Per-agent teacher context should be permutation equivariant.
    assert torch.allclose(ctx_a, ctx_b[:, :, inv_perm, :], atol=1e-6)
    assert torch.allclose(ctx_a, ctx_c[:, :, : obs.shape[2], :], atol=1e-6)

    # Added inactive padding slots should have zero context by construction.
    assert torch.allclose(ctx_c[:, :, obs.shape[2] :, :], torch.zeros_like(ctx_c[:, :, obs.shape[2] :, :]), atol=1e-7)


def test_set_token_teacher_all_inactive_is_finite():
    torch.manual_seed(1)
    teacher = SetTokenTeacher(obs_dim=3, tok_dim=6, num_tokens=2, teacher_hidden_sizes=(8, 8))
    obs = torch.randn(2, 3, 5, 3)
    mask = torch.zeros(2, 3, 5)

    tokens, ctx = teacher(obs, mask)

    assert torch.isfinite(tokens).all()
    assert torch.isfinite(ctx).all()
    assert torch.allclose(ctx, torch.zeros_like(ctx), atol=1e-7)


def test_act_uses_argmax_when_deterministic():
    np.random.seed(0)
    torch.manual_seed(0)

    pimac = PIMACAEC(
        state_size=3,
        action_space_size=4,
        num_agents=1,
        buffer_size=8,
        batch_size=2,
        lr=0.01,
        num_epochs=1,
        num_hidden=1,
        widths=(8, 8),
        rnn_hidden_dim=8,
        tok_dim=8,
    )
    pimac.reset_episode()
    pimac.deterministic = True

    with torch.no_grad():
        pimac.actor_net.input_layer.weight.zero_()
        pimac.actor_net.input_layer.bias.zero_()
        for layer in pimac.actor_net.hidden_layers:
            layer.weight.zero_()
            layer.bias.zero_()
        for p in pimac.actor_net.rnn.parameters():
            p.zero_()
        pimac.actor_net.policy_head.weight.zero_()
        pimac.actor_net.policy_head.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))

    obs = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    action = pimac.act(obs, agent_index=0)
    assert action == 3


def test_gae_matches_manual_recursion():
    pimac = PIMACAEC(state_size=2, action_space_size=2, num_agents=2, gamma=0.9, gae_lambda=0.8)

    rewards = np.array([1.0, 2.0], dtype=np.float32)
    values = np.array([0.5, 0.2], dtype=np.float32)
    next_values = np.array([0.2, 0.0], dtype=np.float32)
    dones = np.array([0.0, 1.0], dtype=np.float32)

    adv, ret = pimac._compute_gae(rewards, values, next_values, dones)

    d1 = rewards[1] + 0.9 * (1.0 - dones[1]) * next_values[1] - values[1]
    a1 = d1
    d0 = rewards[0] + 0.9 * (1.0 - dones[0]) * next_values[0] - values[0]
    a0 = d0 + 0.9 * 0.8 * (1.0 - dones[0]) * a1

    expected_adv = np.array([a0, a1], dtype=np.float32)
    expected_ret = expected_adv + values

    np.testing.assert_allclose(adv, expected_adv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ret, expected_ret, rtol=1e-6, atol=1e-6)


def test_store_episode_finalization_contains_ppo_fields():
    np.random.seed(1)
    torch.manual_seed(1)

    pimac = PIMACAEC(state_size=4, action_space_size=3, num_agents=3, buffer_size=8, batch_size=1)

    obs = np.random.randn(3, 4).astype(np.float32)
    actions = np.array([0, 1, 2], dtype=np.int64)
    rewards = np.array([1.0, 0.5, -0.25], dtype=np.float32)
    active = np.ones(3, dtype=np.float32)

    pimac.store_episode(obs, actions, rewards, active)

    assert len(pimac.memory) == 1
    ep = pimac.memory[0]
    assert ep["old_log_probs"].shape == (1, 3)
    assert ep["advantages"].shape == (1,)
    assert ep["returns"].shape == (1,)


def test_learn_updates_parameters_and_logs_vnext_metrics():
    np.random.seed(2)
    torch.manual_seed(2)

    pimac = PIMACAEC(
        state_size=3,
        action_space_size=2,
        num_agents=2,
        buffer_size=16,
        batch_size=2,
        lr=0.01,
        num_epochs=2,
        num_hidden=1,
        widths=(16, 16),
        rnn_hidden_dim=16,
        num_tokens=3,
        tok_dim=16,
        teacher_hidden_sizes=(16, 16),
        teacher_drop_prob=0.2,
        teacher_ema_tau=0.05,
        distill_weight=0.1,
        critic_hidden_sizes=(16, 16),
    )

    for _ in range(2):
        obs_batch = np.random.randn(2, 3).astype(np.float32)
        actions_batch = np.array([0, 1], dtype=np.int64)
        rewards_batch = np.array([1.0, -0.5], dtype=np.float32)
        active_mask = np.array([1.0, 1.0], dtype=np.float32)
        pimac.store_episode(obs_batch, actions_batch, rewards_batch, active_mask)

    before_params = [p.detach().clone() for p in pimac.actor_net.parameters()]
    assert len(pimac.memory) == 2

    pimac.learn()

    after_params = [p.detach() for p in pimac.actor_net.parameters()]
    changed = any(not torch.allclose(p0, p1) for p0, p1 in zip(before_params, after_params))

    assert len(pimac.loss) == 1
    assert changed
    assert len(pimac.memory) == 0
    assert len(pimac.loss_history) >= 1

    last = pimac.loss_history[-1]
    expected_keys = {
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
        "approx_kl",
        "clip_frac",
        "explained_variance",
        "distill_loss",
        "distill_mse",
        "ctx_logvar_mean",
        "ctx_logvar_std",
        "gate_mean",
        "gate_std",
        "token_norm_mean",
        "teacher_ctx_norm_mean",
        "ctx_consistency_l2",
        "active_decisions",
        "active_agents_mean",
    }
    assert expected_keys.issubset(last.keys())
    assert last["active_decisions"] > 0
    for key in expected_keys:
        assert np.isfinite(float(last[key]))


def test_variable_n_batching_smoke():
    np.random.seed(3)
    torch.manual_seed(3)

    pimac = PIMACAEC(
        state_size=4,
        action_space_size=3,
        num_agents=7,
        buffer_size=4,
        batch_size=2,
        num_epochs=1,
        lr=1e-3,
        num_hidden=1,
        widths=(16, 16),
        rnn_hidden_dim=16,
        num_tokens=3,
        tok_dim=16,
        teacher_hidden_sizes=(16, 16),
        critic_hidden_sizes=(16, 16),
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
    assert len(pimac.memory) == 0


def test_parallel_api_smoke():
    np.random.seed(4)
    torch.manual_seed(4)

    pimac = PIMACParallel(state_size=3, action_space_size=4, num_agents=5)
    obs_dict = {f"agent_{i}": np.random.randn(3).astype(np.float32) for i in range(5)}
    action_dict = pimac.act_parallel(obs_dict)

    assert set(action_dict.keys()) == set(obs_dict.keys())


def test_aec_cycle_storage_smoke():
    np.random.seed(5)
    torch.manual_seed(5)

    pimac = PIMACAEC(state_size=3, action_space_size=4, num_agents=3, buffer_size=8, batch_size=1)
    agent_ids = ["a", "b", "c"]
    pimac.aec_begin_cycle(agent_ids)

    for aid in agent_ids:
        obs = np.random.randn(3).astype(np.float32)
        pimac.aec_record(aid, obs=obs, action=1, reward=1.0, next_obs=obs, done=False)

    pimac.aec_end_cycle(done_all=True)
    assert len(pimac.memory) == 1


def test_pimac_alias_points_to_aec():
    pimac = PIMAC(state_size=2, action_space_size=2, num_agents=2)
    assert isinstance(pimac, PIMACAEC)


def test_parallel_done_dict_without_all_key_finalizes_only_when_all_true():
    np.random.seed(6)
    torch.manual_seed(6)

    pimac = PIMACParallel(state_size=3, action_space_size=2, num_agents=2, buffer_size=8, batch_size=1)
    obs = {
        "a": np.random.randn(3).astype(np.float32),
        "b": np.random.randn(3).astype(np.float32),
    }
    actions = {"a": 0, "b": 1}
    rewards = {"a": 0.5, "b": -0.25}
    next_obs = {
        "a": np.random.randn(3).astype(np.float32),
        "b": np.random.randn(3).astype(np.float32),
    }

    pimac.store_parallel_step(obs, actions, rewards, next_obs, done_dict={"a": False, "b": True})
    assert len(pimac.memory) == 0

    pimac.store_parallel_step(obs, actions, rewards, next_obs, done_dict={"a": True, "b": True})
    assert len(pimac.memory) == 1

    pimac_2 = PIMACParallel(state_size=3, action_space_size=2, num_agents=2, buffer_size=8, batch_size=1)
    pimac_2.store_parallel_step(obs, actions, rewards, next_obs, done_dict={"__all__": True, "a": False, "b": False})
    assert len(pimac_2.memory) == 1


def test_parallel_terminal_empty_next_obs_dict_finalizes_without_crash():
    np.random.seed(8)
    torch.manual_seed(8)

    pimac = PIMACParallel(state_size=3, action_space_size=2, num_agents=2, buffer_size=8, batch_size=1)
    obs = {
        "a": np.random.randn(3).astype(np.float32),
        "b": np.random.randn(3).astype(np.float32),
    }
    actions = {"a": 0, "b": 1}
    rewards = {"a": 0.5, "b": -0.25}

    # Regression case: terminal step reports no next observations.
    pimac.store_parallel_step(obs, actions, rewards, next_obs_dict={}, done_dict={"__all__": True})

    assert len(pimac.memory) == 1
    ep = pimac.memory[0]
    assert ep["next_active_mask"].shape == (1, 2)
    np.testing.assert_array_equal(ep["next_active_mask"][0], np.zeros(2, dtype=np.float32))


def test_parallel_act_accepts_non_integer_agent_ids():
    np.random.seed(7)
    torch.manual_seed(7)

    pimac = PIMACParallel(state_size=4, action_space_size=3, num_agents=2)
    obs_step1 = {
        "agent_alpha": np.random.randn(4).astype(np.float32),
        "agent_beta": np.random.randn(4).astype(np.float32),
    }
    actions_step1 = pimac.act_parallel(obs_step1)
    assert set(actions_step1.keys()) == set(obs_step1.keys())

    obs_step2 = {
        "agent_beta": np.random.randn(4).astype(np.float32),
        "agent_alpha": np.random.randn(4).astype(np.float32),
    }
    actions_step2 = pimac.act_parallel(obs_step2)
    assert set(actions_step2.keys()) == set(obs_step2.keys())
    action_single = pimac.act(np.random.randn(4).astype(np.float32), agent_index=("tuple", 1))
    assert isinstance(action_single, int)
