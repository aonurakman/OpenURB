import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from algorithms.pimac import (
    PIMAC,
    PIMACActorRNN,
    SetTokenTeacher,
    SetValueCritic,
)


def test_network_width_mismatch_raises():
    """Actor width schedules must align with the configured hidden-layer count."""
    with pytest.raises(AssertionError):
        PIMACActorRNN(
            obs_dim=3,
            action_dim=2,
            num_hidden=1,
            widths=(8, 8, 4),
            rnn_hidden_dim=8,
            ctx_dim=8,
        )


def test_actor_uses_film_hyper_projection_and_returns_aux():
    """Actor keeps FiLM and exposes hypernetwork diagnostics in aux outputs."""
    torch.manual_seed(0)

    actor = PIMACActorRNN(
        obs_dim=5,
        action_dim=3,
        num_hidden=1,
        widths=(8, 8),
        rnn_hidden_dim=7,
        ctx_dim=6,
        hypernet_rank=2,
        hypernet_hidden_sizes=(8, 8),
    )
    assert actor.policy_head.in_features == 7
    expected_hyper_out = (actor.hidden_dim * actor.hypernet_rank) + (actor.hypernet_rank * actor.action_dim) + actor.action_dim
    assert actor.hypernet_delta_head[-1].out_features == expected_hyper_out

    obs = torch.randn(2, 4, 5)
    logits, _, aux = actor(obs, return_aux=True)
    assert logits.shape == (2, 4, 3)
    assert aux["ctx_mu"].shape == (2, 4, 6)
    assert aux["ctx_logvar"].shape == (2, 4, 6)
    assert aux["gate"].shape == (2, 4, 1)
    assert aux["delta_w_l2"].shape == (2, 4)
    assert aux["delta_b_l2"].shape == (2, 4)
    assert aux["delta_w_norm"].shape == (2, 4)
    assert aux["delta_b_norm"].shape == (2, 4)
    assert torch.all(aux["gate"] >= 0.0)
    assert torch.all(aux["gate"] <= 1.0)
    for key in ("ctx_mu", "ctx_logvar", "gate", "delta_w_l2", "delta_b_l2", "delta_w_norm", "delta_b_norm"):
        assert torch.isfinite(aux[key]).all()


def test_actor_gate_scales_hypernetwork_delta_contribution():
    """Lower uncertainty should increase gate values and hyper-delta impact on logits."""
    torch.manual_seed(9)

    actor = PIMACActorRNN(
        obs_dim=4,
        action_dim=2,
        num_hidden=1,
        widths=(8, 8),
        rnn_hidden_dim=6,
        ctx_dim=5,
        hypernet_rank=2,
        hypernet_hidden_sizes=(),
        hypernet_delta_init_scale=1.0,
        ctx_logvar_min=-6.0,
        ctx_logvar_max=4.0,
    )

    with torch.no_grad():
        actor.input_layer.weight.zero_()
        actor.input_layer.bias.zero_()
        for layer in actor.hidden_layers:
            layer.weight.zero_()
            layer.bias.zero_()
        for parameter in actor.rnn.parameters():
            parameter.zero_()

        actor.ctx_mu_head.weight.zero_()
        actor.ctx_mu_head.bias.fill_(1.0)
        actor.ctx_logvar_head.weight.zero_()

        actor.film_head.weight.zero_()
        actor.film_head.bias.zero_()

        actor.policy_head.weight.zero_()
        actor.policy_head.bias.zero_()

        for parameter in actor.hypernet_delta_head.parameters():
            parameter.zero_()
        # Make db non-zero so gate directly controls logit magnitude through bias deltas.
        actor.hypernet_delta_head[-1].bias[-actor.action_dim :] = 1.0

        actor.gate_weight.fill_(1.0)
        actor.gate_bias.zero_()

    obs = torch.randn(1, 3, 4)

    with torch.no_grad():
        actor.ctx_logvar_head.bias.fill_(4.0)
        logits_high_uncertainty, _, aux_high_uncertainty = actor(obs, return_aux=True)

        actor.ctx_logvar_head.bias.fill_(-6.0)
        logits_low_uncertainty, _, aux_low_uncertainty = actor(obs, return_aux=True)

    high_uncertainty_gate = aux_high_uncertainty["gate"].mean()
    low_uncertainty_gate = aux_low_uncertainty["gate"].mean()

    assert low_uncertainty_gate > high_uncertainty_gate
    assert logits_low_uncertainty.abs().mean() > logits_high_uncertainty.abs().mean()


def test_set_teacher_permutation_padding_and_context_equivariance():
    """Teacher tokens must be set-invariant and per-agent contexts permutation-equivariant."""
    torch.manual_seed(0)

    teacher = SetTokenTeacher(
        obs_dim=4,
        set_embed_dim=8,
        num_tokens=3,
        set_encoder_hidden_sizes=(16, 16),
    )
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

    assert torch.allclose(tokens_a, tokens_b, atol=1e-6)
    assert torch.allclose(tokens_a, tokens_c, atol=1e-6)

    assert torch.allclose(ctx_a, ctx_b[:, :, inv_perm, :], atol=1e-6)
    assert torch.allclose(ctx_a, ctx_c[:, :, : obs.shape[2], :], atol=1e-6)
    assert torch.allclose(
        ctx_c[:, :, obs.shape[2] :, :],
        torch.zeros_like(ctx_c[:, :, obs.shape[2] :, :]),
        atol=1e-7,
    )


def test_set_teacher_all_inactive_is_finite():
    """All-inactive sets should remain numerically stable."""
    torch.manual_seed(1)

    teacher = SetTokenTeacher(
        obs_dim=3,
        set_embed_dim=8,
        num_tokens=2,
        set_encoder_hidden_sizes=(16, 16),
    )
    obs = torch.randn(2, 3, 5, 3)
    active_mask = torch.zeros(2, 3, 5)

    tokens, context = teacher(obs, active_mask)

    assert torch.isfinite(tokens).all()
    assert torch.isfinite(context).all()
    assert torch.allclose(context, torch.zeros_like(context), atol=1e-7)


def test_set_critic_permutation_invariance():
    """Shuffling the agent axis should not change team values."""
    torch.manual_seed(0)

    critic = SetValueCritic(
        obs_dim=4,
        set_embed_dim=8,
        num_tokens=3,
        set_encoder_hidden_sizes=(16, 16),
        critic_hidden_sizes=(16, 16),
        include_team_size_feature=True,
    )
    critic.eval()

    obs = torch.randn(2, 3, 5, 4)
    active_mask = torch.tensor(
        [
            [[1, 1, 1, 0, 1], [1, 0, 1, 1, 0], [0, 0, 0, 0, 0]],
            [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 0, 1, 0, 1]],
        ],
        dtype=torch.float32,
    )

    perm = torch.tensor([2, 4, 0, 3, 1])
    obs_perm = obs[:, :, perm, :]
    mask_perm = active_mask[:, :, perm]

    v_a = critic(obs, active_mask)
    v_b = critic(obs_perm, mask_perm)

    assert torch.allclose(v_a, v_b, atol=1e-6)


def test_set_critic_inactive_padding_invariance():
    """Appending inactive padded agents should not change values."""
    torch.manual_seed(1)

    critic = SetValueCritic(
        obs_dim=3,
        set_embed_dim=8,
        num_tokens=3,
        set_encoder_hidden_sizes=(16, 16),
        critic_hidden_sizes=(16, 16),
        include_team_size_feature=True,
    )
    critic.eval()

    obs = torch.randn(2, 4, 3, 3)
    active_mask = torch.tensor(
        [
            [[1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]],
            [[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
        ],
        dtype=torch.float32,
    )

    padded_obs = torch.cat([obs, torch.randn(2, 4, 2, 3)], dim=2)
    padded_mask = torch.cat([active_mask, torch.zeros(2, 4, 2)], dim=2)

    v_a = critic(obs, active_mask)
    v_b = critic(padded_obs, padded_mask)

    assert torch.allclose(v_a, v_b, atol=1e-6)


def test_set_critic_all_inactive_is_finite():
    """All-inactive timesteps should stay finite and not produce NaNs."""
    torch.manual_seed(2)

    critic = SetValueCritic(
        obs_dim=5,
        set_embed_dim=8,
        num_tokens=3,
        set_encoder_hidden_sizes=(16, 16),
        critic_hidden_sizes=(16, 16),
        include_team_size_feature=True,
    )

    obs = torch.randn(3, 2, 4, 5)
    active_mask = torch.zeros(3, 2, 4)

    values = critic(obs, active_mask)

    assert values.shape == (3, 2)
    assert torch.isfinite(values).all()


def test_act_uses_argmax_when_deterministic():
    np.random.seed(0)
    torch.manual_seed(0)

    pimac = PIMAC(
        state_size=3,
        action_space_size=4,
        buffer_size=8,
        batch_size=2,
        lr=0.01,
        num_epochs=1,
        num_hidden=1,
        widths=(8, 8),
        rnn_hidden_dim=8,
        critic_hidden_sizes=(8, 8),
        set_embed_dim=8,
        set_encoder_hidden_sizes=(8, 8),
        num_tokens=2,
        hypernet_delta_init_scale=0.0,
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
    pimac = PIMAC(state_size=2, action_space_size=2, gamma=0.9, gae_lambda=0.8)

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

    pimac = PIMAC(state_size=4, action_space_size=3, buffer_size=8, batch_size=1)

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


def test_learn_updates_parameters_and_logs_metrics():
    np.random.seed(2)
    torch.manual_seed(2)

    pimac = PIMAC(
        state_size=3,
        action_space_size=2,
        buffer_size=16,
        batch_size=2,
        lr=0.01,
        num_epochs=2,
        num_hidden=1,
        widths=(16, 16),
        rnn_hidden_dim=16,
        critic_hidden_sizes=(16, 16),
        set_embed_dim=16,
        set_encoder_hidden_sizes=(16, 16),
        num_tokens=3,
        distill_weight=0.2,
        teacher_ema_tau=0.05,
    )

    for _ in range(2):
        obs_batch = np.random.randn(2, 3).astype(np.float32)
        actions_batch = np.array([0, 1], dtype=np.int64)
        rewards_batch = np.array([1.0, -0.5], dtype=np.float32)
        active_mask = np.array([1.0, 1.0], dtype=np.float32)
        pimac.store_episode(obs_batch, actions_batch, rewards_batch, active_mask)

    before_params = [p.detach().clone() for p in pimac.actor_net.parameters()]
    before_base_policy_weight = pimac.actor_net.policy_head.weight.detach().clone()
    assert len(pimac.memory) == 2

    pimac.learn()

    after_params = [p.detach() for p in pimac.actor_net.parameters()]
    after_base_policy_weight = pimac.actor_net.policy_head.weight.detach()
    changed = any(not torch.allclose(p0, p1) for p0, p1 in zip(before_params, after_params))
    base_head_changed = not torch.allclose(before_base_policy_weight, after_base_policy_weight)

    assert len(pimac.loss) == 1
    assert changed
    assert base_head_changed
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
        "hyper_l2",
        "ctx_logvar_mean",
        "ctx_logvar_std",
        "gate_mean",
        "gate_std",
        "delta_w_norm_mean",
        "delta_b_norm_mean",
        "token_norm_mean",
        "teacher_ctx_norm_mean",
        "active_decisions",
        "active_agents_mean",
    }
    assert expected_keys.issubset(last.keys())
    assert last["active_decisions"] > 0
    for key in expected_keys:
        assert np.isfinite(float(last[key]))


def test_variable_n_batching_smoke_without_fixed_max_critic_dependency():
    np.random.seed(3)
    torch.manual_seed(3)

    pimac = PIMAC(
        state_size=4,
        action_space_size=3,
        buffer_size=4,
        batch_size=2,
        num_epochs=1,
        lr=1e-3,
        num_hidden=1,
        widths=(16, 16),
        rnn_hidden_dim=16,
        critic_hidden_sizes=(16, 16),
        set_embed_dim=16,
        set_encoder_hidden_sizes=(16, 16),
        num_tokens=3,
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


def test_aec_cycle_storage_smoke():
    np.random.seed(5)
    torch.manual_seed(5)

    pimac = PIMAC(state_size=3, action_space_size=4, buffer_size=8, batch_size=1)
    agent_ids = ["a", "b", "c"]
    pimac.aec_begin_cycle(agent_ids)

    for aid in agent_ids:
        obs = np.random.randn(3).astype(np.float32)
        pimac.aec_record(aid, obs=obs, action=1, reward=1.0, next_obs=obs, done=False)

    pimac.aec_end_cycle(done_all=True)
    assert len(pimac.memory) == 1


def test_ema_teacher_updates_toward_online_teacher():
    """EMA update should move target-teacher parameters toward online teacher parameters."""
    torch.manual_seed(11)

    pimac = PIMAC(
        state_size=3,
        action_space_size=2,
        set_embed_dim=8,
        set_encoder_hidden_sizes=(8, 8),
        critic_hidden_sizes=(8, 8),
        num_tokens=2,
        teacher_ema_tau=0.2,
    )

    target_params_before = [parameter.detach().clone() for parameter in pimac.target_teacher.parameters()]

    with torch.no_grad():
        for parameter in pimac.critic.teacher.parameters():
            parameter.add_(0.5)

    pimac._update_target_teacher()

    target_params_after = [parameter.detach().clone() for parameter in pimac.target_teacher.parameters()]
    online_params_after = [parameter.detach().clone() for parameter in pimac.critic.teacher.parameters()]

    for before, after, online in zip(target_params_before, target_params_after, online_params_after):
        expected = before * 0.8 + online * 0.2
        assert torch.allclose(after, expected, atol=1e-6)

