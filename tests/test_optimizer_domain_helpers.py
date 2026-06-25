import numpy as np
import pytest

from vmec_jax.optimizers.fixed_boundary.history import ResidualHistoryPolicy
from vmec_jax.optimizers.fixed_boundary.history import build_run_history_dump
from vmec_jax.optimizers.fixed_boundary.history import history_entry_from_residuals
from vmec_jax.optimizers.fixed_boundary.history import monotone_final_wall_time
from vmec_jax.optimizers.fixed_boundary.history import qs_objective_from_residuals
from vmec_jax.optimizers.fixed_boundary.scalar_trust import scalar_cost_only_trials_enabled
from vmec_jax.optimizers.fixed_boundary.scalar_trust import scalar_trust_direction


def test_residual_history_policy_reconstructs_qs_and_aspect():
    policy = ResidualHistoryPolicy(
        aspect_target=5.0,
        aspect_weight=2.0,
        n_non_qs=2,
        n_qs=None,
        has_residual_block_metadata=True,
    )
    residual = np.asarray([0.4, 99.0, 3.0, 4.0])

    assert policy.can_build_history_from_residuals()
    assert qs_objective_from_residuals(residual, policy) == pytest.approx(25.0)

    entry = history_entry_from_residuals(residual, wall_time_s=1.25, policy=policy)
    assert entry["wall_time_s"] == pytest.approx(1.25)
    assert entry["aspect"] == pytest.approx(5.2)
    assert entry["cost"] == pytest.approx(0.5 * float(np.dot(residual, residual)))
    assert entry["qs_objective"] == pytest.approx(25.0)


def test_residual_history_policy_uses_explicit_qs_tail_block():
    policy = ResidualHistoryPolicy(
        aspect_target=4.0,
        aspect_weight=1.0,
        n_non_qs=5,
        n_qs=2,
    )

    assert qs_objective_from_residuals([10.0, 20.0, 1.0, 2.0], policy) == pytest.approx(5.0)


def test_residual_history_policy_rejects_iota_or_missing_aspect():
    with_iota = ResidualHistoryPolicy(
        aspect_target=5.0,
        aspect_weight=1.0,
        has_residual_block_metadata=True,
        has_iota_callback=True,
    )
    missing_aspect = ResidualHistoryPolicy(
        aspect_target=None,
        aspect_weight=1.0,
        has_residual_block_metadata=True,
    )

    assert not with_iota.can_build_history_from_residuals()
    assert not missing_aspect.can_build_history_from_residuals()
    with pytest.raises(ValueError, match="aspect residual weight"):
        history_entry_from_residuals([1.0], wall_time_s=0.0, policy=missing_aspect)


def test_build_run_history_dump_preserves_public_keys():
    dump = build_run_history_dump(
        max_nfev=3,
        ftol=1e-2,
        gtol=1e-3,
        xtol=1e-4,
        method_key="scipy_matrix_free",
        method_requested="auto",
        method_auto_reason="cpu_high_mode",
        exact_path="scan",
        scipy_tr_solver="exact",
        scipy_lsmr_maxiter=4,
        lbfgs_step_bound=0.2,
        scalar_step_bound=None,
        scalar_cost_only_trials_used=True,
        solver_device="cpu",
        inner_max_iter=10,
        inner_ftol=1e-9,
        trial_max_iter=5,
        trial_ftol=1e-7,
        final_wall_time_s=2.5,
        result={"nfev": 3, "njev": 2, "success": False, "message": "budget"},
        cost0=10.0,
        cost_final=2.0,
        qs_total0=8.0,
        qs_total_final=1.0,
        aspect0=6.0,
        aspect_final=5.5,
        history=[{"cost": 10.0}],
        profile={"solve": 1.0},
        selected_best_exact=True,
        rejected_trial_exact_history_count=2,
        optimizer_exception=RuntimeError("trial failed"),
        iota_fn_present=True,
        entry0={"iota": 0.1},
        entry_final={"iota": 0.2},
        target_iota=0.3,
        target_aspect=5.0,
        callback_trace=[{"kind": "residual"}],
    )

    assert dump["scipy_tr_solver"] == "lsmr"
    assert dump["objective_initial"] == pytest.approx(20.0)
    assert dump["objective_final"] == pytest.approx(4.0)
    assert dump["selected_best_exact_point"] is True
    assert dump["iota_initial"] == pytest.approx(0.1)
    assert dump["target_iota"] == pytest.approx(0.3)
    assert dump["callback_trace"] == [{"kind": "residual"}]
    assert "trial failed" in dump["optimizer_exception"]


def test_monotone_final_wall_time_uses_last_history_entry():
    assert monotone_final_wall_time(now_s=1.0, history=[{"wall_time_s": 3.0}]) == pytest.approx(3.0)
    assert monotone_final_wall_time(now_s=4.0, history=[{"wall_time_s": 3.0}]) == pytest.approx(4.0)


def test_scalar_trust_policy_and_direction(monkeypatch):
    class Owner:
        def __init__(self):
            self.profile = {}

        def _profile_add(self, name, value):
            self.profile[name] = self.profile.get(name, 0.0) + float(value)

    owner = Owner()
    monkeypatch.delenv("VMEC_JAX_OPT_SCALAR_COST_ONLY_TRIALS", raising=False)
    assert not scalar_cost_only_trials_enabled(owner, None)
    owner._scalar_trust_cost_only_trials = True
    assert scalar_cost_only_trials_enabled(owner, None)
    monkeypatch.setenv("VMEC_JAX_OPT_SCALAR_COST_ONLY_TRIALS", "false")
    assert not scalar_cost_only_trials_enabled(owner, None)
    assert scalar_cost_only_trials_enabled(owner, True)

    grad = np.asarray([2.0, -1.0])
    np.testing.assert_allclose(scalar_trust_direction(owner, grad, []), -grad)
    assert "scalar_trust_gradient_direction" in owner.profile

