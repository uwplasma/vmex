from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.namelist import InData
from vmec_jax.optimization import BoundaryParamSpec, FixedBoundaryExactOptimizer


def _policy_optimizer(*, lasym: bool = False, solver_device: str | None = None, max_mode: int = 3):
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = solver_device
    opt._indata = InData(scalars={"LASYM": lasym}, indexed={})
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=lasym))
    opt._objective_family = "qs"
    opt._helicity_m = 1
    opt._helicity_n = 0
    opt._specs = [BoundaryParamSpec(f"rc{max_mode}0", "rc", 0, max_mode, 0)]
    return opt


@pytest.mark.parametrize("solver_device", [None, "gpu"])
def test_auto_scalar_routes_high_mode_stellsym_production_to_scalar_trust(solver_device):
    opt = _policy_optimizer(lasym=False, solver_device=solver_device, max_mode=3)

    method, lsmr_maxiter, reason = opt._resolve_optimizer_method("auto_scalar", None)

    assert method == "scalar_trust"
    assert lsmr_maxiter is None
    assert reason is not None
    assert reason.startswith("auto_scalar:")
    assert reason.endswith("high-mode-scalar-trust")


@pytest.mark.parametrize(
    ("lasym", "max_mode", "expected_reasons"),
    [
        (True, 3, {"auto_scalar:dense-lasym"}),
        (False, 2, {"auto_scalar:dense-default", "auto_scalar:dense-preserves-gpu"}),
    ],
)
def test_auto_scalar_keeps_lasym_and_low_mode_on_dense(lasym, max_mode, expected_reasons):
    opt = _policy_optimizer(lasym=lasym, solver_device="gpu", max_mode=max_mode)

    method, lsmr_maxiter, actual_reason = opt._resolve_optimizer_method("auto_scalar", None)

    assert method == "scipy"
    assert lsmr_maxiter is None
    assert actual_reason in expected_reasons


def test_explicit_scalar_trust_routing_is_unaffected_on_gpu():
    opt = _policy_optimizer(lasym=False, solver_device="gpu", max_mode=3)

    assert opt._resolve_optimizer_method("scalar_trust", 5) == ("scalar_trust", 5, None)


def test_gpu_scalar_gradient_initial_tangent_projection_policy(monkeypatch):
    opt = _policy_optimizer(lasym=False, solver_device="gpu", max_mode=3)

    assert opt._scalar_gradient_initial_tangents_enabled(24)
    assert not opt._scalar_gradient_initial_tangents_enabled(23)

    opt._solver_device_name = "cpu"
    assert not opt._scalar_gradient_initial_tangents_enabled(80)

    opt._solver_device_name = "gpu"
    opt._indata = InData(scalars={"LASYM": True}, indexed={})
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=True))
    assert not opt._scalar_gradient_initial_tangents_enabled(80)

    monkeypatch.setenv("VMEC_JAX_OPT_SCALAR_GRADIENT_INITIAL_TANGENTS", "1")
    assert opt._scalar_gradient_initial_tangents_enabled(1)
    monkeypatch.setenv("VMEC_JAX_OPT_SCALAR_GRADIENT_INITIAL_TANGENTS", "0")
    assert not opt._scalar_gradient_initial_tangents_enabled(80)


def _run_ready_scalar_optimizer() -> FixedBoundaryExactOptimizer:
    residual0 = np.asarray([0.0, 2.0], dtype=float)
    state0 = SimpleNamespace(name="state-0")
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._scan_exact_path = "tape"
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._trial_residual_cache = OrderedDict()
    opt._exact_cache = {}
    opt._exact_state_cache = {}
    opt._exact_residual_cache = {}
    opt._exact_jacobian_cache = {}
    opt._initial_tangent_cache = {}
    opt._last_jacobian_key = [None]
    opt._last_jacobian_residual = None
    opt._indata = InData(scalars={"LASYM": False}, indexed={})
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))
    opt._layout = SimpleNamespace(size=2)
    opt._objective_family = "qs"
    opt._helicity_m = 1
    opt._helicity_n = 0
    opt._specs = [BoundaryParamSpec("rc30", "rc", 0, 3, 0)]
    opt._inner_max_iter = 1
    opt._inner_ftol = 1.0e-9
    opt._trial_max_iter = 1
    opt._trial_ftol = 1.0e-6
    opt._exact_solver_kwargs = {"use_scan": False, "light_history": True, "resume_state_mode": "none"}
    opt._trial_solver_kwargs = {"use_scan": False, "light_history": True, "resume_state_mode": "none"}
    opt._post_jacobian_clear = lambda *args, **kwargs: None
    opt._profile_dump = lambda: {}
    opt._profile_add = lambda *args, **kwargs: None
    opt._exact_cache_key = lambda params: np.asarray(params, dtype=float).round(12).reshape(-1).tobytes()
    opt._base_params_vector = lambda: np.zeros(1)
    opt._aspect_target = 7.0
    opt._aspect_weight = 1.0
    opt._n_non_qs = 1
    opt._n_qs = None
    opt._has_residual_block_metadata = True
    opt._last_residual_size = residual0.size
    opt._last_jacobian_shape = None
    opt._last_jacobian_source = "exact_tape_replay"
    opt._last_scalar_gradient_source = None
    opt._last_scalar_cost_only_trials = None
    opt._qs_total_from_state_fn = None
    opt._residuals_fn = SimpleNamespace(_residual_block_summary=None)
    opt._evaluate_residuals_from_state = lambda _state: residual0.copy()
    opt._solve_forward = lambda _params, trial=True: state0
    opt.forward_residual_fun = lambda _params: residual0.copy()
    opt._remember_exact_state = FixedBoundaryExactOptimizer._remember_exact_state.__get__(
        opt, FixedBoundaryExactOptimizer
    )
    opt._remember_exact_residual = FixedBoundaryExactOptimizer._remember_exact_residual.__get__(
        opt, FixedBoundaryExactOptimizer
    )
    opt._remember_best_exact_point = FixedBoundaryExactOptimizer._remember_best_exact_point.__get__(
        opt, FixedBoundaryExactOptimizer
    )
    opt._cached_exact_state = FixedBoundaryExactOptimizer._cached_exact_state.__get__(opt, FixedBoundaryExactOptimizer)
    opt._cached_exact_residual = FixedBoundaryExactOptimizer._cached_exact_residual.__get__(
        opt, FixedBoundaryExactOptimizer
    )

    def solve_exact(params, return_payload=False):
        key = opt._exact_cache_key(params)
        opt._exact_cache[key] = (state0, {})
        opt._remember_exact_state(key, state0)
        opt._remember_exact_residual(key, residual0)
        return (state0, {}) if return_payload else state0

    opt._solve_exact_with_tape = solve_exact
    opt.residual_fun = lambda params: solve_exact(params) and residual0.copy()
    opt.objective_and_gradient_fun = lambda _params: (0.5 * float(np.dot(residual0, residual0)), np.asarray([0.0]))
    return opt


@pytest.mark.parametrize(
    ("method", "expected_reason", "expected_cost_only"),
    [
        ("auto_scalar", "auto_scalar:high-mode-scalar-trust", False),
        ("scalar_trust", None, False),
    ],
)
def test_scalar_trust_run_records_auto_scalar_production_defaults(method, expected_reason, expected_cost_only):
    opt = _run_ready_scalar_optimizer()

    result = opt.run(np.asarray([0.0]), method=method, max_nfev=1, verbose=0)

    assert result["method"] == "scalar_trust"
    assert result["method_auto_reason"] == expected_reason
    assert result["_history_dump"]["scalar_cost_only_trials"] is expected_cost_only
    assert result["_history_dump"]["exact_callback_metadata"]["scalar_cost_only_trials"] is expected_cost_only
