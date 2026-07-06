from __future__ import annotations

from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.optimization as opt_module
from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.namelist import InData
from vmec_jax.optimization import BoundaryParamSpec, FixedBoundaryExactOptimizer


def _boundary(k: int = 4) -> BoundaryCoeffs:
    return BoundaryCoeffs(
        R_cos=np.arange(1.0, k + 1.0),
        R_sin=np.arange(11.0, 11.0 + k),
        Z_cos=np.arange(21.0, 21.0 + k),
        Z_sin=np.arange(31.0, 31.0 + k),
    )


def _run_ready_optimizer(residual: np.ndarray | None = None) -> FixedBoundaryExactOptimizer:
    residual = np.asarray([0.5, 3.0, 4.0] if residual is None else residual, dtype=float)
    state = SimpleNamespace(name="state")
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._scan_exact_path = "tape"
    opt._solver_device_name = None
    opt._indata = InData(scalars={"LASYM": False}, indexed={})
    opt._inside_solver_device_context = False
    opt._trial_residual_cache = OrderedDict()
    opt._exact_cache = {}
    opt._exact_state_cache = {}
    opt._exact_residual_cache = {}
    opt._initial_tangent_cache = {}
    opt._last_jacobian_key = [None]
    opt._last_jacobian_residual = None
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))
    opt._inner_max_iter = 11
    opt._inner_ftol = 1.0e-9
    opt._trial_max_iter = 3
    opt._trial_ftol = 1.0e-6
    opt._post_jacobian_clear = lambda *args, **kwargs: None
    opt._profile_dump = lambda: {}
    opt._exact_cache_key = lambda params: np.asarray(params, dtype=float).round(12).reshape(-1).tobytes()
    opt._aspect_target = 7.0
    opt._aspect_weight = 2.0
    opt._n_non_qs = 1
    opt._n_qs = None
    opt._has_residual_block_metadata = True
    opt._qs_total_from_state_fn = lambda _state: (_ for _ in ()).throw(
        AssertionError("metadata residuals should avoid state QS callback")
    )
    opt._objective_family = "qs"
    opt._helicity_m = 1
    opt._helicity_n = 0
    opt._specs = [BoundaryParamSpec("rc30", "rc", 0, 3, 0)]
    opt._base_params_vector = lambda: np.asarray([0.1], dtype=float)
    opt._evaluate_residuals_from_state = lambda _state: residual.copy()
    opt._solve_exact_with_tape = lambda _params, return_payload=False: (state, {}) if return_payload else state
    opt._solve_forward = lambda _params, trial=True: state
    opt.residual_fun = lambda _params: residual.copy()
    opt.forward_residual_fun = lambda _params: residual.copy() + 1.0
    opt._cached_exact_state = lambda _params: None
    opt._cached_exact_residual = lambda *args, **kwargs: residual.copy()
    opt._remember_exact_residual = lambda cache_key, res: opt._exact_residual_cache.update(
        {cache_key: np.asarray(res, dtype=float)}
    )
    return opt


def test_boundary_param_numpy_partial_updates_and_unknown_kind() -> None:
    boundary = _boundary(k=3)
    specs = [
        BoundaryParamSpec("rc10", "rc", 0, 1, 0),
        BoundaryParamSpec("rs10", "rs", 1, 1, 0),
        BoundaryParamSpec("zc10", "zc", 1, 1, 0),
        BoundaryParamSpec("zs10", "zs", 2, 1, 0),
    ]

    updated = opt_module._apply_boundary_params_numpy(boundary, specs, np.asarray([10.0, 20.0, 30.0, 40.0]))

    np.testing.assert_allclose(updated.R_cos, [11.0, 2.0, 3.0])
    np.testing.assert_allclose(updated.R_sin, [11.0, 32.0, 13.0])
    np.testing.assert_allclose(updated.Z_cos, [21.0, 52.0, 23.0])
    np.testing.assert_allclose(updated.Z_sin, [31.0, 32.0, 73.0])

    partial = opt_module._apply_boundary_params_numpy(boundary, specs, np.asarray([10.0]))
    np.testing.assert_allclose(partial.R_cos, [11.0, 2.0, 3.0])
    np.testing.assert_allclose(partial.R_sin, boundary.R_sin)

    bad = [BoundaryParamSpec("bad10", "bad", 0, 1, 0)]
    with pytest.raises(ValueError, match="Unknown boundary parameter kind"):
        opt_module._apply_boundary_params_numpy(boundary, bad, np.asarray([1.0]))


def test_fixed_boundary_exact_optimizer_init_can_be_unit_constructed(monkeypatch):
    state0 = SimpleNamespace(layout=SimpleNamespace(size=12))
    static = SimpleNamespace(s=np.asarray([0.0, 0.5, 1.0]), cfg=SimpleNamespace(lasym=False))
    indata = InData(
        scalars={"NITER_ARRAY": [2, 5], "FTOL_ARRAY": np.asarray([1.0e-7, 2.0e-8]), "DELT": 0.25},
        indexed={},
        source_path="input.test",
    )
    boundary = _boundary()
    flux = object()
    residuals_fn = lambda _state: np.asarray([1.0, 2.0])
    residuals_fn._n_non_qs = 0
    residuals_fn._n_qs = 2
    residuals_fn._aspect_target = None
    residuals_fn._aspect_weight = 3.0
    residuals_fn._qs_total_from_state = lambda _state: 99.0

    monkeypatch.setattr(FixedBoundaryExactOptimizer, "_make_residuals_eval_fn", lambda self, fn: fn)
    monkeypatch.setattr(opt_module, "initial_guess_from_boundary", lambda *args, **kwargs: state0)
    monkeypatch.setattr(opt_module, "eval_geom", lambda *_args, **_kwargs: SimpleNamespace(sqrtg=np.ones((2, 2))))
    monkeypatch.setattr(opt_module, "signgs_from_sqrtg", lambda *_args, **_kwargs: -1)
    monkeypatch.setattr(opt_module, "flux_profiles_from_indata", lambda *_args, **_kwargs: flux)

    opt = FixedBoundaryExactOptimizer(
        static,
        indata,
        boundary,
        [BoundaryParamSpec("rc10", "rc", 0, 1, 0)],
        residuals_fn,
        inner_max_iter=9,
        inner_ftol=3.0e-8,
        trial_max_iter=0,
        trial_ftol=0.0,
        solver_device="default",
    )

    assert opt.static is static
    assert opt.indata is indata
    assert opt.signgs == -1
    assert opt.flux is flux
    assert opt._layout is state0.layout
    assert opt._inner_max_iter == 9
    assert opt._inner_ftol == pytest.approx(3.0e-8)
    assert opt._trial_max_iter == 9
    assert opt._trial_ftol == pytest.approx(3.0e-8)
    assert opt._step_size == pytest.approx(0.25)
    assert opt._n_qs == 2
    assert opt._n_non_qs == 0
    assert opt._has_residual_block_metadata is True
    assert opt._scan_exact_path == "tape"
    assert opt._trial_residual_cache_max == 8


def test_fixed_boundary_exact_optimizer_init_uses_profiled_trial_scan_policy(monkeypatch):
    state0 = SimpleNamespace(layout=SimpleNamespace(size=8))
    static = SimpleNamespace(s=np.asarray([0.0, 0.5, 1.0]), cfg=SimpleNamespace(lasym=False))
    indata = InData(scalars={"NITER": 12, "FTOL": 1.0e-12, "DELT": 0.2}, indexed={})
    boundary = _boundary()

    residuals_fn = lambda _state: np.asarray([1.0, 2.0])
    residuals_fn._n_non_qs = 2

    monkeypatch.setattr(FixedBoundaryExactOptimizer, "_make_residuals_eval_fn", lambda self, fn: fn)
    monkeypatch.setattr(opt_module, "initial_guess_from_boundary", lambda *args, **kwargs: state0)
    monkeypatch.setattr(opt_module, "eval_geom", lambda *_args, **_kwargs: SimpleNamespace(sqrtg=np.ones((2, 2))))
    monkeypatch.setattr(opt_module, "signgs_from_sqrtg", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(opt_module, "flux_profiles_from_indata", lambda *_args, **_kwargs: object())

    monkeypatch.delenv("VMEC_JAX_OPT_TRIAL_SCAN", raising=False)
    monkeypatch.delenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE", raising=False)
    monkeypatch.delenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE_MAX_DOFS", raising=False)
    opt_gpu = FixedBoundaryExactOptimizer(
        static,
        indata,
        boundary,
        [BoundaryParamSpec("rc10", "rc", 0, 1, 0)],
        residuals_fn,
        inner_max_iter=7,
        inner_ftol=2.0e-9,
        solver_device="gpu",
    )

    assert opt_gpu._solver_device_name == "gpu"
    assert opt_gpu._trial_solver_kwargs["use_scan"] is True
    assert opt_gpu._trial_solver_kwargs["resume_state_mode"] == "none"
    assert opt_gpu._exact_solver_kwargs["use_scan"] is False
    assert opt_gpu._exact_solver_kwargs["preconditioner_use_precomputed_tridi"] is True
    assert opt_gpu._scan_exact_path == "tape"

    monkeypatch.setenv("VMEC_JAX_OPT_TRIAL_SCAN", "scan")
    opt_forced_scan = FixedBoundaryExactOptimizer(
        static,
        indata,
        boundary,
        [BoundaryParamSpec("rc10", "rc", 0, 1, 0)],
        residuals_fn,
        inner_max_iter=7,
        inner_ftol=2.0e-9,
        solver_device="gpu",
    )

    assert opt_forced_scan._trial_solver_kwargs["use_scan"] is True
    assert opt_forced_scan._exact_solver_kwargs["use_scan"] is False

    monkeypatch.delenv("VMEC_JAX_OPT_TRIAL_SCAN", raising=False)
    residuals_fn._objective_family = "qi"
    opt_qi = FixedBoundaryExactOptimizer(
        static,
        indata,
        boundary,
        [BoundaryParamSpec("rc10", "rc", 0, 1, 0)],
        residuals_fn,
        inner_max_iter=7,
        inner_ftol=2.0e-9,
        solver_device="gpu",
    )
    assert opt_qi._trial_solver_kwargs["use_scan"] is False

    monkeypatch.delenv("VMEC_JAX_OPT_TRIAL_SCAN", raising=False)
    residuals_fn._objective_family = "qs"
    residuals_fn._helicity_m = 0
    residuals_fn._helicity_n = -1
    opt_qp = FixedBoundaryExactOptimizer(
        static,
        indata,
        boundary,
        [BoundaryParamSpec("rc10", "rc", 0, 1, 0)],
        residuals_fn,
        inner_max_iter=7,
        inner_ftol=2.0e-9,
        solver_device="gpu",
    )
    assert opt_qp._trial_solver_kwargs["use_scan"] is True

    opt_qp_high = FixedBoundaryExactOptimizer(
        static,
        indata,
        boundary,
        [BoundaryParamSpec("rc40", "rc", 0, 4, 0)],
        residuals_fn,
        inner_max_iter=7,
        inner_ftol=2.0e-9,
        solver_device="gpu",
    )
    assert opt_qp_high._trial_solver_kwargs["use_scan"] is False


def test_auto_method_resolver_keeps_dense_for_high_mode_cpu_cases(monkeypatch):
    opt = _run_ready_optimizer()

    assert opt._resolve_optimizer_method("auto", None) == (
        "scipy",
        None,
        "auto:qa-dense-default",
    )

    opt._solver_device_name = "gpu"
    assert opt._resolve_optimizer_method("auto", None) == (
        "scipy",
        None,
        "auto:dense-preserves-gpu",
    )

    opt._solver_device_name = "metal"
    assert opt._resolve_optimizer_method("auto", None) == (
        "scipy",
        None,
        "auto:dense-preserves-metal",
    )

    opt._solver_device_name = None
    opt._helicity_n = -1
    assert opt._resolve_optimizer_method("auto", 7) == (
        "scipy",
        7,
        "auto:qh-dense-default",
    )

    opt._helicity_m = 0
    opt._helicity_n = 1
    assert opt._resolve_optimizer_method("auto", None) == (
        "scipy",
        None,
        "auto:qp-dense-default",
    )

    opt._objective_family = "qi"
    opt._helicity_m = None
    opt._helicity_n = None
    assert opt._resolve_optimizer_method("auto", None) == (
        "scipy",
        None,
        "auto:qi-dense-default",
    )
    assert opt._resolve_optimizer_method("auto_scalar", None) == (
        "scalar_trust",
        None,
        "auto_scalar:high-mode-scalar-trust",
    )

    opt._objective_family = "qs"
    opt._helicity_m = 1
    opt._helicity_n = 0
    opt._specs = [BoundaryParamSpec("rs30", "rs", 0, 3, 0)]
    assert opt._resolve_optimizer_method("auto", None) == (
        "scipy",
        None,
        "auto:dense-lasym",
    )
    assert opt._resolve_optimizer_method("auto_adjoint", None) == (
        "scipy",
        None,
        "auto_scalar:dense-lasym",
    )

    opt._specs = [BoundaryParamSpec("rc30", "rc", 0, 3, 0)]
    opt._indata = InData(scalars={"LASYM": True}, indexed={})
    assert opt._resolve_optimizer_method("auto", None) == (
        "scipy",
        None,
        "auto:dense-lasym",
    )
    opt._indata = InData(scalars={"LASYM": False}, indexed={})

    opt._specs = []
    assert opt._resolve_optimizer_method("auto", None) == (
        "scipy",
        None,
        "auto:dense-default",
    )
    assert opt._resolve_optimizer_method("auto_scalar", None) == (
        "scipy",
        None,
        "auto_scalar:dense-default",
    )

    opt._specs = [BoundaryParamSpec("rc20", "rc", 0, 2, 0)]
    assert opt._resolve_optimizer_method("matrix-free", 9) == ("scipy_matrix_free", 9, None)
    assert opt._resolve_optimizer_method("scipy-mf", None) == ("scipy_matrix_free", None, None)
    assert opt._resolve_optimizer_method("trf", None) == ("scipy", None, None)


def test_optimizer_private_policy_error_branches(monkeypatch):
    import vmec_jax._compat as compat

    opt = _run_ready_optimizer()

    def bad_default_backend():
        raise RuntimeError("synthetic backend failure")

    monkeypatch.setattr(compat, "jax", SimpleNamespace(default_backend=bad_default_backend))
    assert opt._resolve_solver_device("gpu") == "gpu"

    opt._solver_device_name = None
    assert opt._use_scan_for_trial_solves() is False
    assert opt._use_precomputed_tridi_for_exact_tape() is None

    opt._solver_device_name = "gpu"
    monkeypatch.setenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE_MAX_DOFS", "not-an-int")
    assert opt._use_precomputed_tridi_for_exact_tape() is True
    assert opt._select_exact_path() == "tape"

    opt._indata = SimpleNamespace(get_bool=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad")))
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=True))
    opt._specs = []
    assert opt._has_stellarator_asymmetric_configuration() is True

    opt._profile_add_counter("synthetic_counter", 3)
    assert opt._profile["synthetic_counter"] == {"count": 1, "wall_time_s": 3.0}
    solver_total = opt._profile_solver_timing(
        {
            "timing": {
                "scan_total_s": 2.0,
                "scan_runner_cache_hit_count": 4,
                "scan_runner_cache_miss_count": object(),
                "scan_runner_cache_miss_category_tolerance_count": 2,
            }
        },
        profile_prefix="scan",
        phase_wall_s=5.0,
        unattributed_name="scan_unattributed_total",
    )
    assert solver_total == pytest.approx(2.0)
    assert opt._profile["scan_scan_runner_cache_hit_count"]["wall_time_s"] == pytest.approx(4.0)
    assert opt._profile["scan_scan_runner_cache_miss_category_tolerance_count"]["wall_time_s"] == pytest.approx(2.0)
    assert opt._profile["scan_unattributed_total"]["wall_time_s"] == pytest.approx(3.0)


def test_gpu_projected_replay_policy_is_stellsym_mode3_plus_by_default(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_OPT_PROJECTED_REPLAY_RESIDUALS", raising=False)

    opt = _run_ready_optimizer()
    opt._solver_device_name = "gpu"
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))

    assert opt._projected_replay_residuals_enabled(24) is False
    assert opt._projected_replay_residuals_enabled(48) is True

    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=True))
    assert opt._projected_replay_residuals_enabled(48) is False
    assert opt._lasym_replay_column_chunk(48) == 8

    opt._solver_device_name = "cpu"
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))
    assert opt._projected_replay_residuals_enabled(7) is False
    assert opt._projected_replay_residuals_enabled(8) is True

    monkeypatch.setenv("VMEC_JAX_OPT_PROJECTED_REPLAY_RESIDUALS", "1")
    assert opt._projected_replay_residuals_enabled(1) is True


def test_optimizer_backend_name_preserves_explicit_device_and_falls_back(monkeypatch):
    import vmec_jax._compat as compat

    assert opt_module._optimizer_backend_name("gpu") == "gpu"

    monkeypatch.setattr(compat, "jax", SimpleNamespace(default_backend=lambda: "METAL"))
    assert opt_module._optimizer_backend_name(None) == "metal"

    def bad_default_backend():
        raise RuntimeError("synthetic backend probe failure")

    monkeypatch.setattr(compat, "jax", SimpleNamespace(default_backend=bad_default_backend))
    assert opt_module._optimizer_backend_name(None) == "cpu"


def test_move_to_solver_device_recurses_dataclasses_namedtuples_and_containers(monkeypatch):
    import vmec_jax._compat as compat

    class FakeArray:
        pass

    @dataclass(frozen=True)
    class Payload:
        arr: np.ndarray
        items: list

    Pair = namedtuple("Pair", ["left", "right"])

    puts = []
    fake_jax = SimpleNamespace(
        Array=FakeArray,
        devices=lambda name: [f"{name}:0"],
        device_put=lambda obj, device: puts.append((obj, device)) or ("put", np.asarray(obj).tolist(), device),
    )
    monkeypatch.setattr(compat, "jax", fake_jax)

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = "gpu"
    value = {
        "payload": Payload(arr=np.asarray([1.0, 2.0]), items=[np.asarray([3.0])]),
        "pair": Pair(np.asarray([4.0]), "kept"),
        "scalar": 5,
    }

    moved = opt._move_to_solver_device(value)

    assert moved["scalar"] == 5
    assert moved["payload"].arr == ("put", [1.0, 2.0], "gpu:0")
    assert moved["payload"].items == [("put", [3.0], "gpu:0")]
    assert moved["pair"] == Pair(("put", [4.0], "gpu:0"), "kept")
    assert len(puts) == 3

    opt._solver_device_name = None
    assert opt._move_to_solver_device(value) is value


def test_read_last_array_covers_numpy_empty_and_fallback_values():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._indata = InData(
        scalars={
            "A": np.asarray([1, 2, 3]),
            "B": np.asarray([]),
            "B_FALLBACK": 8,
            "C": [],
            "C_FALLBACK": 9.5,
        },
        indexed={},
        source_path=None,
    )

    assert opt._read_last_array("A", "MISSING", 0, int) == 3
    assert opt._read_last_array("B", "B_FALLBACK", 0, int) == 8
    assert opt._read_last_array("C", "C_FALLBACK", 0.0, float) == pytest.approx(9.5)
    assert opt._read_last_array("MISSING", "ALSO_MISSING", 4, int) == 4


def test_boundary_param_methods_cover_input_convention_and_all_kinds(monkeypatch):
    import vmec_jax.boundary as boundary_module

    converted = SimpleNamespace(converted=True)
    boundary = _boundary()
    specs = [
        BoundaryParamSpec("rc10", "rc", 0, 1, 0),
        BoundaryParamSpec("rs10", "rs", 1, 1, 0),
        BoundaryParamSpec("zc10", "zc", 2, 1, 0),
        BoundaryParamSpec("zs10", "zs", 3, 1, 0),
    ]
    calls = {}

    def fake_convert(boundary_arg, modes, *, lasym, apply_m1_constraint):
        calls["convert"] = (boundary_arg, modes, lasym, apply_m1_constraint)
        return converted

    monkeypatch.setattr(boundary_module, "boundary_from_input_convention", fake_convert)

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._boundary = boundary
    opt._boundary_input = boundary
    opt._specs = specs
    opt._static = SimpleNamespace(modes="modes", cfg=SimpleNamespace(lasym=True))

    params = np.asarray([0.1, 0.2, 0.3, 0.4])
    assert opt._boundary_from_params(params) is converted
    updated_input = opt._boundary_input_from_params(params)
    np.testing.assert_allclose(np.asarray(updated_input.R_cos), [1.1, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(np.asarray(updated_input.R_sin), [11.0, 12.2, 13.0, 14.0])
    np.testing.assert_allclose(np.asarray(updated_input.Z_cos), [21.0, 22.0, 23.3, 24.0])
    np.testing.assert_allclose(np.asarray(updated_input.Z_sin), [31.0, 32.0, 33.0, 34.4])
    np.testing.assert_allclose(opt._base_params_vector(), [1.0, 12.0, 23.0, 34.0])
    assert calls["convert"][1:] == ("modes", True, False)

    opt._specs = [BoundaryParamSpec("bad", "bad", 0, 0, 0)]
    with pytest.raises(ValueError, match="Unknown boundary parameter kind"):
        opt._base_params_vector()


def test_solve_forward_uses_trial_and_exact_solver_kwargs(monkeypatch):
    import vmec_jax.solve as solve_module

    calls = []
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._static = object()
    opt._indata = object()
    opt._profile = {}
    opt._trial_max_iter = 3
    opt._trial_ftol = 1.0e-4
    opt._inner_max_iter = 9
    opt._inner_ftol = 1.0e-9
    opt._trial_solver_kwargs = {"trial": True, "use_scan": True}
    opt._exact_solver_kwargs = {"exact": True}
    opt._boundary_from_params = lambda params: ("boundary", tuple(np.asarray(params, dtype=float)))

    monkeypatch.setattr(opt_module, "initial_guess_from_boundary", lambda *_args, **_kwargs: "state0")

    def fake_solve(state, static, **kwargs):
        calls.append((state, static, kwargs))
        return SimpleNamespace(state=f"solved:{len(calls)}")

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", fake_solve)

    assert opt._solve_forward(np.asarray([1.0]), trial=True) == "solved:1"
    assert opt._solve_forward(np.asarray([2.0]), trial=False) == "solved:2"
    assert calls[0][2]["max_iter"] == 3
    assert calls[0][2]["ftol"] == pytest.approx(1.0e-4)
    assert calls[0][2]["trial"] is True
    assert calls[0][2]["use_scan"] is True
    assert calls[0][2]["state_only"] is True
    assert calls[1][2]["max_iter"] == 9
    assert calls[1][2]["ftol"] == pytest.approx(1.0e-9)
    assert calls[1][2]["exact"] is True
    assert "state_only" not in calls[1][2]
    assert opt._profile["initial_guess_trial"]["count"] == 1
    assert opt._profile["initial_guess_forward"]["count"] == 1
    assert opt._profile["solve_forward_trial"]["count"] == 1
    assert opt._profile["solve_forward_exact"]["count"] == 1


def test_solve_forward_preserves_explicit_trial_state_only(monkeypatch):
    import vmec_jax.solve as solve_module

    calls = []
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._static = object()
    opt._indata = object()
    opt._profile = {}
    opt._trial_max_iter = 3
    opt._trial_ftol = 1.0e-4
    opt._trial_solver_kwargs = {"use_scan": True, "state_only": False}
    opt._boundary_from_params = lambda params: ("boundary", tuple(np.asarray(params, dtype=float)))

    monkeypatch.setattr(opt_module, "initial_guess_from_boundary", lambda *_args, **_kwargs: "state0")

    def fake_solve(state, static, **kwargs):
        calls.append((state, static, kwargs))
        return SimpleNamespace(state="solved")

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", fake_solve)

    assert opt._solve_forward(np.asarray([1.0]), trial=True) == "solved"
    assert calls[0][2]["use_scan"] is True
    assert calls[0][2]["state_only"] is False


def test_scan_exact_state_cache_miss_and_hit():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._exact_state_cache = {}
    opt._exact_residual_cache = {}
    opt._profile = {}
    opt._exact_cache_key = lambda _params: b"scan"
    helper_calls = []
    opt._scan_exact_helpers = lambda: {
        "state": lambda params: helper_calls.append(np.asarray(params, dtype=float).copy()) or "scan-state"
    }

    assert opt._solve_scan_exact_state(np.asarray([1.0])) == "scan-state"
    assert opt._solve_scan_exact_state(np.asarray([2.0])) == "scan-state"
    assert len(helper_calls) == 1
    np.testing.assert_allclose(helper_calls[0], [1.0])
    assert opt._profile["scan_exact_state_solve"]["count"] == 1
    assert opt._profile["scan_exact_state_cache_hit"]["count"] == 1


def test_residual_callbacks_remember_exact_and_trial_residuals():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._scan_exact_path = "scan"
    opt._profile = {}
    opt._trial_residual_cache = OrderedDict()
    opt._trial_residual_cache_max = 2
    opt._exact_residual_cache = {}
    opt._exact_cache_key = lambda params: np.asarray(params, dtype=float).reshape(-1).tobytes()
    opt._solve_scan_exact_state = lambda _params: "scan-state"
    opt._solve_exact_with_tape = lambda _params: "tape-state"
    opt._solve_forward = lambda _params, trial: "trial-state"
    opt._evaluate_residuals_from_state = lambda state: np.asarray(
        {"scan-state": [1.0], "tape-state": [2.0], "trial-state": [3.0]}[state],
        dtype=float,
    )

    np.testing.assert_allclose(opt.residual_fun(np.asarray([0.0])), [1.0])
    opt._scan_exact_path = "tape"
    np.testing.assert_allclose(opt.residual_fun(np.asarray([1.0])), [2.0])
    np.testing.assert_allclose(opt.forward_residual_fun(np.asarray([1.0])), [2.0])
    np.testing.assert_allclose(opt.forward_residual_fun(np.asarray([2.0])), [3.0])
    np.testing.assert_allclose(opt.forward_residual_fun(np.asarray([2.0])), [3.0])
    assert opt._profile["scan_residual_eval_exact"]["count"] == 1
    assert opt._profile["residual_eval_exact"]["count"] == 1
    assert opt._profile["trial_residual_exact_cache_hit"]["count"] == 1
    assert opt._profile["residual_eval_trial"]["count"] == 1
    assert opt._profile["trial_residual_cache_hit"]["count"] == 1


def test_residual_fun_reuses_cached_exact_residual_without_resolve():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._profile = {}
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._scan_exact_path = "tape"
    opt._last_jacobian_key = [None]
    opt._exact_cache_key = lambda params: np.asarray(params, dtype=float).reshape(-1).tobytes()
    key = opt._exact_cache_key(np.asarray([1.0]))
    opt._exact_residual_cache = {key: np.asarray([3.0, 4.0])}
    opt._solve_exact_with_tape = lambda _params: (_ for _ in ()).throw(
        AssertionError("cached residual should avoid exact resolve")
    )

    np.testing.assert_allclose(opt.residual_fun(np.asarray([1.0])), [3.0, 4.0])
    assert opt._profile["exact_residual_cache_hit"]["count"] == 1
    assert opt._profile["residual_exact_cache_hit"]["count"] == 1


def test_cache_helpers_and_clear_caches_manage_small_payloads():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._profile = {}
    opt._exact_cache_key = lambda params: np.asarray(params, dtype=float).reshape(-1).tobytes()
    key0 = opt._exact_cache_key(np.asarray([0.0]))
    key1 = opt._exact_cache_key(np.asarray([1.0]))
    opt._exact_cache = {key0: ("state0", "payload")}
    opt._exact_state_cache = {key1: "state1"}
    opt._exact_residual_cache = {key1: np.asarray([9.0])}
    opt._trial_residual_cache = OrderedDict()
    opt._trial_residual_cache_max = 0
    opt._initial_tangent_cache = {"x": object()}
    opt._initial_tangent_direction_cache = {"directions": object()}
    opt._discrete_jacobian_helper_cache = {"helper": object()}
    opt._scan_exact_helper_cache = {"scan": object()}
    opt._last_jacobian_residual = np.asarray([1.0])
    cleared = []
    opt._post_jacobian_clear = lambda *, clear_compiled=False: cleared.append(clear_compiled)

    assert opt._cached_exact_state(np.asarray([0.0])) == "state0"
    opt._exact_state_cache[key1] = "state1"
    assert opt._cached_exact_state(np.asarray([1.0])) == "state1"
    opt._remember_exact_state(key0, "new-state")
    assert opt._exact_state_cache == {key0: "new-state"}
    assert opt._exact_residual_cache == {}
    opt._remember_trial_residual(np.asarray([0.0]), np.asarray([1.0]))
    assert opt._trial_residual_cache == OrderedDict()

    opt.clear_caches()

    assert opt._exact_cache == {}
    assert opt._exact_state_cache == {}
    assert opt._exact_residual_cache == {}
    assert opt._initial_tangent_cache == {}
    assert opt._initial_tangent_direction_cache == {}
    assert opt._discrete_jacobian_helper_cache == {}
    assert opt._scan_exact_helper_cache == {}
    assert opt._last_jacobian_residual is None
    assert cleared == [True]


def test_initial_tangent_directions_reuse_identity_matrix():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._profile = {}
    opt._solver_device_name = "cpu"
    opt._initial_tangent_direction_cache = {}

    first = opt._initial_tangent_directions(np.asarray([0.0, 1.0]), profile_prefix="jacobian")
    second = opt._initial_tangent_directions(np.asarray([2.0, 3.0]), profile_prefix="jacobian")

    assert first is second
    np.testing.assert_allclose(np.asarray(first), np.eye(2))
    assert opt._profile["jacobian_initial_tangents_eye_cache_miss"]["count"] == 1
    assert opt._profile["jacobian_initial_tangents_eye_cache_hit"]["count"] == 1


def test_aspect_and_quasisymmetry_wrappers_use_configured_exact_path(monkeypatch):
    monkeypatch.setattr("vmec_jax.wout.equilibrium_aspect_ratio_from_state", lambda **_kwargs: 8.25)

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._scan_exact_path = "scan"
    opt._static = object()
    opt._solve_scan_exact_state = lambda _params: "scan-state"
    opt._solve_exact_with_tape = lambda _params: "tape-state"
    opt._evaluate_residuals_from_state = lambda state: np.asarray([1.0, 2.0]) if state == "scan-state" else np.asarray([3.0])
    opt._qs_total_from_state = lambda state, res: float(len(state) + np.sum(res))

    assert opt.aspect_ratio(np.asarray([0.0])) == pytest.approx(8.25)
    assert opt.quasisymmetry_objective(np.asarray([0.0])) == pytest.approx(len("scan-state") + 3.0)

    opt._scan_exact_path = "tape"
    assert opt.quasisymmetry_objective(np.asarray([0.0])) == pytest.approx(len("tape-state") + 3.0)


def test_run_scipy_path_scales_callbacks_and_records_trace(monkeypatch):
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    calls = {}

    def fake_least_squares(residuals, y0, *, jac, method, tr_solver, tr_options, max_nfev, ftol, gtol, xtol, verbose):
        calls["args"] = {
            "y0": np.asarray(y0, dtype=float),
            "method": method,
            "tr_solver": tr_solver,
            "tr_options": tr_options,
            "max_nfev": max_nfev,
            "ftol": ftol,
            "gtol": gtol,
            "xtol": xtol,
            "verbose": verbose,
        }
        np.testing.assert_allclose(residuals(y0), [0.5, 3.0, 4.0])
        np.testing.assert_allclose(jac(y0), [[2.0], [4.0], [6.0]])
        return SimpleNamespace(
            x=np.asarray([0.2]),
            cost=0.25,
            nfev=2,
            njev=1,
            success=True,
            status=1,
            message="synthetic scipy",
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)
    opt = _run_ready_optimizer()
    opt._jacobian_fun_tracked = lambda _x: np.asarray([[1.0], [2.0], [3.0]], dtype=float)

    result = opt.run(
        np.asarray([0.0]),
        method="scipy",
        max_nfev=4,
        ftol=1.0e-4,
        gtol=2.0e-4,
        xtol=3.0e-4,
        x_scale=np.asarray([2.0]),
        scipy_tr_solver="lsmr",
        scipy_lsmr_maxiter=6,
        trace_callbacks=True,
        verbose=1,
    )

    np.testing.assert_allclose(calls["args"]["y0"], [0.05])
    assert calls["args"]["tr_options"] == {"maxiter": 6}
    np.testing.assert_allclose(result["x"], [0.3])
    assert result["_history_dump"]["method"] == "scipy"
    assert result["_history_dump"]["callback_trace"]["summary"]["residual:exact_residual_cache"]["count"] == 1
    assert result["_history_dump"]["callback_trace"]["summary"]["jacobian:exact_tape_replay"]["count"] == 1


def test_run_auto_method_records_resolved_dense_policy(monkeypatch):
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    jacobian_calls = {}

    def fake_least_squares(residuals, y0, *, jac, method, tr_solver, tr_options, max_nfev, ftol, gtol, xtol, verbose):
        np.testing.assert_allclose(residuals(y0), [0.5, 3.0, 4.0])
        jacobian_calls["dense"] = jac(y0)
        np.testing.assert_allclose(jacobian_calls["dense"], [[2.0], [4.0], [6.0]])
        assert method == "trf"
        assert tr_solver == "lsmr"
        assert tr_options == {}
        return SimpleNamespace(
            x=np.asarray([0.2]),
            cost=0.25,
            nfev=2,
            njev=1,
            success=True,
            status=1,
            message="synthetic dense auto",
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)
    opt = _run_ready_optimizer()
    opt._jacobian_fun_tracked = lambda _x: np.asarray([[1.0], [2.0], [3.0]], dtype=float)

    result = opt.run(
        np.asarray([0.0]),
        method="auto",
        max_nfev=4,
        x_scale=np.asarray([2.0]),
        verbose=0,
    )

    np.testing.assert_allclose(jacobian_calls["dense"], [[2.0], [4.0], [6.0]])
    np.testing.assert_allclose(result["x"], [0.0])
    assert result["_history_dump"]["method"] == "scipy"
    assert result["_history_dump"]["method_requested"] == "auto"
    assert result["_history_dump"]["method_auto_reason"] == "auto:qa-dense-default"
    assert result["_history_dump"]["scipy_tr_solver"] == "lsmr"
    assert opt._profile["method_auto_scipy"]["count"] == 1


def test_run_scipy_dense_sanitizes_nonfinite_trial_residuals_and_jacobian(monkeypatch):
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    def fake_least_squares(residuals, y0, *, jac, **kwargs):
        np.testing.assert_allclose(residuals(y0), [1.0, 2.0])
        bad_trial = residuals(y0 + 1.0)
        assert bad_trial.shape == (2,)
        assert np.all(np.isfinite(bad_trial))
        assert np.max(np.abs(bad_trial)) >= 1.0e11
        np.testing.assert_allclose(jac(y0), [[0.0], [0.0]])
        return SimpleNamespace(
            x=np.asarray(y0, dtype=float),
            cost=0.5,
            nfev=2,
            njev=1,
            success=True,
            status=1,
            message="dense finite guard ok",
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)
    opt = _run_ready_optimizer(residual=np.asarray([1.0, 2.0]))
    x0_key = opt._exact_cache_key(np.asarray([0.0]))

    def cached_residual(*args, **kwargs):
        cache_key = kwargs.get("cache_key")
        return np.asarray([1.0, 2.0], dtype=float) if cache_key == x0_key else None

    opt._cached_exact_residual = cached_residual
    opt._cached_exact_state = lambda _x: None
    opt._cached_trial_residual = lambda _x: None
    opt.forward_residual_fun = lambda _x: np.asarray([np.nan, np.inf], dtype=float)
    opt._jacobian_fun_tracked = lambda _x: np.asarray([[np.nan], [-np.inf]], dtype=float)

    result = opt.run(
        np.asarray([0.0]),
        method="scipy",
        x_scale=np.asarray([2.0]),
        verbose=0,
    )

    assert result["success"] is True
    assert opt._profile["dense_nonfinite_residual"]["count"] == 1
    assert opt._profile["dense_nonfinite_jacobian"]["count"] == 1


def test_run_scipy_matrix_free_scales_multi_parameter_linear_operator(monkeypatch):
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    class FakeOperator:
        shape = (2, 2)

        def matvec(self, vector):
            return np.asarray([vector[0] + 2.0 * vector[1], 3.0 * vector[0] - vector[1]], dtype=float)

        def matmat(self, matrix):
            return np.vstack(
                [
                    matrix[0] + 2.0 * matrix[1],
                    3.0 * matrix[0] - matrix[1],
                ]
            )

        def rmatvec(self, vector):
            return np.asarray(
                [
                    vector[0] + 3.0 * vector[1],
                    2.0 * vector[0] - vector[1],
                ],
                dtype=float,
            )

    def fake_least_squares(residuals, y0, *, jac, **kwargs):
        np.testing.assert_allclose(residuals(y0), [1.0, -2.0])
        op = jac(y0)
        np.testing.assert_allclose(op.matvec(np.asarray([5.0, 7.0])), [68.0, 46.0])
        np.testing.assert_allclose(op.matmat(np.asarray([[5.0, 11.0], [7.0, 13.0]])), [[68.0, 124.0], [46.0, 100.0]])
        np.testing.assert_allclose(op.rmatvec(np.asarray([2.0, -3.0])), [-28.0, 14.0])
        return SimpleNamespace(
            x=np.asarray([0.0, 0.0]),
            cost=0.0,
            nfev=1,
            njev=1,
            success=True,
            status=1,
            message="synthetic multi-parameter matrix-free",
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)
    opt = _run_ready_optimizer(residual=np.asarray([1.0, -2.0]))
    opt._base_params_vector = lambda: np.zeros(2)
    opt.residual_linear_operator = lambda _x: FakeOperator()
    opt._exact_cache_key = lambda params: np.asarray(params, dtype=float).round(12).reshape(-1).tobytes()

    result = opt.run(
        np.asarray([0.0, 0.0]),
        method="scipy_matrix_free",
        x_scale=np.asarray([2.0, 4.0]),
        verbose=0,
    )

    np.testing.assert_allclose(result["x"], [0.0, 0.0])
    assert result["_history_dump"]["method"] == "scipy_matrix_free"


def test_run_scipy_matrix_free_sanitizes_nonfinite_trial_residuals_and_products(monkeypatch):
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    class FakeOperator:
        shape = (2, 1)

        def matvec(self, vector):
            assert np.all(np.isfinite(vector))
            return np.asarray([np.nan, np.inf], dtype=float)

        def matmat(self, matrix):
            assert np.all(np.isfinite(matrix))
            return np.asarray([[np.nan], [-np.inf]], dtype=float)

        def rmatvec(self, vector):
            assert np.all(np.isfinite(vector))
            return np.asarray([np.nan], dtype=float)

    def fake_least_squares(residuals, y0, *, jac, **kwargs):
        np.testing.assert_allclose(residuals(y0), [1.0, 2.0])
        bad_trial = residuals(y0 + 1.0)
        assert bad_trial.shape == (2,)
        assert np.all(np.isfinite(bad_trial))
        assert np.max(np.abs(bad_trial)) >= 1.0e11
        op = jac(y0)
        np.testing.assert_allclose(op.matvec(np.asarray([3.0])), [0.0, 0.0])
        np.testing.assert_allclose(op.matmat(np.asarray([[3.0]])), [[0.0], [0.0]])
        np.testing.assert_allclose(op.rmatvec(np.asarray([1.0, 2.0])), [0.0])
        return SimpleNamespace(
            x=np.asarray(y0, dtype=float),
            cost=0.5,
            nfev=2,
            njev=1,
            success=True,
            status=1,
            message="matrix-free finite guard ok",
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)
    opt = _run_ready_optimizer(residual=np.asarray([1.0, 2.0]))

    def cached_residual(x=None, *, cache_key=None):
        if cache_key is not None:
            return np.asarray([1.0, 2.0], dtype=float)
        return np.asarray([1.0, 2.0], dtype=float) if np.allclose(np.asarray(x, dtype=float), [0.0]) else None

    opt._cached_exact_residual = cached_residual
    opt.forward_residual_fun = lambda _x: np.asarray([np.nan, np.inf], dtype=float)
    opt.residual_linear_operator = lambda _x: FakeOperator()

    result = opt.run(
        np.asarray([0.0]),
        method="scipy_matrix_free",
        x_scale=np.asarray([2.0]),
        verbose=0,
    )

    assert result["success"] is True
    assert opt._profile["matrix_free_nonfinite_residual"]["count"] == 1
    assert opt._profile["matrix_free_nonfinite_matvec"]["count"] == 1
    assert opt._profile["matrix_free_nonfinite_matmat"]["count"] == 1
    assert opt._profile["matrix_free_nonfinite_rmatvec"]["count"] == 1


def test_run_scipy_residual_callback_prefers_cache_policies_before_trial_solve(monkeypatch):
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    residual_sources = []
    forward_calls = []
    state_cache = {"state0": object()}
    trial_cache = {0.5: np.asarray([5.0, 6.0], dtype=float)}

    def fake_least_squares(residuals, y0, *, jac, **kwargs):
        np.testing.assert_allclose(residuals(y0), [1.0, 2.0])
        np.testing.assert_allclose(residuals(np.asarray([0.6])), [5.0, 6.0])
        np.testing.assert_allclose(residuals(np.asarray([1.1])), [9.0, 10.0])
        np.testing.assert_allclose(jac(y0), [[0.0], [0.0]])
        return SimpleNamespace(
            x=np.asarray(y0, dtype=float),
            cost=2.5,
            nfev=3,
            njev=1,
            success=True,
            status=1,
            message="synthetic cache policy",
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)
    opt = _run_ready_optimizer(residual=np.asarray([1.0, 2.0]))
    opt._cached_exact_residual = lambda *args, **kwargs: None

    def cached_state(x):
        x0 = float(np.asarray(x, dtype=float)[0])
        if x0 == pytest.approx(0.0):
            residual_sources.append("exact_state")
            return state_cache["state0"]
        return None

    def cached_trial(x):
        x0 = float(np.asarray(x, dtype=float)[0])
        if x0 in trial_cache:
            residual_sources.append("trial_cache")
            return trial_cache[x0]
        return None

    def forward_residual(x):
        forward_calls.append(float(np.asarray(x, dtype=float)[0]))
        residual_sources.append("trial_solve")
        return np.asarray([9.0, 10.0], dtype=float)

    opt._cached_exact_state = cached_state
    opt._cached_trial_residual = cached_trial
    opt._evaluate_residuals_from_state = lambda state: np.asarray([1.0, 2.0], dtype=float)
    opt.forward_residual_fun = forward_residual
    opt._jacobian_fun_tracked = lambda _x: np.zeros((2, 1), dtype=float)

    result = opt.run(np.asarray([0.0]), method="scipy", max_nfev=3, trace_callbacks=True, verbose=0)

    assert residual_sources == ["exact_state", "trial_cache", "trial_solve", "exact_state"]
    assert forward_calls == [1.0]
    assert result["_history_dump"]["callback_trace"]["summary"]["residual:exact_state_cache"]["count"] == 1
    assert result["_history_dump"]["callback_trace"]["summary"]["residual:trial_residual_cache"]["count"] == 1
    assert result["_history_dump"]["callback_trace"]["summary"]["residual:trial_solve"]["count"] == 1


def test_run_scipy_matrix_free_returns_best_exact_point_after_scipy_failure(monkeypatch):
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    def fake_least_squares(residuals, y0, *, jac, **kwargs):
        np.testing.assert_allclose(residuals(y0), [0.5, 3.0, 4.0])
        raise ValueError("synthetic non-finite trial Jacobian")

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)
    opt = _run_ready_optimizer()

    result = opt.run(
        np.asarray([0.0]),
        method="scipy_matrix_free",
        max_nfev=4,
        x_scale=np.asarray([2.0]),
        scipy_lsmr_maxiter=5,
        verbose=0,
    )

    np.testing.assert_allclose(result["x"], [0.0])
    assert result["success"] is False
    assert result["status"] == -1
    assert "scipy matrix-free least_squares failed" in result["message"]
    assert result["_history_dump"]["selected_best_exact_point"] is True
    assert "synthetic non-finite trial Jacobian" in result["_history_dump"]["optimizer_exception"]
