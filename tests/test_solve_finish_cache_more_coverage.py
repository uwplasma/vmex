from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.solve as solve
from vmec_jax._compat import jnp
from vmec_jax.solvers.fixed_boundary.scan import controller as scan_controller
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.vmec_tomnsp import TomnspsRZL


class _FakeInData:
    scalars = {}
    indexed = {}

    def __init__(self, *, lmove_axis: bool = True):
        self._lmove_axis = bool(lmove_axis)

    def get_float(self, name, default=0.0):
        return {"DELT": 0.1, "FTOL": 1.0e-6, "GAMMA": 0.0, "TCON0": 0.0}.get(str(name).upper(), default)

    def get_bool(self, name, default=False):
        return {"LFORBAL": False, "LMOVE_AXIS": self._lmove_axis, "LRFP": False}.get(str(name).upper(), default)

    def get_int(self, name, default=0):
        return {"NCURR": 0, "NSTEP": 1}.get(str(name).upper(), default)


def _state() -> VMECState:
    layout = StateLayout(ns=3, K=2, lasym=False)
    zeros = np.zeros((3, 2), dtype=float)
    rcos = zeros.copy()
    rcos[:, 0] = 1.0
    rcos[:, 1] = 0.1
    zsin = zeros.copy()
    zsin[:, 1] = 0.2
    return VMECState(
        layout=layout,
        Rcos=rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zsin,
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )


def _grid() -> SimpleNamespace:
    return SimpleNamespace(theta=np.asarray([0.0, np.pi]), zeta=np.asarray([0.0]), nfp=1)


def _trig(**overrides) -> SimpleNamespace:
    base = dict(
        mscale=np.ones(2),
        nscale=np.ones(1),
        ntheta1=4,
        cosnv=np.ones((1, 1)),
        cosmu=np.ones((1, 2)),
        r0scale=1.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _static(*, lfreeb: bool = False, grid=None, trig=None) -> SimpleNamespace:
    cfg = SimpleNamespace(
        ns=3,
        mpol=2,
        ntor=0,
        nfp=1,
        ntheta=4,
        nzeta=1,
        lasym=False,
        lthreed=True,
        lconm1=True,
        lfreeb=bool(lfreeb),
        nvacskip=2,
    )
    modes = SimpleNamespace(m=np.asarray([0, 1]), n=np.asarray([0, 0]), K=2)
    return SimpleNamespace(
        cfg=cfg,
        s=np.asarray([0.0, 0.5, 1.0]),
        modes=modes,
        grid=_grid() if grid is None else grid,
        trig_vmec=trig,
    )


def _quiet_solve_env(monkeypatch) -> None:
    for name in (*solve._HEAVY_DUMP_ENVS, *solve._LIGHT_DUMP_ENVS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VMEC_JAX_SCAN_FALLBACK", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_PRINT", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_PRINT_CHUNKED", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_PREFLIGHT", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_EXTRA_ITERS", "0")
    monkeypatch.setenv("VMEC_JAX_TIMING", "0")


def _install_setup_fakes(monkeypatch, *, build_static=None, eval_profiles=None, trig_factory=None) -> None:
    import vmec_jax.boundary as boundary_mod
    import vmec_jax.energy as energy_mod
    import vmec_jax.profiles as profiles_mod
    import vmec_jax.static as static_mod
    import vmec_jax.vmec_tomnsp as tomnsp_mod

    def fake_flux(_indata, s, signgs):
        del signgs
        arr = jnp.ones_like(jnp.asarray(s))
        return SimpleNamespace(chipf=jnp.zeros_like(arr), phipf=arr, phips=arr)

    monkeypatch.setattr(energy_mod, "flux_profiles_from_indata", fake_flux)
    monkeypatch.setattr(boundary_mod, "boundary_from_indata", lambda *_args, **_kwargs: SimpleNamespace(R_cos=np.ones(2)))
    monkeypatch.setattr(tomnsp_mod, "vmec_angle_grid", lambda **_kwargs: _grid())
    monkeypatch.setattr(tomnsp_mod, "vmec_trig_tables", trig_factory or (lambda **_kwargs: _trig()))
    monkeypatch.setattr(static_mod, "build_static", build_static or (lambda cfg, grid, **_kwargs: _static(grid=grid)))
    monkeypatch.setattr(profiles_mod, "eval_profiles", eval_profiles or (lambda _indata, _s: {"pressure": np.ones(1)}))
    monkeypatch.setattr(solve, "_mass_half_mesh_from_indata", lambda **_kwargs: jnp.zeros(3))
    monkeypatch.setattr(solve, "_pressure_half_mesh_from_indata", lambda **_kwargs: jnp.zeros(3))
    monkeypatch.setattr(solve, "_icurv_full_mesh_from_indata", lambda **_kwargs: jnp.zeros(3))


def _run_precompile_only(static, indata=None, **kwargs):
    params = dict(
        indata=_FakeInData() if indata is None else indata,
        signgs=1,
        max_iter=1,
        step_size=0.1,
        vmec2000_control=True,
        strict_update=True,
        backtracking=False,
        use_scan=False,
        jit_forces=False,
        precompile_only=True,
        verbose=False,
        verbose_vmec2000_table=False,
    )
    params.update(kwargs)
    return solve.solve_fixed_boundary_residual_iter(_state(), static, **params)


def test_precompile_setup_disables_jit_for_debug_dumps_and_sanitizes_freeb_controls(monkeypatch, capsys) -> None:
    pytest.importorskip("jax")
    _quiet_solve_env(monkeypatch)
    _install_setup_fakes(monkeypatch)
    monkeypatch.setenv("VMEC_JAX_DUMP_GC", "1")
    monkeypatch.setenv("VMEC_JAX_AXIS_RESET_FSQ_MIN", "-2")

    result = _run_precompile_only(
        _static(lfreeb=True),
        use_scan=True,
        jit_forces=True,
        stage_transition_factor=0.0,
        verbose=True,
    )

    assert result.diagnostics == {"precompile_only": True}
    assert "jit_forces disabled" in capsys.readouterr().out


def test_precompile_setup_rebuilds_static_after_grid_probe_failure_and_handles_profile_errors(monkeypatch) -> None:
    pytest.importorskip("jax")
    _quiet_solve_env(monkeypatch)
    build_calls = []

    class BrokenGrid:
        nfp = 1
        zeta = np.asarray([0.0])

        @property
        def theta(self):
            raise RuntimeError("synthetic grid failure")

    def fake_build_static(cfg, grid, **_kwargs):
        build_calls.append((cfg, grid))
        return _static(lfreeb=True, grid=grid)

    def fail_profiles(_indata, _s):
        raise RuntimeError("synthetic profile failure")

    _install_setup_fakes(monkeypatch, build_static=fake_build_static, eval_profiles=fail_profiles)
    monkeypatch.setenv("VMEC_JAX_AXIS_RESET_FSQ_MIN", "not-a-float")

    result = _run_precompile_only(_static(lfreeb=True, grid=BrokenGrid()))

    assert result.diagnostics == {"precompile_only": True}
    assert len(build_calls) == 1


def test_precompile_setup_rebuilds_mismatched_trig_tables(monkeypatch) -> None:
    pytest.importorskip("jax")
    _quiet_solve_env(monkeypatch)
    trig_calls = []

    def fake_trig_tables(**kwargs):
        trig_calls.append(kwargs)
        return _trig()

    bad_trig = _trig(ntheta1=99, cosnv=np.ones((2, 2)), cosmu=np.ones((1, 1)))
    _install_setup_fakes(monkeypatch, trig_factory=fake_trig_tables)

    result = _run_precompile_only(_static(trig=bad_trig))

    assert result.diagnostics == {"precompile_only": True}
    assert len(trig_calls) == 1
    assert trig_calls[0]["mmax"] == 1
    assert trig_calls[0]["nmax"] == 0


def _install_scan_fakes(monkeypatch) -> None:
    import vmec_jax.preconditioner_1d_jax as precond_mod
    import vmec_jax.vmec_forces as forces_mod
    import vmec_jax.vmec_residue as residue_mod

    _install_setup_fakes(monkeypatch)

    def fake_forces(*, state, **_kwargs):
        dtype = jnp.asarray(state.Rcos).dtype
        one = jnp.ones((3, 1, 1), dtype=dtype)
        jac = SimpleNamespace(sqrtg=one, r12=one, ru12=one, zu12=one, tau=one)
        bc = SimpleNamespace(
            guu=one,
            bsupu=one,
            bsupv=one,
            bsubu=one,
            bsubv=one,
            bsq=one,
            jac=jac,
            lamscale=1.0,
        )
        return SimpleNamespace(state=state, bc=bc, tcon=jnp.zeros(3, dtype=dtype))

    def fake_residual(k, **_kwargs):
        arr = 0.01 * jnp.ones((3, 2, 1), dtype=jnp.asarray(k.state.Rcos).dtype)
        return TomnspsRZL(frcc=arr, frss=None, fzsc=arr, fzcs=None, flsc=arr, flcs=None)

    def fake_norms(**_kwargs):
        return SimpleNamespace(
            r1=jnp.asarray(1.0),
            fnorm=jnp.asarray(1.0),
            fnormL=jnp.asarray(1.0),
            wb=jnp.asarray(0.0),
            wp=jnp.asarray(0.0),
            volume=jnp.asarray(1.0),
            r2=jnp.asarray(1.0),
        )

    monkeypatch.setattr(forces_mod, "vmec_forces_rz_from_wout", fake_forces)
    monkeypatch.setattr(forces_mod, "vmec_residual_internal_from_kernels", fake_residual)
    monkeypatch.setattr(residue_mod, "vmec_apply_m1_constraints", lambda *, frzl, lconm1: frzl)
    monkeypatch.setattr(residue_mod, "vmec_apply_scalxc_to_tomnsps", lambda *, frzl, s: frzl)
    monkeypatch.setattr(residue_mod, "vmec_zero_m1_zforce", lambda *, frzl, enabled: frzl)
    monkeypatch.setattr(residue_mod, "vmec_gcx2_from_tomnsps", lambda **_kwargs: (jnp.asarray(0.01),) * 3)
    monkeypatch.setattr(residue_mod, "vmec_force_norms_from_bcovar_dynamic", fake_norms)
    monkeypatch.setattr(residue_mod, "vmec_scalxc_from_s", lambda *, s, mpol: jnp.ones((len(s), mpol)))
    monkeypatch.setattr(residue_mod, "vmec_wint_from_trig", lambda _trig, nzeta: np.ones((1, nzeta)))
    monkeypatch.setattr(solve, "_scan_math_ptau_minmax_from_k_jax", lambda _k, **_kwargs: (jnp.asarray(0.1), jnp.asarray(0.2)))
    monkeypatch.setattr(precond_mod, "lambda_preconditioner_cached", lambda **_kwargs: jnp.ones((3, 2, 1)))
    monkeypatch.setattr(
        precond_mod,
        "rz_preconditioner_matrices",
        lambda **_kwargs: ({"dr": jnp.ones((2, 2, 1)), "dz": jnp.ones((2, 2, 1))}, 0, 2),
    )
    monkeypatch.setattr(precond_mod, "rz_preconditioner_apply", lambda **kwargs: kwargs["frzl_in"])


_COND_SENTINEL = object()


def _python_cond(pred, true_fun, false_fun, *operands, operand=_COND_SENTINEL):
    selected = true_fun if bool(np.asarray(pred)) else false_fun
    if operand is not _COND_SENTINEL:
        return selected(operand)
    if not operands:
        return selected()
    if len(operands) == 1:
        return selected(operands[0])
    return selected(*operands)


def _python_scan(fn, carry, xs, reverse=False):
    values = list(np.asarray(xs))
    if reverse:
        values = list(reversed(values))
    ys = []
    for value in values:
        carry, y = fn(carry, jnp.asarray(value, dtype=getattr(xs, "dtype", jnp.int32)))
        ys.append(y)
    if not ys:
        return carry, ()
    hist = solve.jax.tree_util.tree_map(lambda *parts: jnp.stack(parts, axis=0), *ys)
    if reverse:
        hist = solve.jax.tree_util.tree_map(lambda a: jnp.flip(a, axis=0), hist)
    return carry, hist


def test_nonscan_reuses_preconditioner_seed_from_same_bcovar_refresh(monkeypatch) -> None:
    pytest.importorskip("jax")
    _quiet_solve_env(monkeypatch)
    _install_scan_fakes(monkeypatch)

    import vmec_jax.preconditioner_1d_jax as precond_mod
    import vmec_jax.preconditioner_1d as precond_np_mod

    calls = {"lambda": 0, "mats": 0}

    def fake_lambda_preconditioner_cached(**_kwargs):
        calls["lambda"] += 1
        return jnp.ones((3, 2, 1))

    def fake_rz_preconditioner_matrices(**_kwargs):
        calls["mats"] += 1
        mats = {
            "dr": jnp.ones((2, 2, 1)),
            "dz": jnp.ones((2, 2, 1)),
        }
        return mats, 0, 2

    monkeypatch.setattr(precond_mod, "lambda_preconditioner_cached", fake_lambda_preconditioner_cached)
    monkeypatch.setattr(precond_np_mod, "lambda_preconditioner", fake_lambda_preconditioner_cached)
    monkeypatch.setattr(precond_mod, "rz_preconditioner_matrices", fake_rz_preconditioner_matrices)
    monkeypatch.setattr(precond_mod, "rz_preconditioner_matrices_numpy_host", fake_rz_preconditioner_matrices)
    monkeypatch.setattr(precond_mod, "rz_preconditioner_apply_jit", lambda **kwargs: kwargs["frzl_in"])
    monkeypatch.setattr(precond_mod, "rz_preconditioner_apply_numpy", lambda **kwargs: kwargs["frzl_in"])
    monkeypatch.setattr(solve, "_scan_math_ptau_minmax_from_k_host", lambda _k, **_kwargs: (0.1, 0.2))
    monkeypatch.setenv("VMEC_JAX_TIMING", "1")

    result = solve.solve_fixed_boundary_residual_iter(
        _state(),
        _static(),
        indata=_FakeInData(lmove_axis=False),
        signgs=1,
        max_iter=1,
        step_size=0.1,
        vmec2000_control=True,
        strict_update=True,
        backtracking=False,
        use_scan=False,
        jit_forces=False,
        auto_flip_force=False,
        limit_dt_from_force=False,
        limit_update_rms=False,
        verbose=False,
        verbose_vmec2000_table=False,
    )

    assert calls == {"lambda": 1, "mats": 1}
    timing = result.diagnostics["timing"]
    assert timing["precond_refresh_calls"] == 1
    assert timing["precond_cache_hit_count"] == 1
    assert timing["precond_refresh_seed_reuse_count"] == 1
    assert timing["precond_refresh_seed_s"] >= 0.0


def test_vmec2000_state_only_scan_runner_cache_reports_miss_then_hit_and_replays_resume_cache(monkeypatch) -> None:
    pytest.importorskip("jax")
    _quiet_solve_env(monkeypatch)
    _install_scan_fakes(monkeypatch)
    monkeypatch.setenv("VMEC_JAX_TIMING", "1")
    monkeypatch.setattr(solve, "jit", lambda fn, *args, **kwargs: fn)
    monkeypatch.setattr(scan_controller, "jit", lambda fn, *args, **kwargs: fn)
    monkeypatch.setattr(solve.jax.lax, "cond", _python_cond)
    monkeypatch.setattr(solve.jax.lax, "scan", _python_scan)
    monkeypatch.setattr(solve.jax, "block_until_ready", lambda value: value)
    solve._SCAN_RUNNER_CACHE.clear()

    resume_state = {
        "vmec2000_cache_valid": True,
        "cache_precond_diag": (jnp.zeros(3), jnp.zeros(3)),
        "cache_tcon": jnp.zeros(3),
        "cache_norms": SimpleNamespace(r1=jnp.asarray(1.0), fnorm=jnp.asarray(1.0), fnormL=jnp.asarray(1.0)),
        "cache_rz_scale": jnp.ones(3),
        "cache_l_scale": jnp.ones(3),
        "cache_rz_norm": object(),
        "cache_f_norm1": object(),
        "cache_prec_rz_mats": {"dr": jnp.ones((2, 2, 1)), "dz": jnp.ones((2, 2, 1))},
        "cache_prec_lam_prec": jnp.ones((3, 2, 1)),
    }
    params = dict(
        indata=_FakeInData(lmove_axis=False),
        signgs=1,
        max_iter=1,
        step_size=0.1,
        vmec2000_control=True,
        strict_update=True,
        backtracking=False,
        use_scan=True,
        jit_forces=False,
        state_only=True,
        resume_state=resume_state,
        scan_minimal_default=True,
        verbose=False,
        verbose_vmec2000_table=False,
    )

    first = solve.solve_fixed_boundary_residual_iter(_state(), _static(), **params)
    second = solve.solve_fixed_boundary_residual_iter(_state(), _static(), **params)

    assert first.diagnostics["timing"]["scan_runner_cache_miss_count"] == 1
    assert first.diagnostics["timing"]["scan_runner_cache_hit_count"] == 0
    assert second.diagnostics["timing"]["scan_runner_cache_miss_count"] == 0
    assert second.diagnostics["timing"]["scan_runner_cache_hit_count"] == 1
    assert len(solve._SCAN_RUNNER_CACHE) == 1


def test_precompile_only_jit_precompile_exercises_force_cache_and_lower(monkeypatch) -> None:
    pytest.importorskip("jax")
    _quiet_solve_env(monkeypatch)
    _install_scan_fakes(monkeypatch)
    solve._COMPUTE_FORCES_CACHE.clear()
    compiled = []

    class FakeJit:
        def __init__(self, fn):
            self.fn = fn
            self.lower_calls = []

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def lower(self, *args, **kwargs):
            self.lower_calls.append((args, kwargs))
            self.fn(*args, **kwargs)
            return SimpleNamespace(compile=lambda: "compiled")

    def fake_jit(fn, *args, **kwargs):
        del args, kwargs
        wrapped = FakeJit(fn)
        compiled.append(wrapped)
        return wrapped

    monkeypatch.setattr(solve, "jit", fake_jit)

    first = _run_precompile_only(_static(), jit_forces=True, jit_precompile=True, host_update_assembly=False)
    second = _run_precompile_only(_static(), jit_forces=True, jit_precompile=True, host_update_assembly=False)

    assert first.diagnostics == {"precompile_only": True}
    assert second.diagnostics == {"precompile_only": True}
    assert len(compiled) == 1
    assert [len(obj.lower_calls) for obj in compiled] == [4]
    assert len(solve._COMPUTE_FORCES_CACHE) == 1


def test_precompile_only_compute_force_cache_is_owned_and_limited(monkeypatch) -> None:
    pytest.importorskip("jax")
    _quiet_solve_env(monkeypatch)
    _install_scan_fakes(monkeypatch)
    monkeypatch.setenv("VMEC_JAX_COMPUTE_FORCES_CACHE_SIZE", "1")
    solve._COMPUTE_FORCES_CACHE.clear()
    compiled = []

    class FakeJit:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def lower(self, *args, **kwargs):
            self.fn(*args, **kwargs)
            return SimpleNamespace(compile=lambda: "compiled")

    def fake_jit(fn, *args, **kwargs):
        del args, kwargs
        wrapped = FakeJit(fn)
        compiled.append(wrapped)
        return wrapped

    static_a = _static()
    static_b = _static()
    static_b.cfg.nfp = 2
    monkeypatch.setattr(solve, "jit", fake_jit)

    _run_precompile_only(static_a, jit_forces=True, jit_precompile=True, host_update_assembly=False)
    first_key = next(iter(solve._COMPUTE_FORCES_CACHE))
    _run_precompile_only(static_b, jit_forces=True, jit_precompile=True, host_update_assembly=False)
    second_key = next(iter(solve._COMPUTE_FORCES_CACHE))
    _run_precompile_only(static_a, jit_forces=True, jit_precompile=True, host_update_assembly=False)
    third_key = next(iter(solve._COMPUTE_FORCES_CACHE))

    assert first_key != second_key
    assert third_key == first_key
    assert len(solve._COMPUTE_FORCES_CACHE) == 1
    assert len(compiled) == 3


def test_precompile_only_jit_precompile_swallows_compile_failure(monkeypatch) -> None:
    pytest.importorskip("jax")
    _quiet_solve_env(monkeypatch)
    _install_scan_fakes(monkeypatch)
    solve._COMPUTE_FORCES_CACHE.clear()

    class FailingJit:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def lower(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("synthetic compile failure")

    monkeypatch.setattr(solve, "jit", lambda fn, *args, **kwargs: FailingJit(fn))

    result = _run_precompile_only(_static(), jit_forces=True, jit_precompile=True)

    assert result.diagnostics == {"precompile_only": True}
    assert len(solve._COMPUTE_FORCES_CACHE) == 1


def test_accelerated_scan_runner_cache_reports_timing_hit_and_miss(monkeypatch) -> None:
    pytest.importorskip("jax")
    _quiet_solve_env(monkeypatch)
    _install_scan_fakes(monkeypatch)
    monkeypatch.setenv("VMEC_JAX_TIMING", "1")
    monkeypatch.setattr(solve, "jit", lambda fn, *args, **kwargs: fn)
    monkeypatch.setattr(solve.jax.lax, "cond", _python_cond)
    monkeypatch.setattr(solve.jax.lax, "scan", _python_scan)
    monkeypatch.setattr(solve.jax, "block_until_ready", lambda value: value)
    solve._SCAN_RUNNER_CACHE.clear()

    params = dict(
        indata=_FakeInData(lmove_axis=False),
        signgs=1,
        max_iter=2,
        step_size=0.1,
        vmec2000_control=False,
        strict_update=False,
        backtracking=False,
        use_scan=True,
        auto_flip_force=False,
        limit_dt_from_force=False,
        limit_update_rms=False,
        use_restart_triggers=False,
        use_direct_fallback=False,
        reference_mode=False,
        jit_forces=False,
        precond_radial_alpha=0.0,
        precond_lambda_alpha=0.0,
        verbose=False,
        verbose_vmec2000_table=False,
    )

    first = solve.solve_fixed_boundary_residual_iter(_state(), _static(), **params)
    second = solve.solve_fixed_boundary_residual_iter(_state(), _static(), **params)

    assert first.diagnostics["accelerated_scan"] is True
    assert first.diagnostics["scan_path"] == "accelerated"
    assert second.diagnostics["accelerated_scan"] is True
    assert first.diagnostics["timing"]["scan_runner_cache_miss_count"] == 1
    assert first.diagnostics["timing"]["scan_runner_cache_hit_count"] == 0
    assert second.diagnostics["timing"]["scan_runner_cache_miss_count"] == 0
    assert second.diagnostics["timing"]["scan_runner_cache_hit_count"] == 1
    assert len(solve._SCAN_RUNNER_CACHE) == 1


def test_nonscan_non_strict_backtracking_accepts_momentum_update(monkeypatch) -> None:
    pytest.importorskip("jax")
    _quiet_solve_env(monkeypatch)
    _install_scan_fakes(monkeypatch)

    result = solve.solve_fixed_boundary_residual_iter(
        _state(),
        _static(),
        indata=_FakeInData(lmove_axis=False),
        signgs=1,
        max_iter=1,
        step_size=0.1,
        vmec2000_control=False,
        strict_update=False,
        backtracking=True,
        use_scan=False,
        auto_flip_force=False,
        limit_dt_from_force=False,
        limit_update_rms=False,
        use_restart_triggers=False,
        use_direct_fallback=False,
        reference_mode=False,
        jit_forces=False,
        verbose=False,
        verbose_vmec2000_table=False,
    )

    assert result.n_iter == 0
    assert result.diagnostics["converged"] is False
    assert result.step_history.shape == (1,)
    assert result.diagnostics["step_status_history"][0] in {"momentum", "rejected"}
    assert result.diagnostics["restart_path_history"][0] == "non_strict"


def test_nonscan_debug_force_path_runs_with_m1_and_zeroing(monkeypatch) -> None:
    pytest.importorskip("jax")
    _quiet_solve_env(monkeypatch)
    _install_scan_fakes(monkeypatch)
    monkeypatch.setenv("VMEC_JAX_SCAN_DEBUG_FORCE", "1")

    result = solve.solve_fixed_boundary_residual_iter(
        _state(),
        _static(),
        indata=_FakeInData(lmove_axis=False),
        signgs=1,
        max_iter=1,
        step_size=0.1,
        vmec2000_control=False,
        strict_update=False,
        backtracking=True,
        use_scan=False,
        auto_flip_force=False,
        limit_dt_from_force=False,
        limit_update_rms=False,
        use_restart_triggers=False,
        use_direct_fallback=False,
        reference_mode=False,
        jit_forces=False,
        verbose=False,
        verbose_vmec2000_table=False,
    )

    assert result.w_history.shape == (1,)
    assert result.diagnostics["restart_path_history"][0] == "non_strict"
