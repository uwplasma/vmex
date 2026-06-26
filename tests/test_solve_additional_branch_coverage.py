from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.solve as solve
import vmec_jax.solvers.fixed_boundary.jit_cache as jit_cache_helpers
from vmec_jax._compat import has_jax, jnp
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.kernels.tomnsp import TomnspsRZL


def _static(*, ns: int = 2, mpol: int = 2, ntor: int = 1, lthreed: bool = True, lasym: bool = False):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            ns=ns,
            mpol=mpol,
            ntor=ntor,
            ntheta=4,
            lthreed=lthreed,
            lasym=lasym,
        )
    )


def _gd_static():
    return SimpleNamespace(
        cfg=SimpleNamespace(nfp=1),
        modes=SimpleNamespace(m=np.asarray([0, 1]), n=np.asarray([0, 0])),
        s=np.asarray([0.0, 0.5, 1.0]),
        grid=SimpleNamespace(theta=np.asarray([0.0, np.pi]), zeta=np.asarray([0.0])),
    )


def _gd_state(*, interior: float) -> VMECState:
    layout = StateLayout(ns=3, K=2, lasym=False)
    zeros = np.zeros((3, 2), dtype=float)
    rcos = zeros.copy()
    rcos[:, 0] = 1.0
    rcos[1, 1] = float(interior)
    rcos[-1, 1] = 0.25
    return VMECState(
        layout=layout,
        Rcos=rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )


def _install_quadratic_geometry(monkeypatch):
    def fake_eval_geom(state, _static):
        rsum = jnp.sum(jnp.asarray(state.Rcos), axis=1)
        return SimpleNamespace(sqrtg=rsum[:, None, None] ** 2 + 1.0)

    monkeypatch.setattr(solve, "eval_geom", fake_eval_geom)
    monkeypatch.setattr(solve, "bsup_from_geom", lambda _g, **_kwargs: (0.0, 0.0))
    monkeypatch.setattr(solve, "b2_from_bsup", lambda g, _bsupu, _bsupv: jnp.ones_like(g.sqrtg))


@pytest.mark.skipif(not has_jax(), reason="fixed-boundary GD requires JAX")
def test_fixed_boundary_gd_stops_on_explicit_gradient_tolerance(monkeypatch, capsys):
    _install_quadratic_geometry(monkeypatch)

    result = solve.solve_fixed_boundary_gd(
        _gd_state(interior=1.5),
        _gd_static(),
        phipf=np.ones(3),
        chipf=np.zeros(3),
        signgs=1,
        lamscale=1.0,
        pressure=np.zeros(3),
        max_iter=2,
        grad_tol=1.0e9,
        verbose=True,
    )

    assert result.n_iter == 0
    assert result.grad_rms_history.shape == (1,)
    assert result.step_history.shape == (0,)
    assert "[solve_fixed_boundary_gd] iter=000" in capsys.readouterr().out


@pytest.mark.skipif(not has_jax(), reason="fixed-boundary GD requires JAX")
def test_fixed_boundary_gd_backtracks_then_reports_line_search_failure(monkeypatch, capsys):
    _install_quadratic_geometry(monkeypatch)
    state0 = _gd_state(interior=2.0)

    result = solve.solve_fixed_boundary_gd(
        state0,
        _gd_static(),
        phipf=np.ones(3),
        chipf=np.zeros(3),
        signgs=1,
        lamscale=1.0,
        pressure=np.zeros(3),
        max_iter=1,
        step_size=1_000.0,
        grad_tol=0.0,
        max_backtracks=1,
        bt_factor=0.5,
        verbose=True,
    )

    assert result.n_iter == 0
    np.testing.assert_allclose(np.asarray(result.state.Rcos), np.asarray(state0.Rcos))
    np.testing.assert_allclose(result.step_history, [500.0])
    assert "line search failed to improve objective" in capsys.readouterr().out


@pytest.mark.skipif(not has_jax(), reason="fixed-boundary L-BFGS requires JAX")
def test_fixed_boundary_lbfgs_backtracks_then_reports_line_search_failure(monkeypatch, capsys):
    _install_quadratic_geometry(monkeypatch)
    state0 = _gd_state(interior=2.0)

    result = solve.solve_fixed_boundary_lbfgs(
        state0,
        _gd_static(),
        phipf=np.ones(3),
        chipf=np.zeros(3),
        signgs=1,
        lamscale=1.0,
        pressure=np.zeros(3),
        max_iter=1,
        step_size=1_000.0,
        grad_tol=0.0,
        max_backtracks=1,
        bt_factor=0.5,
        verbose=True,
    )

    assert result.n_iter == 0
    np.testing.assert_allclose(np.asarray(result.state.Rcos), np.asarray(state0.Rcos))
    np.testing.assert_allclose(result.step_history, [500.0])
    assert "line search failed; stopping" in capsys.readouterr().out


def test_top_level_force_block_and_cache_helpers_cover_unusual_branches(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_TEST_CACHE_LIMIT", "not-an-int")
    assert solve._jit_cache_limit("VMEC_JAX_TEST_CACHE_LIMIT", 3) == 3

    cache = {}
    assert solve._jit_cache_get(cache, ("missing",)) is None
    ordered = solve.OrderedDict([(("a",), 1), (("b",), 2)])
    assert solve._jit_cache_get(ordered, ("a",)) == 1
    assert list(ordered) == [("b",), ("a",)]

    monkeypatch.setenv("VMEC_JAX_TEST_CACHE_LIMIT", "1")
    solve._jit_cache_put(ordered, ("c",), 3, env_name="VMEC_JAX_TEST_CACHE_LIMIT", default=3)
    assert list(ordered) == [("c",)]
    monkeypatch.setenv("VMEC_JAX_TEST_CACHE_LIMIT", "0")
    assert solve._jit_cache_put(ordered, ("d",), 4, env_name="VMEC_JAX_TEST_CACHE_LIMIT", default=3) == 4
    assert ("d",) not in ordered

    base = np.arange(8.0).reshape(2, 2, 2) + 1.0
    blocks = solve._ForceBlocks(
        frcc=base,
        frss=None,
        fzsc=base + 1.0,
        fzcs=None,
        flsc=base + 2.0,
        flcs=None,
        frsc=None,
        frcs=None,
        fzcc=None,
        fzss=None,
        flcc=None,
        flss=None,
    )
    zero = np.zeros_like(base)
    weighted = solve._mode_weight_force_blocks_np(blocks, w_mode_mn=np.asarray([[2.0, 3.0], [4.0, 5.0]]), zeros_coeff=zero)
    np.testing.assert_allclose(weighted.frcc, base * np.asarray([[[2.0, 3.0], [4.0, 5.0]]]))
    assert weighted.frss is zero

    frzl_pre = SimpleNamespace(flsc=np.ones((2, 1, 1)), flcs=None, flcc=None, flss=None)
    assert float(np.asarray(solve._lambda_preconditioned_full_norm(frzl_pre, use_jax=False))) == pytest.approx(1.0)
    frzl_pre_full = SimpleNamespace(
        flsc=np.ones((2, 1, 1)),
        flcs=np.ones((2, 1, 1)) * 2.0,
        flcc=np.ones((2, 1, 1)) * 3.0,
        flss=np.ones((2, 1, 1)) * 4.0,
    )
    assert float(np.asarray(solve._lambda_preconditioned_full_norm(frzl_pre_full, use_jax=True))) == pytest.approx(30.0)

    finite_dt = solve._safe_dt_from_force_blocks(dt_nominal=1.0, max_coeff_delta_rms=1.0e-4, blocks=blocks)
    assert finite_dt < 1.0
    zero_blocks = blocks._replace(frcc=np.zeros_like(base), fzsc=np.zeros_like(base), flsc=np.zeros_like(base))
    assert solve._safe_dt_from_force_blocks(dt_nominal=0.25, max_coeff_delta_rms=1.0e-4, blocks=zero_blocks) == pytest.approx(0.25)


def test_scan_cache_miss_category_recording_sanitizes_and_falls_back(monkeypatch):
    """Cover extracted scan-cache miss diagnostics without running a scan."""

    stats: dict[str, int | float] = {"scan_runner_cache_miss_category_mode_count": 2}

    def fake_counts(_requested_key, _existing_keys):
        return {"Mode": 3, "shape/name": 2, "": 1}

    monkeypatch.setattr(jit_cache_helpers, "scan_cache_miss_category_counts", fake_counts)
    jit_cache_helpers.record_scan_runner_cache_miss_categories(
        stats,
        requested_key=("new",),
        existing_keys=(("old",),),
    )
    assert stats["scan_runner_cache_miss_category_mode_count"] == 5
    assert stats["scan_runner_cache_miss_category_shape_name_count"] == 2
    assert stats["scan_runner_cache_miss_category_unknown_count"] == 1

    def broken_counts(_requested_key, _existing_keys):
        raise RuntimeError("synthetic category failure")

    monkeypatch.setattr(jit_cache_helpers, "scan_cache_miss_category_counts", broken_counts)
    jit_cache_helpers.record_scan_runner_cache_miss_categories(
        stats,
        requested_key=("new",),
        existing_keys=(),
    )
    assert stats["scan_runner_cache_miss_category_unknown_count"] == 2


@pytest.mark.skipif(not has_jax(), reason="fused preconditioner-output scaling requires JAX")
def test_preconditioner_output_scaling_jit_covers_missing_optional_blocks(monkeypatch):
    solve._PRECOND_OUTPUT_SCALE_JIT_CACHE.clear()

    base = np.arange(8.0).reshape(2, 2, 2) + 1.0
    frzl = TomnspsRZL(
        frcc=jnp.asarray(base),
        frss=None,
        fzsc=jnp.asarray(base + 1.0),
        fzcs=None,
        flsc=jnp.asarray(base + 2.0),
        flcs=None,
        frsc=None,
        frcs=None,
        fzcc=None,
        fzss=None,
        flcc=None,
        flss=None,
    )
    lam_prec = jnp.ones_like(frzl.flsc) * 0.5
    weights = jnp.asarray([[2.0, 3.0], [4.0, 5.0]])

    scaler = solve._preconditioner_output_scaling_jit(apply_lambda_update_scale=False)
    assert scaler is solve._preconditioner_output_scaling_jit(apply_lambda_update_scale=False)
    raw, scaled = scaler(frzl, lam_prec, weights, 0.25)

    weight_np = np.asarray(weights)[None, :, :]
    np.testing.assert_allclose(np.asarray(raw[4]), (base + 2.0) * 0.5)
    np.testing.assert_allclose(np.asarray(raw[6]), np.zeros_like(base))
    np.testing.assert_allclose(np.asarray(scaled[0]), base * weight_np)
    np.testing.assert_allclose(np.asarray(scaled[4]), (base + 2.0) * 0.5 * weight_np)


def test_free_boundary_vmec_cadence_handles_invalid_fsq_activation_and_full_updates(monkeypatch):
    ivac, ivacskip, nvacskip = solve._free_boundary_iter_controls_vmec(
        iter2=5,
        iter1=1,
        ivac=-1,
        nvacskip=3,
        nvskip0=2,
        fsq_rz_prev=np.nan,
    )
    assert (ivac, ivacskip, nvacskip) == (-1, 0, 3)

    monkeypatch.setenv("VMEC_JAX_FREEB_ACTIVATE_FSQ", "0.25")
    ivac, ivacskip, nvacskip = solve._free_boundary_iter_controls_vmec(
        iter2=2,
        iter1=1,
        ivac=2,
        nvacskip=4,
        nvskip0=2,
        fsq_rz_prev=0.1,
    )
    assert (ivac, ivacskip, nvacskip) == (3, 1, 4)

    ivac, ivacskip, nvacskip = solve._free_boundary_iter_controls_vmec(
        iter2=1,
        iter1=1,
        ivac=1,
        nvacskip=4,
        nvskip0=3,
        fsq_rz_prev=-1.0,
    )
    assert (ivac, ivacskip, nvacskip) == (1, 0, 3)


def test_coefficient_and_axis_helpers_cover_edge_branches():
    assert solve._mode00_index(SimpleNamespace(m=np.asarray([1, 2]), n=np.asarray([0, 1]))) is None

    lcos = np.arange(6.0).reshape(2, 3)
    lsin = lcos + 10.0
    assert solve._enforce_lambda_gauge(lcos, lsin, idx00=None) == (lcos, lsin)
    np.testing.assert_allclose(np.asarray(solve._axis_m0_mask(SimpleNamespace(m_is_m0=[True, False]), dtype=float)), [1.0, 0.0])

    np.testing.assert_allclose(np.asarray(solve._zero_coeff_column(lcos, idx=-1)), lcos)
    np.testing.assert_allclose(np.asarray(solve._zero_coeff_column(np.ones((2, 1)), idx=0)), np.zeros((2, 1)))
    last_zeroed = np.asarray(solve._zero_coeff_column(lcos, idx=2))
    np.testing.assert_allclose(last_zeroed[:, 2], 0.0)
    middle_zeroed = np.asarray(solve._zero_coeff_column(lcos, idx=1))
    np.testing.assert_allclose(middle_zeroed[:, 1], 0.0)

    arr3 = np.arange(12.0).reshape(2, 3, 2)
    repl = np.full((2, 2), -1.0)
    assert solve._replace_mode_slice(None, mode_idx=0, replacement=repl) is None
    np.testing.assert_allclose(np.asarray(solve._replace_mode_slice(arr3, mode_idx=-1, replacement=repl)), arr3)
    np.testing.assert_allclose(np.asarray(solve._replace_mode_slice(arr3[:, :1, :], mode_idx=0, replacement=repl))[:, 0, :], repl)
    np.testing.assert_allclose(np.asarray(solve._replace_mode_slice(arr3, mode_idx=2, replacement=repl))[:, 2, :], repl)
    np.testing.assert_allclose(np.asarray(solve._replace_mode_slice(arr3, mode_idx=1, replacement=repl))[:, 1, :], repl)

    assert solve._scale_mode_slice(None, mode_idx=0, scale=np.ones(2)) is None
    np.testing.assert_allclose(np.asarray(solve._scale_mode_slice(arr3, mode_idx=9, scale=np.ones(2))), arr3)
    scaled = np.asarray(solve._scale_mode_slice(arr3, mode_idx=1, scale=np.asarray([2.0, 3.0])))
    np.testing.assert_allclose(scaled[:, 1, :], arr3[:, 1, :] * np.asarray([[2.0], [3.0]]))

    np.testing.assert_allclose(solve._zero_coeff_column_np(lcos, idx=1)[:, 1], 0.0)
    assert solve._replace_mode_slice_np(None, mode_idx=0, replacement=repl) is None
    np.testing.assert_allclose(solve._replace_mode_slice_np(arr3, mode_idx=1, replacement=repl)[:, 1, :], repl)
    assert solve._scale_mode_slice_np(None, mode_idx=0, scale=np.ones(2)) is None
    np.testing.assert_allclose(
        solve._scale_mode_slice_np(arr3, mode_idx=1, scale=np.asarray([2.0, 3.0]))[:, 1, :],
        arr3[:, 1, :] * np.asarray([[2.0], [3.0]]),
    )

    empty = np.empty((0, 2))
    assert np.asarray(solve._enforce_field_rows(empty)).shape == (0, 2)
    one = np.asarray([[1.0, 2.0]])
    np.testing.assert_allclose(np.asarray(solve._enforce_field_rows(one, edge_row=[5.0, 6.0])), [[5.0, 6.0]])
    np.testing.assert_allclose(np.asarray(solve._enforce_field_rows(one, edge_row=[5.0, 6.0], zero_axis=True)), 0.0)
    np.testing.assert_allclose(
        np.asarray(solve._enforce_field_rows(one, edge_row=[5.0, 6.0], axis_mask=[1.0, 0.0])),
        [[5.0, 0.0]],
    )
    np.testing.assert_allclose(np.asarray(solve._enforce_field_rows(lcos, edge_row=[7.0, 8.0, 9.0]))[-1], [7.0, 8.0, 9.0])


def test_apply_vmec_lambda_axis_rules_to_state_disabled_host_and_device_paths():
    state = _gd_state(interior=1.0)
    state = VMECState(
        layout=state.layout,
        Rcos=state.Rcos,
        Rsin=state.Rsin,
        Zcos=state.Zcos,
        Zsin=state.Zsin,
        Lcos=np.ones_like(state.Lcos),
        Lsin=np.ones_like(state.Lsin) * 2.0,
    )

    disabled = solve._apply_vmec_lambda_axis_rules_to_state(
        state,
        enforce_vmec_lambda_axis=False,
        host_update_assembly=True,
        idx00=0,
    )
    assert disabled is state

    host = solve._apply_vmec_lambda_axis_rules_to_state(
        state,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=True,
        idx00=0,
    )
    np.testing.assert_allclose(np.asarray(host.Lcos)[:, 0], 0.0)
    np.testing.assert_allclose(np.asarray(host.Lsin)[:, 0], 0.0)

    device = solve._apply_vmec_lambda_axis_rules_to_state(
        state,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=False,
        idx00=1,
    )
    np.testing.assert_allclose(np.asarray(device.Lcos)[:, 1], 0.0)
    np.testing.assert_allclose(np.asarray(device.Lsin)[:, 1], 0.0)


def test_lam_prec_dump_iter_fallback_filter_and_asymmetric_replication(monkeypatch, tmp_path):
    monkeypatch.setenv("VMEC_JAX_DUMP_LAM", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "3")
    monkeypatch.setenv("VMEC_JAX_DUMP_LAM_DIR", str(tmp_path))
    monkeypatch.delenv("VMEC_JAX_DUMP_LAM_ITER", raising=False)

    static = _static(lthreed=True, lasym=True)
    lam_prec = np.arange(8.0).reshape(2, 2, 2)
    faclam = lam_prec + 100.0

    solve._maybe_dump_lam_prec(lam_prec=lam_prec, faclam=faclam, static=static, iter_idx=2)
    assert not (tmp_path / "lam_prec_ns2_iter2.npz").exists()

    solve._maybe_dump_lam_prec(lam_prec=lam_prec, faclam=faclam, static=static, iter_idx=3)

    data = np.load(tmp_path / "lam_prec_ns2_iter3.npz")
    expected = np.transpose(lam_prec, (0, 2, 1))
    assert data["pfaclam"].shape == (2, 2, 2, 4)
    np.testing.assert_allclose(data["pfaclam"][..., 0], expected)
    expected_extra_plane = expected.copy()
    expected_extra_plane[:, 0, 0] = 0.0
    np.testing.assert_allclose(data["pfaclam"][..., 1], expected_extra_plane)
    np.testing.assert_allclose(data["pfaclam"][:, 0, 0, 1:], 0.0)
    np.testing.assert_allclose(data["faclam"][..., 0], np.transpose(faclam, (0, 2, 1)))
    assert bool(data["lasym"])


def test_lam_prec_dump_keeps_mismatched_faclam_and_rejects_bad_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("VMEC_JAX_DUMP_LAM", "yes")
    monkeypatch.setenv("VMEC_JAX_DUMP_LAM_DIR", str(tmp_path))
    monkeypatch.delenv("VMEC_JAX_DUMP_ITER", raising=False)
    monkeypatch.delenv("VMEC_JAX_DUMP_LAM_ITER", raising=False)

    static = _static(ntor=0, lthreed=False, lasym=False)
    lam_prec = np.arange(4.0).reshape(2, 2, 1)
    faclam = np.asarray([9.0, 10.0])

    solve._maybe_dump_lam_prec(lam_prec=lam_prec, faclam=faclam, static=static, iter_idx=4)

    data = np.load(tmp_path / "lam_prec_ns2_iter4.npz")
    assert data["pfaclam"].shape == (2, 1, 2, 1)
    np.testing.assert_allclose(data["faclam"], faclam)
    assert not bool(data["lthreed"])

    with pytest.raises(ValueError, match="lam_prec expected 3D"):
        solve._maybe_dump_lam_prec(lam_prec=np.ones((2, 2)), faclam=None, static=static, iter_idx=5)


def test_precond_matrix_and_lambda_fsql1_dumps_use_shared_iter_filter(monkeypatch, tmp_path):
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "7")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_PRECOND_MATS", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_LAM", "1")
    monkeypatch.delenv("VMEC_JAX_DUMP_PRECOND_MATS_ITER", raising=False)
    monkeypatch.delenv("VMEC_JAX_DUMP_PRECOND_MATS_DIR", raising=False)
    monkeypatch.delenv("VMEC_JAX_DUMP_LAM_ITER", raising=False)
    monkeypatch.delenv("VMEC_JAX_DUMP_LAM_DIR", raising=False)

    static = _static(ns=3, mpol=2, ntor=0)
    mats = {"ar": np.asarray([1.0, 2.0]), "ignored": np.asarray([99.0])}

    solve._maybe_dump_precond_mats(mats=mats, static=static, iter_idx=6, jmax=2, used_cache=True)
    assert not (tmp_path / "precond_mats_ns3_iter6.npz").exists()

    solve._maybe_dump_precond_mats(mats=mats, static=static, iter_idx=7, jmax=2, used_cache=True)
    data = np.load(tmp_path / "precond_mats_ns3_iter7.npz")
    np.testing.assert_allclose(data["ar"], [1.0, 2.0])
    assert "ignored" not in data.files
    assert bool(data["used_cache"])
    assert int(data["jmax"]) == 2

    solve._maybe_dump_lam_fsql1(fsql1_pre=1.25, fsql1_post=np.asarray(0.5), static=static, iter_idx=7)
    text = (tmp_path / "lam_fsql1_ns3_iter7.dat").read_text(encoding="utf-8")
    assert "lambda fsql1 dump" in text
    assert "1.2500000000000000e+00" in text
    assert "5.0000000000000000e-01" in text


def test_hlo_kernel_dump_writes_once_and_respects_cache(monkeypatch, tmp_path):
    jnp = pytest.importorskip("jax.numpy")

    solve._HLO_DUMPED_KEYS.clear()
    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_DIR", str(tmp_path))

    static = _static(ns=2, mpol=2, ntor=0)
    wout_like = SimpleNamespace(mpol=2, ntor=0, nfp=1, lasym=False)

    def tiny_kernel(x):
        return x + 1.0

    solve._maybe_dump_hlo_kernel(
        label="branchcov",
        fn=tiny_kernel,
        args=(jnp.asarray([1.0]),),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=True,
    )

    path = tmp_path / "hlo_branchcov_ns2_mpol2_ntor0.txt"
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip()

    path.write_text("sentinel", encoding="utf-8")
    solve._maybe_dump_hlo_kernel(
        label="branchcov",
        fn=tiny_kernel,
        args=(jnp.asarray([2.0]),),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=True,
    )
    assert path.read_text(encoding="utf-8") == "sentinel"


def test_hlo_kernel_dump_uses_legacy_and_string_fallbacks(monkeypatch, tmp_path):
    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_VERBOSE", "1")
    monkeypatch.setattr(solve, "has_jax", lambda: True)
    solve._HLO_DUMPED_KEYS.clear()

    static = _static(ns=2, mpol=2, ntor=0)
    wout_like = SimpleNamespace(mpol=2, ntor=0, nfp=1, lasym=False)

    class _StringLowered:
        def compiler_ir(self, *, dialect):
            assert dialect == "hlo"
            return SimpleNamespace()

    class _StringJit:
        def lower(self, *args, **kwargs):
            assert args == (1,)
            assert kwargs == {}
            return _StringLowered()

    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(jit=lambda _fn: _StringJit()))
    solve._maybe_dump_hlo_kernel(
        label="stringfallback",
        fn=lambda x: x,
        args=(1,),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=True,
    )
    assert "namespace" in (tmp_path / "hlo_stringfallback_ns2_mpol2_ntor0.txt").read_text(encoding="utf-8")

    class _JitLowerRaises:
        def lower(self, *args, **kwargs):
            raise RuntimeError("lower failed")

    class _LegacyHlo:
        def as_hlo_text(self):
            return "legacy hlo text"

    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(jit=lambda _fn: _JitLowerRaises(), xla_computation=lambda _fn: lambda *args, **kwargs: _LegacyHlo()),
    )
    solve._maybe_dump_hlo_kernel(
        label="legacyfallback",
        fn=lambda x: x,
        args=(2,),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=True,
    )
    assert (tmp_path / "hlo_legacyfallback_ns2_mpol2_ntor0.txt").read_text(encoding="utf-8") == "legacy hlo text"

    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(
            jit=lambda _fn: _JitLowerRaises(),
            xla_computation=lambda _fn: (_ for _ in ()).throw(RuntimeError("legacy failed")),
        ),
    )
    solve._maybe_dump_hlo_kernel(
        label="errorfallback",
        fn=lambda x: x,
        args=(3,),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=True,
    )
    assert "xla_computation failed" in (
        tmp_path / "hlo_errorfallback_error_ns2_mpol2_ntor0.txt"
    ).read_text(encoding="utf-8")


def test_hlo_kernel_dump_covers_as_text_and_write_error_branches(monkeypatch, tmp_path):
    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_VERBOSE", "1")
    monkeypatch.setattr(solve, "has_jax", lambda: True)
    solve._HLO_DUMPED_KEYS.clear()

    static = _static(ns=2, mpol=2, ntor=0)
    wout_like = SimpleNamespace(mpol=2, ntor=0, nfp=1, lasym=False)

    class _AsText:
        def as_text(self):
            return "as-text hlo"

    class _AsTextLowered:
        def compiler_ir(self, *, dialect):
            assert dialect == "hlo"
            return _AsText()

    class _AsTextJit:
        def lower(self, *args, **kwargs):
            return _AsTextLowered()

    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(jit=lambda _fn: _AsTextJit()))
    solve._maybe_dump_hlo_kernel(
        label="astext",
        fn=lambda x: x,
        args=(1,),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=True,
    )
    assert (tmp_path / "hlo_astext_ns2_mpol2_ntor0.txt").read_text(encoding="utf-8") == "as-text hlo"

    class _JitLowerRaises:
        def lower(self, *args, **kwargs):
            raise RuntimeError("lower failed")

    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(jit=lambda _fn: _JitLowerRaises(), xla_computation=lambda _fn: lambda *args, **kwargs: object()),
    )
    solve._maybe_dump_hlo_kernel(
        label="legacystr",
        fn=lambda x: x,
        args=(2,),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=True,
    )
    assert "object" in (tmp_path / "hlo_legacystr_ns2_mpol2_ntor0.txt").read_text(encoding="utf-8")

    def raise_write_text(self, text, *args, **kwargs):
        raise OSError(f"blocked write for {self.name}: {text[:8]}")

    monkeypatch.setattr(solve.Path, "write_text", raise_write_text)
    solve._maybe_dump_hlo_kernel(
        label="writeerror",
        fn=lambda x: x,
        args=(3,),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=True,
    )

    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(
            jit=lambda _fn: _JitLowerRaises(),
            xla_computation=lambda _fn: (_ for _ in ()).throw(RuntimeError("legacy failed")),
        ),
    )
    solve._maybe_dump_hlo_kernel(
        label="errorwrite",
        fn=lambda x: x,
        args=(4,),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=True,
    )
