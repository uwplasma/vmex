from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.state import StateLayout, VMECState


def _tiny_state(*, jnp=None) -> VMECState:
    if jnp is None:
        xp = np
    else:
        xp = jnp
    layout = StateLayout(ns=2, K=1, lasym=False)
    zeros = xp.zeros((2, 1), dtype=float)
    return VMECState(
        layout=layout,
        Rcos=xp.asarray([[1.0], [2.0]], dtype=float),
        Rsin=zeros,
        Zcos=zeros,
        Zsin=xp.asarray([[0.0], [0.5]], dtype=float),
        Lcos=zeros,
        Lsin=zeros,
    )


def _tiny_static():
    return SimpleNamespace(
        cfg=SimpleNamespace(
            ns=2,
            nfp=1,
            mpol=1,
            ntor=0,
            ntheta=2,
            nzeta=1,
            lasym=False,
            lconm1=True,
        ),
        modes=SimpleNamespace(m=np.asarray([0]), n=np.asarray([0]), K=1),
        grid=SimpleNamespace(theta=np.asarray([0.0, np.pi]), zeta=np.asarray([0.0])),
        s=np.asarray([0.0, 1.0]),
        trig_vmec=SimpleNamespace(ntheta3=1, cosmui3=np.ones((1, 1)), mscale=np.ones((1,))),
    )


class _Indata:
    def __init__(self, scalars=None):
        self.scalars = dict(scalars or {})

    def get(self, name, default=None):
        return self.scalars.get(name.upper(), default)

    def get_bool(self, name, default=False):
        value = self.get(name, default)
        return bool(value[0] if isinstance(value, list) and value else value)

    def get_int(self, name, default=0):
        value = self.get(name, default)
        if isinstance(value, list):
            value = value[0] if value else default
        try:
            return int(value)
        except Exception:
            return int(default)

    def get_float(self, name, default=0.0):
        value = self.get(name, default)
        if isinstance(value, list):
            value = value[0] if value else default
        try:
            return float(value)
        except Exception:
            return float(default)


def test_lineax_bicgstab_optional_paths(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.implicit as implicit
    from vmec_jax._compat import jnp

    monkeypatch.setattr(implicit, "lx", None)
    value, success, stats = implicit._lineax_bicgstab_solve(lambda x: x, jnp.ones(2), tol=1e-8, max_iter=3)
    assert value is None
    assert success is False
    assert stats == {}

    class _FakeSolution:
        value = jnp.asarray([2.0, 3.0])
        stats = {"num_steps": jnp.asarray(2)}

    class _FakeLineax:
        class FunctionLinearOperator:
            def __init__(self, matvec, input_structure):
                self.matvec = matvec
                self.input_structure = input_structure

        class BiCGStab:
            def __init__(self, *, rtol, atol, max_steps):
                self.rtol = rtol
                self.atol = atol
                self.max_steps = max_steps

        @staticmethod
        def linear_solve(operator, b, *, solver, options, throw):
            assert options["y0"].shape == b.shape
            assert solver.max_steps == 4
            assert throw is False
            operator.matvec(b)
            return _FakeSolution()

    monkeypatch.setattr(implicit, "lx", _FakeLineax)
    value, success, stats = implicit._lineax_bicgstab_solve(
        lambda x: 2.0 * x,
        jnp.ones(2),
        x0=jnp.zeros(2),
        tol=1e-6,
        max_iter=4,
    )
    np.testing.assert_allclose(np.asarray(value), [2.0, 3.0])
    assert success is True
    assert "num_steps" in stats


def test_stellsym_index_helpers_cover_unmasked_and_no_00_branches():
    pytest.importorskip("jax")

    from vmec_jax.implicit import (
        _stellsym_feasible_indices_np,
        _stellsym_lambda_mn_indices,
        _stellsym_reduced_z_indices,
        _stellsym_structural_active_keep_indices,
    )

    static = SimpleNamespace(
        cfg=SimpleNamespace(ns=3),
        modes=SimpleNamespace(m=np.asarray([0, 1, 2]), n=np.asarray([0, 0, 1])),
    )
    rz_idx, lam_idx, ns, k = _stellsym_feasible_indices_np(static, idx00=None, mask_lambda_axis=False)

    assert ns == 3
    assert k == 3
    np.testing.assert_array_equal(rz_idx, [0, 3, 4, 5])
    np.testing.assert_array_equal(lam_idx, np.arange(9))

    z_idx = _stellsym_reduced_z_indices(rz_idx=rz_idx, K=k, idx00=None)
    np.testing.assert_array_equal(np.asarray(z_idx), rz_idx)

    lam_sc_idx, lam_cs_idx, lam_maps = _stellsym_lambda_mn_indices(
        static,
        idx00=None,
        mask_lambda_axis=False,
    )
    assert np.asarray(lam_sc_idx).size > 0
    assert np.asarray(lam_cs_idx).size > 0
    assert lam_maps.mpol == 3

    keep = _stellsym_structural_active_keep_indices(rz_idx=rz_idx, lam_idx=lam_idx, K=k, idx00=None)
    expected_keep = np.arange(2 * len(rz_idx) + len(lam_idx))
    np.testing.assert_array_equal(np.asarray(keep), expected_keep)


def test_fixed_boundary_implicit_validates_edges_and_zeroes_unconverged_vjp(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)

    state0 = _tiny_state(jnp=jnp)
    static = _tiny_static()

    with pytest.raises(ValueError, match="solver must be 'gd' or 'lbfgs'"):
        implicit.solve_fixed_boundary_state_implicit(
            state0,
            static,
            phipf=jnp.ones(2),
            chipf=jnp.ones(2),
            signgs=1,
            lamscale=jnp.ones(2),
            pressure=jnp.zeros(2),
            solver="newton",
        )

    with pytest.raises(ValueError, match="must be provided together"):
        implicit.solve_fixed_boundary_state_implicit(
            state0,
            static,
            phipf=jnp.ones(2),
            chipf=jnp.ones(2),
            signgs=1,
            lamscale=jnp.ones(2),
            pressure=jnp.zeros(2),
            edge_Rcos=jnp.ones(1),
        )

    def fake_gd(state, *_args, **_kwargs):
        return SimpleNamespace(state=state, grad_rms_history=[1.0], diagnostics={"grad_tol": 1e-6})

    monkeypatch.setattr(implicit, "solve_fixed_boundary_gd", fake_gd)

    def objective(scale):
        st = implicit.solve_fixed_boundary_state_implicit(
            state0,
            static,
            phipf=scale * jnp.asarray([1.0, 2.0]),
            chipf=jnp.asarray([0.0, 0.1]),
            signgs=1,
            lamscale=jnp.ones(2),
            pressure=jnp.zeros(2),
            solver="gd",
            max_iter=1,
            implicit_converge_tol=1e-9,
        )
        return jnp.sum(jnp.asarray(st.Rcos)) + jnp.sum(jnp.asarray(st.Zsin))

    assert float(jax.grad(objective)(2.0)) == pytest.approx(0.0, abs=0.0)


def test_vmec_residual_implicit_forward_uses_host_callback_seam(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.boundary as boundary_module
    import vmec_jax.implicit as implicit
    import vmec_jax.init_guess as init_guess_module
    from vmec_jax._compat import jnp
    from vmec_jax.state import pack_state

    state0 = _tiny_state(jnp=jnp)
    static = _tiny_static()
    indata = _Indata({"NCURR": 0, "LRFP": False, "GAMMA": 0.0, "TCON0": 1.0})
    boundary = boundary_module.BoundaryCoeffs(
        R_cos=np.asarray([2.0]),
        R_sin=np.asarray([0.0]),
        Z_cos=np.asarray([0.0]),
        Z_sin=np.asarray([0.5]),
    )

    monkeypatch.setattr(
        implicit,
        "flux_profiles_from_indata",
        lambda *_args, **_kwargs: SimpleNamespace(
            phipf=jnp.asarray([0.0, 1.0]),
            phips=jnp.asarray([0.0, 1.0]),
            chipf=jnp.asarray([0.0, 0.2]),
        ),
    )
    monkeypatch.setattr(implicit, "_mass_half_mesh_from_indata", lambda **_kwargs: jnp.asarray([0.0, 1.0]))
    monkeypatch.setattr(implicit, "_pressure_half_mesh_from_indata", lambda **_kwargs: jnp.asarray([0.0, 0.0]))
    monkeypatch.setattr(implicit, "_icurv_full_mesh_from_indata", lambda **_kwargs: jnp.asarray([0.0, 0.0]))
    monkeypatch.setattr(
        implicit,
        "_vmec_force_flux_profiles",
        lambda **kwargs: (kwargs["phipf"], kwargs["chipf"], kwargs["chipf"]),
    )
    monkeypatch.setattr(boundary_module, "boundary_from_indata", lambda *_args, **_kwargs: boundary)
    monkeypatch.setattr(init_guess_module, "initial_guess_from_boundary", lambda *_args, **_kwargs: state0)
    monkeypatch.setattr(implicit, "eval_geom", lambda *_args, **_kwargs: SimpleNamespace(sqrtg=np.ones((2, 1, 1))))
    monkeypatch.setattr(implicit, "signgs_from_sqrtg", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        implicit,
        "solve_fixed_boundary_residual_iter",
        lambda state, *_args, **_kwargs: SimpleNamespace(
            state=state,
            n_iter=1,
            fsqz2_history=np.asarray([1e-8]),
        ),
    )

    out = implicit.solve_fixed_boundary_state_implicit_vmec_residual(
        state0,
        static,
        indata=indata,
        signgs=1,
        max_iter=1,
        step_size=0.25,
    )
    np.testing.assert_allclose(np.asarray(pack_state(out)), np.asarray(pack_state(state0)))


def test_wout_weight_geometry_and_profile_edge_branches():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.wout import (
        _chipf_from_chips,
        _compute_aspectratio,
        _compute_eqfor_beta,
        _compute_eqfor_betaxis,
        _jxbforce_nyquist_limits,
        _pshalf_from_s,
        _safe_divide,
        _vmec_wint_from_trig,
        _vmec_wint_from_trig_jax,
    )

    trig = SimpleNamespace(
        ntheta3=2,
        cosmui3=np.asarray([[2.0], [4.0]]),
        mscale=np.asarray([2.0]),
        cosnv=np.ones((3, 1)),
        ntheta2=5,
    )
    np.testing.assert_allclose(_vmec_wint_from_trig(trig), [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    np.testing.assert_allclose(np.asarray(_vmec_wint_from_trig_jax(trig)), [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    with pytest.raises(ValueError, match="shape"):
        _vmec_wint_from_trig(SimpleNamespace(cosmui3=np.ones(2), mscale=np.ones(1), cosnv=np.ones((1, 1))))
    with pytest.raises(ValueError, match="non-empty"):
        _vmec_wint_from_trig(SimpleNamespace(cosmui3=np.ones((1, 1)), mscale=np.asarray([]), cosnv=np.ones((1, 1))))
    with pytest.raises(ValueError, match="non-empty"):
        _vmec_wint_from_trig_jax(SimpleNamespace(cosmui3=jnp.ones((1, 1)), mscale=jnp.asarray([]), cosnv=np.ones((1, 1))))

    np.testing.assert_allclose(_pshalf_from_s(np.asarray([4.0])), [2.0])
    np.testing.assert_allclose(_pshalf_from_s(np.asarray([0.0, 0.5, 1.0])), [0.5, 0.5, np.sqrt(0.75)])
    np.testing.assert_allclose(_safe_divide(np.asarray([2.0, 4.0]), np.asarray([0.0, 2.0])), [2.0, 2.0])
    assert _jxbforce_nyquist_limits(trig) == (4, 1)
    assert _jxbforce_nyquist_limits(SimpleNamespace(ntheta2=0, cosnv=np.asarray(1.0))) == (0, 0)

    chips = np.asarray([0.0, 2.0, 4.0, 8.0])
    np.testing.assert_allclose(_chipf_from_chips(chips), [1.0, 3.0, 6.0, 10.0])
    np.testing.assert_allclose(_chipf_from_chips(np.asarray([3.0])), [3.0])
    np.testing.assert_allclose(np.asarray(_chipf_from_chips(jnp.asarray([1.0, 3.0]))), [3.0, 4.0])

    zeros = np.zeros((2, 1, 1))
    assert _compute_eqfor_beta(
        pres=np.zeros(2),
        vp=np.ones(2),
        bsq=zeros,
        r12=zeros,
        bsupv=zeros,
        sqrtg=zeros,
        wint=np.ones((1, 1)),
        signgs=1,
    ) == (0.0, 0.0, 0.0, 0.0)
    assert _compute_eqfor_betaxis(
        pres=np.zeros(2),
        vp=np.ones(2),
        bsq=zeros,
        sqrtg=zeros,
        wint=np.ones((1, 1)),
        signgs=1,
    ) == 0.0

    beta = _compute_eqfor_beta(
        pres=np.asarray([0.0, 1.0, 2.0]),
        vp=np.asarray([1.0, 2.0, 2.0]),
        bsq=np.asarray([[[0.0]], [[4.0]], [[5.0]]]),
        r12=np.asarray([[[0.0]], [[2.0]], [[2.0]]]),
        bsupv=np.asarray([[[0.0]], [[0.5]], [[0.5]]]),
        sqrtg=np.ones((3, 1, 1)),
        wint=np.ones((1, 1)),
        signgs=1,
    )
    assert all(np.isfinite(beta))

    assert _compute_eqfor_betaxis(
        pres=np.asarray([0.0, 1.0, 2.0]),
        vp=np.asarray([1.0, 2.0, 2.0]),
        bsq=np.asarray([[[0.0]], [[4.0]], [[8.0]]]),
        sqrtg=np.ones((3, 1, 1)),
        wint=np.ones((1, 1)),
        signgs=1,
    ) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="shape"):
        _compute_aspectratio(R=np.ones((2, 2)), Zu=np.ones((2, 2, 1)), wint=np.ones((2, 1)))
    with pytest.raises(ValueError, match="wint shape"):
        _compute_aspectratio(R=np.ones((2, 2, 1)), Zu=np.ones((2, 2, 1)), wint=np.ones((1, 1)))
    assert _compute_aspectratio(R=np.ones((2, 2, 1)), Zu=np.zeros((2, 2, 1)), wint=np.ones((2, 1))) == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    aspect = _compute_aspectratio(
        R=np.asarray([[[1.0], [1.0]], [[2.0], [2.0]]]),
        Zu=np.asarray([[[0.0], [0.0]], [[1.0], [1.0]]]),
        wint=np.ones((2, 1)),
    )
    assert aspect[0] == pytest.approx(np.sqrt(8.0))
    assert aspect[1] == pytest.approx(1.0)


def test_wout_bsubv_ctor_and_current_helpers(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.profiles as profiles_module
    from vmec_jax._compat import jnp
    from vmec_jax.wout import (
        _apply_bsubv_equif_correction,
        _compute_ctor_from_buco,
        _icurv_full_mesh_from_indata,
    )

    trig = SimpleNamespace(ntheta3=2, cosmui3=np.ones((2, 1)), mscale=np.ones(1))
    bsubv = np.arange(6.0).reshape(3, 2, 1)
    bsubv_e = bsubv + 1.0
    corrected = _apply_bsubv_equif_correction(bsubv=bsubv, bsubv_e=bsubv_e, trig=trig)
    assert corrected.shape == bsubv.shape
    np.testing.assert_allclose(corrected[0], bsubv[0])

    small = _apply_bsubv_equif_correction(bsubv=bsubv[:2], bsubv_e=bsubv_e[:2], trig=trig)
    np.testing.assert_allclose(small, bsubv[:2])

    bad_trig = SimpleNamespace(ntheta3=1, cosmui3=np.ones((1, 1)), mscale=np.ones(1))
    with pytest.raises(ValueError, match="pwint shape mismatch"):
        _apply_bsubv_equif_correction(bsubv=bsubv, bsubv_e=bsubv_e, trig=bad_trig)

    assert _compute_ctor_from_buco(buco=np.asarray([1.0]), signgs=1, indata=_Indata()) == 0.0
    default_ctor = _compute_ctor_from_buco(buco=np.asarray([2.0, 4.0]), signgs=-1, indata=_Indata())
    assert default_ctor < 0.0
    exact_ctor = _compute_ctor_from_buco(
        buco=np.asarray([2.0, 4.0]),
        signgs=1,
        indata=_Indata({"LFREEB": True, "ICTRL_PREC2D": 1, "LHESS_EXACT": True}),
    )
    inexact_ctor = _compute_ctor_from_buco(
        buco=np.asarray([2.0, 4.0]),
        signgs=1,
        indata=_Indata({"LFREEB": True, "ICTRL_PREC2D": 2, "LHESS_EXACT": False}),
    )
    assert exact_ctor == pytest.approx(inexact_ctor)

    np.testing.assert_allclose(
        np.asarray(_icurv_full_mesh_from_indata(indata=_Indata({"NCURR": 0}), s_full=jnp.asarray([0.0, 1.0]), signgs=1)),
        [0.0, 0.0],
    )
    np.testing.assert_allclose(
        np.asarray(
            _icurv_full_mesh_from_indata(
                indata=_Indata({"NCURR": 1, "CURTOR": 0.0}),
                s_full=jnp.asarray([0.0, 1.0]),
                signgs=1,
            )
        ),
        [0.0, 0.0],
    )

    monkeypatch.setattr(
        profiles_module,
        "eval_profiles",
        lambda _indata, s: {"current": jnp.asarray(s, dtype=float) + 1.0},
    )
    cur = _icurv_full_mesh_from_indata(
        indata=_Indata({"NCURR": 1, "CURTOR": 2.0}),
        s_full=jnp.asarray([0.0, 0.5, 1.0]),
        signgs=-1,
    )
    assert np.asarray(cur)[0] == pytest.approx(0.0)
    assert np.all(np.asarray(cur)[1:] < 0.0)


def test_driver_small_policy_and_serialization_helpers(monkeypatch, tmp_path):
    import vmec_jax.driver as driver
    import vmec_jax.wout as wout_module

    monkeypatch.setattr(driver, "_default_backend_name", lambda: "gpu")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "1")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "99")
    assert driver._dynamic_scan_probe_settings(5) == (1, True, "gpu")

    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "off")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "bad")
    assert driver._dynamic_scan_probe_settings(5) == (2, False, "gpu")

    assert driver._normalize_solver_mode(solver_mode=None, performance_mode=True) == "default"
    assert driver._normalize_solver_mode(solver_mode=None, performance_mode=False) == "parity"
    assert driver._normalize_solver_mode(solver_mode="FAST", performance_mode=False) == "default"
    with pytest.raises(ValueError, match="Unknown solver_mode"):
        driver._normalize_solver_mode(solver_mode="mystery", performance_mode=False)

    assert driver._requested_final_ftol(indata=_Indata({"FTOL": 2e-4}), ftol_list_input=None) == pytest.approx(2e-4)
    assert driver._requested_final_ftol(indata=_Indata({"FTOL": 2e-4}), ftol_list_input=["1e-3", "-1"]) == pytest.approx(0.0)
    assert driver._as_float_list(object()) is None
    assert driver._as_list_like(np.asarray([1, 2])) == [1, 2]
    assert driver._as_list_like(3) == [3]

    captured = []

    def fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(path=kwargs["path"], fsqr=kwargs["fsqr"], fsqz=kwargs["fsqz"], fsql=kwargs["fsql"])

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: (9.0, 8.0, 7.0))

    run = driver.FixedBoundaryRun(
        cfg=SimpleNamespace(),
        indata=_Indata(),
        static=SimpleNamespace(),
        state=SimpleNamespace(),
        result=SimpleNamespace(
            diagnostics={"converged": True},
            fsqr2_history=np.asarray([1.0, 2.0]),
            fsqz2_history=np.asarray([3.0, 5.0]),
            fsql2_history=np.asarray([7.0, 11.0]),
        ),
        flux=SimpleNamespace(),
        profiles={},
        signgs=-1,
    )

    monkeypatch.setenv("VMEC_JAX_WOUT_FAST_BCOVAR", "keep")
    wout = driver.wout_from_fixed_boundary_run(run, include_fsq=True, path=tmp_path / "custom.nc", fast_bcovar=True)
    assert wout.fsqr == pytest.approx(2.0)
    assert wout.fsqz == pytest.approx(5.0)
    assert wout.fsql == pytest.approx(11.0)
    assert captured[-1]["fsqt"][0] == pytest.approx(4.0)
    assert captured[-1]["fsqt"][1] == pytest.approx(7.0)
    assert captured[-1]["converged"] is True
    assert driver.os.environ["VMEC_JAX_WOUT_FAST_BCOVAR"] == "keep"

    run.result.diagnostics["converged"] = False
    driver.wout_from_fixed_boundary_run(run, include_fsq=True, fast_bcovar=True)
    assert captured[-1]["converged"] is False

    run_fallback = driver.FixedBoundaryRun(
        cfg=SimpleNamespace(),
        indata=_Indata(),
        static=SimpleNamespace(),
        state=SimpleNamespace(),
        result=SimpleNamespace(diagnostics={}),
        flux=SimpleNamespace(),
        profiles={},
        signgs=1,
    )
    fallback = driver.wout_from_fixed_boundary_run(run_fallback, include_fsq=True, fast_bcovar=False)
    assert (fallback.fsqr, fallback.fsqz, fallback.fsql) == pytest.approx((9.0, 8.0, 7.0))

    no_fsq = driver.wout_from_fixed_boundary_run(run_fallback, include_fsq=False)
    assert (no_fsq.fsqr, no_fsq.fsqz, no_fsq.fsql) == pytest.approx((0.0, 0.0, 0.0))

    written = {}
    monkeypatch.setattr(driver, "wout_from_fixed_boundary_run", lambda *_args, **_kwargs: no_fsq)
    monkeypatch.setattr(
        wout_module,
        "write_wout",
        lambda path, wout_obj, *, overwrite: written.update(path=Path(path), wout=wout_obj, overwrite=overwrite),
    )
    out_path = tmp_path / "nested" / "wout_test.nc"
    assert driver.write_wout_from_fixed_boundary_run(out_path, run_fallback, include_fsq=False) is no_fsq
    assert out_path.parent.exists()
    assert written == {"path": out_path, "wout": no_fsq, "overwrite": True}

    input_dir = tmp_path / "examples" / "data"
    input_dir.mkdir(parents=True)
    (input_dir / "input.demo").write_text("&INDATA /\n")
    (input_dir / "wout_demo.nc").write_text("placeholder\n")
    assert driver.example_paths("demo", root=tmp_path) == (input_dir / "input.demo", input_dir / "wout_demo.nc")
    assert driver.example_paths("missing", root=tmp_path)[1] is None

    npz_path = driver.save_npz(tmp_path / "arrays" / "x.npz", x=np.asarray([1.0, 2.0]))
    assert npz_path.exists()
    with np.load(npz_path) as data:
        np.testing.assert_allclose(data["x"], [1.0, 2.0])
