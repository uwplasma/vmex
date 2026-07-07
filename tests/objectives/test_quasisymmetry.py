from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest


def _optional_wout_path(filename: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(root, "examples", "data", filename)
    if not os.path.exists(path):
        pytest.skip(f"Optional WOUT fixture is missing: {filename}. Run tools/fetch_assets.py --bundle wout-fixtures.")
    return path


def _require_slow() -> None:
    if os.environ.get("RUN_SLOW", "") != "1":
        pytest.skip("Set RUN_SLOW=1 to run slow quasisymmetry checks")


def test_quasisymmetry_ratio_residual_from_state_is_self_consistent(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.quasisymmetry import (
        quasisymmetry_diagnostics_from_state,
        quasisymmetry_ratio_residual_from_state,
        quasisymmetry_ratio_residual_from_wout,
    )
    from vmec_jax.solve import solve_fixed_boundary_residual_iter

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))

    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-13)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces=True,
        use_scan=False,
        light_history=True,
    )

    diag = quasisymmetry_diagnostics_from_state(
        state=result.state,
        static=static,
        indata=indata,
        signgs=signgs,
    )
    qs_state = quasisymmetry_ratio_residual_from_state(
        state=result.state,
        static=static,
        indata=indata,
        signgs=signgs,
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
    )
    qs_diag = quasisymmetry_ratio_residual_from_wout(
        diag,
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
    )

    assert np.asarray(diag.gmnc).ndim == 2
    assert np.asarray(diag.bmnc).ndim == 2
    assert np.asarray(diag.bsubumnc).ndim == 2
    assert np.asarray(diag.bsupumnc).ndim == 2
    np.testing.assert_allclose(
        np.asarray(qs_state["residuals1d"]),
        np.asarray(qs_diag["residuals1d"]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert float(np.asarray(qs_state["total"])) > 0.0


def test_as_jax_array_is_tracer_safe():
    pytest.importorskip("jax")

    import jax
    import jax.numpy as jnp

    from vmec_jax.quasisymmetry import _as_jax_array

    @jax.jit
    def traced(values):
        arr = _as_jax_array(values, dtype=np.float64)
        return jnp.sum(arr * arr)

    result = traced(jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64))
    np.testing.assert_allclose(np.asarray(result), 14.0, rtol=0.0, atol=0.0)


def test_quasisymmetry_surface_and_weight_helpers():
    pytest.importorskip("jax")

    from vmec_jax.quasisymmetry import _as_surface_array, _as_weight_array, _half_grid, _interp_half_grid

    np.testing.assert_allclose(np.asarray(_as_surface_array(0.5)), [0.5])
    np.testing.assert_allclose(np.asarray(_as_surface_array([0.25, 0.75])), [0.25, 0.75])
    np.testing.assert_allclose(np.asarray(_as_weight_array(None, 2)), [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(_as_weight_array([2.0, 3.0], 2)), [2.0, 3.0])
    np.testing.assert_allclose(np.asarray(_half_grid(4, np.float64)), [1.0 / 6.0, 0.5, 5.0 / 6.0])

    samples = np.array([[10.0, 20.0], [30.0, 60.0], [50.0, 100.0]])
    s_half = np.array([0.0, 0.5, 1.0])
    interp = _interp_half_grid(samples, [0.25, 0.75], s_half)
    np.testing.assert_allclose(np.asarray(interp), [[20.0, 40.0], [40.0, 80.0]])

    single = _interp_half_grid(np.array([[7.0, 9.0]]), [0.25, 0.75], np.array([0.5]))
    np.testing.assert_allclose(np.asarray(single), [[7.0, 9.0], [7.0, 9.0]])

    with pytest.raises(ValueError, match="half-grid interpolation"):
        _interp_half_grid(np.zeros((0,)), [0.5], np.zeros((0,)))


def test_boozer_mode_quasisymmetry_residual_masks_non_qs_modes():
    pytest.importorskip("jax")

    from vmec_jax.quasisymmetry import quasisymmetry_boozer_mode_residual_from_boozer_output

    booz = {
        "bmnc_b": np.asarray([[2.0, 0.3, 0.4]]),
        "bmns_b": np.asarray([[0.0, 0.1, 0.2]]),
        "ixm_b": np.asarray([0, 1, 1]),
        "ixn_b": np.asarray([0, 0, 2]),
        "nfp_b": np.asarray([2]),
    }

    qa = quasisymmetry_boozer_mode_residual_from_boozer_output(
        booz,
        helicity_m=1,
        helicity_n=0,
    )
    expected_denominator = 2.0**2 + 0.3**2 + 0.4**2 + 0.1**2 + 0.2**2
    np.testing.assert_allclose(np.asarray(qa["total"]), (0.4**2 + 0.2**2) / expected_denominator)
    np.testing.assert_array_equal(np.asarray(qa["non_qs_mask"]), [False, False, True])

    qh = quasisymmetry_boozer_mode_residual_from_boozer_output(
        booz,
        helicity_m=1,
        helicity_n=1,
        normalize=False,
    )
    np.testing.assert_allclose(np.asarray(qh["total"]), 0.3**2 + 0.1**2)
    np.testing.assert_array_equal(np.asarray(qh["non_qs_mask"]), [False, True, False])

    with pytest.raises(ValueError, match="Boozer mode arrays"):
        quasisymmetry_boozer_mode_residual_from_boozer_output({**booz, "ixn_b": np.asarray([0, 1])})


def test_quasisymmetry_coefficient_shape_helpers():
    pytest.importorskip("jax")

    from vmec_jax.quasisymmetry import _optional_radial_mode_matrix, _radial_mode_matrix

    direct = _radial_mode_matrix(np.arange(6.0).reshape(3, 2), radial_count=3, mode_count=2)
    transposed = _radial_mode_matrix(np.arange(6.0).reshape(2, 3), radial_count=3, mode_count=2)

    np.testing.assert_allclose(np.asarray(direct), np.arange(6.0).reshape(3, 2))
    np.testing.assert_allclose(np.asarray(transposed), np.arange(6.0).reshape(2, 3).T)

    like = np.ones((3, 2))
    zeros = _optional_radial_mode_matrix(
        SimpleNamespace(lasym=False, bmns=np.ones((3, 2))),
        "bmns",
        radial_count=3,
        mode_count=2,
        like=like,
    )
    np.testing.assert_allclose(np.asarray(zeros), np.zeros((3, 2)))

    values = _optional_radial_mode_matrix(
        SimpleNamespace(lasym=True, bmns=np.arange(6.0).reshape(3, 2)),
        "bmns",
        radial_count=3,
        mode_count=2,
        like=like,
    )
    np.testing.assert_allclose(np.asarray(values), np.arange(6.0).reshape(3, 2))

    with pytest.raises(ValueError, match="expected a rank-2"):
        _radial_mode_matrix(np.ones((2, 2, 2)), radial_count=2, mode_count=2)
    with pytest.raises(ValueError, match="unexpected coefficient shape"):
        _radial_mode_matrix(np.ones((4, 4)), radial_count=3, mode_count=2)


def test_quasisymmetry_nyquist_coeff_helpers_project_known_modes_and_validate_inputs():
    pytest.importorskip("jax")

    from vmec_jax.modes import ModeTable
    from vmec_jax.quasisymmetry import (
        _as_jax_array,
        _half_grid,
        _vmec_symoutput_split_jax,
        _vmec_wrout_nyquist_cos_coeffs_jax,
        _vmec_wrout_nyquist_sin_coeffs_jax,
        quasisymmetry_ratio_residual_from_wout,
    )
    from vmec_jax.kernels.tomnsp import vmec_trig_tables

    trig = vmec_trig_tables(ntheta=6, nzeta=5, nfp=1, mmax=2, nmax=2, lasym=False)
    modes = ModeTable(m=np.array([0, 1, 1, 2]), n=np.array([0, 1, -1, 0]))
    theta = 2.0 * np.pi * np.arange(int(trig.ntheta3)) / float(trig.ntheta1)
    zeta = 2.0 * np.pi * np.arange(5) / 5.0

    constant = np.ones((2, int(trig.ntheta3), 5))
    cos_coeff = np.asarray(_vmec_wrout_nyquist_cos_coeffs_jax(f=constant, modes=modes, trig=trig))
    sin_coeff = np.asarray(_vmec_wrout_nyquist_sin_coeffs_jax(f=constant, modes=modes, trig=trig))
    np.testing.assert_allclose(cos_coeff[:, 0], 1.0, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(cos_coeff[:, 1:], 0.0, atol=1e-13)
    np.testing.assert_allclose(sin_coeff, 0.0, atol=1e-13)

    helical_cos = np.cos(theta[:, None] - zeta[None, :])[None, :, :]
    helical_sin = np.sin(theta[:, None] - zeta[None, :])[None, :, :]
    cos_projected = np.asarray(_vmec_wrout_nyquist_cos_coeffs_jax(f=helical_cos, modes=modes, trig=trig))
    sin_projected = np.asarray(_vmec_wrout_nyquist_sin_coeffs_jax(f=helical_sin, modes=modes, trig=trig))
    np.testing.assert_allclose(cos_projected[0, 1], 1.0, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(sin_projected[0, 1], 1.0, rtol=1e-13, atol=1e-13)

    empty_modes = ModeTable(m=np.array([], dtype=int), n=np.array([], dtype=int))
    empty = _vmec_wrout_nyquist_cos_coeffs_jax(f=constant, modes=empty_modes, trig=trig)
    empty_sin = _vmec_wrout_nyquist_sin_coeffs_jax(f=constant, modes=empty_modes, trig=trig)
    assert np.asarray(empty).shape == (2, 0)
    assert np.asarray(empty_sin).shape == (2, 0)

    np.testing.assert_allclose(np.asarray(_half_grid(1, np.float64)), [])
    big_endian = np.array([1.0, 2.0], dtype=">f8")
    np.testing.assert_allclose(np.asarray(_as_jax_array(big_endian)), [1.0, 2.0])

    with pytest.raises(ValueError, match="Expected f with shape"):
        _vmec_wrout_nyquist_cos_coeffs_jax(f=np.zeros((2, 3)), modes=modes, trig=trig)
    with pytest.raises(ValueError, match="Expected f with shape"):
        _vmec_wrout_nyquist_sin_coeffs_jax(f=np.zeros((2, 3)), modes=modes, trig=trig)
    with pytest.raises(ValueError, match="smaller than VMEC ntheta2"):
        _vmec_wrout_nyquist_cos_coeffs_jax(f=np.zeros((1, int(trig.ntheta2) - 1, 5)), modes=modes, trig=trig)
    with pytest.raises(ValueError, match="smaller than VMEC ntheta2"):
        _vmec_wrout_nyquist_sin_coeffs_jax(f=np.zeros((1, int(trig.ntheta2) - 1, 5)), modes=modes, trig=trig)
    with pytest.raises(ValueError, match="Input theta grid"):
        _vmec_symoutput_split_jax(f=np.zeros((1, int(trig.ntheta2) - 1, 5)), trig=trig)
    with pytest.raises(ValueError, match="Expected f with shape"):
        _vmec_symoutput_split_jax(f=np.zeros((2, 3)), trig=trig)
    with pytest.raises(ValueError, match="weights must have the same length"):
        quasisymmetry_ratio_residual_from_wout(SimpleNamespace(), surfaces=[0.25, 0.5], weights=[1.0])


def test_quasisymmetry_diagnostics_from_state_uses_lightweight_vmec_dependencies(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.booz_input as booz_input
    import vmec_jax.driver as driver
    import vmec_jax.energy as energy
    import vmec_jax.profiles as profiles
    import vmec_jax.kernels.bcovar as vmec_bcovar
    import vmec_jax.kernels.lforbal as vmec_lforbal
    from vmec_jax.quasisymmetry import quasisymmetry_diagnostics_from_state

    class InData:
        def get_bool(self, _name, default=False):
            return bool(default)

        def get_int(self, name, default=0):
            return 1 if name == "NCURR" else int(default)

    def eval_profiles(_indata, s):
        return {"pressure": np.linspace(0.0, 0.3, len(s))}

    def flux_profiles_from_indata(_indata, s, signgs):
        del signgs
        return SimpleNamespace(
            phipf=np.ones(len(s)),
            chipf=np.linspace(0.0, 0.2, len(s)),
            phips=np.ones(len(s)),
        )

    def final_flux_profiles_from_state(**kwargs):
        s = np.asarray(kwargs["static_in"].s)
        return flux_profiles_from_indata(None, s, 1), {"pressure": np.linspace(0.0, 0.3, len(s))}

    def vmec_bcovar_half_mesh_from_wout(**kwargs):
        static = kwargs["static"]
        trig = kwargs["trig"]
        ns = len(static.s)
        shape = (ns, int(trig.ntheta3), int(static.cfg.nzeta))
        radial = np.arange(ns, dtype=float)[:, None, None]
        theta = np.arange(shape[1], dtype=float)[None, :, None]
        zeta = np.arange(shape[2], dtype=float)[None, None, :]
        field = np.ones(shape) + 0.05 * radial + 0.03 * theta + 0.02 * zeta
        return SimpleNamespace(
            jac=SimpleNamespace(sqrtg=field),
            bsq=2.0 + 0.2 * field,
            bsubu=0.2 + 0.01 * field,
            bsubv=0.4 + 0.02 * field,
            bsupu=0.5 + 0.03 * field,
            bsupv=0.6 + 0.04 * field,
        )

    def currents_from_bcovar(**kwargs):
        ns = len(kwargs["s"])
        return np.linspace(0.0, 0.1, ns), np.linspace(1.0, 1.2, ns), None, None

    def parity_filter(**kwargs):
        return kwargs["bsubu_even"], kwargs["bsubv_even"]

    monkeypatch.setattr(profiles, "eval_profiles", eval_profiles)
    monkeypatch.setattr(energy, "flux_profiles_from_indata", flux_profiles_from_indata)
    monkeypatch.setattr(driver, "_final_flux_profiles_from_state", final_flux_profiles_from_state)
    monkeypatch.setattr(vmec_bcovar, "vmec_bcovar_half_mesh_from_wout", vmec_bcovar_half_mesh_from_wout)
    monkeypatch.setattr(vmec_lforbal, "currents_from_bcovar", currents_from_bcovar)
    monkeypatch.setattr(booz_input, "_filter_bsubuv_jxbforce_parity_jax", parity_filter)

    from collections import namedtuple

    Cfg = namedtuple("Cfg", "mpol ntor ntheta nzeta nfp lasym lthreed")
    for lasym in (False, True):
        cfg = Cfg(mpol=2, ntor=1, ntheta=6, nzeta=4, nfp=1, lasym=lasym, lthreed=True)
        static = SimpleNamespace(s=np.linspace(0.0, 1.0, 4), cfg=cfg)
        diag = quasisymmetry_diagnostics_from_state(
            state=SimpleNamespace(Rcos=np.ones((4, 3))),
            static=static,
            indata=InData(),
            signgs=1,
        )

        assert diag.lasym is lasym
        assert np.asarray(diag.buco).shape == (4,)
        assert np.asarray(diag.bvco).shape == (4,)
        assert np.asarray(diag.gmnc).shape[0] == 4
        assert np.asarray(diag.bmnc).shape == np.asarray(diag.gmnc).shape
        assert np.all(np.isfinite(np.asarray(diag.bsubumnc)))
        assert np.all(np.isfinite(np.asarray(diag.bsubvmnc)))
        if lasym:
            assert np.asarray(diag.bmns).shape == np.asarray(diag.bmnc).shape
        else:
            np.testing.assert_allclose(np.asarray(diag.bmns), 0.0)


def test_quasisymmetry_symoutput_split_reconstructs_half_grid():
    pytest.importorskip("jax")

    from vmec_jax.quasisymmetry import _vmec_symoutput_split_jax

    trig = SimpleNamespace(ntheta2=3, ntheta1=4)
    f = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)

    sym, asym = _vmec_symoutput_split_jax(f=f, trig=trig)
    rev_sym, rev_asym = _vmec_symoutput_split_jax(f=f, trig=trig, reversed_sym=True)

    np.testing.assert_allclose(np.asarray(sym + asym), f[:, :3, :])
    np.testing.assert_allclose(np.asarray(rev_sym + rev_asym), f[:, :3, :])
    np.testing.assert_allclose(np.asarray(rev_sym), np.asarray(asym))
    np.testing.assert_allclose(np.asarray(rev_asym), np.asarray(sym))


def test_quasisymmetry_ratio_residual_returns_diagnostic_fields():
    pytest.importorskip("jax")

    from vmec_jax import load_wout
    from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_wout

    wout = load_wout(_optional_wout_path("wout_li383_low_res.nc"))
    qs = quasisymmetry_ratio_residual_from_wout(
        wout,
        surfaces=[0.5],
        helicity_m=1,
        helicity_n=1,
        ntheta=17,
        nphi=18,
    )

    for key in (
        "d_B_d_theta",
        "d_B_d_phi",
        "bsubu",
        "bsubv",
        "bsupu",
        "bsupv",
        "d_psi_d_s",
        "V_prime",
    ):
        assert key in qs

    np.testing.assert_allclose(
        np.asarray(qs["bsupu"] * qs["d_B_d_theta"] + qs["bsupv"] * qs["d_B_d_phi"]),
        np.asarray(qs["B_dot_grad_B"]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(
            qs["d_psi_d_s"]
            * (qs["bsubu"] * qs["d_B_d_phi"] - qs["bsubv"] * qs["d_B_d_theta"])
            / qs["sqrtg"]
        ),
        np.asarray(qs["B_cross_grad_B_dot_grad_psi"]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_quasisymmetry_wout_residual_gradient_matches_finite_difference():
    pytest.importorskip("jax")

    import jax
    import jax.numpy as jnp

    from vmec_jax._compat import enable_x64
    from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_wout

    enable_x64(True)

    constant_mode = jnp.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=jnp.float64,
    )

    def objective(alpha):
        bmnc = jnp.asarray(
            [
                [1.0, alpha],
                [1.0, alpha],
                [1.0, alpha],
            ],
            dtype=jnp.float64,
        )
        wout_like = SimpleNamespace(
            nfp=2,
            lasym=False,
            iotas=jnp.asarray([0.0, 0.4, 0.5], dtype=jnp.float64),
            buco=jnp.asarray([0.0, 0.2, 0.25], dtype=jnp.float64),
            bvco=jnp.asarray([0.0, 1.0, 1.1], dtype=jnp.float64),
            gmnc=constant_mode,
            bmnc=bmnc,
            bsubumnc=0.2 * constant_mode,
            bsubvmnc=0.3 * constant_mode,
            bsupumnc=0.4 * constant_mode,
            bsupvmnc=0.5 * constant_mode,
            xm_nyq=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
            xn_nyq=jnp.asarray([0.0, 2.0], dtype=jnp.float64),
            phi=jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64),
        )
        return quasisymmetry_ratio_residual_from_wout(
            wout_like,
            surfaces=[0.5],
            helicity_m=1,
            helicity_n=-1,
            ntheta=9,
            nphi=10,
        )["total"]

    alpha0 = jnp.asarray(0.08, dtype=jnp.float64)
    eps = jnp.asarray(1.0e-5, dtype=jnp.float64)
    value, grad_ad = jax.value_and_grad(objective)(alpha0)
    grad_fd = (objective(alpha0 + eps) - objective(alpha0 - eps)) / (2.0 * eps)

    assert float(np.asarray(value)) > 0.0
    assert np.isfinite(float(np.asarray(grad_ad)))
    np.testing.assert_allclose(np.asarray(grad_ad), np.asarray(grad_fd), rtol=1.0e-7, atol=1.0e-10)


def test_quasisymmetry_wout_residual_jvp_and_vjp_match_finite_difference():
    pytest.importorskip("jax")

    import jax
    import jax.numpy as jnp

    from vmec_jax._compat import enable_x64
    from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_wout

    enable_x64(True)

    constant_mode = jnp.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=jnp.float64,
    )

    def residuals(alpha):
        bmnc = jnp.asarray(
            [
                [1.0, alpha],
                [1.0, alpha],
                [1.0, alpha],
            ],
            dtype=jnp.float64,
        )
        wout_like = SimpleNamespace(
            nfp=2,
            lasym=False,
            iotas=jnp.asarray([0.0, 0.4, 0.5], dtype=jnp.float64),
            buco=jnp.asarray([0.0, 0.2, 0.25], dtype=jnp.float64),
            bvco=jnp.asarray([0.0, 1.0, 1.1], dtype=jnp.float64),
            gmnc=constant_mode,
            bmnc=bmnc,
            bsubumnc=0.2 * constant_mode,
            bsubvmnc=0.3 * constant_mode,
            bsupumnc=0.4 * constant_mode,
            bsupvmnc=0.5 * constant_mode,
            xm_nyq=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
            xn_nyq=jnp.asarray([0.0, 2.0], dtype=jnp.float64),
            phi=jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64),
        )
        return quasisymmetry_ratio_residual_from_wout(
            wout_like,
            surfaces=[0.5],
            helicity_m=1,
            helicity_n=-1,
            ntheta=9,
            nphi=10,
        )["residuals1d"]

    alpha0 = jnp.asarray(0.08, dtype=jnp.float64)
    direction = jnp.asarray(-0.37, dtype=jnp.float64)
    eps = jnp.asarray(1.0e-5, dtype=jnp.float64)

    residual0, jvp_ad = jax.jvp(residuals, (alpha0,), (direction,))
    jvp_fd = (residuals(alpha0 + eps * direction) - residuals(alpha0 - eps * direction)) / (2.0 * eps)

    cotangent = jnp.linspace(-0.25, 0.35, int(residual0.size), dtype=jnp.float64)
    _, pullback = jax.vjp(residuals, alpha0)
    (vjp_ad,) = pullback(cotangent)
    vjp_fd = (
        jnp.vdot(cotangent, residuals(alpha0 + eps))
        - jnp.vdot(cotangent, residuals(alpha0 - eps))
    ) / (2.0 * eps)

    assert np.all(np.isfinite(np.asarray(jvp_ad)))
    assert np.isfinite(float(np.asarray(vjp_ad)))
    np.testing.assert_allclose(np.asarray(jvp_ad), np.asarray(jvp_fd), rtol=2.0e-6, atol=1.0e-9)
    np.testing.assert_allclose(np.asarray(vjp_ad), np.asarray(vjp_fd), rtol=2.0e-6, atol=1.0e-9)


def test_quasisymmetry_angle_cache_matches_uncached_wout():
    pytest.importorskip("jax")

    from vmec_jax import load_wout
    from vmec_jax.quasisymmetry import (
        quasisymmetry_angle_cache,
        quasisymmetry_ratio_residual_from_wout,
    )

    wout = load_wout(_optional_wout_path("wout_nfp4_QH_warm_start.nc"))
    cache = quasisymmetry_angle_cache(
        nfp=int(wout.nfp),
        xm_nyq=wout.xm_nyq,
        xn_nyq=wout.xn_nyq,
        ntheta=13,
        nphi=14,
    )
    uncached = quasisymmetry_ratio_residual_from_wout(
        wout,
        surfaces=[0.25, 0.5, 0.75],
        helicity_m=1,
        helicity_n=-1,
        ntheta=13,
        nphi=14,
    )
    cached = quasisymmetry_ratio_residual_from_wout(
        wout,
        surfaces=[0.25, 0.5, 0.75],
        helicity_m=1,
        helicity_n=-1,
        angle_cache=cache,
    )

    assert np.asarray(cached["residuals1d"]).shape == (3 * 13 * 14,)
    np.testing.assert_allclose(
        np.asarray(cached["residuals1d"]),
        np.asarray(uncached["residuals1d"]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(np.asarray(cached["total"]), np.asarray(uncached["total"]), rtol=0.0, atol=0.0)


def test_quasisymmetry_angle_cache_from_static_matches_generic_cache():
    pytest.importorskip("jax")

    from types import SimpleNamespace

    from vmec_jax.modes import nyquist_mode_table_from_grid
    from vmec_jax.quasisymmetry import quasisymmetry_angle_cache, quasisymmetry_angle_cache_from_static

    static = SimpleNamespace(cfg=SimpleNamespace(nfp=2, mpol=3, ntor=2, ntheta=10, nzeta=8))
    nyq_modes = nyquist_mode_table_from_grid(
        mpol=static.cfg.mpol,
        ntor=static.cfg.ntor,
        ntheta=static.cfg.ntheta,
        nzeta=static.cfg.nzeta,
    )

    from_static = quasisymmetry_angle_cache_from_static(static, ntheta=7, nphi=9)
    generic = quasisymmetry_angle_cache(
        nfp=static.cfg.nfp,
        xm_nyq=nyq_modes.m,
        xn_nyq=nyq_modes.n * static.cfg.nfp,
        ntheta=7,
        nphi=9,
    )

    assert int(from_static["ntheta"]) == 7
    assert int(from_static["nphi"]) == 9
    np.testing.assert_allclose(np.asarray(from_static["cosangle"]), np.asarray(generic["cosangle"]))
    np.testing.assert_allclose(np.asarray(from_static["sinangle"]), np.asarray(generic["sinangle"]))


@pytest.mark.parametrize(
    ("filename", "helicity_m", "helicity_n", "expected_total", "expected_profile"),
    [
        (
            "wout_li383_low_res.nc",
            1,
            1,
            0.20657479140083462,
            [0.026328741242275744, 0.06337277437029094, 0.11687327578826792],
        ),
        (
            "wout_nfp4_QH_warm_start.nc",
            1,
            -1,
            0.06535102501039897,
            [0.004964336247390136, 0.017396361463970656, 0.042990327299038164],
        ),
        (
            "wout_LandremanPaul2021_QA_lowres.nc",
            1,
            0,
            7.800132285797323e-08,
            [1.8742987587697922e-08, 2.3570159194276813e-08, 3.568817607599847e-08],
        ),
    ],
)
def test_quasisymmetry_ratio_residual_regression_on_bundled_wouts(
    filename,
    helicity_m,
    helicity_n,
    expected_total,
    expected_profile,
):
    pytest.importorskip("jax")

    from vmec_jax import load_wout
    from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_wout

    wout = load_wout(_optional_wout_path(filename))
    qs = quasisymmetry_ratio_residual_from_wout(
        wout,
        surfaces=[0.25, 0.5, 0.75],
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ntheta=13,
        nphi=14,
    )

    assert np.asarray(qs["residuals1d"]).shape == (3 * 13 * 14,)
    np.testing.assert_allclose(np.asarray(qs["total"]), expected_total, rtol=1.0e-8, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(qs["profile"]), expected_profile, rtol=1.0e-7, atol=1.0e-12)


def test_accepted_wout_output_preserves_qs_metric_against_bundled_vmec2000(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    from vmec_jax import build_static, load_config, load_wout
    from vmec_jax.optimization import FixedBoundaryExactOptimizer
    from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_wout
    from vmec_jax.wout import state_from_wout

    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_path = os.path.join(root, "examples", "data", "input.LandremanPaul2021_QA_lowres")
    wout_path = os.path.join(root, "examples", "data", "wout_LandremanPaul2021_QA_lowres.nc")
    if not os.path.exists(input_path) or not os.path.exists(wout_path):
        pytest.skip("Missing bundled QA VMEC2000 fixture")

    cfg, indata = load_config(input_path)
    static = build_static(cfg)
    ref_wout = load_wout(wout_path)
    accepted_state = state_from_wout(ref_wout)

    # Exercise the accepted-state writer used by optimization final artifacts.
    optimizer = object.__new__(FixedBoundaryExactOptimizer)
    optimizer._static = static
    optimizer._indata = indata
    optimizer._flux = object()
    optimizer._signgs = int(ref_wout.signgs)
    optimizer._exact_cache = {}
    optimizer._exact_state_cache = {}
    optimizer._profile = {}

    out_path = tmp_path / "wout_accepted.nc"
    optimizer.save_wout(out_path, state=accepted_state)
    accepted_wout = load_wout(out_path)

    qs_kwargs = {
        "surfaces": [0.25, 0.5, 0.75],
        "helicity_m": 1,
        "helicity_n": 0,
        "ntheta": 13,
        "nphi": 14,
    }
    ref_qs = quasisymmetry_ratio_residual_from_wout(ref_wout, **qs_kwargs)
    accepted_qs = quasisymmetry_ratio_residual_from_wout(accepted_wout, **qs_kwargs)

    np.testing.assert_allclose(
        np.asarray(accepted_qs["residuals1d"]),
        np.asarray(ref_qs["residuals1d"]),
        rtol=1.0e-4,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(accepted_qs["total"]),
        np.asarray(ref_qs["total"]),
        rtol=1.0e-4,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(accepted_wout.rmnc),
        np.asarray(ref_wout.rmnc),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(accepted_wout.zmns),
        np.asarray(ref_wout.zmns),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_quasisymmetry_ratio_residual_supports_lasym_wout():
    pytest.importorskip("jax")

    from vmec_jax import load_wout
    from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_wout

    wout = load_wout(_optional_wout_path("wout_basic_non_stellsym_simsopt.nc"))
    assert bool(wout.lasym)

    qs = quasisymmetry_ratio_residual_from_wout(
        wout,
        surfaces=[0.5],
        helicity_m=1,
        helicity_n=0,
        ntheta=13,
        nphi=14,
    )

    assert np.asarray(qs["residuals1d"]).ndim == 1
    assert np.all(np.isfinite(np.asarray(qs["residuals1d"])))
    assert np.isfinite(float(np.asarray(qs["total"])))
    assert float(np.linalg.norm(np.asarray(wout.bmns))) > 0.0


def test_scan_cache_lru_helpers_evict_oldest(monkeypatch):
    from collections import OrderedDict

    from vmec_jax.discrete_adjoint import _lru_cache_get, _lru_cache_put

    monkeypatch.setenv("VMEC_JAX_SCAN_CACHE_LIMIT", "2")
    cache = OrderedDict()
    _lru_cache_put(cache, ("a",), 1)
    _lru_cache_put(cache, ("b",), 2)
    assert list(cache.keys()) == [("a",), ("b",)]

    assert _lru_cache_get(cache, ("a",)) == 1
    assert list(cache.keys()) == [("b",), ("a",)]

    _lru_cache_put(cache, ("c",), 3)
    assert list(cache.keys()) == [("a",), ("c",)]
