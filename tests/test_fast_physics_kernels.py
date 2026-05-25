from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import jnp
from vmec_jax.energy import (
    TWOPI,
    FluxProfiles,
    _iotaf_from_iotas,
    _poly_no_const,
    _poly_no_const_deriv,
    flux_profiles_from_indata,
    integrate_volume_density,
)
from vmec_jax.integrals import (
    cumrect_s_halfmesh,
    cumtrapz_s,
    dvds_from_sqrtg,
    dvds_from_sqrtg_zeta,
    volume_from_sqrtg,
    volume_from_sqrtg_vmec,
)
from vmec_jax.modes import ModeTable
from vmec_jax.namelist import InData
from vmec_jax.profiles import MU0, eval_profiles, profiles_from_indata
from vmec_jax.residuals import _rms, _sum_squares_state, force_residuals_from_state
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.vmec2000_exec import (
    _find_threed1_file,
    _infer_case_name,
    _parse_vmec2000_threed1,
    _patch_indata,
    find_vmec2000_exec,
    flatten_threed1,
    run_xvmec2000,
    threed1_fsq_total,
)


def test_angle_volume_integrals_match_constant_jacobian_conventions() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
    s = np.array([0.0, 0.25, 1.0])
    sqrtg = np.full((s.size, theta.size, zeta.size), 2.0)

    # dvds_from_sqrtg integrates over physical toroidal angle phi=zeta/nfp.
    dvds = dvds_from_sqrtg(sqrtg, theta, zeta, nfp=2)
    np.testing.assert_allclose(np.asarray(dvds), np.full(s.size, 4.0 * np.pi**2))

    # VMEC wout-style zeta integration is over one field-period zeta grid and
    # uses signgs instead of abs(sqrtg).
    dvds_vmec = dvds_from_sqrtg_zeta(-sqrtg, theta, zeta, signgs=-1)
    np.testing.assert_allclose(np.asarray(dvds_vmec), np.full(s.size, 8.0 * np.pi**2))

    _, volume_trap = volume_from_sqrtg(sqrtg, s, theta, zeta, nfp=2)
    np.testing.assert_allclose(np.asarray(volume_trap), [0.0, np.pi**2, 4.0 * np.pi**2])

    _, volume_rect = volume_from_sqrtg_vmec(-sqrtg, s, theta, zeta, signgs=-1)
    np.testing.assert_allclose(np.asarray(volume_rect), [0.0, 2.0 * np.pi**2, 8.0 * np.pi**2])


def test_radial_integral_helpers_validate_shapes_and_half_mesh_rule() -> None:
    y = jnp.asarray([10.0, 20.0, 40.0])
    s = jnp.asarray([0.0, 0.25, 1.0])

    np.testing.assert_allclose(np.asarray(cumtrapz_s(y, s)), [0.0, 3.75, 26.25])
    np.testing.assert_allclose(np.asarray(cumrect_s_halfmesh(y, s)), [0.0, 5.0, 35.0])

    with pytest.raises(ValueError, match="same length"):
        cumtrapz_s(jnp.asarray([1.0, 2.0]), s)
    with pytest.raises(ValueError, match="non-empty"):
        dvds_from_sqrtg(np.zeros((1, 0, 2)), np.asarray([]), np.asarray([0.0, 1.0]), nfp=1)
    with pytest.raises(ValueError, match="nfp must be positive"):
        dvds_from_sqrtg(np.ones((1, 2, 2)), np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), nfp=0)


def test_integrate_volume_density_uses_full_torus_rectangle_rule() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    s = np.array([0.0, 0.5, 1.0])
    density = np.full((s.size, theta.size, zeta.size), 3.0)
    sqrtg = np.full_like(density, 2.0)

    integral = integrate_volume_density(density, sqrtg, s, theta, zeta, nfp=5, signgs=1)
    # The helper intentionally uses a VMEC-style rectangle rule over the stored
    # radial grid: sum_j density*sqrtg*ds*4*pi^2.
    np.testing.assert_allclose(np.asarray(integral), 36.0 * np.pi**2)


def test_polynomial_flux_profiles_follow_vmec_normalization() -> None:
    s = jnp.asarray([0.0, 0.25, 0.5, 1.0])
    indata = InData(
        scalars={
            "PHIEDGE": TWOPI,
            "APHI": [1.0, 1.0],  # torflux'(s)=1+2s; torflux(1)=2
            "PIOTA_TYPE": "power_series",
            "AI": [0.5],
            "LRFP": False,
        },
        indexed={},
    )

    flux = flux_profiles_from_indata(indata, s, signgs=1)

    expected_deriv = 1.0 + 2.0 * np.asarray(s)
    np.testing.assert_allclose(np.asarray(flux.phipf), 0.5 * expected_deriv, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(flux.chipf), 0.25 * expected_deriv, rtol=1e-12, atol=1e-12)
    assert int(flux.signgs) == 1
    assert np.asarray(flux.phips)[0] == 0.0
    assert np.isfinite(float(np.asarray(flux.lamscale)))


def test_iotaf_smoothing_matches_arithmetic_and_rfp_harmonic_closures() -> None:
    iotas = jnp.asarray([0.0, 2.0, 4.0, 8.0])
    np.testing.assert_allclose(np.asarray(_iotaf_from_iotas(iotas, lrfp=False)), [1.0, 3.0, 6.0, 10.0])

    harmonic = _iotaf_from_iotas(iotas, lrfp=True)
    expected0 = 1.0 / (1.5 / 2.0 - 0.5 / 4.0)
    expected_mid = [2.0 / (1.0 / 2.0 + 1.0 / 4.0), 2.0 / (1.0 / 4.0 + 1.0 / 8.0)]
    expectedN = 1.0 / (1.5 / 8.0 - 0.5 / 4.0)
    np.testing.assert_allclose(np.asarray(harmonic), [expected0, *expected_mid, expectedN])


def test_profile_two_power_and_cubic_spline_current_contracts() -> None:
    s = jnp.asarray([0.0, 0.25, 0.5, 1.0])
    indata = InData(
        scalars={
            "PMASS_TYPE": "two_power",
            "PIOTA_TYPE": "power_series",
            "PCURR_TYPE": "two_power",
            "AM": [8.0, 2.0, 1.0],  # p = 8 * (1 - s^2)
            "AI": [0.25, 0.5],
            "AC": [2.0, 1.0, 1.0],  # I' = 2*(1-s), so I = 2s - s^2
            "PRES_SCALE": 0.5,
            "BLOAT": 1.0,
            "SPRES_PED": 1.0,
            "LRFP": False,
            "NCURR": 1,
        },
        indexed={},
    )
    profiles = eval_profiles(indata, s)

    s_np = np.asarray(s)
    pressure_pa = 4.0 * (1.0 - s_np**2)
    np.testing.assert_allclose(np.asarray(profiles["pressure_pa"]), pressure_pa, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(profiles["pressure"]), MU0 * pressure_pa, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(profiles["iota"]), 0.25 + 0.5 * s_np, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(profiles["current"]), 2.0 * s_np - s_np**2, rtol=1e-12, atol=1e-12)

    spline = InData(
        scalars={
            "PMASS_TYPE": "power_series",
            "PIOTA_TYPE": "power_series",
            "PCURR_TYPE": "cubic_spline_i",
            "AM": [0.0],
            "AI": [0.0],
            "AC": [0.0],
            "AC_AUX_S": [0.0, 1.0, 0.0],
            "AC_AUX_F": [3.0, 7.0, 0.0],
        },
        indexed={},
    )
    spline_profiles = eval_profiles(spline, s)
    np.testing.assert_allclose(np.asarray(spline_profiles["current"]), 3.0 + 4.0 * s_np, rtol=1e-12, atol=1e-12)


def test_profile_error_paths_lrfp_and_empty_spline_auxiliaries() -> None:
    s = jnp.asarray([0.0, 0.5, 1.0])

    lrfp = InData(
        scalars={
            "PMASS_TYPE": "power_series",
            "PIOTA_TYPE": "power_series",
            "PCURR_TYPE": "power_series",
            "AM": [0.0],
            "AI": [0.0, 2.0],
            "AC": [],
            "LRFP": True,
        },
        indexed={},
    )
    iota = np.asarray(eval_profiles(lrfp, s)["iota"])
    assert np.isinf(iota[0])
    np.testing.assert_allclose(iota[1:], [1.0, 0.5])

    empty_spline_i = InData(
        scalars={"PMASS_TYPE": "power_series", "PCURR_TYPE": "cubic_spline_i", "AM": [0.0], "AI": [], "AC": [0.0]},
        indexed={},
    )
    np.testing.assert_allclose(np.asarray(eval_profiles(empty_spline_i, s)["current"]), 0.0)

    empty_spline_ip = InData(
        scalars={"PMASS_TYPE": "power_series", "PCURR_TYPE": "cubic_spline_ip", "AM": [0.0], "AI": [], "AC": [0.0]},
        indexed={},
    )
    np.testing.assert_allclose(np.asarray(eval_profiles(empty_spline_ip, s)["current"]), 0.0)

    aux = profiles_from_indata(
        InData(
            scalars={
                "AC_AUX_S": [0.0, 0.4, 0.3, 1.0],
                "AC_AUX_F": [1.0, 2.0, 99.0, 100.0],
                "SPRES_PED": -0.2,
            },
            indexed={},
        )
    )
    np.testing.assert_allclose(np.asarray(aux.ac_aux_s), [0.0, 0.4])
    np.testing.assert_allclose(np.asarray(aux.ac_aux_f), [1.0, 2.0])
    assert aux.spres_ped == pytest.approx(0.2)

    with pytest.raises(NotImplementedError, match="pmass_type"):
        eval_profiles(InData(scalars={"PMASS_TYPE": "unsupported", "AM": [0.0], "AI": [], "AC": []}, indexed={}), s)
    with pytest.raises(NotImplementedError, match="piota_type"):
        eval_profiles(
            InData(
                scalars={"PMASS_TYPE": "power_series", "PIOTA_TYPE": "unsupported", "AM": [0.0], "AI": [0.1], "AC": []},
                indexed={},
            ),
            s,
        )
    with pytest.raises(NotImplementedError, match="pcurr_type"):
        eval_profiles(
            InData(
                scalars={"PMASS_TYPE": "power_series", "PCURR_TYPE": "unsupported", "AM": [0.0], "AI": [], "AC": [1.0]},
                indexed={},
            ),
            s,
        )


def test_energy_polynomial_helpers_are_derivative_consistent() -> None:
    x = jnp.asarray([0.0, 0.25, 0.5, 1.0])
    coeffs = [2.0, -3.0, 4.0]

    poly = _poly_no_const(coeffs, x)
    deriv = _poly_no_const_deriv(coeffs, x)

    x_np = np.asarray(x)
    np.testing.assert_allclose(np.asarray(poly), 2.0 * x_np - 3.0 * x_np**2 + 4.0 * x_np**3)
    np.testing.assert_allclose(np.asarray(deriv), 2.0 - 6.0 * x_np + 12.0 * x_np**2)


def test_residual_bookkeeping_sums_all_state_blocks() -> None:
    layout = StateLayout(ns=1, K=2, lasym=True)
    state = VMECState(
        layout=layout,
        Rcos=np.asarray([[1.0, 2.0]]),
        Rsin=np.asarray([[3.0, 4.0]]),
        Zcos=np.asarray([[5.0, 6.0]]),
        Zsin=np.asarray([[7.0, 8.0]]),
        Lcos=np.asarray([[9.0, 10.0]]),
        Lsin=np.asarray([[11.0, 12.0]]),
    )

    gR, gZ, gL = _sum_squares_state(state)
    assert gR == 1.0**2 + 2.0**2 + 3.0**2 + 4.0**2
    assert gZ == 5.0**2 + 6.0**2 + 7.0**2 + 8.0**2
    assert gL == 9.0**2 + 10.0**2 + 11.0**2 + 12.0**2
    assert _rms(np.asarray([3.0, 4.0])) == 3.5355339059327378
    assert _rms(np.asarray([])) == 0.0


def test_force_residuals_validate_jax_gamma_and_pressure_shape(monkeypatch) -> None:
    import vmec_jax.residuals as residuals_module

    monkeypatch.setattr(residuals_module, "has_jax", lambda: False)
    with pytest.raises(ImportError, match="requires JAX"):
        force_residuals_from_state(
            None,
            None,
            flux=None,
            pressure=np.asarray([0.0]),
            gamma=0.0,
        )

    monkeypatch.setattr(residuals_module, "has_jax", lambda: True)
    with pytest.raises(ValueError, match="gamma=1"):
        force_residuals_from_state(
            None,
            None,
            flux=None,
            pressure=np.asarray([0.0]),
            gamma=1.0,
        )

    static = SimpleNamespace(
        s=np.asarray([0.0, 1.0]),
        modes=ModeTable(m=np.asarray([0]), n=np.asarray([0])),
    )
    with pytest.raises(ValueError, match="pressure must have shape"):
        force_residuals_from_state(
            None,
            static,
            flux=None,
            pressure=np.asarray([0.0, 0.5, 1.0]),
            gamma=0.0,
        )


def test_force_residuals_compute_normalized_gradient_diagnostics(monkeypatch) -> None:
    import vmec_jax.residuals as residuals_module

    layout = StateLayout(ns=2, K=1, lasym=False)
    state = VMECState(
        layout=layout,
        Rcos=jnp.asarray([[1.0], [1.5]]),
        Rsin=jnp.zeros((2, 1)),
        Zcos=jnp.zeros((2, 1)),
        Zsin=jnp.asarray([[0.0], [0.2]]),
        Lcos=jnp.zeros((2, 1)),
        Lsin=jnp.zeros((2, 1)),
    )
    static = SimpleNamespace(
        s=jnp.asarray([0.0, 1.0]),
        grid=SimpleNamespace(theta=jnp.asarray([0.0, np.pi]), zeta=jnp.asarray([0.0, np.pi])),
        cfg=SimpleNamespace(nfp=1, ntheta=2, nzeta=2),
        modes=ModeTable(m=np.asarray([0]), n=np.asarray([0])),
    )
    flux = FluxProfiles(
        phipf=jnp.asarray([1.0, 1.0]),
        chipf=jnp.asarray([0.25, 0.25]),
        phips=jnp.asarray([0.0, 1.0]),
        signgs=1,
        lamscale=jnp.asarray(2.0),
    )

    def fake_eval_geom(st, _static):
        shape = (2, 2, 2)
        r = st.Rcos[:, :1, None]
        z = st.Zsin[:, :1, None]
        return SimpleNamespace(
            sqrtg=jnp.ones(shape) * (2.0 + 0.1 * r + 0.05 * z),
            g_tt=jnp.ones(shape) * 3.0,
        )

    def fake_bsup_from_geom(g, **_kwargs):
        return jnp.ones_like(g.sqrtg) * 2.0, jnp.ones_like(g.sqrtg) * 3.0

    def fake_b2_from_bsup(_g, bsupu, bsupv):
        return bsupu * bsupu + 0.25 * bsupv * bsupv

    def fake_bsub_from_bsup(g, _bsupu, _bsupv):
        return jnp.ones_like(g.sqrtg) * 4.0, jnp.ones_like(g.sqrtg) * 5.0

    monkeypatch.setattr(residuals_module, "eval_geom", fake_eval_geom)
    monkeypatch.setattr(residuals_module, "bsup_from_geom", fake_bsup_from_geom)
    monkeypatch.setattr(residuals_module, "b2_from_bsup", fake_b2_from_bsup)
    monkeypatch.setattr(residuals_module, "bsub_from_bsup", fake_bsub_from_bsup)
    monkeypatch.setattr(residuals_module, "_mask_grad_for_constraints", lambda grad, _static, idx00: grad)

    residuals = residuals_module.force_residuals_from_state(
        state,
        static,
        flux=flux,
        pressure=jnp.asarray([0.0, 0.5]),
        gamma=5.0 / 3.0,
        jacobian_penalty=0.0,
    )

    assert residuals.diagnostics["idx00"] == 0
    assert residuals.diagnostics["objective"] > 0.0
    assert residuals.diagnostics["volume"] > 0.0
    assert residuals.diagnostics["wb"] > 0.0
    assert residuals.diagnostics["wp"] > 0.0
    assert residuals.diagnostics["fnorm"] > 0.0
    assert residuals.diagnostics["fnormL"] > 0.0
    assert residuals.fsqr_like > 0.0
    assert residuals.fsqz_like > 0.0
    assert residuals.fsql_like == 0.0
    assert residuals.fsq_like == pytest.approx(residuals.fsqr_like + residuals.fsqz_like)
    assert residuals.grad_rms > 0.0
    assert residuals.grad_rms_rz > 0.0
    assert residuals.grad_rms_l == 0.0


def test_vmec2000_trace_parser_handles_multiple_stages_and_d_exponents(tmp_path: Path) -> None:
    threed1 = tmp_path / "threed1.case"
    threed1.write_text(
        "\n".join(
            [
                "  NS =   13 NO. FOURIER MODES =   7 FTOLV =  1.000D-10 NITER =    20",
                " ITER    FSQR      FSQZ      FSQL      fsqr1      fsqz1      fsql1      DELT0R     R00      W",
                "    0  1.0D-02  2.0D-02  3.0D-02  4.0D-02  5.0D-02  6.0D-02  9.0D-01  1.0D+00  7.0D+00",
                " MHD Energy",
                "  NS =   25 NO. FOURIER MODES =   7 FTOLV =  1.000E-12 NITER =    30",
                " ITER    FSQR      FSQZ      FSQL      fsqr1      fsqz1      fsql1      DELT0R",
                "    1  1.0E-03  2.0E-03  3.0E-03  4.0E-03  5.0E-03  6.0E-03  8.0E-01",
                "",
            ]
        )
    )

    stages = _parse_vmec2000_threed1(threed1)
    assert [stage.ns for stage in stages] == [13, 25]
    assert [stage.niter for stage in stages] == [20, 30]
    rows = flatten_threed1(stages)
    assert [row.it for row in rows] == [0, 1]
    np.testing.assert_allclose(threed1_fsq_total(rows), [0.06, 0.006])
    assert rows[0].r00 == 1.0
    assert rows[0].w == 7.0
    assert rows[1].r00 is None


def test_vmec2000_indata_patch_and_file_discovery(tmp_path: Path) -> None:
    text = "\n".join(
        [
            "&INDATA",
            "  niter = 100",
            "  FTOL_ARRAY = 1.0e-10",
            "/",
            "",
        ]
    )
    patched = _patch_indata(text, updates={"NITER": "5", "NS_ARRAY": "7"})

    assert "NITER = 5" in patched
    assert "FTOL_ARRAY = 1.0e-10" in patched
    assert "NS_ARRAY = 7" in patched
    assert patched.index("NS_ARRAY = 7") < patched.index("/")

    assert _infer_case_name(Path("input.circular_tokamak")) == "circular_tokamak"
    assert _infer_case_name(Path("custom_input")) == "custom_input"

    fallback = tmp_path / "threed1_anything"
    fallback.write_text("")
    assert _find_threed1_file(tmp_path, case="missing") == fallback

    direct = tmp_path / "threed1.case"
    direct.write_text("")
    assert _find_threed1_file(tmp_path, case="case") == direct


def test_vmec2000_exec_discovery_and_fake_run(monkeypatch, tmp_path: Path) -> None:
    import vmec_jax.vmec2000_exec as vx

    env_exec = tmp_path / "xvmec_env"
    env_exec.write_text("#!/bin/sh\n")
    monkeypatch.setenv("VMEC2000_EXEC", str(env_exec))
    assert find_vmec2000_exec(root=tmp_path / "empty") == env_exec

    monkeypatch.setenv("VMEC2000_EXEC", str(tmp_path / "missing"))
    default_exec = tmp_path / "STELLOPT" / "VMEC2000" / "Release" / "xvmec2000"
    default_exec.parent.mkdir(parents=True)
    default_exec.write_text("#!/bin/sh\n")
    assert find_vmec2000_exec(root=tmp_path) == default_exec

    monkeypatch.delenv("VMEC2000_EXEC", raising=False)
    assert find_vmec2000_exec(root=tmp_path / "none" / "child") is None

    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n  NITER = 20\n  MGRID_FILE = 'mgrid.test'\n/\n")
    (tmp_path / "mgrid.test").write_text("synthetic mgrid placeholder")
    workdir = tmp_path / "work"

    with pytest.raises(FileNotFoundError, match="VMEC2000 executable"):
        run_xvmec2000(input_path, exec_path=tmp_path / "no_exec")

    def fake_run(cmd, *, cwd, capture_output, text, timeout, check):
        assert cmd == [str(default_exec), "input.case"]
        assert (Path(cwd) / "mgrid.test").read_text() == "synthetic mgrid placeholder"
        assert capture_output is True
        assert text is True
        assert timeout == 12.0
        assert check is False
        (Path(cwd) / "threed1.case").write_text(
            "\n".join(
                [
                    " NS =  3 NO. FOURIER MODES =  1 FTOLV =  1.0E-10 NITER =  2",
                    " ITER FSQR FSQZ FSQL fsqr1 fsqz1 fsql1 DELT0R",
                    " 1 1.0E-2 2.0E-2 3.0E-2 4.0E-2 5.0E-2 6.0E-2 7.0E-1",
                ]
            )
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    times = iter([10.0, 12.5])
    monkeypatch.setattr(vx.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(vx.subprocess, "run", fake_run)

    result = run_xvmec2000(
        input_path,
        exec_path=default_exec,
        workdir=workdir,
        timeout_s=12.0,
        indata_updates={"NITER": "2"},
        keep_workdir=True,
    )

    assert result.workdir == workdir
    assert result.input_path == workdir / "input.case"
    assert result.returncode == 0
    assert "NITER = 2" in result.input_path.read_text()
    assert result.stdout == "ok"
    assert result.stderr == ""
    assert result.runtime_s == pytest.approx(2.5)
    assert result.threed1_path == workdir / "threed1.case"
    assert len(result.stages) == 1
    assert threed1_fsq_total(flatten_threed1(result.stages))[0] == pytest.approx(0.06)
