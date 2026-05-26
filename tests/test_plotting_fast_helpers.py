from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.plotting import (
    _best_so_far_stage_segments,
    _default_example_outdir,
    _is_tracer,
    _lcfs_xyz,
    _line_contour_levels,
    _load_wout_if_path,
    _mode_table_from_wout,
    _objective_iota_series,
    axis_rz_from_state_physical,
    axis_rz_from_wout_physical,
    bmag_from_state_physical,
    bmag_from_state_vmec_realspace,
    bmag_from_wout_physical,
    boozer_bmag_grid_from_state,
    bsub_from_wout,
    bsup_from_wout,
    plot_3d_boundary_comparison,
    plot_bmag_contours,
    plot_objective_history,
    surface_rz_from_state_physical,
    surface_rz_from_wout,
    surface_rz_from_wout_physical,
    vmecplot2_bmag_grid,
)
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data"


def _small_circular_state_static():
    data_dir = _data_dir()
    wout_path = data_dir / "wout_circular_tokamak.nc"
    if not wout_path.exists():
        pytest.skip("Optional WOUT fixtures are missing. Run tools/fetch_assets.py --bundle wout-fixtures.")
    cfg, indata = load_config(data_dir / "input.circular_tokamak")
    wout = read_wout(wout_path)
    cfg = replace(
        cfg,
        ns=int(wout.ns),
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
        lthreed=bool(int(wout.ntor) > 0),
        ntheta=8,
        nzeta=1,
    )
    return state_from_wout(wout), build_static(cfg), indata, wout


def test_surface_rz_from_wout_respects_lasym_geometry_parity() -> None:
    theta = np.asarray([0.5 * np.pi])
    zeta = np.asarray([0.0])
    common = dict(
        nfp=1,
        xm=np.asarray([0, 1]),
        xn=np.asarray([0, 0]),
        xm_nyq=np.asarray([0, 1]),
        xn_nyq=np.asarray([0, 0]),
        rmnc=np.asarray([[1.0, 2.0]]),
        rmns=np.asarray([[10.0, 20.0]]),
        zmnc=np.asarray([[100.0, 200.0]]),
        zmns=np.asarray([[0.0, 3.0]]),
    )

    symmetric = SimpleNamespace(**common, lasym=False)
    R_sym, Z_sym = surface_rz_from_wout(symmetric, theta=theta, zeta=zeta, s_index=0)
    np.testing.assert_allclose(R_sym, [[1.0]])
    np.testing.assert_allclose(Z_sym, [[3.0]])

    asymmetric = SimpleNamespace(**common, lasym=True)
    R_asym, Z_asym = surface_rz_from_wout(asymmetric, theta=theta, zeta=zeta, s_index=0)
    np.testing.assert_allclose(R_asym, [[21.0]])
    np.testing.assert_allclose(Z_asym, [[103.0]])


def test_surface_rz_from_wout_physical_uses_vmec_xn_scaling() -> None:
    wout = SimpleNamespace(
        nfp=2,
        lasym=False,
        xm=np.asarray([0, 0]),
        xn=np.asarray([0, 2]),
        xm_nyq=np.asarray([0, 0]),
        xn_nyq=np.asarray([0, 2]),
        rmnc=np.asarray([[1.0, 2.0]]),
        rmns=np.asarray([[0.0, 0.0]]),
        zmnc=np.asarray([[0.0, 0.0]]),
        zmns=np.asarray([[0.0, 0.0]]),
    )

    R, Z = surface_rz_from_wout_physical(
        wout,
        theta=np.asarray([0.0]),
        phi=np.asarray([0.25 * np.pi]),
        s_index=0,
    )

    np.testing.assert_allclose(R, [[1.0]], atol=1e-14)
    np.testing.assert_allclose(Z, [[0.0]], atol=1e-14)
    np.testing.assert_array_equal(_mode_table_from_wout(wout, nyq=False, physical=False).n, [0, 1])
    np.testing.assert_array_equal(_mode_table_from_wout(wout, nyq=False, physical=True).n, [0, 2])


def test_line_contour_levels_handle_constant_and_nonfinite_fields() -> None:
    varying = _line_contour_levels(np.asarray([[1.0, 2.0], [3.0, 4.0]]), count=4)
    np.testing.assert_allclose(varying, [1.0, 2.0, 3.0, 4.0])

    constant = _line_contour_levels(np.full((2, 2), 2.0), count=3)
    assert constant[0] < 2.0 < constant[-1]
    np.testing.assert_allclose(constant[1], 2.0)

    fallback = _line_contour_levels(np.asarray([np.nan, np.inf]), count=3)
    np.testing.assert_allclose(fallback, [0.0, 0.5, 1.0])


def test_objective_history_series_helpers_preserve_stage_boundaries_and_missing_iota() -> None:
    segments = _best_so_far_stage_segments([4.0, 0.0, 3.0, 2.0], [1, 1, 99])

    assert len(segments) == 2
    np.testing.assert_array_equal(segments[0][0], [0, 1])
    np.testing.assert_allclose(segments[0][1], [4.0, 1e-16])
    np.testing.assert_array_equal(segments[1][0], [2, 3])
    np.testing.assert_allclose(segments[1][1], [3.0, 2.0])

    assert _objective_iota_series([]) is None
    assert _objective_iota_series([{"objective": 1.0}]) is None
    iota_series = _objective_iota_series([{"objective": 1.0}, {"objective": 0.5, "iota": 0.2}])
    assert iota_series is not None
    assert np.isnan(iota_series[0])
    assert iota_series[1] == 0.2


def test_load_wout_if_path_dispatches_only_for_paths_and_lcfs_xyz(monkeypatch, tmp_path) -> None:
    import vmec_jax.wout as wout_module

    loaded = SimpleNamespace(kind="loaded")
    monkeypatch.setattr(wout_module, "read_wout", lambda path: loaded)

    assert _load_wout_if_path(tmp_path / "wout_toy.nc") is loaded
    existing = SimpleNamespace(kind="already_loaded")
    assert _load_wout_if_path(existing) is existing

    R = np.asarray([[1.0, 2.0]])
    Z = np.asarray([[3.0, 4.0]])
    phi = np.asarray([0.0, 0.5 * np.pi])
    X, Y, Z_out = _lcfs_xyz(R, Z, phi)
    np.testing.assert_allclose(X, [[1.0, 0.0]], atol=1e-14)
    np.testing.assert_allclose(Y, [[0.0, 2.0]], atol=1e-14)
    assert Z_out is Z


def test_symmetric_field_helpers_ignore_asymmetric_sine_channels() -> None:
    theta = np.asarray([0.5 * np.pi])
    phi = np.asarray([0.0])
    wout = SimpleNamespace(
        nfp=1,
        lasym=False,
        xm=np.asarray([0, 1]),
        xn=np.asarray([0, 0]),
        xm_nyq=np.asarray([0, 1]),
        xn_nyq=np.asarray([0, 0]),
        rmnc=np.asarray([[1.0, 0.0]]),
        rmns=np.asarray([[0.0, 0.0]]),
        zmnc=np.asarray([[0.0, 0.0]]),
        zmns=np.asarray([[0.0, 0.0]]),
        bmnc=np.asarray([[2.0, 0.0]]),
        bmns=np.asarray([[100.0, 100.0]]),
        bsupumnc=np.asarray([[3.0, 0.0]]),
        bsupumns=np.asarray([[100.0, 100.0]]),
        bsupvmnc=np.asarray([[4.0, 0.0]]),
        bsupvmns=np.asarray([[100.0, 100.0]]),
        bsubumnc=np.asarray([[5.0, 0.0]]),
        bsubumns=np.asarray([[100.0, 100.0]]),
        bsubvmnc=np.asarray([[6.0, 0.0]]),
        bsubvmns=np.asarray([[100.0, 100.0]]),
    )

    np.testing.assert_allclose(bmag_from_wout_physical(wout, theta=theta, phi=phi, s_index=0), [[2.0]])
    bsupu, bsupv = bsup_from_wout(wout, theta=theta, zeta=phi, s_index=0)
    bsubu, bsubv = bsub_from_wout(wout, theta=theta, zeta=phi, s_index=0)
    np.testing.assert_allclose(bsupu, [[3.0]])
    np.testing.assert_allclose(bsupv, [[4.0]])
    np.testing.assert_allclose(bsubu, [[5.0]])
    np.testing.assert_allclose(bsubv, [[6.0]])


def test_vmecplot2_and_axis_fallback_defaults() -> None:
    wout = SimpleNamespace(
        nfp=2,
        lasym=False,
        xm_nyq=np.asarray([0.0]),
        xn_nyq=np.asarray([0.0]),
        bmnc=np.asarray([[7.0]]),
        bmns=np.asarray([[100.0]]),
        rmnc=np.asarray([[1.25]]),
    )

    theta, zeta, B = vmecplot2_bmag_grid(wout, s_index=0, ntheta=3, nzeta=4)
    np.testing.assert_allclose(theta, [0.0, np.pi, 2.0 * np.pi])
    np.testing.assert_allclose(zeta, [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0, 2.0 * np.pi])
    np.testing.assert_allclose(B, np.full((3, 4), 7.0))

    R_axis, Z_axis = axis_rz_from_wout_physical(wout, phi=np.asarray([0.0, 0.5]))
    np.testing.assert_allclose(R_axis, [1.25, 1.25])
    np.testing.assert_allclose(Z_axis, [0.0, 0.0])
    assert _default_example_outdir("subdir", "case", None).parts[-4:] == ("examples", "outputs", "subdir", "case")
    assert _is_tracer(object()) is False


def test_bmag_from_state_paths_require_flux_source_and_return_positive_fields() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    state, static, indata, wout = _small_circular_state_static()

    theta = np.asarray(static.grid.theta, dtype=float)
    phi = np.asarray(static.grid.zeta, dtype=float) / float(static.cfg.nfp)
    s_index = int(wout.ns) - 1

    with pytest.raises(ValueError, match="indata must be provided"):
        bmag_from_state_physical(
            state,
            static,
            theta=theta,
            phi=phi,
            s_index=s_index,
            signgs=int(wout.signgs),
        )
    with pytest.raises(ValueError, match="indata must be provided"):
        bmag_from_state_vmec_realspace(
            state,
            static,
            s_index=s_index,
            signgs=int(wout.signgs),
        )

    b_from_indata = bmag_from_state_physical(
        state,
        static,
        indata,
        theta=theta,
        phi=phi,
        s_index=s_index,
        signgs=int(wout.signgs),
    )
    b_with_explicit_flux = bmag_from_state_physical(
        state,
        static,
        theta=theta,
        phi=phi,
        s_index=s_index,
        signgs=int(wout.signgs),
        phipf=np.asarray(wout.phipf),
        chipf=np.asarray(wout.chipf),
        lamscale=float(getattr(wout, "lamscale", 1.0)),
        flux_is_internal=True,
        sqrtg_floor=1.0e-12,
        bmag_floor=1.0e-30,
    )
    b_with_derived_flux_scaling = bmag_from_state_physical(
        state,
        static,
        theta=theta,
        phi=phi,
        s_index=s_index,
        signgs=int(wout.signgs),
        phipf=np.asarray(wout.phipf),
        chipf=np.asarray(wout.chipf),
    )
    b_vmec_realspace = bmag_from_state_vmec_realspace(
        state,
        static,
        s_index=s_index,
        signgs=int(wout.signgs),
        phipf=np.asarray(wout.phipf),
        chipf=np.asarray(wout.chipf),
        lamscale=float(getattr(wout, "lamscale", 1.0)),
        flux_is_internal=True,
        sqrtg_floor=1.0e-12,
    )
    b_vmec_realspace_derived_scaling = bmag_from_state_vmec_realspace(
        state,
        static,
        s_index=s_index,
        signgs=int(wout.signgs),
        phipf=np.asarray(wout.phipf),
        chipf=np.asarray(wout.chipf),
    )

    assert b_from_indata.shape == (theta.size, phi.size)
    assert b_with_explicit_flux.shape == b_from_indata.shape
    assert b_with_derived_flux_scaling.shape == b_from_indata.shape
    assert b_vmec_realspace.ndim == 2
    assert b_vmec_realspace_derived_scaling.shape == b_vmec_realspace.shape
    assert np.all(np.isfinite(b_from_indata))
    assert np.all(np.isfinite(b_with_explicit_flux))
    assert np.all(np.isfinite(b_with_derived_flux_scaling))
    assert np.all(np.isfinite(b_vmec_realspace))
    assert np.all(np.isfinite(b_vmec_realspace_derived_scaling))
    assert float(np.min(b_from_indata)) > 0.0
    assert float(np.min(b_vmec_realspace)) > 0.0


def test_state_physical_surface_helpers_scale_phi_by_field_period() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    state, static, _indata, _wout = _small_circular_state_static()
    theta = np.asarray([0.0, np.pi])
    phi = np.asarray([0.0])

    R, Z = surface_rz_from_state_physical(
        state,
        static.modes,
        theta=theta,
        phi=phi,
        s_index=int(static.cfg.ns) - 1,
        nfp=int(static.cfg.nfp),
    )
    R_axis, Z_axis = axis_rz_from_state_physical(state, static.modes, phi=phi, nfp=int(static.cfg.nfp))

    assert R.shape == (2, 1)
    assert Z.shape == (2, 1)
    assert R_axis.shape == (1,)
    assert Z_axis.shape == (1,)
    assert np.all(np.isfinite(R))
    assert np.all(np.isfinite(Z))
    assert np.all(np.isfinite(R_axis))
    assert np.all(np.isfinite(Z_axis))


def test_plot_wrappers_default_outdir_without_rendering(monkeypatch, tmp_path) -> None:
    import vmec_jax.plotting as plotting

    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plotting, "_load_wout_if_path", lambda value: f"loaded:{value}")
    monkeypatch.setattr(
        plotting,
        "_plot_3d_boundary_comparison",
        lambda initial, final, outdir: calls.append(("3d", initial, final, outdir)) or outdir / "3d.png",
    )
    monkeypatch.setattr(
        plotting,
        "_plot_bmag_contours",
        lambda initial, final, outdir: calls.append(("bmag", initial, final, outdir)) or outdir / "bmag.png",
    )
    monkeypatch.setattr(
        plotting,
        "_plot_objective_history",
        lambda history, outdir: calls.append(("history", history, outdir)) or outdir / "history.png",
    )

    assert plot_3d_boundary_comparison("initial.nc", "final.nc") == Path("3d.png")
    assert plot_bmag_contours("initial.nc", "final.nc") == Path("bmag.png")
    history_path = tmp_path / "history.json"
    assert plot_objective_history(history_path) == tmp_path / "history.png"
    assert calls == [
        ("3d", "loaded:initial.nc", "loaded:final.nc", Path(".")),
        ("bmag", "loaded:initial.nc", "loaded:final.nc", Path(".")),
        ("history", history_path, tmp_path),
    ]


def test_boozer_bmag_grid_validates_and_synthesizes_optional_sine_modes(monkeypatch) -> None:
    import sys
    import vmec_jax.booz_input as booz_input
    import vmec_jax.quasi_isodynamic as qi

    calls = []

    def fake_prepare(*, inputs, mboz, nboz, asym):
        calls.append(("prepare", inputs.rmnc.shape, mboz, nboz, asym))
        return "constants", "grids"

    def fake_booz(*, inputs, constants, grids, surface_indices, jit):
        calls.append(("booz", constants, grids, np.asarray(surface_indices), jit))
        return {
            "bmnc_b": np.asarray([[2.0, 0.5], [3.0, 1.0]]),
            "ixm_b": np.asarray([0.0, 1.0]),
            "ixn_b": np.asarray([0.0, 2.0]),
            "nfp_b": np.asarray([2]),
        }

    fake_module = SimpleNamespace(
        prepare_booz_xform_constants_from_inputs=fake_prepare,
        booz_xform_from_inputs=fake_booz,
    )
    monkeypatch.setitem(sys.modules, "booz_xform_jax", fake_module)
    monkeypatch.setattr(
        booz_input,
        "booz_xform_inputs_from_state",
        lambda **_kwargs: SimpleNamespace(rmnc=np.zeros((4, 1))),
    )
    monkeypatch.setattr(qi, "_nearest_half_mesh_indices", lambda surfaces, n_half: np.asarray([0, n_half - 1]))
    static = SimpleNamespace(cfg=SimpleNamespace(lasym=False, nfp=5))

    with pytest.raises(ValueError, match="surfaces must contain at least one value"):
        boozer_bmag_grid_from_state("state", static=static, indata="indata", signgs=1, surfaces=())
    with pytest.raises(ValueError, match="ntheta and nphi must both be at least 4"):
        boozer_bmag_grid_from_state("state", static=static, indata="indata", signgs=1, ntheta=3)

    theta, phi, bmag, booz = boozer_bmag_grid_from_state(
        "state",
        static=static,
        indata="indata",
        signgs=-1,
        surfaces=(0.25, 1.0),
        surface_index=-1,
        mboz=4,
        nboz=5,
        ntheta=4,
        nphi=5,
        phimin=0.25,
        jit_booz=True,
    )

    assert theta.shape == (4,)
    assert phi[0] == pytest.approx(0.25)
    assert phi[-1] == pytest.approx(0.25 + np.pi)
    assert bmag.shape == (4, 5)
    expected = 3.0 + np.cos(theta[:, None] - 2.0 * phi[None, :])
    np.testing.assert_allclose(bmag, expected)
    assert "bmnc_b" in booz
    assert calls[0] == ("prepare", (4, 1), 4, 5, False)
    assert calls[1][0] == "booz"
    np.testing.assert_array_equal(calls[1][3], [0, 3])
    assert calls[1][4] is True

    with pytest.raises(IndexError, match="surface_index 2 is outside Boozer surface range"):
        boozer_bmag_grid_from_state(
            "state",
            static=static,
            indata="indata",
            signgs=1,
            surfaces=(1.0,),
            surface_index=2,
            ntheta=4,
            nphi=4,
        )
