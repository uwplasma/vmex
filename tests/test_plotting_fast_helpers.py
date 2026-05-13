from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vmec_jax.plotting import (
    _best_so_far_stage_segments,
    _default_example_outdir,
    _is_tracer,
    _lcfs_xyz,
    _line_contour_levels,
    _load_wout_if_path,
    _mode_table_from_wout,
    _objective_iota_series,
    axis_rz_from_wout_physical,
    bmag_from_wout_physical,
    bsub_from_wout,
    bsup_from_wout,
    plot_3d_boundary_comparison,
    plot_bmag_contours,
    plot_objective_history,
    surface_rz_from_wout,
    surface_rz_from_wout_physical,
    vmecplot2_bmag_grid,
)


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
