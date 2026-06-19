from __future__ import annotations

import json
from pathlib import Path
import site
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.plotting import (
    _case_from_input_path,
    _default_example_outdir,
    _extent_from_grids,
    _pi_label,
    _selected_boozer_surface_label,
    axis_rz_from_wout,
    axis_rz_from_wout_physical,
    boozer_bmag_grid_from_state,
    bmag_from_wout,
    bmag_from_wout_physical,
    bsub_from_wout,
    bsup_from_wout,
    closed_theta_grid,
    fix_matplotlib_3d,
    prepare_matplotlib_3d,
    plot_3d_boundary_comparison,
    plot_bmag_contours,
    plot_boozmn,
    plot_boozmn_bmag_contours,
    plot_boozmn_mode_families,
    plot_boozmn_spectrum,
    plot_boozer_bmag_contours_from_state,
    plot_boozer_lcfs_bmag_comparison,
    plot_objective_history,
    plot_qh_optimization,
    plot_wout,
    profiles_from_wout,
    select_zeta_slices,
    surface_data_from_wout,
    surface_rz_from_wout,
    surface_rz_from_wout_physical,
    surface_stack,
    vmecplot2_bmag_grid,
    vmecplot2_cross_section_indices,
    vmecplot2_lcfs_3d_grid,
    vmecplot2_surface_grid,
    write_axisym_overview,
    write_bmag_parity_figures,
    write_bsub_parity_figures,
    write_bsup_parity_figures,
    zeta_grid,
    zeta_grid_field_period,
)


def _snapshot_mpl_toolkits_modules():
    return {name: module for name, module in sys.modules.items() if name == "mpl_toolkits" or name.startswith("mpl_toolkits.")}


def _restore_mpl_toolkits_modules(snapshot) -> None:
    for name in list(sys.modules):
        if name == "mpl_toolkits" or name.startswith("mpl_toolkits."):
            del sys.modules[name]
    sys.modules.update(snapshot)


def _toy_wout(*, lasym: bool = False):
    ns = 2
    main = np.asarray([0.0, 1.0])
    nyq = np.asarray([0.0, 1.0])
    rmnc = np.asarray([[1.0, 0.0], [1.0, 0.1]])
    rmns = np.asarray([[0.0, 0.0], [0.0, 0.05 if lasym else 0.0]])
    zmns = np.asarray([[0.0, 0.0], [0.0, 0.2]])
    zmnc = np.asarray([[0.0, 0.0], [0.0, 0.07 if lasym else 0.0]])
    bmnc = np.asarray([[2.0, 0.0], [2.0, 0.3]])
    bmns = np.asarray([[0.0, 0.0], [0.0, 0.4]])
    return SimpleNamespace(
        ns=ns,
        nfp=2,
        lasym=lasym,
        xm=main,
        xn=np.asarray([0.0, 0.0]),
        xm_nyq=nyq,
        xn_nyq=np.asarray([0.0, 0.0]),
        rmnc=rmnc,
        rmns=rmns,
        zmns=zmns,
        zmnc=zmnc,
        bmnc=bmnc,
        bmns=bmns,
        bsupumnc=bmnc + 1.0,
        bsupumns=bmns + 0.2,
        bsupvmnc=bmnc + 2.0,
        bsupvmns=bmns + 0.3,
        bsubumnc=bmnc + 3.0,
        bsubumns=bmns + 0.4,
        bsubvmnc=bmnc + 4.0,
        bsubvmns=bmns + 0.5,
        iotaf=np.asarray([0.0, 0.4]),
        iotas=np.asarray([0.0, 0.35]),
        presf=np.asarray([1.0, 0.0]),
        pres=np.asarray([0.9, 0.1]),
        raxis_cc=np.asarray([1.0, 0.1]),
        raxis_cs=np.asarray([0.0, 0.2]),
        zaxis_cs=np.asarray([0.0, 0.3]),
        zaxis_cc=np.asarray([0.0, 0.4]),
    )


def _toy_wout_with_flux(*, lasym: bool = False):
    wout = _toy_wout(lasym=lasym)
    wout.ntor = 1
    wout.phi = np.linspace(0.0, 1.0, int(wout.ns))
    wout.phips = 1.0
    wout.phipf = np.asarray([1.0, 1.0])
    wout.chipf = np.asarray([0.2, 0.2])
    wout.signgs = 1
    wout.buco = np.asarray([0.0, 0.1])
    wout.bvco = np.asarray([0.0, 0.2])
    wout.jcuru = np.asarray([0.0, 0.3])
    wout.jcurv = np.asarray([0.0, 0.4])
    wout.DMerc = np.asarray([0.0, 0.5])
    return wout


def test_grids_slices_and_path_helpers():
    np.testing.assert_allclose(closed_theta_grid(3), [0.0, np.pi, 2.0 * np.pi])
    np.testing.assert_allclose(zeta_grid(3, endpoint=True), [0.0, np.pi, 2.0 * np.pi])
    np.testing.assert_allclose(zeta_grid_field_period(3, nfp=2), [0.0, np.pi / 3.0, 2.0 * np.pi / 3.0])
    np.testing.assert_array_equal(vmecplot2_cross_section_indices(8), [0, 2, 4, 6])
    with pytest.raises(ValueError, match="nzeta>=8"):
        vmecplot2_cross_section_indices(6)
    with pytest.raises(ValueError, match="positive"):
        select_zeta_slices(np.arange(4), n=0)

    zeta = np.asarray([0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(select_zeta_slices(zeta, n=3), [0.0, 1.0, 1.5])
    assert _case_from_input_path("/tmp/input.nfp4_QH") == "nfp4_QH"
    assert _case_from_input_path("/tmp/wout_test.nc") == "wout_test"
    assert _default_example_outdir("sub", "case", "/tmp/out") == Path("/tmp/out")
    assert _extent_from_grids(np.asarray([2.0]), np.asarray([3.0])) == (2.5, 3.5, 1.5, 2.5)
    assert [_pi_label(v) for v in (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)] == ["0", "π/2", "π", "3π/2"]
    assert _selected_boozer_surface_label((0.25, 0.5, 1.0), -2) == "mid radius"
    assert _selected_boozer_surface_label((0.25, 0.5, 1.0), -1) == "plasma boundary"
    assert _selected_boozer_surface_label((0.25,), 0) == "Boozer surface s=0.25"


def test_wout_surface_and_field_helpers_respect_lasym():
    theta = np.asarray([0.0, 0.5 * np.pi])
    zeta = np.asarray([0.0, 0.3])
    wout = _toy_wout(lasym=False)

    R, Z = surface_rz_from_wout(wout, theta=theta, zeta=zeta, s_index=1)
    np.testing.assert_allclose(R[:, 0], [1.1, 1.0])
    np.testing.assert_allclose(Z[:, 0], [0.0, 0.2])
    B = bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=1)
    np.testing.assert_allclose(B[:, 0], [2.3, 2.0])

    wout_asym = _toy_wout(lasym=True)
    R_asym, Z_asym = surface_rz_from_wout(wout_asym, theta=theta, zeta=zeta, s_index=1)
    np.testing.assert_allclose(R_asym[:, 0], [1.1, 1.05])
    np.testing.assert_allclose(Z_asym[:, 0], [0.07, 0.2])
    B_asym = bmag_from_wout(wout_asym, theta=theta, zeta=zeta, s_index=1)
    np.testing.assert_allclose(B_asym[:, 0], [2.3, 2.4])

    bsupu, bsupv = bsup_from_wout(wout_asym, theta=theta, zeta=zeta, s_index=1)
    bsubu, bsubv = bsub_from_wout(wout_asym, theta=theta, zeta=zeta, s_index=1)
    assert bsupu.shape == bsupv.shape == bsubu.shape == bsubv.shape == (2, 2)
    assert float(bsupu[1, 0]) > float(bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=1)[1, 0])


def test_physical_angle_helpers_match_field_period_helpers_for_axisymmetric_data():
    wout = _toy_wout(lasym=True)
    theta = np.asarray([0.0, 0.25 * np.pi, 0.5 * np.pi])
    phi = np.asarray([0.0, 0.2])
    zeta = phi * float(wout.nfp)

    R_phys, Z_phys = surface_rz_from_wout_physical(wout, theta=theta, phi=phi, s_index=1)
    R_zeta, Z_zeta = surface_rz_from_wout(wout, theta=theta, zeta=zeta, s_index=1)
    np.testing.assert_allclose(R_phys, R_zeta)
    np.testing.assert_allclose(Z_phys, Z_zeta)

    B_phys = bmag_from_wout_physical(wout, theta=theta, phi=phi, s_index=1)
    B_zeta = bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=1)
    np.testing.assert_allclose(B_phys, B_zeta)


def test_axis_profiles_surface_stack_and_surface_data():
    wout = _toy_wout(lasym=True)
    zeta = np.asarray([0.0, np.pi / 4.0])
    R_axis, Z_axis = axis_rz_from_wout(wout, zeta=zeta)
    R_axis_phys, Z_axis_phys = axis_rz_from_wout_physical(wout, phi=zeta)
    np.testing.assert_allclose(R_axis, R_axis_phys)
    np.testing.assert_allclose(Z_axis, Z_axis_phys)
    np.testing.assert_allclose(R_axis, [1.1, 0.8])
    np.testing.assert_allclose(Z_axis, [0.4, -0.3])

    fallback = SimpleNamespace(rmnc=np.asarray([[1.25]]))
    np.testing.assert_allclose(axis_rz_from_wout(fallback, zeta=zeta)[0], [1.25, 1.25])

    profiles = profiles_from_wout(wout)
    assert set(profiles) == {"s", "s_half", "iotaf", "iotas", "presf", "pres", "buco", "bvco", "jcuru", "jcurv"}
    np.testing.assert_allclose(profiles["s"], [0.0, 1.0])
    np.testing.assert_allclose(profiles["s_half"], [0.5])
    np.testing.assert_allclose(profiles["buco"], [0.0, 0.0])

    theta = np.asarray([0.0, np.pi / 2.0])
    R_stack, Z_stack = surface_stack(wout, theta=theta, zeta_list=[0.0, 0.1, 0.2], s_index=1)
    assert R_stack.shape == Z_stack.shape == (2, 3)
    data = surface_data_from_wout(wout, theta=theta, zeta=np.asarray([0.0]), s_index=1, with_bmag=True)
    assert data.R.shape == data.Z.shape == data.B.shape == (2, 1)
    assert surface_data_from_wout(wout, theta=theta, zeta=np.asarray([0.0]), s_index=1).B is None


def test_vmecplot2_grid_helpers_return_vmecplot2_shapes():
    wout = _toy_wout(lasym=True)
    theta, zeta, B = vmecplot2_bmag_grid(wout, s_index=1, ntheta=4, nzeta=5, zeta_max=np.pi)
    assert theta.shape == (4,)
    assert zeta.shape == (5,)
    assert B.shape == (4, 5)

    theta_s, zeta_s, R_s, Z_s = vmecplot2_surface_grid(wout, s_index=1, ntheta=6, nzeta=3)
    assert theta_s.shape == (6,)
    assert zeta_s.shape == (3,)
    assert R_s.shape == Z_s.shape == (6, 3)

    theta_3d, phi_3d, R_3d, Z_3d, B_3d = vmecplot2_lcfs_3d_grid(wout, s_index=1, ntheta=5, nzeta=7)
    assert theta_3d.shape == (5,)
    assert phi_3d.shape == (7,)
    assert R_3d.shape == Z_3d.shape == B_3d.shape == (5, 7)


def test_public_optimization_plot_helpers_render_synthetic_outputs(tmp_path):
    pytest.importorskip("matplotlib")

    initial = _toy_wout(lasym=False)
    final = _toy_wout(lasym=True)
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps(
            {
                "label": "Synthetic optimization",
                "target_aspect": 5.0,
                "target_iota": 0.42,
                "total_wall_time_s": 1.5,
                "nfev": 3,
                "history": [
                    {"objective": 3.0, "qs_objective": 2.0, "aspect": 6.0, "iota": 0.1},
                    {"objective": 2.0, "qs_objective": 1.5, "aspect": 5.5, "iota": 0.3},
                    {"objective": 1.0, "qs_objective": 0.5, "aspect": 5.0, "iota": 0.42},
                ],
            }
        )
    )

    boundary_path = plot_3d_boundary_comparison(initial, final, outdir=tmp_path)
    bmag_path = plot_bmag_contours(initial, final, outdir=tmp_path)
    history_plot_path = plot_objective_history(history_path, outdir=tmp_path)

    assert boundary_path.name == "boundary_comparison.png"
    assert bmag_path.name == "bmag_surface.png"
    assert history_plot_path.name == "objective_history.png"
    assert boundary_path.stat().st_size > 0
    assert bmag_path.stat().st_size > 0
    assert history_plot_path.stat().st_size > 0


def test_boozer_bmag_contour_helper_uses_line_contours(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import vmec_jax.booz_input as booz_input

    fake_booz_module = ModuleType("booz_xform_jax")
    fake_booz_module.prepare_booz_xform_constants_from_inputs = (
        lambda *, inputs, mboz, nboz, asym: ("constants", "grids")
    )
    fake_booz_module.booz_xform_from_inputs = lambda **_kwargs: {
        "bmnc_b": np.asarray([[1.0, 0.2]], dtype=float),
        "bmns_b": np.asarray([[0.0, 0.1]], dtype=float),
        "ixm_b": np.asarray([0, 1], dtype=int),
        "ixn_b": np.asarray([0, 0], dtype=int),
        "nfp_b": np.asarray(1, dtype=int),
        "iota_b": np.asarray([0.5], dtype=float),
    }
    monkeypatch.setitem(sys.modules, "booz_xform_jax", fake_booz_module)
    monkeypatch.setattr(
        booz_input,
        "booz_xform_inputs_from_state",
        lambda **_kwargs: SimpleNamespace(rmnc=np.zeros((3, 2)), nfp=1),
    )

    static = SimpleNamespace(cfg=SimpleNamespace(lasym=False, nfp=1))
    theta, phi, bmag, _booz = boozer_bmag_grid_from_state(
        object(),
        static=static,
        indata=object(),
        signgs=1,
        ntheta=8,
        nphi=9,
    )
    assert theta.shape == (8,)
    assert phi.shape == (9,)
    assert bmag.shape == (8, 9)
    assert np.ptp(bmag) > 0.0

    out = plot_boozer_bmag_contours_from_state(
        object(),
        static=static,
        indata=object(),
        signgs=1,
        outdir=tmp_path,
        ntheta=8,
        nphi=9,
    )
    assert out.name == "boozer_bmag_surface.png"
    assert out.stat().st_size > 0


def test_boozmn_plot_helpers_render_synthetic_boozer_output(tmp_path):
    pytest.importorskip("matplotlib")
    import inspect
    import vmec_jax.plotting as plotting

    booz = SimpleNamespace(
        nfp=2,
        s_b=np.asarray([0.25, 0.5, 1.0]),
        xm_b=np.asarray([0, 1, 1, 2, 0]),
        xn_b=np.asarray([0, 0, 2, -4, 2]),
        bmnc_b=np.asarray(
            [
                [1.0, 1.0, 1.0],
                [0.1, 0.08, 0.05],
                [0.03, 0.02, 0.01],
                [0.02, 0.03, 0.04],
                [0.01, 0.01, 0.02],
            ]
        ),
        bmns_b=np.zeros((5, 3)),
    )

    bmag = plot_boozmn_bmag_contours(booz, outdir=tmp_path, name="toy", ntheta=16, nphi=18)
    families = plot_boozmn_mode_families(booz, outdir=tmp_path, name="toy")
    spectrum = plot_boozmn_spectrum(booz, outdir=tmp_path, name="toy")

    assert bmag.name == "toy_BoozerB_contours.png"
    assert families.name == "toy_Boozer_mode_families.png"
    assert spectrum.name == "toy_Boozer_lcfs_spectrum.png"
    for path in (bmag, families, spectrum):
        assert path.stat().st_size > 0
    assert plotting._booz_surface_label(booz, 1, outer=False) == "mid radius"
    assert plotting._booz_surface_label(booz, 2, outer=True) == "plasma boundary"
    source = inspect.getsource(plotting._plot_boozer_bmag_axis)
    assert "_BOOZER_PHI_LABEL" in source
    assert "_BOOZER_THETA_LABEL" in source
    assert r"$\phi_{B}$" == plotting._BOOZER_PHI_LABEL.split("angle ", 1)[1].removesuffix(" (rad)")
    assert r"$\theta_{B}$" == plotting._BOOZER_THETA_LABEL.split("angle ", 1)[1].removesuffix(" (rad)")


def test_boozer_lcfs_comparison_runs_booz_and_labels_boozer_axes(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import vmec_jax.plotting as plotting

    calls: list[tuple[Path, Path | None, tuple[float, ...] | None]] = []
    initial_wout = tmp_path / "wout_initial.nc"
    final_wout = tmp_path / "wout_final.nc"
    initial_wout.write_text("initial")
    final_wout.write_text("final")

    def fake_run_booz_xform(wout_path, *, output_path, surfaces, **_kwargs):
        output_path = Path(output_path)
        output_path.write_text("booz")
        calls.append((Path(wout_path), output_path, surfaces))
        return output_path

    def fake_load_booz(path):
        factor = 0.1 if "initial" in Path(path).name else 0.2
        return SimpleNamespace(
            nfp=2,
            s_b=np.asarray([1.0]),
            xm_b=np.asarray([0, 1, 0]),
            xn_b=np.asarray([0, 0, 2]),
            bmnc_b=np.asarray([[1.0], [factor], [0.03]]),
            bmns_b=np.zeros((3, 1)),
        )

    monkeypatch.setattr(plotting, "run_booz_xform", fake_run_booz_xform)
    monkeypatch.setattr(plotting, "_load_booz_if_path", fake_load_booz)

    out = plot_boozer_lcfs_bmag_comparison(
        initial_wout,
        final_wout,
        outdir=tmp_path / "plots",
        ntheta=16,
        nphi=18,
    )

    assert out.name == "boozer_lcfs_bmag_comparison.png"
    assert out.stat().st_size > 0
    assert [call[0] for call in calls] == [initial_wout, final_wout]
    assert all(call[2] == (1.0,) for call in calls)


def test_plot_boozmn_dispatches_all_public_boozer_helpers(tmp_path, monkeypatch):
    import vmec_jax.plotting as plotting

    calls: list[tuple[str, Path, Path, str | None]] = []
    source = tmp_path / "boozmn_case.nc"
    source.write_text("placeholder")

    def fake(name: str):
        def _impl(path, *, outdir, name=None, **_kwargs):
            calls.append((name or "", Path(path), Path(outdir), name))
            out = Path(outdir) / f"{name or 'case'}_{fake.__name__}_{len(calls)}.png"
            out.write_text("plot")
            return out

        return _impl

    monkeypatch.setattr(plotting, "plot_boozmn_bmag_contours", fake("bmag"))
    monkeypatch.setattr(plotting, "plot_boozmn_mode_families", fake("families"))
    monkeypatch.setattr(plotting, "plot_boozmn_spectrum", fake("spectrum"))

    paths = plot_boozmn(source, outdir=tmp_path / "plots", name="custom")

    assert set(paths) == {"bmag_contours", "mode_families", "lcfs_spectrum"}
    assert [call[1] for call in calls] == [source, source, source]
    assert all(call[2] == tmp_path / "plots" for call in calls)
    assert all(call[3] == "custom" for call in calls)


def test_plot_qh_optimization_wrapper_dispatches_to_public_helpers(tmp_path, monkeypatch, capsys):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    import vmec_jax.plotting as plotting

    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps({"history": [{"objective": 1.0, "aspect": 5.0}], "label": "toy"}))
    calls = []

    def fake_boundary(wout_initial, wout_final, *, outdir):
        calls.append(("boundary", Path(wout_initial), Path(wout_final), Path(outdir)))
        return Path(outdir) / "boundary.png"

    def fake_bmag(wout_initial, wout_final, *, outdir):
        calls.append(("bmag", Path(wout_initial), Path(wout_final), Path(outdir)))
        return Path(outdir) / "bmag.png"

    def fake_history(path, *, outdir):
        calls.append(("history", Path(path), Path(outdir)))
        return Path(outdir) / "history.png"

    monkeypatch.setattr(plotting, "plot_3d_boundary_comparison", fake_boundary)
    monkeypatch.setattr(plotting, "plot_bmag_contours", fake_bmag)
    monkeypatch.setattr(plotting, "plot_objective_history", fake_history)
    monkeypatch.setattr(plotting, "prepare_matplotlib_3d", lambda: calls.append(("prepare",)))
    monkeypatch.setattr(plt, "show", lambda: calls.append(("show",)))

    outdir = tmp_path / "plots"
    paths = plot_qh_optimization("wout_initial.nc", "wout_final.nc", history_path, outdir=outdir, show=True)

    assert paths == {
        "boundary_comparison": outdir / "boundary.png",
        "bmag_surface": outdir / "bmag.png",
        "objective_history": outdir / "history.png",
    }
    assert calls[:3] == [
        ("boundary", Path("wout_initial.nc"), Path("wout_final.nc"), outdir),
        ("bmag", Path("wout_initial.nc"), Path("wout_final.nc"), outdir),
        ("history", history_path, outdir),
    ]
    assert calls[-2:] == [("prepare",), ("show",)]
    assert "Saved" in capsys.readouterr().out


def test_axisym_and_parity_plot_writers_render_with_synthetic_wout(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import vmec_jax.plotting as plotting

    wout = _toy_wout_with_flux(lasym=True)
    theta = np.asarray([0.0, np.pi / 2.0])
    zeta = np.asarray([0.0, 0.2])
    static = SimpleNamespace(
        grid=SimpleNamespace(theta=theta, zeta=zeta),
        cfg=SimpleNamespace(nfp=wout.nfp),
        s=np.asarray([0.0, 1.0]),
    )

    monkeypatch.setattr(plotting, "example_paths", lambda case: (Path(f"input.{case}"), Path(f"wout_{case}.nc")))
    monkeypatch.setattr(plotting, "read_wout", lambda path: wout)
    monkeypatch.setattr(plotting, "load_config", lambda path: ("cfg", "indata"))
    monkeypatch.setattr(plotting, "build_static", lambda cfg: static)
    monkeypatch.setattr(plotting, "state_from_wout", lambda loaded_wout: "state")
    monkeypatch.setattr(
        plotting,
        "bmag_from_state_physical",
        lambda *args, **kwargs: np.ones((len(theta), len(zeta))) * 3.0,
    )
    monkeypatch.setattr(
        plotting,
        "eval_geom",
        lambda state, static_arg: SimpleNamespace(dummy=True),
    )
    monkeypatch.setattr(plotting, "lamscale_from_phips", lambda phips, s: 1.0)
    monkeypatch.setattr(
        plotting,
        "bsup_from_geom",
        lambda *args, **kwargs: (
            np.ones((2, len(theta), len(zeta))) * 4.0,
            np.ones((2, len(theta), len(zeta))) * 5.0,
        ),
    )
    monkeypatch.setattr(
        plotting,
        "bsub_from_bsup",
        lambda geom, bsupu, bsupv: (np.asarray(bsupu) + 1.0, np.asarray(bsupv) + 1.0),
    )

    overview = write_axisym_overview("toy", outdir=tmp_path / "overview")
    bmag = write_bmag_parity_figures(input_path=tmp_path / "input.toy", wout_path=tmp_path / "wout_toy.nc", outdir=tmp_path / "bmag")
    bsup = write_bsup_parity_figures(input_path=tmp_path / "input.toy", wout_path=tmp_path / "wout_toy.nc", outdir=tmp_path / "bsup")
    bsub = write_bsub_parity_figures(input_path=tmp_path / "input.toy", wout_path=tmp_path / "wout_toy.nc", outdir=tmp_path / "bsub")

    for path in (overview, bmag, bsup, bsub):
        assert path.exists()
        assert path.stat().st_size > 0


def test_axisym_overview_requires_reference_wout(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    import vmec_jax.plotting as plotting

    monkeypatch.setattr(plotting, "example_paths", lambda case: (Path(f"input.{case}"), None))
    with pytest.raises(FileNotFoundError, match="Reference wout"):
        write_axisym_overview("missing", outdir=tmp_path)


def test_plot_wout_renders_cli_diagnostic_outputs(monkeypatch, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    import inspect
    import vmec_jax.plotting as plotting
    import vmec_jax.wout as wout_module

    monkeypatch.setattr(wout_module, "read_wout", lambda path: _toy_wout_with_flux(lasym=True))

    paths = plot_wout(tmp_path / "wout_toy_case.nc", outdir=tmp_path / "plots", name="toy_case")

    assert set(paths) == {"vmec_params", "poloidal_plot", "vmec_surfaces", "3d_plot"}
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
    assert "[vmec_params]" in capsys.readouterr().out
    source = inspect.getsource(plotting.plot_wout)
    assert "Mid radius |B|" in source
    assert "Plasma boundary |B|" in source
    assert "1-based idx" not in source


def test_fix_matplotlib_3d_sets_equal_radius_limits():
    class _Axis:
        def __init__(self):
            self.xlim = (0.0, 2.0)
            self.ylim = (-2.0, 2.0)
            self.zlim = (10.0, 11.0)

        def get_xlim3d(self):
            return self.xlim

        def get_ylim3d(self):
            return self.ylim

        def get_zlim3d(self):
            return self.zlim

        def set_xlim3d(self, value):
            self.xlim = tuple(value)

        def set_ylim3d(self, value):
            self.ylim = tuple(value)

        def set_zlim3d(self, value):
            self.zlim = tuple(value)

    axis = _Axis()
    fix_matplotlib_3d(axis)
    assert axis.xlim == (-1.0, 3.0)
    assert axis.ylim == (-2.0, 2.0)
    assert axis.zlim == (8.5, 12.5)


def test_prepare_matplotlib_3d_replaces_mixed_system_toolkit(tmp_path, monkeypatch):
    snapshot = _snapshot_mpl_toolkits_modules()
    try:
        site_root = tmp_path / "site"
        toolkit = site_root / "mpl_toolkits"
        (toolkit / "mplot3d").mkdir(parents=True)
        (toolkit / "__init__.py").write_text("")
        (toolkit / "mplot3d" / "__init__.py").write_text("from .axes3d import Axes3D\n")
        (toolkit / "mplot3d" / "axes3d.py").write_text("class Axes3D:\n    name = '3d'\n")
        monkeypatch.setattr(site, "getusersitepackages", lambda: str(site_root))
        monkeypatch.setattr(site, "getsitepackages", lambda: [])

        system_toolkit = ModuleType("mpl_toolkits")
        system_toolkit.__file__ = "/usr/lib/python3/dist-packages/mpl_toolkits/__init__.py"
        system_toolkit.__path__ = ["/usr/lib/python3/dist-packages/mpl_toolkits"]
        monkeypatch.setitem(sys.modules, "mpl_toolkits", system_toolkit)

        prepare_matplotlib_3d()

        prepared = sys.modules["mpl_toolkits"]
        prepared_paths = list(prepared.__path__)
        assert all("/usr/lib/python3/dist-packages" not in path for path in prepared_paths)
        assert any((Path(path) / "mplot3d" / "axes3d.py").exists() for path in prepared_paths)
    finally:
        _restore_mpl_toolkits_modules(snapshot)


def test_prepare_matplotlib_3d_replaces_broken_user_toolkit(tmp_path, monkeypatch):
    snapshot = _snapshot_mpl_toolkits_modules()
    try:
        broken = tmp_path / "broken" / "mpl_toolkits"
        broken.mkdir(parents=True)
        (broken / "__init__.py").write_text("")
        good_site = tmp_path / "good_site"
        toolkit = good_site / "mpl_toolkits"
        (toolkit / "mplot3d").mkdir(parents=True)
        (toolkit / "__init__.py").write_text("")
        (toolkit / "mplot3d" / "__init__.py").write_text("from .axes3d import Axes3D\n")
        (toolkit / "mplot3d" / "axes3d.py").write_text("class Axes3D:\n    name = '3d'\n")
        monkeypatch.setattr(site, "getusersitepackages", lambda: str(good_site))
        monkeypatch.setattr(site, "getsitepackages", lambda: [])

        user_toolkit = ModuleType("mpl_toolkits")
        user_toolkit.__file__ = str(broken / "__init__.py")
        user_toolkit.__path__ = [str(broken)]
        monkeypatch.setitem(sys.modules, "mpl_toolkits", user_toolkit)

        prepare_matplotlib_3d()

        prepared = sys.modules["mpl_toolkits"]
        prepared_paths = list(prepared.__path__)
        assert str(broken) not in prepared_paths
        assert any((Path(path) / "mplot3d" / "axes3d.py").exists() for path in prepared_paths)
    finally:
        _restore_mpl_toolkits_modules(snapshot)
