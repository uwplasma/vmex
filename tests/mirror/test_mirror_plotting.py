from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax import cli
from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    mirror_boozer_like_summary_metrics as public_mirror_boozer_like_summary_metrics,
    plot_mirror_output,
    run_mirror_fixed_boundary,
    write_mirror_output,
)
from vmec_jax.mirror.core.boundary import MirrorBoundary
from vmec_jax.mirror.io.mout import load_mirror_output
from vmec_jax.mirror.plotting.bfield import mirror_bfield_boundary_data, mirror_bmag_boundary_data, mirror_bmag_sxi_data
from vmec_jax.mirror.plotting.bfield import mirror_boundary_field_line_data
from vmec_jax.mirror.plotting.diagnostics import (
    mirror_boozer_like_diagnostics_data,
    mirror_boozer_like_summary_metrics,
    mirror_field_line_pitch_profile_data,
    mirror_jacobian_data,
    mirror_pressure_profile_data,
    mirror_radial_diagnostics_data,
    mirror_residual_history_data,
)
from vmec_jax.mirror.plotting.geometry import (
    mirror_boundary_3d_data,
    mirror_cross_sections_data,
    mirror_surfaces_rz_data,
)

pytestmark = pytest.mark.mirror


def _mout_file(tmp_path: Path) -> Path:
    config = MirrorConfig(MirrorResolution(ns=7, ntheta=1, nxi=11, mpol=0), z_min=-1.0, z_max=1.0)
    boundary = MirrorBoundary.polynomial_radius(r0=0.3, a2=0.05)
    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(0.01),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.polynomial([0.1, -0.05], gamma=2.0),
        options=MirrorSolveOptions(maxiter=2, step_size=1.0e-4, tolerance=1.0e-12, mu0=1.0),
    )
    return write_mirror_output(tmp_path / "mout_plot.nc", result)


def _finite_current_mout_file(tmp_path: Path) -> Path:
    config = MirrorConfig(MirrorResolution(ns=7, ntheta=1, nxi=11, mpol=0), z_min=-1.0, z_max=1.0)
    boundary = MirrorBoundary.polynomial_radius(r0=0.3, a2=0.05)
    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(0.01),
        i_prime=IPrimeProfile.constant(0.02),
        pressure=PressureProfile.zero(),
        options=MirrorSolveOptions(maxiter=1, step_size=1.0e-4, tolerance=1.0e-12, mu0=1.0),
    )
    return write_mirror_output(tmp_path / "mout_finite_current_plot.nc", result)


def test_mirror_plot_data_helpers_expose_numerical_content(tmp_path):
    output = load_mirror_output(_mout_file(tmp_path))

    surfaces = mirror_surfaces_rz_data(output, num_surfaces=4)
    bmag_sxi = mirror_bmag_sxi_data(output)
    bmag_boundary = mirror_bmag_boundary_data(output)
    bfield_boundary = mirror_bfield_boundary_data(output, stride_theta=3, stride_xi=2)
    field_lines = mirror_boundary_field_line_data(output, num_lines=4)
    boundary = mirror_boundary_3d_data(output, ntheta_axisym=12)
    cross_sections = mirror_cross_sections_data(output, num_sections=3, num_surfaces=4, ntheta_axisym=10)
    jacobian = mirror_jacobian_data(output)
    pressure = mirror_pressure_profile_data(output)
    pitch = mirror_field_line_pitch_profile_data(output)
    radial = mirror_radial_diagnostics_data(output)
    boozer_like = mirror_boozer_like_diagnostics_data(output)
    boozer_summary = mirror_boozer_like_summary_metrics(output)
    history = mirror_residual_history_data(output)

    assert surfaces.radii.shape == (4, output.nxi)
    assert np.allclose(surfaces.boundary_radius, output.geometry.boundary_r[0])
    assert np.allclose(bmag_sxi.bmag, np.mean(output.field.bmag, axis=1))
    assert np.allclose(bmag_boundary.bmag, output.field.bmag[-1])
    assert bfield_boundary.x.shape == bfield_boundary.bx.shape
    assert field_lines.x.shape == (4, output.nxi)
    assert np.allclose(field_lines.z, output.z[None, :])
    assert boundary.x.shape == boundary.y.shape == boundary.z.shape == boundary.bmag.shape
    assert cross_sections.x.shape == (3, 4, 10)
    assert cross_sections.y.shape == cross_sections.x.shape
    assert jacobian.min_sqrtg == pytest.approx(float(np.min(jacobian.sqrtg)))
    assert np.allclose(pressure.pressure, output.profiles.pressure)
    assert pitch.turns_mean.shape == output.s.shape
    assert np.allclose(pitch.turns_mean, 0.0)
    assert radial.mean_bmag.shape == output.s.shape
    assert radial.iota_like_twist.shape == output.s.shape
    assert radial.field_line_turns.shape == output.s.shape
    assert np.allclose(radial.field_line_turns, pitch.turns_mean)
    assert boozer_like.bmag_flux_surface_average.shape == output.s.shape
    assert np.all(boozer_like.bmag_min <= boozer_like.bmag_flux_surface_average)
    assert np.all(boozer_like.bmag_flux_surface_average <= boozer_like.bmag_max)
    assert np.all(boozer_like.surface_mirror_ratio >= 1.0)
    assert np.all(boozer_like.normalized_bmag_ripple_rms >= 0.0)
    assert np.allclose(boozer_like.field_line_turns, pitch.turns_mean)
    assert np.allclose(boozer_like.contravariant_pitch_mean, 0.0)
    assert np.allclose(boozer_like.contravariant_pitch_rms, 0.0)
    assert boozer_summary["boozer_like_surface_mirror_ratio_max"] == pytest.approx(
        float(np.max(boozer_like.surface_mirror_ratio))
    )
    assert boozer_summary["boozer_like_field_line_turns_mean"] == pytest.approx(0.0)
    assert public_mirror_boozer_like_summary_metrics(output) == boozer_summary
    assert np.allclose(history.residual_norm, output.history.residual_norm)
    assert np.allclose(history.fsq, output.history.fsq)
    assert np.allclose(history.normalized_force, output.history.normalized_force)
    assert np.allclose(history.step_size, output.history.step_size)


def test_mirror_boozer_like_diagnostics_capture_finite_current_pitch(tmp_path):
    output = load_mirror_output(_finite_current_mout_file(tmp_path))

    pitch = mirror_field_line_pitch_profile_data(output, num_lines=2)
    boozer_like = mirror_boozer_like_diagnostics_data(output)
    summary = mirror_boozer_like_summary_metrics(output)

    assert np.all(boozer_like.iota_like_twist > 0.0)
    assert np.min(pitch.turns_mean) > 0.0
    assert np.min(boozer_like.field_line_turns) > 0.0
    assert np.allclose(boozer_like.field_line_turns, pitch.turns_mean)
    assert np.min(boozer_like.contravariant_pitch_mean) > 0.0
    assert np.max(boozer_like.contravariant_pitch_rms) > 0.0
    assert summary["boozer_like_field_line_turns_min"] == pytest.approx(float(np.min(pitch.turns_mean)))
    assert summary["boozer_like_contravariant_pitch_mean_min"] == pytest.approx(
        float(np.min(boozer_like.contravariant_pitch_mean))
    )
    assert summary["boozer_like_contravariant_pitch_rms_max"] == pytest.approx(
        float(np.max(boozer_like.contravariant_pitch_rms))
    )


def test_plot_mirror_output_writes_expected_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    mout = _mout_file(tmp_path)

    paths = plot_mirror_output(mout, outdir=tmp_path / "figures", name="case")

    assert set(paths) == {
        "surfaces_rz",
        "cross_sections",
        "boundary_3d",
        "bfield_boundary",
        "bmag_sxi",
        "bmag_boundary",
        "jacobian",
        "pressure_profile",
        "radial_diagnostics",
        "boozer_like_diagnostics",
        "residual_history",
    }
    for path in paths.values():
        assert path.suffix == ".png"
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_mirror_output_writes_nonaxisymmetric_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy.optimize")
    config = MirrorConfig(MirrorResolution(ns=5, ntheta=9, nxi=9, mpol=3), z_min=-1.0, z_max=1.0)
    boundary = MirrorBoundary.cosine_modulated_radius(r0=0.28, a2=0.06, epsilon=0.04, theta_mode=2)
    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(0.01),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        options=MirrorSolveOptions(optimizer="lbfgs", maxiter=2, tolerance=1.0e-10, mu0=1.0),
    )
    mout = write_mirror_output(tmp_path / "mout_nonaxisymmetric_plot.nc", result)

    paths = plot_mirror_output(mout, outdir=tmp_path / "figures", name="nonaxisymmetric")

    assert paths["boundary_3d"].exists()
    assert paths["cross_sections"].exists()
    assert paths["bmag_boundary"].exists()
    for path in paths.values():
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


def test_cli_plot_dispatches_mirror_output(monkeypatch, tmp_path):
    mout = _mout_file(tmp_path)
    calls = []

    def fake_plot_mirror_output(path, *, outdir):
        calls.append((Path(path), Path(outdir)))
        return {"surfaces_rz": Path(outdir) / "surfaces.png"}

    import vmec_jax.mirror.plotting.export as export

    monkeypatch.setattr(export, "plot_mirror_output", fake_plot_mirror_output)

    assert cli.main(["--plot", str(mout), "--outdir", str(tmp_path / "plots")]) == 0
    assert calls == [(mout.resolve(), (tmp_path / "plots").resolve())]
