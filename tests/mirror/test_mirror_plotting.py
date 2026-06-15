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
    plot_mirror_output,
    run_mirror_fixed_boundary,
    write_mirror_output,
)
from vmec_jax.mirror.core.boundary import MirrorBoundary
from vmec_jax.mirror.io.mout import load_mirror_output
from vmec_jax.mirror.plotting.bfield import mirror_bmag_boundary_data, mirror_bmag_sxi_data
from vmec_jax.mirror.plotting.diagnostics import (
    mirror_jacobian_data,
    mirror_pressure_profile_data,
    mirror_residual_history_data,
)
from vmec_jax.mirror.plotting.geometry import mirror_boundary_3d_data, mirror_surfaces_rz_data

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


def test_mirror_plot_data_helpers_expose_numerical_content(tmp_path):
    output = load_mirror_output(_mout_file(tmp_path))

    surfaces = mirror_surfaces_rz_data(output, num_surfaces=4)
    bmag_sxi = mirror_bmag_sxi_data(output)
    bmag_boundary = mirror_bmag_boundary_data(output)
    boundary = mirror_boundary_3d_data(output, ntheta_axisym=12)
    jacobian = mirror_jacobian_data(output)
    pressure = mirror_pressure_profile_data(output)
    history = mirror_residual_history_data(output)

    assert surfaces.radii.shape == (4, output.nxi)
    assert np.allclose(surfaces.boundary_radius, output.geometry.boundary_r[0])
    assert np.allclose(bmag_sxi.bmag, np.mean(output.field.bmag, axis=1))
    assert np.allclose(bmag_boundary.bmag, output.field.bmag[-1])
    assert boundary.x.shape == boundary.y.shape == boundary.z.shape == boundary.bmag.shape
    assert jacobian.min_sqrtg == pytest.approx(float(np.min(jacobian.sqrtg)))
    assert np.allclose(pressure.pressure, output.profiles.pressure)
    assert np.allclose(history.residual_norm, output.history.residual_norm)


def test_plot_mirror_output_writes_expected_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    mout = _mout_file(tmp_path)

    paths = plot_mirror_output(mout, outdir=tmp_path / "figures", name="case")

    assert set(paths) == {
        "surfaces_rz",
        "boundary_3d",
        "bmag_sxi",
        "bmag_boundary",
        "jacobian",
        "pressure_profile",
        "residual_history",
    }
    for path in paths.values():
        assert path.suffix == ".png"
        assert path.exists()
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
