"""The ESSOS field handoff and alpha tracing (``vmex.core.tracing``).

Small and honest: 8 particles over ``tmax = 1e-5`` s on the solovev quick
case.  Gates: ``essos_vmec_field`` builds the same field from a wout path
and from an in-memory equilibrium, the trace runs on the released ESSOS
surface, the counts are mutually consistent, the loss fraction is a
fraction, the in-memory equilibrium route (temporary-wout hop) reproduces
the file route, and ``vmex --trace`` scales to ARIES-CS size in memory and
writes its JSON/NPZ summary and figures end to end.  Skips cleanly
without ESSOS.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

from vmex.core import cli
from vmex.core.tracing import essos_vmec_field, trace_alphas
from vmex.core.wout import read_wout

# Skip per test, not at import: a module that collects nothing breaks the
# manifest ownership check wherever ESSOS is not installed.
pytestmark = [
    pytest.mark.usefixtures("_module_jit_enabled"),
    pytest.mark.skipif(importlib.util.find_spec("essos") is None, reason="requires ESSOS"),
]

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
SOLOVEV_DECK = DATA_DIR / "input.solovev"

TRACE_KWARGS = dict(
    tmax=1e-5, nparticles=8, s=0.25, seed=1, timestep=5e-7, times_to_trace=12,
    mboz=8, nboz=8,
)


@pytest.fixture(scope="module")
def solovev_wout(tmp_path_factory) -> Path:
    """One quiet CLI solve of the solovev deck, shared by the tests below."""
    outdir = tmp_path_factory.mktemp("trace_wout")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = cli.main([str(SOLOVEV_DECK), "--outdir", str(outdir), "--quiet"])
    assert rc == 0, buffer.getvalue()
    return outdir / "wout_solovev.nc"


@pytest.fixture(scope="module")
def traced(solovev_wout):
    return trace_alphas(solovev_wout, **TRACE_KWARGS)


def test_counts_are_consistent(traced):
    n = traced.nparticles
    assert n == 8
    lost = int(np.sum(traced.lost_times >= 0.0))
    assert lost == traced.particles_lost
    assert traced.particles_failed == 0 and traced.particles_thermalized == 0
    assert 0.0 <= traced.loss_fraction <= 1.0
    assert traced.loss_fraction == pytest.approx(lost / n)
    assert traced.loss_fraction == pytest.approx(float(traced.loss_fractions[-1]))
    assert np.all(np.diff(traced.loss_fractions) >= 0.0)  # cumulative


def test_shapes_births_and_energy(traced):
    assert traced.initial_conditions.shape == (8, 4)
    np.testing.assert_allclose(traced.initial_conditions[:, 0], 0.25)
    assert np.all(np.abs(traced.initial_conditions[:, 3]) <= 1.0)
    assert traced.final_states.shape == (8, 5)
    assert traced.times.shape == (12,) and traced.times[-1] == pytest.approx(1e-5)
    # Collisionless RK4 conserves the orbit energy to far below a percent.
    assert np.max(traced.energy_error) < 1e-3


def test_volume_births_follow_the_fusion_profile(solovev_wout):
    """Volume births peak in the core: <s> of the D-T source is well below 1/2."""
    from vmex.core.tracing import boozer_field, dt_reactivity, sample_births
    from vmex.core.scaling import aries_cs_scales, scale_wout

    wout = read_wout(solovev_wout)
    b_scale, r_scale = aries_cs_scales(wout)
    field, _ = boozer_field(scale_wout(wout, b_scale=b_scale, r_scale=r_scale), mboz=8, nboz=8)
    births = sample_births(field, 2000, birth="volume", seed=0)
    assert births.shape == (2000, 4)
    assert 0.1 < births[:, 0].mean() < 0.4
    with pytest.raises(ValueError, match="birth"):
        sample_births(field, 4, birth="line")
    assert dt_reactivity(10.0) == pytest.approx(1.136e-22, rel=2e-3)  # Bosch & Hale 1992


def test_essos_field_handoff_matches_the_file_route(solovev_wout):
    """The field seam itself: both sources build the same ESSOS field."""
    from_file = essos_vmec_field(solovev_wout)
    from_memory = essos_vmec_field(read_wout(solovev_wout))
    assert int(from_memory.nfp) == int(from_file.nfp)
    assert int(from_memory.ns) == int(from_file.ns)
    points = np.array([[0.3, 0.4, 0.1], [0.7, 2.2, 0.9], [0.9, 5.0, 3.0]])
    for point in points:
        # The temporary wout is already deleted here; ESSOS read its tables
        # eagerly, so the field stays usable.
        np.testing.assert_allclose(
            float(from_memory.AbsB(point)), float(from_file.AbsB(point)), rtol=0.0)
        np.testing.assert_allclose(
            np.asarray(from_memory.to_xyz(point)),
            np.asarray(from_file.to_xyz(point)), rtol=0.0)


def test_in_memory_equilibrium_matches_the_file_route(traced, solovev_wout):
    result = trace_alphas(read_wout(solovev_wout), **TRACE_KWARGS)
    np.testing.assert_allclose(result.final_states, traced.final_states)


def test_cli_trace_writes_summary_files_and_figures(solovev_wout, tmp_path):
    """The --trace contract: scaled in memory, JSON/NPZ summary, two figures."""
    import json

    from vmex.core.scaling import aries_cs_scales

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = cli.main([
            str(solovev_wout), "--trace", "--outdir", str(tmp_path),
            "--trace-particles", "8", "--trace-tmax", "1e-5",
            "--trace-times", "12", "--trace-seed", "1", "--mbooz", "8", "--nbooz", "8",
            "--collisional", "--trace-birth", "volume",
        ])
    stdout = buffer.getvalue()
    assert rc == 0, stdout
    for line in ("Loss fraction:", "Thermalized:", "Solver failures:",
                 "Scaling: B_scale=", "compile"):
        assert line in stdout, line
    for suffix in ("trace.png", "trace_3d.png", "trace.npz"):
        assert (tmp_path / f"solovev_{suffix}").exists(), suffix
    summary = json.loads((tmp_path / "solovev_trace.json").read_text())
    b_scale, r_scale = aries_cs_scales(read_wout(solovev_wout))
    assert summary["b_scale"] == pytest.approx(b_scale)
    assert summary["r_scale"] == pytest.approx(r_scale)
    assert summary["volavgB"] == pytest.approx(5.8646)
    assert summary["Aminor_p"] == pytest.approx(1.7044)
    assert summary["timestep"] == pytest.approx(1.25e-7)
    assert summary["nparticles"] == 8 and summary["scale_target"] == "volavgB"
    assert summary["collisions"] is True
    assert {"loss_fraction_sigma", "compile_time_s", "devices", "versions"} <= set(summary)
    arrays = np.load(tmp_path / "solovev_trace.npz")
    assert arrays["initial_conditions"].shape == (8, 4)
    assert np.all((arrays["initial_conditions"][:, 0] > 0) & (arrays["initial_conditions"][:, 0] < 1))
    assert arrays["loss_fractions"][-1] == pytest.approx(summary["loss_fraction"])


def test_no_scale_traces_the_equilibrium_as_given(solovev_wout):
    result = trace_alphas(solovev_wout, **{**TRACE_KWARGS, "timestep": None}, scale=None)
    wout = read_wout(solovev_wout)
    assert (result.metadata["b_scale"], result.metadata["r_scale"]) == (1.0, 1.0)
    assert result.metadata["Aminor_p"] == pytest.approx(wout.Aminor_p)
    # The default step keeps the alpha step length a fixed fraction of the device.
    assert result.metadata["timestep"] == pytest.approx(1.25e-7 * wout.Aminor_p / 1.7044)


def test_plot_tracing_writes_the_surface_birth_figures(traced, tmp_path):
    import dataclasses

    from vmex.core.plotting import plot_tracing

    lost_times = np.where(np.arange(8) < 2, 5e-6, -1.0)  # two losses for the loss panels
    result = dataclasses.replace(traced, lost_times=lost_times)
    result.metadata["birth"] = "volume"
    written = plot_tracing(result, tmp_path, name="solovev")
    assert set(written) == {"summary", "3d"}
    assert all(path.stat().st_size > 0 for path in written.values())


def test_collisional_requires_trace(solovev_wout):
    with pytest.raises(SystemExit):
        cli.main([str(solovev_wout), "--plot", "--collisional"])
