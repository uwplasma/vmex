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


@pytest.mark.parametrize("phase", [0.0, 0.7, np.pi / 2])
def test_mode_cut_spectrum_uses_physical_boozer_angular_derivatives(phase):
    from types import SimpleNamespace

    from benchmarks.trace_mode_cut import spectrum

    s = np.array([0.04, 0.25, 0.64, 1.0])
    bx = SimpleNamespace(s_b=s, xm_b=np.array([0, 0, 1]),
                         xn_b=np.array([0, 1, 0]),
                         bmnc_b=np.array([np.ones(4), np.full(4, 0.02),
                                          5e-4 * np.sqrt(s)]))
    bx.asym = phase != 0
    bx.bmns_b = np.zeros_like(bx.bmnc_b)
    bx.bmns_b[1:] = bx.bmnc_b[1:] * np.sin(phase)
    bx.bmnc_b[1:] *= np.cos(phase)
    cut = spectrum(bx)[0]["cuts"]
    assert cut["0.0001"]["modes"] == 3
    assert cut["0.001"]["modes"] == 2
    expected = 2.5e-4 / np.hypot(0.02, 2.5e-4)
    assert cut["0.001"]["angle_gradient_rms_rel"] == pytest.approx(expected)


def test_cut_audit_scales_boozer_tables_without_retransform(solovev_wout, tmp_path):
    from booz_xform_jax import Booz_xform
    from essos.boozer import BoozerField

    from benchmarks.trace_mode_cut import scaled_field
    from vmex.core.scaling import scale_wout
    from vmex.core.wout import write_wout

    wout = read_wout(solovev_wout)
    b, r = 2.3, 1.7
    scaled_path = tmp_path / "wout_scaled.nc"
    write_wout(scaled_path, scale_wout(wout, b_scale=b, r_scale=r))
    bx = []
    for path in (solovev_wout, scaled_path):
        transform = Booz_xform(verbose=0, mboz=8, nboz=8)
        transform.read_wout(str(path), flux=False)
        transform.run()
        bx.append(transform)
    direct = scaled_field(bx[0], wout, b, r, 1e-4)
    reference = BoozerField.from_booz_xform(
        bx[1], -float(np.asarray(wout.phi)[-1]) * b * r**2 / (2 * np.pi), 1e-4)
    np.testing.assert_allclose(direct.b_coef, reference.b_coef, rtol=1e-8, atol=5e-11)
    np.testing.assert_allclose(direct.profile_coef, reference.profile_coef, rtol=1e-8, atol=5e-11)
    assert direct.psi0 == pytest.approx(reference.psi0)


def test_cut_audit_records_common_births_and_individual_losses(solovev_wout):
    from booz_xform_jax import Booz_xform
    from benchmarks.trace_mode_cut import orbits

    bx = Booz_xform(verbose=0, mboz=8, nboz=8)
    bx.read_wout(str(solovev_wout), flux=False)
    bx.run()
    result = orbits(solovev_wout, bx, [6e-5, 1e-4], 8, 1e-5, 1, 3, 1e-4, 1,
                    s=0.25, birth_samples=16)
    assert result["reference_cut"] == 6e-5
    assert len(result["birth_sha256"]) == 64
    for row in result["cuts"].values():
        assert row["lost_indices"] == np.flatnonzero(np.array(row["loss_times"]) >= 0).tolist()
        assert row["lost"] == len(row["lost_indices"])


def test_cut_audit_exits_nonzero_after_writing_case_error(tmp_path, monkeypatch):
    import json
    import sys

    from benchmarks.trace_mode_cut import main

    target = tmp_path / "cuts.json"
    monkeypatch.setattr(sys, "argv", ["trace_mode_cut.py", str(tmp_path / "wout_missing.nc"),
                                      "--out", str(target)])
    with pytest.raises(SystemExit, match="one or more WOUT audits failed"):
        main()
    assert "traceback" in json.loads(target.read_text())["cases"][0]


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


def test_vmec_flux_sign_gives_the_boozer_radial_drift(solovev_wout):
    """VMEC Phi and the guiding-centre psi have opposite signs (FIRM3D/SIMSOPT)."""
    import jax.numpy as jnp
    from essos import constants as c
    from essos.boozer import guiding_center_rhs
    from vmex.core.tracing import boozer_field

    wout = read_wout(solovev_wout)
    field, _ = boozer_field(wout, mboz=8, nboz=8)
    psi = -float(wout.phi[-1]) / (2 * np.pi)
    assert field.psi0 == pytest.approx(psi)
    s, theta, zeta, pitch = 0.25, 0.7, 0.3, 0.4
    r = np.sqrt(s)
    speed = np.sqrt(2 * c.FUSION_ALPHA_PARTICLE_ENERGY / c.ALPHA_PARTICLE_MASS)
    vpar = pitch * speed
    B, _, Btheta_over_r, Bzeta = map(float, field.modB_derivatives(r, theta, zeta))
    mu = speed**2 * (1 - pitch**2) / (2 * B)
    (iota, G, current), (_, Gprime, Iprime) = np.asarray(field.profiles(s))
    C = -c.ALPHA_PARTICLE_CHARGE * iota + c.ALPHA_PARTICLE_MASS * vpar * Gprime / (psi * B)
    F = c.ALPHA_PARTICLE_CHARGE + c.ALPHA_PARTICLE_MASS * vpar * Iprime / (psi * B)
    D = F * G - C * current
    expected_sdot = ((current * Bzeta - r * G * Btheta_over_r) *
                     c.ALPHA_PARTICLE_MASS * (vpar**2 / B + mu) / (D * psi))
    y = jnp.array([r * np.cos(theta), r * np.sin(theta), zeta, vpar])
    dy = np.asarray(guiding_center_rhs(field, y, mu, c.ALPHA_PARTICLE_MASS,
                                       c.ALPHA_PARTICLE_CHARGE))
    actual_sdot = 2 * (y[0] * dy[0] + y[1] * dy[1])
    assert abs(expected_sdot) > 1.0
    assert actual_sdot == pytest.approx(expected_sdot, rel=1e-11)


@pytest.mark.parametrize("failure_source", ["status", "energy", "drift"])
def test_failed_orbit_cannot_produce_a_loss_fraction(solovev_wout, monkeypatch, failure_source):
    from types import SimpleNamespace
    import essos.boozer

    def failed_trace(_field, s, *_angles, **_kwargs):
        n = len(s)
        data = dict(states=np.zeros((n, 2, 5)), loss_times=np.full(n, -1.0),
                    thermalized_times=np.full(n, -1.0), energy_error=np.zeros(n))
        if failure_source == "status":
            data["failed"] = np.arange(n) == 0
        else:
            data["energy_error"][0] = 1.1e-3 if failure_source == "drift" else np.nan
        return SimpleNamespace(**data)

    monkeypatch.setattr(essos.boozer, "trace_boozer", failed_trace)
    with pytest.raises(ValueError, match="loss fraction is undefined"):
        trace_alphas(solovev_wout, **TRACE_KWARGS)


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


def test_births_are_invariant_under_field_direction(solovev_wout):
    import equinox as eqx

    from vmex.core.tracing import boozer_field, sample_births

    field, _ = boozer_field(read_wout(solovev_wout), mboz=8, nboz=8)
    reversed_field = eqx.tree_at(
        lambda f: f.profile_coef, field,
        field.profile_coef.at[:, 1:, :].multiply(-1))
    for birth in ("surface", "volume"):
        np.testing.assert_array_equal(sample_births(field, 16, birth=birth, seed=7),
                                      sample_births(reversed_field, 16, birth=birth, seed=7))
    zero_field = eqx.tree_at(lambda f: f.profile_coef, field,
                             field.profile_coef.at[:, 1:, :].set(0))
    with pytest.raises(ValueError, match="positive finite"):
        sample_births(zero_field, 16)


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

    buffer, progress = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(progress):
        rc = cli.main([
            str(solovev_wout), "--trace", "--outdir", str(tmp_path),
            "--trace-particles", "8", "--trace-tmax", "1e-5",
            "--trace-times", "12", "--trace-seed", "1", "--mbooz", "8", "--nbooz", "8",
            "--collisional", "--trace-birth", "volume", "--trace-mode-cut", "1e-3",
        ])
    stdout = buffer.getvalue()
    assert rc == 0, stdout
    for line in ("Loss fraction:", "Thermalized:", "Solver failures:",
                 "Scaling: B_scale=", "compile", "volavgB=5.8646 T, Aminor_p=1.7044 m",
                 "mode cut 0.001 of B00", "Change with --trace-particles N"):
        assert line in stdout, line
    assert "traced 100% of tmax" in progress.getvalue()
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


@pytest.mark.parametrize("levels, expected", [((10, 4), 10), ((4, 6), 14), (None, 14)])
def test_trace_cpu_devices_skip_efficiency_cores_only_when_fewer(monkeypatch, levels, expected):
    """M3 Max (10 P + 4 E) keeps its performance cores; M4 (4 + 6) and Linux use every usable core."""
    import subprocess
    import types

    from vmex.core import parallel

    monkeypatch.setattr(parallel, "available_cpus", lambda: 14)

    def fake_sysctl(command, **_):
        if levels is None:
            raise FileNotFoundError("sysctl")
        return types.SimpleNamespace(stdout=str(levels[int(command[-1][12])]))

    monkeypatch.setattr(subprocess, "run", fake_sysctl)
    assert cli._trace_cpu_devices() == expected


def test_progress_leaves_the_trace_unchanged(traced, solovev_wout):
    """Reporting progress runs the horizon in chunks and changes no orbit."""
    calls = []
    reported = trace_alphas(solovev_wout, **TRACE_KWARGS, progress=lambda d, n: calls.append((d, n)))
    assert calls[-1][0] == calls[-1][1] and len(calls) > 1
    np.testing.assert_array_equal(reported.lost_times, traced.lost_times)
    np.testing.assert_array_equal(reported.trajectories, traced.trajectories)


@pytest.mark.parametrize("tty", [True, False])
def test_trace_progress_estimates_after_the_first_chunk(monkeypatch, tty):
    """One rewritten line in a terminal, a line per chunk in a log; no estimate from the compiling chunk."""
    stream = io.StringIO()
    stream.isatty = lambda: tty
    monkeypatch.setattr(cli.sys, "stderr", stream)
    meter = cli._TraceProgress()
    for done in (1, 2, 4):
        meter(done, 4)
    text = stream.getvalue()
    assert "25% of tmax" in text and "estimating the rest" in text
    assert "s left" in text and "100% of tmax" in text and "done" in text
    assert text.count("\r") == (3 if tty else 0) and text.endswith("\n")


def test_cli_trace_names_the_upgrade_for_an_outdated_essos(solovev_wout, tmp_path, monkeypatch):
    """A stale environment gets the pip command, not a TypeError."""
    from vmex import _compat

    monkeypatch.setitem(_compat.OPTIONAL_MINIMUMS, "essos", "999.0")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = cli.main([str(solovev_wout), "--trace", "--outdir", str(tmp_path), "--quiet"])
    assert rc != 0
    assert 'pip install -U "essos>=999.0"' in buffer.getvalue()
    assert "MISSING OR OUTDATED OPTIONAL DEPENDENCY" in buffer.getvalue()


def test_mode_cut_scaling_preserves_sine_spectra(monkeypatch):
    from types import SimpleNamespace
    from essos.boozer import BoozerField
    from benchmarks.trace_mode_cut import scaled_field

    bx = SimpleNamespace(asym=True, s_b=np.array([0.1, 0.9]), nfp=2,
                         bmnc_b=np.ones((1, 2)), bmns_b=np.full((1, 2), 0.2),
                         xm_b=np.array([0]), xn_b=np.array([0]), iota=np.ones(2),
                         Boozer_G=np.ones(2), Boozer_I=np.zeros(2))
    monkeypatch.setattr(BoozerField, "from_booz", lambda *a, **kw: kw)
    result = scaled_field(bx, SimpleNamespace(phi=np.array([0, 1.])), 3, 2, 1e-4)
    np.testing.assert_array_equal(result["bmns"], 3 * bx.bmns_b)
