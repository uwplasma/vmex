"""Released ESSOS field handoff, alpha tracing, and CLI output contracts."""

from __future__ import annotations

import contextlib
import dataclasses
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


def _birth_field(orientation=1.0, B=1.0, G=1.0, current=0.2, *, cosine=0.2,
                 sine=0.0, second=0.1, toroidal=False):
    from essos.boozer import BoozerField

    r, s = np.array([0.2, 0.65, 0.85]), np.array([0.1, 0.55, 0.9])
    b, profiles = np.zeros((2, 3, 4)), np.zeros((2, 3, 4))
    b[:, :, 3] = B * np.array([1, cosine, second])
    profiles[:, 0, 2], profiles[:, 0, 3] = 0.1, 0.7 + 0.1 * s[:-1]
    profiles[:, 1, 2], profiles[:, 1, 3] = orientation * G * 0.1, orientation * G * (1 + 0.1 * s[:-1])
    profiles[:, 2, 2], profiles[:, 2, 3] = orientation * current, orientation * current * s[:-1]
    bs = np.zeros_like(b)
    bs[:, 1, 3] = B * sine
    values = [r, b, s, profiles, [0, 0, 0] if toroidal else [0, 1, 2], [0, 4, 8]]
    return BoozerField(*map(jax.numpy.asarray, values), -1.0, 4,
                       None if sine == 0 else jax.numpy.asarray(bs))


@pytest.mark.parametrize("birth", ["surface", "volume"])
def test_birth_measure_is_independent_of_chart_orientation(birth):
    from vmex.core.tracing import sample_births

    positive = sample_births(_birth_field(), 64, birth=birth, seed=9)
    negative = sample_births(_birth_field(orientation=-1), 64, birth=birth, seed=9)
    np.testing.assert_array_equal(positive, negative)
    assert np.all((positive[:, 0] >= 0) & (positive[:, 0] <= 1))
    assert np.all((positive[:, 1] >= 0) & (positive[:, 1] < 2 * np.pi))
    assert np.all((positive[:, 2] >= 0) & (positive[:, 2] < 2 * np.pi / 4))
    assert np.all(np.abs(positive[:, 3]) <= 1)


@pytest.mark.parametrize("B", [0.0, -1.0, np.inf, np.nan])
def test_birth_sampling_rejects_invalid_field_strength(B):
    from vmex.core.tracing import sample_births

    with pytest.raises(ValueError, match="finite positive"):
        sample_births(_birth_field(B=B), 2)


@pytest.mark.parametrize("G", [0.0, np.inf, np.nan])
def test_birth_sampling_rejects_invalid_jacobian_measure(G):
    from vmex.core.tracing import sample_births

    with pytest.raises(ValueError, match="positive support"):
        sample_births(_birth_field(G=G, current=0), 2)


def test_volume_birth_sampling_rejects_zero_source():
    from vmex.core.tracing import sample_births

    with pytest.raises(ValueError, match="positive support"):
        sample_births(_birth_field(), 2, birth="volume", ne0=0)


@pytest.mark.parametrize("birth", ["surface", "volume"])
def test_birth_bounds_does_not_depend_on_a_small_proposal_batch(monkeypatch, birth):
    """A batch missing the high-weight region must not accept its local maximum."""
    from vmex.core.tracing import sample_births

    class Proposals:
        theta_batches = 0
        draws = 0

        def uniform(self, low, high, size):
            position = self.draws % (5 if birth == "volume" else 4)
            self.draws += 1
            if birth == "volume" and position == 0:
                return np.full(size, 0.8 if self.theta_batches == 0 else 0.0)
            if high == 2 * np.pi:
                self.theta_batches += 1
                return np.full(size, 0.0 if self.theta_batches == 1 else np.pi)
            if high == np.pi / 2 or low == -1:
                return np.zeros(size)
            return np.full(size, high * 0.05)

    rng = Proposals()
    monkeypatch.setattr(np.random, "default_rng", lambda _seed: rng)
    field = _birth_field(cosine=0 if birth == "volume" else 0.9, second=0)
    births = sample_births(field, 1, s=0.81, birth=birth)
    assert rng.theta_batches == 2
    column, expected = (0, 0.0) if birth == "volume" else (1, np.pi)
    assert births[0, column] == expected


@pytest.mark.parametrize("sign", [1, -1])
def test_surface_births_sample_the_cosine_and_sine_jacobian(sign):
    from vmex.core.tracing import sample_births

    n, s = 4096, 0.64
    field = _birth_field(cosine=0.6, sine=0.4, orientation=sign, second=0)
    births = sample_births(field, n, s=s, seed=17)
    phase = births[:, 1] - 4 * births[:, 2]
    # For B=1+a*cos(phase)+b*sin(phase), density proportional to B^-2 gives <cos>=-a, <sin>=-b.
    assert abs(np.cos(phase).mean() + 0.6 * np.sqrt(s)) < 6 / np.sqrt(n)
    assert abs(np.sin(phase).mean() + 0.4 * np.sqrt(s)) < 6 / np.sqrt(n)
    assert abs(births[:, 3].mean()) < 6 / np.sqrt(3 * n)
    np.testing.assert_array_equal(births, sample_births(field, n, s=s, seed=17))


def test_volume_births_match_radial_fusion_and_asymmetric_angle_moments():
    from scipy.integrate import quad
    from vmex.core.tracing import dt_reactivity, sample_births

    n, T0 = 4096, 12.0
    field = _birth_field(cosine=0, sine=0.4, second=0, toroidal=True)
    births = sample_births(field, n, birth="volume", seed=29, T0_keV=T0)

    def source(s):
        return (1 + 0.24 * s + 0.02 * s**2) * (1 - s**5)**2 * dt_reactivity(T0 * (1 - s)) / dt_reactivity(T0)

    moments = [quad(lambda s: s**k * source(s), 0, 1, epsabs=1e-10)[0] for k in range(3)]
    mean, variance = moments[1] / moments[0], moments[2] / moments[0] - (moments[1] / moments[0])**2
    assert abs(births[:, 0].mean() - mean) < 6 * np.sqrt(variance / n)
    assert abs(np.sin(4 * births[:, 2]).mean() - 0.4) < 6 / np.sqrt(n)
    # Density scale and Jacobian orientation cancel in the normalized birth measure.
    flipped = _birth_field(cosine=0, sine=0.4, second=0, toroidal=True, orientation=-1)
    np.testing.assert_array_equal(births, sample_births(flipped, n, birth="volume", seed=29, ne0=1e250))


@pytest.mark.parametrize("volume", [False, True])
def test_birth_bound_covers_positive_fields_with_nonpositive_fourier_l1_bound(volume):
    from vmex.core.tracing import _birth_bounds

    field = _birth_field(cosine=0.95, second=0.45)
    lower, upper = _birth_bounds(field, 1.0, volume)
    # At r=1, B=0.55+0.95*u+0.9*u² has a positive minimum despite sum(amplitudes)>1.
    minimum = 0.55 - 0.95**2 / 3.6
    assert 0 < lower <= minimum and upper >= 1.26


def test_birth_bounds_include_axis_and_boundary_spline_extrapolation():
    import dataclasses
    from vmex.core.tracing import _birth_bounds

    field = _birth_field(cosine=0, second=0)
    coefficient = np.zeros_like(field.b_coef)
    r0 = np.asarray(field.r_knots)[:-1]
    coefficient[:, 0, :] = np.stack([np.full(2, 0.2), 0.6 * r0, 0.6 * r0**2,
                                    0.5 + 0.2 * r0**3], axis=1)
    field = dataclasses.replace(field, b_coef=jax.numpy.asarray(coefficient))
    lower, upper = _birth_bounds(field, 0.0, False)
    assert 0 < lower <= 0.5 and upper >= 1.0
    lower, upper = _birth_bounds(field, 1.0, False)
    assert 0 < lower <= 0.7 and upper >= 1.26
    lower, upper = _birth_bounds(field, 0.25, True)
    assert 0 < lower <= 0.5 and upper >= 1.26


@pytest.mark.parametrize("temperature", [1e-3, 0.1, 1.0, 12.0, 100.0, 1e6])
def test_fusion_envelope_covers_the_complete_temperature_profile(temperature):
    from vmex.core.tracing import _fusion_envelope, dt_reactivity

    bound = _fusion_envelope(temperature)
    temperatures = np.geomspace(1e-3, max(1e-3, temperature), 1001)
    assert bound > 0 and np.isfinite(bound)
    assert np.all(dt_reactivity(temperatures) <= bound)


@pytest.mark.parametrize("kwargs", [dict(n=0), dict(n=1.5), dict(s=-0.1), dict(s=np.nan),
    dict(birth="line"), dict(birth="volume", ne0=np.inf),
    dict(birth="volume", T0_keV=-1), dict(birth="volume", T0_keV=np.nan)])
def test_birth_sampling_rejects_invalid_parameters(kwargs):
    from vmex.core.tracing import sample_births

    with pytest.raises(ValueError, match="Birth sampling|Volume births"):
        sample_births(_birth_field(), **{**dict(n=1), **kwargs})


def test_birth_spline_bound_keeps_the_correct_adjacent_float_piece():
    from vmex.core.tracing import _spline_range

    lo = np.nextafter(0.5, 1.0)
    hi = np.nextafter(lo, 1.0)
    assert (lo + hi) / 2 == hi
    knots = np.array([0.0, lo, hi, 1.0])
    coefficient = np.zeros((3, 1, 4))
    coefficient[:, 0, 3] = [2, 2, 3]
    coefficient[1, 0, 2] = 1 / (hi - lo)
    bounds = _spline_range(knots, coefficient, np.array([lo]), np.array([hi]))
    assert bounds[0, 0, 0] <= 2 and 3 <= bounds[0, 0, 1] < 4


def test_birth_sampling_preserves_float32_runtime_with_steep_radial_splines(monkeypatch):
    from vmex.core.tracing import sample_births

    import dataclasses
    from essos.boozer import BoozerField

    dtypes, original = set(), BoozerField.modB
    def observe(field, s, theta, zeta):
        dtypes.add(s.dtype)
        return original(field, s, theta, zeta)
    monkeypatch.setattr(BoozerField, "modB", observe)
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    try:
        surface = 0.3
        field = _birth_field(cosine=0.6, sine=0.4, second=0)
        coefficient = np.asarray(field.b_coef).copy()
        coefficient[:, 1, 2] = 1e4
        coefficient[:, 1, 3] = 0.6 + 1e4 * (np.asarray(field.r_knots, float)[:-1] - np.sqrt(surface))
        field = dataclasses.replace(field, b_coef=jax.numpy.asarray(coefficient))
        assert field.b_coef.dtype == jax.numpy.float32
        births = sample_births(field, 512, s=surface, seed=17)
        assert not jax.config.jax_enable_x64 and dtypes == {jax.numpy.dtype("float64")}
        np.testing.assert_array_equal(field.b_coef, coefficient)
        phase = births[:, 1] - 4 * births[:, 2]
        assert np.isfinite(births).all() and births.shape == (512, 4)
        assert abs(np.cos(phase).mean() + 0.6 * np.sqrt(surface)) < 6 / np.sqrt(512)
        assert abs(np.sin(phase).mean() + 0.4 * np.sqrt(surface)) < 6 / np.sqrt(512)
        with pytest.raises(ValueError, match="finite positive"):
            sample_births(_birth_field(B=0), 1)
        assert not jax.config.jax_enable_x64
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_volume_birth_sampling_accepts_a_floored_temperature_with_small_n(monkeypatch):
    from vmex.core import tracing

    monkeypatch.setattr(tracing, "_BIRTH_MAX_BATCHES", 3)
    births = tracing.sample_births(_birth_field(cosine=0, second=0), 1,
                                   birth="volume", T0_keV=1e-3)
    assert births.shape == (1, 4)


def test_birth_sampling_rejects_invalid_fields_and_bounds(monkeypatch):
    import dataclasses
    from vmex.core import tracing

    field = _birth_field(cosine=0, second=0)
    invalid = dataclasses.replace(field, sine_coef=jax.numpy.full_like(field.b_coef, np.nan))
    with pytest.raises(ValueError, match="finite"):
        tracing.sample_births(invalid, 1)
    with pytest.raises(ValueError, match="positive n/nfp"):
        tracing.sample_births(dataclasses.replace(field, nfp=0), 1)
    negative = np.asarray(field.b_coef).copy()
    negative[:, 0, 3] = -1
    with pytest.raises(ValueError, match="positive Boozer"):
        tracing.sample_births(dataclasses.replace(field, b_coef=jax.numpy.asarray(negative)), 1)
    zero = dataclasses.replace(field, profile_coef=jax.numpy.zeros_like(field.profile_coef))
    for birth in ("surface", "volume"):
        with pytest.raises(ValueError, match="positive support"):
            tracing.sample_births(zero, 1, birth=birth)
    monkeypatch.setattr(tracing, "_birth_bounds", lambda *_args: (10, 1))
    with pytest.raises(ValueError, match="envelope violation"):
        tracing.sample_births(field, 1)


@pytest.mark.parametrize("name,value,match", [
    ("xm", [1, 2, 3], "constant mode"),
    ("r_knots", [0.2, 1.1, 1.2], "radial knots"),
    ("s_knots", [0.1, 1.1, 1.2], "profile knots"),
    ("b_coef", np.zeros((2, 3, 3)), "finite cubic"),
], ids=["constant-mode", "radial-knots", "profile-knots", "cubic-shape"])
def test_birth_sampling_rejects_malformed_boozer_tables(name, value, match):
    import dataclasses
    from vmex.core.tracing import sample_births

    field = dataclasses.replace(_birth_field(cosine=0, second=0), **{name: jax.numpy.asarray(value)})
    with pytest.raises(ValueError, match=match):
        sample_births(field, 1, birth="volume")


def test_birth_sampling_stops_with_no_partial_result_at_the_proposal_limit(monkeypatch):
    from vmex.core import tracing

    field = _birth_field(cosine=0, second=0)

    class Rejections:
        def uniform(self, low, high, size):
            return np.full(size, high / 2)

    monkeypatch.setattr(np.random, "default_rng", lambda _seed: Rejections())
    monkeypatch.setattr(tracing, "_birth_bounds", lambda *_args: (1e-5, 1))
    monkeypatch.setattr(tracing, "_BIRTH_MAX_BATCHES", 3)
    with pytest.raises(ValueError, match="proposal limit"):
        tracing.sample_births(field, 1)


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
                 "mode cut 0.001 of largest amplitude", "Change with --trace-particles N",
                 "Max energy error:"):
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
    assert summary["method"] == "adaptive8"
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
    # Chunks change how ESSOS groups the particles over CPU devices, which can move the last bit of a sum.
    np.testing.assert_allclose(reported.lost_times, traced.lost_times, rtol=1e-12, atol=0)
    np.testing.assert_allclose(reported.trajectories, traced.trajectories, rtol=1e-12, atol=0)


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


def _catalog_birth_field(case, polarity=1):
    from essos.boozer import BoozerField

    # Eight native rows and 32 modes from actual ITER/NCSX/QHS46/LHD tables.
    # The fixture embeds input/table hashes, retained indices and converged
    # independent SciPy/NumPy quadrature; it is not a dynamics equilibrium.
    with np.load(Path(__file__).parent / "data" / "boozer_birth_measures.npz") as data:
        args = [data[f"{case}_{key}"] for key in
                ("s", "bmnc", "xm", "xn", "iota", "G", "I", "psi0", "nfp")]
    for k in (5, 6, 7):
        args[k] = polarity * args[k]
    return BoozerField.from_booz(*args, mode_tolerance=0)


@pytest.mark.parametrize("case", ["iter", "ncsx", "qhs46", "lhd"])
@pytest.mark.parametrize("birth", ["surface", "volume"])
def test_real_equilibrium_birth_sign_seed_and_quadrature(case, birth):
    from vmex.core.tracing import sample_births

    positive = sample_births(_catalog_birth_field(case), 2048, birth=birth, seed=29)
    negative = sample_births(_catalog_birth_field(case, -1), 2048, birth=birth, seed=29)
    np.testing.assert_array_equal(positive, negative)
    ss, theta, zeta, pitch = positive.T
    field = _catalog_birth_field(case)
    basis = np.column_stack([ss, ss**2, np.cos(theta), np.sin(theta),
                             np.cos(field.nfp * zeta), np.sin(field.nfp * zeta)])
    with np.load(Path(__file__).parent / "data" / "boozer_birth_measures.npz") as data:
        expected = data[f"{case}_{birth}_moments"]
    error = 6 * basis.std(axis=0, ddof=1) / np.sqrt(len(ss)) + 2e-5
    assert np.all(abs(basis.mean(axis=0) - expected) <= error)
    assert np.all(abs(pitch) <= 1)


@pytest.mark.parametrize("B", [1.0, 1e-200, 1e200])
def test_birth_sampling_with_near_degenerate_positive_measure(B):
    from vmex.core.tracing import sample_births

    ordinary = sample_births(_birth_field(current=0), 64, seed=4)
    tiny = sample_births(_birth_field(B=B, G=2.0**-900, current=0), 64, seed=4)
    np.testing.assert_array_equal(tiny, ordinary)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_optional_solver_cli_writes_method_and_energy(solovev_wout, tmp_path, method):
    import inspect
    import json
    from essos.boozer import trace_boozer
    from essos.constants import ALPHA_PARTICLE_MASS

    if "method" not in inspect.signature(trace_boozer).parameters:
        pytest.skip("requires ESSOS method support")
    traced = trace_alphas(solovev_wout, **{**TRACE_KWARGS, "timestep": 1.5625e-8})
    assert cli.main([
        str(solovev_wout), "--trace", "--quiet", "--outdir", str(tmp_path),
        "--trace-method", method, "--trace-particles", "8", "--trace-tmax", "1e-5",
        "--trace-times", "12", "--mbooz", "8", "--nbooz", "8",
        "--trace-seed", "1", "--trace-timestep", "1.5625e-8",
    ]) == 0
    summary = json.loads((tmp_path / "solovev_trace.json").read_text())
    assert summary["method"] == method and summary["integrator"].startswith(method.upper())
    assert summary["particles_failed"] == 0 and summary["max_energy_error"] < 1e-3
    arrays = np.load(tmp_path / "solovev_trace.npz")
    np.testing.assert_array_equal(arrays["initial_conditions"], traced.initial_conditions)
    np.testing.assert_array_equal(arrays["lost_times"], traced.lost_times)
    birth_speed = np.sqrt(2 * traced.particle_energy / ALPHA_PARTICLE_MASS)
    scale = [1, 1, 1, birth_speed, birth_speed]
    np.testing.assert_allclose(arrays["final_states"] / scale, traced.final_states / scale,
                               rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("failure_source", ["status", "energy", "drift"])
def test_orbit_failure_is_separate_from_energy_drift(solovev_wout, monkeypatch, failure_source):
    from types import SimpleNamespace
    import essos.boozer

    def failed_trace(_field, s, *_angles, **_kwargs):
        n = len(s)
        data = dict(states=np.zeros((n, 2, 5)), loss_times=np.full(n, -1.0),
                    thermalized_times=np.full(n, -1.0), energy_error=np.zeros(n))
        if failure_source == "status":
            data["failed"] = np.arange(n) == 0
        else:
            data["energy_error"][0] = 2e-3 if failure_source == "drift" else np.nan
        return SimpleNamespace(**data)

    monkeypatch.setattr(essos.boozer, "trace_boozer", failed_trace)
    if failure_source == "drift":
        result = trace_alphas(solovev_wout, **TRACE_KWARGS)
        assert result.loss_fraction == 0 and result.energy_error[0] == 2e-3
    else:
        with pytest.raises(ValueError, match="loss fraction is undefined"):
            trace_alphas(solovev_wout, **TRACE_KWARGS)


def test_asymmetric_trace_requires_complete_backend(solovev_wout, monkeypatch):
    from dataclasses import replace
    from essos.boozer import BoozerField
    from vmex.core.tracing import boozer_field

    monkeypatch.setattr(BoozerField, "from_booz", lambda *args, **kwargs: None)
    with pytest.raises(ImportError, match="sine-spectrum support"):
        boozer_field(replace(read_wout(solovev_wout), lasym=True))


def test_trace_boundary_handoff_keeps_sine_partners(solovev_wout, monkeypatch):
    import vmex.core.tracing as tracing

    wout = read_wout(solovev_wout)
    field, bx = tracing.boozer_field(wout, mboz=8, nboz=8)
    bx.asym = True
    names = ("rmns_b", "zmnc_b", "numnc_b")
    for name in names:
        setattr(bx, name, np.full_like(bx.rmnc_b, 0.125))
    monkeypatch.setattr(tracing, "boozer_field", lambda *a, **k: (field, bx))
    result = trace_alphas(wout, scale=None, **{
        **TRACE_KWARGS, "tmax": 1e-7, "timestep": 1e-7, "times_to_trace": 2})
    assert result.boozer["asym"] is True
    for name in names:
        np.testing.assert_array_equal(result.boozer[name], getattr(bx, name)[:, -1])


def test_asymmetric_trace_preserves_sine_boundary(tmp_path):
    from vmex.core.tracing import boozer_field

    assert cli.main([str(DATA_DIR / "input.up_down_asymmetric_tokamak"),
                     "--ftol", "1e-10", "--quiet", "--outdir", str(tmp_path)]) == 0
    wout = read_wout(tmp_path / "wout_up_down_asymmetric_tokamak.nc")
    native = [essos_vmec_field(source, ntheta=8, nphi=8)
              for source in (tmp_path / "wout_up_down_asymmetric_tokamak.nc", wout)]
    for field in native:
        for name in ("rmns", "zmnc", "bmns", "gmns", "bsubsmnc", "bsubumns",
                     "bsubvmns", "bsupumns", "bsupvmns"):
            expected, actual = getattr(wout, name), getattr(field, name)
            if actual is None:
                assert expected is None or not np.any(expected)
            else:
                np.testing.assert_array_equal(actual, expected)
    for point in ([0.3, 0.4, 0.1], [0.7, 2.2, 0.9]):
        for name in ("AbsB", "to_xyz", "B_covariant", "B_contravariant", "sqrtg"):
            np.testing.assert_array_equal(getattr(native[0], name)(point),
                                          getattr(native[1], name)(point))
    field, bx = boozer_field(wout, mboz=8, nboz=8)
    assert field.sine_coef is not None and np.max(np.abs(field.sine_coef)) > 1e-10
    result = trace_alphas(wout, scale=None, tmax=1e-6, timestep=1e-8, nparticles=4,
                          times_to_trace=3, mboz=8, nboz=8)
    assert result.boozer["asym"] is True and result.particles_failed == 0
    assert np.isfinite(result.trajectories).all() and result.energy_error.max() < 1e-5
    assert {"rmns_b", "zmnc_b", "numnc_b"} <= result.boozer.keys()


def test_asymmetric_boundary_cartesian_coordinates():
    from vmex.core.plotting import _boozer_boundary_xyz

    theta, zeta = np.array([0.2, 0.7]), np.array([0.3, 0.4])
    bz = dict(xm_b=np.array([0, 1]), xn_b=np.array([0, 2]), asym=True,
              rmnc_b=np.array([10, 0.3]), rmns_b=np.array([0, -0.4]),
              zmns_b=np.array([0, 0.2]), zmnc_b=np.array([0, 0.05]),
              numns_b=np.array([0, 0.04]), numnc_b=np.array([0, 0.06]))
    angle = theta - 2*zeta
    radius = 10 + 0.3*np.cos(angle) - 0.4*np.sin(angle)
    phi = zeta - 0.04*np.sin(angle) - 0.06*np.cos(angle)
    expected = (radius*np.cos(phi), radius*np.sin(phi),
                0.2*np.sin(angle) + 0.05*np.cos(angle))
    np.testing.assert_allclose(_boozer_boundary_xyz(bz, theta, zeta), expected, atol=1e-14)


@pytest.mark.parametrize("energy, expected", [(1e-6, "(converged, below 0.001)"),
                                              (5e-2, "the orbits are not converged. Rerun with --trace-tolerance 3e-08")])
def test_cli_reports_whether_the_orbits_converged(solovev_wout, tmp_path, monkeypatch, energy, expected):
    """The energy check names the step to rerun with when the orbits are not converged."""
    from vmex.core import tracing

    real = tracing.trace_alphas

    def traced(*args, **kwargs):
        result = real(*args, **kwargs)
        return dataclasses.replace(result, energy_error=np.full_like(result.energy_error, energy))

    monkeypatch.setattr(tracing, "trace_alphas", traced)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = cli.main([str(solovev_wout), "--trace", "--outdir", str(tmp_path), "--trace-particles", "4",
                       "--trace-tmax", "1e-6", "--mbooz", "8", "--nbooz", "8"])
    assert rc == 0 and expected in buffer.getvalue()


def test_integrator_keywords_reach_essos(solovev_wout, monkeypatch):
    """Adaptive Dopri8 at TOLERANCE by default; fixed methods get no tolerance; compaction passes through."""
    import essos.boozer

    from vmex.core.tracing import TOLERANCE

    original, seen = essos.boozer.trace_boozer, []

    def recording(*args, **kwargs):
        seen.append({k: kwargs.get(k) for k in ("method", "tolerance", "compact")})
        return original(*args, **kwargs)

    monkeypatch.setattr(essos.boozer, "trace_boozer", recording)
    kwargs = {k: v for k, v in TRACE_KWARGS.items() if k != "method"}
    default = trace_alphas(solovev_wout, **kwargs)
    trace_alphas(solovev_wout, **kwargs, method="rk4", compact=False)
    assert seen == [dict(method="adaptive8", tolerance=TOLERANCE, compact=True),
                    dict(method="rk4", tolerance=None, compact=False)]
    assert default.metadata["integrator"].startswith("adaptive Dopri8, tolerance")
    with pytest.raises(ValueError, match="method must be one of"):
        trace_alphas(None, method="unknown")
    with pytest.raises(ValueError, match="tolerance must be positive"):
        trace_alphas(solovev_wout, **kwargs, tolerance=0.0)



def test_trace_devices_default_to_every_cpu_or_one_gpu(solovev_wout, monkeypatch):
    """CPU runs split over every device, GPU runs keep one; an explicit list wins."""
    import essos.boozer
    import jax

    original, seen = essos.boozer.trace_boozer, []

    def recording(*args, **kwargs):
        seen.append(kwargs["devices"])
        return original(*args, **kwargs)

    monkeypatch.setattr(essos.boozer, "trace_boozer", recording)
    cpu = trace_alphas(solovev_wout, **TRACE_KWARGS)
    assert seen[-1] == jax.devices() and cpu.metadata["devices"] == len(jax.devices())
    explicit = trace_alphas(solovev_wout, **TRACE_KWARGS, devices=jax.devices()[:1])
    assert seen[-1] == jax.devices()[:1] and explicit.metadata["devices"] == 1
    gpus = [jax.devices()[0]] * 2
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "devices", lambda: gpus)
    trace_alphas(solovev_wout, **TRACE_KWARGS)
    assert seen[-1] == gpus[:1]
