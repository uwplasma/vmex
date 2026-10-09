"""The vmex-to-ESSOS field handoff, and alpha tracing on top of it.

- :func:`essos_vmec_field` — hand a solved equilibrium (or a wout file) to
  ESSOS as an ``essos.fields.Vmec``, ready for ESSOS tracing, surfaces and
  field queries.  An ESSOS coil field entering a vmex free-boundary solve goes
  the other way through :meth:`~vmex.core.mgrid.MgridField.from_coils`.
- :func:`trace_alphas` — trace fusion-born alpha particles and return the
  loss diagnostics as an :class:`AlphaTracingResult` (``vmex --trace``).

Tracing runs in Boozer coordinates: ``booz_xform_jax`` transforms the
equilibrium, and ``essos.boozer`` integrates the guiding-centre equations
(White; the ``K = 0`` form of SIMSOPT ``GuidingCenterNoKBoozerRHS``) with
fixed-step RK4 over a spline of the ``|B|`` spectrum, in a chart that is
regular on the magnetic axis.  One right-hand side costs about 1 µs per
particle, against 8-10 µs for the VMEC-coordinate field.
A particle is lost when it reaches ``s = 1``.

Births: on one surface ``s`` (default) or through the volume in proportion
to the D-T fusion rate (``birth="volume"``), uniform in pitch ``v_par/v`` over
``[-1, 1)`` and distributed over the angles with the Boozer Jacobian
``abs(G + iota I) / B^2``.  The plasma profiles, for volume births and for
``collisions=True``, are those of Landreman, Buller & Drevlak, PoP 29, 082501
(2022): ``n_D = n_T = n_e / 2 = (n_e0 / 2)(1 - s^5)`` and
``T = T_0 (1 - s)`` with ``n_e0 = 4e20 m^-3`` and ``T_0 = 12 keV``, and the
Bosch-Hale D-T reactivity.  With collisions the Monte Carlo operator of
``essos.boozer`` (pitch-angle scattering, slowing down and energy diffusion
on electrons, D and T) acts after every step, and an alpha whose energy falls
below 1.5 times the local temperature is thermalised (confined).

By default the equilibrium is first scaled in memory to ARIES-CS size
(:func:`~vmex.core.scaling.aries_cs_scales`, ``scale="volavgB"``), because
alpha orbit widths, and hence losses, depend on the absolute field and size.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import time
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from .._compat import require_optional

# Reference orbit step [s] at Aminor_p = 1.7044 m, scaled with Aminor_p, and
# the relative amplitude below which Boozer |B| modes are dropped
# (docs/howto/trace-alpha-particles.md, convergence table).
TIMESTEP = 1.25e-7
MODE_TOLERANCE = 2e-4
# largest relative energy error of converged collisionless orbits
ENERGY_TOLERANCE = 1e-3
# default integrator: per-particle error-controlled Dopri8(7) at this tolerance
# (21 equilibria: worst energy error 1.7e-4, six times below ENERGY_TOLERANCE)
METHOD = "adaptive8"
TOLERANCE = 3e-7
METHODS = ("adaptive8", "adaptive", "rk4", "dopri5", "dopri8")
# Landreman, Buller & Drevlak (2022) profiles: n_e0 [m^-3], T_0 [keV].
NE0, T0_KEV = 4e20, 12.0
_DT_COEFFICIENTS = (1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3, 1.35e-2, -1.0675e-4, 1.366e-5)
_BIRTH_MAX_BATCHES = 1000
_COMPILE_S = [0.0, 0.0]  # compile seconds, listener registered


def _compile_listener(event: str, duration: float, **_: Any) -> None:
    if event.startswith("/jax/core/compile/"):
        _COMPILE_S[0] += float(duration)


def _essos_imports():
    require_optional("essos", "alpha-particle tracing")
    try:
        from essos import constants, dynamics, fields
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "alpha-particle tracing requires ESSOS (`pip install essos`)"
        ) from exc
    return constants, dynamics, fields


def dt_reactivity(T_keV):
    """Bosch & Hale (NF 32, 611, 1992) D-T ``<sigma v>`` [m^3/s] at ``T`` [keV]."""
    T = np.maximum(np.asarray(T_keV, float), 1e-3)
    c = _DT_COEFFICIENTS
    theta = T / (1 - T * (c[1] + T * (c[3] + T * c[5])) / (1 + T * (c[2] + T * (c[4] + T * c[6]))))
    xi = (34.3827**2 / (4 * theta)) ** (1 / 3)
    return 1e-6 * c[0] * theta * np.sqrt(xi / (1124656.0 * T**3)) * np.exp(-3 * xi)


def background_species(ne0: float = NE0, T0_keV: float = T0_KEV):
    """Electrons, D and T on the default profiles, as an ESSOS ``BackgroundSpecies``."""
    import jax.numpy as jnp
    from essos.background_species import BackgroundSpecies
    from essos.constants import ELECTRON_MASS, PROTON_MASS

    s = np.linspace(0.0, 1.0, 101)
    n = ne0 * (1 - s**5)
    T = np.maximum(T0_keV * 1e3 * (1 - s), 10.0)  # eV, floored at the edge
    return BackgroundSpecies(
        3, jnp.array([ELECTRON_MASS / PROTON_MASS, 2.014, 3.016]), jnp.array([-1.0, 1.0, 1.0]),
        jnp.array([n, n / 2, n / 2]), jnp.array([T, T, T]), radial_grid=s)


@dataclasses.dataclass(frozen=True)
class AlphaTracingResult:
    """Alpha-loss diagnostics from one :func:`trace_alphas` call.

    ``initial_conditions`` holds the births ``(s, theta_B, zeta_B, v_par/v)``;
    ``final_states`` holds ``(s, theta_B, zeta_B, v_par, v)`` at the loss,
    thermalisation or final time; ``lost_times`` and ``thermalized_times`` are
    ``-1`` for particles without that outcome. Collisionless ``energy_error``
    is the maximum ``|E/E_initial - 1|`` over accepted steps, per particle.
    With collisions, the reference resets after each collision kick, so the
    metric measures orbital step error separately from collisional changes.
    """

    nparticles: int
    loss_fraction: float
    particles_lost: int
    particles_thermalized: int
    particles_failed: int
    wall_time_s: float
    particle_energy: float
    times: np.ndarray = dataclasses.field(repr=False)
    loss_fractions: np.ndarray = dataclasses.field(repr=False)
    lost_times: np.ndarray = dataclasses.field(repr=False)
    thermalized_times: np.ndarray = dataclasses.field(repr=False)
    initial_conditions: np.ndarray = dataclasses.field(repr=False)
    final_states: np.ndarray = dataclasses.field(repr=False)
    energy_error: np.ndarray = dataclasses.field(repr=False)
    trajectories: np.ndarray = dataclasses.field(repr=False)
    boozer: dict = dataclasses.field(repr=False, default_factory=dict)
    metadata: dict = dataclasses.field(repr=False, default_factory=dict)

    @property
    def loss_fraction_sigma(self) -> float:
        """Binomial standard error of :attr:`loss_fraction`."""
        f = self.loss_fraction
        return float(np.sqrt(f * (1.0 - f) / max(self.nparticles, 1)))

    def save(self, stem: str | Path) -> tuple[Path, Path]:
        """Write ``<stem>_trace.json`` (counts, factors, versions, timing) and
        ``<stem>_trace.npz`` (times, loss fractions, loss times, births and
        final states), so a run can be replotted and compared."""
        stem = Path(stem)
        summary = {
            "nparticles": self.nparticles, "loss_fraction": self.loss_fraction,
            "loss_fraction_sigma": self.loss_fraction_sigma,
            "particles_lost": self.particles_lost,
            "particles_thermalized": self.particles_thermalized,
            "particles_failed": self.particles_failed,
            "wall_time_s": self.wall_time_s,
            "max_energy_error": float(np.max(self.energy_error, initial=0.0)),
            "particle_energy_J": self.particle_energy, **self.metadata,
        }
        json_path = stem.with_name(stem.name + "_trace.json")
        json_path.write_text(json.dumps(summary, indent=2) + "\n")
        npz_path = stem.with_name(stem.name + "_trace.npz")
        np.savez_compressed(
            npz_path, times=self.times, loss_fractions=self.loss_fractions,
            lost_times=self.lost_times, thermalized_times=self.thermalized_times,
            initial_conditions=self.initial_conditions, final_states=self.final_states)
        return json_path, npz_path


def essos_vmec_field(source: Any, **kwargs: Any) -> Any:
    """Build an ESSOS VMEC field from a wout path or :class:`WoutData`.

    In-memory data use a temporary wout loaded eagerly by ESSOS; this
    diagnostic handoff severs gradients. Constructor ``kwargs`` pass through.
    """
    _, _, fields = _essos_imports()

    if hasattr(source, "rmnc") and hasattr(source, "xm"):  # WoutData
        from .wout import write_wout

        with tempfile.TemporaryDirectory(prefix="vmex_essos_") as tmp:
            wout_path = Path(tmp) / "wout_equilibrium.nc"
            write_wout(wout_path, source)
            return fields.Vmec(str(wout_path), **kwargs)

    return fields.Vmec(str(Path(source)), **kwargs)


def boozer_field(wout, *, mboz: int = 32, nboz: int = 32, mode_tolerance: float = MODE_TOLERANCE):
    """Run ``booz_xform_jax`` on every surface of ``wout`` and return
    ``(essos.boozer.BoozerField, Booz_xform)``."""
    from booz_xform_jax import Booz_xform
    from essos.boozer import BoozerField

    from .wout import write_wout

    if bool(wout.lasym):
        from inspect import signature

        if "bmns" not in signature(BoozerField.from_booz).parameters:
            raise ImportError("Non-symmetric tracing requires ESSOS sine-spectrum support; upgrade ESSOS")
    bx = Booz_xform(verbose=0, mboz=int(mboz), nboz=int(nboz))
    with tempfile.TemporaryDirectory(prefix="vmex_booz_") as tmp:
        path = Path(tmp) / "wout_trace.nc"
        write_wout(path, wout)
        bx.read_wout(str(path), flux=False)
    bx.run()
    # Boozer guiding-centre equations use psi = -Phi_tor/(2 pi) for VMEC's phi.
    psi0 = -float(np.asarray(wout.phi)[-1]) / (2 * np.pi)
    return BoozerField.from_booz_xform(bx, psi0, mode_tolerance), bx


def _interval_product(a, b):
    products = a[..., :, None] * b[..., None, :]
    return np.stack([np.nextafter(products.min(axis=(-2, -1)), -np.inf),
                     np.nextafter(products.max(axis=(-2, -1)), np.inf)], axis=-1)


def _interval_sum(a, b):
    return np.stack([np.nextafter(a[..., 0] + b[..., 0], -np.inf),
                     np.nextafter(a[..., 1] + b[..., 1], np.inf)], axis=-1)


def _spline_range(knots, coefficients, lo, hi, *, derivative=False):
    """Outward-rounded Bernstein enclosure, including extrapolated end pieces."""
    from math import comb

    knots, coefficients = np.asarray(knots, float), np.asarray(coefficients, float)
    if (knots.ndim != 1 or len(knots) < 2 or np.any(np.diff(knots) <= 0)
            or coefficients.ndim != 3 or coefficients.shape[0] != len(knots) - 1
            or coefficients.shape[-1] != 4 or not np.isfinite(knots).all()
            or not np.isfinite(coefficients).all()):
        raise ValueError("Birth sampling requires finite cubic Boozer splines")
    index = np.clip(np.searchsorted(knots, lo, side="right") - 1, 0, len(knots) - 2)
    c = np.stack([coefficients[index, :, ::-1]] * 2, axis=-1)
    if derivative:
        c = _interval_product(c[..., 1:, :], np.stack([np.arange(1, 4)] * 2, axis=-1))
    degree = c.shape[-2] - 1
    left = np.stack([np.nextafter(lo - knots[index], -np.inf),
                     np.nextafter(lo - knots[index], np.inf)], axis=-1)[:, None, :]
    width = np.stack([np.nextafter(hi - lo, -np.inf),
                      np.nextafter(hi - lo, np.inf)], axis=-1)[:, None, :]
    powers_l, powers_w = [np.ones_like(left)], [np.ones_like(width)]
    for _ in range(degree):
        powers_l.append(_interval_product(powers_l[-1], left))
        powers_w.append(_interval_product(powers_w[-1], width))
    power = []
    for k in range(degree + 1):
        value = np.zeros_like(c[..., 0, :])
        for j in range(k, degree + 1):
            term = _interval_product(c[..., j, :], powers_l[j - k])
            value = _interval_sum(value, _interval_product(term, np.array([comb(j, k)] * 2)))
        power.append(_interval_product(value, powers_w[k]))
    bernstein = []
    for j in range(degree + 1):
        value = np.zeros_like(power[0])
        for k in range(j + 1):
            ratio = comb(j, k) / comb(degree, k)
            factor = np.array([np.nextafter(ratio, -np.inf), np.nextafter(ratio, np.inf)])
            value = _interval_sum(value, _interval_product(power[k], factor))
        bernstein.append(value)
    values = np.stack(bernstein)
    if not np.isfinite(values).all():
        raise ValueError("Birth spline bounds must be finite")
    return np.stack([values[..., 0].min(axis=0), values[..., 1].max(axis=0)], axis=-1)


def _birth_bounds(field, s, volume):
    """Conservative |B| lower and |G+iota I| upper bounds for the stored splines."""
    from essos.boozer import BoozerField

    if not isinstance(field, BoozerField):
        raise TypeError("Birth sampling requires an ESSOS BoozerField")
    xm, xn = np.asarray(field.xm), np.asarray(field.xn)
    zero = (xm == 0) & (xn == 0)
    if (xm.ndim != 1 or xm.shape != xn.shape or not zero.any()
            or not np.isfinite(xm).all() or not np.isfinite(xn).all()):
        raise ValueError("Birth sampling requires finite Boozer modes and a constant mode")
    if not np.isfinite(field.b_coef).all() or (field.sine_coef is not None and not np.isfinite(field.sine_coef).all()):
        raise ValueError("Birth sampling requires finite positive |B| splines")
    edges = np.r_[0.0, np.asarray(field.r_knots)[1:-1], 1.0] if volume else np.array([np.sqrt(s)] * 2)
    if volume and (np.any(np.diff(edges) <= 0) or edges[0] != 0 or edges[-1] != 1):
        raise ValueError("Boozer radial knots must lie within the plasma")
    eps = np.finfo(np.asarray(field.b_coef).dtype).eps
    if field.sine_coef is not None:
        eps = max(eps, np.finfo(np.asarray(field.sine_coef).dtype).eps)
    for refinement in range(6):
        lo, hi = edges[:-1], edges[1:]
        if len(lo) * len(xm) > 1_000_000:
            raise ValueError("Cannot bound a finite positive Boozer |B| within the birth-bound work limit")
        cosine = _spline_range(field.r_knots, field.b_coef, lo, hi)
        sine = (np.zeros_like(cosine) if field.sine_coef is None
                else _spline_range(field.r_knots, field.sine_coef, lo, hi))
        radial = np.stack([lo, hi], axis=-1)[:, None, :]
        scale = np.where((xm > 0)[None, :, None], radial, 1.0)
        c, t = _interval_product(cosine, scale), _interval_product(sine, scale)
        amplitude = np.hypot(np.abs(c).max(axis=-1), np.abs(t).max(axis=-1))
        rounding = 128 * (len(xm) + 1 + np.abs(xm).max() + np.abs(xn).max() / field.nfp) * eps * amplitude.sum(axis=1)
        lower = c[:, zero, 0].sum(axis=1) - amplitude[:, ~zero].sum(axis=1) - rounding
        if np.all(lower > 0):
            break
        # Derivative bounds cover the radial and angular gaps between grid points.
        dc = _spline_range(field.r_knots, field.b_coef, lo, hi, derivative=True)
        ds = (np.zeros_like(dc) if field.sine_coef is None
              else _spline_range(field.r_knots, field.sine_coef, lo, hi, derivative=True))
        dc = np.where((xm > 0)[None, :, None], _interval_sum(cosine, _interval_product(radial, dc)), dc)
        ds = np.where((xm > 0)[None, :, None], _interval_sum(sine, _interval_product(radial, ds)), ds)
        dr = np.hypot(np.abs(dc).max(axis=-1), np.abs(ds).max(axis=-1)).sum(axis=1)
        nt = 1 if np.all(xm == 0) else 16 << refinement
        nz = 1 if np.all(xn == 0) else 16 << refinement
        if len(lo) * len(xm) * nt * nz > 100_000_000:
            raise ValueError("Cannot bound a finite positive Boozer |B| within the birth-bound work limit")
        theta, zeta = np.meshgrid((np.arange(nt) + 0.5) * 2 * np.pi / nt,
                                  (np.arange(nz) + 0.5) * 2 * np.pi / (field.nfp * nz))
        mid = (lo + hi) / 2
        # The same polynomial at a point gives a narrow, rounded value enclosure.
        cm = _spline_range(field.r_knots, field.b_coef, mid, mid)
        sm = (np.zeros_like(cm) if field.sine_coef is None
              else _spline_range(field.r_knots, field.sine_coef, mid, mid))
        point_scale = np.stack([np.where(xm > 0, mid[:, None], 1)] * 2, axis=-1)
        cm, sm = _interval_product(cm, point_scale), _interval_product(sm, point_scale)
        point_error = np.diff(cm, axis=-1).sum(axis=(1, 2)) + np.diff(sm, axis=-1).sum(axis=(1, 2))
        grid_min = np.full(len(mid), np.inf)
        chunk = max(1, min(512, 1_000_000 // len(xm)))
        for start in range(0, theta.size, chunk):
            phase = xm[:, None] * theta.ravel()[None, start:start + chunk] - xn[:, None] * zeta.ravel()[None, start:start + chunk]
            grid_min = np.minimum(grid_min, (cm[..., 0] @ np.cos(phase) + sm[..., 0] @ np.sin(phase)).min(axis=1))
        grid_bound = (grid_min - dr * np.maximum(mid - lo, hi - mid) - rounding - point_error
                      - np.pi * (amplitude @ np.abs(xm) / nt + amplitude @ np.abs(xn) / (field.nfp * nz)))
        lower = np.maximum(lower, grid_bound)
        if np.all(lower > 0):
            break
        if volume:
            edges = np.sort(np.r_[edges, mid])
    else:
        raise ValueError("Cannot bound a finite positive Boozer |B|; inspect the field or radial resolution")
    knots, coefficients = np.asarray(field.s_knots), np.asarray(field.profile_coef)
    if not np.isfinite(coefficients).all():
        raise ValueError("Birth sampling requires finite weights with positive support")
    if volume:
        edges = np.r_[0.0, knots[1:-1], 1.0]
        if np.any(np.diff(edges) <= 0):
            raise ValueError("Boozer profile knots must lie within the plasma")
        lo, hi = edges[:-1], edges[1:]
    else:
        lo = hi = np.array([s])
    profiles = _spline_range(knots, coefficients, lo, hi)
    jacobian = _interval_sum(profiles[:, 1], _interval_product(profiles[:, 0], profiles[:, 2]))
    numerator = np.abs(jacobian).max()
    magnitude = np.abs(profiles[:, 1]).max() + np.abs(profiles[:, 0]).max() * np.abs(profiles[:, 2]).max()
    numerator += 128 * np.finfo(coefficients.dtype).eps * magnitude
    if volume:
        if not coefficients[:, 1].any() and (not coefficients[:, 0].any() or not coefficients[:, 2].any()):
            numerator = 0.0
    else:
        index = np.clip(np.searchsorted(knots, s, side="right") - 1, 0, len(knots) - 2)
        c, d = coefficients[index], s - knots[index]
        iota, G, current = ((c[:, 0] * d + c[:, 1]) * d + c[:, 2]) * d + c[:, 3]
        if G + iota * current == 0:
            numerator = 0.0
    if not np.isfinite(numerator) or numerator <= 0:
        raise ValueError("Birth sampling requires finite weights with positive support")
    return lower.min(), np.nextafter(numerator, np.inf)


def _fusion_envelope(T0_keV):
    """Bound Bosch–Hale reactivity over the floored radial temperature profile."""
    c = _DT_COEFFICIENTS
    # theta/T is a convex combination of these positive polynomial coefficient ratios.
    ratio = max(1.0, c[2] / (c[2] - c[1]), c[4] / (c[4] - c[3]), c[6] / (c[6] - c[5]))
    temperature = max(T0_keV, 1e-3)
    ratio = min(ratio, 1 + temperature * (c[1] + temperature * c[3]))
    ratio *= 1 + 64 * np.finfo(float).eps
    k = 34.3827**2 / 4
    x = max(2 / 3, np.exp((np.log(k) - np.log(temperature) - np.log(ratio)) / 3)
            * (1 - 64 * np.finfo(float).eps))
    # x^2 exp(-3x) decreases for x >= 2/3; the density factor (1-s^5)^2 <= 1.
    bound = 1e-6 * c[0] * ratio**1.5 * x**2 * np.exp(-3 * x) / np.sqrt(1124656.0 * k)
    return np.nextafter(bound * (1 + 64 * np.finfo(float).eps), np.inf)


def sample_births(field, n: int, *, s: float = 0.25, birth: str = "surface",
                  seed: int = 42, ne0: float = NE0, T0_keV: float = T0_KEV):
    """Sample with a fixed conservative bound and 64-bit birth probabilities.

    The caller's runtime precision is restored; tracing precision is unchanged.
    """
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(True):
        if (not isinstance(n, (int, np.integer)) or n < 1 or birth not in ("surface", "volume")
                or not np.isfinite(s) or not 0 <= s <= 1 or not isinstance(field.nfp, (int, np.integer))
                or field.nfp < 1):
            raise ValueError("Birth sampling requires positive n/nfp, s in [0,1], and birth='surface' or 'volume'")
        volume = birth == "volume"
        if volume and (not np.isfinite(ne0) or ne0 <= 0 or not np.isfinite(T0_keV) or T0_keV <= 0):
            raise ValueError("Volume births require finite positive density and temperature for positive support")
        B_lower, jacobian_bound = _birth_bounds(field, s, volume)
        source_bound = _fusion_envelope(T0_keV) if volume else 1.0
        if not np.isfinite(source_bound) or source_bound <= 0:
            raise ValueError("Volume births require a finite, positive fusion envelope")
        rng = np.random.default_rng(int(seed))
        modB = jax.jit(jax.vmap(field.modB))
        out: list[np.ndarray] = []
        accepted = 0
        m = min(65_536, max(64, 4 * int(n)), max(1, 1_000_000 // field.xm.size))
        for _ in range(_BIRTH_MAX_BATCHES):
            ss = np.full(m, float(s)) if birth == "surface" else rng.uniform(0.0, 1.0, m)
            th = rng.uniform(0.0, 2 * np.pi, m)
            ze = rng.uniform(0.0, 2 * np.pi / field.nfp, m)
            B = np.asarray(modB(jnp.asarray(ss), jnp.asarray(th), jnp.asarray(ze)))
            if not np.isfinite(B).all() or np.any(B <= 0):
                raise ValueError("Birth sampling requires finite positive |B|")
            iota, G, current = field.profiles(jnp.asarray(ss))[0].T
            # Normalize before squaring: field strength and Jacobian scales cancel.
            weight = np.abs(np.asarray(G + iota * current)) / jacobian_bound * (B_lower / B)**2
            if volume:
                # The constant density scale cancels from the normalized distribution.
                weight *= (1 - ss**5) ** 2 * dt_reactivity(T0_keV * (1 - ss)) / source_bound
            if not np.isfinite(weight).all() or np.any(weight < 0) or np.any(weight > 1):
                raise ValueError("Invalid birth weight or rejection envelope violation")
            keep = rng.uniform(0.0, 1.0, m) < weight
            out.append(np.stack([ss, th, ze, rng.uniform(-1.0, 1.0, m)], axis=1)[keep])
            accepted += int(keep.sum())
            if accepted >= n:
                break
        else:
            raise ValueError("Birth sampling exceeded its proposal limit; inspect the field and acceptance rate")
        return np.concatenate(out)[:n]


def trace_alphas(
    source: Any,
    *,
    tmax: float = 1e-2,
    nparticles: int = 500,
    s: float = 0.25,
    seed: int = 42,
    timestep: float | None = None,
    method: str = METHOD,
    tolerance: float = TOLERANCE,
    compact: bool | None = None,
    times_to_trace: int = 1000,
    scale: str | None = "volavgB",
    birth: str = "surface",
    collisions: bool = False,
    ne0: float = NE0,
    T0_keV: float = T0_KEV,
    mboz: int = 32,
    nboz: int = 32,
    mode_tolerance: float = MODE_TOLERANCE,
    progress: Any = None,
    devices: Any = None,
) -> AlphaTracingResult:
    """Trace fusion alphas through a wout file or in-memory equilibrium.

    Parameters
    ----------
    source:
        Path to a ``wout_*.nc`` file, or an in-memory
        :class:`~vmex.core.wout.WoutData`.
    tmax, timestep, times_to_trace:
        Horizon [s]; RK4 step [s] (``None``: :data:`TIMESTEP` times
        ``Aminor_p / 1.7044 m``); samples of the loss-fraction curve.
    nparticles, s, seed, birth:
        Ensemble size, launch surface (``birth="surface"``) and seed;
        ``birth="volume"`` samples the D-T birth profile instead.
    scale:
        :data:`~vmex.core.scaling.SCALE_TARGETS` convention to scale the
        equilibrium to ARIES-CS size in memory first, or ``None``.
    collisions, ne0, T0_keV:
        Monte Carlo collisions on the default background, with on-axis
        electron density [m^-3] and temperature [keV] (also the volume
        birth profile).
    mboz, nboz, mode_tolerance:
        Boozer resolution and the relative amplitude of dropped modes.
    method, tolerance:
        ``"adaptive8"`` (default) and ``"adaptive"`` control each particle's
        Dopri8 or Dopri5 step to the embedded-error ``tolerance``, so
        ``timestep`` is only the first trial step; ``"rk4"``, ``"dopri5"`` and
        ``"dopri8"`` take fixed steps of ``timestep``.
    compact:
        Enable survivor compaction when supported; ``False`` disables it.
    progress:
        ``None``, or ``progress(done, total)``, called as the horizon advances
        (ESSOS runs it in host-side chunks; the orbits are unchanged).
    devices:
        JAX devices the particles are split over.  ``None`` uses every CPU
        device but only the first GPU: each adaptive step ends with a
        cross-device check, which costs little between CPU cores but made
        two A4000s 3x slower than one.  Pass ``jax.devices()`` to split
        over every GPU anyway.
    """
    import jax

    require_optional("essos", "alpha-particle tracing")
    from essos import constants
    from essos.boozer import trace_boozer

    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    if not (np.isfinite(tolerance) and tolerance > 0):
        raise ValueError("tolerance must be positive and finite")
    compact = compact is None or bool(compact)
    if devices is None:
        devices = jax.devices()
        if jax.default_backend() != "cpu":
            devices = devices[:1]
    devices = list(devices)
    trace_kwargs = dict(compact=compact, method=method)
    if method.startswith("adaptive"):
        trace_kwargs["tolerance"] = float(tolerance)

    from .scaling import SCALE_TARGETS, aries_cs_scales, scale_wout
    from .wout import read_wout

    wout = source if hasattr(source, "rmnc") else read_wout(source)
    b_scale = r_scale = 1.0
    if scale is not None:
        b_scale, r_scale = aries_cs_scales(wout, scale)
        wout = scale_wout(wout, b_scale=b_scale, r_scale=r_scale)
    if timestep is None:
        timestep = TIMESTEP * float(wout.Aminor_p) / SCALE_TARGETS["volavgB"][1]
    if not _COMPILE_S[1]:
        jax.monitoring.register_event_duration_secs_listener(_compile_listener)
        _COMPILE_S[1] = 1.0
    compile_start = _COMPILE_S[0]
    field, bx = boozer_field(wout, mboz=mboz, nboz=nboz, mode_tolerance=mode_tolerance)
    births = sample_births(field, nparticles, s=s, birth=birth, seed=seed, ne0=ne0, T0_keV=T0_keV)
    mass, charge = constants.ALPHA_PARTICLE_MASS, constants.ALPHA_PARTICLE_CHARGE
    energy = constants.FUSION_ALPHA_PARTICLE_ENERGY
    start = time.perf_counter()
    trace = trace_boozer(
        field, *births.T, speed=float(np.sqrt(2 * energy / mass)), mass=mass,
        charge=charge, tmax=float(tmax), timestep=float(timestep),
        n_save=min(int(times_to_trace), 101), seed=int(seed),
        species=background_species(ne0, T0_keV) if collisions else None,
        progress=progress, devices=devices, **trace_kwargs)
    wall = time.perf_counter() - start
    times = np.linspace(0.0, float(tmax), int(times_to_trace))
    lost = trace.loss_times >= 0
    failed = (~np.isfinite(trace.states).all(axis=(1, 2)) |
              ~np.isfinite(trace.energy_error) |
              np.asarray(getattr(trace, "failed", np.zeros(nparticles, dtype=bool))))
    if failed.any():
        raise ValueError(f"{failed.sum()} alpha trajectories failed; loss fraction is undefined. "
                         "Reduce the timestep or inspect the field")
    loss_fractions = np.array([(trace.loss_times[lost] <= t).sum() for t in times]) / nparticles
    last = -1
    keys = ("rmnc_b", "zmns_b", "numns_b")
    if bool(bx.asym):
        keys += ("rmns_b", "zmnc_b", "numnc_b")
    boundary = {key: np.asarray(getattr(bx, key))[:, last] for key in keys}
    result = AlphaTracingResult(
        nparticles=int(nparticles), loss_fraction=float(lost.mean()),
        particles_lost=int(lost.sum()),
        particles_thermalized=int((trace.thermalized_times >= 0).sum()),
        particles_failed=int(failed.sum()), wall_time_s=float(wall),
        particle_energy=float(energy), times=times, loss_fractions=loss_fractions,
        lost_times=trace.loss_times, thermalized_times=trace.thermalized_times,
        initial_conditions=births, final_states=trace.states[:, -1],
        energy_error=trace.energy_error, trajectories=trace.states,
        boozer=dict(boundary, xm_b=np.asarray(bx.xm_b), xn_b=np.asarray(bx.xn_b),
                    nfp=int(bx.nfp), asym=bool(bx.asym), s=np.asarray(bx.s_b), iota=np.asarray(bx.iota)),
    )
    result.metadata.update(
        tmax=float(tmax), timestep=float(timestep), s=float(s), seed=int(seed),
        birth=birth, collisions=bool(collisions), ne0=float(ne0), T0_keV=float(T0_keV),
        method=method, tolerance=float(tolerance) if method.startswith("adaptive") else None,
        integrator=(f"adaptive {'Dopri8' if method == 'adaptive8' else 'Dopri5'}, tolerance {tolerance:g}"
                    if method.startswith("adaptive") else f"{method.upper()}") + " (Boozer guiding centre)",
        compact=bool(compact), boozer_modes=int(field.xm.size),
        mode_tolerance=float(mode_tolerance), mboz=int(mboz), nboz=int(nboz),
        scale_target=scale, b_scale=b_scale, r_scale=r_scale,
        volavgB=float(wout.volavgB), Aminor_p=float(wout.Aminor_p),
        compile_time_s=_COMPILE_S[0] - compile_start,
        devices=len(devices), platform=devices[0].platform,
        # vmex's own version also covers a source tree without installed package metadata.
        versions={"vmex": import_module("vmex").__version__,
                  **{name: version(name) for name in ("essos", "jax", "booz_xform_jax")}},
    )
    return result
