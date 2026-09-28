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
``(G + iota I) / B^2``.  The plasma profiles, for volume births and for
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
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

# Converged RK4 step [s] at Aminor_p = 1.7044 m, scaled with Aminor_p, and
# the relative amplitude below which Boozer |B| modes are dropped
# (docs/howto/trace-alpha-particles.md, convergence table).
TIMESTEP = 1.25e-7
MODE_TOLERANCE = 1e-3
# Landreman, Buller & Drevlak (2022) profiles: n_e0 [m^-3], T_0 [keV].
NE0, T0_KEV = 4e20, 12.0
_COMPILE_S = [0.0, 0.0]  # compile seconds, listener registered


def _compile_listener(event: str, duration: float, **_: Any) -> None:
    if event.startswith("/jax/core/compile/"):
        _COMPILE_S[0] += float(duration)


def _essos_imports():
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
    c = (1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3, 1.35e-2, -1.0675e-4, 1.366e-5)
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
    ``-1`` for particles without that outcome.  ``energy_error`` is the largest
    relative change of the orbit energy over one step, per particle.
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
    """Return the ``essos.fields.Vmec`` field for an equilibrium or wout file.

    ``source`` is a path to a ``wout_*.nc`` file or an in-memory
    :class:`~vmex.core.wout.WoutData`.  Released ESSOS reads a wout *file*,
    so an in-memory equilibrium is written to a temporary wout; ESSOS loads
    every table eagerly in its constructor, so the file is gone by the time
    the field is returned.  That write severs the gradient — this seam is
    for diagnostics, not for differentiating through ESSOS.

    ``kwargs`` reach ``essos.fields.Vmec`` unchanged (``ntheta``, ``nphi``,
    ``close`` and ``range_torus`` on the released constructor, which set the
    resolution of the ``field.surface`` ESSOS builds alongside the field).

    Released ESSOS reads the stellarator-symmetric wout tables only, so an
    ``lasym`` equilibrium is rejected rather than silently half-transferred.
    """
    _, _, fields = _essos_imports()

    if hasattr(source, "rmnc") and hasattr(source, "xm"):  # WoutData
        if bool(source.lasym):
            raise ValueError(
                "released ESSOS reads stellarator-symmetric wout tables only; "
                "the lasym partner tables would be silently dropped"
            )
        from .wout import write_wout

        with tempfile.TemporaryDirectory(prefix="vmex_essos_") as tmp:
            wout_path = Path(tmp) / "wout_equilibrium.nc"
            write_wout(wout_path, source)
            return fields.Vmec(str(wout_path), **kwargs)

    wout_path = Path(source)
    import netCDF4

    with netCDF4.Dataset(str(wout_path)) as ds:
        if bool(int(ds.variables["lasym__logical__"][()])):
            raise ValueError(
                "released ESSOS reads stellarator-symmetric wout tables only; "
                f"{wout_path.name} is an lasym equilibrium"
            )
    return fields.Vmec(str(wout_path), **kwargs)


def boozer_field(wout, *, mboz: int = 32, nboz: int = 32, mode_tolerance: float = MODE_TOLERANCE):
    """Run ``booz_xform_jax`` on every surface of ``wout`` and return
    ``(essos.boozer.BoozerField, Booz_xform)``."""
    from booz_xform_jax import Booz_xform
    from essos.boozer import BoozerField

    from .wout import write_wout

    if bool(wout.lasym):
        raise ValueError("--trace supports stellarator-symmetric equilibria only")
    bx = Booz_xform(verbose=0, mboz=int(mboz), nboz=int(nboz))
    with tempfile.TemporaryDirectory(prefix="vmex_booz_") as tmp:
        path = Path(tmp) / "wout_trace.nc"
        write_wout(path, wout)
        bx.read_wout(str(path), flux=False)
    bx.run()
    psi0 = float(np.asarray(wout.phi)[-1]) / (2 * np.pi)
    return BoozerField.from_booz_xform(bx, psi0, mode_tolerance), bx


def sample_births(field, n: int, *, s: float = 0.25, birth: str = "surface",
                  seed: int = 42, ne0: float = NE0, T0_keV: float = T0_KEV):
    """Birth ``(s, theta_B, zeta_B, v_par/v)`` for ``n`` alphas (module notes)."""
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(int(seed))
    modB = jax.jit(jax.vmap(field.modB))
    out: list[np.ndarray] = []
    while sum(o.shape[0] for o in out) < n:
        m = 4 * n
        ss = np.full(m, float(s)) if birth == "surface" else rng.uniform(0.0, 1.0, m)
        th = rng.uniform(0.0, 2 * np.pi, m)
        ze = rng.uniform(0.0, 2 * np.pi / field.nfp, m)
        B = np.asarray(modB(jnp.asarray(ss), jnp.asarray(th), jnp.asarray(ze)))
        iota, G, current = field.profiles(jnp.asarray(ss))[0].T
        weight = np.asarray(G + iota * current) / B**2
        if birth == "volume":
            weight = weight * (ne0 / 2 * (1 - ss**5)) ** 2 * dt_reactivity(T0_keV * (1 - ss))
        elif birth != "surface":
            raise ValueError(f"birth must be 'surface' or 'volume', got {birth!r}")
        keep = rng.uniform(0.0, weight.max(), m) < weight
        out.append(np.stack([ss, th, ze, rng.uniform(-1.0, 1.0, m)], axis=1)[keep])
    return np.concatenate(out)[:n]


def trace_alphas(
    source: Any,
    *,
    tmax: float = 1e-2,
    nparticles: int = 1000,
    s: float = 0.25,
    seed: int = 42,
    timestep: float | None = None,
    times_to_trace: int = 1000,
    scale: str | None = "volavgB",
    birth: str = "surface",
    collisions: bool = False,
    ne0: float = NE0,
    T0_keV: float = T0_KEV,
    mboz: int = 32,
    nboz: int = 32,
    mode_tolerance: float = MODE_TOLERANCE,
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
    """
    import jax
    from essos import constants
    from essos.boozer import trace_boozer

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
        species=background_species(ne0, T0_keV) if collisions else None)
    wall = time.perf_counter() - start
    times = np.linspace(0.0, float(tmax), int(times_to_trace))
    lost = trace.loss_times >= 0
    failed = ~np.isfinite(trace.states).all(axis=(1, 2)) & ~lost
    loss_fractions = np.array([(trace.loss_times[lost] <= t).sum() for t in times]) / nparticles
    last = -1
    boundary = {key: np.asarray(getattr(bx, key))[:, last] for key in ("rmnc_b", "zmns_b", "numns_b")}
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
                    nfp=int(bx.nfp), s=np.asarray(bx.s_b), iota=np.asarray(bx.iota)),
    )
    result.metadata.update(
        tmax=float(tmax), timestep=float(timestep), s=float(s), seed=int(seed),
        birth=birth, collisions=bool(collisions), ne0=float(ne0), T0_keV=float(T0_keV),
        integrator="RK4 (Boozer guiding centre)", boozer_modes=int(field.xm.size),
        mode_tolerance=float(mode_tolerance), mboz=int(mboz), nboz=int(nboz),
        scale_target=scale, b_scale=b_scale, r_scale=r_scale,
        volavgB=float(wout.volavgB), Aminor_p=float(wout.Aminor_p),
        compile_time_s=_COMPILE_S[0] - compile_start,
        devices=len(jax.devices()), platform=jax.default_backend(),
        versions={name: version(name) for name in ("vmex", "essos", "jax", "booz_xform_jax")},
    )
    return result
