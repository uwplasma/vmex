"""``vmex --turbulence``: a quick gyrokinetic look at a solved equilibrium.

Samples one flux tube of a WOUT (default ``s = 0.5``, ``alpha = 0``) and runs
`GKX <https://github.com/uwplasma/GKX>`_ (``pip install 'vmex[turbulence]'``)
through its public API:

1. a linear ``k_y`` scan (growth rate and real frequency of the dominant mode),
2. one linear run at the peak ``k_y`` for the ballooning eigenfunction,
3. a short nonlinear electrostatic ITG simulation with adiabatic electrons
   (optionally kinetic electrons), stopped by GKX's saturation policy or by
   ``t_max``.

The defaults follow a GPU resolution study (96 x 96, (4, 8) moments, saturation window from
``t = 300``) and still carry about 10% from the random seed alone, which the printout states.  Every
number is in GKX's normalized units (``rho_star`` diagnostic norm; ``k_y rho_i``, ``gamma a / v_ti``,
``Q / Q_gB``).  The deck GKX ran is written next to the figures so the same case can be rerun at
other settings with the ``gkx`` command.

Geometry panels use :func:`vmex.core.turbulence.gk_fieldline_geometry_from_wout`
(GS2/GX normalization); GKX itself builds its flux tube from the same WOUT
through ``booz_xform_jax``.
"""

from __future__ import annotations

import json
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

__all__ = ["TurbulenceSettings", "plot_turbulence", "run_turbulence", "saturated_mean"]


@dataclass(frozen=True)
class TurbulenceSettings:
    """Settings of ``vmex --turbulence``.

    The command line exposes ``s``, ``alpha``, the gradients, kinetic electrons and the ``t_max``
    cap; everything else is set here (Python API) or in the GKX decks written next to the output.
    ``nx``, ``ny`` and ``nz`` default to the resolution validated for the electron model.
    """

    s: float = 0.5  # normalized toroidal flux of the flux tube
    alpha: float = 0.0  # field-line label
    tprim: float = 3.0  # a/L_Ti (ITG drive; Cyclone-like, GX stellarator papers)
    fprim: float = 1.0  # a/L_n
    ky_min: float = 0.1
    ky_max: float = 1.0
    nky: int = 8
    nx: int | None = None  # nonlinear grid; nz points per poloidal turn
    ny: int | None = None
    nz: int | None = None
    npol: float = 3.0  # poloidal turns of the flux tube (VMEC field lines need >= 3)
    nl: int = 4  # ion Laguerre moments
    nm: int = 8  # ion Hermite moments
    t_max: float = 800.0  # nonlinear cap, a / v_ti; GKX stops earlier on saturation
    t_sat_min: float = 300.0  # earliest time the saturation window may start
    kinetic_electrons: bool = False
    d_hyper: float = 0.05  # k_perp hyperdiffusion coefficient
    precision: str = "f32"  # "f32" or "f64"
    out: str | None = None  # GKX NetCDF bundle; default <outdir>/<case>_turbulence.out.nc

    def __post_init__(self) -> None:
        nx, ny, nz = (96, 96, 24) if not self.kinetic_electrons else (16, 16, 16)
        for name, value in (("nx", nx), ("ny", ny), ("nz", nz)):
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)

    @property
    def ntheta(self) -> int:
        """Total parallel points ``nz * npol``."""
        return int(round(self.nz * self.npol))


# Linear-run horizon (a / v_ti, capped at t_max); GKX falls back to a Krylov eigensolve when the
# time fit is not a clean growing mode.
_T_LINEAR = 50.0
_NU_ION = 0.01
_BETA_KINETIC = 1e-4  # small finite beta with A_par: GKX's kinetic-electron stepper needs it
_ELECTRON_MOMENTS = (4, 48)  # (Nl, Nm), GKX's nonlinear kinetic-electron default
_M_ELECTRON = 0.00027  # m_e / m_i = 1 / 3670
# The linear time route's measured CFL step (0.9 of the bound) diverges with kinetic electrons on VMEC
# lines (non-finite fields on the 32 x 32 x 48 QA and shaped-tokamak decks, f32 and f64); 0.2 is finite and
# gives the same rate as 0.1.
_LINEAR_CFL_KINETIC = 0.2


_DECK = """schema_version = 1
# Written by vmex --turbulence. Rerun with: gkx run {name}
{species}
[grid]
Nx = {nx}
Ny = {ny}
Nz = {ntheta}
Lx = {lx:.6g}
Ly = {ly:.6g}
boundary = "{boundary}"
y0 = {y0:.6g}
ntheta = {ntheta}
nperiod = 1
{jtwist}
[time]
t_max = {t_max:.6g}
sample_stride = 5
diagnostics_stride = 5
{time_extra}
[geometry]
model = "vmec"
vmec_file = "{wout}"
torflux = {s:.6g}
alpha = {alpha:.6g}
npol = {npol:.6g}

[init]
# Seed every (kx, ky) mode: a single seeded mode never couples nonlinearly.
init_field = "density"
init_amp = 1.0e-3
init_single = false
gaussian_init = false

[physics]
linear = {linear}
nonlinear = {nonlinear}
electrostatic = {electrostatic}
adiabatic_electrons = {adiabatic}
tau_e = 1.0
collisions = true
hypercollisions = true
{beta}
[collisions]
# k_perp hyperdiffusion, as in GKX's nonlinear stellarator decks.
D_hyper = {d_hyper:.6g}

[terms]
# GKX leaves both off unless asked.
nonlinear = {nl_term}
hyperdiffusion = 1.0
apar = {apar}
bpar = 0.0

[normalization]
contract = "kinetic"
diagnostic_norm = "rho_star"
"""

_SPECIES = """
[[species]]
name = "{name}"
charge = {charge}
mass = {mass}
density = 1.0
temperature = 1.0
tprim = {tprim:.6g}
fprim = {fprim:.6g}
nu = {nu}
kinetic = true
{moments}"""


def flux_tube_box(cfg: TurbulenceSettings) -> dict:
    """Radial box of the nonlinear flux tube: the one place the ``jtwist`` / ``Lx`` rule lives.

    ``jtwist = None`` leaves GKX's own rule (``jtwist = round(2 s_hat gds21 / gds22)`` at the end
    of the line, ``x0 = y0 jtwist / that factor``, ``Lx = 2 pi x0``); an integer is written to the deck.
    """
    y0 = 1.0 / cfg.ky_min
    return {"y0": y0, "lx": 2 * np.pi * y0, "ly": 2 * np.pi * y0, "jtwist": None}


def twist_estimate(geom: dict, cfg: TurbulenceSettings) -> tuple[int, float]:
    """``(jtwist, Lx)`` GKX's default rule gives, from the VMEX field-line geometry (for the printout)."""
    fac = abs(2.0 * float(geom["s_hat"]) * float(geom["gds21"][0]) / float(geom["gds22"][0]))
    jt = max(1, int(round(fac)))
    return jt, 2 * np.pi * jt / (cfg.ky_min * max(fac, 1e-12))


def write_deck(path: Path, wout: Path, cfg: TurbulenceSettings, *, nonlinear: bool,
               saturation: bool = True) -> Path:
    """Write the GKX TOML deck for the linear scan or the nonlinear run."""
    species = _SPECIES.format(name="ion", charge=1.0, mass=1.0, tprim=cfg.tprim, fprim=cfg.fprim, nu=_NU_ION,
                              moments=f"Nl = {cfg.nl}\nNm = {cfg.nm}\n" if cfg.kinetic_electrons else "")
    if cfg.kinetic_electrons:
        species += _SPECIES.format(name="electron", charge=-1.0, mass=_M_ELECTRON, tprim=cfg.tprim, fprim=cfg.fprim,
                                   nu=0.0, moments="Nl = {}\nNm = {}\n".format(*_ELECTRON_MOMENTS))
    box = flux_tube_box(cfg)
    es = not cfg.kinetic_electrons
    path.write_text(_DECK.format(
        name=path.name, species=species, wout=wout.resolve(), s=cfg.s, alpha=cfg.alpha,
        t_max=cfg.t_max if nonlinear else min(_T_LINEAR, cfg.t_max),
        nx=cfg.nx if nonlinear else 1, ny=cfg.ny if nonlinear else 3 * int(ky_grid(cfg)[-1] * box["y0"] + 0.5) + 1,
        ntheta=cfg.ntheta, npol=cfg.npol, lx=box["lx"], ly=box["ly"], y0=box["y0"],
        boundary="fix aspect" if nonlinear else "linked",
        linear=str(not nonlinear).lower(), nonlinear=str(nonlinear).lower(),
        electrostatic=str(es).lower(), adiabatic=str(es).lower(),
        beta="" if es else f"use_apar = true\nuse_bpar = false\nbeta = {_BETA_KINETIC:g}\n",
        time_extra=(f"saturation_t_min = {cfg.t_sat_min:g}\n" if saturation else "") if nonlinear
        else (f"cfl = {_LINEAR_CFL_KINETIC:g}\n" if cfg.kinetic_electrons else ""),
        jtwist="" if box["jtwist"] is None else f"jtwist = {box['jtwist']}\n",
        d_hyper=cfg.d_hyper, nl_term=1.0 if nonlinear else 0.0, apar=0.0 if es else 1.0,
    ))
    return path


def ky_grid(cfg: TurbulenceSettings) -> np.ndarray:
    """Scan points: ``nky`` values in ``[ky_min, ky_max]`` snapped to multiples of ``ky_min``.

    GKX's box has ``k_y = j / y0``; with ``y0 = 1 / ky_min`` every scan point is a grid mode.
    """
    j = np.unique(np.rint(np.linspace(cfg.ky_min, cfg.ky_max, cfg.nky) / cfg.ky_min).astype(int))
    return cfg.ky_min * j[j >= 1]


def saturated_mean(t, q, *, t_start: float | None = None, batches: int = 8) -> tuple[float, float, float]:
    """Mean and batch-means standard error of ``q`` for ``t >= t_start``.

    Returns ``(mean, sem, t_start)``.  Without ``t_start`` the window is the
    last half of the run, the usual convention for a short flux-tube run.
    """
    t = np.asarray(t, float)
    q = np.asarray(q, float)
    if t.size == 0:
        return float("nan"), float("nan"), float("nan")
    t0 = 0.5 * (t[0] + t[-1]) if t_start is None else float(t_start)
    w = q[t >= t0]
    if w.size < 2 * batches:
        return float(np.mean(w)), float(np.std(w) / np.sqrt(max(w.size, 1))), float(t0)
    means = np.array([b.mean() for b in np.array_split(w, batches)])
    return float(w.mean()), float(means.std(ddof=1) / np.sqrt(batches)), float(t0)


class _Stages:
    """Print stage banners, elapsed time and a running ETA."""

    def __init__(self, emit, weights: dict[str, float]):
        self.emit, self.weights, self.done, self.start = emit, weights, 0.0, time.perf_counter()
        self.total = sum(weights.values())
        self.t_stage = self.start

    def begin(self, key: str, text: str) -> None:
        elapsed = time.perf_counter() - self.start
        eta = ""
        if self.done > 0:
            eta = f", about {elapsed * (self.total - self.done) / self.done:.0f} s left"
        self.emit(f" [{elapsed:6.1f} s{eta}] {text}")
        self.t_stage, self.key = time.perf_counter(), key

    def end(self, text: str = "") -> float:
        dt = time.perf_counter() - self.t_stage
        self.done += self.weights[self.key]
        self.emit(f"          done in {dt:.1f} s{('  ' + text) if text else ''}")
        return dt


@contextmanager
def _gkx_warnings(emit):
    """Show GKX's fit-quality warnings as one indented line each."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    notes = dict.fromkeys(str(w.message).split(" -- ")[0].split("\n")[0] for w in caught
                          if issubclass(w.category, RuntimeWarning))
    for msg in list(notes)[:3]:
        emit(f"          GKX note: {msg}")
    if len(notes) > 3:
        emit(f"          (+{len(notes) - 3} more GKX fit/step notes)")


def _gkx():
    from .._compat import require_optional

    require_optional("gkx", "vmex --turbulence")
    import gkx

    return gkx


UNCERTAINTY = (
    "the printed +/- is the statistical error of one run; the random initial seed alone moves Q by about 10% on "
    "tokamaks, (4, 8) moments read 8-14% above (6, 12), 96 x 96 is within 7% of 128 x 128"
)


def settings_block(wout: Path, cfg: TurbulenceSettings, *, backend: str, out: Path, geom: dict,
                   rel_sem: float | None = None, t_sat_min: float | None = None) -> list[str]:
    """Everything that determines the answer, one compact block (mirrors the ``--trace`` banner)."""
    box = flux_tube_box(cfg)
    jt, lx = twist_estimate(geom, cfg) if box["jtwist"] is None else (box["jtwist"], box["lx"])
    kin = cfg.kinetic_electrons
    ele = (f"kinetic (m_e/m_i = 1/3670, nu_e = 0, (Nl, Nm) = {_ELECTRON_MOMENTS}, beta = {_BETA_KINETIC:g} with A_par)"
           if kin else "adiabatic (tau_e = 1)")
    window = ("window starts no earlier than t = " + f"{t_sat_min:g}" if t_sat_min is not None else
              "GKX's default window start (this GKX predates saturation_t_min)")
    stop = (f"stop when the heat-flux mean has SEM <= {rel_sem:.0%} and is stationary" if rel_sem is not None
            else "stop on GKX's saturation rule")
    return [
        f" Gyrokinetic turbulence on {wout.name} ({backend}, {'float64' if cfg.precision == 'f64' else 'float32'})",
        f"   flux tube : s = {cfg.s:g} (rho = {np.sqrt(cfg.s):.3f}), alpha = {cfg.alpha:g}, npol = {cfg.npol:g}, "
        f"q = {float(geom['q']):.3f}, s_hat = {float(geom['s_hat']):.3f}",
        f"   drive     : a/LT = {cfg.tprim:g}, a/Ln = {cfg.fprim:g} (every species); electrons {ele}",
        f"   box       : Lx = {lx:.1f}, Ly = {box['ly']:.1f} (ky_min = {cfg.ky_min:g}, y0 = {box['y0']:g}), "
        f"jtwist = {jt}" + ("" if box["jtwist"] is not None else " (GKX default rule)"),
        f"   grid      : Nx x Ny x Nz = {cfg.nx} x {cfg.ny} x {cfg.ntheta} ({cfg.nz} per turn), ions (Nl, Nm) = ({cfg.nl}, {cfg.nm})",
        f"   time      : GKX CFL-controlled stepper, cap t_max = {cfg.t_max:g} a/v_ti; {stop}; {window}",
        f"   collisions: nu_i = {_NU_ION:g}; hypercollisions on; k_perp hyperdiffusion D = {cfg.d_hyper:g} "
        "(normalized to the grid cutoff)",
        f"   output    : {out}",
        f"   uncertainty: {UNCERTAINTY}",
    ]


def run_turbulence(wout_path: Path, outdir: Path, cfg: TurbulenceSettings = TurbulenceSettings(),
                   *, emit=print) -> dict:
    """Linear ``k_y`` scan + nonlinear run on one flux tube; returns the summary dict."""
    import jax

    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", cfg.precision == "f64")
    try:
        return _run(wout_path, outdir, cfg, emit)
    finally:
        jax.config.update("jax_enable_x64", prev)


def _run(wout_path: Path, outdir: Path, cfg: TurbulenceSettings, emit) -> dict:
    import dataclasses

    from gkx.config import TimeConfig

    from .turbulence import gk_fieldline_geometry_from_wout

    gkx = _gkx()
    import jax

    wout_path = Path(wout_path).resolve()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    label = wout_path.stem.removeprefix("wout_")
    out = Path(cfg.out) if cfg.out else outdir / f"{label}_turbulence.out.nc"
    stages = _Stages(emit, {"geom": 1.0, "scan": 3.0, "mode": 1.0, "nl": 6.0})

    stages.begin("geom", "Sampling field-line geometry")
    geom = {k: v if isinstance(v, dict) else np.asarray(v) for k, v in gk_fieldline_geometry_from_wout(
        wout_path, s=cfg.s, alpha=cfg.alpha, ntheta=128, equal_arc=False).items()}
    stages.end(f"q={float(geom['q']):.3f}, s_hat={float(geom['s_hat']):.3f}")
    if not abs(float(geom["q"])) < 1e3:
        raise ValueError(f"iota is ~0 at s={cfg.s:g}: the field line never closes poloidally, so no flux tube is defined")

    # GKX releases up to 2.9.0 have no saturation_t_min and reject it in a deck.
    has_t_min = "saturation_t_min" in {f.name for f in dataclasses.fields(TimeConfig)}
    lin_case = gkx.load(write_deck(outdir / f"{label}_gkx_linear.toml", wout_path, cfg, nonlinear=False))
    nl_case = gkx.load(write_deck(outdir / f"{label}_gkx_nonlinear.toml", wout_path, cfg, nonlinear=True,
                                  saturation=has_t_min))
    for line in settings_block(wout_path, cfg, backend=f"GKX {gkx.__version__}, {jax.default_backend()}", out=out,
                               geom=geom, rel_sem=getattr(nl_case.time, "saturation_rel_sem", None),
                               t_sat_min=cfg.t_sat_min if has_t_min else None):
        emit(line)

    ky = ky_grid(cfg)
    lin_nm = _ELECTRON_MOMENTS[1] if cfg.kinetic_electrons else max(cfg.nm, 8)
    lin_nl = max(cfg.nl, 4)
    stages.begin("scan", f"Linear ky scan ({ky.size} points)")
    with _gkx_warnings(emit):
        scan = gkx.scan(lin_case, ky, Nl=lin_nl, Nm=lin_nm)
    gamma, omega = np.asarray(scan.gamma, float), np.asarray(scan.omega, float)
    ipk = int(np.nanargmax(gamma))
    stages.end(f"peak gamma={gamma[ipk]:.4f} at ky={ky[ipk]:.3g} (omega={omega[ipk]:.4f})")

    stages.begin("mode", f"Linear eigenfunction at ky={ky[ipk]:.3g}")
    with _gkx_warnings(emit):
        mode = gkx.run_runtime_linear(lin_case, ky_target=float(ky[ipk]), Nl=lin_nl, Nm=lin_nm)
    stages.end(f"gamma={float(mode.gamma):.4f}")

    stages.begin("nl", f"Nonlinear run to saturation or t={cfg.t_max:g}")
    with _gkx_warnings(emit):
        try:  # GKX >= the progress reporter (GKX #495); older GKX gets its raw status lines
            from gkx.workflows.runtime.chunks import ProgressReporter

            rep = ProgressReporter()
        except ImportError:
            rep = lambda msg: emit(f"          {msg}")  # noqa: E731
        nl = gkx.solve(nl_case, Nl=cfg.nl, Nm=_ELECTRON_MOMENTS[1] if cfg.kinetic_electrons else cfg.nm, return_state=True, status_callback=rep)
        if hasattr(rep, "finish"):  # summary line and Q(t) / Q(ky) plots next to the output file
            rep.finish(nl, plot=str(out).removesuffix(".out.nc").removesuffix(".nc"),
                       ky=np.asarray(gkx.build_spectral_grid(nl_case.grid).ky))
    d = nl.diagnostics
    t = np.asarray(d.t, float)
    # GKX's own decision (window, mean, SEM, status); the last half of the run when it made none.
    sat = nl.saturation or {}
    t_win = sat.get("window_tmin")
    q_mean, q_err, t_sat = saturated_mean(t, d.heat_flux_t, t_start=t_win)
    if sat.get("mean") is not None and sat.get("sem") is not None:
        q_mean, q_err = float(sat["mean"]), float(sat["sem"])
    g_mean, g_err, _ = saturated_mean(t, d.particle_flux_t, t_start=t_win)
    status = "saturated" if sat.get("saturated", False) else "NOT saturated"
    if status != "saturated":
        emit(f"          not saturated by t_max = {cfg.t_max:g}: the mean is indicative only; raise --turbulence-tmax")
    stages.end(f"t_end={t[-1]:.1f}, Q/Q_gB={q_mean:.4g} +/- {q_err:.2g}")

    # GKX NetCDF bundle (traces, spectra, saturation verdict): an internal GKX call since 2.6.0; no public writer yet.
    try:
        from gkx.workflows.runtime.artifacts import write_runtime_nonlinear_artifacts
    except ImportError as exc:
        raise ImportError(f"vmex --turbulence needs gkx.workflows.runtime.artifacts.write_runtime_nonlinear_artifacts, "
                          f"absent from GKX {gkx.__version__}; install a GKX release that provides it (2.6.0 does)") from exc

    write_runtime_nonlinear_artifacts(out, nl, nl_case)
    summary = {
        "wout": wout_path.name, "settings": asdict(cfg), "gkx_version": gkx.__version__,
        "backend": jax.default_backend(), "q": float(geom["q"]), "shat": float(geom["s_hat"]),
        "ky": ky.tolist(), "gamma": gamma.tolist(), "omega": omega.tolist(),
        "ky_peak": float(ky[ipk]), "gamma_peak": float(gamma[ipk]),
        "t_end": float(t[-1]), "window_start": t_sat,
        "heat_flux": [q_mean, q_err], "particle_flux": [g_mean, g_err],
        "status": status, "saturation": nl.saturation, "output": str(out),
        "steps": int(getattr(nl.diagnostics, "n_steps", 0) or 0), "wall_time_s": time.perf_counter() - stages.start,
    }
    grid = gkx.build_spectral_grid(nl_case.grid)
    written = plot_turbulence(summary, geom, mode, nl, np.asarray(grid.ky), outdir, name=label)
    path = outdir / f"{label}_turbulence.json"
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    written["json"] = path
    emit(f" Turbulence summary: Q_i/Q_gB = {q_mean:.4g} +/- {q_err:.2g}, "
         + (f"Gamma/Gamma_gB = {g_mean:.3g} +/- {g_err:.2g}, " if cfg.kinetic_electrons else "")
         + f"(t >= {t_sat:.1f}), "
         f"gamma_max = {gamma[ipk]:.4f} v_ti/a at ky rho_i = {ky[ipk]:.3g}; "
         f"{summary['wall_time_s']:.0f} s total, {status}")
    emit(f"   Uncertainty: {UNCERTAINTY}")
    for key, p in written.items():
        emit(f"   Saved {key}: {p}")
    summary["files"] = {k: str(v) for k, v in written.items()}
    return summary


def _fold_ky(ky, values):
    """Sum an FFT-ordered ``k_y`` axis (last) onto ``|k_y|``; returns ``(|k_y|, folded)``."""
    key = np.round(np.abs(np.asarray(ky)), 8)
    uniq = np.unique(key)
    out = np.zeros(values.shape[:-1] + (uniq.size,))
    for i, k in enumerate(uniq):
        out[..., i] = values[..., key == k].sum(axis=-1)
    return uniq, out


def plot_turbulence(summary: dict, geom: dict, mode, nl, ky_grid, outdir: str | Path,
                    *, name: str = "turbulence") -> dict[str, Path]:
    """Write the ``--turbulence`` figure set .

    ``<name>_turbulence.png`` is the one-page summary in the layout of the
    GX/stella stellarator papers (Mandell et al. 2024; Landreman et al.
    2025; Gonzalez-Jerez et al. 2022): heat-flux trace with its saturated
    mean +/- standard error, linear ``gamma(k_y)``/``omega(k_y)``,
    time-averaged nonlinear ``Q(k_y)`` and ``|phi|^2(k_y)`` spectra, a
    snapshot of ``phi(x, y)`` at the outboard midplane, the linear
    eigenfunction ``phi(theta)`` over ``|B|``, and zonal vs non-zonal
    ``|phi|^2``.  ``<name>_turbulence_geometry.png`` shows ``|B|``, the
    grad-B and curvature drifts, and ``gds2/gds21/gds22`` along the field
    line; ``<name>_turbulence_fluxes.png`` the heat and particle flux traces
    on linear and logarithmic axes.
    """
    from .plotting import _LINE_COLORS, _ensure_outdir, _import_matplotlib, _rc_context, _save_figure

    plt = _import_matplotlib()
    outdir = _ensure_outdir(outdir)
    d = nl.diagnostics
    rd = d.resolved
    t = np.asarray(d.t, float)
    q_t, g_t = np.asarray(d.heat_flux_t, float), np.asarray(d.particle_flux_t, float)
    (qm, qe), (gm, ge), t0 = summary["heat_flux"], summary["particle_flux"], summary["window_start"]
    win = t >= t0
    th = np.asarray(geom["theta"]) / np.pi
    cfg = summary["settings"]
    written = {}

    def _flux(ax, y, mean, err, label, log=False):
        ax.plot(t, y, color=_LINE_COLORS[0])
        if not log:
            ax.axvspan(t0, t[-1], color="0.85", zorder=0)
            ax.axhline(mean, color=_LINE_COLORS[1], ls="--", label=f"{mean:.3g} ± {err:.2g}")
            ax.axhspan(mean - err, mean + err, color=_LINE_COLORS[1], alpha=0.25)
            ax.legend(loc="upper left")
        else:
            ax.set_yscale("log")
        ax.set(xlabel=r"$t\,v_{ti}/a$", ylabel=label)

    title = (f"{name}: s={cfg['s']:g}, α={cfg['alpha']:g}, q={summary['q']:.3f}, "
             f"ŝ={summary['shat']:.3f}, a/L_Ti={cfg['tprim']:g}, a/L_n={cfg['fprim']:g}, "
             f"{'kinetic' if cfg['kinetic_electrons'] else 'adiabatic'} electrons — "
             f"Q/Q_gB = {qm:.3g} ± {qe:.2g}")
    with _rc_context():
        fig, ax = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
        _flux(ax[0, 0], q_t, qm, qe, r"ion heat flux $Q_i/Q_{gB}$")
        ax[0, 0].set_title("heat flux (shaded: averaging window)")

        a = ax[0, 1]
        ky, gam, om = (np.asarray(summary[k]) for k in ("ky", "gamma", "omega"))
        a.plot(ky, gam, "o-", label=r"$\gamma$")
        a.set(xlabel=r"$k_y\rho_i$", ylabel=r"$\gamma\,a/v_{ti}$", title="linear growth rate and frequency")
        a2 = a.twinx()
        a2.plot(ky, om, "s--", color=_LINE_COLORS[1], label=r"$\omega$")
        a2.set_ylabel(r"$\omega\,a/v_{ti}$")
        a2.grid(False)
        a.legend(handles=a.get_lines() + a2.get_lines(), loc="upper left")

        a = ax[0, 2]
        kyp, qk = _fold_ky(ky_grid, np.asarray(rd.HeatFlux_kyst, float).sum(axis=1)[win].mean(axis=0)[None])
        _, pk = _fold_ky(ky_grid, np.asarray(rd.Phi2_kyt, float)[win].mean(axis=0)[None])
        keep = qk[0] != 0
        a.bar(kyp[keep], qk[0][keep], width=0.8 * np.min(np.diff(kyp)), label=r"$Q(k_y)$")
        a.set_xlim(0, 1.05 * kyp[keep].max())
        a.set(xlabel=r"$k_y\rho_i$", ylabel=r"$\langle Q\rangle_t(k_y)$", title="nonlinear spectra (window mean)")
        a2 = a.twinx()
        a2.semilogy(kyp[1:], pk[0][1:], "o-", color=_LINE_COLORS[1], label=r"$|\phi|^2(k_y)$")
        a2.set_ylabel(r"$\langle|\phi|^2\rangle_t$")
        a2.grid(False)
        a.legend(handles=[*a.containers, *a2.get_lines()], loc="upper right")

        a = ax[1, 0]
        phi = np.asarray(nl.fields.phi)
        iz = int(np.argmin(np.abs(np.linspace(-np.pi, np.pi, phi.shape[-1], endpoint=False))))
        # zero-padded inverse FFT: a smooth 128x128 picture of the same modes
        pk = np.fft.fftshift(phi[:, :, iz])
        pad = [((m - n) // 2, m - n - (m - n) // 2) for n in pk.shape for m in [max(128, n)]]
        xy = np.real(np.fft.ifft2(np.fft.ifftshift(np.pad(pk, pad))))
        ext = 2 * np.pi / cfg["ky_min"]
        vmax = float(np.max(np.abs(xy))) or 1.0
        im = a.imshow(xy / vmax, origin="lower", extent=(0, ext, 0, ext), cmap="RdBu_r", vmin=-1, vmax=1,
                      aspect="equal")
        fig.colorbar(im, ax=a, label=r"$\phi/\max|\phi|$")
        a.set(xlabel=r"$x/\rho_i$", ylabel=r"$y/\rho_i$", title=f"φ(x, y), outboard midplane, t={t[-1]:.0f}")
        a.grid(False)

        a = ax[1, 1]
        if mode.eigenfunction is not None and mode.z is not None:
            z, ef = np.asarray(mode.z, float) / np.pi, np.asarray(mode.eigenfunction)
            ef = ef / ef[np.argmax(np.abs(ef))]
            a.plot(z, np.abs(ef), label=r"$|\phi|$")
            a.plot(z, ef.real, "--", label=r"Re $\phi$")
            a.plot(z, ef.imag, ":", label=r"Im $\phi$")
        else:  # GKX's Krylov route returns no eigenfunction (e.g. a damped mode)
            a.text(0.5, 0.5, "no eigenfunction returned", ha="center", transform=a.transAxes)
        a.set(xlabel=r"$\theta/\pi$", ylabel=r"$\phi/\phi_{\max}$",
              title=f"linear eigenfunction, $k_y\\rho_i$={summary['ky_peak']:.2g}")
        a2 = a.twinx()
        a2.plot(th, geom["bmag"], color="0.5", lw=1.0, label="|B|")
        a2.set_ylabel(r"$|B|/B_{ref}$")
        a2.grid(False)
        a.legend(loc="upper right")

        a = ax[1, 2]
        p2 = np.asarray(rd.Phi2_kyt, float).sum(axis=1)
        zon = np.asarray(rd.Phi2_zonal_t, float)
        a.semilogy(t, zon, label="zonal")
        a.semilogy(t, np.maximum(p2 - zon, 1e-300), label="non-zonal")
        a.set(xlabel=r"$t\,v_{ti}/a$", ylabel=r"$|\phi|^2$", title="zonal vs non-zonal energy")
        a.legend(loc="lower right")
        fig.suptitle(title)
        written["summary"] = outdir / f"{name}_turbulence.png"
        _save_figure(fig, written["summary"], dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained", sharex=True)
        ax[0, 0].plot(th, geom["bmag"])
        ax[0, 0].set(ylabel=r"$|B|/B_{ref}$", title="field strength")
        ax[0, 1].plot(th, geom["gbdrift"], label="gbdrift")
        ax[0, 1].plot(th, geom["cvdrift"], "--", label="cvdrift")
        ax[0, 1].axhline(0, color="0.4", lw=0.8)
        ax[0, 1].set(title="grad-B and curvature drift (<0: bad curvature)")
        ax[0, 1].legend()
        ax[1, 0].plot(th, geom["gds2"], label="gds2")
        ax[1, 0].plot(th, geom["gds22"], "--", label="gds22")
        ax[1, 0].set(xlabel=r"$\theta/\pi$", title=r"$|\nabla\alpha|^2$, $|\nabla\psi|^2$ metrics")
        ax[1, 0].legend()
        ax[1, 1].plot(th, geom["gds21"])
        ax[1, 1].set(xlabel=r"$\theta/\pi$", title="gds21 (local shear)")
        fig.suptitle(f"{name}: flux-tube geometry, s={cfg['s']:g}, α={cfg['alpha']:g}")
        written["geometry"] = outdir / f"{name}_turbulence_geometry.png"
        _save_figure(fig, written["geometry"], dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained", sharex=True)
        _flux(ax[0, 0], q_t, qm, qe, r"$Q_i/Q_{gB}$")
        _flux(ax[0, 1], np.abs(q_t) + 1e-300, qm, qe, r"$|Q_i|/Q_{gB}$", log=True)
        _flux(ax[1, 0], g_t, gm, ge, r"$\Gamma/\Gamma_{gB}$")
        ax[1, 1].semilogy(t, np.asarray(d.Wg_t, float), label=r"$W_g$")
        ax[1, 1].semilogy(t, np.asarray(d.Wphi_t, float), label=r"$W_\phi$")
        ax[1, 1].set(xlabel=r"$t\,v_{ti}/a$", title="free energy")
        ax[1, 1].legend()
        ax[0, 0].set_title("heat flux")
        ax[0, 1].set_title("heat flux, log scale (linear phase)")
        ax[1, 0].set_title("particle flux" + (" (zero with adiabatic electrons)"
                                              if not cfg["kinetic_electrons"] else ""))
        fig.suptitle(title)
        written["fluxes"] = outdir / f"{name}_turbulence_fluxes.png"
        _save_figure(fig, written["fluxes"], dpi=130)
        plt.close(fig)
    return written
