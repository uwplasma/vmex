"""Plotting helpers for VMEC wout data.

These utilities are adapted from the standalone `vmecPlot2.py` script, but
refactored to use vmec_jax's vectorized Fourier evaluation for speed and
consistency. The functions return NumPy arrays; any plotting backend (e.g.
matplotlib) can be layered on top.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.machinery
from pathlib import Path
import site
import sys
import types
from typing import Iterable

import numpy as np

from ._compat import has_jax, jnp
from .fourier import build_helical_basis, eval_fourier
from .geom import _eval_geom_jit, eval_geom
from .grids import AngleGrid
from .modes import ModeTable
from .field import b2_from_bsup, bsub_from_bsup, bsup_from_geom, bsup_from_sqrtg_lambda, lamscale_from_phips
from .vmec_jacobian import vmec_half_mesh_jacobian_from_state
from .vmec_realspace import vmec_realspace_geom_from_state
from .vmec_tomnsp import vmec_trig_tables
from .vmec_parity import vmec_m1_internal_to_physical_signed
from .energy import flux_profiles_from_indata
from .field import signgs_from_sqrtg
from .config import load_config
from .static import build_static
from .driver import example_paths
from .wout import read_wout, state_from_wout


@dataclass(frozen=True)
class SurfaceData:
    """Surface data on a (theta, zeta) grid."""

    R: np.ndarray
    Z: np.ndarray
    B: np.ndarray | None = None


def _is_tracer(x) -> bool:
    if not has_jax():
        return False
    try:
        import jax
    except Exception:
        return False
    return isinstance(x, jax.core.Tracer)


def fix_matplotlib_3d(ax):
    """Fix 3D matplotlib aspect so structures do not look distorted."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def prepare_matplotlib_3d() -> None:
    """Prefer the Matplotlib-matched ``mpl_toolkits`` namespace before plotting.

    Some Linux installations preload ``mpl_toolkits`` from the system
    ``dist-packages`` while importing a newer pip-installed ``matplotlib`` from
    user site-packages.  That mixed state makes ``projection="3d"`` fail with
    errors such as ``cannot import name 'docstring' from 'matplotlib'``.  If we
    detect that state and a user/site ``mpl_toolkits.mplot3d`` is available,
    replace the preloaded namespace before ``pyplot`` imports Matplotlib
    projections.
    """

    def _register_projection() -> bool:
        try:
            axes3d = importlib.import_module("mpl_toolkits.mplot3d.axes3d")
        except Exception:
            return False
        try:
            from matplotlib.projections import register_projection

            register_projection(axes3d.Axes3D)
        except Exception:
            pass
        return True

    loaded = sys.modules.get("mpl_toolkits")
    loaded_file = str(getattr(loaded, "__file__", "")) if loaded is not None else ""
    loaded_paths = [str(path) for path in getattr(loaded, "__path__", [])] if loaded is not None else []
    loaded_from_system_dist = "/usr/lib/python3/dist-packages" in loaded_file or any(
        "/usr/lib/python3/dist-packages" in path for path in loaded_paths
    )
    loaded_has_mplot3d = any((Path(path) / "mplot3d" / "axes3d.py").exists() for path in loaded_paths)
    if loaded is not None and loaded_has_mplot3d and not loaded_from_system_dist and _register_projection():
        return

    candidate_bases: list[str] = []
    for getter in (site.getusersitepackages, site.getsitepackages):
        try:
            value = getter()
        except Exception:
            continue
        if isinstance(value, str):
            candidate_bases.append(value)
        else:
            candidate_bases.extend(value)
    candidate_bases.extend(sys.path)

    toolkit_path: Path | None = None
    for base in candidate_bases:
        if not base:
            continue
        candidate = Path(base) / "mpl_toolkits"
        if "/usr/lib/python3/dist-packages" in str(candidate):
            continue
        if (candidate / "mplot3d" / "axes3d.py").exists():
            toolkit_path = candidate
            break

    if toolkit_path is None:
        return

    for name in list(sys.modules):
        if name == "mpl_toolkits" or name.startswith("mpl_toolkits."):
            del sys.modules[name]

    module = types.ModuleType("mpl_toolkits")
    module.__path__ = [str(toolkit_path)]
    module.__package__ = "mpl_toolkits"
    module.__file__ = str(toolkit_path / "__init__.py")
    spec = importlib.machinery.ModuleSpec("mpl_toolkits", loader=None, is_package=True)
    spec.submodule_search_locations = [str(toolkit_path)]
    module.__spec__ = spec
    sys.modules["mpl_toolkits"] = module
    _register_projection()


def _mode_table_from_wout(wout, *, nyq: bool, physical: bool = False) -> ModeTable:
    if nyq:
        m = np.asarray(wout.xm_nyq, dtype=int)
        n_raw = np.asarray(wout.xn_nyq, dtype=int)
    else:
        m = np.asarray(wout.xm, dtype=int)
        n_raw = np.asarray(wout.xn, dtype=int)
    if physical:
        n = n_raw
    else:
        n = n_raw // int(wout.nfp)
    return ModeTable(m=m, n=n)


def _basis_from_wout(wout, theta: np.ndarray, zeta: np.ndarray, *, nyq: bool, physical: bool = False) -> AngleGrid:
    modes = _mode_table_from_wout(wout, nyq=nyq, physical=physical)
    grid = AngleGrid(theta=theta, zeta=zeta, nfp=int(wout.nfp))
    basis = build_helical_basis(modes, grid)
    return basis


def bsup_from_wout(
    wout,
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    s_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return contravariant components (bsupu, bsupv) on a surface from wout Nyquist data."""
    basis = _basis_from_wout(wout, theta, zeta, nyq=True, physical=False)
    bsupumnc = np.asarray(wout.bsupumnc)
    bsupumns = np.asarray(getattr(wout, "bsupumns", np.zeros_like(bsupumnc)))
    bsupvmnc = np.asarray(wout.bsupvmnc)
    bsupvmns = np.asarray(getattr(wout, "bsupvmns", np.zeros_like(bsupvmnc)))
    if not bool(getattr(wout, "lasym", False)):
        bsupumns = np.zeros_like(bsupumnc)
        bsupvmns = np.zeros_like(bsupvmnc)
    bsupu = np.asarray(eval_fourier(bsupumnc[s_index], bsupumns[s_index], basis))
    bsupv = np.asarray(eval_fourier(bsupvmnc[s_index], bsupvmns[s_index], basis))
    return bsupu, bsupv


def bsub_from_wout(
    wout,
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    s_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return covariant components (bsubu, bsubv) on a surface from wout Nyquist data."""
    basis = _basis_from_wout(wout, theta, zeta, nyq=True, physical=False)
    bsubumnc = np.asarray(wout.bsubumnc)
    bsubumns = np.asarray(getattr(wout, "bsubumns", np.zeros_like(bsubumnc)))
    bsubvmnc = np.asarray(wout.bsubvmnc)
    bsubvmns = np.asarray(getattr(wout, "bsubvmns", np.zeros_like(bsubvmnc)))
    if not bool(getattr(wout, "lasym", False)):
        bsubumns = np.zeros_like(bsubumnc)
        bsubvmns = np.zeros_like(bsubvmnc)
    bsubu = np.asarray(eval_fourier(bsubumnc[s_index], bsubumns[s_index], basis))
    bsubv = np.asarray(eval_fourier(bsubvmnc[s_index], bsubvmns[s_index], basis))
    return bsubu, bsubv


def surface_rz_from_wout(
    wout,
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    s_index: int,
    nyq: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return R,Z on a surface from wout Fourier coefficients."""
    basis = _basis_from_wout(wout, theta, zeta, nyq=nyq, physical=False)
    rmnc = np.asarray(wout.rmnc)
    rmns = np.asarray(getattr(wout, "rmns", np.zeros_like(rmnc)))
    zmns = np.asarray(wout.zmns)
    zmnc = np.asarray(getattr(wout, "zmnc", np.zeros_like(zmns)))
    if not bool(getattr(wout, "lasym", False)):
        rmns = np.zeros_like(rmnc)
        zmnc = np.zeros_like(zmns)

    R = np.asarray(eval_fourier(rmnc[s_index], rmns[s_index], basis))
    Z = np.asarray(eval_fourier(zmnc[s_index], zmns[s_index], basis))
    return R, Z


def surface_rz_from_wout_physical(
    wout,
    *,
    theta: np.ndarray,
    phi: np.ndarray,
    s_index: int,
    nyq: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return R,Z on a surface using physical toroidal angle phi.

    This matches vmecPlot2's convention: phase = m*theta - xn*phi, where
    `xn` already includes the nfp factor.
    """
    basis = _basis_from_wout(wout, theta, phi, nyq=nyq, physical=True)
    rmnc = np.asarray(wout.rmnc)
    rmns = np.asarray(getattr(wout, "rmns", np.zeros_like(rmnc)))
    zmns = np.asarray(wout.zmns)
    zmnc = np.asarray(getattr(wout, "zmnc", np.zeros_like(zmns)))
    if not bool(getattr(wout, "lasym", False)):
        rmns = np.zeros_like(rmnc)
        zmnc = np.zeros_like(zmns)

    R = np.asarray(eval_fourier(rmnc[s_index], rmns[s_index], basis))
    Z = np.asarray(eval_fourier(zmnc[s_index], zmns[s_index], basis))
    return R, Z


def surface_rz_from_state(
    state,
    modes: ModeTable,
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    s_index: int,
    nfp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return R,Z on a surface from a VMECState on a field-period grid."""
    grid = AngleGrid(theta=theta, zeta=zeta, nfp=int(nfp))
    basis = build_helical_basis(modes, grid)
    Rcos = np.asarray(state.Rcos)
    Rsin = np.asarray(state.Rsin)
    Zcos = np.asarray(state.Zcos)
    Zsin = np.asarray(state.Zsin)
    lthreed = bool(np.any(np.asarray(modes.n)))
    lasym = bool(np.any(np.asarray(Rsin))) or bool(np.any(np.asarray(Zcos)))
    lconm1 = bool(lthreed or lasym)
    if lconm1 and int(np.max(np.asarray(modes.m))) > 0:
        Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
            Rcos=Rcos,
            Zsin=Zsin,
            Rsin=Rsin,
            Zcos=Zcos,
            modes=modes,
            lthreed=lthreed,
            lasym=lasym,
            lconm1=lconm1,
        )
    R = np.asarray(eval_fourier(Rcos[s_index], Rsin[s_index], basis, coeffs_internal=True))
    Z = np.asarray(eval_fourier(Zcos[s_index], Zsin[s_index], basis, coeffs_internal=True))
    return R, Z


def surface_rz_from_state_physical(
    state,
    modes: ModeTable,
    *,
    theta: np.ndarray,
    phi: np.ndarray,
    s_index: int,
    nfp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return R,Z on a surface using physical toroidal angle phi."""
    zeta = np.asarray(phi) * float(nfp)
    return surface_rz_from_state(state, modes, theta=theta, zeta=zeta, s_index=s_index, nfp=nfp)


def axis_rz_from_state_physical(
    state,
    modes: ModeTable,
    *,
    phi: np.ndarray,
    nfp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Axis curve from state using physical toroidal angle phi."""
    theta0 = np.zeros((1,), dtype=float)
    R, Z = surface_rz_from_state_physical(
        state,
        modes,
        theta=theta0,
        phi=phi,
        s_index=0,
        nfp=nfp,
    )
    return R[0], Z[0]


def bmag_from_state_physical(
    state,
    static,
    indata=None,
    *,
    theta: np.ndarray,
    phi: np.ndarray,
    s_index: int,
    signgs: int | None = None,
    phipf: np.ndarray | None = None,
    chipf: np.ndarray | None = None,
    lamscale: float | None = None,
    flux_is_internal: bool | None = None,
    sqrtg_floor: float | None = None,
    bmag_floor: float | None = None,
    eps: float = 1e-14,
) -> np.ndarray:
    """Compute B magnitude on a surface using physical toroidal angle phi.

    Notes
    -----
    - If ``phipf/chipf`` are provided, they override ``indata``-derived profiles.
    - If ``indata`` is None and no flux profiles are provided, this raises.
    - ``bmag_floor`` adds a small positive value inside the sqrt for smoother gradients.
    """
    nfp = int(static.cfg.nfp)
    theta_use = jnp.asarray(theta)
    phi_use = jnp.asarray(phi)
    zeta = phi_use * float(nfp)
    grid = AngleGrid(theta=theta_use, zeta=zeta, nfp=nfp)
    basis = build_helical_basis(static.modes, grid)
    # Mirror eval_geom's m=1 conversion when using custom grids.
    cfg = static.cfg
    lconm1 = bool(getattr(cfg, "lconm1", True))
    lthreed = bool(getattr(cfg, "lthreed", int(getattr(cfg, "ntor", 0)) > 0))
    lasym = bool(getattr(cfg, "lasym", False))
    if lconm1 and (lthreed or lasym) and int(getattr(cfg, "mpol", 0)) > 1:
        from .vmec_parity import vmec_m1_internal_to_physical_signed
        from .state import VMECState

        Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
            Rcos=state.Rcos,
            Zsin=state.Zsin,
            Rsin=state.Rsin,
            Zcos=state.Zcos,
            modes=static.modes,
            lthreed=lthreed,
            lasym=lasym,
            lconm1=lconm1,
        )
        state = VMECState(
            layout=state.layout,
            Rcos=Rcos,
            Rsin=Rsin,
            Zcos=Zcos,
            Zsin=Zsin,
            Lcos=state.Lcos,
            Lsin=state.Lsin,
        )
    geom = _eval_geom_jit(state, basis, static.s, grid.zeta)
    if signgs is None:
        if _is_tracer(geom.sqrtg):
            raise ValueError("signgs must be provided when tracing bmag_from_state_physical")
        signgs = signgs_from_sqrtg(geom.sqrtg)
    signgs = int(signgs)

    if phipf is None or chipf is None:
        if indata is None:
            raise ValueError("indata must be provided when phipf/chipf are not supplied")
        flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
        phipf_use = jnp.asarray(flux.phipf)
        chipf_use = jnp.asarray(flux.chipf)
        lamscale_use = jnp.asarray(flux.lamscale)
        flux_is_internal_use = True
    else:
        phipf_use = jnp.asarray(phipf)
        chipf_use = jnp.asarray(chipf)
        if lamscale is None:
            phips = (signgs * phipf_use) / (2.0 * np.pi)
            lamscale_use = lamscale_from_phips(phips, static.s)
        else:
            lamscale_use = jnp.asarray(lamscale)
        if flux_is_internal is None:
            flux_is_internal_use = False
        else:
            flux_is_internal_use = bool(flux_is_internal)

    if sqrtg_floor is None:
        bsupu, bsupv = bsup_from_geom(
            geom,
            phipf=phipf_use,
            chipf=chipf_use,
            nfp=nfp,
            signgs=signgs,
            lamscale=lamscale_use,
            flux_is_internal=flux_is_internal_use,
            eps=eps,
        )
    else:
        sqrtg = jnp.asarray(geom.sqrtg)
        sqrtg_use = jnp.sign(sqrtg) * jnp.maximum(jnp.abs(sqrtg), jnp.asarray(sqrtg_floor))
        bsupu, bsupv = bsup_from_sqrtg_lambda(
            sqrtg=sqrtg_use,
            lam_u=geom.L_theta,
            lam_v=geom.L_phi,
            phipf=phipf_use,
            chipf=chipf_use,
            signgs=signgs,
            lamscale=lamscale_use,
            flux_is_internal=flux_is_internal_use,
            eps=eps,
        )
    B2 = b2_from_bsup(geom, bsupu, bsupv)
    B2 = jnp.maximum(jnp.asarray(B2), 0.0)
    if bmag_floor is not None:
        B2 = B2 + jnp.asarray(bmag_floor)
    B = jnp.sqrt(B2)
    out = B[s_index]
    if _is_tracer(out):
        return out
    return np.asarray(out)


def bmag_from_state_vmec_realspace(
    state,
    static,
    indata=None,
    *,
    s_index: int,
    signgs: int | None = None,
    phipf: np.ndarray | None = None,
    chipf: np.ndarray | None = None,
    lamscale: float | None = None,
    flux_is_internal: bool | None = None,
    sqrtg_floor: float | None = None,
) -> np.ndarray:
    """Compute the magnetic field magnitude using VMEC real-space synthesis."""
    nfp = int(static.cfg.nfp)
    trig = vmec_trig_tables(
        ntheta=static.cfg.ntheta,
        nzeta=static.cfg.nzeta,
        nfp=nfp,
        mmax=static.cfg.mpol - 1,
        nmax=static.cfg.ntor,
        lasym=static.cfg.lasym,
    )
    geom = vmec_realspace_geom_from_state(state=state, modes=static.modes, trig=trig)
    if signgs is None:
        signgs = 1

    if phipf is None or chipf is None:
        if indata is None:
            raise ValueError("indata must be provided when phipf/chipf are not supplied")
        flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
        phipf_use = flux.phipf
        chipf_use = flux.chipf
        lamscale_use = flux.lamscale
        flux_is_internal_use = True
    else:
        phipf_use = np.asarray(phipf)
        chipf_use = np.asarray(chipf)
        if lamscale is None:
            phips = (signgs * np.asarray(phipf_use)) / (2.0 * np.pi)
            lamscale_use = float(np.asarray(lamscale_from_phips(phips, static.s)))
        else:
            lamscale_use = float(lamscale)
        if flux_is_internal is None:
            flux_is_internal_use = False
        else:
            flux_is_internal_use = bool(flux_is_internal)

    jac = vmec_half_mesh_jacobian_from_state(
        state=state,
        modes=static.modes,
        trig=trig,
        s=np.asarray(static.s),
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        lthreed=bool(getattr(static.cfg, "lthreed", True)),
    )
    sqrtg = np.asarray(jac.sqrtg)
    if sqrtg_floor is not None:
        sqrtg = np.sign(sqrtg) * np.maximum(np.abs(sqrtg), float(sqrtg_floor))
    lam_u = np.asarray(geom["Lu"]) if geom["Lu"] is not None else np.zeros_like(sqrtg)
    lam_v = np.asarray(geom["Lv"]) if geom["Lv"] is not None else np.zeros_like(sqrtg)

    bsupu, bsupv = bsup_from_sqrtg_lambda(
        sqrtg=sqrtg,
        lam_u=lam_u,
        lam_v=lam_v,
        phipf=phipf_use,
        chipf=chipf_use,
        signgs=signgs,
        lamscale=lamscale_use,
        flux_is_internal=flux_is_internal_use,
    )

    Ru = np.asarray(geom["Ru"])
    Zu = np.asarray(geom["Zu"])
    Rv = np.asarray(geom["Rv"])
    Zv = np.asarray(geom["Zv"])
    R = np.asarray(geom["R"])

    g_tt = Ru * Ru + Zu * Zu
    g_tp = Ru * Rv + Zu * Zv
    g_pp = Rv * Rv + Zv * Zv + R * R

    B2 = g_tt * bsupu**2 + 2.0 * g_tp * bsupu * bsupv + g_pp * bsupv**2
    B = np.sqrt(np.maximum(B2, 0.0))
    return B[s_index]


def bmag_from_wout(
    wout,
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    s_index: int,
) -> np.ndarray:
    """Return B magnitude on a surface from wout Nyquist Fourier coefficients."""
    basis = _basis_from_wout(wout, theta, zeta, nyq=True, physical=False)
    bmnc = np.asarray(wout.bmnc)
    bmns = np.asarray(getattr(wout, "bmns", np.zeros_like(bmnc)))
    if not bool(getattr(wout, "lasym", False)):
        bmns = np.zeros_like(bmnc)
    B = np.asarray(eval_fourier(bmnc[s_index], bmns[s_index], basis))
    return B


def bmag_from_wout_physical(
    wout,
    *,
    theta: np.ndarray,
    phi: np.ndarray,
    s_index: int,
) -> np.ndarray:
    """Return B magnitude on a surface using physical toroidal angle phi."""
    basis = _basis_from_wout(wout, theta, phi, nyq=True, physical=True)
    bmnc = np.asarray(wout.bmnc)
    bmns = np.asarray(getattr(wout, "bmns", np.zeros_like(bmnc)))
    if not bool(getattr(wout, "lasym", False)):
        bmns = np.zeros_like(bmnc)
    B = np.asarray(eval_fourier(bmnc[s_index], bmns[s_index], basis))
    return B


def vmecplot2_bmag_grid(
    wout,
    *,
    s_index: int,
    ntheta: int = 30,
    nzeta: int = 65,
    zeta_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (theta, zeta, B) on a grid matching vmecPlot2.py defaults.

    Parameters
    ----------
    zeta_max:
        Upper bound of the toroidal angle range.  Defaults to ``2π`` (full
        toroidal circle).  Pass ``2π/nfp`` to restrict to one field period.
    """
    if zeta_max is None:
        zeta_max = 2.0 * np.pi
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta))
    zeta = np.linspace(0.0, float(zeta_max), int(nzeta))
    zeta2d, theta2d = np.meshgrid(zeta, theta)
    xm_nyq = np.asarray(wout.xm_nyq, dtype=float)
    xn_nyq = np.asarray(wout.xn_nyq, dtype=float)
    bmnc = np.asarray(wout.bmnc, dtype=float)[int(s_index)]
    bmns = np.asarray(getattr(wout, "bmns", np.zeros_like(wout.bmnc)), dtype=float)[int(s_index)]
    if not bool(getattr(wout, "lasym", False)):
        bmns = np.zeros_like(bmns)
    angle = xm_nyq[:, None, None] * theta2d[None, :, :] - xn_nyq[:, None, None] * zeta2d[None, :, :]
    B = np.tensordot(bmnc, np.cos(angle), axes=(0, 0)) + np.tensordot(bmns, np.sin(angle), axes=(0, 0))
    return theta, zeta, np.asarray(B)


def vmecplot2_surface_grid(
    wout,
    *,
    s_index: int,
    ntheta: int = 200,
    nzeta: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (theta, zeta, R, Z) grids matching vmecPlot2.py surface defaults."""
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta))
    zeta = np.linspace(0.0, 2.0 * np.pi / float(wout.nfp), int(nzeta), endpoint=False)
    zeta2d, theta2d = np.meshgrid(zeta, theta)
    xm = np.asarray(wout.xm, dtype=float)
    xn = np.asarray(wout.xn, dtype=float)
    rmnc = np.asarray(wout.rmnc, dtype=float)[int(s_index)]
    rmns = np.asarray(getattr(wout, "rmns", np.zeros_like(wout.rmnc)), dtype=float)[int(s_index)]
    zmns = np.asarray(wout.zmns, dtype=float)[int(s_index)]
    zmnc = np.asarray(getattr(wout, "zmnc", np.zeros_like(wout.zmns)), dtype=float)[int(s_index)]
    angle = xm[:, None, None] * theta2d[None, :, :] - xn[:, None, None] * zeta2d[None, :, :]
    R = np.tensordot(rmnc, np.cos(angle), axes=(0, 0)) + np.tensordot(rmns, np.sin(angle), axes=(0, 0))
    Z = np.tensordot(zmns, np.sin(angle), axes=(0, 0)) + np.tensordot(zmnc, np.cos(angle), axes=(0, 0))
    return theta, zeta, np.asarray(R), np.asarray(Z)


def vmecplot2_lcfs_3d_grid(
    wout,
    *,
    s_index: int,
    ntheta: int = 80,
    nzeta: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (theta, phi, R, Z, B) grids matching vmecPlot2.py 3D defaults."""
    if nzeta is None:
        nzeta = int(150 * int(wout.nfp))
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta))
    phi = np.linspace(0.0, 2.0 * np.pi, int(nzeta))
    phi2d, theta2d = np.meshgrid(phi, theta)
    xm = np.asarray(wout.xm, dtype=float)
    xn = np.asarray(wout.xn, dtype=float)
    xm_nyq = np.asarray(wout.xm_nyq, dtype=float)
    xn_nyq = np.asarray(wout.xn_nyq, dtype=float)
    rmnc = np.asarray(wout.rmnc, dtype=float)[int(s_index)]
    rmns = np.asarray(getattr(wout, "rmns", np.zeros_like(wout.rmnc)), dtype=float)[int(s_index)]
    zmns = np.asarray(wout.zmns, dtype=float)[int(s_index)]
    zmnc = np.asarray(getattr(wout, "zmnc", np.zeros_like(wout.zmns)), dtype=float)[int(s_index)]
    bmnc = np.asarray(wout.bmnc, dtype=float)[int(s_index)]
    bmns = np.asarray(getattr(wout, "bmns", np.zeros_like(wout.bmnc)), dtype=float)[int(s_index)]
    if not bool(getattr(wout, "lasym", False)):
        bmns = np.zeros_like(bmns)

    angle = xm[:, None, None] * theta2d[None, :, :] - xn[:, None, None] * phi2d[None, :, :]
    R = np.tensordot(rmnc, np.cos(angle), axes=(0, 0)) + np.tensordot(rmns, np.sin(angle), axes=(0, 0))
    Z = np.tensordot(zmns, np.sin(angle), axes=(0, 0)) + np.tensordot(zmnc, np.cos(angle), axes=(0, 0))

    angle_b = xm_nyq[:, None, None] * theta2d[None, :, :] - xn_nyq[:, None, None] * phi2d[None, :, :]
    B = np.tensordot(bmnc, np.cos(angle_b), axes=(0, 0)) + np.tensordot(bmns, np.sin(angle_b), axes=(0, 0))
    return theta, phi, np.asarray(R), np.asarray(Z), np.asarray(B)


def axis_rz_from_wout(wout, *, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis curve from wout Fourier coefficients."""
    zeta = np.asarray(zeta)
    if not hasattr(wout, "raxis_cc") or not hasattr(wout, "zaxis_cs"):
        # Fallback: use the m=0,n=0 mode from rmnc for a constant axis estimate.
        r0 = float(np.asarray(wout.rmnc)[0, 0]) if np.asarray(wout.rmnc).size else 0.0
        return np.full_like(zeta, r0, dtype=float), np.zeros_like(zeta, dtype=float)

    n = np.arange(len(wout.raxis_cc), dtype=float)
    angle = (-n[:, None] * float(wout.nfp)) * zeta[None, :]
    raxis_cc = np.asarray(wout.raxis_cc, dtype=float)[:, None]
    raxis_cs = np.asarray(getattr(wout, "raxis_cs", np.zeros_like(wout.raxis_cc)), dtype=float)[:, None]
    zaxis_cs = np.asarray(wout.zaxis_cs, dtype=float)[:, None]
    zaxis_cc = np.asarray(getattr(wout, "zaxis_cc", np.zeros_like(wout.zaxis_cs)), dtype=float)[:, None]

    R = np.sum(raxis_cc * np.cos(angle) + raxis_cs * np.sin(angle), axis=0)
    Z = np.sum(zaxis_cs * np.sin(angle) + zaxis_cc * np.cos(angle), axis=0)
    return R, Z


def axis_rz_from_wout_physical(wout, *, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis curve using physical toroidal angle phi (vmecPlot2 convention)."""
    if not hasattr(wout, "raxis_cc") or not hasattr(wout, "zaxis_cs"):
        r0 = float(np.asarray(wout.rmnc)[0, 0]) if np.asarray(wout.rmnc).size else 0.0
        return np.full_like(phi, r0, dtype=float), np.zeros_like(phi, dtype=float)

    phi = np.asarray(phi)
    n = np.arange(len(wout.raxis_cc), dtype=float)
    angle = (-n[:, None] * float(wout.nfp)) * phi[None, :]
    raxis_cc = np.asarray(wout.raxis_cc, dtype=float)[:, None]
    raxis_cs = np.asarray(getattr(wout, "raxis_cs", np.zeros_like(wout.raxis_cc)), dtype=float)[:, None]
    zaxis_cs = np.asarray(wout.zaxis_cs, dtype=float)[:, None]
    zaxis_cc = np.asarray(getattr(wout, "zaxis_cc", np.zeros_like(wout.zaxis_cs)), dtype=float)[:, None]

    R = np.sum(raxis_cc * np.cos(angle) + raxis_cs * np.sin(angle), axis=0)
    Z = np.sum(zaxis_cs * np.sin(angle) + zaxis_cc * np.cos(angle), axis=0)
    return R, Z


def profiles_from_wout(wout) -> dict[str, np.ndarray]:
    """Return common radial profiles from wout."""
    ns = int(wout.ns)
    s = np.linspace(0.0, 1.0, ns)
    s_half = (np.arange(1, ns, dtype=float) - 0.5) / float(ns - 1)
    return {
        "s": s,
        "s_half": s_half,
        "iotaf": np.asarray(wout.iotaf),
        "iotas": np.asarray(wout.iotas),
        "presf": np.asarray(wout.presf),
        "pres": np.asarray(wout.pres),
        "buco": np.asarray(getattr(wout, "buco", np.zeros_like(wout.presf))),
        "bvco": np.asarray(getattr(wout, "bvco", np.zeros_like(wout.presf))),
        "jcuru": np.asarray(getattr(wout, "jcuru", np.zeros_like(wout.presf))),
        "jcurv": np.asarray(getattr(wout, "jcurv", np.zeros_like(wout.presf))),
    }


def surface_data_from_wout(
    wout,
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    s_index: int,
    with_bmag: bool = False,
) -> SurfaceData:
    """Convenience wrapper returning R/Z (and optionally B magnitude) on a surface."""
    R, Z = surface_rz_from_wout(wout, theta=theta, zeta=zeta, s_index=s_index, nyq=False)
    if with_bmag:
        B = bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=s_index)
    else:
        B = None
    return SurfaceData(R=R, Z=Z, B=B)


def closed_theta_grid(ntheta: int) -> np.ndarray:
    """Theta grid including the 2π endpoint (good for closed cross-sections)."""
    return np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=True)


def zeta_grid(nzeta: int, *, endpoint: bool = False) -> np.ndarray:
    """Uniform zeta grid over one field period."""
    return np.linspace(0.0, 2.0 * np.pi, int(nzeta), endpoint=bool(endpoint))


def zeta_grid_field_period(nzeta: int, *, nfp: int) -> np.ndarray:
    """Uniform zeta grid over one field period (0..2π/nfp)."""
    nfp = max(1, int(nfp))
    return np.linspace(0.0, 2.0 * np.pi / float(nfp), int(nzeta), endpoint=False)


def vmecplot2_cross_section_indices(nzeta: int) -> np.ndarray:
    """Indices used by vmecPlot2 for cross sections (0,2,4,6)."""
    if nzeta < 7:
        raise ValueError("vmecPlot2 cross sections expect nzeta>=8")
    return np.asarray([0, 2, 4, 6], dtype=int)


def select_zeta_slices(zeta: np.ndarray, *, n: int) -> np.ndarray:
    """Pick evenly spaced zeta indices from a zeta grid."""
    zeta = np.asarray(zeta)
    if n <= 0:
        raise ValueError("n must be positive")
    idx = np.linspace(0, len(zeta) - 1, num=n).round().astype(int)
    return zeta[idx]


def surface_stack(
    wout,
    *,
    theta: np.ndarray,
    zeta_list: Iterable[float],
    s_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack R,Z slices for multiple zeta values."""
    zeta = np.asarray(list(zeta_list), dtype=float)
    R, Z = surface_rz_from_wout(wout, theta=theta, zeta=zeta, s_index=s_index, nyq=False)
    return R, Z


def _import_matplotlib():
    try:
        prepare_matplotlib_3d()
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for this plotting helper") from exc


def _case_from_input_path(input_path: str | Path) -> str:
    name = Path(input_path).name
    if name.startswith("input."):
        return name.split("input.", 1)[1]
    return Path(name).stem


def _default_example_outdir(subdir: str, case: str, outdir: str | Path | None) -> Path:
    if outdir is not None:
        return Path(outdir)
    root = Path(__file__).resolve().parents[1]
    return root / "examples" / "outputs" / subdir / case


def _extent_from_grids(theta: np.ndarray, zeta: np.ndarray) -> tuple[float, float, float, float]:
    z0 = float(np.min(zeta))
    z1 = float(np.max(zeta))
    t0 = float(np.min(theta))
    t1 = float(np.max(theta))
    if z0 == z1:
        z0 -= 0.5
        z1 += 0.5
    if t0 == t1:
        t0 -= 0.5
        t1 += 0.5
    return (z0, z1, t0, t1)


def _line_contour_levels(values: np.ndarray, *, count: int = 25) -> np.ndarray:
    """Return robust line-contour levels for finite, constant, or bad-valued fields."""
    arr = np.asarray(values, dtype=float)
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        pad = max(abs(vmin), 1.0) * 1.0e-12
        vmin -= pad
        vmax += pad
    return np.linspace(vmin, vmax, int(count))


def write_axisym_overview(case: str, *, outdir: str | Path | None = None) -> Path:
    """Write a quick axisymmetric overview plot from bundled reference wout."""
    plt = _import_matplotlib()
    input_path, wout_path = example_paths(case)
    if wout_path is None:
        raise FileNotFoundError(f"Reference wout not found for case={case!r}")
    wout = read_wout(wout_path)
    s_index = int(wout.ns) - 1

    theta = closed_theta_grid(256)
    zeta = np.asarray([0.0], dtype=float)

    R, Z = surface_rz_from_wout(wout, theta=theta, zeta=zeta, s_index=s_index, nyq=False)
    B = bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=s_index)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(R[:, 0], Z[:, 0], lw=1.5)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("R")
    axes[0].set_ylabel("Z")
    axes[0].set_title(f"{case} cross-section")

    axes[1].plot(theta, B[:, 0], lw=1.2)
    axes[1].set_xlabel("theta")
    axes[1].set_ylabel("|B|")
    axes[1].set_title("|B| on LCFS")

    outdir = _default_example_outdir("overview", case, outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{case}_overview.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def write_bmag_parity_figures(
    *,
    input_path: str | Path,
    wout_path: str | Path,
    outdir: str | Path | None = None,
    s_index: int | None = None,
) -> Path:
    """Write magnetic-field-magnitude parity figures for wout vs vmec_jax."""
    plt = _import_matplotlib()
    cfg, indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    static = build_static(cfg)
    state = state_from_wout(wout)
    s_idx = int(wout.ns) - 1 if s_index is None else int(s_index)
    theta = np.asarray(static.grid.theta)
    zeta = np.asarray(static.grid.zeta)

    B_ref = bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=s_idx)
    phi = zeta / float(max(1, int(static.cfg.nfp)))
    B_jax = bmag_from_state_physical(
        state,
        static,
        indata=indata,
        theta=theta,
        phi=phi,
        s_index=s_idx,
        signgs=int(getattr(wout, "signgs", 1)),
        phipf=np.asarray(wout.phipf),
        chipf=np.asarray(wout.chipf),
        flux_is_internal=False,
    )
    diff = B_jax - B_ref

    extent = _extent_from_grids(theta, zeta)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, data, title in zip(axes, [B_ref, B_jax, diff], ["wout", "vmec_jax", "diff"]):
        im = ax.imshow(data, origin="lower", aspect="auto", extent=extent)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    case = _case_from_input_path(input_path)
    fig.suptitle(f"|B| parity ({case})")

    outdir = _default_example_outdir("bmag_parity", case, outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "bmag_parity.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def write_bsup_parity_figures(
    *,
    input_path: str | Path,
    wout_path: str | Path,
    outdir: str | Path | None = None,
    s_index: int | None = None,
) -> Path:
    """Write (bsupu, bsupv) parity figures comparing wout vs vmec_jax geometry."""
    plt = _import_matplotlib()
    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    static = build_static(cfg)
    state = state_from_wout(wout)
    s_idx = int(wout.ns) - 1 if s_index is None else int(s_index)
    theta = np.asarray(static.grid.theta)
    zeta = np.asarray(static.grid.zeta)

    bsupu_ref, bsupv_ref = bsup_from_wout(wout, theta=theta, zeta=zeta, s_index=s_idx)

    geom = eval_geom(state, static)
    lamscale = float(np.asarray(lamscale_from_phips(np.asarray(wout.phips), np.asarray(static.s))))
    bsupu_full, bsupv_full = bsup_from_geom(
        geom,
        phipf=np.asarray(wout.phipf),
        chipf=np.asarray(wout.chipf),
        nfp=int(static.cfg.nfp),
        signgs=int(getattr(wout, "signgs", 1)),
        lamscale=lamscale,
        flux_is_internal=False,
    )
    bsupu_jax = np.asarray(bsupu_full)[s_idx]
    bsupv_jax = np.asarray(bsupv_full)[s_idx]

    case = _case_from_input_path(input_path)
    outdir = _default_example_outdir("bsup_parity", case, outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    extent = _extent_from_grids(theta, zeta)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, (ref, jax, label) in enumerate(
        [(bsupu_ref, bsupu_jax, "bsupu"), (bsupv_ref, bsupv_jax, "bsupv")]
    ):
        diff = jax - ref
        for col, (data, title) in enumerate([(ref, "wout"), (jax, "vmec_jax"), (diff, "diff")]):
            ax = axes[row, col]
            im = ax.imshow(data, origin="lower", aspect="auto", extent=extent)
            ax.set_title(f"{label} {title}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"bsup parity ({case})")
    outpath = outdir / "bsup_parity.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def write_bsub_parity_figures(
    *,
    input_path: str | Path,
    wout_path: str | Path,
    outdir: str | Path | None = None,
    s_index: int | None = None,
) -> Path:
    """Write (bsubu, bsubv) parity figures comparing wout vs vmec_jax geometry."""
    plt = _import_matplotlib()
    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    static = build_static(cfg)
    state = state_from_wout(wout)
    s_idx = int(wout.ns) - 1 if s_index is None else int(s_index)
    theta = np.asarray(static.grid.theta)
    zeta = np.asarray(static.grid.zeta)

    bsubu_ref, bsubv_ref = bsub_from_wout(wout, theta=theta, zeta=zeta, s_index=s_idx)

    geom = eval_geom(state, static)
    lamscale = float(np.asarray(lamscale_from_phips(np.asarray(wout.phips), np.asarray(static.s))))
    bsupu_full, bsupv_full = bsup_from_geom(
        geom,
        phipf=np.asarray(wout.phipf),
        chipf=np.asarray(wout.chipf),
        nfp=int(static.cfg.nfp),
        signgs=int(getattr(wout, "signgs", 1)),
        lamscale=lamscale,
        flux_is_internal=False,
    )
    bsubu_full, bsubv_full = bsub_from_bsup(geom, bsupu_full, bsupv_full)
    bsubu_jax = np.asarray(bsubu_full)[s_idx]
    bsubv_jax = np.asarray(bsubv_full)[s_idx]

    case = _case_from_input_path(input_path)
    outdir = _default_example_outdir("bsub_parity", case, outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    extent = _extent_from_grids(theta, zeta)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, (ref, jax, label) in enumerate(
        [(bsubu_ref, bsubu_jax, "bsubu"), (bsubv_ref, bsubv_jax, "bsubv")]
    ):
        diff = jax - ref
        for col, (data, title) in enumerate([(ref, "wout"), (jax, "vmec_jax"), (diff, "diff")]):
            ax = axes[row, col]
            im = ax.imshow(data, origin="lower", aspect="auto", extent=extent)
            ax.set_title(f"{label} {title}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"bsub parity ({case})")
    outpath = outdir / "bsub_parity.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# QH optimisation result plots
# ─────────────────────────────────────────────────────────────────────────────

def _lcfs_xyz(R: np.ndarray, Z: np.ndarray, phi: np.ndarray):
    """Convert cylindrical (R, Z, phi) grids to Cartesian (X, Y, Z)."""
    X = R * np.cos(phi[None, :])
    Y = R * np.sin(phi[None, :])
    return X, Y, Z


def _plot_3d_boundary_comparison(wout_init, wout_final, outdir: Path) -> Path:
    """3-D LCFS plots coloured by |B|, initial (left) vs optimised (right)."""
    prepare_matplotlib_3d()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    ns_init = int(np.asarray(wout_init.ns))
    ns_final = int(np.asarray(wout_final.ns))
    theta_i, phi_i, R_i, Z_i, B_i = vmecplot2_lcfs_3d_grid(
        wout_init, s_index=ns_init - 1, ntheta=60, nzeta=None
    )
    theta_f, phi_f, R_f, Z_f, B_f = vmecplot2_lcfs_3d_grid(
        wout_final, s_index=ns_final - 1, ntheta=60, nzeta=None
    )
    X_i, Y_i, _ = _lcfs_xyz(R_i, Z_i, phi_i)
    X_f, Y_f, _ = _lcfs_xyz(R_f, Z_f, phi_f)

    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(12, 5))
    for col, (X, Y, Zp, B, title) in enumerate([
        (X_i, Y_i, Z_i, B_i, "Initial boundary"),
        (X_f, Y_f, Z_f, B_f, "Optimised boundary"),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        norm = Normalize(vmin=float(np.nanmin(B)), vmax=float(np.nanmax(B)))
        fcolors = cmap(norm(B))
        ax.plot_surface(X, Y, Zp, facecolors=fcolors, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title, fontsize=11)
        fix_matplotlib_3d(ax)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="|B| (T)", shrink=0.62, pad=0.04)
    nfp = int(np.asarray(wout_init.nfp)) if hasattr(wout_init, "nfp") else 4
    fig.suptitle(f"LCFS coloured by |B| — nfp={nfp}", fontsize=13, y=1.01)

    out = outdir / "boundary_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _load_wout_if_path(wout_or_path):
    if isinstance(wout_or_path, (str, Path)):
        from .wout import read_wout as _read_wout

        return _read_wout(str(wout_or_path))
    return wout_or_path


def plot_3d_boundary_comparison(
    wout_initial,
    wout_final,
    *,
    outdir=None,
) -> Path:
    """Plot initial/final LCFS 3-D surfaces colored by ``|B|``.

    ``wout_initial`` and ``wout_final`` can be loaded ``WoutData`` objects or
    paths to ``wout_*.nc`` files.  The returned path points to
    ``boundary_comparison.png`` in *outdir*.
    """

    wout_init = _load_wout_if_path(wout_initial)
    wout_final_obj = _load_wout_if_path(wout_final)
    if outdir is None:
        outdir = Path(".")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return _plot_3d_boundary_comparison(wout_init, wout_final_obj, outdir)


def _pi_label(v: float) -> str:
    """Format a radian value as a human-readable fraction of π (e.g. 'π/4')."""
    from fractions import Fraction
    if abs(v) < 1e-14:
        return "0"
    frac = Fraction(v / float(np.pi)).limit_denominator(128)
    n, d = frac.numerator, frac.denominator
    if d == 1:
        return "π" if n == 1 else f"{n}π"
    return f"π/{d}" if n == 1 else f"{n}π/{d}"


def _plot_bmag_contours(wout_init, wout_final, outdir: Path) -> Path:
    """Unrolled |B|(θ, ζ) contour lines on LCFS — initial (top) vs optimised (bottom).

    Uses ``ax.contour`` (line contours only, no fill) so the helically-aligned
    contours of a quasi-helically/quasi-axially symmetric configuration are
    visually obvious.  The toroidal axis covers exactly **one field period**:
    ζ ∈ [0, 2π/nfp].  Poloidal axis: θ ∈ [0, 2π].
    """
    prepare_matplotlib_3d()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns_init = int(np.asarray(wout_init.ns))
    ns_final = int(np.asarray(wout_final.ns))
    nfp = int(np.asarray(wout_init.nfp))
    zeta_max = 2.0 * np.pi / nfp   # one field period

    theta_i, zeta_i, B_i = vmecplot2_bmag_grid(
        wout_init, s_index=ns_init - 1, ntheta=128, nzeta=256, zeta_max=zeta_max
    )
    theta_f, zeta_f, B_f = vmecplot2_bmag_grid(
        wout_final, s_index=ns_final - 1, ntheta=128, nzeta=256, zeta_max=zeta_max
    )

    # Dynamic ticks for ζ ∈ [0, 2π/nfp].
    xtick_vals = np.linspace(0.0, zeta_max, 5)
    xtick_lbls = [_pi_label(v) for v in xtick_vals]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, B, zeta, theta, title in [
        (axes[0], B_i, zeta_i, theta_i, "Initial"),
        (axes[1], B_f, zeta_f, theta_f, "Optimised"),
    ]:
        # B has shape (ntheta, nzeta); meshgrid for contour
        ZETA, THETA = np.meshgrid(zeta, theta)
        levels = _line_contour_levels(B, count=25)
        cs = ax.contour(
            ZETA, THETA, B,
            levels=levels,
            cmap="viridis",
            linewidths=1.2,
        )
        ax.set_facecolor("white")
        ax.set_ylabel("Poloidal angle θ (rad)")
        ax.set_title(f"|B| on LCFS — {title}", fontsize=11)
        ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
        ax.set_ylim(0, 2 * np.pi)
        fig.colorbar(cs, ax=ax, label="|B| (T)")

    axes[-1].set_xlabel(f"Toroidal angle ζ (rad, one field period = 2π/{nfp})")
    axes[-1].set_xticks(xtick_vals)
    axes[-1].set_xticklabels(xtick_lbls)
    axes[-1].set_xlim(0, zeta_max)
    fig.suptitle("|B| on LCFS — contour lines", fontsize=13)
    fig.tight_layout()

    out = outdir / "bmag_surface.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_bmag_contours(
    wout_initial,
    wout_final,
    *,
    outdir=None,
) -> Path:
    """Plot line contours of ``|B|`` on the initial/final LCFS.

    ``wout_initial`` and ``wout_final`` can be loaded ``WoutData`` objects or
    paths to ``wout_*.nc`` files.  The returned path points to
    ``bmag_surface.png`` in *outdir*.
    """

    wout_init = _load_wout_if_path(wout_initial)
    wout_final_obj = _load_wout_if_path(wout_final)
    if outdir is None:
        outdir = Path(".")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return _plot_bmag_contours(wout_init, wout_final_obj, outdir)


def boozer_bmag_grid_from_state(
    state,
    *,
    static,
    indata,
    signgs: int,
    surfaces=(1.0,),
    surface_index: int = -1,
    mboz: int = 18,
    nboz: int = 18,
    ntheta: int = 128,
    nphi: int = 256,
    phimin: float = 0.0,
    jit_booz: bool = False,
):
    """Evaluate Boozer-coordinate ``|B|(theta_B, phi_B)`` from a VMEC state.

    This is the visual diagnostic needed for QI/omnigenity review.  VMEC-angle
    ``plot_bmag_contours`` is useful for VMEC parity, but QI contour closure
    must be judged in Boozer coordinates.
    """

    try:
        from booz_xform_jax import booz_xform_from_inputs, prepare_booz_xform_constants_from_inputs
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "boozer_bmag_grid_from_state requires booz_xform_jax. "
            "Install it with `pip install booz_xform_jax` or from github.com/uwplasma/booz_xform_jax."
        ) from exc

    from .booz_input import booz_xform_inputs_from_state
    from .quasi_isodynamic import _nearest_half_mesh_indices

    surface_values = tuple(float(s) for s in surfaces)
    if not surface_values:
        raise ValueError("surfaces must contain at least one value")
    if int(ntheta) < 4 or int(nphi) < 4:
        raise ValueError("ntheta and nphi must both be at least 4")

    inputs = booz_xform_inputs_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    constants, grids = prepare_booz_xform_constants_from_inputs(
        inputs=inputs,
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(static.cfg.lasym),
    )
    surface_indices = _nearest_half_mesh_indices(surface_values, n_half=int(inputs.rmnc.shape[0]))
    booz = booz_xform_from_inputs(
        inputs=inputs,
        constants=constants,
        grids=grids,
        surface_indices=jnp.asarray(surface_indices, dtype=jnp.int32),
        jit=bool(jit_booz),
    )

    bmnc = np.asarray(booz["bmnc_b"], dtype=float)
    bmns_value = booz.get("bmns_b")
    bmns = np.zeros_like(bmnc) if bmns_value is None else np.asarray(bmns_value, dtype=float)
    xm = np.asarray(booz["ixm_b"], dtype=float)
    xn = np.asarray(booz["ixn_b"], dtype=float)
    nfp_arr = np.asarray(booz.get("nfp_b", getattr(static.cfg, "nfp", 1)))
    nfp = int(nfp_arr.ravel()[0]) if nfp_arr.size else int(static.cfg.nfp)

    nsurf = int(bmnc.shape[0])
    selected = int(surface_index)
    if selected < 0:
        selected += nsurf
    if selected < 0 or selected >= nsurf:
        raise IndexError(f"surface_index {surface_index} is outside Boozer surface range 0..{nsurf - 1}")

    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=True)
    phi = np.linspace(float(phimin), float(phimin) + 2.0 * np.pi / float(nfp), int(nphi), endpoint=True)
    angle = theta[:, None, None] * xm[None, None, :] - phi[None, :, None] * xn[None, None, :]
    bmag = np.sum(
        bmnc[selected][None, None, :] * np.cos(angle)
        + bmns[selected][None, None, :] * np.sin(angle),
        axis=-1,
    )
    return theta, phi, bmag, booz


def plot_boozer_bmag_contours_from_state(
    state,
    *,
    static,
    indata,
    signgs: int,
    outdir: str | Path,
    filename: str = "boozer_bmag_surface.png",
    surfaces=(1.0,),
    surface_index: int = -1,
    mboz: int = 18,
    nboz: int = 18,
    ntheta: int = 128,
    nphi: int = 256,
    phimin: float = 0.0,
    title: str = "Boozer |B| contours",
):
    """Write a Boozer-coordinate line-contour plot of ``|B|`` for QI review."""

    prepare_matplotlib_3d()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta, phi, bmag, _booz = boozer_bmag_grid_from_state(
        state,
        static=static,
        indata=indata,
        signgs=signgs,
        surfaces=surfaces,
        surface_index=surface_index,
        mboz=mboz,
        nboz=nboz,
        ntheta=ntheta,
        nphi=nphi,
        phimin=phimin,
    )
    phi2d, theta2d = np.meshgrid(phi, theta)
    levels = _line_contour_levels(bmag, count=25)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    cs = ax.contour(phi2d, theta2d, bmag, levels=levels, cmap="viridis", linewidths=1.0)
    fig.colorbar(cs, ax=ax, label="|B| (T)")
    ax.set_title(title)
    ax.set_xlabel("Boozer toroidal angle φ_B (rad)")
    ax.set_ylabel("Boozer poloidal angle θ_B (rad)")
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.set_ylim(0, 2.0 * np.pi)
    fig.tight_layout()
    out = outdir / filename
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def _objective_iota_series(hist: list[dict]) -> list[float] | None:
    """Return optional iota trajectory, preserving missing entries as NaN."""
    if not hist or not any("iota" in h for h in hist):
        return None
    return [h.get("iota", np.nan) for h in hist]


def _best_so_far_stage_segments(
    values: Iterable[float],
    stage_boundaries: Iterable[int],
    *,
    floor: float = 1e-16,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split objective values by stage and return best-so-far traces per stage."""
    vals = [float(v) for v in values]
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    stops = [int(b) + 1 for b in stage_boundaries if int(b) + 1 < len(vals)]
    for stop in stops + [len(vals)]:
        if stop <= start:
            continue
        x_segment = np.arange(start, stop, dtype=int)
        y_segment = np.minimum.accumulate([max(v, floor) for v in vals[start:stop]])
        segments.append((x_segment, y_segment))
        start = stop
    return segments


def _plot_objective_history(history_path: Path, outdir: Path) -> Path:
    """Objective value, aspect ratio, and (optionally) iota vs Jacobian evaluation."""
    import json
    prepare_matplotlib_3d()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(history_path) as f:
        data = json.load(f)

    hist = data["history"]
    # Use qs_objective (QS residuals only) if available, else fall back to total objective
    qs_vals = [h.get("qs_objective", h["objective"]) for h in hist]
    aspects = [h["aspect"] for h in hist]
    # Iota trajectory: present when iota_fn was passed to optimizer
    iotas = _objective_iota_series(hist)
    target_iota = data.get("target_iota", None)
    iota_abs_min = data.get("iota_abs_min", None)
    # Also show iota panel when target_iota is specified even if trajectory is missing
    show_iota = iotas is not None
    iters = list(range(len(hist)))
    total_time = data.get("total_wall_time_s", 0.0)
    nfev = data.get("nfev", len(hist))

    n_panels = 3 if show_iota else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 3 * n_panels), sharex=True)
    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if n_panels == 3 else None

    # --- panel 1: QS residuals ---
    # Stage-preseeded optimizations can switch objective definitions (for
    # example QP preseed -> QI preseed -> full constrained QI). Plot each stage
    # separately and show the best-so-far value within that stage so rejected
    # trial points or objective switches are not drawn as false increases.
    for x_segment, y_segment in _best_so_far_stage_segments(qs_vals, data.get("stage_boundaries", [])):
        ax1.semilogy(x_segment, y_segment, "o-", color="steelblue", linewidth=2, markersize=6)
    qs_pos = [max(v, 1e-16) for v in qs_vals]  # avoid log(0)
    ax1.set_ylabel("QS residuals ∑r²", fontsize=11)
    opt_label = data.get("label", "Optimisation")
    ax1.set_title(
        f"{opt_label}  ({nfev} evals, {total_time:.0f} s)",
        fontsize=11,
    )
    ax1.axhline(qs_pos[-1], color="steelblue", linestyle="--", alpha=0.4,
                label=f"Final: {qs_vals[-1]:.2e}")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- panel 2: aspect ratio ---
    ax2.plot(iters, aspects, "s-", color="darkorange", linewidth=2, markersize=6)
    target_aspect = data.get("target_aspect", None)
    if target_aspect is not None:
        ax2.axhline(target_aspect, color="k", linestyle=":", alpha=0.5,
                    label=f"Target A={target_aspect:.4g}")
    ax2.set_ylabel("Aspect ratio", fontsize=11)
    if ax3 is None:
        ax2.set_xlabel("Jacobian evaluation index", fontsize=11)
    if target_aspect is not None:
        ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- panel 3: mean iota ---
    if ax3 is not None and iotas is not None:
        ax3.plot(iters, iotas, "^-", color="forestgreen", linewidth=2, markersize=6)
        if target_iota is not None:
            ax3.axhline(target_iota, color="k", linestyle=":", alpha=0.5,
                        label=f"Target ι={target_iota:.4g}")
        if iota_abs_min is not None:
            ax3.axhline(iota_abs_min, color="k", linestyle=":", alpha=0.45,
                        label=f"Min |ι|={iota_abs_min:.4g}")
            ax3.axhline(-float(iota_abs_min), color="k", linestyle=":", alpha=0.45)
        ax3.set_ylabel("Mean iota ι", fontsize=11)
        ax3.set_xlabel("Jacobian evaluation index", fontsize=11)
        if target_iota is not None or iota_abs_min is not None:
            ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out = outdir / "objective_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_objective_history(
    history_path,
    *,
    outdir=None,
) -> Path:
    """Plot objective, aspect-ratio, and optional iota history from JSON."""

    history_path = Path(history_path)
    if outdir is None:
        outdir = history_path.parent
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return _plot_objective_history(history_path, outdir)


def plot_qh_optimization(
    wout_initial_path,
    wout_final_path,
    history_path,
    *,
    outdir=None,
    show: bool = False,
) -> dict:
    """Generate optimization result plots and return their paths.

    This compatibility wrapper calls :func:`plot_3d_boundary_comparison`,
    :func:`plot_bmag_contours`, and :func:`plot_objective_history`.  New
    examples call those functions directly so users can choose which plots to
    create for QA/QH/QP/QI or custom objectives.

    Produces three figures:

    * ``boundary_comparison.png``  — 3-D LCFS coloured by |B| (before/after)
    * ``bmag_surface.png``         — |B| contour lines on LCFS unrolled to (θ, φ)
    * ``objective_history.png``    — Objective and aspect ratio vs iteration

    Parameters
    ----------
    wout_initial_path, wout_final_path:
        Paths to the initial and final ``wout_*.nc`` files.
    history_path:
        Path to the ``history.json`` file produced by
        :meth:`~vmec_jax.FixedBoundaryExactOptimizer.save_history`.
    outdir:
        Directory for saved figures.  Defaults to the directory of *history_path*.
    show:
        If ``True``, call ``plt.show()`` after saving.

    Returns
    -------
    dict
        Mapping ``{"boundary_comparison", "bmag_surface", "objective_history"}``
        to their saved :class:`~pathlib.Path` objects.
    """
    wout_initial_path = Path(wout_initial_path)
    wout_final_path = Path(wout_final_path)
    history_path = Path(history_path)

    if outdir is None:
        outdir = history_path.parent
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p1 = plot_3d_boundary_comparison(wout_initial_path, wout_final_path, outdir=outdir)
    p2 = plot_bmag_contours(wout_initial_path, wout_final_path, outdir=outdir)
    p3 = plot_objective_history(history_path, outdir=outdir)

    for p in (p1, p2, p3):
        print(f"  Saved {p}")

    if show:
        prepare_matplotlib_3d()
        import matplotlib.pyplot as plt
        plt.show()

    return {
        "boundary_comparison": p1,
        "bmag_surface": p2,
        "objective_history": p3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# plot_wout — standalone wout diagnostic viewer (replicates vmecPlot2.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_wout(
    wout_path: str | Path,
    outdir: str | Path | None = None,
    *,
    name: str | None = None,
    s_plot_ignore: float = 0.2,
    show: bool = False,
) -> dict:
    """Generate diagnostic plots from a VMEC ``wout_*.nc`` file.

    Replicates the output of the standalone ``vmecPlot2.py`` script in a
    vectorised, vmec_jax-native form.  Four figures are written:

    * ``<name>_VMECparams.pdf``  — 9-panel profile + ``|B|`` overview
    * ``<name>_poloidal_plot.png`` — LCFS cross-sections at multiple toroidal angles
    * ``<name>_VMECsurfaces.pdf``  — nested flux-surface cross-sections (8 panels)
    * ``<name>_VMEC_3Dplot.png``   — 3-D LCFS surface coloured by ``|B|``

    Parameters
    ----------
    wout_path:
        Path to the ``wout_*.nc`` file.
    outdir:
        Directory to save figures.  Defaults to the directory containing
        *wout_path*.
    name:
        Base name for output files.  Defaults to the wout stem with the
        leading ``wout_`` stripped (e.g. ``wout_nfp4_QH.nc`` → ``nfp4_QH``).
    s_plot_ignore:
        Fraction of flux surfaces near the axis to ignore when plotting DMerc.
    show:
        If ``True``, call ``plt.show()`` after saving all figures.

    Returns
    -------
    dict
        ``{"vmec_params", "poloidal_plot", "vmec_surfaces", "3d_plot"}``
        mapping to saved :class:`~pathlib.Path` objects.
    """
    prepare_matplotlib_3d()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    from .wout import read_wout as _read_wout

    wout_path = Path(wout_path)
    if outdir is None:
        outdir = wout_path.parent
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if name is None:
        stem = wout_path.stem  # e.g. "wout_nfp4_QH"
        name = stem[5:] if stem.startswith("wout_") else stem

    wout = _read_wout(str(wout_path))

    ns = int(wout.ns)
    nfp = int(wout.nfp)
    ntor = int(wout.ntor)
    lasym = bool(wout.lasym)

    xm = np.asarray(wout.xm, dtype=float)
    xn = np.asarray(wout.xn, dtype=float)
    xm_nyq = np.asarray(wout.xm_nyq, dtype=float)
    xn_nyq = np.asarray(wout.xn_nyq, dtype=float)
    rmnc = np.asarray(wout.rmnc, dtype=float)
    zmns = np.asarray(wout.zmns, dtype=float)
    bmnc = np.asarray(wout.bmnc, dtype=float)
    rmns = np.asarray(wout.rmns, dtype=float) if lasym else np.zeros_like(rmnc)
    zmnc = np.asarray(wout.zmnc, dtype=float) if lasym else np.zeros_like(rmnc)
    bmns = np.asarray(wout.bmns, dtype=float) if lasym else np.zeros_like(bmnc)

    raxis_cc = np.asarray(wout.raxis_cc, dtype=float)
    raxis_cs = np.asarray(wout.raxis_cs, dtype=float)
    zaxis_cs = np.asarray(wout.zaxis_cs, dtype=float)
    zaxis_cc = np.asarray(wout.zaxis_cc, dtype=float)

    phi = np.asarray(wout.phi, dtype=float)
    iotaf = np.asarray(wout.iotaf, dtype=float)
    presf = np.asarray(wout.presf, dtype=float)
    iotas = np.asarray(wout.iotas, dtype=float)
    pres = np.asarray(wout.pres, dtype=float)
    buco = np.asarray(wout.buco, dtype=float)
    bvco = np.asarray(wout.bvco, dtype=float)
    jcuru = np.asarray(wout.jcuru, dtype=float)
    jcurv = np.asarray(wout.jcurv, dtype=float)
    DMerc = np.asarray(wout.DMerc, dtype=float)

    s = np.linspace(0.0, 1.0, ns)
    s_half = [(i - 0.5) / (ns - 1) for i in range(1, ns)]
    xLabel = r"$s = \psi/\psi_b$"

    # ── Helper: evaluate Fourier series on a (ntheta, nzeta) grid ──────────────
    def _eval_rz(isurf: int, theta: np.ndarray, zeta: np.ndarray):
        """(ntheta, nzeta) R and Z arrays for surface isurf."""
        zeta2d, theta2d = np.meshgrid(zeta, theta)
        angles = (xm[:, None, None] * theta2d[None] - xn[:, None, None] * zeta2d[None])
        R = np.tensordot(rmnc[isurf] + rmns[isurf], np.zeros_like(angles[0]), axes=0)
        Z = np.zeros_like(R)
        R = (np.tensordot(rmnc[isurf], np.cos(angles), axes=([0], [0]))
             + np.tensordot(rmns[isurf], np.sin(angles), axes=([0], [0])))
        Z = (np.tensordot(zmns[isurf], np.sin(angles), axes=([0], [0]))
             + np.tensordot(zmnc[isurf], np.cos(angles), axes=([0], [0])))
        return R, Z

    def _eval_bmag(isurf: int, theta: np.ndarray, zeta: np.ndarray):
        """(ntheta, nzeta) |B| array for surface isurf (Nyquist)."""
        zeta2d, theta2d = np.meshgrid(zeta, theta)
        angles = (xm_nyq[:, None, None] * theta2d[None] - xn_nyq[:, None, None] * zeta2d[None])
        B = (np.tensordot(bmnc[isurf], np.cos(angles), axes=([0], [0]))
             + np.tensordot(bmns[isurf], np.sin(angles), axes=([0], [0])))
        return B

    # ── Plot 1: VMECparams — 9-panel diagnostics ────────────────────────────────
    ntheta_b = 30
    nzeta_b = 65
    theta_b = np.linspace(0.0, 2.0 * np.pi, ntheta_b)
    zeta_b = np.linspace(0.0, 2.0 * np.pi, nzeta_b)

    fig1, axes1 = plt.subplots(3, 3, figsize=(14, 7))
    fig1.patch.set_facecolor("white")

    ax = axes1[0, 0]
    ax.plot(s, iotaf, ".-")
    ax.set_xlabel(xLabel)
    ax.set_ylabel(r"$\iota$")

    ax = axes1[0, 1]
    ax.plot(s, presf, ".-", label="presf")
    ax.plot(s_half, pres[1:], ".-", label="pres")
    ax.legend(fontsize="x-small")
    ax.set_xlabel(xLabel)
    ax.set_title("pressure")

    ax = axes1[0, 2]
    ax.plot(s_half, buco[1:], ".-")
    ax.set_title("buco")
    ax.set_xlabel(xLabel)

    ax = axes1[1, 0]
    ax.plot(s_half, bvco[1:], ".-")
    ax.set_title("bvco")
    ax.set_xlabel(xLabel)

    ax = axes1[1, 1]
    ax.plot(s, jcuru, ".-")
    ax.set_title("jcuru")
    ax.set_xlabel(xLabel)

    ax = axes1[1, 2]
    ax.plot(s, jcurv, ".-")
    ax.set_title("jcurv")
    ax.set_xlabel(xLabel)

    ax = axes1[2, 0]
    ign = int(s_plot_ignore * len(s))
    ax.plot(s[ign:-2], DMerc[ign:-2], ".-")
    ax.set_title("DMerc")
    ax.set_xlabel(xLabel)

    _titles_b = ["|B| at half radius", "|B| at LCFS"]
    _iradii_b = [int(round(ns * 0.25)), ns - 1]
    for _col, (_irad, _ttl) in enumerate(zip(_iradii_b, _titles_b)):
        B_b = _eval_bmag(_irad, theta_b, zeta_b)
        zeta2d_b, theta2d_b = np.meshgrid(zeta_b, theta_b)
        ax = axes1[2, 1 + _col]
        cf = ax.contour(zeta2d_b, theta2d_b, B_b, 20, cmap="viridis", linewidths=0.8)
        ax.set_title(f"{_ttl}\n(1-based idx {_irad + 1})")
        ax.set_xlabel("ζ")
        ax.set_ylabel("θ")
        fig1.colorbar(cf, ax=ax)
        iota_val = float(iotaf[_irad])
        if iota_val > 0:
            ax.plot([0, zeta_b.max()], [0, zeta_b.max() * iota_val], "k")
        else:
            ax.plot([0, zeta_b.max()], [-zeta_b.max() * iota_val, 0], "k")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 2 * np.pi])

    fig1.tight_layout()
    fig1.text(0.5, 0.995, str(wout_path.resolve()), ha="center", va="top", fontsize=6)
    out_params = outdir / f"{name}_VMECparams.pdf"
    fig1.savefig(out_params, bbox_inches="tight", pad_inches=0)
    plt.close(fig1)

    # ── Plot 2: Poloidal cross-sections at LCFS ──────────────────────────────────
    ntheta_p = 200
    nzeta_p = 8
    theta_p = np.linspace(0.0, 2.0 * np.pi, ntheta_p)
    zeta_p = np.linspace(0.0, 2.0 * np.pi / nfp, nzeta_p, endpoint=False)
    R_lcfs, Z_lcfs = _eval_rz(ns - 1, theta_p, zeta_p)

    # Axis positions
    nzeta_ax = nzeta_p
    zeta_ax = zeta_p
    n_arr = np.arange(ntor + 1)
    angles_ax = -n_arr[:, None] * nfp * zeta_ax[None, :]  # (ntor+1, nzeta)
    Raxis = (raxis_cc[:ntor + 1, None] * np.cos(angles_ax)
             + raxis_cs[:ntor + 1, None] * np.sin(angles_ax)).sum(axis=0)
    Zaxis = (zaxis_cs[:ntor + 1, None] * np.sin(angles_ax)
             + zaxis_cc[:ntor + 1, None] * np.cos(angles_ax)).sum(axis=0)

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    fig2.patch.set_facecolor("white")
    _zeta_lbls = [
        (0, r"$\phi=0$"),
        (2, r"$\phi=\pi/2$"),
        (4, r"$\phi=\pi$"),
        (6, r"$\phi=3\pi/2$"),
    ]
    for _iz, _lbl in _zeta_lbls:
        if _iz < nzeta_p:
            ax2.plot(R_lcfs[:, _iz], Z_lcfs[:, _iz], "-", label=_lbl)
    ax2.set_aspect("equal", adjustable="box")
    ax2.legend(fontsize=18)
    ax2.set_xlabel("R", fontsize=22)
    ax2.set_ylabel("Z", fontsize=22)
    ax2.tick_params(axis="x", labelsize=16)
    ax2.tick_params(axis="y", labelsize=16)
    fig2.tight_layout()
    out_poloidal = outdir / f"{name}_poloidal_plot.png"
    fig2.savefig(out_poloidal)
    plt.close(fig2)

    # ── Plot 3: VMECsurfaces — 8 nested cross-section panels ─────────────────────
    ntheta_s = 200
    nzeta_s = 8
    nradius_s = 8
    theta_s = np.linspace(0.0, 2.0 * np.pi, ntheta_s)
    zeta_s = np.linspace(0.0, 2.0 * np.pi / nfp, nzeta_s, endpoint=False)
    iradii_s = np.round(np.linspace(0, ns - 1, nradius_s)).astype(int)

    fig3, axes3 = plt.subplots(2, 4, figsize=(14, 7))
    fig3.patch.set_facecolor("white")
    axes3_flat = axes3.ravel()

    for _iz in range(nzeta_s):
        ax = axes3_flat[_iz]
        for _ir, _irad in enumerate(iradii_s):
            R_s, Z_s = _eval_rz(_irad, theta_s, zeta_s)
            ax.plot(R_s[:, _iz], Z_s[:, _iz], "-")
        ax.plot(Raxis[_iz], Zaxis[_iz], "xr")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("R", fontsize=10)
        ax.set_ylabel("Z", fontsize=10)
        ax.set_title(rf"$\phi$ = {round(float(zeta_s[_iz]), 2)}")

    fig3.tight_layout()
    out_surfaces = outdir / f"{name}_VMECsurfaces.pdf"
    fig3.savefig(out_surfaces, bbox_inches="tight", pad_inches=0)
    plt.close(fig3)

    # ── Plot 4: 3-D LCFS surface coloured by |B| ─────────────────────────────────
    ntheta_3d = 80
    nzeta_3d = max(500, int(150 * nfp))
    theta_3d = np.linspace(0.0, 2.0 * np.pi, ntheta_3d)
    zeta_3d = np.linspace(0.0, 2.0 * np.pi, nzeta_3d)

    R_3d, Z_3d = _eval_rz(ns - 1, theta_3d, zeta_3d)
    B_3d = _eval_bmag(ns - 1, theta_3d, zeta_3d)

    zeta2d_3d, _ = np.meshgrid(zeta_3d, theta_3d)
    X_3d = R_3d * np.cos(zeta2d_3d)
    Y_3d = R_3d * np.sin(zeta2d_3d)

    B_rescaled = (B_3d - B_3d.min()) / (B_3d.max() - B_3d.min() + 1e-30)

    fig4 = plt.figure(figsize=(5, 4), frameon=False)
    ax4 = fig4.add_subplot(111, projection="3d")
    ax4.plot_surface(
        X_3d, Y_3d, Z_3d,
        facecolors=cm.jet(B_rescaled),
        rstride=1, cstride=1, antialiased=False,
    )
    scale = 0.7 * max(abs(X_3d).max(), abs(Y_3d).max())
    ax4.auto_scale_xyz(
        [-scale, scale],
        [-scale, scale],
        [-scale, scale],
    )
    ax4.set_box_aspect([1, 1, 1])
    ax4.axis("off")

    cax4 = fig4.add_axes([0.21, 0.80, 0.60, 0.03])
    norm4 = Normalize(vmin=float(B_3d.min()), vmax=float(B_3d.max()))
    sm4 = cm.ScalarMappable(cmap=cm.jet, norm=norm4)
    sm4.set_array([])
    cbar4 = plt.colorbar(sm4, orientation="horizontal", cax=cax4)
    cbar4.set_label("|B| [T]")

    out_3d = outdir / f"{name}_VMEC_3Dplot.png"
    fig4.savefig(out_3d, bbox_inches="tight", pad_inches=0, dpi=400)
    plt.close(fig4)

    if show:
        plt.show()

    results = {
        "vmec_params": out_params,
        "poloidal_plot": out_poloidal,
        "vmec_surfaces": out_surfaces,
        "3d_plot": out_3d,
    }
    for _k, _p in results.items():
        print(f"  [{_k}] {_p}")
    return results
