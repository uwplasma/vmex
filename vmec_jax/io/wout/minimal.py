"""Assembly helpers for VMEC-compatible minimal ``wout`` output.

The public constructor remains :func:`vmec_jax.wout.wout_minimal_from_fixed_boundary`.
This module keeps passive data-shaping pieces out of that high-level routine so
the delicate diagnostic assembly is easier to review and test.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np

from ...vmec_parity import vmec_m1_internal_to_physical_signed_host


class WoutMainGeometryCoefficients(NamedTuple):
    """Physical full-mesh geometry coefficients written to ``wout``."""

    rmnc: np.ndarray
    rmns: np.ndarray
    zmnc: np.ndarray
    zmns: np.ndarray
    lmnc_internal: np.ndarray
    lmns_internal: np.ndarray
    raxis_cc: np.ndarray
    raxis_cs: np.ndarray
    zaxis_cc: np.ndarray
    zaxis_cs: np.ndarray


class WoutProfilePayload(NamedTuple):
    """Flux, pressure, mass, and iota profiles used while assembling WOUT."""

    flux: Any
    chipf_wout: np.ndarray
    phips: np.ndarray
    pres: np.ndarray
    s_half: np.ndarray
    mass: np.ndarray
    ncurr: int
    iotas: np.ndarray
    iotaf: np.ndarray
    gamma: float
    phipf_internal: np.ndarray


def prepare_profile_payload(
    *,
    state: Any,
    static: Any,
    indata: Any,
    modes: Any,
    s: np.ndarray,
    ns: int,
    signgs: int,
    flux_override: Any | None,
    profiles_override: dict | None,
    equilibrium_iota_profiles_from_state_func: Any,
    chipf_from_chips_func: Any,
) -> WoutProfilePayload:
    """Prepare radial profiles for minimal WOUT output.

    This preserves the VMEC output convention that current-driven runs
    recompute ``iota``/``chipf`` from the accepted equilibrium state unless the
    explicit debug environment disables that recompute.
    """

    from ...boundary import boundary_from_indata
    from ...energy import _iotaf_from_iotas, flux_profiles_from_indata
    from ...profiles import eval_profiles

    s_arr = np.asarray(s)
    flux = flux_override if flux_override is not None else flux_profiles_from_indata(indata, s_arr, signgs=int(signgs))
    chipf_wout = np.asarray(flux.chipf)
    phips = np.asarray(flux.phips)
    if phips.size:
        phips = phips.copy()
        phips[0] = 0.0

    if int(ns) < 2:
        s_half = s_arr
    else:
        s_half = np.concatenate([s_arr[:1], 0.5 * (s_arr[1:] + s_arr[:-1])], axis=0)
    prof = dict(profiles_override) if profiles_override is not None else eval_profiles(indata, s_half)
    pres = np.asarray(prof.get("pressure", np.zeros((int(ns),), dtype=float)))
    if pres.size:
        pres = pres.copy()
        pres[0] = 0.0

    boundary = boundary_from_indata(indata, modes)
    idx00 = np.where((np.asarray(modes.m) == 0) & (np.asarray(modes.n) == 0))[0]
    r00 = float(boundary.R_cos[int(idx00[0])]) if idx00.size else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    vnorm = phips
    if lrfp:
        chipf = np.asarray(flux.chipf)
        if chipf.size:
            vnorm = np.concatenate([chipf[:1], 0.5 * (chipf[1:] + chipf[:-1])], axis=0)
    mass = pres * (np.abs(vnorm) * r00) ** gamma
    if mass.size:
        mass = mass.copy()
        mass[0] = 0.0

    ncurr = int(indata.get_int("NCURR", 0))
    iotas = np.asarray(prof.get("iota", np.zeros((int(ns),), dtype=float)))
    if iotas.size:
        iotas = iotas.copy()
        iotas[0] = 0.0
    iotaf = np.asarray(_iotaf_from_iotas(iotas, lrfp=bool(indata.get_bool("LRFP", False))))

    if ncurr == 1 and os.getenv("VMEC_JAX_DISABLE_WOUT_NCURR_RECOMPUTE", "0") in ("", "0"):
        chips, iotas, iotaf = equilibrium_iota_profiles_from_state_func(
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
        )
        chips = np.asarray(chips, dtype=float)
        iotas = np.asarray(iotas, dtype=float)
        iotaf = np.asarray(iotaf, dtype=float)
        chipf_wout = np.asarray(chipf_from_chips_func(chips), dtype=float)

    return WoutProfilePayload(
        flux=flux,
        chipf_wout=np.asarray(chipf_wout),
        phips=phips,
        pres=pres,
        s_half=s_half,
        mass=mass,
        ncurr=int(ncurr),
        iotas=np.asarray(iotas),
        iotaf=np.asarray(iotaf),
        gamma=float(gamma),
        phipf_internal=np.asarray(flux.phipf, dtype=float),
    )


def build_main_geometry_coefficients(
    *,
    state: Any,
    modes: Any,
    ntor: int,
    lasym: bool,
    lconm1: bool,
) -> WoutMainGeometryCoefficients:
    """Convert internal VMEC-JAX coefficients to VMEC ``wout`` convention.

    VMEC's internal ``m=1`` representation and output normalization differ from
    the Fourier coefficients stored in ``wout``.  Keep that conversion in one
    pure NumPy helper so the WOUT builder can focus on diagnostics and file
    schema assembly.
    """

    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    sqrt2 = np.sqrt(2.0)
    mscale = np.where(m_arr == 0, 1.0, sqrt2)
    nscale = np.where(np.abs(n_arr) == 0, 1.0, sqrt2)
    mode_scale = (mscale * nscale)[None, :]

    Rcos_use, Zsin_use, Rsin_use, Zcos_use = vmec_m1_internal_to_physical_signed_host(
        Rcos=np.asarray(state.Rcos, dtype=float),
        Zsin=np.asarray(state.Zsin, dtype=float),
        Rsin=np.asarray(state.Rsin, dtype=float),
        Zcos=np.asarray(state.Zcos, dtype=float),
        modes=modes,
        lthreed=bool(ntor > 0),
        lasym=bool(lasym),
        lconm1=bool(lconm1),
    )
    rmnc = np.asarray(Rcos_use, dtype=float) * mode_scale
    rmns = np.asarray(Rsin_use, dtype=float) * mode_scale
    zmnc = np.asarray(Zcos_use, dtype=float) * mode_scale
    zmns = np.asarray(Zsin_use, dtype=float) * mode_scale
    if not bool(lasym):
        rmns = np.zeros_like(rmnc)
        zmnc = np.zeros_like(zmns)

    lmnc_internal = np.asarray(state.Lcos, dtype=float) * mode_scale
    lmns_internal = np.asarray(state.Lsin, dtype=float) * mode_scale

    raxis_cc, raxis_cs, zaxis_cc, zaxis_cs = axis_coefficients_from_main_modes(
        rmnc=rmnc,
        rmns=rmns,
        zmnc=zmnc,
        zmns=zmns,
        modes=modes,
        ntor=int(ntor),
    )

    return WoutMainGeometryCoefficients(
        rmnc=rmnc,
        rmns=rmns,
        zmnc=zmnc,
        zmns=zmns,
        lmnc_internal=lmnc_internal,
        lmns_internal=lmns_internal,
        raxis_cc=raxis_cc,
        raxis_cs=raxis_cs,
        zaxis_cc=zaxis_cc,
        zaxis_cs=zaxis_cs,
    )


def axis_coefficients_from_main_modes(
    *,
    rmnc: np.ndarray,
    rmns: np.ndarray,
    zmnc: np.ndarray,
    zmns: np.ndarray,
    modes: Any,
    ntor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract magnetic-axis Fourier coefficients from ``m=0`` output modes."""

    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    raxis_cc = np.zeros((int(ntor) + 1,), dtype=float)
    raxis_cs = np.zeros_like(raxis_cc)
    zaxis_cc = np.zeros_like(raxis_cc)
    zaxis_cs = np.zeros_like(raxis_cc)
    for nval in range(int(ntor) + 1):
        mask = (m_arr == 0) & (n_arr == nval)
        if np.any(mask):
            idx = int(np.where(mask)[0][0])
            raxis_cc[nval] = float(np.asarray(rmnc)[0, idx])
            raxis_cs[nval] = float(np.asarray(rmns)[0, idx])
            zaxis_cc[nval] = float(np.asarray(zmnc)[0, idx])
            zaxis_cs[nval] = float(np.asarray(zmns)[0, idx])
    return raxis_cc, raxis_cs, zaxis_cc, zaxis_cs


class WoutMinimalVmecLike:
    """Small VMEC-like payload consumed by bcovar/force reconstruction helpers."""

    __slots__ = (
        "phipf",
        "phips",
        "chipf",
        "iotaf",
        "iotas",
        "signgs",
        "nfp",
        "mpol",
        "ntor",
        "lasym",
        "flux_is_internal",
        "ncurr",
        "lcurrent",
        "icurv",
        "mass",
        "gamma",
    )

    def __init__(
        self,
        *,
        flux: Any,
        chipf: np.ndarray,
        iotaf: np.ndarray,
        iotas: np.ndarray,
        signgs: int,
        nfp: int,
        mpol: int,
        ntor: int,
        lasym: bool,
        ncurr: int,
        mass: np.ndarray,
        gamma: float,
        indata: Any,
        s_full: np.ndarray,
        icurv_full_mesh_from_indata_func: Any,
    ) -> None:
        self.phipf = np.asarray(flux.phipf)
        self.phips = np.asarray(flux.phips)
        self.chipf = np.asarray(chipf)
        self.iotaf = np.asarray(iotaf)
        self.iotas = np.asarray(iotas)
        self.signgs = int(signgs)
        self.nfp = int(nfp)
        self.mpol = int(mpol)
        self.ntor = int(ntor)
        self.lasym = bool(lasym)
        self.flux_is_internal = True
        self.ncurr = int(ncurr)
        self.lcurrent = bool(int(ncurr) == 1)
        self.icurv = np.asarray(
            icurv_full_mesh_from_indata_func(
                indata=indata,
                s_full=np.asarray(s_full, dtype=float),
                signgs=int(signgs),
            )
        )
        self.mass = np.asarray(mass)
        self.gamma = float(gamma)


def build_minimal_wout_data_kwargs(
    context: Mapping[str, Any],
    *,
    path: str | Path,
    converged: bool,
) -> dict[str, Any]:
    """Map a fixed-boundary diagnostic payload to ``WoutData`` kwargs.

    The full WOUT builder computes geometry, field, profile, and stability
    diagnostics.  This helper is intentionally limited to VMEC schema assembly:
    it performs output normalization and dtype coercion, but no physics
    calculations.  Keeping the final schema mapping here makes the high-level
    builder easier to audit while avoiding an import cycle with
    :class:`vmec_jax.io.wout.schema.WoutData`.
    """

    main_modes = context["main_modes"]
    nyq_modes = context["nyq_modes"]
    nfp = int(context["nfp"])
    ns = int(context["ns"])
    indata = context["indata"]
    converged_bool = bool(converged)

    return {
        "path": Path(path),
        "ns": ns,
        "mpol": int(context["mpol"]),
        "ntor": int(context["ntor"]),
        "nfp": nfp,
        "lasym": bool(context["lasym"]),
        "signgs": int(context["signgs"]),
        "mnmax": int(main_modes.K),
        "mpol_nyq": int(np.max(nyq_modes.m)) if int(nyq_modes.K) > 0 else 0,
        "ntor_nyq": int(np.max(np.abs(nyq_modes.n))) if int(nyq_modes.K) > 0 else 0,
        "mnmax_nyq": int(nyq_modes.K),
        "xm": np.asarray(main_modes.m, dtype=int),
        "xn": np.asarray(main_modes.n * nfp, dtype=int),
        "xm_nyq": np.asarray(nyq_modes.m, dtype=int),
        "xn_nyq": np.asarray(nyq_modes.n * nfp, dtype=int),
        "rmnc": np.asarray(context["rmnc"], dtype=float),
        "rmns": np.asarray(context["rmns"], dtype=float),
        "zmnc": np.asarray(context["zmnc"], dtype=float),
        "zmns": np.asarray(context["zmns"], dtype=float),
        "lmnc": np.asarray(context["lmnc"], dtype=float),
        "lmns": np.asarray(context["lmns"], dtype=float),
        "phipf": np.asarray(context["phipf_out"], dtype=float),
        "chipf": np.asarray(context["chipf_out"], dtype=float),
        "phips": np.asarray(context["flux"].phips, dtype=float),
        "iotaf": np.asarray(context["iotaf"], dtype=float),
        "iotas": np.asarray(context["iotas"], dtype=float),
        "gmnc": np.asarray(context["gmnc"], dtype=float),
        "gmns": np.asarray(context["gmns"], dtype=float),
        "bsupumnc": np.asarray(context["bsupumnc"], dtype=float),
        "bsupumns": np.asarray(context["bsupumns"], dtype=float),
        "bsupvmnc": np.asarray(context["bsupvmnc"], dtype=float),
        "bsupvmns": np.asarray(context["bsupvmns"], dtype=float),
        "bsubumnc": np.asarray(context["bsubumnc"], dtype=float),
        "bsubumns": np.asarray(context["bsubumns"], dtype=float),
        "bsubvmnc": np.asarray(context["bsubvmnc"], dtype=float),
        "bsubvmns": np.asarray(context["bsubvmns"], dtype=float),
        "bsubsmns": np.asarray(context["bsubsmns"], dtype=float),
        "bsubsmnc": np.asarray(context["bsubsmnc"], dtype=float),
        "bmnc": np.asarray(context["bmnc"], dtype=float),
        "bmns": np.asarray(context["bmns"], dtype=float),
        "wb": float(context["wb"]),
        "volume_p": float(context["volume_p"]),
        "gamma": float(getattr(indata, "get_float", lambda *_: 0.0)("GAMMA", 0.0)),
        "wp": float(context["wp"]),
        "vp": np.asarray(context["vp"], dtype=float),
        "pres": np.asarray(context["pres"], dtype=float),
        "presf": np.asarray(context["presf"], dtype=float),
        "fsqr": float(context["fsqr"]),
        "fsqz": float(context["fsqz"]),
        "fsql": float(context["fsql"]),
        "fsqt": np.asarray(context["fsqt_out"], dtype=float),
        "equif": np.asarray(context["equif"], dtype=float),
        "phi": np.asarray(context["phi"], dtype=float),
        "buco": np.asarray(context["buco"], dtype=float),
        "bvco": np.asarray(context["bvco"], dtype=float),
        "jcuru": np.asarray(context["jcuru"], dtype=float),
        "jcurv": np.asarray(context["jcurv"], dtype=float),
        "raxis_cc": np.asarray(context["raxis_cc"], dtype=float),
        "zaxis_cs": np.asarray(context["zaxis_cs"], dtype=float),
        "raxis_cs": np.asarray(context["raxis_cs"], dtype=float),
        "zaxis_cc": np.asarray(context["zaxis_cc"], dtype=float),
        "Aminor_p": float(context["Aminor_p"]),
        "Rmajor_p": float(context["Rmajor_p"]),
        "aspect": float(context["aspect"]),
        "betatotal": float(context["betatotal"]),
        "betapol": float(context["betapol"]),
        "betator": float(context["betator"]),
        "betaxis": float(context["betaxis"]),
        "ctor": float(context["ctor"]),
        "DMerc": np.asarray(context["DMerc"], dtype=float),
        "Dshear": np.asarray(context["Dshear"], dtype=float),
        "Dwell": np.asarray(context["Dwell"], dtype=float),
        "Dcurr": np.asarray(context["Dcurr"], dtype=float),
        "Dgeod": np.asarray(context["Dgeod"], dtype=float),
        "D_R": np.asarray(context["D_R"], dtype=float),
        "H": np.asarray(context["H_glasser"], dtype=float),
        "glasser_correction": np.asarray(context["glasser_correction"], dtype=float),
        "glasser_shear_valid": np.asarray(context["glasser_shear_valid"], dtype=bool),
        "jdotb": np.asarray(context["jdotb"], dtype=float),
        "bdotb": np.asarray(context["bdotb"], dtype=float),
        "bdotgradv": np.asarray(context["bdotgradv"], dtype=float),
        "ac": np.asarray(context["ac"], dtype=float),
        "ac_aux_s": np.asarray(context["ac_aux_s"], dtype=float),
        "ac_aux_f": np.asarray(context["ac_aux_f"], dtype=float),
        "pcurr_type": str(context["pcurr_type"]),
        "piota_type": str(context["piota_type"]),
        "ier_flag": 0 if converged_bool else 1,
        "vmec_jax_converged": converged_bool,
        "vmec_jax_status": "converged" if converged_bool else "nonconverged",
    }
