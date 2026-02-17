"""Minimal `wout_*.nc` reader helpers.

This module is intentionally small and only depends on `netCDF4` when used.
It is meant for regression comparisons against VMEC2000 outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .modes import vmec_mode_table
from .modes import nyquist_mode_table
from .state import StateLayout, VMECState
from .vmec_parity import vmec_m1_internal_to_physical_signed, vmec_m1_physical_to_internal_signed


MU0 = 4e-7 * np.pi  # N/A^2


def _chipf_from_chips(chips: np.ndarray) -> np.ndarray:
    """VMEC add_fluxes: build half-mesh chipf from chips on full mesh."""
    chips = np.asarray(chips, dtype=float)
    ns = int(chips.shape[0])
    if ns <= 1:
        return chips.copy()
    chipf = np.zeros((ns,), dtype=chips.dtype)
    if ns >= 3:
        chipf[0] = 1.5 * chips[1] - 0.5 * chips[2]
    else:
        chipf[0] = chips[1]
    if ns > 2:
        chipf[1:-1] = 0.5 * (chips[1:-1] + chips[2:])
    chipf[-1] = 1.5 * chips[-1] - 0.5 * chips[-2]
    return chipf


def _icurv_full_mesh_from_indata(*, indata, s_full: np.ndarray, signgs: int) -> np.ndarray:
    from .profiles import eval_profiles

    s_full = np.asarray(s_full, dtype=float)
    ncurr = int(indata.get_int("NCURR", 0))
    if ncurr != 1:
        return np.zeros_like(s_full)

    curtor = float(indata.get_float("CURTOR", 0.0))
    if abs(curtor) <= np.finfo(float).eps:
        return np.zeros_like(s_full)

    if s_full.size < 2:
        s_half = s_full
    else:
        s_half = np.concatenate([s_full[:1], 0.5 * (s_full[1:] + s_full[:-1])], axis=0)
    prof = eval_profiles(indata, s_half)
    icurv_raw = np.asarray(prof.get("current", np.zeros_like(s_half)))
    if int(icurv_raw.shape[0]) != int(s_full.shape[0]):
        icurv_raw = np.zeros_like(s_half)

    pedge_prof = eval_profiles(indata, np.asarray([1.0], dtype=s_full.dtype))
    pedge = float(np.asarray(pedge_prof.get("current", np.asarray([0.0], dtype=s_full.dtype)))[0])
    if abs(pedge) <= abs(np.finfo(float).eps * curtor):
        return np.zeros_like(s_full)

    mu0 = 4e-7 * np.pi
    currv = mu0 * curtor
    scale = float(signgs) * currv / (2.0 * np.pi * pedge)
    icurv = np.asarray(scale, dtype=icurv_raw.dtype) * icurv_raw
    if int(icurv.shape[0]) > 0:
        icurv[0] = 0.0
    return icurv


@dataclass(frozen=True)
class WoutData:
    path: Path
    ns: int
    mpol: int
    ntor: int
    nfp: int
    lasym: bool
    signgs: int

    # main mode table
    xm: np.ndarray
    xn: np.ndarray

    # nyquist mode table
    xm_nyq: np.ndarray
    xn_nyq: np.ndarray

    # geometry coefficients (full mesh)
    rmnc: np.ndarray
    rmns: np.ndarray
    zmnc: np.ndarray
    zmns: np.ndarray
    lmnc: np.ndarray
    lmns: np.ndarray

    # flux functions / profiles
    phipf: np.ndarray
    chipf: np.ndarray
    phips: np.ndarray
    iotaf: np.ndarray  # (ns,) iota on full mesh (VMEC convention)
    iotas: np.ndarray  # (ns,) iota on half mesh (VMEC convention)

    # nyquist Fourier coefficients for derived fields
    gmnc: np.ndarray
    gmns: np.ndarray
    bsupumnc: np.ndarray
    bsupumns: np.ndarray
    bsupvmnc: np.ndarray
    bsupvmns: np.ndarray

    # nyquist Fourier coefficients for covariant field components (for parity checks)
    bsubumnc: np.ndarray
    bsubumns: np.ndarray
    bsubvmnc: np.ndarray
    bsubvmns: np.ndarray

    # nyquist Fourier coefficients for |B|
    bmnc: np.ndarray
    bmns: np.ndarray

    wb: float
    volume_p: float

    # pressure / energy scalars (VMEC internal units)
    gamma: float
    wp: float
    vp: np.ndarray  # (ns,) volume derivative on half mesh, normalized by (2π)^2
    pres: np.ndarray  # (ns,) pressure on half mesh in mu0*Pa (B^2 units)
    presf: np.ndarray  # (ns,) pressure on full mesh in mu0*Pa (B^2 units)

    # force residual diagnostics (VMEC scalars)
    fsqr: float  # radial force residual
    fsqz: float  # vertical force residual
    fsql: float  # lambda/constraint residual
    fsqt: np.ndarray  # force trace vs iteration (if present)
    equif: np.ndarray  # (ns,) flux-surface-averaged force balance (if present)

    # additional wout fields used by vmecPlot2 and diagnostics
    phi: np.ndarray  # (ns,) toroidal flux
    buco: np.ndarray  # (ns,)
    bvco: np.ndarray  # (ns,)
    jcuru: np.ndarray  # (ns,)
    jcurv: np.ndarray  # (ns,)
    raxis_cc: np.ndarray  # (ntor+1,)
    zaxis_cs: np.ndarray  # (ntor+1,)
    raxis_cs: np.ndarray  # (ntor+1,)
    zaxis_cc: np.ndarray  # (ntor+1,)
    Aminor_p: float
    Rmajor_p: float
    aspect: float
    betatotal: float
    betapol: float
    betator: float
    betaxis: float
    ctor: float
    DMerc: np.ndarray  # (ns,)
    ac: np.ndarray  # (nac,)
    ac_aux_s: np.ndarray  # (ndfmax,)
    ac_aux_f: np.ndarray  # (ndfmax,)
    pcurr_type: str
    piota_type: str


def _bool_from_nc(x: Any) -> bool:
    # VMEC stores *_logical__ as 0/1 integers in netcdf.
    try:
        return bool(int(np.asarray(x)))
    except Exception:
        return bool(x)


def read_wout(path: str | Path) -> WoutData:
    """Read a subset of `wout_*.nc` needed for regression comparisons."""
    path = Path(path)
    try:
        import netCDF4  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("netCDF4 is required to read wout files (pip install -e .[netcdf])") from e

    with netCDF4.Dataset(path) as ds:
        ns = int(ds.variables["ns"][:])
        mpol = int(ds.variables["mpol"][:])
        ntor = int(ds.variables["ntor"][:])
        nfp = int(ds.variables["nfp"][:])
        lasym = _bool_from_nc(ds.variables.get("lasym__logical__", 0)[:])
        signgs = int(ds.variables["signgs"][:])

        xm = np.asarray(ds.variables["xm"][:], dtype=int)
        xn = np.asarray(ds.variables["xn"][:], dtype=int)
        xm_nyq = np.asarray(ds.variables["xm_nyq"][:], dtype=int)
        xn_nyq = np.asarray(ds.variables["xn_nyq"][:], dtype=int)

        rmnc = np.asarray(ds.variables["rmnc"][:])
        rmns = np.asarray(ds.variables.get("rmns", np.zeros_like(rmnc))[:])
        zmns = np.asarray(ds.variables["zmns"][:])
        zmnc = np.asarray(ds.variables.get("zmnc", np.zeros_like(zmns))[:])
        lmns = np.asarray(ds.variables["lmns"][:])
        lmnc = np.asarray(ds.variables.get("lmnc", np.zeros_like(lmns))[:])

        phipf = np.asarray(ds.variables["phipf"][:])
        chipf = np.asarray(ds.variables["chipf"][:])
        phips = np.asarray(ds.variables["phips"][:])
        iotaf = np.asarray(ds.variables.get("iotaf", np.zeros_like(phips))[:])
        iotas = np.asarray(ds.variables.get("iotas", np.zeros_like(phips))[:])

        gmnc = np.asarray(ds.variables["gmnc"][:])
        gmns = np.asarray(ds.variables.get("gmns", np.zeros_like(gmnc))[:])
        bsupumnc = np.asarray(ds.variables["bsupumnc"][:])
        bsupumns = np.asarray(ds.variables.get("bsupumns", np.zeros_like(bsupumnc))[:])
        bsupvmnc = np.asarray(ds.variables["bsupvmnc"][:])
        bsupvmns = np.asarray(ds.variables.get("bsupvmns", np.zeros_like(bsupvmnc))[:])

        bsubumnc = np.asarray(ds.variables.get("bsubumnc", np.zeros_like(bsupumnc))[:])
        bsubumns = np.asarray(ds.variables.get("bsubumns", np.zeros_like(bsupumnc))[:])
        bsubvmnc = np.asarray(ds.variables.get("bsubvmnc", np.zeros_like(bsupvmnc))[:])
        bsubvmns = np.asarray(ds.variables.get("bsubvmns", np.zeros_like(bsupvmnc))[:])

        bmnc = np.asarray(ds.variables.get("bmnc", np.zeros_like(gmnc))[:])
        bmns = np.asarray(ds.variables.get("bmns", np.zeros_like(gmnc))[:])

        wb = float(ds.variables["wb"][:])
        volume_p = float(ds.variables["volume_p"][:])
        gamma = float(ds.variables.get("gamma", 0.0)[:]) if "gamma" in ds.variables else 0.0
        wp = float(ds.variables.get("wp", 0.0)[:]) if "wp" in ds.variables else 0.0
        vp = np.asarray(ds.variables.get("vp", np.zeros((ns,), dtype=float))[:])

        # `wout` stores pres/presf divided by mu0. Convert back to VMEC internal
        # units (mu0*Pa) so it matches the energy functional.
        pres_pa = np.asarray(ds.variables.get("pres", np.zeros((ns,), dtype=float))[:])
        presf_pa = np.asarray(ds.variables.get("presf", np.zeros((ns,), dtype=float))[:])
        pres = MU0 * pres_pa
        presf = MU0 * presf_pa

        # Force residual scalars (present in most VMEC wout files).
        fsqr = float(ds.variables.get("fsqr", 0.0)[:]) if "fsqr" in ds.variables else 0.0
        fsqz = float(ds.variables.get("fsqz", 0.0)[:]) if "fsqz" in ds.variables else 0.0
        fsql = float(ds.variables.get("fsql", 0.0)[:]) if "fsql" in ds.variables else 0.0
        fsqt = np.asarray(ds.variables.get("fsqt", np.zeros((0,), dtype=float))[:])
        equif = np.asarray(ds.variables.get("equif", np.zeros((ns,), dtype=float))[:])

        # Additional fields used by vmecPlot2 and diagnostics.
        if "phi" in ds.variables:
            phi = np.asarray(ds.variables["phi"][:])
        else:
            from .integrals import cumrect_s_halfmesh

            if ns < 2:
                s = np.zeros((ns,), dtype=float)
            else:
                s = np.linspace(0.0, 1.0, ns, dtype=float)
            phi = np.asarray(cumrect_s_halfmesh(phipf, s))

        buco = np.asarray(ds.variables.get("buco", np.zeros((ns,), dtype=float))[:])
        bvco = np.asarray(ds.variables.get("bvco", np.zeros((ns,), dtype=float))[:])
        jcuru = np.asarray(ds.variables.get("jcuru", np.zeros((ns,), dtype=float))[:])
        jcurv = np.asarray(ds.variables.get("jcurv", np.zeros((ns,), dtype=float))[:])

        raxis_cc = np.asarray(ds.variables.get("raxis_cc", np.zeros((ntor + 1,), dtype=float))[:])
        zaxis_cs = np.asarray(ds.variables.get("zaxis_cs", np.zeros((ntor + 1,), dtype=float))[:])
        raxis_cs = np.asarray(ds.variables.get("raxis_cs", np.zeros_like(raxis_cc))[:])
        zaxis_cc = np.asarray(ds.variables.get("zaxis_cc", np.zeros_like(zaxis_cs))[:])

        Aminor_p = float(ds.variables.get("Aminor_p", 0.0)[:]) if "Aminor_p" in ds.variables else 0.0
        Rmajor_p = float(ds.variables.get("Rmajor_p", 0.0)[:]) if "Rmajor_p" in ds.variables else 0.0
        aspect = float(ds.variables.get("aspect", 0.0)[:]) if "aspect" in ds.variables else 0.0
        betatotal = float(ds.variables.get("betatotal", 0.0)[:]) if "betatotal" in ds.variables else 0.0
        betapol = float(ds.variables.get("betapol", 0.0)[:]) if "betapol" in ds.variables else 0.0
        betator = float(ds.variables.get("betator", 0.0)[:]) if "betator" in ds.variables else 0.0
        betaxis = float(ds.variables.get("betaxis", 0.0)[:]) if "betaxis" in ds.variables else 0.0
        ctor = float(ds.variables.get("ctor", 0.0)[:]) if "ctor" in ds.variables else 0.0

        DMerc = np.asarray(ds.variables.get("DMerc", np.zeros((ns,), dtype=float))[:])

        ac = np.asarray(ds.variables.get("ac", np.zeros((0,), dtype=float))[:])
        ac_aux_s = np.asarray(ds.variables.get("ac_aux_s", -np.ones((101,), dtype=float))[:])
        ac_aux_f = np.asarray(ds.variables.get("ac_aux_f", np.zeros((101,), dtype=float))[:])

        def _read_type_field(name: str) -> str:
            if name not in ds.variables:
                return ""
            raw = np.asarray(ds.variables[name][:])
            if raw.dtype.kind in ("S", "U"):
                if raw.ndim == 0:
                    out = str(raw)
                else:
                    out = b"".join(raw.astype("S1")).decode("utf-8", "ignore")
            else:
                try:
                    out = "".join(raw.tolist())
                except Exception:
                    out = str(raw)
            return out.rstrip()

        pcurr_type = _read_type_field("pcurr_type")
        piota_type = _read_type_field("piota_type")

    return WoutData(
        path=path,
        ns=ns,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        lasym=lasym,
        signgs=signgs,
        xm=xm,
        xn=xn,
        xm_nyq=xm_nyq,
        xn_nyq=xn_nyq,
        rmnc=rmnc,
        rmns=rmns,
        zmnc=zmnc,
        zmns=zmns,
        lmnc=lmnc,
        lmns=lmns,
        phipf=phipf,
        chipf=chipf,
        phips=phips,
        iotaf=iotaf,
        iotas=iotas,
        gmnc=gmnc,
        gmns=gmns,
        bsupumnc=bsupumnc,
        bsupumns=bsupumns,
        bsupvmnc=bsupvmnc,
        bsupvmns=bsupvmns,
        bsubumnc=bsubumnc,
        bsubumns=bsubumns,
        bsubvmnc=bsubvmnc,
        bsubvmns=bsubvmns,
        bmnc=bmnc,
        bmns=bmns,
        wb=wb,
        volume_p=volume_p,
        gamma=gamma,
        wp=wp,
        vp=vp,
        pres=pres,
        presf=presf,
        fsqr=fsqr,
        fsqz=fsqz,
        fsql=fsql,
        fsqt=fsqt,
        equif=equif,
        phi=phi,
        buco=buco,
        bvco=bvco,
        jcuru=jcuru,
        jcurv=jcurv,
        raxis_cc=raxis_cc,
        zaxis_cs=zaxis_cs,
        raxis_cs=raxis_cs,
        zaxis_cc=zaxis_cc,
        Aminor_p=Aminor_p,
        Rmajor_p=Rmajor_p,
        aspect=aspect,
        betatotal=betatotal,
        betapol=betapol,
        betator=betator,
        betaxis=betaxis,
        ctor=ctor,
        DMerc=DMerc,
        ac=ac,
        ac_aux_s=ac_aux_s,
        ac_aux_f=ac_aux_f,
        pcurr_type=pcurr_type,
        piota_type=piota_type,
    )


def write_wout(path: str | Path, wout: WoutData, *, overwrite: bool = False) -> None:
    """Write a minimal VMEC-style ``wout_*.nc`` file.

    This is intended for:
    - round-tripping reference ``wout`` files (read -> write -> read),
    - emitting VMEC-compatible output containers from vmec_jax as parity work progresses.

    Notes
    -----
    - Only the subset of variables represented in :class:`WoutData` is written.
    - ``pres`` and ``presf`` are written in **Pa** (VMEC convention: netCDF stores
      pressure divided by ``mu0``). Internally :class:`WoutData` stores pressure in
      VMEC internal units (``mu0*Pa``).
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True to overwrite)")

    try:
        import netCDF4  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("netCDF4 is required to write wout files (pip install -e .[netcdf])") from e

    # Dimensions.
    ns = int(wout.ns)
    mnmax = int(np.asarray(wout.xm).size)
    mnmax_nyq = int(np.asarray(wout.xm_nyq).size)
    nstore = int(np.asarray(wout.fsqt).size)
    n_tor = int(wout.ntor) + 1
    ac = np.asarray(getattr(wout, "ac", np.zeros((0,), dtype=float)))
    if ac.size == 0:
        ac = np.zeros((1,), dtype=float)
    ac_aux_s = np.asarray(getattr(wout, "ac_aux_s", -np.ones((101,), dtype=float)))
    ac_aux_f = np.asarray(getattr(wout, "ac_aux_f", np.zeros((101,), dtype=float)))
    if ac_aux_s.size == 0:
        ac_aux_s = -np.ones((1,), dtype=float)
    if ac_aux_f.size == 0:
        ac_aux_f = np.zeros((1,), dtype=float)
    ndfmax = int(ac_aux_s.size)
    preset = int(ac.size)

    # Convert pressures back to VMEC netcdf convention (Pa).
    pres_pa = np.asarray(wout.pres) / MU0
    presf_pa = np.asarray(wout.presf) / MU0

    # Use VMEC-like dimension names for better interoperability with external tools.
    with netCDF4.Dataset(path, mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("radius", ns)
        ds.createDimension("mn_mode", mnmax)
        ds.createDimension("mn_mode_nyq", mnmax_nyq)
        ds.createDimension("nstore_seq", nstore)
        ds.createDimension("n_tor", n_tor)
        ds.createDimension("ndfmax", ndfmax)
        ds.createDimension("preset", preset)
        ds.createDimension("dim_00020", 20)

        def _var_i(name: str, dims: tuple[str, ...], data: np.ndarray) -> None:
            v = ds.createVariable(name, "i4", dims)
            v[:] = np.asarray(data, dtype=np.int32)

        def _var_f(name: str, dims: tuple[str, ...], data: np.ndarray) -> None:
            v = ds.createVariable(name, "f8", dims)
            v[:] = np.asarray(data, dtype=np.float64)

        # Scalars.
        _var_i("ns", (), np.asarray(ns))
        _var_i("mpol", (), np.asarray(int(wout.mpol)))
        _var_i("ntor", (), np.asarray(int(wout.ntor)))
        _var_i("nfp", (), np.asarray(int(wout.nfp)))
        _var_i("signgs", (), np.asarray(int(wout.signgs)))
        _var_i("lasym__logical__", (), np.asarray(int(bool(wout.lasym))))

        _var_f("wb", (), np.asarray(float(wout.wb)))
        _var_f("volume_p", (), np.asarray(float(wout.volume_p)))
        _var_f("gamma", (), np.asarray(float(wout.gamma)))
        _var_f("wp", (), np.asarray(float(wout.wp)))
        _var_f("fsqr", (), np.asarray(float(wout.fsqr)))
        _var_f("fsqz", (), np.asarray(float(wout.fsqz)))
        _var_f("fsql", (), np.asarray(float(wout.fsql)))

        # Mode tables.
        _var_i("xm", ("mn_mode",), np.asarray(wout.xm))
        _var_i("xn", ("mn_mode",), np.asarray(wout.xn))
        _var_i("xm_nyq", ("mn_mode_nyq",), np.asarray(wout.xm_nyq))
        _var_i("xn_nyq", ("mn_mode_nyq",), np.asarray(wout.xn_nyq))

        # Geometry coefficients (full mesh).
        _var_f("rmnc", ("radius", "mn_mode"), np.asarray(wout.rmnc))
        _var_f("rmns", ("radius", "mn_mode"), np.asarray(wout.rmns))
        _var_f("zmnc", ("radius", "mn_mode"), np.asarray(wout.zmnc))
        _var_f("zmns", ("radius", "mn_mode"), np.asarray(wout.zmns))
        _var_f("lmnc", ("radius", "mn_mode"), np.asarray(wout.lmnc))
        _var_f("lmns", ("radius", "mn_mode"), np.asarray(wout.lmns))

        # Flux functions / profiles.
        _var_f("phipf", ("radius",), np.asarray(wout.phipf))
        _var_f("chipf", ("radius",), np.asarray(wout.chipf))
        _var_f("phips", ("radius",), np.asarray(wout.phips))
        _var_f("iotaf", ("radius",), np.asarray(wout.iotaf))
        _var_f("iotas", ("radius",), np.asarray(wout.iotas))
        _var_f("phi", ("radius",), np.asarray(getattr(wout, "phi", np.zeros((ns,), dtype=float))))

        # Nyquist Fourier fields.
        _var_f("gmnc", ("radius", "mn_mode_nyq"), np.asarray(wout.gmnc))
        _var_f("gmns", ("radius", "mn_mode_nyq"), np.asarray(wout.gmns))
        _var_f("bsupumnc", ("radius", "mn_mode_nyq"), np.asarray(wout.bsupumnc))
        _var_f("bsupumns", ("radius", "mn_mode_nyq"), np.asarray(wout.bsupumns))
        _var_f("bsupvmnc", ("radius", "mn_mode_nyq"), np.asarray(wout.bsupvmnc))
        _var_f("bsupvmns", ("radius", "mn_mode_nyq"), np.asarray(wout.bsupvmns))

        _var_f("bsubumnc", ("radius", "mn_mode_nyq"), np.asarray(wout.bsubumnc))
        _var_f("bsubumns", ("radius", "mn_mode_nyq"), np.asarray(wout.bsubumns))
        _var_f("bsubvmnc", ("radius", "mn_mode_nyq"), np.asarray(wout.bsubvmnc))
        _var_f("bsubvmns", ("radius", "mn_mode_nyq"), np.asarray(wout.bsubvmns))

        _var_f("bmnc", ("radius", "mn_mode_nyq"), np.asarray(wout.bmnc))
        _var_f("bmns", ("radius", "mn_mode_nyq"), np.asarray(wout.bmns))

        # 1D radial fields.
        _var_f("vp", ("radius",), np.asarray(wout.vp))
        _var_f("pres", ("radius",), np.asarray(pres_pa))
        _var_f("presf", ("radius",), np.asarray(presf_pa))
        _var_f("equif", ("radius",), np.asarray(getattr(wout, "equif", np.zeros((ns,), dtype=float))))
        _var_f("buco", ("radius",), np.asarray(getattr(wout, "buco", np.zeros((ns,), dtype=float))))
        _var_f("bvco", ("radius",), np.asarray(getattr(wout, "bvco", np.zeros((ns,), dtype=float))))
        _var_f("jcuru", ("radius",), np.asarray(getattr(wout, "jcuru", np.zeros((ns,), dtype=float))))
        _var_f("jcurv", ("radius",), np.asarray(getattr(wout, "jcurv", np.zeros((ns,), dtype=float))))
        _var_f("DMerc", ("radius",), np.asarray(getattr(wout, "DMerc", np.zeros((ns,), dtype=float))))

        # Iteration trace (optional).
        _var_f("fsqt", ("nstore_seq",), np.asarray(wout.fsqt))

        # Axis coefficients and geometric scalars.
        _var_f("raxis_cc", ("n_tor",), np.asarray(getattr(wout, "raxis_cc", np.zeros((n_tor,), dtype=float))))
        _var_f("zaxis_cs", ("n_tor",), np.asarray(getattr(wout, "zaxis_cs", np.zeros((n_tor,), dtype=float))))
        _var_f("raxis_cs", ("n_tor",), np.asarray(getattr(wout, "raxis_cs", np.zeros((n_tor,), dtype=float))))
        _var_f("zaxis_cc", ("n_tor",), np.asarray(getattr(wout, "zaxis_cc", np.zeros((n_tor,), dtype=float))))

        _var_f("Aminor_p", (), np.asarray(float(getattr(wout, "Aminor_p", 0.0))))
        _var_f("Rmajor_p", (), np.asarray(float(getattr(wout, "Rmajor_p", 0.0))))
        _var_f("aspect", (), np.asarray(float(getattr(wout, "aspect", 0.0))))
        _var_f("betatotal", (), np.asarray(float(getattr(wout, "betatotal", 0.0))))
        _var_f("betapol", (), np.asarray(float(getattr(wout, "betapol", 0.0))))
        _var_f("betator", (), np.asarray(float(getattr(wout, "betator", 0.0))))
        _var_f("betaxis", (), np.asarray(float(getattr(wout, "betaxis", 0.0))))
        _var_f("ctor", (), np.asarray(float(getattr(wout, "ctor", 0.0))))

        _var_f("ac_aux_s", ("ndfmax",), np.asarray(ac_aux_s))
        _var_f("ac_aux_f", ("ndfmax",), np.asarray(ac_aux_f))
        _var_f("ac", ("preset",), np.asarray(ac))

        pcurr = str(getattr(wout, "pcurr_type", "") or "")
        pcurr = (pcurr[:20]).ljust(20)
        v = ds.createVariable("pcurr_type", "S1", ("dim_00020",))
        v[:] = np.asarray(list(pcurr), dtype="S1")

        piota = str(getattr(wout, "piota_type", "") or "")
        piota = (piota[:20]).ljust(20)
        v = ds.createVariable("piota_type", "S1", ("dim_00020",))
        v[:] = np.asarray(list(piota), dtype="S1")


def assert_main_modes_match_wout(*, wout: WoutData) -> None:
    """Ensure vmec_jax mode ordering matches the `wout` file (important for parity)."""
    modes = vmec_mode_table(wout.mpol, wout.ntor)
    if modes.K != int(wout.xm.size):
        raise ValueError(f"Mode count mismatch: vmec_jax K={modes.K} wout mnmax={wout.xm.size}")
    if not np.array_equal(modes.m, wout.xm.astype(int)):
        raise ValueError("wout xm ordering does not match vmec_jax vmec_mode_table")
    if not np.array_equal(modes.n, (wout.xn // wout.nfp).astype(int)):
        raise ValueError("wout xn ordering does not match vmec_jax (expected xn = n*nfp)")


def wout_minimal_from_fixed_boundary(
    *,
    path: str | Path,
    state: VMECState,
    static,
    indata,
    signgs: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
) -> WoutData:
    """Build a minimal :class:`WoutData` from an input-only fixed-boundary run.

    This helper is intended for producing VMEC-compatible output files from
    vmec_jax *without* reading any existing `wout_*.nc` as an input.

    Scope:
    - Writes the main Fourier coefficients (R/Z/lambda) using `vmec_mode_table`.
    - Writes flux functions and profiles derived from `indata` (same path used by the solver).
    - Sets Nyquist-derived fields (gmnc/bsup*/bsub*/bmnc) to zeros for now.
      These can be filled in later once the full VMEC nyquist output path is
      fully ported end-to-end.
    """
    from .energy import flux_profiles_from_indata
    from .profiles import eval_profiles
    from .field import full_mesh_from_half_mesh_avg
    from .integrals import cumrect_s_halfmesh
    from .vmec_tomnsp import vmec_trig_tables
    from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout
    from .vmec_realspace import vmec_realspace_analysis
    from .vmec_residue import vmec_force_norms_from_bcovar_dynamic
    from .vmec_lforbal import equif_from_bcovar, currents_from_bcovar
    from .plotting import surface_rz_from_state_physical

    cfg = static.cfg
    ns = int(cfg.ns)
    mpol = int(cfg.mpol)
    ntor = int(cfg.ntor)
    nfp = int(cfg.nfp)
    lasym = bool(cfg.lasym)

    main_modes = vmec_mode_table(mpol, ntor)
    if int(main_modes.K) != int(state.layout.K):
        raise ValueError("state mode count does not match vmec_mode_table(mpol,ntor)")

    nyq_modes = nyquist_mode_table(mpol, ntor)

    # Flux and profiles on VMEC half mesh.
    s = np.asarray(static.s)
    flux = flux_profiles_from_indata(indata, s, signgs=int(signgs))
    chipf_wout = np.asarray(flux.chipf)

    if ns < 2:
        s_half = s
    else:
        s_half = np.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])], axis=0)
    prof = eval_profiles(indata, s_half)
    pres = np.asarray(prof.get("pressure", np.zeros((ns,), dtype=float)))
    if pres.size:
        pres = pres.copy()
        pres[0] = 0.0
    if ns < 2:
        presf = pres.copy()
    else:
        presf = np.zeros_like(pres)
        if ns >= 3:
            presf[0] = 1.5 * pres[1] - 0.5 * pres[2]
        else:
            presf[0] = pres[1]
        presf[1:-1] = 0.5 * (pres[1:-1] + pres[2:])
        presf[-1] = 1.5 * pres[-1] - 0.5 * pres[-2]
    ncurr = int(indata.get_int("NCURR", 0))
    iotas = np.asarray(prof.get("iota", np.zeros((ns,), dtype=float)))
    if iotas.size:
        iotas = iotas.copy()
        iotas[0] = 0.0
    from .energy import _iotaf_from_iotas

    iotaf = np.asarray(_iotaf_from_iotas(iotas, lrfp=bool(indata.get_bool("LRFP", False))))

    # For current-driven runs (ncurr=1), VMEC updates iota from the force balance
    # (add_fluxes). Recompute iotas/iotaf from the final state to match VMEC2000.
    if ncurr == 1:
        from types import SimpleNamespace
        from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout
        from .vmec_residue import vmec_pwint_from_trig
        from .vmec_tomnsp import vmec_trig_tables

        icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=int(signgs))
        phips = np.asarray(flux.phips, dtype=float)
        if phips.size:
            phips = phips.copy()
            phips[0] = 0.0

        wout_like_pre = SimpleNamespace(
            phipf=np.asarray(flux.phipf, dtype=float),
            phips=phips,
            chipf=np.zeros_like(np.asarray(flux.phipf, dtype=float)),
            signgs=int(signgs),
            nfp=int(static.cfg.nfp),
            mpol=int(static.cfg.mpol),
            ntor=int(static.cfg.ntor),
            lasym=bool(static.cfg.lasym),
            ncurr=0,
            lcurrent=False,
            icurv=np.zeros_like(np.asarray(flux.phipf, dtype=float)),
            flux_is_internal=True,
        )

        trig = getattr(static, "trig_vmec", None)
        if trig is None:
            trig = vmec_trig_tables(
                ntheta=int(static.cfg.ntheta),
                nzeta=int(static.cfg.nzeta),
                nfp=int(static.cfg.nfp),
                mmax=int(static.cfg.mpol) - 1,
                nmax=int(static.cfg.ntor),
                lasym=bool(static.cfg.lasym),
                dtype=np.asarray(state.Rcos).dtype,
            )

        bc_pre = vmec_bcovar_half_mesh_from_wout(
            state=state,
            static=static,
            wout=wout_like_pre,
            pres=pres,
            use_vmec_synthesis=True,
            trig=trig,
        )

        sqrtg = np.asarray(bc_pre.jac.sqrtg)
        overg = np.zeros_like(sqrtg)
        np.divide(1.0, sqrtg, out=overg, where=(sqrtg != 0.0))
        pwint = np.asarray(
            vmec_pwint_from_trig(trig, ns=int(overg.shape[0]), nzeta=int(overg.shape[2])),
            dtype=overg.dtype,
        )
        guu = np.asarray(bc_pre.guu)
        guv = np.asarray(bc_pre.guv)
        bsupu = np.asarray(bc_pre.bsupu)
        bsupv = np.asarray(bc_pre.bsupv)

        top = np.asarray(icurv, dtype=overg.dtype) - np.sum(
            pwint * ((guu * bsupu) + (guv * bsupv)),
            axis=(1, 2),
        )
        bot = np.sum(pwint * (overg * guu), axis=(1, 2))
        chips = np.zeros_like(top)
        np.divide(top, bot, out=chips, where=(bot != 0.0))
        if chips.size:
            chips[0] = 0.0

        iotas = np.zeros_like(chips)
        np.divide(chips, phips, out=iotas, where=(phips != 0.0))
        if iotas.size:
            iotas[0] = 0.0
        iotaf = np.asarray(_iotaf_from_iotas(iotas, lrfp=bool(indata.get_bool("LRFP", False))))
        chipf_wout = _chipf_from_chips(chips)

    # Geometry coefficients on the full mesh (convert internal -> external).
    m_arr = np.asarray(main_modes.m, dtype=int)
    n_arr = np.asarray(main_modes.n, dtype=int)
    sqrt2 = np.sqrt(2.0)
    mscale = np.where(m_arr == 0, 1.0, sqrt2)
    nscale = np.where(np.abs(n_arr) == 0, 1.0, sqrt2)
    mode_scale = (mscale * nscale)[None, :]
    lconm1 = bool(getattr(cfg, "lconm1", True))
    Rcos_use, Zsin_use, Rsin_use, Zcos_use = vmec_m1_internal_to_physical_signed(
        Rcos=np.asarray(state.Rcos, dtype=float),
        Zsin=np.asarray(state.Zsin, dtype=float),
        Rsin=np.asarray(state.Rsin, dtype=float),
        Zcos=np.asarray(state.Zcos, dtype=float),
        modes=main_modes,
        lthreed=bool(ntor > 0),
        lasym=bool(lasym),
        lconm1=bool(lconm1),
    )
    rmnc = np.asarray(Rcos_use, dtype=float) * mode_scale
    rmns = np.asarray(Rsin_use, dtype=float) * mode_scale
    zmnc = np.asarray(Zcos_use, dtype=float) * mode_scale
    zmns = np.asarray(Zsin_use, dtype=float) * mode_scale
    lmnc_internal = np.asarray(state.Lcos, dtype=float)
    lmns_internal = np.asarray(state.Lsin, dtype=float)
    lmnc_internal = lmnc_internal * mode_scale
    lmns_internal = lmns_internal * mode_scale

    mnmax_nyq = int(nyq_modes.K)
    z2 = np.zeros((ns, mnmax_nyq), dtype=float)
    z1 = np.zeros((ns,), dtype=float)

    # Toroidal flux (VMEC `phi`) in physical units.
    phipf_internal = np.asarray(flux.phipf, dtype=float)
    phipf_out = phipf_internal * float(2.0 * np.pi * signgs)
    chipf_out = np.asarray(chipf_wout, dtype=float) * float(2.0 * np.pi * signgs)
    phi = np.asarray(cumrect_s_halfmesh(phipf_out, s))

    # Axis coefficients from m=0 modes on the magnetic axis.
    raxis_cc = np.zeros((ntor + 1,), dtype=float)
    raxis_cs = np.zeros_like(raxis_cc)
    zaxis_cs = np.zeros_like(raxis_cc)
    zaxis_cc = np.zeros_like(raxis_cc)
    for nval in range(ntor + 1):
        mask = (m_arr == 0) & (n_arr == nval)
        if np.any(mask):
            idx = int(np.where(mask)[0][0])
            raxis_cc[nval] = float(rmnc[0, idx])
            raxis_cs[nval] = float(rmns[0, idx])
            zaxis_cs[nval] = float(zmns[0, idx])
            zaxis_cc[nval] = float(zmnc[0, idx])

    # Simple geometric measures for plotting (VMEC-style Aminor/Rmajor).
    if ns >= 1:
        theta_dense = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
        Rb, Zb = surface_rz_from_state_physical(
            state,
            static.modes,
            theta=theta_dense,
            phi=np.asarray([0.0]),
            s_index=int(ns - 1),
            nfp=int(nfp),
        )
        Rb = np.asarray(Rb)[:, 0]
        Rmin = float(np.min(Rb))
        Rmax = float(np.max(Rb))
        Aminor_p = 0.5 * (Rmax - Rmin)
        Rmajor_p = 0.5 * (Rmax + Rmin)
        aspect = Rmajor_p / Aminor_p if Aminor_p != 0.0 else 0.0
    else:
        Aminor_p = 0.0
        Rmajor_p = 0.0
        aspect = 0.0

    # Build VMEC parity grids for Nyquist outputs.
    mmax = int(np.max(nyq_modes.m)) if int(nyq_modes.K) > 0 else 0
    nmax = int(np.max(np.abs(nyq_modes.n))) if int(nyq_modes.K) > 0 else 0
    ntheta_base = int(cfg.ntheta)
    trig = vmec_trig_tables(
        ntheta=int(ntheta_base),
        nzeta=int(cfg.nzeta),
        nfp=int(nfp),
        mmax=int(mmax),
        nmax=int(nmax),
        lasym=bool(lasym),
        dtype=np.asarray(state.Rcos).dtype,
    )

    class _WoutLike:
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
        )

        def __init__(self):
            self.phipf = np.asarray(flux.phipf)
            self.phips = np.asarray(flux.phips)
            self.chipf = np.asarray(chipf_wout)
            self.iotaf = np.asarray(iotaf)
            self.iotas = np.asarray(iotas)
            self.signgs = int(signgs)
            self.nfp = int(nfp)
            self.mpol = int(mpol)
            self.ntor = int(ntor)
            self.lasym = bool(lasym)
            self.flux_is_internal = True

    wout_like = _WoutLike()
    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        pres=pres,
        use_wout_bsup=False,
        use_wout_bsub_for_lambda=False,
        use_wout_bmag_for_bsq=False,
        use_vmec_synthesis=True,
        trig=trig,
    )

    # Derived 1D profiles and scalars.
    norms = vmec_force_norms_from_bcovar_dynamic(bc=bc, trig=trig, s=s, signgs=int(signgs))
    vp = np.asarray(norms.vp, dtype=float)
    wb = float(np.asarray(norms.wb))
    wp = float(np.asarray(norms.wp))
    volume = float(np.asarray(norms.volume))
    volume_p = volume * float(4.0 * np.pi**2)
    betatotal = (wp / wb) if wb != 0.0 else 0.0

    # buco/bvco/jcur* will be computed after aligning bsubu sign for wout output.
    buco = bvco = jcuru = jcurv = None
    equif = None

    # Nyquist Fourier coefficients for fields stored in wout.
    bsupu_out = np.asarray(bc.bsupu)
    bsupv_out = np.asarray(bc.bsupv)
    bsubu_out = np.asarray(bc.bsubu)
    bsubv_out = np.asarray(bc.bsubv)

    if bool(lasym):
        gmnc = z2.copy()
        gmns = z2.copy()
        bsupumnc = z2.copy()
        bsupumns = z2.copy()
        bsupvmnc = z2.copy()
        bsupvmns = z2.copy()
        bsubumnc = z2.copy()
        bsubumns = z2.copy()
        bsubvmnc = z2.copy()
        bsubvmns = z2.copy()
        bmnc = z2.copy()
        bmns = z2.copy()
    else:
        parity = "both" if bool(lasym) else "cos"
        gmnc, gmns = vmec_realspace_analysis(f=np.asarray(bc.jac.sqrtg), modes=nyq_modes, trig=trig, parity=parity)
        bsupu_out = np.asarray(bc.bsupu)
        bsupv_out = np.asarray(bc.bsupv)
        bsubu_out = np.asarray(bc.bsubu)
        bsubv_out = np.asarray(bc.bsubv)
        bsupumnc, bsupumns = vmec_realspace_analysis(f=bsupu_out, modes=nyq_modes, trig=trig, parity=parity)
        bsupvmnc, bsupvmns = vmec_realspace_analysis(f=bsupv_out, modes=nyq_modes, trig=trig, parity=parity)
        bsubumnc, bsubumns = vmec_realspace_analysis(f=bsubu_out, modes=nyq_modes, trig=trig, parity=parity)
        bsubvmnc, bsubvmns = vmec_realspace_analysis(f=bsubv_out, modes=nyq_modes, trig=trig, parity=parity)

        pres_h = np.asarray(pres, dtype=float)[:, None, None]
        bmag = np.sqrt(np.maximum(2.0 * (np.asarray(bc.bsq) - pres_h), 0.0))
        bmnc, bmns = vmec_realspace_analysis(f=bmag, modes=nyq_modes, trig=trig, parity=parity)

    # buco/bvco/jcur* and equif use the sign-aligned bsubu/bsubv for wout output.
    from types import SimpleNamespace

    bc_curr = SimpleNamespace(bsubu=bsubu_out, bsubv=bsubv_out)
    buco, bvco, jcuru, jcurv = currents_from_bcovar(bc=bc_curr, trig=trig, wout=wout_like, s=s)
    # Replace jcuru/jcurv with VMEC eqfor-style endpoint extrapolation.
    if ns >= 3:
        hs = float(s[1] - s[0])
        ohs = 1.0 / hs if hs != 0.0 else 0.0
        signgs_f = float(signgs)
        jcuru = np.zeros_like(buco)
        jcurv = np.zeros_like(buco)
        for js in range(1, ns - 1):
            jcurv[js] = signgs_f * ohs * (buco[js + 1] - buco[js])
            jcuru[js] = -signgs_f * ohs * (bvco[js + 1] - bvco[js])
        jcuru[0] = 2.0 * jcuru[1] - jcuru[2]
        jcuru[-1] = 2.0 * jcuru[-2] - jcuru[-3]
        jcurv[0] = 2.0 * jcurv[1] - jcurv[2]
        jcurv[-1] = 2.0 * jcurv[-2] - jcurv[-3]
        from .vmec_lforbal import MU0

        jcuru = jcuru / MU0
        jcurv = jcurv / MU0
    equif = equif_from_bcovar(bc=bc_curr, trig=trig, wout=wout_like, s=s)

    # Current profile metadata for vmecPlot2.
    pcurr_type = indata.get("PCURR_TYPE", None)
    if pcurr_type is None:
        pcurr_type = "power_series"
    pcurr_type = str(pcurr_type)

    piota_type = indata.get("PIOTA_TYPE", None)
    if piota_type is None:
        piota_type = "power_series"
    piota_type = str(piota_type)

    ac_raw = indata.get("AC", [])
    if isinstance(ac_raw, (int, float, np.floating)):
        ac = np.asarray([float(ac_raw)], dtype=float)
    elif isinstance(ac_raw, list):
        ac = np.asarray([float(v) for v in ac_raw], dtype=float)
    else:
        ac = np.asarray([], dtype=float)
    if ac.size == 0:
        ac = np.zeros((1,), dtype=float)

    ndfmax = 101
    ac_aux_s = -np.ones((ndfmax,), dtype=float)
    ac_aux_f = np.zeros((ndfmax,), dtype=float)
    DMerc = np.zeros((ns,), dtype=float)

    # Convert internal lambda coefficients to VMEC wout convention.
    from .field import lamscale_from_phips

    lamscale = float(np.asarray(lamscale_from_phips(flux.phips, s)))

    def _lambda_wout_from_full(*, lam_full: np.ndarray, m_modes: np.ndarray) -> np.ndarray:
        lam_full = np.asarray(lam_full, dtype=float)
        if lam_full.ndim != 2 or lam_full.shape[0] != ns:
            raise ValueError("Expected lam_full with shape (ns, K)")
        m_modes = np.asarray(m_modes, dtype=int)
        if m_modes.ndim != 1 or m_modes.shape[0] != lam_full.shape[1]:
            raise ValueError("Expected m_modes with shape (K,)")

        if lamscale == 0.0:
            lam_ext = np.zeros_like(lam_full)
        else:
            phipf_safe = np.where(phipf_internal == 0.0, 1.0, phipf_internal)
            lam_ext = lam_full * (lamscale / phipf_safe[:, None])

        if ns < 2:
            return np.zeros_like(lam_ext)

        hs = float(s[1] - s[0])
        sqrts_f = np.zeros((ns + 1,), dtype=float)
        shalf_f = np.zeros((ns + 1,), dtype=float)
        for i in range(1, ns + 1):
            sqrts_f[i] = np.sqrt(max(hs * float(i - 1), 0.0))
            shalf_f[i] = np.sqrt(hs * abs(float(i) - 1.5))
        sqrts_f[ns] = 1.0

        sm_f = np.zeros((ns + 1,), dtype=float)
        sp_f = np.zeros((ns + 1,), dtype=float)
        for i in range(2, ns + 1):
            sm_f[i] = shalf_f[i] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
            if i < ns:
                sp_f[i] = shalf_f[i + 1] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
            else:
                sp_f[i] = 1.0 / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
        sm_f[1] = 0.0
        sp_f[0] = 0.0
        sp_f[1] = sm_f[2] if ns >= 2 else 0.0

        lam_half = lam_ext.copy()
        mask_m_le1 = m_modes <= 1
        if np.any(mask_m_le1):
            lam_half[0, mask_m_le1] = lam_half[1, mask_m_le1]

        even_mask = (m_modes % 2) == 0
        odd_mask = ~even_mask
        for js_idx in range(ns - 1, 0, -1):
            if np.any(even_mask):
                lam_half[js_idx, even_mask] = 0.5 * (
                    lam_half[js_idx, even_mask] + lam_half[js_idx - 1, even_mask]
                )
            if np.any(odd_mask):
                sm_val = sm_f[js_idx + 1]
                sp_val = sp_f[js_idx]
                lam_half[js_idx, odd_mask] = 0.5 * (
                    sm_val * lam_half[js_idx, odd_mask] + sp_val * lam_half[js_idx - 1, odd_mask]
                )

        lam_half[0, :] = 0.0
        return lam_half

    lmns = _lambda_wout_from_full(lam_full=lmns_internal, m_modes=np.asarray(main_modes.m, dtype=int))
    lmnc = _lambda_wout_from_full(lam_full=lmnc_internal, m_modes=np.asarray(main_modes.m, dtype=int))

    return WoutData(
        path=Path(path),
        ns=ns,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        lasym=lasym,
        signgs=int(signgs),
        xm=np.asarray(main_modes.m, dtype=int),
        xn=np.asarray(main_modes.n * nfp, dtype=int),
        xm_nyq=np.asarray(nyq_modes.m, dtype=int),
        xn_nyq=np.asarray(nyq_modes.n * nfp, dtype=int),
        rmnc=rmnc,
        rmns=rmns,
        zmnc=zmnc,
        zmns=zmns,
        lmnc=lmnc,
        lmns=lmns,
        phipf=phipf_out,
        chipf=chipf_out,
        phips=np.asarray(flux.phips, dtype=float),
        iotaf=np.asarray(iotaf, dtype=float),
        iotas=np.asarray(iotas, dtype=float),
        gmnc=np.asarray(gmnc, dtype=float),
        gmns=np.asarray(gmns, dtype=float),
        bsupumnc=np.asarray(bsupumnc, dtype=float),
        bsupumns=np.asarray(bsupumns, dtype=float),
        bsupvmnc=np.asarray(bsupvmnc, dtype=float),
        bsupvmns=np.asarray(bsupvmns, dtype=float),
        bsubumnc=np.asarray(bsubumnc, dtype=float),
        bsubumns=np.asarray(bsubumns, dtype=float),
        bsubvmnc=np.asarray(bsubvmnc, dtype=float),
        bsubvmns=np.asarray(bsubvmns, dtype=float),
        bmnc=np.asarray(bmnc, dtype=float),
        bmns=np.asarray(bmns, dtype=float),
        wb=float(wb),
        volume_p=float(volume_p),
        gamma=float(getattr(indata, "get_float", lambda *_: 0.0)("GAMMA", 0.0)),
        wp=float(wp),
        vp=np.asarray(vp, dtype=float),
        pres=np.asarray(pres, dtype=float),
        presf=np.asarray(presf, dtype=float),
        fsqr=float(fsqr),
        fsqz=float(fsqz),
        fsql=float(fsql),
        fsqt=np.zeros((0,), dtype=float),
        equif=np.asarray(equif, dtype=float),
        phi=np.asarray(phi, dtype=float),
        buco=np.asarray(buco, dtype=float),
        bvco=np.asarray(bvco, dtype=float),
        jcuru=np.asarray(jcuru, dtype=float),
        jcurv=np.asarray(jcurv, dtype=float),
        raxis_cc=np.asarray(raxis_cc, dtype=float),
        zaxis_cs=np.asarray(zaxis_cs, dtype=float),
        raxis_cs=np.asarray(raxis_cs, dtype=float),
        zaxis_cc=np.asarray(zaxis_cc, dtype=float),
        Aminor_p=float(Aminor_p),
        Rmajor_p=float(Rmajor_p),
        aspect=float(aspect),
        betatotal=float(betatotal),
        betapol=0.0,
        betator=0.0,
        betaxis=0.0,
        ctor=0.0,
        DMerc=np.asarray(DMerc, dtype=float),
        ac=np.asarray(ac, dtype=float),
        ac_aux_s=np.asarray(ac_aux_s, dtype=float),
        ac_aux_f=np.asarray(ac_aux_f, dtype=float),
        pcurr_type=str(pcurr_type),
        piota_type=str(piota_type),
    )


def state_from_wout(wout: WoutData) -> VMECState:
    """Build a :class:`~vmec_jax.state.VMECState` from `wout` Fourier coefficients.

    Notes
    -----
    VMEC's ``wout`` files do **not** store the internal lambda coefficients in the
    same units VMEC uses in ``bcovar`` / ``totzsps``.

    In ``wrout.f`` VMEC writes (schematically, for each radial surface ``js``)::

        lmns_wout(:,js) = (lmns_internal(:,js) / phipf(js)) * lamscale

    to preserve an older output convention.

    For parity-style kernels that re-use VMEC's ``bcovar`` formulas, we therefore
    invert this scaling when constructing the state:

        lmns_internal = lmns_wout * phipf / lamscale
    """
    assert_main_modes_match_wout(wout=wout)
    layout = StateLayout(ns=wout.ns, K=int(wout.xm.size), lasym=bool(wout.lasym))

    # Reconstruct VMEC's internal lambda coefficients from the `wout` convention.
    # See `VMEC2000/Sources/Input_Output/wrout.f` (comment: "IF B^v ~ phip + lamu,
    # MUST DIVIDE BY phipf(js) below to maintain old-style format").
    from .field import lamscale_from_phips

    ns = int(wout.ns)
    if ns < 2:
        s = np.asarray([0.0], dtype=float)
    else:
        s = np.linspace(0.0, 1.0, ns, dtype=float)
    lamscale = float(np.asarray(lamscale_from_phips(wout.phips, s)))
    # VMEC's `wout` stores phipf scaled by 2π*signgs. Internally, lambda scaling
    # uses the unscaled phipf (= phipf_internal). Align the reconstruction with
    # bcovar's bsupv formula by undoing the 2π*signgs factor here.
    scale = float(2.0 * np.pi * float(getattr(wout, "signgs", 1)))
    phipf_internal = np.asarray(wout.phipf, dtype=float) / scale if scale != 0.0 else np.asarray(wout.phipf, dtype=float)
    if lamscale == 0.0:
        lam_scale = np.zeros((ns,), dtype=float)
    else:
        lam_scale = phipf_internal / lamscale  # (ns,)

    # VMEC writes lambda in a backward-compatible *half-mesh* convention (wrout.f),
    # which is not the internal full-mesh representation used by `totzsps`/`bcovar`.
    # We reproduce VMEC's own recovery logic from `load_xc_from_wout.f`:
    #   - undo the half-mesh interpolation (recurrence in `js`)
    #   - multiply by `phipf(js)` (undo old-style division)
    #   - divide by `lamscale` (undo old-style multiply)
    #
    # This yields lambda coefficients that are consistent with VMEC's internal
    # `bcovar` formulas when used with our `lamscale` scaling.
    def _lambda_full_from_wout(*, lam_wout: np.ndarray, m_modes: np.ndarray, phipf: np.ndarray, lamscale: float) -> np.ndarray:
        lam_wout = np.asarray(lam_wout, dtype=float)
        if lam_wout.ndim != 2 or lam_wout.shape[0] != ns:
            raise ValueError("Expected lam_wout with shape (ns, K)")
        m_modes = np.asarray(m_modes, dtype=int)
        if m_modes.ndim != 1 or m_modes.shape[0] != lam_wout.shape[1]:
            raise ValueError("Expected m_modes with shape (K,)")
        phipf = np.asarray(phipf, dtype=float)
        if phipf.shape != (ns,):
            raise ValueError("Expected phipf with shape (ns,)")
        if ns < 2:
            return lam_wout.copy()

        hs = float(s[1] - s[0])
        # Fortran-style 1-based arrays for sm/sp (profil1d.f).
        sqrts_f = np.zeros((ns + 1,), dtype=float)
        shalf_f = np.zeros((ns + 1,), dtype=float)
        for i in range(1, ns + 1):
            sqrts_f[i] = np.sqrt(max(hs * float(i - 1), 0.0))
            shalf_f[i] = np.sqrt(hs * abs(float(i) - 1.5))
        sqrts_f[ns] = 1.0  # avoid roundoff at boundary

        sm_f = np.zeros((ns + 1,), dtype=float)
        sp_f = np.zeros((ns + 1,), dtype=float)  # sp(0) exists in VMEC but is always 0
        for i in range(2, ns + 1):
            sm_f[i] = shalf_f[i] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
            if i < ns:
                sp_f[i] = shalf_f[i + 1] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
            else:
                sp_f[i] = 1.0 / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
        sm_f[1] = 0.0
        sp_f[0] = 0.0
        sp_f[1] = sm_f[2] if ns >= 2 else 0.0

        lam_full = np.zeros_like(lam_wout)
        # Axis initialization (load_xc_from_wout.f).
        m = m_modes
        is_m0 = m == 0
        is_m1 = m == 1
        lam_full[0, is_m0] = lam_wout[1, is_m0]
        denom_m1 = sm_f[2] + sp_f[1]
        if denom_m1 != 0.0:
            lam_full[0, is_m1] = 2.0 * lam_wout[1, is_m1] / denom_m1
        lam_full[0, ~(is_m0 | is_m1)] = 0.0

        # Undo the half-mesh interpolation.
        for mval in range(0, int(np.max(m_modes)) + 1):
            mask = m_modes == mval
            if not np.any(mask):
                continue
            if (mval % 2) == 0:
                for js in range(2, ns + 1):
                    lam_full[js - 1, mask] = 2.0 * lam_wout[js - 1, mask] - lam_full[js - 2, mask]
            else:
                for js in range(2, ns + 1):
                    denom = sm_f[js]
                    if denom == 0.0:
                        lam_full[js - 1, mask] = 0.0
                    else:
                        lam_full[js - 1, mask] = (2.0 * lam_wout[js - 1, mask] - sp_f[js - 1] * lam_full[js - 2, mask]) / denom

        # Undo the old-style `phipf` division and `lamscale` multiply done in `wrout.f`.
        #
        # Important: VMEC's `wrout.f` writes (schematically)
        #   lmns_wout(:,js) = (lmns1(:,js)/phipf(js)) * lamscale
        # for **all** radial indices `js=1..ns` before interpolating onto the
        # half mesh and finally setting `lmns(:,1)=0`. To recover internal full
        # mesh lambda coefficients consistent with the fields stored in `wout`
        # (notably `bsupu`), we must therefore multiply back by `phipf(js)` on
        # *every* surface, including the axis surface.
        lam_full = lam_full * phipf[:, None]
        if lamscale != 0.0:
            lam_full = lam_full / float(lamscale)
        return lam_full

    lmns_full = _lambda_full_from_wout(
        lam_wout=np.asarray(wout.lmns),
        m_modes=np.asarray(wout.xm),
        phipf=np.asarray(phipf_internal),
        lamscale=lamscale,
    )
    lmnc_full = _lambda_full_from_wout(
        lam_wout=np.asarray(wout.lmnc),
        m_modes=np.asarray(wout.xm),
        phipf=np.asarray(phipf_internal),
        lamscale=lamscale,
    )

    m_arr = np.asarray(wout.xm, dtype=int)
    n_arr = (np.asarray(wout.xn, dtype=int) // int(wout.nfp)).astype(int)
    sqrt2 = np.sqrt(2.0)
    mscale = np.where(m_arr == 0, 1.0, sqrt2)
    nscale = np.where(np.abs(n_arr) == 0, 1.0, sqrt2)
    mode_scale = (1.0 / (mscale * nscale))[None, :]

    Rcos = np.asarray(wout.rmnc) * mode_scale
    Rsin = np.asarray(wout.rmns) * mode_scale
    Zcos = np.asarray(wout.zmnc) * mode_scale
    Zsin = np.asarray(wout.zmns) * mode_scale

    modes = vmec_mode_table(wout.mpol, wout.ntor)
    lthreed = bool(int(wout.ntor) > 0)
    lasym = bool(wout.lasym)
    lconm1 = bool(lthreed or lasym)
    Rcos, Zsin, Rsin, Zcos = vmec_m1_physical_to_internal_signed(
        Rcos=Rcos,
        Zsin=Zsin,
        Rsin=Rsin,
        Zcos=Zcos,
        modes=modes,
        lthreed=lthreed,
        lasym=lasym,
        lconm1=lconm1,
    )

    return VMECState(
        layout=layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=np.asarray(lmnc_full) * mode_scale,
        Lsin=np.asarray(lmns_full) * mode_scale,
    )
