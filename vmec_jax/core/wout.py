"""Complete VMEC2000-compatible ``wout_*.nc`` schema, writer and reader.

This module implements the full variable set written by VMEC2000's
``wrout.f`` (plan.md Appendix A), with the exact netCDF names, dimensions,
dtypes and unit conventions of the reference implementation:

- ``presf``/``pres``/``mass`` are stored in Pa (``wrout.f`` divides the
  internal ``mu0*Pa`` values by ``mu0`` on write);
- ``jcuru``/``jcurv``/``ctor``/``currumnc``/``currvmnc`` are in A (again
  ``1/mu0`` applied on write);
- ``phipf``/``chipf`` carry the ``twopi*signgs`` factor relative to the
  internal ``phips``/``chips`` arrays;
- ``q_factor = 1/iotaf`` (``HUGE`` at iota zeros);
- ``lmns`` is on the half mesh (interpolated from VMEC's internal full-mesh
  lambda), ``bsubsmns`` on the full mesh (converted in ``jxbforce.f``);
- ``lasym`` partner tables (``rmns``, ``zmnc``, ...) exist **only** for
  asymmetric runs; the free-boundary potential/surface tables (``potsin``,
  ``xmpot``, ``xnpot``, ``curlabel``, ``*_sur``) only when ``lfreeb``.

:class:`WoutData` stores every field in **file convention** (exactly the
values found in the netCDF file), so ``read_wout(write_wout(x)) == x``.

:func:`wout_from_state` builds the complete dataset from a converged
fixed-boundary spectral state.  The parity-proven Nyquist/jxbforce/Mercier
post-processing engine from :mod:`vmec_jax.io.wout_files` (ports of
``wrout.f``/``jxbforce.f``/``mercier.f``/``bss.f``) supplies the core
tables; the remaining VMEC2000 output quantities (``eqfor.f``/
``spectrum.f``/``Compute_Currents`` et al.) come from
:mod:`vmec_jax.core.postprocess`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as _dc_fields
from pathlib import Path

import numpy as np

from . import postprocess as _pp

MU0 = _pp.MU0

# netCDF dimension names used by wrout.f (ezcdf conventions).
_DIM_RADIUS = "radius"
_DIM_MN = "mn_mode"
_DIM_MN_NYQ = "mn_mode_nyq"
_DIM_TIME = "time"
_NDFMAX = 101   # spline aux-array length (vmec_input.f ndatafmax)
_PRESET = 21    # am/ac/ai polynomial storage (0:20)
_NSTORE = 100   # fsqt/wdot history length (nstore_seq)

# (netcdf_name, long_name, units) attribute table copied from VMEC2000 output.
_ATTRS = {
    "mnyq": ("Poloidal modes (Nyquist)", None),
    "nnyq": ("Toroidal modes (Nyquist)", None),
    "xm": ("Poloidal mode numbers", None),
    "xn": ("Toroidal mode numbers", None),
    "xm_nyq": ("Poloidal mode numbers (Nyquist)", None),
    "xn_nyq": ("Toroidal mode numbers (Nyquist)", None),
    "raxis_cc": ("raxis (cosnv)", None),
    "zaxis_cs": ("zaxis (sinnv)", None),
    "raxis_cs": ("raxis (sinnv)", None),
    "zaxis_cc": ("zaxis (cosnv)", None),
    "iotaf": ("Rotational Transform (iota) on full mesh", None),
    "q_factor": ("Safety-factor (q) on full mesh", None),
    "presf": ("Pressure on full mesh [Pa]", "Pa"),
    "phi": ("Toroidal flux on full mesh [Wb]", "wb"),
    "phipf": ("d(phi)/ds: Toroidal flux deriv on full mesh", None),
    "chi": ("Poloidal flux on full mesh [Wb]", "wb"),
    "chipf": ("d(chi)/ds: Poroidal flux deriv on full mesh", None),
    "iotas": ("Rotational transform (iota) on half mesh", None),
    "mass": ("Mass on half mesh", None),
    "pres": ("Pressure half mesh [Pa]", "Pa"),
    "rmnc": ("cosmn component of cylindrical R, full mesh", "m"),
    "zmns": ("sinmn component of cylindrical Z, full mesh", "m"),
    "lmns": ("sinmn component of lambda, half mesh", None),
    "gmnc": ("cosmn component of jacobian, half mesh", None),
    "bmnc": ("cosmn component of mod-B, half mesh", None),
    "bsubumnc": ("cosmn covariant u-component of B, half mesh", None),
    "bsubvmnc": ("cosmn covariant v-component of B, half mesh", None),
    "bsubsmns": ("sinmn covariant s-component of B, half mesh", None),
    "currumnc": ("cosmn covariant u-component of J, full mesh", None),
    "currvmnc": ("cosmn covariant v-component of J, full mesh", None),
    "rmns": ("sinmn component of cylindrical R, full mesh", "m"),
    "zmnc": ("cosmn component of cylindrical Z, full mesh", "m"),
    "lmnc": ("cosmn component of lambda, half mesh", None),
    "gmns": ("sinmn component of jacobian, half mesh", None),
    "bmns": ("sinmn component of mod-B, half mesh", None),
    "bsubumns": ("sinmn covariant u-component of B, half mesh", None),
    "bsubvmns": ("sinmn covariant v-component of B, half mesh", None),
    "bsubsmnc": ("cosmn covariant s-component of B, half mesh", None),
    "currumns": ("sinmn covariant u-component of J, full mesh", None),
    "currvmns": ("sinmn covariant v-component of J, full mesh", None),
    "potsin": ("Vacuum potential sin modes", None),
    "potcos": ("Vacuum potential cos modes", None),
    "bsubumnc_sur": ("cosmn covaiant u-component of B, surface", None),
    "bsubvmnc_sur": ("cosmn covaiant v-component of B, surface", None),
    "bsupumnc_sur": ("cosmn contravariant u-component of B, surface", None),
    "bsupvmnc_sur": ("cosmn contravariant v-component of B, surface", None),
    "bsubumns_sur": ("sinmn covaiant u-component of B, surface", None),
    "bsubvmns_sur": ("sinmn covaiant v-component of B, surface", None),
    "bsupumns_sur": ("sinmn contravariant u-component of B, surface", None),
    "bsupvmns_sur": ("sinmn contravariant v-component of B, surface", None),
}

_LOGICALS = ("lasym", "lrecon", "lfreeb", "lmove_axis", "lrfp")
_INT_SCALARS = ("nfp", "ns", "mpol", "ntor", "mnmax", "mnyq", "nnyq",
                "mnmax_nyq", "niter", "itfsq", "ier_flag", "signgs", "nextcur")
_FLOAT_SCALARS = ("version_", "wb", "wp", "gamma", "rmax_surf", "rmin_surf",
                  "zmax_surf", "aspect", "betatotal", "betapol", "betator",
                  "betaxis", "b0", "rbtor0", "rbtor", "IonLarmor", "volavgB",
                  "ctor", "Aminor_p", "Rmajor_p", "volume_p", "ftolv",
                  "fsql", "fsqr", "fsqz")
_STRINGS = (("input_extension", "dim_00100", 100), ("mgrid_file", "dim_00200", 200),
            ("pcurr_type", "dim_00020", 20), ("pmass_type", "dim_00020", 20),
            ("piota_type", "dim_00020", 20))
_RADIUS_1D = ("iotaf", "q_factor", "presf", "phi", "phipf", "chi", "chipf",
              "jcuru", "jcurv", "iotas", "mass", "pres", "beta_vol", "buco",
              "bvco", "vp", "specw", "phips", "over_r", "jdotb", "bdotb",
              "bdotgradv", "DMerc", "DShear", "DWell", "DCurr", "DGeod", "equif")
_PROFILE_1D = (("am", "preset"), ("ac", "preset"), ("ai", "preset"),
               ("am_aux_s", "ndfmax"), ("am_aux_f", "ndfmax"),
               ("ai_aux_s", "ndfmax"), ("ai_aux_f", "ndfmax"),
               ("ac_aux_s", "ndfmax"), ("ac_aux_f", "ndfmax"))
_MN2D_SYM = ("rmnc", "zmns", "lmns")
_NYQ2D_SYM = ("gmnc", "bmnc", "bsubumnc", "bsubvmnc", "bsubsmns",
              "currumnc", "currvmnc", "bsupumnc", "bsupvmnc")
_MN2D_ASYM = ("rmns", "zmnc", "lmnc")
_NYQ2D_ASYM = ("gmns", "bmns", "bsubumns", "bsubvmns", "bsubsmnc",
               "currumns", "currvmns", "bsupumns", "bsupvmns")
_SUR_SYM = ("bsubumnc_sur", "bsubvmnc_sur", "bsupumnc_sur", "bsupvmnc_sur")
_SUR_ASYM = ("bsubumns_sur", "bsubvmns_sur", "bsupumns_sur", "bsupvmns_sur")


@dataclass(frozen=True)
class WoutData:
    """Full VMEC2000 ``wout`` dataset in **file conventions** (wrout.f).

    Every field name matches its netCDF variable name (logicals drop the
    ``__logical__`` suffix).  Radial arrays are ``(ns,)``; Fourier tables
    ``(ns, mnmax)`` / ``(ns, mnmax_nyq)``.  Optional groups are ``None``
    when not applicable: lasym partners for symmetric runs, free-boundary
    tables for fixed-boundary runs.
    """

    # -- scalars ---------------------------------------------------------
    version_: float
    input_extension: str
    mgrid_file: str
    pcurr_type: str
    pmass_type: str
    piota_type: str
    wb: float
    wp: float
    gamma: float
    rmax_surf: float
    rmin_surf: float
    zmax_surf: float
    nfp: int
    ns: int
    mpol: int
    ntor: int
    mnmax: int
    mnyq: int
    nnyq: int
    mnmax_nyq: int
    niter: int
    itfsq: int
    lasym: bool
    lrecon: bool
    lfreeb: bool
    lmove_axis: bool
    lrfp: bool
    ier_flag: int
    aspect: float
    betatotal: float
    betapol: float
    betator: float
    betaxis: float
    b0: float
    rbtor0: float
    rbtor: float
    signgs: int
    IonLarmor: float
    volavgB: float
    ctor: float                      # toroidal current [A] (internal / mu0)
    Aminor_p: float
    Rmajor_p: float
    volume_p: float
    ftolv: float
    fsql: float
    fsqr: float
    fsqz: float
    nextcur: int
    extcur: np.ndarray               # (max(nextcur,1),) coil currents [A]
    mgrid_mode: str
    # -- mode tables / axis ---------------------------------------------
    xm: np.ndarray
    xn: np.ndarray
    xm_nyq: np.ndarray
    xn_nyq: np.ndarray
    raxis_cc: np.ndarray             # (ntor+1,)
    zaxis_cs: np.ndarray
    # -- profile presets --------------------------------------------------
    am: np.ndarray                   # (21,) pressure polynomial
    ac: np.ndarray                   # (21,) current polynomial
    ai: np.ndarray                   # (21,) iota polynomial
    am_aux_s: np.ndarray             # (101,) spline knots (-1 fill)
    am_aux_f: np.ndarray             # (101,) spline values (0 fill)
    ai_aux_s: np.ndarray
    ai_aux_f: np.ndarray
    ac_aux_s: np.ndarray
    ac_aux_f: np.ndarray
    # -- radial profiles (full mesh unless noted) -------------------------
    iotaf: np.ndarray
    q_factor: np.ndarray             # 1/iotaf
    presf: np.ndarray                # [Pa]
    phi: np.ndarray                  # toroidal flux [Wb]
    phipf: np.ndarray                # 2*pi*signgs*d(phi_int)/ds
    chi: np.ndarray                  # poloidal flux [Wb]
    chipf: np.ndarray                # 2*pi*signgs*chips
    jcuru: np.ndarray                # [A] surface-averaged current density
    jcurv: np.ndarray                # [A]
    iotas: np.ndarray                # half mesh
    mass: np.ndarray                 # half mesh [Pa]
    pres: np.ndarray                 # half mesh [Pa]
    beta_vol: np.ndarray             # half mesh
    buco: np.ndarray                 # half mesh <B_u> (internal units)
    bvco: np.ndarray                 # half mesh <B_v>
    vp: np.ndarray                   # half mesh dV/ds/(2*pi)^2
    specw: np.ndarray                # spectral width <M>
    phips: np.ndarray                # half mesh internal phip
    over_r: np.ndarray               # half mesh <1/R>
    jdotb: np.ndarray                # [A*T] (1/mu0 in jxbforce)
    bdotb: np.ndarray
    bdotgradv: np.ndarray
    DMerc: np.ndarray
    DShear: np.ndarray
    DWell: np.ndarray
    DCurr: np.ndarray
    DGeod: np.ndarray
    equif: np.ndarray
    fsqt: np.ndarray                 # (100,) residual history
    wdot: np.ndarray                 # (100,) energy decay history
    # -- Fourier tables ----------------------------------------------------
    rmnc: np.ndarray                 # (ns, mnmax) full mesh [m]
    zmns: np.ndarray                 # full mesh [m]
    lmns: np.ndarray                 # HALF mesh
    gmnc: np.ndarray                 # (ns, mnmax_nyq) half mesh
    bmnc: np.ndarray                 # half mesh
    bsubumnc: np.ndarray             # half mesh
    bsubvmnc: np.ndarray             # half mesh
    bsubsmns: np.ndarray             # FULL mesh (converted in jxbforce.f)
    currumnc: np.ndarray             # full mesh [A]
    currvmnc: np.ndarray             # full mesh [A]
    bsupumnc: np.ndarray             # half mesh
    bsupvmnc: np.ndarray             # half mesh
    # -- lasym partners (None for symmetric runs) --------------------------
    raxis_cs: np.ndarray | None = None
    zaxis_cc: np.ndarray | None = None
    rmns: np.ndarray | None = None
    zmnc: np.ndarray | None = None
    lmnc: np.ndarray | None = None
    gmns: np.ndarray | None = None
    bmns: np.ndarray | None = None
    bsubumns: np.ndarray | None = None
    bsubvmns: np.ndarray | None = None
    bsubsmnc: np.ndarray | None = None
    currumns: np.ndarray | None = None
    currvmns: np.ndarray | None = None
    bsupumns: np.ndarray | None = None
    bsupvmns: np.ndarray | None = None
    # -- free boundary (None for fixed-boundary runs) ----------------------
    mnmaxpot: int | None = None
    nobser: int | None = None
    nobd: int | None = None
    nbsets: int | None = None
    nbfld: np.ndarray | None = None
    potsin: np.ndarray | None = None
    potcos: np.ndarray | None = None  # only when also lasym
    xmpot: np.ndarray | None = None
    xnpot: np.ndarray | None = None
    curlabel: tuple[str, ...] | None = None
    bsubumnc_sur: np.ndarray | None = None
    bsubvmnc_sur: np.ndarray | None = None
    bsupumnc_sur: np.ndarray | None = None
    bsupvmnc_sur: np.ndarray | None = None
    bsubumns_sur: np.ndarray | None = None
    bsubvmns_sur: np.ndarray | None = None
    bsupumns_sur: np.ndarray | None = None
    bsupvmns_sur: np.ndarray | None = None


# ==========================================================================
# netCDF writer / reader
# ==========================================================================

def _put_string(ds, name: str, dim: str, width: int, value: str | None) -> None:
    var = ds.createVariable(name, "S1", (dim,))
    _ATTRS.get(name)  # strings carry no attributes in VMEC output
    if value is None:
        return  # leave at fill (VMEC skips mgrid_mode when nextcur == 0)
    text = (str(value)[:width]).ljust(width)
    var[:] = np.frombuffer(text.encode("ascii", "replace"), dtype="S1")


def _put(ds, name: str, dims: tuple[str, ...], data, dtype: str = "f8") -> None:
    var = ds.createVariable(name, dtype, dims)
    ln_units = _ATTRS.get(name)
    if ln_units is not None:
        ln, units = ln_units
        if ln:
            var.long_name = ln
        if units:
            var.units = units
    if data is not None:
        if dims:
            arr = np.asarray(data, dtype=np.float64 if dtype == "f8" else np.int32)
            var[:] = np.reshape(arr, var.shape)
        else:
            var.assignValue(np.float64(data) if dtype == "f8" else np.int32(data))


def write_wout(path: str | Path, data: WoutData, *, overwrite: bool = True) -> Path:
    """Write ``data`` to ``path`` in VMEC2000 ``wout_*.nc`` layout.

    Uses NETCDF3 64-bit-offset format (as VMEC2000's ezcdf) and reproduces
    wrout.f's variable set, ordering, dimensions and attributes.  ``extcur``
    and ``mgrid_mode`` are created but left unwritten (netCDF fill) when
    ``nextcur == 0``, matching VMEC2000.
    """
    import netCDF4

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True)")
    d = data
    lasym, lfreeb = bool(d.lasym), bool(d.lfreeb)
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        for name, size in (
            ("dim_00100", 100), ("dim_00200", 200), ("dim_00020", 20),
            *((("ext_current", int(d.nextcur)),) if int(d.nextcur) > 0 else ()),
            ("dim_00001", 1),
            (_DIM_MN, int(np.asarray(d.xm).size)),
            (_DIM_MN_NYQ, int(np.asarray(d.xm_nyq).size)),
            ("n_tor", int(d.ntor) + 1), ("preset", _PRESET),
            ("ndfmax", _NDFMAX), (_DIM_RADIUS, int(d.ns)),
            (_DIM_TIME, int(np.asarray(d.fsqt).size)),
        ):
            ds.createDimension(name, size)
        if lfreeb:
            mnpot = int(np.asarray(d.xmpot).size) if d.xmpot is not None else max(int(d.mnmaxpot or 1), 1)
            ds.createDimension("mn_mode_pot", max(mnpot, 1))
            ds.createDimension("current_label", max(int(d.nextcur), 1))

        _put(ds, "version_", (), float(d.version_))
        for name, dim, width in _STRINGS:
            _put_string(ds, name, dim, width, getattr(d, name))
        for name in ("wb", "wp", "gamma", "rmax_surf", "rmin_surf", "zmax_surf"):
            _put(ds, name, (), float(getattr(d, name)))
        for name in ("nfp", "ns", "mpol", "ntor", "mnmax", "mnyq", "nnyq",
                     "mnmax_nyq", "niter", "itfsq"):
            _put(ds, name, (), int(getattr(d, name)), "i4")
        for name in _LOGICALS:
            _put(ds, f"{name}__logical__", (), int(bool(getattr(d, name))), "i4")
        _put(ds, "ier_flag", (), int(d.ier_flag), "i4")
        for name in ("aspect", "betatotal", "betapol", "betator", "betaxis",
                     "b0", "rbtor0", "rbtor"):
            _put(ds, name, (), float(getattr(d, name)))
        _put(ds, "signgs", (), int(d.signgs), "i4")
        for name in ("IonLarmor", "volavgB", "ctor", "Aminor_p", "Rmajor_p",
                     "volume_p", "ftolv", "fsql", "fsqr", "fsqz"):
            _put(ds, name, (), float(getattr(d, name)))
        _put(ds, "nextcur", (), int(d.nextcur), "i4")
        # VMEC leaves extcur/mgrid_mode unwritten when nextcur == 0; ezcdf
        # then stores a dimensionless scalar placeholder (netCDF fill).
        if int(d.nextcur) > 0:
            _put(ds, "extcur", ("ext_current",), np.asarray(d.extcur, dtype=float))
        else:
            _put(ds, "extcur", (), None)
        _put_string(ds, "mgrid_mode", "dim_00001", 1,
                    d.mgrid_mode if int(d.nextcur) > 0 else None)
        if lfreeb:
            for name in ("mnmaxpot", "nobser", "nobd", "nbsets"):
                _put(ds, name, (), int(getattr(d, name) or 0), "i4")
            if d.nbsets and d.nbfld is not None and int(d.nbsets) > 0:
                ds.createDimension("nbsets_dim", int(d.nbsets))
                _put(ds, "nbfld", ("nbsets_dim",), np.asarray(d.nbfld, float))

        for name in ("xm", "xn"):
            _put(ds, name, (_DIM_MN,), np.asarray(getattr(d, name), float))
        for name in ("xm_nyq", "xn_nyq"):
            _put(ds, name, (_DIM_MN_NYQ,), np.asarray(getattr(d, name), float))
        _put(ds, "raxis_cc", ("n_tor",), d.raxis_cc)
        _put(ds, "zaxis_cs", ("n_tor",), d.zaxis_cs)
        if lasym:
            _put(ds, "raxis_cs", ("n_tor",), d.raxis_cs)
            _put(ds, "zaxis_cc", ("n_tor",), d.zaxis_cc)
        for name, dim in _PROFILE_1D:
            _put(ds, name, (dim,), np.asarray(getattr(d, name), float))
        for name in _RADIUS_1D:
            _put(ds, name, (_DIM_RADIUS,), np.asarray(getattr(d, name), float))
        _put(ds, "fsqt", (_DIM_TIME,), np.asarray(d.fsqt, float))
        _put(ds, "wdot", (_DIM_TIME,), np.asarray(d.wdot, float))
        if lfreeb:
            _put(ds, "potsin", ("mn_mode_pot",), d.potsin)
            _put(ds, "xmpot", ("mn_mode_pot",), d.xmpot)
            _put(ds, "xnpot", ("mn_mode_pot",), d.xnpot)
            if lasym:
                _put(ds, "potcos", ("mn_mode_pot",), d.potcos)
            if d.curlabel is not None and int(d.nextcur) > 0:
                var = ds.createVariable("curlabel", "S1", ("current_label", "dim_00020"))
                arr = np.full((len(d.curlabel), 20), b" ", dtype="S1")
                for i, lbl in enumerate(d.curlabel):
                    for j, ch in enumerate(str(lbl)[:20].ljust(20)):
                        arr[i, j] = ch.encode("ascii", "replace")
                var[:] = arr
        for name in _MN2D_SYM:
            _put(ds, name, (_DIM_RADIUS, _DIM_MN), getattr(d, name))
        for name in _NYQ2D_SYM:
            _put(ds, name, (_DIM_RADIUS, _DIM_MN_NYQ), getattr(d, name))
        if lfreeb:
            for name in _SUR_SYM:
                _put(ds, name, (_DIM_MN_NYQ,), getattr(d, name))
        if lasym:
            for name in _MN2D_ASYM:
                _put(ds, name, (_DIM_RADIUS, _DIM_MN), getattr(d, name))
            for name in _NYQ2D_ASYM:
                _put(ds, name, (_DIM_RADIUS, _DIM_MN_NYQ), getattr(d, name))
            if lfreeb:
                for name in _SUR_ASYM:
                    _put(ds, name, (_DIM_MN_NYQ,), getattr(d, name))
    return path


def _read_string(ds, name: str) -> str:
    if name not in ds.variables:
        return ""
    raw = np.ma.filled(ds.variables[name][:], b"\x00")
    return raw.tobytes().decode("ascii", "ignore").rstrip(" \x00")


def _rd(ds, name: str, default=None):
    if name not in ds.variables:
        return default
    return np.asarray(np.ma.filled(ds.variables[name][:], 0.0))


def read_wout(path: str | Path) -> WoutData:
    """Read a VMEC2000-compatible ``wout_*.nc`` file into :class:`WoutData`.

    All values are kept in file conventions (no unit conversions), so a
    :func:`write_wout` / :func:`read_wout` round trip is the identity.
    """
    import netCDF4

    kw: dict = {}
    with netCDF4.Dataset(Path(path)) as ds:
        for name in _FLOAT_SCALARS:
            v = _rd(ds, name, 0.0)
            kw[name] = float(np.ravel(v)[0]) if v is not None and np.asarray(v).size else 0.0
        for name in _INT_SCALARS:
            v = _rd(ds, name, 0)
            kw[name] = int(np.ravel(v)[0]) if v is not None and np.asarray(v).size else 0
        for name in _LOGICALS:
            v = _rd(ds, f"{name}__logical__", 0)
            kw[name] = bool(int(np.ravel(v)[0])) if v is not None else False
        for name, _, _ in _STRINGS:
            kw[name] = _read_string(ds, name)
        kw["mgrid_mode"] = _read_string(ds, "mgrid_mode")
        kw["extcur"] = np.atleast_1d(_rd(ds, "extcur", np.zeros(1)))
        for name in ("xm", "xn", "xm_nyq", "xn_nyq", "raxis_cc", "zaxis_cs",
                     "fsqt", "wdot"):
            kw[name] = _rd(ds, name)
        for name, _dim in _PROFILE_1D:
            kw[name] = _rd(ds, name)
        for name in _RADIUS_1D:
            kw[name] = _rd(ds, name)
        for name in _MN2D_SYM + _NYQ2D_SYM:
            kw[name] = _rd(ds, name)
        lasym, lfreeb = kw["lasym"], kw["lfreeb"]
        if lasym:
            for name in ("raxis_cs", "zaxis_cc") + _MN2D_ASYM + _NYQ2D_ASYM:
                kw[name] = _rd(ds, name)
        if lfreeb:
            for name in ("mnmaxpot", "nobser", "nobd", "nbsets"):
                v = _rd(ds, name, 0)
                kw[name] = int(np.ravel(v)[0])
            kw["nbfld"] = _rd(ds, "nbfld")
            for name in ("potsin", "xmpot", "xnpot") + _SUR_SYM:
                kw[name] = _rd(ds, name)
            if lasym:
                kw["potcos"] = _rd(ds, "potcos")
                for name in _SUR_ASYM:
                    kw[name] = _rd(ds, name)
            if "curlabel" in ds.variables:
                raw = np.ma.filled(ds.variables["curlabel"][:], b" ")
                kw["curlabel"] = tuple(
                    row.tobytes().decode("ascii", "ignore").rstrip(" \x00")
                    for row in np.atleast_2d(raw)
                )
    return WoutData(**kw)


# ==========================================================================
# wout construction from a solved fixed-boundary state
# ==========================================================================

def _preset_array(indata, key: str, size: int = _PRESET, fill: float = 0.0) -> np.ndarray:
    """Fixed-width profile array from a namelist entry (wrout.f preset dims)."""
    raw = indata.get(key, None) if hasattr(indata, "get") else None
    out = np.full((size,), float(fill), dtype=float)
    if raw is None:
        return out
    vals = np.atleast_1d(np.asarray(raw, dtype=float)).ravel()
    n = min(int(vals.size), size)
    out[:n] = vals[:n]
    return out


def _indata_str(indata, key: str, default: str) -> str:
    """Namelist string with surrounding quotes/blanks stripped."""
    raw = indata.get(key, None) if hasattr(indata, "get") else None
    if raw is None:
        return default
    return str(raw).strip().strip("'\"").strip() or default


def _ftolv_from_indata(indata, *, ns: int) -> float:
    """``ftol_array(MAXLOC(ns_array))`` exactly as wrout.f evaluates ftolv."""
    ns_arr = np.atleast_1d(np.asarray(indata.get("NS_ARRAY", [ns]), dtype=float))
    ftol_arr = np.atleast_1d(np.asarray(
        indata.get("FTOL_ARRAY", [indata.get_float("FTOL", 1.0e-10)]), dtype=float))
    idx = int(np.argmax(ns_arr))
    return float(ftol_arr[min(idx, ftol_arr.size - 1)])


def wout_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    fsqt=None,
    wdot=None,
    niter: int = 0,
    itfsq: int = 0,
    converged: bool = True,
    input_extension: str = "",
    version: float = 9.0,
    path: str | Path = "wout_vmec_jax.nc",
    flux_override=None,
    profiles_override=None,
    force_payload_override=None,
    fast_bcovar: bool = False,
) -> WoutData:
    """Build a complete :class:`WoutData` from a solved fixed-boundary state.

    ``state``/``static``/``indata``/``signgs`` are the solved-run objects
    (as returned by the driver).  The parity-proven wout engine in
    :mod:`vmec_jax.io.wout_files` supplies the geometry/Nyquist/Mercier
    tables; this function adds every remaining VMEC2000 ``wrout.f``
    variable via :mod:`vmec_jax.core.postprocess` and the input deck.

    Unlike VMEC2000 (which zeroes the late ``eqfor.f`` scalars when the run
    hits NITER), all derived quantities are always computed - vmec_jax's
    zero-crash policy keeps diagnostic output for non-converged states and
    records convergence in ``ier_flag`` (0 = converged, 2 = more iterations
    needed, matching vmec_params.f).
    """
    import os

    from ..wout import wout_minimal_from_fixed_boundary

    # The legacy engine reads its bcovar lane from the environment; force the
    # parity-faithful (slow) lane so bsubsmns/jxbforce outputs match wrout.f.
    prev_fast = os.environ.get("VMEC_JAX_WOUT_FAST_BCOVAR")
    os.environ["VMEC_JAX_WOUT_FAST_BCOVAR"] = "1" if fast_bcovar else "0"
    try:
        legacy = wout_minimal_from_fixed_boundary(
            path=path,
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
            fsqr=float(fsqr),
            fsqz=float(fsqz),
            fsql=float(fsql),
            fsqt=fsqt,
            converged=bool(converged),
            flux_override=flux_override,
            profiles_override=profiles_override,
            force_payload_override=force_payload_override,
        )
    finally:
        if prev_fast is None:
            os.environ.pop("VMEC_JAX_WOUT_FAST_BCOVAR", None)
        else:
            os.environ["VMEC_JAX_WOUT_FAST_BCOVAR"] = prev_fast

    ns = int(legacy.ns)
    nfp = int(legacy.nfp)
    lasym = bool(legacy.lasym)
    cfg = static.cfg
    ntheta = int(getattr(cfg, "ntheta", 0)) or (2 * int(legacy.mpol) + 6)
    nzeta = int(getattr(cfg, "nzeta", 0)) or 1
    gamma = float(indata.get_float("GAMMA", 0.0))

    pres_pa = np.asarray(legacy.pres, dtype=float) / MU0
    presf_pa = np.asarray(legacy.presf, dtype=float) / MU0
    pres_pa[0] = 0.0
    vp = np.asarray(legacy.vp, dtype=float).copy()
    vp[0] = 0.0
    iotas = np.asarray(legacy.iotas, dtype=float).copy()
    iotas[0] = 0.0
    phips = np.asarray(legacy.phips, dtype=float).copy()
    phips[0] = 0.0
    phipf_out = np.asarray(legacy.phipf, dtype=float)
    chipf_out = np.asarray(legacy.chipf, dtype=float)

    a = {"rmns": None, "zmnc": None}
    if lasym:
        a = {k: np.asarray(getattr(legacy, k), dtype=float) for k in a}

    # -- official VMEC2000 Nyquist mode set (grid Nyquist, fixaray.f) ------
    # VMEC2000 sizes the Nyquist table from the *grid* (mnyq = ntheta1/2,
    # nnyq = nzeta/2 with the deck's NZETA), even when ntor = 0. The solver
    # may have run with a reduced toroidal grid; expand the tables (the
    # extra toroidal harmonics vanish identically for such runs).
    xm_nyq = np.asarray(legacy.xm_nyq, dtype=float)
    xn_nyq = np.asarray(legacy.xn_nyq, dtype=float)
    nyq = {name: np.asarray(getattr(legacy, name), dtype=float)
           for name in ("gmnc", "bmnc", "bsubumnc", "bsubvmnc", "bsubsmns",
                        "bsupumnc", "bsupvmnc")}
    nyq_a = {name: None for name in ("gmns", "bmns", "bsubumns", "bsubvmns",
                                     "bsubsmnc", "bsupumns", "bsupvmns")}
    if lasym:
        nyq_a = {k: np.asarray(getattr(legacy, k), dtype=float) for k in nyq_a}
    nzeta_vmec = int(indata.get_int("NZETA", 0)) or (1 if int(legacy.ntor) == 0 else nzeta)
    nnyq_target = int(nzeta_vmec) // 2
    nnyq_legacy = int(np.max(xn_nyq)) // nfp if xn_nyq.size else 0
    if nnyq_target != nnyq_legacy:
        mnyq = int(np.max(xm_nyq)) if xm_nyq.size else 0
        xm_new, xn_new = _pp.nyquist_mode_table(mnyq=mnyq, nnyq=nnyq_target, nfp=nfp)
        for name, tab in nyq.items():
            nyq[name] = _pp.expand_mode_columns(tab, xm_nyq, xn_nyq, xm_new, xn_new)
        if lasym:
            for name, tab in nyq_a.items():
                nyq_a[name] = _pp.expand_mode_columns(tab, xm_nyq, xn_nyq, xm_new, xn_new)
        xm_nyq, xn_nyq = xm_new, xn_new

    # -- fbal.f/bcovar.f: current averages recomputed from the proven
    #    bsub[uv]mnc tables (also fixes the legacy lasym wint normalization)
    buco, bvco, jcuru, jcurv, equif, ctor = _pp.force_balance(
        bsubumnc=nyq["bsubumnc"], bsubvmnc=nyq["bsubvmnc"],
        xm_nyq=xm_nyq, xn_nyq=xn_nyq,
        phipf=phipf_out, chipf=chipf_out, pres=pres_pa, vp=vp,
        signgs=int(signgs))

    # jxbforce.f jdotb: the legacy lasym lane misses VMEC2000's 2013 output
    # integration-norm change (four factor-2 normalizations: wint doubling
    # plus the (u,v) current pair on both J and B legs); measured exactly 16
    # against golden VMEC2000 lasym output. Symmetric runs are unaffected.
    jdotb = np.asarray(legacy.jdotb, dtype=float)
    if lasym:
        jdotb = 16.0 * jdotb

    # -- eqfor.f / spectrum.f / Compute_Currents ports -------------------
    currumnc, currvmnc, currumns, currvmns = _pp.compute_currents(
        bsubsmns=nyq["bsubsmns"], bsubumnc=nyq["bsubumnc"],
        bsubvmnc=nyq["bsubvmnc"], xm_nyq=xm_nyq, xn_nyq=xn_nyq,
        bsubsmnc=nyq_a["bsubsmnc"], bsubumns=nyq_a["bsubumns"],
        bsubvmns=nyq_a["bsubvmns"], lasym=lasym)
    specw = _pp.spectral_width(rmnc=legacy.rmnc, zmns=legacy.zmns,
                               xm=legacy.xm, xn=legacy.xn,
                               rmns=a["rmns"], zmnc=a["zmnc"])
    chi = _pp.poloidal_flux(phips=phips, iotas=iotas)
    q_factor = _pp.safety_factor(legacy.iotaf)
    mass = _pp.mass_profile(pres=pres_pa, vp=vp, gamma=gamma)
    beta_vol, betaxis, over_r = _pp.beta_volume_profiles(
        bmnc=nyq["bmnc"], gmnc=nyq["gmnc"], xm_nyq=xm_nyq,
        xn_nyq=xn_nyq, pres=pres_pa, vp=vp, signgs=int(signgs),
        rmnc=legacy.rmnc, xm=legacy.xm, xn=legacy.xn, ntheta=ntheta,
        nzeta=nzeta, nfp=nfp, lasym=lasym,
        bmns=nyq_a["bmns"], gmns=nyq_a["gmns"], rmns=a["rmns"])
    rmax_surf, rmin_surf, zmax_surf = _pp.surface_extrema(
        rmnc=legacy.rmnc, zmns=legacy.zmns, xm=legacy.xm, xn=legacy.xn,
        ntheta=ntheta, nzeta=nzeta, nfp=nfp, lasym=lasym,
        rmns=a["rmns"], zmnc=a["zmnc"])
    rbtor0, rbtor, b0, volavgb, ion_larmor = _pp.field_scalars(
        bvco=bvco, raxis_cc=legacy.raxis_cc, wb=float(legacy.wb),
        volume_p=float(legacy.volume_p))

    # -- histories --------------------------------------------------------
    fsqt_out = np.zeros((_NSTORE,), dtype=float)
    src = np.asarray(legacy.fsqt, dtype=float).ravel()
    fsqt_out[: min(src.size, _NSTORE)] = src[:_NSTORE]
    wdot_out = np.zeros((_NSTORE,), dtype=float)
    if wdot is not None:
        w = np.asarray(wdot, dtype=float).ravel()
        wdot_out[: min(w.size, _NSTORE)] = w[:_NSTORE]
    if itfsq <= 0:
        itfsq = int(np.count_nonzero(fsqt_out)) or 1

    return WoutData(
        version_=float(version),
        input_extension=str(input_extension),
        mgrid_file=_indata_str(indata, "MGRID_FILE", "NONE"),
        pcurr_type=_indata_str(indata, "PCURR_TYPE", "power_series"),
        pmass_type=_indata_str(indata, "PMASS_TYPE", "power_series"),
        piota_type=_indata_str(indata, "PIOTA_TYPE", "power_series"),
        wb=float(legacy.wb), wp=float(legacy.wp), gamma=gamma,
        rmax_surf=rmax_surf, rmin_surf=rmin_surf, zmax_surf=zmax_surf,
        nfp=nfp, ns=ns, mpol=int(legacy.mpol), ntor=int(legacy.ntor),
        mnmax=int(legacy.mnmax),
        mnyq=int(np.max(xm_nyq)) if xm_nyq.size else 0,
        nnyq=int(np.max(xn_nyq)) // nfp if xn_nyq.size else 0,
        mnmax_nyq=int(xm_nyq.size),
        niter=int(niter), itfsq=int(itfsq),
        lasym=lasym, lrecon=False,
        lfreeb=bool(indata.get_bool("LFREEB", False)),
        lmove_axis=bool(indata.get_bool("LMOVE_AXIS", True)),
        lrfp=bool(indata.get_bool("LRFP", False)),
        ier_flag=0 if bool(converged) else 2,
        aspect=float(legacy.aspect), betatotal=float(legacy.betatotal),
        betapol=float(legacy.betapol), betator=float(legacy.betator),
        betaxis=betaxis, b0=b0, rbtor0=rbtor0, rbtor=rbtor,
        signgs=int(signgs), IonLarmor=ion_larmor, volavgB=volavgb,
        ctor=ctor, Aminor_p=float(legacy.Aminor_p),
        Rmajor_p=float(legacy.Rmajor_p), volume_p=float(legacy.volume_p),
        ftolv=_ftolv_from_indata(indata, ns=ns),
        fsql=float(fsql), fsqr=float(fsqr), fsqz=float(fsqz),
        nextcur=0, extcur=np.zeros((1,), dtype=float), mgrid_mode="",
        xm=np.asarray(legacy.xm, dtype=float),
        xn=np.asarray(legacy.xn, dtype=float),
        xm_nyq=np.asarray(xm_nyq, dtype=float),
        xn_nyq=np.asarray(xn_nyq, dtype=float),
        raxis_cc=np.asarray(legacy.raxis_cc, dtype=float),
        zaxis_cs=np.asarray(legacy.zaxis_cs, dtype=float),
        am=_preset_array(indata, "AM"), ac=_preset_array(indata, "AC"),
        ai=_preset_array(indata, "AI"),
        am_aux_s=_preset_array(indata, "AM_AUX_S", _NDFMAX, -1.0),
        am_aux_f=_preset_array(indata, "AM_AUX_F", _NDFMAX, 0.0),
        ai_aux_s=_preset_array(indata, "AI_AUX_S", _NDFMAX, -1.0),
        ai_aux_f=_preset_array(indata, "AI_AUX_F", _NDFMAX, 0.0),
        ac_aux_s=_preset_array(indata, "AC_AUX_S", _NDFMAX, -1.0),
        ac_aux_f=_preset_array(indata, "AC_AUX_F", _NDFMAX, 0.0),
        iotaf=np.asarray(legacy.iotaf, dtype=float), q_factor=q_factor,
        presf=presf_pa, phi=np.asarray(legacy.phi, dtype=float),
        phipf=np.asarray(legacy.phipf, dtype=float), chi=chi,
        chipf=np.asarray(legacy.chipf, dtype=float),
        jcuru=jcuru, jcurv=jcurv,
        iotas=iotas, mass=mass, pres=pres_pa, beta_vol=beta_vol,
        buco=buco, bvco=bvco, vp=vp, specw=specw, phips=phips,
        over_r=over_r, jdotb=jdotb,
        bdotb=np.asarray(legacy.bdotb, dtype=float),
        bdotgradv=np.asarray(legacy.bdotgradv, dtype=float),
        DMerc=np.asarray(legacy.DMerc, dtype=float),
        DShear=np.asarray(legacy.Dshear, dtype=float),
        DWell=np.asarray(legacy.Dwell, dtype=float),
        DCurr=np.asarray(legacy.Dcurr, dtype=float),
        DGeod=np.asarray(legacy.Dgeod, dtype=float),
        equif=equif,
        fsqt=fsqt_out, wdot=wdot_out,
        rmnc=np.asarray(legacy.rmnc, dtype=float),
        zmns=np.asarray(legacy.zmns, dtype=float),
        lmns=np.asarray(legacy.lmns, dtype=float),
        gmnc=nyq["gmnc"], bmnc=nyq["bmnc"],
        bsubumnc=nyq["bsubumnc"], bsubvmnc=nyq["bsubvmnc"],
        bsubsmns=nyq["bsubsmns"],
        currumnc=currumnc, currvmnc=currvmnc,
        bsupumnc=nyq["bsupumnc"], bsupvmnc=nyq["bsupvmnc"],
        raxis_cs=np.asarray(legacy.raxis_cs, dtype=float) if lasym else None,
        zaxis_cc=np.asarray(legacy.zaxis_cc, dtype=float) if lasym else None,
        rmns=a["rmns"], zmnc=a["zmnc"],
        lmnc=np.asarray(legacy.lmnc, dtype=float) if lasym else None,
        gmns=nyq_a["gmns"], bmns=nyq_a["bmns"],
        bsubumns=nyq_a["bsubumns"], bsubvmns=nyq_a["bsubvmns"],
        bsubsmnc=nyq_a["bsubsmnc"],
        currumns=currumns, currvmns=currvmns,
        bsupumns=nyq_a["bsupumns"], bsupvmns=nyq_a["bsupvmns"],
    )


def wout_from_run(run, *, input_extension: str = "",
                  path: str | Path = "wout_vmec_jax.nc") -> WoutData:
    """Convenience wrapper: build :class:`WoutData` from a driver run object.

    Extracts the residual scalars, iteration counts and VMEC-style ``fsqt``
    history from a ``run_fixed_boundary`` result, then delegates to
    :func:`wout_from_state`.
    """
    res = getattr(run, "result", None)
    diag = getattr(res, "diagnostics", {}) or {}
    converged = bool(diag.get("converged", True))
    niter = int(getattr(res, "n_iter", 0) or 0)

    fsqr = fsqz = fsql = None
    fsqt = None
    rh = getattr(res, "fsqr2_history", None)
    zh = getattr(res, "fsqz2_history", None)
    lh = getattr(res, "fsql2_history", None)
    if rh is not None and zh is not None:
        rh = np.asarray(rh, dtype=float)
        zh = np.asarray(zh, dtype=float)
        hist = rh + zh
        n = int(hist.size)
        if n:
            stride = n // _NSTORE + 1
            fsqt = np.zeros((_NSTORE,), dtype=float)
            picks = hist[stride - 1::stride][:_NSTORE]
            fsqt[: picks.size] = picks
        fsqr = float(rh[-1])
        fsqz = float(zh[-1])
    if lh is not None:
        fsql = float(np.asarray(lh, dtype=float)[-1])
    if fsqr is None or fsqz is None or fsql is None:
        from ..drivers.output import residual_scalars_from_state

        fsqr, fsqz, fsql = residual_scalars_from_state(
            state=run.state, static=run.static, indata=run.indata,
            signgs=int(run.signgs), use_vmec_synthesis=True)

    return wout_from_state(
        state=run.state, static=run.static, indata=run.indata,
        signgs=int(run.signgs), fsqr=fsqr, fsqz=fsqz, fsql=fsql,
        fsqt=fsqt, niter=niter, converged=converged,
        input_extension=input_extension, path=path,
        flux_override=getattr(run, "flux", None),
        profiles_override=getattr(run, "profiles", None),
        force_payload_override=getattr(res, "_final_force_payload", None),
    )


def wout_field_names() -> tuple[str, ...]:
    """All :class:`WoutData` field names (for completeness checks)."""
    return tuple(f.name for f in _dc_fields(WoutData))
