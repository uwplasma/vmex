"""Low-level netCDF helpers for VMEC ``wout_*.nc`` I/O."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .schema import _bool_from_nc, _nc_scalar


# VMEC Nyquist Fourier fields are stored with (radius, mn_mode_nyq) dimensions.
NYQUIST_FOURIER_FIELD_NAMES: tuple[str, ...] = (
    "gmnc",
    "gmns",
    "bsupumnc",
    "bsupumns",
    "bsupvmnc",
    "bsupvmns",
    "bsubumnc",
    "bsubumns",
    "bsubvmnc",
    "bsubvmns",
    "bsubsmns",
    "bsubsmnc",
    "bmnc",
    "bmns",
)

_REQUIRED_NYQUIST_FOURIER_FIELDS = ("gmnc", "bsupumnc", "bsupvmnc")
_NYQUIST_DEFAULT_TEMPLATES = {
    "gmns": "gmnc",
    "bsupumns": "bsupumnc",
    "bsupvmns": "bsupvmnc",
    "bsubumnc": "bsupumnc",
    "bsubumns": "bsupumnc",
    "bsubvmnc": "bsupvmnc",
    "bsubvmns": "bsupvmnc",
    "bsubsmns": "bsupvmnc",
    "bsubsmnc": "bsupvmnc",
    "bmnc": "gmnc",
    "bmns": "gmnc",
}


def read_mode_table(variables: Any, name: str, *, path: Path) -> np.ndarray:
    """Read a required VMEC mode table and reject fully masked metadata."""
    raw = variables[name][:]
    if np.ma.isMaskedArray(raw):
        mask = np.asarray(raw.mask)
        if mask.size > 0 and bool(np.all(mask)):
            raise ValueError(f"Incomplete or masked wout mode metadata ({name}) in {path}")
    return np.asarray(np.ma.filled(raw, 0.0), dtype=int)


def read_optional_int_scalar(variables: Any, name: str, default: int | float) -> int:
    """Read an optional integer scalar from a netCDF variable mapping."""
    if name not in variables:
        return int(default)
    return int(_nc_scalar(variables[name][:], default, as_int=True))


def read_wout_scalar_metadata(variables: Any, *, path: Path) -> tuple[int, int, int, int, bool, int]:
    """Extract and validate scalar metadata required before reading WOUT arrays."""

    ns = int(_nc_scalar(variables["ns"][:], 0.0, as_int=True))
    mpol = int(_nc_scalar(variables["mpol"][:], 0.0, as_int=True))
    ntor = int(_nc_scalar(variables["ntor"][:], 0.0, as_int=True))
    nfp = int(_nc_scalar(variables["nfp"][:], 0.0, as_int=True))

    lasym_var = variables.get("lasym__logical__")
    lasym = _bool_from_nc(lasym_var[:] if lasym_var is not None else 0)
    signgs_var = variables.get("signgs")
    signgs = int(_nc_scalar(signgs_var[:] if signgs_var is not None else 1.0, 1.0, as_int=True))
    if ns <= 0 or mpol <= 0 or ntor < 0 or nfp <= 0:
        raise ValueError(f"Incomplete or masked wout scalar metadata in {path}")
    return ns, mpol, ntor, nfp, lasym, signgs


def read_type_field(variables: Any, name: str) -> str:
    """Read a VMEC fixed-width string field from netCDF character storage."""
    if name not in variables:
        return ""
    raw = np.asarray(variables[name][:])
    if raw.dtype.kind in ("S", "U"):
        if raw.ndim == 0:
            out = str(raw)
        else:
            out = b"".join(raw.astype("S1").ravel()).decode("utf-8", "ignore")
    else:
        try:
            out = "".join(raw.tolist())
        except Exception:
            out = str(raw)
    return out.rstrip()


def read_nyquist_fourier_fields(variables: Any) -> dict[str, np.ndarray]:
    """Read Nyquist Fourier field groups from a VMEC wout variable mapping.

    VMEC omits sine/asymmetric channels from some stellarator-symmetric output
    files.  Missing optional channels are returned as zeros shaped like the
    matching cosine field group so downstream code can treat every field as
    present.
    """

    required_fields = {name: np.asarray(variables[name][:]) for name in _REQUIRED_NYQUIST_FOURIER_FIELDS}
    fields: dict[str, np.ndarray] = {}
    for name in NYQUIST_FOURIER_FIELD_NAMES:
        if name in required_fields:
            fields[name] = required_fields[name]
            continue
        if name in variables:
            fields[name] = np.asarray(variables[name][:])
            continue
        fields[name] = np.zeros_like(fields[_NYQUIST_DEFAULT_TEMPLATES[name]])
    return fields


def write_int_variable(ds: Any, name: str, dims: tuple[str, ...], data: Any) -> None:
    """Create and write an int32 netCDF variable."""
    var = ds.createVariable(name, "i4", dims)
    var[:] = np.asarray(data, dtype=np.int32)


def write_float_variable(ds: Any, name: str, dims: tuple[str, ...], data: Any) -> None:
    """Create and write a float64 netCDF variable."""
    var = ds.createVariable(name, "f8", dims)
    var[:] = np.asarray(data, dtype=np.float64)


def write_fixed_width_string_variable(
    ds: Any,
    name: str,
    value: Any,
    *,
    dim: str = "dim_00020",
    width: int = 20,
) -> None:
    """Create and write a fixed-width VMEC string variable."""
    text = (str(value or "")[:width]).ljust(width)
    var = ds.createVariable(name, "S1", (dim,))
    var[:] = np.asarray(list(text), dtype="S1")


def write_nyquist_fourier_fields(ds: Any, wout: Any) -> None:
    """Write Nyquist Fourier field groups from a WoutData-like object."""
    dims = ("radius", "mn_mode_nyq")
    for name in NYQUIST_FOURIER_FIELD_NAMES:
        write_float_variable(ds, name, dims, np.asarray(getattr(wout, name)))


def read_wout_payload(
    path: str | Path,
    *,
    mu0: float,
    phi_profile_from_variables_func: Callable[..., np.ndarray],
    glasser_profiles_from_variables_func: Callable[..., Any],
) -> dict[str, Any]:
    """Read a VMEC ``wout`` netCDF file into ``WoutData`` constructor kwargs."""

    path = Path(path)
    try:
        import netCDF4  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("netCDF4 is required to read wout files (pip install vmec-jax)") from e

    with netCDF4.Dataset(path) as ds:
        ns, mpol, ntor, nfp, lasym, signgs = read_wout_scalar_metadata(ds.variables, path=path)

        xm = read_mode_table(ds.variables, "xm", path=path)
        xn = read_mode_table(ds.variables, "xn", path=path)
        xm_nyq = read_mode_table(ds.variables, "xm_nyq", path=path)
        xn_nyq = read_mode_table(ds.variables, "xn_nyq", path=path)
        mpol_nyq_default = int(np.max(xm_nyq)) if xm_nyq.size else 0
        ntor_nyq_default = int(np.max(np.abs(xn_nyq // nfp))) if xn_nyq.size else 0
        mnmax = read_optional_int_scalar(ds.variables, "mnmax", xm.size)
        mnmax_nyq = read_optional_int_scalar(ds.variables, "mnmax_nyq", xm_nyq.size)
        mpol_nyq = read_optional_int_scalar(ds.variables, "mpol_nyq", mpol_nyq_default)
        ntor_nyq = read_optional_int_scalar(ds.variables, "ntor_nyq", ntor_nyq_default)

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

        nyquist_fields = read_nyquist_fourier_fields(ds.variables)

        wb = float(_nc_scalar(ds.variables["wb"][:], 0.0))
        volume_p = float(_nc_scalar(ds.variables["volume_p"][:], 0.0))
        gamma = float(_nc_scalar(ds.variables.get("gamma", 0.0)[:], 0.0)) if "gamma" in ds.variables else 0.0
        wp = float(_nc_scalar(ds.variables.get("wp", 0.0)[:], 0.0)) if "wp" in ds.variables else 0.0
        vp = np.asarray(ds.variables.get("vp", np.zeros((ns,), dtype=float))[:])

        pres_pa = np.asarray(ds.variables.get("pres", np.zeros((ns,), dtype=float))[:])
        presf_pa = np.asarray(ds.variables.get("presf", np.zeros((ns,), dtype=float))[:])
        pres = float(mu0) * pres_pa
        presf = float(mu0) * presf_pa

        fsqr = float(_nc_scalar(ds.variables.get("fsqr", 0.0)[:], 0.0)) if "fsqr" in ds.variables else 0.0
        fsqz = float(_nc_scalar(ds.variables.get("fsqz", 0.0)[:], 0.0)) if "fsqz" in ds.variables else 0.0
        fsql = float(_nc_scalar(ds.variables.get("fsql", 0.0)[:], 0.0)) if "fsql" in ds.variables else 0.0
        fsqt = np.asarray(ds.variables.get("fsqt", np.zeros((0,), dtype=float))[:])
        equif = np.asarray(ds.variables.get("equif", np.zeros((ns,), dtype=float))[:])

        phi = phi_profile_from_variables_func(ds.variables, ns=ns, phipf=phipf)

        buco = np.asarray(ds.variables.get("buco", np.zeros((ns,), dtype=float))[:])
        bvco = np.asarray(ds.variables.get("bvco", np.zeros((ns,), dtype=float))[:])
        jcuru = np.asarray(ds.variables.get("jcuru", np.zeros((ns,), dtype=float))[:])
        jcurv = np.asarray(ds.variables.get("jcurv", np.zeros((ns,), dtype=float))[:])

        raxis_cc = np.asarray(ds.variables.get("raxis_cc", np.zeros((ntor + 1,), dtype=float))[:])
        zaxis_cs = np.asarray(ds.variables.get("zaxis_cs", np.zeros((ntor + 1,), dtype=float))[:])
        raxis_cs = np.asarray(ds.variables.get("raxis_cs", np.zeros_like(raxis_cc))[:])
        zaxis_cc = np.asarray(ds.variables.get("zaxis_cc", np.zeros_like(zaxis_cs))[:])

        Aminor_p = float(_nc_scalar(ds.variables.get("Aminor_p", 0.0)[:], 0.0)) if "Aminor_p" in ds.variables else 0.0
        Rmajor_p = float(_nc_scalar(ds.variables.get("Rmajor_p", 0.0)[:], 0.0)) if "Rmajor_p" in ds.variables else 0.0
        aspect = float(_nc_scalar(ds.variables.get("aspect", 0.0)[:], 0.0)) if "aspect" in ds.variables else 0.0
        betatotal = (
            float(_nc_scalar(ds.variables.get("betatotal", 0.0)[:], 0.0)) if "betatotal" in ds.variables else 0.0
        )
        betapol = float(_nc_scalar(ds.variables.get("betapol", 0.0)[:], 0.0)) if "betapol" in ds.variables else 0.0
        betator = float(_nc_scalar(ds.variables.get("betator", 0.0)[:], 0.0)) if "betator" in ds.variables else 0.0
        betaxis = float(_nc_scalar(ds.variables.get("betaxis", 0.0)[:], 0.0)) if "betaxis" in ds.variables else 0.0
        ctor = float(_nc_scalar(ds.variables.get("ctor", 0.0)[:], 0.0)) if "ctor" in ds.variables else 0.0

        DMerc = np.asarray(ds.variables.get("DMerc", np.zeros((ns,), dtype=float))[:])
        Dshear = np.asarray(ds.variables.get("DShear", np.zeros((ns,), dtype=float))[:])
        Dwell = np.asarray(ds.variables.get("DWell", np.zeros((ns,), dtype=float))[:])
        Dcurr = np.asarray(ds.variables.get("DCurr", np.zeros((ns,), dtype=float))[:])
        Dgeod = np.asarray(ds.variables.get("DGeod", np.zeros((ns,), dtype=float))[:])
        jdotb = np.asarray(ds.variables.get("jdotb", np.zeros((ns,), dtype=float))[:])
        bdotb = np.asarray(ds.variables.get("bdotb", np.zeros((ns,), dtype=float))[:])
        bdotgradv = np.asarray(ds.variables.get("bdotgradv", np.zeros((ns,), dtype=float))[:])
        glasser_profiles = glasser_profiles_from_variables_func(
            ds.variables,
            DMerc=DMerc,
            Dshear=Dshear,
            Dcurr=Dcurr,
        )

        ac = np.asarray(ds.variables.get("ac", np.zeros((0,), dtype=float))[:])
        ac_aux_s = np.asarray(ds.variables.get("ac_aux_s", -np.ones((101,), dtype=float))[:])
        ac_aux_f = np.asarray(ds.variables.get("ac_aux_f", np.zeros((101,), dtype=float))[:])

        pcurr_type = read_type_field(ds.variables, "pcurr_type")
        piota_type = read_type_field(ds.variables, "piota_type")
        ier_flag = read_optional_int_scalar(ds.variables, "ier_flag", 0)
        vmec_jax_converged_var = ds.variables.get("vmec_jax_converged__logical__")
        if vmec_jax_converged_var is None:
            vmec_jax_converged = ier_flag == 0
        else:
            vmec_jax_converged = _bool_from_nc(vmec_jax_converged_var[:])
        vmec_jax_status = read_type_field(ds.variables, "vmec_jax_status")
        if not vmec_jax_status:
            vmec_jax_status = "converged" if bool(vmec_jax_converged) else "nonconverged"

    return dict(
        path=path,
        ns=ns,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        lasym=lasym,
        signgs=signgs,
        mnmax=mnmax,
        mpol_nyq=mpol_nyq,
        ntor_nyq=ntor_nyq,
        mnmax_nyq=mnmax_nyq,
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
        **nyquist_fields,
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
        Dshear=Dshear,
        Dwell=Dwell,
        Dcurr=Dcurr,
        Dgeod=Dgeod,
        D_R=glasser_profiles.D_R,
        H=glasser_profiles.H,
        glasser_correction=glasser_profiles.correction,
        glasser_shear_valid=glasser_profiles.shear_valid,
        jdotb=jdotb,
        bdotb=bdotb,
        bdotgradv=bdotgradv,
        ac=ac,
        ac_aux_s=ac_aux_s,
        ac_aux_f=ac_aux_f,
        pcurr_type=pcurr_type,
        piota_type=piota_type,
        ier_flag=ier_flag,
        vmec_jax_converged=bool(vmec_jax_converged),
        vmec_jax_status=vmec_jax_status,
    )


def write_wout_payload(
    path: str | Path,
    wout: Any,
    *,
    overwrite: bool = False,
    mu0: float,
    glasser_profiles_from_wout_data_func: Callable[..., Any],
    getenv: Callable[[str, str], str] = os.getenv,
    print_func: Callable[..., Any] = print,
) -> None:
    """Write a WoutData-like object to VMEC-style ``wout_*.nc`` netCDF."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True to overwrite)")

    try:
        import netCDF4  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("netCDF4 is required to write wout files (pip install vmec-jax)") from e

    ns = int(wout.ns)
    mnmax = int(np.asarray(wout.xm).size)
    mnmax_nyq = int(np.asarray(wout.xm_nyq).size)
    nstore = int(np.asarray(wout.fsqt).size)
    n_tor = int(wout.ntor) + 1
    ac = np.asarray(getattr(wout, "ac", np.zeros((0,), dtype=float)))
    if ac.size == 0:
        ac = np.zeros((21,), dtype=float)
    ac_aux_s = np.asarray(getattr(wout, "ac_aux_s", -np.ones((101,), dtype=float)))
    ac_aux_f = np.asarray(getattr(wout, "ac_aux_f", np.zeros((101,), dtype=float)))
    if ac_aux_s.size == 0:
        ac_aux_s = -np.ones((1,), dtype=float)
    if ac_aux_f.size == 0:
        ac_aux_f = np.zeros((1,), dtype=float)
    ndfmax = int(ac_aux_s.size)
    preset = int(ac.size)

    pres_pa = np.asarray(wout.pres) / float(mu0)
    presf_pa = np.asarray(wout.presf) / float(mu0)

    with netCDF4.Dataset(path, mode="w", format="NETCDF3_CLASSIC") as ds:
        try:
            ds.set_fill_off()
        except Exception as exc:
            if getenv("VMEC_JAX_MERCIER_RAISE", "") not in ("", "0", "false", "no"):
                raise
            if getenv("VMEC_JAX_MERCIER_LOG", "") not in ("", "0", "false", "no"):
                print_func(f"[vmec_jax] Mercier/jdotb computation failed: {exc}", flush=True)
        for name, size in (
            ("radius", ns),
            ("mn_mode", mnmax),
            ("mn_mode_nyq", mnmax_nyq),
            ("nstore_seq", nstore),
            ("n_tor", n_tor),
            ("ndfmax", ndfmax),
            ("preset", preset),
            ("dim_00020", 20),
        ):
            ds.createDimension(name, size)

        wout_converged = bool(getattr(wout, "vmec_jax_converged", True))
        ier_flag = int(getattr(wout, "ier_flag", 0 if wout_converged else 1))
        int_scalars = (
            ("ns", ns),
            ("mpol", int(wout.mpol)),
            ("ntor", int(wout.ntor)),
            ("nfp", int(wout.nfp)),
            ("signgs", int(wout.signgs)),
            ("lasym__logical__", int(bool(wout.lasym))),
            ("ier_flag", ier_flag),
            ("vmec_jax_converged__logical__", int(wout_converged)),
            ("mnmax", int(getattr(wout, "mnmax", mnmax))),
            ("mpol_nyq", int(getattr(wout, "mpol_nyq", np.max(np.asarray(wout.xm_nyq)) if mnmax_nyq > 0 else 0))),
            (
                "ntor_nyq",
                int(
                    getattr(
                        wout,
                        "ntor_nyq",
                        np.max(np.abs(np.asarray(wout.xn_nyq) // int(wout.nfp))) if mnmax_nyq > 0 else 0,
                    )
                ),
            ),
            ("mnmax_nyq", int(getattr(wout, "mnmax_nyq", mnmax_nyq))),
        )
        for name, value in int_scalars:
            write_int_variable(ds, name, (), np.asarray(value))

        for name in ("wb", "volume_p", "gamma", "wp", "fsqr", "fsqz", "fsql"):
            write_float_variable(ds, name, (), np.asarray(float(getattr(wout, name))))

        for name, dims in (
            ("xm", ("mn_mode",)),
            ("xn", ("mn_mode",)),
            ("xm_nyq", ("mn_mode_nyq",)),
            ("xn_nyq", ("mn_mode_nyq",)),
            ("rmnc", ("radius", "mn_mode")),
            ("rmns", ("radius", "mn_mode")),
            ("zmnc", ("radius", "mn_mode")),
            ("zmns", ("radius", "mn_mode")),
            ("lmnc", ("radius", "mn_mode")),
            ("lmns", ("radius", "mn_mode")),
        ):
            write_float_variable(ds, name, dims, np.asarray(getattr(wout, name)))

        radius_fields = (
            ("phipf", np.asarray(wout.phipf)),
            ("chipf", np.asarray(wout.chipf)),
            ("phips", np.asarray(wout.phips)),
            ("iotaf", np.asarray(wout.iotaf)),
            ("iotas", np.asarray(wout.iotas)),
            ("phi", np.asarray(getattr(wout, "phi", np.zeros((ns,), dtype=float)))),
        )
        for name, value in radius_fields:
            write_float_variable(ds, name, ("radius",), value)

        write_nyquist_fourier_fields(ds, wout)

        zero_radius = np.zeros((ns,), dtype=float)
        for name, value in (
            ("vp", np.asarray(wout.vp)),
            ("pres", np.asarray(pres_pa)),
            ("presf", np.asarray(presf_pa)),
            ("equif", np.asarray(getattr(wout, "equif", zero_radius))),
            ("buco", np.asarray(getattr(wout, "buco", zero_radius))),
            ("bvco", np.asarray(getattr(wout, "bvco", zero_radius))),
            ("jcuru", np.asarray(getattr(wout, "jcuru", zero_radius))),
            ("jcurv", np.asarray(getattr(wout, "jcurv", zero_radius))),
            ("jdotb", np.asarray(getattr(wout, "jdotb", zero_radius))),
            ("bdotb", np.asarray(getattr(wout, "bdotb", zero_radius))),
            ("bdotgradv", np.asarray(getattr(wout, "bdotgradv", zero_radius))),
            ("DMerc", np.asarray(getattr(wout, "DMerc", zero_radius))),
            ("DShear", np.asarray(getattr(wout, "Dshear", zero_radius))),
            ("DWell", np.asarray(getattr(wout, "Dwell", zero_radius))),
            ("DCurr", np.asarray(getattr(wout, "Dcurr", zero_radius))),
            ("DGeod", np.asarray(getattr(wout, "Dgeod", zero_radius))),
        ):
            write_float_variable(ds, name, ("radius",), value)
        glasser_profiles = glasser_profiles_from_wout_data_func(wout, ns)
        for name, value in (
            ("D_R", glasser_profiles.D_R),
            ("HGlasser", glasser_profiles.H),
            ("GlasserCorrection", glasser_profiles.correction),
            ("GlasserShearValid", np.asarray(glasser_profiles.shear_valid, dtype=float)),
        ):
            write_float_variable(ds, name, ("radius",), value)

        write_float_variable(ds, "fsqt", ("nstore_seq",), np.asarray(wout.fsqt))

        zero_tor = np.zeros((n_tor,), dtype=float)
        for name in ("raxis_cc", "zaxis_cs", "raxis_cs", "zaxis_cc"):
            write_float_variable(ds, name, ("n_tor",), np.asarray(getattr(wout, name, zero_tor)))

        for name in ("Aminor_p", "Rmajor_p", "aspect", "betatotal", "betapol", "betator", "betaxis", "ctor"):
            write_float_variable(ds, name, (), np.asarray(float(getattr(wout, name, 0.0))))

        for name, dims, value in (
            ("ac_aux_s", ("ndfmax",), ac_aux_s),
            ("ac_aux_f", ("ndfmax",), ac_aux_f),
            ("ac", ("preset",), ac),
        ):
            write_float_variable(ds, name, dims, np.asarray(value))

        write_fixed_width_string_variable(ds, "pcurr_type", getattr(wout, "pcurr_type", ""))
        write_fixed_width_string_variable(ds, "piota_type", getattr(wout, "piota_type", ""))
        write_fixed_width_string_variable(
            ds,
            "vmec_jax_status",
            getattr(wout, "vmec_jax_status", "converged" if wout_converged else "nonconverged"),
        )
