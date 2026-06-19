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

from ..._compat import has_jax, jax
from ...namelist import InData
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


class WoutMinimalRuntimeOptions(NamedTuple):
    """Environment-derived switches for minimal WOUT construction."""

    timing_enabled: bool
    light: bool
    fast_bcovar: bool


class WoutMinimalFieldOptions(NamedTuple):
    """Environment-derived switches for Bsub/Mercier field-output assembly."""

    mercier_bsub_source: str
    mercier_use_bsube: bool
    disable_bsubv_equif_corr: bool
    apply_bss_scalxc: bool
    skip_bsub_filter: bool
    filter_from_raw: bool
    use_lasym_loop: bool
    lasym_filter: bool
    lasym_filter_use_parity_channels: bool
    symmetric_wrout_loop: bool
    mercier_use_wrout_bsubuv: bool
    bsub_filter_use_bc_parity: bool
    zero_nonconverged_beta: bool


class WoutBcovarPayload(NamedTuple):
    """Force/bcovar state needed by minimal WOUT assembly."""

    bc: Any
    k_force: Any | None
    indata_wout: Any


class WoutBssSourcePayload(NamedTuple):
    """BSS/JXBFORCE source arrays selected for minimal WOUT assembly."""

    use_force_bss: bool
    k_force: Any | None
    bsupu: np.ndarray
    bsupv: np.ndarray
    ru12: np.ndarray | None
    zu12: np.ndarray | None
    rs: np.ndarray | None
    zs: np.ndarray | None
    crmn_e_sym: np.ndarray | None
    czmn_e_sym: np.ndarray | None
    bzmn_e_sym: np.ndarray | None
    brmn_e_sym: np.ndarray | None
    azmn_e_sym: np.ndarray | None
    armn_e_sym: np.ndarray | None
    geom: dict[str, Any]


def select_bsubuv_diagnostic_fields(
    *,
    bc: Any,
    bsubu_out: np.ndarray,
    bsubv_out: np.ndarray,
    field_options: WoutMinimalFieldOptions,
    trig: Any,
    apply_bsubv_equif_correction_func,
) -> tuple[np.ndarray, np.ndarray]:
    """Select VMEC-compatible ``bsubu``/``bsubv`` fields for diagnostics."""
    bsubu_diag = np.asarray(bsubu_out, dtype=float)
    bsubv_diag = np.asarray(bsubv_out, dtype=float)
    bsub_src = str(field_options.mercier_bsub_source)
    if bsub_src in {"bsubu_e", "bsubu_e_scaled", "bsubu"}:
        u_name = bsub_src
        v_name = bsub_src.replace("bsubu", "bsubv")
        if hasattr(bc, u_name) and hasattr(bc, v_name):
            bsubu_diag = np.asarray(getattr(bc, u_name), dtype=float)
            bsubv_diag = np.asarray(getattr(bc, v_name), dtype=float)
    elif field_options.mercier_use_bsube and hasattr(bc, "bsubu_e") and hasattr(bc, "bsubv_e"):
        bsubu_diag = np.asarray(getattr(bc, "bsubu_e"), dtype=float)
        bsubv_diag = np.asarray(getattr(bc, "bsubv_e"), dtype=float)

    if (not field_options.disable_bsubv_equif_corr) and getattr(bc, "bsubv_e", None) is not None:
        # VMEC fileout.f forces IEQUI=1 before funct3d/wrout at output time.
        bsubv_diag = apply_bsubv_equif_correction_func(
            bsubv=bsubv_diag,
            bsubv_e=np.asarray(bc.bsubv_e),
            trig=trig,
        )
    return bsubu_diag, bsubv_diag


class WoutScalarDiagnostics(NamedTuple):
    """Scalar and radial diagnostics written by the minimal WOUT builder."""

    betatotal: float
    betapol: float
    betator: float
    betaxis: float
    ctor: float
    DMerc: np.ndarray
    Dshear: np.ndarray
    Dcurr: np.ndarray
    Dwell: np.ndarray
    Dgeod: np.ndarray
    D_R: np.ndarray
    H_glasser: np.ndarray
    glasser_correction: np.ndarray
    glasser_shear_valid: np.ndarray
    jdotb: np.ndarray
    bdotb: np.ndarray
    bdotgradv: np.ndarray


def env_enabled(value: str | None, *, false_values: tuple[str, ...] = ("", "0", "false", "no")) -> bool:
    """Return whether a VMEC-JAX environment toggle should be considered enabled."""

    if value is None:
        return False
    return value.strip().lower() not in false_values


def device_get_if_available(value: Any) -> Any:
    """Host-materialize a JAX pytree when JAX is available, otherwise return it."""

    if has_jax():
        try:
            return jax.device_get(value)
        except Exception:
            pass
    return value


def attach_force_payload_geometry(geom: dict[str, Any], k_force: Any) -> Any:
    """Attach VMEC force-kernel geometry channels and return the bcovar payload."""

    geom["pr1_even"] = np.asarray(k_force.pr1_even, dtype=float)
    geom["pr1_odd"] = np.asarray(k_force.pr1_odd, dtype=float)
    geom["pz1_even"] = np.asarray(k_force.pz1_even, dtype=float)
    geom["pz1_odd"] = np.asarray(k_force.pz1_odd, dtype=float)
    geom["pru_even"] = np.asarray(k_force.pru_even, dtype=float)
    geom["pru_odd"] = np.asarray(k_force.pru_odd, dtype=float)
    geom["pzu_even"] = np.asarray(k_force.pzu_even, dtype=float)
    geom["pzu_odd"] = np.asarray(k_force.pzu_odd, dtype=float)
    geom["prv_even"] = np.asarray(k_force.prv_even, dtype=float)
    geom["prv_odd"] = np.asarray(k_force.prv_odd, dtype=float)
    geom["pzv_even"] = np.asarray(k_force.pzv_even, dtype=float)
    geom["pzv_odd"] = np.asarray(k_force.pzv_odd, dtype=float)
    return k_force.bc


def indata_for_wout_force_path(indata: Any, *, force_iequi1: bool) -> Any:
    """Return the input deck used for WOUT force diagnostics."""

    if not bool(force_iequi1):
        return indata
    try:
        out = InData(
            scalars=dict(indata.scalars),
            indexed=dict(indata.indexed),
            source_path=indata.source_path,
        )
        out.scalars["IEQUI"] = 1
        return out
    except Exception:
        return indata


def prepare_wout_bcovar_payload(
    *,
    state: Any,
    static: Any,
    indata: Any,
    wout_like: Any,
    pres: np.ndarray,
    geom: dict[str, Any],
    force_payload_override: Any,
    fast_bcovar: bool,
    timing_enabled: bool,
    timing: dict[str, float],
    vmec_bcovar_half_mesh_from_wout_func: Any,
    vmec_forces_rz_from_wout_func: Any,
    numpy_module_patch_func: Any,
) -> WoutBcovarPayload:
    """Resolve the force/bcovar source used by minimal WOUT diagnostics."""

    if timing_enabled:
        import time as _time

        t0 = _time.perf_counter()

    force_iequi1 = env_enabled(os.getenv("VMEC_JAX_WOUT_FORCE_IEQUI1", "0"))
    indata_wout = indata_for_wout_force_path(indata, force_iequi1=bool(force_iequi1))
    reuse_final_bcovar = env_enabled(
        os.getenv("VMEC_JAX_WOUT_REUSE_FINAL_BCOVAR", ""),
        false_values=("", "0", "false", "no", "off"),
    )

    k_force = None
    if force_payload_override is not None and (reuse_final_bcovar or not fast_bcovar) and (not force_iequi1):
        k_force = device_get_if_available(force_payload_override)
        bc = attach_force_payload_geometry(geom, k_force)
    elif fast_bcovar:
        with numpy_module_patch_func():
            bc = vmec_bcovar_half_mesh_from_wout_func(
                state=state,
                static=static,
                wout=wout_like,
                pres=pres,
                use_wout_bsup=False,
                use_wout_bsub_for_lambda=False,
                use_wout_bmag_for_bsq=False,
                use_vmec_synthesis=True,
                trig=None,
            )
        bc = device_get_if_available(bc)
    else:
        wout_force_vmec_synth = env_enabled(os.getenv("VMEC_JAX_WOUT_FORCE_VMEC_SYNTH", ""))
        k_force = vmec_forces_rz_from_wout_func(
            state=state,
            static=static,
            wout=wout_like,
            indata=indata_wout,
            use_wout_bsup=False,
            use_vmec_synthesis=wout_force_vmec_synth,
            trig=None,
        )
        k_force = device_get_if_available(k_force)
        bc = attach_force_payload_geometry(geom, k_force)

    if timing_enabled:
        timing["forces_bcovar_s"] = _time.perf_counter() - t0
    return WoutBcovarPayload(
        bc=bc,
        k_force=k_force,
        indata_wout=indata_wout,
    )


def prepare_wout_bss_source_payload(
    *,
    state: Any,
    static: Any,
    indata_wout: Any,
    wout_like: Any,
    bc: Any,
    k_force: Any | None,
    trig: Any,
    geom: dict[str, Any],
    lasym: bool,
    force_sym_func: Any,
    vmec_forces_rz_from_wout_func: Any,
    environ: Mapping[str, str] | None = None,
) -> WoutBssSourcePayload:
    """Select raw or force-kernel source arrays for the BSS output path."""

    env = os.environ if environ is None else environ
    force_bss_env = env.get("VMEC_JAX_WOUT_FORCE_BSS", "").strip().lower()
    if force_bss_env == "":
        # Default to bcovar/Jacobian bss inputs. Force-kernel bss inputs remain
        # opt-in for targeted debugging.
        use_force_bss = False
    else:
        use_force_bss = force_bss_env not in ("0", "false", "no")

    bsupu_bss = np.asarray(bc.bsupu, dtype=float)
    bsupv_bss = np.asarray(bc.bsupv, dtype=float)
    ru12_bss = None
    zu12_bss = None
    rs_bss = None
    zs_bss = None
    crmn_e_sym = None
    czmn_e_sym = None
    bzmn_e_sym = None
    brmn_e_sym = None
    azmn_e_sym = None
    armn_e_sym = None
    use_parity_geom_bss = env.get("VMEC_JAX_BSS_FROM_PARITY_GEOM", "1") not in ("", "0")
    geom_bss = geom if use_parity_geom_bss else {}

    if use_force_bss and (k_force is None):
        wout_force_vmec_synth_env = env.get("VMEC_JAX_WOUT_FORCE_VMEC_SYNTH", "").strip().lower()
        wout_force_vmec_synth = wout_force_vmec_synth_env not in ("", "0", "false", "no")
        k_force = vmec_forces_rz_from_wout_func(
            state=state,
            static=static,
            wout=wout_like,
            indata=indata_wout,
            use_wout_bsup=False,
            use_vmec_synthesis=wout_force_vmec_synth,
            trig=None,
        )
        k_force = device_get_if_available(k_force)

    if use_force_bss and (k_force is not None):
        if hasattr(k_force, "crmn_e") and hasattr(k_force, "czmn_e"):
            crmn_e_sym = force_sym_func(k_force.crmn_e, "crs")
            czmn_e_sym = force_sym_func(k_force.czmn_e, "czs")
            bsupu_bss = crmn_e_sym
            bsupv_bss = czmn_e_sym
        if hasattr(k_force, "bzmn_e"):
            bzmn_e_sym = force_sym_func(k_force.bzmn_e, "bzs")
            rs_bss = bzmn_e_sym
        if hasattr(k_force, "brmn_e"):
            brmn_e_sym = force_sym_func(k_force.brmn_e, "brs")
            zs_bss = brmn_e_sym
        if hasattr(k_force, "azmn_e"):
            azmn_e_sym = force_sym_func(k_force.azmn_e, "azs")
            ru12_bss = azmn_e_sym
        if hasattr(k_force, "armn_e"):
            armn_e_sym = force_sym_func(k_force.armn_e, "ars")
            zu12_bss = armn_e_sym

    return WoutBssSourcePayload(
        use_force_bss=bool(use_force_bss),
        k_force=k_force,
        bsupu=bsupu_bss,
        bsupv=bsupv_bss,
        ru12=ru12_bss,
        zu12=zu12_bss,
        rs=rs_bss,
        zs=zs_bss,
        crmn_e_sym=crmn_e_sym,
        czmn_e_sym=czmn_e_sym,
        bzmn_e_sym=bzmn_e_sym,
        brmn_e_sym=brmn_e_sym,
        azmn_e_sym=azmn_e_sym,
        armn_e_sym=armn_e_sym,
        geom=geom_bss,
    )


def minimal_wout_runtime_options_from_env(environ: Mapping[str, str] | None = None) -> WoutMinimalRuntimeOptions:
    """Resolve passive WOUT runtime switches from environment variables."""

    env = os.environ if environ is None else environ
    timing_enabled = env_enabled(env.get("VMEC_JAX_WOUT_TIMING", ""))
    light = env_enabled(env.get("VMEC_JAX_WOUT_LIGHT", ""))
    fast_bcovar = env_enabled(env.get("VMEC_JAX_WOUT_FAST_BCOVAR", ""), false_values=("0", "false", "no", "off"))
    if light:
        fast_bcovar = True
    return WoutMinimalRuntimeOptions(
        timing_enabled=bool(timing_enabled),
        light=bool(light),
        fast_bcovar=bool(fast_bcovar),
    )


def minimal_wout_field_options_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    wout_light: bool = False,
) -> WoutMinimalFieldOptions:
    """Resolve Bsub/Mercier field-output switches from environment variables."""

    env = os.environ if environ is None else environ
    strict_lasym_loop = env_enabled(
        env.get("VMEC_JAX_WROUT_LASYM_STRICT", ""),
        false_values=("", "0", "false", "no"),
    )
    return WoutMinimalFieldOptions(
        mercier_bsub_source=env.get("VMEC_JAX_MERCIER_BSUB_SOURCE", "").strip().lower(),
        mercier_use_bsube=env_enabled(env.get("VMEC_JAX_MERCIER_USE_BSUBE", "0"), false_values=("", "0")),
        disable_bsubv_equif_corr=env_enabled(
            env.get("VMEC_JAX_DISABLE_BSUBV_EQUI_CORR", "1"),
            false_values=("", "0"),
        ),
        apply_bss_scalxc=env_enabled(env.get("VMEC_JAX_BSS_APPLY_SCALXC", "1"), false_values=("", "0")),
        skip_bsub_filter=bool(wout_light)
        or env_enabled(env.get("VMEC_JAX_SKIP_BSUB_FILTER", ""), false_values=("", "0")),
        filter_from_raw=env_enabled(env.get("VMEC_JAX_MERCIER_FILTER_FROM_RAW", "0"), false_values=("", "0")),
        use_lasym_loop=bool(strict_lasym_loop)
        or env_enabled(env.get("VMEC_JAX_WROUT_LASYM_LOOP", "0"), false_values=("", "0", "false", "no")),
        lasym_filter=env_enabled(env.get("VMEC_JAX_WROUT_LASYM_FILTER", "1"), false_values=("", "0")),
        lasym_filter_use_parity_channels=env_enabled(
            env.get("VMEC_JAX_LASYM_FILTER_USE_PARITY_CHANNELS", "0"),
            false_values=("", "0", "false", "no"),
        ),
        symmetric_wrout_loop=env_enabled(env.get("VMEC_JAX_WROUT_LOOP", "0"), false_values=("", "0")),
        mercier_use_wrout_bsubuv=env_enabled(
            env.get("VMEC_JAX_MERCIER_USE_WROUT_BSUBUV", ""),
            false_values=("", "0"),
        ),
        bsub_filter_use_bc_parity=env_enabled(
            env.get("VMEC_JAX_BSUB_FILTER_USE_BC_PARITY", "0"),
            false_values=("", "0", "false", "no"),
        ),
        zero_nonconverged_beta=env_enabled(
            env.get("VMEC_JAX_WOUT_ZERO_NONCONVERGED_BETA", ""),
            false_values=("", "0", "false", "no", "off"),
        ),
    )


def lbsubs_from_indata_and_env(indata: Any, environ: Mapping[str, str] | None = None) -> bool:
    """Resolve VMEC ``LBSUBS`` output policy with the debug env override."""

    env = os.environ if environ is None else environ
    lbsubs = bool(getattr(indata, "get_bool", lambda *_args, **_kwargs: False)("LBSUBS", False))
    if env_enabled(env.get("VMEC_JAX_ENABLE_BSUBS_CORR", "")):
        lbsubs = True
    return bool(lbsubs)


def pressure_profiles_from_mass_vp(
    *,
    mass: np.ndarray,
    vp: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct VMEC half/full-mesh pressure profiles from mass and volume."""

    mass_arr = np.asarray(mass, dtype=float)
    vp_arr = np.asarray(vp, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(vp_arr != 0.0, vp_arr, 1.0)
        pres = np.where(vp_arr != 0.0, mass_arr / (denom**float(gamma)), 0.0)
    if pres.size:
        pres = pres.copy()
        pres[0] = 0.0

    if pres.size < 2:
        presf = pres.copy()
    else:
        presf = np.zeros_like(pres)
        if pres.size >= 3:
            presf[0] = 1.5 * pres[1] - 0.5 * pres[2]
        else:
            presf[0] = pres[1]
        presf[1:-1] = 0.5 * (pres[1:-1] + pres[2:])
        presf[-1] = 1.5 * pres[-1] - 0.5 * pres[-2]
    return pres, presf


def compute_minimal_wout_scalar_diagnostics(
    *,
    ns: int,
    wout_light: bool,
    betatotal: float,
    state: Any,
    static: Any,
    s: np.ndarray,
    lconm1: bool,
    ntor: int,
    nfp: int,
    mpol: int,
    lasym: bool,
    lbsubs: bool,
    signgs: int,
    pres: np.ndarray,
    vp: np.ndarray,
    flux_phips: np.ndarray,
    iotas: np.ndarray,
    bc: Any,
    buco: np.ndarray,
    trig: Any,
    geom_bss: dict[str, Any],
    bsupu_bss: np.ndarray,
    bsupv_bss: np.ndarray,
    rs_bss: np.ndarray | None,
    zs_bss: np.ndarray | None,
    ru12_bss: np.ndarray | None,
    zu12_bss: np.ndarray | None,
    bsubu_diag: np.ndarray,
    bsubv_diag: np.ndarray,
    bsubu_raw: np.ndarray,
    bsubv_raw: np.ndarray,
    bsubu_phys: np.ndarray | None,
    bsubv_phys: np.ndarray | None,
    indata: Any,
    timing_enabled: bool,
    timing: dict[str, float],
    vmec_wint_from_trig_func: Any,
    compute_eqfor_betaxis_func: Any,
    compute_eqfor_beta_func: Any,
    compute_ctor_from_buco_func: Any,
    compute_mercier_func: Any,
    glasser_from_wout_mercier_terms_func: Any,
) -> WoutScalarDiagnostics:
    """Compute beta, current, Mercier, and Glasser profiles for minimal WOUT."""

    betapol = 0.0
    betator = 0.0
    betaxis = 0.0
    ctor = 0.0
    DMerc = np.zeros((ns,), dtype=float)
    Dshear = np.zeros((ns,), dtype=float)
    Dcurr = np.zeros((ns,), dtype=float)
    Dwell = np.zeros((ns,), dtype=float)
    Dgeod = np.zeros((ns,), dtype=float)
    D_R = np.zeros((ns,), dtype=float)
    H_glasser = np.zeros((ns,), dtype=float)
    glasser_correction = np.zeros((ns,), dtype=float)
    glasser_shear_valid = np.zeros((ns,), dtype=bool)
    jdotb = np.zeros((ns,), dtype=float)
    bdotb = np.zeros((ns,), dtype=float)
    bdotgradv = np.zeros((ns,), dtype=float)

    if wout_light:
        return WoutScalarDiagnostics(
            betatotal=float(betatotal),
            betapol=betapol,
            betator=betator,
            betaxis=betaxis,
            ctor=ctor,
            DMerc=DMerc,
            Dshear=Dshear,
            Dcurr=Dcurr,
            Dwell=Dwell,
            Dgeod=Dgeod,
            D_R=D_R,
            H_glasser=H_glasser,
            glasser_correction=glasser_correction,
            glasser_shear_valid=glasser_shear_valid,
            jdotb=jdotb,
            bdotb=bdotb,
            bdotgradv=bdotgradv,
        )

    try:
        bsubu_merc = np.asarray(bsubu_diag, dtype=float)
        bsubv_merc = np.asarray(bsubv_diag, dtype=float)
        if env_enabled(os.getenv("VMEC_JAX_MERCIER_USE_RAW_BSUBUV", "")):
            bsubu_merc = np.asarray(bsubu_raw, dtype=float)
            bsubv_merc = np.asarray(bsubv_raw, dtype=float)
        elif env_enabled(os.getenv("VMEC_JAX_MERCIER_USE_WROUT_BSUBUV", "")):
            bsubu_merc = np.asarray(bsubu_phys, dtype=float)
            bsubv_merc = np.asarray(bsubv_phys, dtype=float)
        elif env_enabled(os.getenv("VMEC_JAX_MERCIER_USE_RAW_BSUBV", "")):
            bsubv_merc = np.asarray(bsubv_raw, dtype=float)

        if timing_enabled:
            import time as _time

            t_beta = _time.perf_counter()
        wint = vmec_wint_from_trig_func(trig)
        betaxis = compute_eqfor_betaxis_func(
            pres=np.asarray(pres, dtype=float),
            vp=np.asarray(vp, dtype=float),
            bsq=np.asarray(bc.bsq, dtype=float),
            sqrtg=np.asarray(bc.jac.sqrtg, dtype=float),
            wint=wint,
            signgs=int(signgs),
        )
        betapol, betator, betatot_eq, betaxis = compute_eqfor_beta_func(
            pres=np.asarray(pres, dtype=float),
            vp=np.asarray(vp, dtype=float),
            bsq=np.asarray(bc.bsq, dtype=float),
            r12=np.asarray(bc.jac.r12, dtype=float),
            bsupv=np.asarray(bc.bsupv, dtype=float),
            sqrtg=np.asarray(bc.jac.sqrtg, dtype=float),
            wint=wint,
            signgs=int(signgs),
        )
        if timing_enabled:
            timing["beta_s"] = _time.perf_counter() - t_beta
        betatotal = float(betatot_eq)
        ctor = compute_ctor_from_buco_func(buco=np.asarray(buco, dtype=float), signgs=int(signgs), indata=indata)

        if timing_enabled:
            t_mercier = _time.perf_counter()
        (
            DMerc,
            Dshear,
            Dcurr,
            Dwell,
            Dgeod,
            jdotb,
            bdotb,
            bdotgradv,
        ) = compute_mercier_func(
            state=state,
            geom_modes=static.modes,
            s=np.asarray(s, dtype=float),
            lconm1=bool(lconm1),
            lthreed=bool(ntor > 0),
            lasym=bool(lasym),
            nfp=int(nfp),
            lbsubs=bool(lbsubs),
            mmax_force=max(int(mpol) - 1, 0),
            nmax_force=int(ntor),
            pres=np.asarray(pres, dtype=float),
            vp=np.asarray(vp, dtype=float),
            phips=np.asarray(flux_phips, dtype=float),
            iotas=np.asarray(iotas, dtype=float),
            bsq=np.asarray(bc.bsq, dtype=float),
            sqrtg=np.asarray(bc.jac.sqrtg, dtype=float),
            bsubu=bsubu_merc,
            bsubv=bsubv_merc,
            bsubu_parity_even=(
                None
                if getattr(bc, "bsubu_parity_even", None) is None
                else np.asarray(getattr(bc, "bsubu_parity_even"), dtype=float)
            ),
            bsubu_parity_odd=(
                None
                if getattr(bc, "bsubu_parity_odd", None) is None
                else np.asarray(getattr(bc, "bsubu_parity_odd"), dtype=float)
            ),
            bsubv_parity_even=(
                None
                if getattr(bc, "bsubv_parity_even", None) is None
                else np.asarray(getattr(bc, "bsubv_parity_even"), dtype=float)
            ),
            bsubv_parity_odd=(
                None
                if getattr(bc, "bsubv_parity_odd", None) is None
                else np.asarray(getattr(bc, "bsubv_parity_odd"), dtype=float)
            ),
            bsupu=np.asarray(bsupu_bss, dtype=float),
            bsupv=np.asarray(bsupv_bss, dtype=float),
            trig=trig,
            geom=geom_bss,
            jac_half=bc.jac,
            force_rs=rs_bss,
            force_zs=zs_bss,
            force_ru12=ru12_bss,
            force_zu12=zu12_bss,
            bsubu_raw=np.asarray(bsubu_raw, dtype=float),
            bsubv_raw=np.asarray(bsubv_raw, dtype=float),
            signgs=int(signgs),
        )
        D_R, H_glasser, glasser_correction, glasser_shear_valid = glasser_from_wout_mercier_terms_func(
            DMerc=DMerc,
            Dshear=Dshear,
            Dcurr=Dcurr,
        )
        if timing_enabled:
            timing["mercier_s"] = _time.perf_counter() - t_mercier
    except Exception:
        if env_enabled(os.getenv("VMEC_JAX_STRICT_WOUT_DIAGNOSTICS", "")):
            raise

    return WoutScalarDiagnostics(
        betatotal=float(betatotal),
        betapol=float(betapol),
        betator=float(betator),
        betaxis=float(betaxis),
        ctor=float(ctor),
        DMerc=np.asarray(DMerc, dtype=float),
        Dshear=np.asarray(Dshear, dtype=float),
        Dcurr=np.asarray(Dcurr, dtype=float),
        Dwell=np.asarray(Dwell, dtype=float),
        Dgeod=np.asarray(Dgeod, dtype=float),
        D_R=np.asarray(D_R, dtype=float),
        H_glasser=np.asarray(H_glasser, dtype=float),
        glasser_correction=np.asarray(glasser_correction, dtype=float),
        glasser_shear_valid=np.asarray(glasser_shear_valid, dtype=bool),
        jdotb=np.asarray(jdotb, dtype=float),
        bdotb=np.asarray(bdotb, dtype=float),
        bdotgradv=np.asarray(bdotgradv, dtype=float),
    )


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
    main_geom = context["main_geom"]
    scalar_diag = context["scalar_diag"]
    current_metadata = context["current_metadata"]

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
        "rmnc": np.asarray(main_geom.rmnc, dtype=float),
        "rmns": np.asarray(main_geom.rmns, dtype=float),
        "zmnc": np.asarray(main_geom.zmnc, dtype=float),
        "zmns": np.asarray(main_geom.zmns, dtype=float),
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
        "raxis_cc": np.asarray(main_geom.raxis_cc, dtype=float),
        "zaxis_cs": np.asarray(main_geom.zaxis_cs, dtype=float),
        "raxis_cs": np.asarray(main_geom.raxis_cs, dtype=float),
        "zaxis_cc": np.asarray(main_geom.zaxis_cc, dtype=float),
        "Aminor_p": float(context["Aminor_p"]),
        "Rmajor_p": float(context["Rmajor_p"]),
        "aspect": float(context["aspect"]),
        "betatotal": float(scalar_diag.betatotal),
        "betapol": float(scalar_diag.betapol),
        "betator": float(scalar_diag.betator),
        "betaxis": float(scalar_diag.betaxis),
        "ctor": float(scalar_diag.ctor),
        "DMerc": np.asarray(scalar_diag.DMerc, dtype=float),
        "Dshear": np.asarray(scalar_diag.Dshear, dtype=float),
        "Dwell": np.asarray(scalar_diag.Dwell, dtype=float),
        "Dcurr": np.asarray(scalar_diag.Dcurr, dtype=float),
        "Dgeod": np.asarray(scalar_diag.Dgeod, dtype=float),
        "D_R": np.asarray(scalar_diag.D_R, dtype=float),
        "H": np.asarray(scalar_diag.H_glasser, dtype=float),
        "glasser_correction": np.asarray(scalar_diag.glasser_correction, dtype=float),
        "glasser_shear_valid": np.asarray(scalar_diag.glasser_shear_valid, dtype=bool),
        "jdotb": np.asarray(scalar_diag.jdotb, dtype=float),
        "bdotb": np.asarray(scalar_diag.bdotb, dtype=float),
        "bdotgradv": np.asarray(scalar_diag.bdotgradv, dtype=float),
        "ac": np.asarray(current_metadata.ac, dtype=float),
        "ac_aux_s": np.asarray(current_metadata.ac_aux_s, dtype=float),
        "ac_aux_f": np.asarray(current_metadata.ac_aux_f, dtype=float),
        "pcurr_type": str(current_metadata.pcurr_type),
        "piota_type": str(current_metadata.piota_type),
        "ier_flag": 0 if converged_bool else 1,
        "vmec_jax_converged": converged_bool,
        "vmec_jax_status": "converged" if converged_bool else "nonconverged",
    }
