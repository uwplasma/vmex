"""Assembly helpers for VMEC-compatible minimal ``wout`` output.

The public constructor remains :func:`vmec_jax.wout.wout_minimal_from_fixed_boundary`.
This module keeps passive data-shaping pieces out of that high-level routine so
the delicate diagnostic assembly is easier to review and test.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Mapping, NamedTuple

import numpy as np

from ..._compat import has_jax, jax
from ...integrals import cumrect_s_halfmesh
from ...modes import nyquist_mode_table_from_grid, vmec_mode_table
from ...namelist import InData
from ...vmec_parity import vmec_m1_internal_to_physical_signed_host
from ...vmec_tomnsp import vmec_trig_tables


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


class WoutMinimalCorePayload(NamedTuple):
    """Input, mode, profile, and geometry setup for minimal WOUT assembly."""

    cfg: Any
    ns: int
    mpol: int
    ntor: int
    nfp: int
    lasym: bool
    runtime_options: WoutMinimalRuntimeOptions
    wout_timing_enabled: bool
    wout_light: bool
    wout_fast_bcovar: bool
    field_options: WoutMinimalFieldOptions
    converged: bool
    lbsubs: bool
    main_modes: Any
    nyq_modes: Any
    trig: Any
    geom: dict[str, Any]
    s: np.ndarray
    flux: Any
    chipf_wout: np.ndarray
    pres: np.ndarray
    mass: np.ndarray
    ncurr: int
    iotas: np.ndarray
    iotaf: np.ndarray
    gamma: float
    phipf_internal: np.ndarray
    lconm1: bool
    main_geom: WoutMainGeometryCoefficients
    phipf_out: np.ndarray
    chipf_out: np.ndarray
    phi: np.ndarray
    wout_like: Any


class WoutForceSourcePayload(NamedTuple):
    """Bcovar and BSS source arrays used by WOUT field diagnostics."""

    bc: Any
    k_force: Any | None
    indata_wout: Any
    use_force_bss: bool
    bsupu_bss: np.ndarray
    bsupv_bss: np.ndarray
    ru12_bss: np.ndarray | None
    zu12_bss: np.ndarray | None
    rs_bss: np.ndarray | None
    zs_bss: np.ndarray | None
    crmn_e_sym: np.ndarray | None
    czmn_e_sym: np.ndarray | None
    bzmn_e_sym: np.ndarray | None
    brmn_e_sym: np.ndarray | None
    azmn_e_sym: np.ndarray | None
    armn_e_sym: np.ndarray | None
    geom_bss: dict[str, Any]


class WoutDerivedProfilePayload(NamedTuple):
    """Radial profiles and global geometry scalars derived from bcovar."""

    vp: np.ndarray
    wb: float
    wp: float
    volume: float
    volume_p: float
    betatotal: float
    pres: np.ndarray
    presf: np.ndarray
    wint: np.ndarray
    Aminor_p: float
    Rmajor_p: float
    aspect: float


class WoutNyquistFieldPayload(NamedTuple):
    """Nyquist field-output coefficients and diagnostic real-space fields."""

    bsupu_out: np.ndarray
    bsupv_out: np.ndarray
    bsubu_out: np.ndarray
    bsubv_out: np.ndarray
    bsubu_raw: np.ndarray
    bsubv_raw: np.ndarray
    bsubu_diag: np.ndarray
    bsubv_diag: np.ndarray
    bsubu_phys: np.ndarray | None
    bsubv_phys: np.ndarray | None
    bsubs_full: np.ndarray
    gmnc: np.ndarray
    gmns: np.ndarray
    bsupumnc: np.ndarray
    bsupumns: np.ndarray
    bsupvmnc: np.ndarray
    bsupvmns: np.ndarray
    bsubumnc: np.ndarray
    bsubumns: np.ndarray
    bsubvmnc: np.ndarray
    bsubvmns: np.ndarray
    bsubsmns: np.ndarray
    bsubsmnc: np.ndarray
    bmnc: np.ndarray
    bmns: np.ndarray


class WoutNyquistSourcePayload(NamedTuple):
    """Initial Bsub/Bsup sources and full-mesh Bsubs used for Nyquist output."""

    bsupu_out: np.ndarray
    bsupv_out: np.ndarray
    bsubu_out: np.ndarray
    bsubv_out: np.ndarray
    bsubu_raw: np.ndarray
    bsubv_raw: np.ndarray
    bsubu_diag: np.ndarray
    bsubv_diag: np.ndarray
    bsubv_lasym_asym_source: np.ndarray | None
    bsubs_full: np.ndarray


class WoutNyquistCoefficientSelection(NamedTuple):
    """Selected Nyquist coefficients and filtered Bsub diagnostics."""

    nyq: tuple[Any, ...]
    bsubu_out: np.ndarray
    bsubv_out: np.ndarray
    bsubu_diag: np.ndarray
    bsubv_diag: np.ndarray


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


def filter_symmetric_bsubuv_diagnostics_for_wout(
    *,
    bsubu_diag: np.ndarray,
    bsubv_diag: np.ndarray,
    bc: Any,
    trig: Any,
    field_options: WoutMinimalFieldOptions,
    mpol: int,
    ntor: int,
    s: np.ndarray,
    pshalf_from_s_func,
    filter_loop_func,
    filter_parity_func,
    dump_parity_inputs_func,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply VMEC jxbforce filtering for stellarator-symmetric diagnostics."""
    if field_options.skip_bsub_filter:
        return np.asarray(bsubu_diag, dtype=float), np.asarray(bsubv_diag, dtype=float)

    s_arr = np.asarray(s, dtype=float)
    if field_options.filter_from_raw:
        return filter_loop_func(
            bsubu=np.asarray(bsubu_diag, dtype=float),
            bsubv=np.asarray(bsubv_diag, dtype=float),
            trig=trig,
            mmax_force=max(int(mpol) - 1, 0),
            nmax_force=int(ntor),
            s=s_arr,
        )

    psh = pshalf_from_s_func(s_arr)[:, None, None]
    if psh.shape[0] > 1:
        psh[0] = psh[1]
    use_bc_parity = field_options.bsub_filter_use_bc_parity
    if use_bc_parity and getattr(bc, "bsubu_parity_even", None) is not None:
        bsubu_even = np.asarray(getattr(bc, "bsubu_parity_even"), dtype=float)
        bsubv_even = np.asarray(getattr(bc, "bsubv_parity_even"), dtype=float)
        bsubu_odd = np.asarray(getattr(bc, "bsubu_parity_odd"), dtype=float)
        bsubv_odd = np.asarray(getattr(bc, "bsubv_parity_odd"), dtype=float)
    else:
        bsubu_even = np.asarray(bsubu_diag, dtype=float)
        bsubv_even = np.asarray(bsubv_diag, dtype=float)
        bsubu_odd = psh * bsubu_even
        bsubv_odd = psh * bsubv_even
    dump_parity_inputs_func(
        bsubu_diag=bsubu_diag,
        bsubv_diag=bsubv_diag,
        bsubu_even=bsubu_even,
        bsubu_odd=bsubu_odd,
        bsubv_even=bsubv_even,
        bsubv_odd=bsubv_odd,
        use_bc_parity=bool(use_bc_parity),
    )
    return filter_parity_func(
        bsubu_even=np.asarray(bsubu_even, dtype=float),
        bsubu_odd=np.asarray(bsubu_odd, dtype=float),
        bsubv_even=np.asarray(bsubv_even, dtype=float),
        bsubv_odd=np.asarray(bsubv_odd, dtype=float),
        trig=trig,
        mmax_force=max(int(mpol) - 1, 0),
        nmax_force=int(ntor),
        s=s_arr,
    )


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


def prepare_minimal_wout_core_payload(
    *,
    state: Any,
    static: Any,
    indata: Any,
    signgs: int,
    converged: bool | None,
    flux_override: Any | None,
    profiles_override: dict | None,
    runtime_options: WoutMinimalRuntimeOptions,
    field_options: WoutMinimalFieldOptions,
    synthesize_geometry_func: Any,
    timing: dict[str, float],
    equilibrium_iota_profiles_from_state_func: Any,
    chipf_from_chips_func: Any,
    icurv_full_mesh_from_indata_func: Any,
) -> WoutMinimalCorePayload:
    """Prepare the passive state needed before WOUT diagnostic assembly.

    The root WOUT constructor still orchestrates physics diagnostics and file
    schema assembly. This helper owns deterministic input/mode/profile setup so
    those pieces stay in the WOUT domain package instead of the compatibility
    facade.
    """

    cfg = static.cfg
    ns = int(cfg.ns)
    mpol = int(cfg.mpol)
    ntor = int(cfg.ntor)
    nfp = int(cfg.nfp)
    lasym = bool(cfg.lasym)
    wout_timing_enabled = bool(runtime_options.timing_enabled)
    wout_light = bool(runtime_options.light)
    wout_fast_bcovar = bool(runtime_options.fast_bcovar)
    converged_out = True if converged is None else bool(converged)

    lbsubs = lbsubs_from_indata_and_env(indata)
    main_modes = vmec_mode_table(mpol, ntor)
    if int(main_modes.K) != int(state.layout.K):
        raise ValueError("state mode count does not match vmec_mode_table(mpol,ntor)")

    nyq_modes = nyquist_mode_table_from_grid(
        mpol=mpol,
        ntor=ntor,
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
    )
    mmax_nyq = int(np.max(nyq_modes.m)) if int(nyq_modes.K) > 0 else 0
    nmax_nyq = int(np.max(np.abs(nyq_modes.n))) if int(nyq_modes.K) > 0 else 0
    mmax_base = max(int(mpol) - 1, mmax_nyq)
    nmax_base = max(int(ntor), nmax_nyq)

    t0 = time.perf_counter() if wout_timing_enabled else None
    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(nfp),
        mmax=int(mmax_base),
        nmax=int(nmax_base),
        lasym=bool(lasym),
        dtype=np.asarray(state.Rcos).dtype,
    )
    if t0 is not None:
        timing["trig_tables_s"] = time.perf_counter() - t0

    geom = synthesize_geometry_func(
        state=state,
        static=static,
        trig=trig,
        light=bool(wout_light),
        timing_enabled=bool(wout_timing_enabled),
        timing=timing,
    )

    s = np.asarray(static.s)
    profile_payload = prepare_profile_payload(
        state=state,
        static=static,
        indata=indata,
        modes=main_modes,
        s=s,
        ns=int(ns),
        signgs=int(signgs),
        flux_override=flux_override,
        profiles_override=profiles_override,
        equilibrium_iota_profiles_from_state_func=equilibrium_iota_profiles_from_state_func,
        chipf_from_chips_func=chipf_from_chips_func,
    )
    (flux, chipf_wout, _, pres, _, mass, ncurr, iotas, iotaf, gamma, phipf_internal) = profile_payload

    lconm1 = bool(getattr(cfg, "lconm1", True))
    main_geom = build_main_geometry_coefficients(
        state=state,
        modes=main_modes,
        ntor=int(ntor),
        lasym=bool(lasym),
        lconm1=bool(lconm1),
    )

    phipf_out = np.asarray(phipf_internal, dtype=float) * float(2.0 * np.pi * signgs)
    chipf_out = np.asarray(chipf_wout, dtype=float) * float(2.0 * np.pi * signgs)
    phi = np.asarray(cumrect_s_halfmesh(phipf_out, s))
    wout_like = WoutMinimalVmecLike(
        flux=flux,
        chipf=np.asarray(chipf_wout),
        iotaf=np.asarray(iotaf),
        iotas=np.asarray(iotas),
        signgs=int(signgs),
        nfp=int(nfp),
        mpol=int(mpol),
        ntor=int(ntor),
        lasym=bool(lasym),
        ncurr=int(ncurr),
        mass=np.asarray(mass),
        gamma=float(gamma),
        indata=indata,
        s_full=np.asarray(s, dtype=float),
        icurv_full_mesh_from_indata_func=icurv_full_mesh_from_indata_func,
    )

    return WoutMinimalCorePayload(
        cfg=cfg,
        ns=int(ns),
        mpol=int(mpol),
        ntor=int(ntor),
        nfp=int(nfp),
        lasym=bool(lasym),
        runtime_options=runtime_options,
        wout_timing_enabled=bool(wout_timing_enabled),
        wout_light=bool(wout_light),
        wout_fast_bcovar=bool(wout_fast_bcovar),
        field_options=field_options,
        converged=bool(converged_out),
        lbsubs=bool(lbsubs),
        main_modes=main_modes,
        nyq_modes=nyq_modes,
        trig=trig,
        geom=geom,
        s=np.asarray(s),
        flux=flux,
        chipf_wout=np.asarray(chipf_wout),
        pres=np.asarray(pres),
        mass=np.asarray(mass),
        ncurr=int(ncurr),
        iotas=np.asarray(iotas),
        iotaf=np.asarray(iotaf),
        gamma=float(gamma),
        phipf_internal=np.asarray(phipf_internal),
        lconm1=bool(lconm1),
        main_geom=main_geom,
        phipf_out=np.asarray(phipf_out),
        chipf_out=np.asarray(chipf_out),
        phi=np.asarray(phi),
        wout_like=wout_like,
    )


def prepare_minimal_wout_force_sources(
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
    lasym: bool,
    trig: Any,
    vmec_bcovar_half_mesh_from_wout_func: Any,
    vmec_forces_rz_from_wout_func: Any,
    numpy_module_patch_func: Any,
    force_sym_func: Any,
    dump_bsub_parity_func: Any,
    dump_bsubh_func: Any,
) -> WoutForceSourcePayload:
    """Resolve bcovar/BSS sources before Nyquist field-output assembly."""

    bcovar_payload = prepare_wout_bcovar_payload(
        state=state,
        static=static,
        indata=indata,
        wout_like=wout_like,
        pres=pres,
        geom=geom,
        force_payload_override=force_payload_override,
        fast_bcovar=bool(fast_bcovar),
        timing_enabled=bool(timing_enabled),
        timing=timing,
        vmec_bcovar_half_mesh_from_wout_func=vmec_bcovar_half_mesh_from_wout_func,
        vmec_forces_rz_from_wout_func=vmec_forces_rz_from_wout_func,
        numpy_module_patch_func=numpy_module_patch_func,
    )
    bc, k_force, indata_wout = bcovar_payload
    bss_payload = prepare_wout_bss_source_payload(
        state=state,
        static=static,
        indata_wout=indata_wout,
        wout_like=wout_like,
        bc=bc,
        k_force=k_force,
        trig=trig,
        geom=geom,
        lasym=bool(lasym),
        force_sym_func=force_sym_func,
        vmec_forces_rz_from_wout_func=vmec_forces_rz_from_wout_func,
    )
    (
        use_force_bss,
        k_force,
        bsupu_bss,
        bsupv_bss,
        ru12_bss,
        zu12_bss,
        rs_bss,
        zs_bss,
        crmn_e_sym,
        czmn_e_sym,
        bzmn_e_sym,
        brmn_e_sym,
        azmn_e_sym,
        armn_e_sym,
        geom_bss,
    ) = bss_payload
    dump_bsub_parity_func(bc=bc)
    dump_bsubh_func(bsupu=bsupu_bss, bsupv=bsupv_bss, bc=bc)
    return WoutForceSourcePayload(
        bc=bc,
        k_force=k_force,
        indata_wout=indata_wout,
        use_force_bss=bool(use_force_bss),
        bsupu_bss=np.asarray(bsupu_bss, dtype=float),
        bsupv_bss=np.asarray(bsupv_bss, dtype=float),
        ru12_bss=ru12_bss,
        zu12_bss=zu12_bss,
        rs_bss=rs_bss,
        zs_bss=zs_bss,
        crmn_e_sym=crmn_e_sym,
        czmn_e_sym=czmn_e_sym,
        bzmn_e_sym=bzmn_e_sym,
        brmn_e_sym=brmn_e_sym,
        azmn_e_sym=azmn_e_sym,
        armn_e_sym=armn_e_sym,
        geom_bss=geom_bss,
    )


def compute_minimal_wout_derived_profiles(
    *,
    bc: Any,
    trig: Any,
    s: np.ndarray,
    signgs: int,
    mass: np.ndarray,
    gamma: float,
    geom: dict[str, Any],
    vmec_force_norms_from_bcovar_dynamic_func: Any,
    vmec_wint_from_trig_func: Any,
    compute_aspectratio_func: Any,
) -> WoutDerivedProfilePayload:
    """Compute radial force norms, pressure profiles, and aspect ratio."""

    norms = vmec_force_norms_from_bcovar_dynamic_func(
        bc=bc,
        trig=trig,
        s=s,
        signgs=int(signgs),
    )
    norms = device_get_if_available(norms)
    vp = np.asarray(norms.vp, dtype=float)
    wb = float(np.asarray(norms.wb))
    wp = float(np.asarray(norms.wp))
    volume = float(np.asarray(norms.volume))
    betatotal = (wp / wb) if wb != 0.0 else 0.0
    pres, presf = pressure_profiles_from_mass_vp(mass=mass, vp=vp, gamma=gamma)
    wint = vmec_wint_from_trig_func(trig)
    Aminor_p, Rmajor_p, aspect, volume_p, _ = compute_aspectratio_func(
        R=np.asarray(geom["R"]),
        Zu=np.asarray(geom["Zu"]),
        wint=wint,
    )
    return WoutDerivedProfilePayload(
        vp=np.asarray(vp, dtype=float),
        wb=float(wb),
        wp=float(wp),
        volume=float(volume),
        volume_p=float(volume_p),
        betatotal=float(betatotal),
        pres=np.asarray(pres, dtype=float),
        presf=np.asarray(presf, dtype=float),
        wint=np.asarray(wint, dtype=float),
        Aminor_p=float(Aminor_p),
        Rmajor_p=float(Rmajor_p),
        aspect=float(aspect),
    )


def _prepare_nyquist_sources_and_bsubs(
    *,
    state: Any,
    static: Any,
    cfg: Any,
    bc: Any,
    k_force: Any | None,
    use_force_bss: bool,
    bsupu_bss: np.ndarray,
    bsupv_bss: np.ndarray,
    ru12_bss: np.ndarray | None,
    zu12_bss: np.ndarray | None,
    rs_bss: np.ndarray | None,
    zs_bss: np.ndarray | None,
    crmn_e_sym: np.ndarray | None,
    czmn_e_sym: np.ndarray | None,
    geom_bss: dict[str, Any],
    field_options: WoutMinimalFieldOptions,
    trig: Any,
    s: np.ndarray,
    ntor: int,
    lasym: bool,
    timing_enabled: bool,
    timing: dict[str, float],
    force_sym_func: Any,
    apply_bsubv_equif_correction_func: Any,
    compute_bsubs_half_mesh_func: Any,
    bsubs_full_mesh_for_wrout_func: Any,
    dump_bsub_sources_func: Any,
) -> WoutNyquistSourcePayload:
    """Select raw/diagnostic field sources and compute full-mesh Bsubs."""

    bsupu_out = np.asarray(bc.bsupu)
    bsupv_out = np.asarray(bc.bsupv)
    if use_force_bss and (k_force is not None) and hasattr(k_force, "crmn_e") and hasattr(k_force, "czmn_e"):
        bsupu_out = crmn_e_sym if crmn_e_sym is not None else force_sym_func(k_force.crmn_e, "crs")
        bsupv_out = czmn_e_sym if czmn_e_sym is not None else force_sym_func(k_force.czmn_e, "czs")

    bsubu_out = np.asarray(bc.bsubu).copy()
    bsubv_out = np.asarray(bc.bsubv).copy()
    bsubu_raw = bsubu_out.copy()
    bsubv_raw = bsubv_out.copy()
    bsubv_lasym_asym_source = None
    if bool(lasym) and hasattr(bc, "bsubv_e"):
        bsubv_lasym_asym_source = apply_bsubv_equif_correction_func(
            bsubv=np.asarray(getattr(bc, "bsubv"), dtype=float),
            bsubv_e=np.asarray(getattr(bc, "bsubv_e"), dtype=float),
            trig=trig,
        )
    dump_bsub_sources_func(bc=bc)

    bsubu_diag, bsubv_diag = select_bsubuv_diagnostic_fields(
        bc=bc,
        bsubu_out=bsubu_out,
        bsubv_out=bsubv_out,
        field_options=field_options,
        trig=trig,
        apply_bsubv_equif_correction_func=apply_bsubv_equif_correction_func,
    )
    t_bsubs = time.perf_counter() if timing_enabled else None
    bsubs_half = compute_bsubs_half_mesh_func(
        state=state,
        geom_modes=static.modes,
        s=np.asarray(s, dtype=float),
        lconm1=bool(getattr(cfg, "lconm1", True)),
        lthreed=bool(ntor > 0),
        lasym=bool(lasym),
        bsupu=bsupu_bss,
        bsupv=bsupv_bss,
        trig=trig,
        geom=geom_bss,
        jac_half=bc.jac,
        force_rs=rs_bss,
        force_zs=zs_bss,
        force_ru12=ru12_bss,
        force_zu12=zu12_bss,
        apply_scalxc=field_options.apply_bss_scalxc,
    )
    if t_bsubs is not None:
        timing["bsubs_half_s"] = time.perf_counter() - t_bsubs
    return WoutNyquistSourcePayload(
        bsupu_out=np.asarray(bsupu_out),
        bsupv_out=np.asarray(bsupv_out),
        bsubu_out=np.asarray(bsubu_out),
        bsubv_out=np.asarray(bsubv_out),
        bsubu_raw=np.asarray(bsubu_raw),
        bsubv_raw=np.asarray(bsubv_raw),
        bsubu_diag=np.asarray(bsubu_diag),
        bsubv_diag=np.asarray(bsubv_diag),
        bsubv_lasym_asym_source=bsubv_lasym_asym_source,
        bsubs_full=np.asarray(bsubs_full_mesh_for_wrout_func(bsubs_half=bsubs_half)),
    )


def _build_nyquist_field_payload(
    *,
    sources: WoutNyquistSourcePayload,
    bsubu_out: np.ndarray,
    bsubv_out: np.ndarray,
    bsubu_diag: np.ndarray,
    bsubv_diag: np.ndarray,
    bsubu_phys: np.ndarray | None,
    bsubv_phys: np.ndarray | None,
    nyq: tuple[Any, ...],
) -> WoutNyquistFieldPayload:
    """Package final Nyquist coefficients without obscuring the assembly path."""

    nyq_fields = {
        name: np.asarray(value)
        for name, value in zip(WoutNyquistFieldPayload._fields[11:], nyq, strict=True)
    }
    return WoutNyquistFieldPayload(
        bsupu_out=np.asarray(sources.bsupu_out),
        bsupv_out=np.asarray(sources.bsupv_out),
        bsubu_out=np.asarray(bsubu_out),
        bsubv_out=np.asarray(bsubv_out),
        bsubu_raw=np.asarray(sources.bsubu_raw),
        bsubv_raw=np.asarray(sources.bsubv_raw),
        bsubu_diag=np.asarray(bsubu_diag),
        bsubv_diag=np.asarray(bsubv_diag),
        bsubu_phys=bsubu_phys,
        bsubv_phys=bsubv_phys,
        bsubs_full=np.asarray(sources.bsubs_full),
        **nyq_fields,
    )


def _select_nyquist_coefficients(
    *,
    sources: WoutNyquistSourcePayload,
    bc: Any,
    k_force: Any | None,
    field_options: WoutMinimalFieldOptions,
    trig: Any,
    nyq_modes: Any,
    pres: np.ndarray,
    s: np.ndarray,
    mpol: int,
    ntor: int,
    ns: int,
    lasym: bool,
    force_sym_func: Any,
    filter_lasym_loop_func: Any,
    lasym_nyquist_coefficients_func: Any,
    symmetric_nyquist_coefficients_func: Any,
    dump_bsub_pre_sym_func: Any,
) -> WoutNyquistCoefficientSelection:
    """Apply VMEC wrout filtering and select Nyquist coefficient arrays."""

    bsupu_out = np.asarray(sources.bsupu_out)
    bsupv_out = np.asarray(sources.bsupv_out)
    bsubu_out = np.asarray(sources.bsubu_out)
    bsubv_out = np.asarray(sources.bsubv_out)
    bsubu_diag = np.asarray(sources.bsubu_diag)
    bsubv_diag = np.asarray(sources.bsubv_diag)
    bsubv_lasym_asym_source = sources.bsubv_lasym_asym_source
    if bool(lasym):
        use_lasym_loop = field_options.use_lasym_loop
        if (not field_options.skip_bsub_filter) and field_options.lasym_filter:
            use_parity_channels = field_options.lasym_filter_use_parity_channels
            bsubu_even_filter = getattr(bc, "bsubu_parity_even", None) if use_parity_channels else None
            bsubu_odd_filter = getattr(bc, "bsubu_parity_odd", None) if use_parity_channels else None
            bsubv_even_filter = getattr(bc, "bsubv_parity_even", None) if use_parity_channels else None
            bsubv_odd_filter = getattr(bc, "bsubv_parity_odd", None) if use_parity_channels else None
            bsubv_lasym_asym_filter_u = np.asarray(bsubu_out, dtype=float).copy()
            bsubu_out, bsubv_out = filter_lasym_loop_func(
                bsubu=np.asarray(bsubu_out, dtype=float),
                bsubv=np.asarray(bsubv_out, dtype=float),
                trig=trig,
                mmax_force=max(int(mpol) - 1, 0),
                nmax_force=int(ntor),
                s=np.asarray(s, dtype=float),
                bsubu_even=None if bsubu_even_filter is None else np.asarray(bsubu_even_filter, dtype=float),
                bsubu_odd=None if bsubu_odd_filter is None else np.asarray(bsubu_odd_filter, dtype=float),
                bsubv_even=None if bsubv_even_filter is None else np.asarray(bsubv_even_filter, dtype=float),
                bsubv_odd=None if bsubv_odd_filter is None else np.asarray(bsubv_odd_filter, dtype=float),
            )
            if bsubv_lasym_asym_source is not None:
                _, bsubv_lasym_asym_source = filter_lasym_loop_func(
                    bsubu=np.asarray(bsubv_lasym_asym_filter_u, dtype=float),
                    bsubv=np.asarray(bsubv_lasym_asym_source, dtype=float),
                    trig=trig,
                    mmax_force=max(int(mpol) - 1, 0),
                    nmax_force=int(ntor),
                    s=np.asarray(s, dtype=float),
                    bsubu_even=None,
                    bsubu_odd=None,
                    bsubv_even=None,
                    bsubv_odd=None,
                )
            bsubu_diag = np.asarray(bsubu_out, dtype=float)
            bsubv_diag = np.asarray(bsubv_out, dtype=float)
        dump_bsub_pre_sym_func(
            trig=trig,
            bsubu=bsubu_out,
            bsubv=bsubv_out,
            bsupu=bsupu_out,
            bsupv=bsupv_out,
            bsubs=sources.bsubs_full,
        )
        nyq = lasym_nyquist_coefficients_func(
            bc=bc,
            bsubu_out=np.asarray(bsubu_out, dtype=float),
            bsubv_out=np.asarray(bsubv_out, dtype=float),
            bsupu_out=np.asarray(bsupu_out, dtype=float),
            bsupv_out=np.asarray(bsupv_out, dtype=float),
            bsubs_full=np.asarray(sources.bsubs_full, dtype=float),
            bsubv_asym_source=bsubv_lasym_asym_source,
            pres=np.asarray(pres, dtype=float),
            ns=int(ns),
            mpol=int(mpol),
            ntor=int(ntor),
            modes=nyq_modes,
            trig=trig,
            use_loop=bool(use_lasym_loop),
        )
    else:
        nyq = symmetric_nyquist_coefficients_func(
            bc=bc,
            bsubu_out=np.asarray(bsubu_out, dtype=float),
            bsubv_out=np.asarray(bsubv_out, dtype=float),
            bsubs_full=np.asarray(sources.bsubs_full, dtype=float),
            pres=np.asarray(pres, dtype=float),
            ns=int(ns),
            modes=nyq_modes,
            trig=trig,
            use_loop=bool(field_options.symmetric_wrout_loop),
        )

    return WoutNyquistCoefficientSelection(
        nyq=tuple(nyq),
        bsubu_out=np.asarray(bsubu_out),
        bsubv_out=np.asarray(bsubv_out),
        bsubu_diag=np.asarray(bsubu_diag),
        bsubv_diag=np.asarray(bsubv_diag),
    )


def prepare_minimal_wout_nyquist_fields(
    *,
    state: Any,
    static: Any,
    cfg: Any,
    bc: Any,
    k_force: Any | None,
    use_force_bss: bool,
    bsupu_bss: np.ndarray,
    bsupv_bss: np.ndarray,
    ru12_bss: np.ndarray | None,
    zu12_bss: np.ndarray | None,
    rs_bss: np.ndarray | None,
    zs_bss: np.ndarray | None,
    crmn_e_sym: np.ndarray | None,
    czmn_e_sym: np.ndarray | None,
    geom_bss: dict[str, Any],
    field_options: WoutMinimalFieldOptions,
    trig: Any,
    nyq_modes: Any,
    pres: np.ndarray,
    s: np.ndarray,
    mpol: int,
    ntor: int,
    nfp: int,
    ns: int,
    lasym: bool,
    timing_enabled: bool,
    timing: dict[str, float],
    force_sym_func: Any,
    apply_bsubv_equif_correction_func: Any,
    compute_bsubs_half_mesh_func: Any,
    bsubs_full_mesh_for_wrout_func: Any,
    filter_lasym_loop_func: Any,
    filter_symmetric_loop_func: Any,
    filter_symmetric_parity_func: Any,
    pshalf_from_s_func: Any,
    lasym_nyquist_coefficients_func: Any,
    symmetric_nyquist_coefficients_func: Any,
    nyquist_cos_coeffs_func: Any,
    zero_first_surface_func: Any,
    eval_fourier_func: Any,
    build_helical_basis_func: Any,
    vmec_angle_grid_func: Any,
    dump_bsub_sources_func: Any,
    dump_bsub_pre_sym_func: Any,
    dump_bsub_parity_inputs_func: Any,
) -> WoutNyquistFieldPayload:
    """Assemble WOUT Nyquist fields and selected Bsub diagnostics."""

    t0 = time.perf_counter() if timing_enabled else None
    sources = _prepare_nyquist_sources_and_bsubs(
        state=state,
        static=static,
        cfg=cfg,
        bc=bc,
        k_force=k_force,
        use_force_bss=use_force_bss,
        bsupu_bss=bsupu_bss,
        bsupv_bss=bsupv_bss,
        ru12_bss=ru12_bss,
        zu12_bss=zu12_bss,
        rs_bss=rs_bss,
        zs_bss=zs_bss,
        crmn_e_sym=crmn_e_sym,
        czmn_e_sym=czmn_e_sym,
        geom_bss=geom_bss,
        field_options=field_options,
        trig=trig,
        s=s,
        ntor=int(ntor),
        lasym=bool(lasym),
        timing_enabled=bool(timing_enabled),
        timing=timing,
        force_sym_func=force_sym_func,
        apply_bsubv_equif_correction_func=apply_bsubv_equif_correction_func,
        compute_bsubs_half_mesh_func=compute_bsubs_half_mesh_func,
        bsubs_full_mesh_for_wrout_func=bsubs_full_mesh_for_wrout_func,
        dump_bsub_sources_func=dump_bsub_sources_func,
    )
    selected = _select_nyquist_coefficients(
        sources=sources,
        bc=bc,
        k_force=k_force,
        field_options=field_options,
        trig=trig,
        nyq_modes=nyq_modes,
        pres=pres,
        s=s,
        mpol=int(mpol),
        ntor=int(ntor),
        ns=int(ns),
        lasym=bool(lasym),
        force_sym_func=force_sym_func,
        filter_lasym_loop_func=filter_lasym_loop_func,
        lasym_nyquist_coefficients_func=lasym_nyquist_coefficients_func,
        symmetric_nyquist_coefficients_func=symmetric_nyquist_coefficients_func,
        dump_bsub_pre_sym_func=dump_bsub_pre_sym_func,
    )
    nyq = selected.nyq
    bsubu_out = np.asarray(selected.bsubu_out)
    bsubv_out = np.asarray(selected.bsubv_out)
    bsubu_diag = np.asarray(selected.bsubu_diag)
    bsubv_diag = np.asarray(selected.bsubv_diag)
    if t0 is not None:
        timing["nyquist_coeffs_s"] = time.perf_counter() - t0

    bsubu_phys = None
    bsubv_phys = None
    if field_options.mercier_use_wrout_bsubuv:
        grid_nyq = vmec_angle_grid_func(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(nfp),
            lasym=bool(lasym),
        )
        basis_nyq = build_helical_basis_func(nyq_modes, grid_nyq, cache=True)
        _, _, _, _, _, _, bsubumnc, bsubumns, bsubvmnc, bsubvmns, *_ = nyq
        bsubu_phys = np.asarray(eval_fourier_func(bsubumnc, bsubumns, basis_nyq))
        bsubv_phys = np.asarray(eval_fourier_func(bsubvmnc, bsubvmns, basis_nyq))

    t_bsub_filter = time.perf_counter() if timing_enabled else None
    if (not bool(lasym)) and (not field_options.skip_bsub_filter):
        bsubu_diag, bsubv_diag = filter_symmetric_bsubuv_diagnostics_for_wout(
            bsubu_diag=bsubu_diag,
            bsubv_diag=bsubv_diag,
            bc=bc,
            trig=trig,
            field_options=field_options,
            mpol=int(mpol),
            ntor=int(ntor),
            s=np.asarray(s, dtype=float),
            pshalf_from_s_func=pshalf_from_s_func,
            filter_loop_func=filter_symmetric_loop_func,
            filter_parity_func=filter_symmetric_parity_func,
            dump_parity_inputs_func=dump_bsub_parity_inputs_func,
        )
    if t_bsub_filter is not None:
        timing["bsub_filter_s"] = time.perf_counter() - t_bsub_filter

    bsubu_out = np.asarray(bsubu_diag, dtype=float)
    bsubv_out = np.asarray(bsubv_diag, dtype=float)
    t_bsub_coeffs = time.perf_counter() if timing_enabled else None
    if not bool(lasym):
        nyq_list = list(nyq)
        bsubumnc = nyquist_cos_coeffs_func(f=bsubu_out, modes=nyq_modes, trig=trig)
        bsubvmnc = nyquist_cos_coeffs_func(f=bsubv_out, modes=nyq_modes, trig=trig)
        zero_first_surface_func(bsubumnc, bsubvmnc)
        nyq_list[6] = bsubumnc
        nyq_list[8] = bsubvmnc
        nyq = tuple(nyq_list)
    if t_bsub_coeffs is not None:
        timing["bsub_coeffs_s"] = time.perf_counter() - t_bsub_coeffs

    return _build_nyquist_field_payload(
        sources=sources,
        bsubu_out=bsubu_out,
        bsubv_out=bsubv_out,
        bsubu_diag=bsubu_diag,
        bsubv_diag=bsubv_diag,
        bsubu_phys=bsubu_phys,
        bsubv_phys=bsubv_phys,
        nyq=nyq,
    )


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
