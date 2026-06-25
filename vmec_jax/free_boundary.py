"""Free-boundary typed config/state and mgrid loader skeleton.

WP0 scope:
- typed runtime state container for free-boundary iteration control,
- deterministic mgrid metadata/field loader,
- lightweight validation hooks that mirror VMEC2000 readin/read_indata behavior.
"""

from __future__ import annotations

from dataclasses import replace
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import VMECConfig
from .solvers.free_boundary.axis_current import (
    axis_current_field_simple as _axis_current_field_simple,
    axis_current_field_vmec_filament as _axis_current_field_vmec_filament,
)
from .solvers.free_boundary.boundary_fields import (
    _FREEB_BOUNDARY_SETUP_CACHE,  # noqa: F401 - private compatibility facade
    _FREEB_HOST_PHASE_CACHE,  # noqa: F401 - private compatibility facade
    _FREEB_TRIG_CACHE,  # noqa: F401 - private compatibility facade
    _FREEB_WINT_CACHE,  # noqa: F401 - private compatibility facade
    boundary_metric_from_rz,  # noqa: F401 - public compatibility facade
    contravariant_boundary_field_from_covariant,
    covariant_boundary_field_from_cylindrical,  # noqa: F401 - public compatibility facade
    freeb_boundary_sample_setup as _freeb_boundary_sample_setup,
    freeb_boundary_trig as _freeb_boundary_trig,  # noqa: F401 - private compatibility facade
    freeb_host_phase_stack as _freeb_host_phase_stack,  # noqa: F401 - private compatibility facade
    sample_free_boundary_external_field,  # noqa: F401 - public compatibility facade
    vacuum_boundary_fields_from_cylindrical,
    vmec_boundary_wint as _vmec_boundary_wint,
    vmec_realspace_synthesis_multi_host as _vmec_realspace_synthesis_multi_host,
)
from .solvers.free_boundary.mgrid import (
    MGRID_FIELD_CACHE as _MGRID_FIELD_CACHE,
    broadcast_xyz as _broadcast_xyz,  # noqa: F401 - public compatibility facade
    decode_char_rows as _decode_char_rows,  # noqa: F401 - public compatibility facade
    decode_char_scalar as _decode_char_scalar,  # noqa: F401 - public compatibility facade
    interpolate_mgrid_bfield,
    load_mgrid,
    normalize_extcur as _normalize_extcur,  # noqa: F401 - public compatibility facade
    prepare_mgrid_for_config as _prepare_mgrid_for_config_impl,
    validate_free_boundary_config,
)
from .solvers.free_boundary.jax_nestor_operator import (
    FREEB_JAX_NESTOR_OPERATOR_FN_CACHE as _FREEB_JAX_NESTOR_OPERATOR_FN_CACHE,  # noqa: F401 - compatibility facade
    JAX_NESTOR_BASIS_KEYS as _JAX_NESTOR_BASIS_KEYS,  # noqa: F401 - compatibility facade
    build_poisson_cache as _build_poisson_cache,
    build_vmec_cmns as _build_vmec_cmns,  # noqa: F401 - private compatibility facade
    build_vmec_like_cache as _build_vmec_like_cache,
    build_vmec_mode_basis as _build_vmec_mode_basis,
    compact_jax_nestor_basis as _compact_jax_nestor_basis,  # noqa: F401 - compatibility facade
    dense_lu_factor as _dense_lu_factor,
    dense_lu_solve as _dense_lu_solve,  # noqa: F401 - private compatibility facade
    digest_array_for_cache as _digest_array_for_cache,  # noqa: F401 - compatibility facade
    ensure_vmec_nonsingular_kernel_tables as _ensure_vmec_nonsingular_kernel_tables,
    jax_nestor_input_signature as _jax_nestor_input_signature,  # noqa: F401 - compatibility facade
    jax_nestor_operator_cache_key as _jax_nestor_operator_cache_key,  # noqa: F401 - compatibility facade
    jax_nestor_operator_guard as _jax_nestor_operator_guard,
    jitted_jax_nestor_operator as _jitted_jax_nestor_operator,  # noqa: F401 - compatibility facade
    mapping_cache_signature as _mapping_cache_signature,  # noqa: F401 - compatibility facade
    solve_vmec_like_mode_from_gsource as _solve_vmec_like_mode_from_gsource,
    solve_vmec_like_dense as _solve_vmec_like_dense,
    solve_vmec_like_mode_with_jax_nestor_operator as _solve_vmec_like_mode_with_jax_nestor_operator_impl,
    spectral_second_derivatives_2d as _spectral_second_derivatives_2d,  # noqa: F401 - private compatibility facade
    vmec_analytic_bvec_from_geometry as _vmec_analytic_bvec_from_geometry,  # noqa: F401 - private compatibility facade
    vmec_analytic_terms_from_geometry as _vmec_analytic_terms_from_geometry,
    vmec_bvec_from_gsource as _vmec_bvec_from_gsource,
    vmec_mode_matrix_from_grpmn as _vmec_mode_matrix_from_grpmn,
    vmec_nonsingular_gsource_from_bexni as _vmec_nonsingular_gsource_from_bexni,
    vmec_nonsingular_terms_from_bexni as _vmec_nonsingular_terms_from_bexni,
    vmec_precal_tan_tables as _vmec_precal_tan_tables,  # noqa: F401 - private compatibility facade
    vmec_source_from_gsource as _vmec_source_from_gsource,
)
from .solvers.free_boundary.types import (
    ExternalBoundarySample,
    FreeBoundaryRuntimeState,
    FreeBoundarySampleSetup as _FreeBoundarySampleSetup,
    MGridData,
    MGridMetadata,  # noqa: F401 - public compatibility facade
    NestorPoissonCache,
    NestorRuntimeState,
    NestorSolveResult,
    NestorVmecLikeCache,
    PreparedMGrid,
    VacuumBoundaryFields,
)

def initial_free_boundary_state(cfg: VMECConfig) -> FreeBoundaryRuntimeState:
    """Initialize free-boundary control state for a VMEC stage."""

    nv = int(cfg.nvacskip)
    return FreeBoundaryRuntimeState(
        ivac=0,
        ivacskip=0,
        nvacskip=nv,
        nvskip0=max(1, nv),
    )


def _external_field_provider(
    *,
    static: Any,
    provider_kind_raw: str | None,
) -> tuple[str, bool, MGridData | None]:
    """Resolve the free-boundary external-field source."""

    provider_kind = "mgrid" if provider_kind_raw is None else str(provider_kind_raw).strip().lower()
    use_mgrid = provider_kind in ("", "mgrid", "legacy_mgrid")
    if not use_mgrid:
        return f"provider:{provider_kind}", False, None

    meta = getattr(static, "mgrid_metadata", None)
    if meta is None:
        raise ValueError("missing_mgrid_metadata")
    mgrid_path = str(getattr(meta, "path", "")).strip()
    if not mgrid_path:
        raise ValueError("missing_mgrid_path")

    mgrid = _MGRID_FIELD_CACHE.get(mgrid_path)
    if mgrid is None:
        loaded = load_mgrid(mgrid_path, load_fields=True)
        if not isinstance(loaded, MGridData):  # pragma: no cover
            raise TypeError("load_mgrid(load_fields=True) must return MGridData")
        mgrid = loaded
        _MGRID_FIELD_CACHE[mgrid_path] = mgrid
    return mgrid_path, True, mgrid


def _freeb_physical_coefficients(*, state: Any, static: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return free-boundary geometry coefficients after VMEC m=1 conversion."""

    if os.getenv("VMEC_JAX_FREEB_DISABLE_M1_CONVERSION", "").strip().lower() in ("1", "true", "yes"):
        return (
            np.asarray(state.Rcos),
            np.asarray(state.Zsin),
            np.asarray(state.Rsin),
            np.asarray(state.Zcos),
        )

    from .vmec_parity import vmec_m1_internal_to_physical_signed_host

    return vmec_m1_internal_to_physical_signed_host(
        Rcos=np.asarray(state.Rcos),
        Zsin=np.asarray(state.Zsin),
        Rsin=np.asarray(state.Rsin),
        Zcos=np.asarray(state.Zcos),
        modes=static.modes,
        lthreed=bool(getattr(static.cfg, "lthreed", True)),
        lasym=bool(getattr(static.cfg, "lasym", False)),
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
    )


def _freeb_boundary_geometry(
    *,
    Rcos_phys: np.ndarray,
    Zsin_phys: np.ndarray,
    Rsin_phys: np.ndarray,
    Zcos_phys: np.ndarray,
    static: Any,
    trig: Any,
    setup: _FreeBoundarySampleSetup,
) -> dict[str, np.ndarray]:
    """Synthesize boundary geometry and modal second derivatives."""

    boundary_cos = np.stack([np.asarray(Rcos_phys)[-1:, :], np.asarray(Zcos_phys)[-1:, :]], axis=0)
    boundary_sin = np.stack([np.asarray(Rsin_phys)[-1:, :], np.asarray(Zsin_phys)[-1:, :]], axis=0)
    boundary_all = _vmec_realspace_synthesis_multi_host(
        coeff_cos=boundary_cos,
        coeff_sin=boundary_sin,
        modes=static.modes,
        trig=trig,
        derivs=("base", "dtheta", "dzeta"),
    )
    second_cos = boundary_cos[:, None, :, :] * setup.second_facs[None, :, :, :]
    second_sin = boundary_sin[:, None, :, :] * setup.second_facs[None, :, :, :]
    second_all = np.asarray(
        _vmec_realspace_synthesis_multi_host(
            coeff_cos=second_cos,
            coeff_sin=second_sin,
            modes=static.modes,
            trig=trig,
            derivs=("base",),
        )[0]
    )
    return {
        "R": np.asarray(boundary_all[0][0, 0]),
        "Ru": np.asarray(boundary_all[1][0, 0]),
        "Rv": np.asarray(boundary_all[2][0, 0]),
        "Z": np.asarray(boundary_all[0][1, 0]),
        "Zu": np.asarray(boundary_all[1][1, 0]),
        "Zv": np.asarray(boundary_all[2][1, 0]),
        "Ruu": np.asarray(second_all[0, 0, 0]),
        "Ruv": np.asarray(second_all[0, 1, 0]),
        "Rvv": np.asarray(second_all[0, 2, 0]),
        "Zuu": np.asarray(second_all[1, 0, 0]),
        "Zuv": np.asarray(second_all[1, 1, 0]),
        "Zvv": np.asarray(second_all[1, 2, 0]),
    }


def _sample_external_provider_field(
    *,
    provider_kind: str,
    use_mgrid_provider: bool,
    mgrid: MGridData | None,
    external_field_provider_static: Any,
    external_field_provider_params: Any,
    R: np.ndarray,
    Z: np.ndarray,
    phi_grid: np.ndarray,
    extcur_eff: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample either a VMEC mgrid or a direct external-field provider."""

    if use_mgrid_provider:
        if mgrid is None:  # pragma: no cover
            raise ValueError("missing_mgrid_data")
        return interpolate_mgrid_bfield(
            mgrid,
            r=R,
            z=Z,
            phi=phi_grid,
            extcur=extcur_eff,
            use_vmec_kv=True,
        )

    from .external_fields import sample_external_field_cylindrical

    return tuple(
        np.asarray(arr, dtype=float)
        for arr in sample_external_field_cylindrical(
            provider_kind,
            external_field_provider_static,
            external_field_provider_params,
            R,
            Z,
            phi_grid,
        )
    )


def _freeb_synthesize_axis_pair(
    *,
    Rcos_phys: np.ndarray,
    Zsin_phys: np.ndarray,
    Rsin_phys: np.ndarray,
    Zcos_phys: np.ndarray,
    static: Any,
    trig: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback axis reconstruction from the axis Fourier coefficients."""

    from .vmec_realspace import vmec_realspace_synthesis

    axis_r = np.asarray(
        vmec_realspace_synthesis(
            coeff_cos=np.asarray(Rcos_phys)[:1, :],
            coeff_sin=np.asarray(Rsin_phys)[:1, :],
            modes=static.modes,
            trig=trig,
            coeffs_internal=True,
        )[0, 0, :]
    )
    axis_z = np.asarray(
        vmec_realspace_synthesis(
            coeff_cos=np.asarray(Zcos_phys)[:1, :],
            coeff_sin=np.asarray(Zsin_phys)[:1, :],
            modes=static.modes,
            trig=trig,
            coeffs_internal=True,
        )[0, 0, :]
    )
    return axis_r, axis_z


def _freeb_axis_from_vmec_pr1(*, state: Any, static: Any, trig: Any, axis_mode: str) -> tuple[np.ndarray, np.ndarray] | None:
    """VMEC-exact stellarator-symmetric raxis/zaxis reconstruction."""

    if bool(getattr(static.cfg, "lasym", False)) or axis_mode not in ("vmec_pr1", "pr1_vmec", "vmec"):
        return None
    try:
        from .vmec_parity import signed_maps_from_modes, _signed_to_mn_cos_host, _signed_to_mn_sin_host

        maps = signed_maps_from_modes(static.modes)
        rcc, _rss = _signed_to_mn_cos_host(np.asarray(state.Rcos), maps=maps)
        _zsc, zcs = _signed_to_mn_sin_host(np.asarray(state.Zsin), maps=maps)
        rcc_js1 = np.asarray(rcc[0], dtype=float)
        zcs_js1 = np.asarray(zcs[0], dtype=float)
        even_m = (np.arange(rcc_js1.shape[0], dtype=int) % 2) == 0
        rmncc_n = np.sum(rcc_js1[even_m, :], axis=0)
        zmncs_n = np.sum(zcs_js1[even_m, :], axis=0)
        nrange = int(rmncc_n.shape[0])
        return (
            np.asarray(np.asarray(trig.cosnv, dtype=float)[:, :nrange] @ rmncc_n, dtype=float),
            np.asarray(np.asarray(trig.sinnv, dtype=float)[:, :nrange] @ zmncs_n, dtype=float),
        )
    except Exception:
        return None


def _freeb_axis_from_parity(
    *,
    Rcos_phys: np.ndarray,
    Zsin_phys: np.ndarray,
    Rsin_phys: np.ndarray,
    Zcos_phys: np.ndarray,
    static: Any,
    trig: Any,
    setup: _FreeBoundarySampleSetup,
    dump_scalpot_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None:
    """Recover axis arrays from VMEC parity channels when requested."""

    try:
        from .vmec_realspace import vmec_realspace_synthesis

        axis_apply_scalxc = _env_truthy("VMEC_JAX_FREEB_AXIS_PARITY_SCALXC")
        coeff_cos = np.stack([np.asarray(Rcos_phys), np.asarray(Zcos_phys)], axis=0) * setup.even_m_mask
        coeff_sin = np.stack([np.asarray(Rsin_phys), np.asarray(Zsin_phys)], axis=0) * setup.even_m_mask
        parity_even = np.asarray(
            vmec_realspace_synthesis(
                coeff_cos=coeff_cos,
                coeff_sin=coeff_sin,
                modes=static.modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=bool(axis_apply_scalxc),
            ),
            dtype=float,
        )
        axis_r = np.asarray(parity_even[0, 0, 0, :], dtype=float)
        axis_z = np.asarray(parity_even[1, 0, 0, :], dtype=float)
        return (
            axis_r,
            axis_z,
            np.asarray(axis_r, dtype=float) if dump_scalpot_enabled else None,
            np.asarray(axis_z, dtype=float) if dump_scalpot_enabled else None,
        )
    except Exception:
        return None


def _freeb_axis_arrays(
    *,
    state: Any,
    static: Any,
    trig: Any,
    setup: _FreeBoundarySampleSetup,
    Rcos_phys: np.ndarray,
    Zsin_phys: np.ndarray,
    Rsin_phys: np.ndarray,
    Zcos_phys: np.ndarray,
    axis_override: tuple[np.ndarray, np.ndarray] | None,
    nzeta: int,
    dump_scalpot_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Choose VMEC-aligned magnetic-axis arrays for the plasma-current field."""

    axis_mode = os.getenv("VMEC_JAX_FREEB_AXIS_MODE", "vmec_pr1").strip().lower()
    axis_r_parity = axis_z_parity = None
    axis_pair: tuple[np.ndarray, np.ndarray] | None = None
    if axis_override is not None:
        try:
            axis_r_o = np.asarray(axis_override[0], dtype=float).reshape(-1)
            axis_z_o = np.asarray(axis_override[1], dtype=float).reshape(-1)
            if axis_r_o.size == int(nzeta) and axis_z_o.size == int(nzeta):
                axis_pair = (axis_r_o, axis_z_o)
        except Exception:
            axis_pair = None

    vmec_pair = _freeb_axis_from_vmec_pr1(state=state, static=static, trig=trig, axis_mode=axis_mode)
    if vmec_pair is not None:
        # Preserve legacy order: VMEC-pr1 reconstruction supersedes a supplied
        # override unless the axis mode explicitly disables the vmec_pr1 path.
        axis_pair = vmec_pair
    if axis_pair is None and _env_truthy("VMEC_JAX_FREEB_AXIS_FROM_PARITY", default=True):
        parity = _freeb_axis_from_parity(
            Rcos_phys=Rcos_phys,
            Zsin_phys=Zsin_phys,
            Rsin_phys=Rsin_phys,
            Zcos_phys=Zcos_phys,
            static=static,
            trig=trig,
            setup=setup,
            dump_scalpot_enabled=bool(dump_scalpot_enabled),
        )
        if parity is not None:
            axis_pair = (parity[0], parity[1])
            axis_r_parity, axis_z_parity = parity[2], parity[3]
    if axis_pair is None:
        axis_pair = _freeb_synthesize_axis_pair(
            Rcos_phys=Rcos_phys,
            Zsin_phys=Zsin_phys,
            Rsin_phys=Rsin_phys,
            Zcos_phys=Zcos_phys,
            static=static,
            trig=trig,
        )

    axis_r_full = axis_z_full = None
    if dump_scalpot_enabled:
        axis_r_full, axis_z_full = _freeb_synthesize_axis_pair(
            Rcos_phys=Rcos_phys,
            Zsin_phys=Zsin_phys,
            Rsin_phys=Rsin_phys,
            Zcos_phys=Zcos_phys,
            static=static,
            trig=trig,
        )
    return axis_pair[0], axis_pair[1], axis_r_full, axis_z_full, axis_r_parity, axis_z_parity


def _freeb_axis_current_field(
    *,
    R: np.ndarray,
    Z: np.ndarray,
    phi_grid: np.ndarray,
    axis_r: np.ndarray,
    axis_z: np.ndarray,
    static: Any,
    plascur: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the plasma-axis current contribution, with VMEC fallback policy."""

    axis_field_mode = os.getenv("VMEC_JAX_FREEB_AXIS_FIELD_MODE", "vmec_filament").strip().lower()
    simple_kwargs = dict(
        R=R,
        Z=Z,
        phi=phi_grid,
        axis_r=axis_r,
        axis_z=axis_z,
        nfp=int(static.cfg.nfp),
        plascur=float(plascur),
    )
    if axis_field_mode in ("simple", "legacy"):
        return _axis_current_field_simple(**simple_kwargs)
    try:
        return _axis_current_field_vmec_filament(
            R=R,
            Z=Z,
            axis_r=axis_r,
            axis_z=axis_z,
            nfp=int(static.cfg.nfp),
            plascur=float(plascur),
        )
    except Exception:
        return _axis_current_field_simple(**simple_kwargs)


def _sample_external_boundary_arrays(
    *,
    state: Any,
    static: Any,
    extcur: tuple[float, ...] | None = None,
    plascur: float = 0.0,
    axis_override: tuple[np.ndarray, np.ndarray] | None = None,
    external_field_provider_kind: str | None = None,
    external_field_provider_static: Any = None,
    external_field_provider_params: Any = None,
) -> ExternalBoundarySample:
    """Return full boundary arrays for mgrid or direct-provider sampling."""

    timing: dict[str, float] = {}
    t_total = time.perf_counter()
    t_phase = t_total
    mgrid_path, use_mgrid_provider, mgrid = _external_field_provider(
        static=static,
        provider_kind_raw=external_field_provider_kind,
    )
    extcur_eff = tuple(extcur) if extcur is not None else tuple(getattr(static, "free_boundary_extcur", ()) or ())

    sample_nzeta = 1 if (not bool(getattr(static.cfg, "lthreed", True))) else int(static.cfg.nzeta)
    dump_scalpot_enabled = _env_truthy("VMEC_JAX_DUMP_SCALPOT")
    setup = _freeb_boundary_sample_setup(static=static, sample_nzeta=int(sample_nzeta))
    trig = setup.trig
    timing["setup_time_s"] = max(0.0, time.perf_counter() - t_phase)
    t_phase = time.perf_counter()

    # Apply VMEC m=1 internal->physical conversion before free-boundary
    # sampling. This matches the convert_sym/convert_asym path feeding NESTOR.
    Rcos_phys, Zsin_phys, Rsin_phys, Zcos_phys = _freeb_physical_coefficients(state=state, static=static)
    # VMEC surface.f uses exact modal second derivatives. Reconstruct those
    # directly from boundary Fourier coefficients (instead of finite/spectral
    # differencing sampled R,Z) for matrix-side parity in analyt/fouri/scalpot.
    geom = _freeb_boundary_geometry(
        Rcos_phys=Rcos_phys,
        Zsin_phys=Zsin_phys,
        Rsin_phys=Rsin_phys,
        Zcos_phys=Zcos_phys,
        static=static,
        trig=trig,
        setup=setup,
    )
    R, Ru, Rv = geom["R"], geom["Ru"], geom["Rv"]
    Z, Zu, Zv = geom["Z"], geom["Zu"], geom["Zv"]
    timing["boundary_geometry_time_s"] = max(0.0, time.perf_counter() - t_phase)
    t_phase = time.perf_counter()

    nzeta = int(R.shape[1])
    phi_grid = setup.phi_grid

    br_mgrid, bp_mgrid, bz_mgrid = _sample_external_provider_field(
        provider_kind=external_field_provider_kind or "mgrid",
        use_mgrid_provider=use_mgrid_provider,
        mgrid=mgrid,
        external_field_provider_static=external_field_provider_static,
        external_field_provider_params=external_field_provider_params,
        R=R,
        Z=Z,
        phi_grid=phi_grid,
        extcur_eff=extcur_eff,
    )
    timing["external_field_time_s"] = max(0.0, time.perf_counter() - t_phase)
    t_phase = time.perf_counter()
    # VMEC funct3d sets:
    #   raxis_nestor(1:nzeta) = pr1(1:nzeta,1,0)
    #   zaxis_nestor(1:nzeta) = pz1(1:nzeta,1,0)
    # where parity index 0 is the even-m channel.
    axis_r, axis_z, axis_r_full, axis_z_full, axis_r_parity, axis_z_parity = _freeb_axis_arrays(
        state=state,
        static=static,
        trig=trig,
        setup=setup,
        Rcos_phys=Rcos_phys,
        Zsin_phys=Zsin_phys,
        Rsin_phys=Rsin_phys,
        Zcos_phys=Zcos_phys,
        axis_override=axis_override,
        nzeta=nzeta,
        dump_scalpot_enabled=bool(dump_scalpot_enabled),
    )
    br_axis, bp_axis, bz_axis = _freeb_axis_current_field(
        R=R,
        Z=Z,
        phi_grid=phi_grid,
        axis_r=axis_r,
        axis_z=axis_z,
        static=static,
        plascur=float(plascur),
    )
    timing["axis_field_time_s"] = max(0.0, time.perf_counter() - t_phase)
    t_phase = time.perf_counter()
    br = np.asarray(br_mgrid, dtype=float) + np.asarray(br_axis, dtype=float)
    bp = np.asarray(bp_mgrid, dtype=float) + np.asarray(bp_axis, dtype=float)
    bz = np.asarray(bz_mgrid, dtype=float) + np.asarray(bz_axis, dtype=float)
    vac = vacuum_boundary_fields_from_cylindrical(
        br=br,
        bp=bp,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
    )
    timing["projection_time_s"] = max(0.0, time.perf_counter() - t_phase)
    timing["total_time_s"] = max(0.0, time.perf_counter() - t_total)
    return ExternalBoundarySample(
        mgrid_path=mgrid_path,
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=phi_grid,
        br=br,
        bp=bp,
        bz=bz,
        br_mgrid=np.asarray(br_mgrid, dtype=float),
        bp_mgrid=np.asarray(bp_mgrid, dtype=float),
        bz_mgrid=np.asarray(bz_mgrid, dtype=float),
        br_axis=np.asarray(br_axis, dtype=float),
        bp_axis=np.asarray(bp_axis, dtype=float),
        bz_axis=np.asarray(bz_axis, dtype=float),
        axis_r=np.asarray(axis_r, dtype=float),
        axis_z=np.asarray(axis_z, dtype=float),
        vac_ext=vac,
        axis_r_full=np.asarray(axis_r_full, dtype=float),
        axis_z_full=np.asarray(axis_z_full, dtype=float),
        axis_r_parity=None if axis_r_parity is None else np.asarray(axis_r_parity, dtype=float),
        axis_z_parity=None if axis_z_parity is None else np.asarray(axis_z_parity, dtype=float),
        ruu=geom["Ruu"],
        ruv=geom["Ruv"],
        rvv=geom["Rvv"],
        zuu=geom["Zuu"],
        zuv=geom["Zuv"],
        zvv=geom["Zvv"],
        timing=timing,
    )


def _base_nestor_mode(mode: str) -> str:
    return str(mode).split("_fallback:", 1)[0]


def _is_dense_mode(mode: str) -> bool:
    return _base_nestor_mode(mode).startswith("vmec2000_like_dense_integral")


def _is_spectral_mode(mode: str) -> bool:
    return _base_nestor_mode(mode).startswith("spectral_poisson_external_only")


def _solve_periodic_poisson_fft(rhs: np.ndarray, cache: NestorPoissonCache) -> np.ndarray:
    rhs_hat = np.fft.fftn(rhs)
    phi_hat = -rhs_hat / cache.lam
    phi_hat[0, 0] = 0.0
    phi = np.fft.ifftn(phi_hat).real
    return phi


def _spectral_grad(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ntheta, nzeta = phi.shape
    ku = 2.0 * np.pi * np.fft.fftfreq(ntheta)
    kv = 2.0 * np.pi * np.fft.fftfreq(nzeta)
    ph = np.fft.fftn(phi)
    du = np.fft.ifftn((1j * ku[:, None]) * ph).real
    dv = np.fft.ifftn((1j * kv[None, :]) * ph).real
    return du, dv


def _as_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _as_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _parse_iter_list_env(name: str) -> set[int]:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return set()
    out: set[int] = set()
    for tok in raw.replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            out.add(int(t))
        except Exception:
            continue
    return out


def _select_nestor_mode(*, ntheta: int, nzeta: int) -> tuple[str, str]:
    """Pick free-boundary NESTOR model mode with fallback-friendly semantics."""

    mode_raw = os.getenv("VMEC_JAX_FREEB_NESTOR_MODE", "auto").strip().lower()
    npts = int(ntheta * nzeta)
    # Parity-first default: keep VMEC-like dense integral enabled on practical
    # free-boundary grids without requiring environment tuning.
    max_pts = max(1, _as_int_env("VMEC_JAX_FREEB_VMEC_LIKE_MAX_POINTS", 1_000_000))

    if mode_raw in ("spectral", "fast", "surrogate", "poisson"):
        return "spectral_poisson_external_only", "forced_fast"
    if mode_raw in ("vmec2000_like", "vmec_like", "vmec2000", "dense", "integral"):
        if npts > max_pts:
            return "spectral_poisson_external_only", f"fallback_max_points:{npts}>{max_pts}"
        return "vmec2000_like_dense_integral", "forced_vmec_like"
    # auto
    if npts <= max_pts:
        return "vmec2000_like_dense_integral", "auto_vmec_like"
    return "spectral_poisson_external_only", f"auto_fast_max_points:{npts}>{max_pts}"


def _vacuum_channels_from_sample_phi(sample: ExternalBoundarySample, phi: np.ndarray) -> VacuumBoundaryFields:
    dphi_u, dphi_v = _spectral_grad(phi)
    bu = np.asarray(sample.vac_ext.bu) + dphi_u
    bv = np.asarray(sample.vac_ext.bv) + dphi_v
    bsupu, bsupv, det = contravariant_boundary_field_from_covariant(
        bu=bu,
        bv=bv,
        g_uu=sample.vac_ext.g_uu,
        g_uv=sample.vac_ext.g_uv,
        g_vv=sample.vac_ext.g_vv,
    )
    # Keep parity with VMEC vacuum.f convention: bsqvac = 0.5*|B|^2.
    bsqvac = 0.5 * (bu * bsupu + bv * bsupv)
    return VacuumBoundaryFields(
        bu=bu,
        bv=bv,
        bsupu=bsupu,
        bsupv=bsupv,
        bsqvac=bsqvac,
        bnormal=sample.vac_ext.bnormal,
        bnormal_unit=sample.vac_ext.bnormal_unit,
        g_uu=sample.vac_ext.g_uu,
        g_uv=sample.vac_ext.g_uv,
        g_vv=sample.vac_ext.g_vv,
        det_guv=det,
    )


def _vacuum_channels_from_sample_potvac(
    *,
    sample: ExternalBoundarySample,
    basis: dict[str, Any],
    potvac: np.ndarray,
) -> VacuumBoundaryFields:
    """Compute VMEC vacuum channels from mode coefficients (potsin/potcos)."""

    pot = np.asarray(potvac, dtype=float).reshape(-1)
    mnpd = int(basis["mnpd"])
    if pot.size < mnpd:
        raise ValueError("potvac_too_small")
    potsin = np.asarray(pot[:mnpd], dtype=float)
    if bool(basis["lasym"]) and pot.size >= 2 * mnpd:
        potcos = np.asarray(pot[mnpd : 2 * mnpd], dtype=float)
    else:
        potcos = np.zeros((mnpd,), dtype=float)

    xmpot = np.asarray(basis["xmpot"], dtype=float)
    n_raw = np.asarray(basis["n_raw"], dtype=float)
    nfp = float(int(basis["nfp"]))
    cos_phase = np.asarray(basis["cos_phase"], dtype=float)
    sin_phase = np.asarray(basis["sin_phase"], dtype=float)

    mfac = xmpot * potsin
    nfac = (-n_raw * nfp) * potsin
    potu = cos_phase @ mfac
    potv = cos_phase @ nfac
    if bool(basis["lasym"]):
        mfac_c = xmpot * potcos
        nfac_c = (-n_raw * nfp) * potcos
        potu = potu - (sin_phase @ mfac_c)
        potv = potv - (sin_phase @ nfac_c)

    potu2 = potu.reshape(sample.R.shape)
    potv2 = potv.reshape(sample.R.shape)
    bu = np.asarray(sample.vac_ext.bu, dtype=float) + potu2
    bv = np.asarray(sample.vac_ext.bv, dtype=float) + potv2
    bsupu, bsupv, det = contravariant_boundary_field_from_covariant(
        bu=bu,
        bv=bv,
        g_uu=sample.vac_ext.g_uu,
        g_uv=sample.vac_ext.g_uv,
        g_vv=sample.vac_ext.g_vv,
    )
    bsqvac = 0.5 * (bu * bsupu + bv * bsupv)
    return VacuumBoundaryFields(
        bu=bu,
        bv=bv,
        bsupu=bsupu,
        bsupv=bsupv,
        bsqvac=bsqvac,
        bnormal=np.asarray(sample.vac_ext.bnormal, dtype=float),
        bnormal_unit=np.asarray(sample.vac_ext.bnormal_unit, dtype=float),
        g_uu=np.asarray(sample.vac_ext.g_uu, dtype=float),
        g_uv=np.asarray(sample.vac_ext.g_uv, dtype=float),
        g_vv=np.asarray(sample.vac_ext.g_vv, dtype=float),
        det_guv=det,
    )


def _maybe_dump_scalpot_jax(
    *,
    iter_idx: int | None,
    ivac: int,
    reused: bool,
    mode: str,
    rhs: np.ndarray,
    phi: np.ndarray,
    vac: VacuumBoundaryFields,
    cache: Any,
    sample: ExternalBoundarySample,
    mf: int,
    nf: int,
    nfp: int,
    lasym: bool,
    wint_vmec: np.ndarray | None = None,
    gsource_vmec: np.ndarray | None = None,
    potvac: np.ndarray | None = None,
    bvec_mode: np.ndarray | None = None,
    bvec_mode_nonsing: np.ndarray | None = None,
    bvec_mode_analytic: np.ndarray | None = None,
    source_cache_iter: int | None = None,
    matrix_override_applied: bool = False,
    amatrix_mode_pre: np.ndarray | None = None,
    amatrix_mode_from_grpmn: np.ndarray | None = None,
    grpmn_nonsing: np.ndarray | None = None,
    grpmn_analytic: np.ndarray | None = None,
    grpmn_total: np.ndarray | None = None,
    plascur: float | None = None,
) -> None:
    env = os.getenv("VMEC_JAX_DUMP_SCALPOT", "").strip().lower()
    if env in ("", "0", "false", "no"):
        return
    if iter_idx is None:
        return
    iters = _parse_iter_list_env("VMEC_JAX_DUMP_ITER")
    if iters and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    out = {
        "iter": np.asarray(int(iter_idx), dtype=np.int64),
        "ivac": np.asarray(int(ivac), dtype=np.int64),
        "reused": np.asarray(1 if bool(reused) else 0, dtype=np.int64),
        "mode": np.asarray(str(mode)),
        "rhs": np.asarray(rhs, dtype=float),
        "phi": np.asarray(phi, dtype=float),
        "R": np.asarray(sample.R, dtype=float),
        "Z": np.asarray(sample.Z, dtype=float),
        "phi_grid": np.asarray(sample.phi, dtype=float),
        "Ru": np.asarray(sample.Ru, dtype=float),
        "Zu": np.asarray(sample.Zu, dtype=float),
        "Rv": np.asarray(sample.Rv, dtype=float),
        "Zv": np.asarray(sample.Zv, dtype=float),
        "br": np.asarray(sample.br, dtype=float),
        "bp": np.asarray(sample.bp, dtype=float),
        "bz": np.asarray(sample.bz, dtype=float),
        "br_mgrid": np.asarray(sample.br_mgrid, dtype=float),
        "bp_mgrid": np.asarray(sample.bp_mgrid, dtype=float),
        "bz_mgrid": np.asarray(sample.bz_mgrid, dtype=float),
        "br_axis": np.asarray(sample.br_axis, dtype=float),
        "bp_axis": np.asarray(sample.bp_axis, dtype=float),
        "bz_axis": np.asarray(sample.bz_axis, dtype=float),
        "axis_r": np.asarray(sample.axis_r, dtype=float),
        "axis_z": np.asarray(sample.axis_z, dtype=float),
        "bexu_ext": np.asarray(sample.vac_ext.bu, dtype=float),
        "bexv_ext": np.asarray(sample.vac_ext.bv, dtype=float),
        "bexn_ext": np.asarray(-sample.vac_ext.bnormal, dtype=float),
        "bnormal": np.asarray(sample.vac_ext.bnormal, dtype=float),
        "bu": np.asarray(vac.bu, dtype=float),
        "bv": np.asarray(vac.bv, dtype=float),
        "bsqvac": np.asarray(vac.bsqvac, dtype=float),
        "bnormal_unit": np.asarray(vac.bnormal_unit, dtype=float),
    }
    # VMEC bextern.f diagnostics: explicit normal components and bexn terms.
    try:
        snr = -np.asarray(sample.R, dtype=float) * np.asarray(sample.Zu, dtype=float)
        snv = (
            np.asarray(sample.Zu, dtype=float) * np.asarray(sample.Rv, dtype=float)
            - np.asarray(sample.Ru, dtype=float) * np.asarray(sample.Zv, dtype=float)
        )
        snz = np.asarray(sample.R, dtype=float) * np.asarray(sample.Ru, dtype=float)
        out["snr"] = np.asarray(snr, dtype=float)
        out["snv"] = np.asarray(snv, dtype=float)
        out["snz"] = np.asarray(snz, dtype=float)
        bexn_term_r = np.asarray(sample.br, dtype=float) * np.asarray(snr, dtype=float)
        bexn_term_phi = np.asarray(sample.bp, dtype=float) * np.asarray(snv, dtype=float)
        bexn_term_z = np.asarray(sample.bz, dtype=float) * np.asarray(snz, dtype=float)
        out["bexn_term_r"] = np.asarray(bexn_term_r, dtype=float)
        out["bexn_term_phi"] = np.asarray(bexn_term_phi, dtype=float)
        out["bexn_term_z"] = np.asarray(bexn_term_z, dtype=float)
        out["bexn_recon"] = -(
            np.asarray(bexn_term_r, dtype=float)
            + np.asarray(bexn_term_phi, dtype=float)
            + np.asarray(bexn_term_z, dtype=float)
        )
    except Exception:
        pass
    for key in ("axis_r_full", "axis_z_full", "axis_r_parity", "axis_z_parity"):
        if (value := getattr(sample, key)) is not None:
            out[key] = np.asarray(value, dtype=float)
    if plascur is not None:
        out["plascur"] = np.asarray(float(plascur), dtype=float)
    for source, key in (
        ("ruu", "Ruu"),
        ("ruv", "Ruv"),
        ("rvv", "Rvv"),
        ("zuu", "Zuu"),
        ("zuv", "Zuv"),
        ("zvv", "Zvv"),
    ):
        if (value := getattr(sample, source)) is not None:
            out[key] = np.asarray(value, dtype=float)
    if source_cache_iter is not None:
        out["source_cache_iter"] = np.asarray(int(source_cache_iter), dtype=np.int64)
    out["matrix_override_applied"] = np.asarray(1 if bool(matrix_override_applied) else 0, dtype=np.int64)
    ntheta, nzeta = rhs.shape
    wint_uniform = np.full((ntheta, nzeta), 1.0 / float(max(1, ntheta * nzeta)), dtype=float)
    out["wint_uniform"] = wint_uniform
    out["bexni_uniform"] = np.asarray(-sample.vac_ext.bnormal, dtype=float) * wint_uniform * ((2.0 * np.pi) ** 2)
    if wint_vmec is not None:
        wv = np.asarray(wint_vmec, dtype=float)
        if wv.shape == (ntheta, nzeta):
            out["wint_vmec"] = wv
            out["bexni_vmec"] = np.asarray(-sample.vac_ext.bnormal, dtype=float) * wv * ((2.0 * np.pi) ** 2)
    if isinstance(cache, NestorVmecLikeCache):
        out["cache_kind"] = np.asarray("dense")
        out["matrix"] = np.asarray(cache.matrix, dtype=float)
        out["rhs_scale"] = np.asarray(cache.rhs_scale, dtype=float)
        if gsource_vmec is not None:
            out["gsource_vmec"] = np.asarray(gsource_vmec, dtype=float)
            try:
                gsrc_flat = np.asarray(gsource_vmec, dtype=float).reshape(-1)
                amat = np.asarray(cache.matrix, dtype=float)
                if amat.ndim == 2 and amat.shape[1] == gsrc_flat.size:
                    out["gsource_kernel"] = (amat @ gsrc_flat).reshape(ntheta, nzeta)
            except Exception:
                pass
        if potvac is not None:
            out["potvac"] = np.asarray(potvac, dtype=float)
        try:
            basis = cache.mode_basis
            if basis is None:
                basis = _build_vmec_mode_basis(
                    ntheta=int(ntheta),
                    nzeta=int(nzeta),
                    nfp=int(nfp),
                    mf=int(mf),
                    nf=int(nf),
                    lasym=bool(lasym),
                    wint=np.asarray(wint_vmec if wint_vmec is not None else np.asarray(cache.rhs_scale).reshape(ntheta, nzeta), dtype=float),
                )
            out["xmpot"] = np.asarray(basis["xmpot"], dtype=np.int64)
            out["xnpot"] = np.asarray(basis["xnpot"], dtype=np.int64)
            out["sinmni"] = np.asarray(basis["sinmni"], dtype=float)
            out["cosmni"] = np.asarray(basis["cosmni"], dtype=float)
            if gsource_vmec is not None:
                src_sym = _vmec_source_from_gsource(
                    gsource=np.asarray(gsource_vmec, dtype=float),
                    basis=basis,
                )
                out["source_sym"] = np.asarray(src_sym, dtype=float).reshape(ntheta, nzeta)

            if bvec_mode is not None:
                bv = np.asarray(bvec_mode, dtype=float).reshape(-1)
            else:
                if gsource_vmec is None:
                    gsource_loc = np.asarray(rhs, dtype=float).reshape(-1)
                else:
                    gsource_loc = np.asarray(gsource_vmec, dtype=float).reshape(-1)
                bv = _vmec_bvec_from_gsource(gsource=gsource_loc, basis=basis)

            mnpd = int(basis["mnpd"])
            out["bvec_mode_sin"] = np.asarray(bv[:mnpd], dtype=float)
            if bool(basis["lasym"]) and bv.size >= 2 * mnpd:
                out["bvec_mode_cos"] = np.asarray(bv[mnpd : 2 * mnpd], dtype=float)
            for prefix, value in (
                ("bvec_mode_nonsing", bvec_mode_nonsing),
                ("bvec_mode_analytic", bvec_mode_analytic),
            ):
                if value is not None:
                    vec = np.asarray(value, dtype=float).reshape(-1)
                    out[f"{prefix}_sin"] = np.asarray(vec[:mnpd], dtype=float)
                    if bool(basis["lasym"]) and vec.size >= 2 * mnpd:
                        out[f"{prefix}_cos"] = np.asarray(vec[mnpd : 2 * mnpd], dtype=float)

            if cache.mode_matrix is not None:
                out["amatrix_mode"] = np.asarray(cache.mode_matrix, dtype=float)
            else:
                sinmni = np.asarray(basis["sinmni"], dtype=float)
                if bool(basis["lasym"]):
                    cosmni = np.asarray(basis["cosmni"], dtype=float)
                    B = np.concatenate([sinmni, cosmni], axis=1)
                else:
                    B = sinmni
                A = np.asarray(cache.matrix, dtype=float)
                out["amatrix_mode"] = B.T @ (A @ B)
            for key, value in {
                "amatrix_mode_pre": amatrix_mode_pre,
                "amatrix_mode_from_grpmn": amatrix_mode_from_grpmn,
                "grpmn_nonsing": grpmn_nonsing,
                "grpmn_analytic": grpmn_analytic,
                "grpmn_total": grpmn_total,
            }.items():
                if value is not None:
                    out[key] = np.asarray(value, dtype=float)
        except Exception:
            pass
    elif isinstance(cache, NestorPoissonCache):
        out["cache_kind"] = np.asarray("spectral")
        out["lam"] = np.asarray(cache.lam, dtype=float)
    else:
        out["cache_kind"] = np.asarray("unknown")
    fpath = outdir / f"scalpot_jax_iter{int(iter_idx)}.npz"
    np.savez_compressed(fpath, **out)


def _freeb_use_greenf_source(ntor: int) -> bool:
    """Resolve Green-function source assembly toggle for free-boundary mode."""

    _ = int(ntor)  # kept for future topology-specific policy hooks
    greenf_env = os.getenv("VMEC_JAX_FREEB_USE_GREENF_SOURCE")
    if greenf_env is None:
        # Default to VMEC-like Green-function non-singular source assembly in
        # all topologies. Environment variable remains a diagnostic override.
        return True
    return _env_truthy("VMEC_JAX_FREEB_USE_GREENF_SOURCE")


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in ("", "0", "false", "no")


def _solve_vmec_like_mode_with_jax_nestor_operator(
    *,
    sample: ExternalBoundarySample,
    basis: dict[str, Any],
    bexni: np.ndarray,
    signgs: int,
    nvper: int,
    include_analytic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    tables = _ensure_vmec_nonsingular_kernel_tables(
        basis=basis,
        nv=int(np.asarray(sample.R).shape[1]),
        nvper=max(1, int(nvper)),
    )
    return _solve_vmec_like_mode_with_jax_nestor_operator_impl(
        sample=sample,
        basis=basis,
        tables=tables,
        bexni=bexni,
        signgs=signgs,
        nvper=nvper,
        include_analytic=include_analytic,
    )


def _nestor_trace_arrays(
    *,
    sample: ExternalBoundarySample,
    vac_total: VacuumBoundaryFields,
    gsource_vmec: np.ndarray,
    potvac: np.ndarray | None,
    bvec_mode: np.ndarray | None,
    bvec_mode_nonsing: np.ndarray | None,
    bvec_mode_analytic: np.ndarray | None,
    grpmn_nonsing: np.ndarray | None,
    grpmn_analytic: np.ndarray | None,
    grpmn_total: np.ndarray | None,
    amatrix_mode_from_grpmn: np.ndarray | None,
    cache: Any,
) -> dict[str, Any]:
    """Pack optional NESTOR trace arrays without cluttering solve policy code."""
    out = {
        name: np.asarray(getattr(sample, name), dtype=float)
        for name in ("R", "Z", "phi", "Ru", "Zu", "Rv", "Zv", "br", "bp", "bz")
    }
    out.update(
        {
            name: np.asarray(value, dtype=float)
            for name, value in {
                "br_mgrid": sample.br_mgrid,
                "bp_mgrid": sample.bp_mgrid,
                "bz_mgrid": sample.bz_mgrid,
                "br_axis": sample.br_axis,
                "bp_axis": sample.bp_axis,
                "bz_axis": sample.bz_axis,
                "bu_ext": sample.vac_ext.bu,
                "bv_ext": sample.vac_ext.bv,
                "bnormal": sample.vac_ext.bnormal,
                "g_uu": sample.vac_ext.g_uu,
                "g_uv": sample.vac_ext.g_uv,
                "g_vv": sample.vac_ext.g_vv,
                "bsqvac": vac_total.bsqvac,
                "gsource_vmec": gsource_vmec,
            }.items()
        }
    )
    for name, value in {
        "potvac": potvac,
        "bvec_mode": bvec_mode,
        "bvec_mode_nonsing": bvec_mode_nonsing,
        "bvec_mode_analytic": bvec_mode_analytic,
        "grpmn_nonsing": grpmn_nonsing,
        "grpmn_analytic": grpmn_analytic,
        "grpmn_total": grpmn_total,
        "mode_matrix": amatrix_mode_from_grpmn,
    }.items():
        if value is not None:
            out[name] = np.asarray(value, dtype=float)
    if "mode_matrix" not in out and isinstance(cache, NestorVmecLikeCache) and cache.mode_matrix is not None:
        out["mode_matrix"] = np.asarray(cache.mode_matrix, dtype=float)
    return out


def _nestor_legacy_fast_reuse(
    *,
    runtime: NestorRuntimeState,
    runtime_cache: Any,
    runtime_mode: str,
) -> tuple[NestorSolveResult, NestorRuntimeState]:
    bsqvac = np.asarray(runtime.bsqvac)
    zeros = {key: np.zeros_like(bsqvac) for key in ("bu", "bv", "bsupu", "bsupv", "bnormal", "bnormal_unit", "g_uu", "g_uv", "g_vv", "det_guv")}
    optional_cache = {
        key: None if getattr(runtime, key, None) is None else np.asarray(getattr(runtime, key), dtype=float)
        for key in ("gsource_cached", "source_sym_cached", "bvec_nonsing_cached")
    }
    return (
        NestorSolveResult(
            vac_total=VacuumBoundaryFields(bsqvac=bsqvac, **zeros),
            phi=np.asarray(runtime.phi),
            reused=True,
            solve_time_s=0.0,
            sample_time_s=0.0,
            model=runtime_mode,
        ),
        NestorRuntimeState(
            operator_cache=runtime_cache,
            phi=np.asarray(runtime.phi),
            bsqvac=np.asarray(bsqvac),
            mode=runtime_mode,
            update_count=int(runtime.update_count),
            reuse_count=int(runtime.reuse_count) + 1,
            source_cache_iter=int(getattr(runtime, "source_cache_iter", -1)),
            **optional_cache,
        ),
    )


def _rms_float(arr: Any) -> float:
    vals = np.asarray(arr, dtype=float)
    return float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0


def _nestor_step_diagnostics(ctx: dict[str, Any]) -> dict[str, Any]:
    sample = ctx["sample"]
    bsqvac = ctx["bsqvac"]
    out = {
        "provider_kind": ctx["provider_kind"],
        "rhs_mode": str(ctx["rhs_mode"]),
        "mode": str(ctx["used_mode"]),
        "jax_nestor_operator_reason": str(ctx["jax_nestor_operator_reason"]),
        "sample_points": int(ctx["ntheta"] * ctx["nzeta"]),
        "bsqvac_mean": float(np.mean(np.asarray(bsqvac, dtype=float))) if np.asarray(bsqvac).size else 0.0,
    }
    for key in ("ntheta", "nzeta"):
        out[f"sample_{key}"] = int(ctx[key])
    for key in "provider_allows_source_reuse source_reused matrix_override_applied jax_nestor_operator_applied jax_nestor_operator_jitted jax_nestor_operator_cache_hit".split():
        out[key] = bool(ctx[key])
    out["reused"] = bool(ctx["reuse_step"])
    for key in ("sample_time", "solve_time"):
        out[f"{key}_s"] = float(ctx[key])
    for key in "cache_build_time_s source_time_s bvec_time_s matrix_time_s linear_solve_time_s vacuum_channels_time_s jax_nestor_operator_time_s".split():
        out[key] = float(ctx[key])
    for key, value in (
        ("br", sample.br),
        ("bp", sample.bp),
        ("bz", sample.bz),
        ("bnormal", sample.vac_ext.bnormal),
        ("bnormal_unit", sample.vac_ext.bnormal_unit),
        ("rhs", ctx["rhs"]),
        ("gsource", ctx["gsource_vmec"]),
        ("bsqvac", bsqvac),
        ("bvec_mode", ctx["bvec_mode"]),
        ("bvec_mode_nonsing", ctx["bvec_mode_nonsing"]),
        ("bvec_mode_analytic", ctx["bvec_mode_analytic"]),
    ):
        if value is not None:
            out[f"{key}_rms"] = _rms_float(value)
    provider_static = ctx["external_field_provider_static"]
    provider_params = ctx["external_field_provider_params"]
    if isinstance(provider_static, dict):
        chunk_size = provider_static.get("chunk_size", getattr(provider_params, "chunk_size", None))
        out.update({
            "provider_coil_geometry_cached": bool("coil_geometry" in provider_static),
            "provider_jit_sampler": bool(provider_static.get("jit_sampler", False)),
            "provider_cache_scope": str(provider_static.get("cache_scope", "")),
            "provider_regularization_epsilon": float(provider_static.get("regularization_epsilon", getattr(provider_params, "regularization_epsilon", 0.0))),
            "provider_chunk_size": None if chunk_size is None else int(chunk_size),
        })
        geometry = provider_static.get("coil_geometry")
        if isinstance(geometry, tuple) and len(geometry) >= 1:
            shape = tuple(int(dim) for dim in getattr(geometry[0], "shape", ())[:2])
            out.update(dict(zip(("provider_coil_count", "provider_segments_per_coil"), shape, strict=False)))
    if isinstance(getattr(sample, "timing", None), dict):
        for key, value in sample.timing.items():
            try:
                out[f"sample_{key}"] = float(value)
            except Exception:
                pass
    if isinstance(cache := ctx["cache"], NestorVmecLikeCache):
        out.update({"physical_matrix_lu_built": bool(cache.matrix_lu is not None), "mode_matrix_lu_built": bool(cache.mode_matrix_lu is not None)})
    return out


def _nestor_runtime_next_and_dump_source(ctx: dict[str, Any]) -> tuple[NestorRuntimeState, np.ndarray]:
    """Build the next NESTOR runtime cache and source array used by debug dumps."""

    runtime = ctx["runtime"]
    cache = ctx["cache"]
    reuse_step = bool(ctx["reuse_step"])
    provider_allows_source_reuse = bool(ctx["provider_allows_source_reuse"])
    gsource_cached = ctx["runtime_gsource_cached"]
    source_sym_cached = ctx["runtime_source_sym_cached"]
    bvec_nonsing_cached = ctx["runtime_bvec_nonsing_cached"]
    source_cache_iter = int(ctx["runtime_source_cache_iter"])
    if isinstance(cache, NestorVmecLikeCache) and (cache.mode_basis is not None):
        basis = cache.mode_basis
        if (not reuse_step) or (not provider_allows_source_reuse) or (gsource_cached is None):
            gsource_cached = np.asarray(ctx["gsource_vmec"], dtype=float)
        if (not reuse_step) or (not provider_allows_source_reuse) or (source_sym_cached is None):
            try:
                source_sym_cached = _vmec_source_from_gsource(
                    gsource=np.asarray(gsource_cached, dtype=float),
                    basis=basis,
                )
            except Exception:
                source_sym_cached = ctx["runtime_source_sym_cached"]
        if (not reuse_step) or (not provider_allows_source_reuse) or (bvec_nonsing_cached is None):
            if ctx["bvec_mode_nonsing"] is not None:
                bvec_nonsing_cached = np.asarray(ctx["bvec_mode_nonsing"], dtype=float)
            else:
                try:
                    bvec_nonsing_cached = _vmec_bvec_from_gsource(
                        gsource=np.asarray(gsource_cached, dtype=float),
                        basis=basis,
                    )
                except Exception:
                    bvec_nonsing_cached = ctx["runtime_bvec_nonsing_cached"]
        if (not reuse_step) and (ctx["iter_idx"] is not None):
            source_cache_iter = int(ctx["iter_idx"])
    runtime_next = NestorRuntimeState(
        operator_cache=cache,
        phi=np.asarray(ctx["phi"]),
        bsqvac=np.asarray(ctx["bsqvac"]),
        mode=ctx["used_mode"],
        update_count=(0 if runtime is None else int(runtime.update_count)) + (0 if reuse_step else 1),
        reuse_count=(0 if runtime is None else int(runtime.reuse_count)) + (1 if reuse_step else 0),
        source_cache_iter=int(source_cache_iter),
        gsource_cached=None if gsource_cached is None else np.asarray(gsource_cached, dtype=float),
        source_sym_cached=None if source_sym_cached is None else np.asarray(source_sym_cached, dtype=float),
        bvec_nonsing_cached=None if bvec_nonsing_cached is None else np.asarray(bvec_nonsing_cached, dtype=float),
    )
    gsource_dump = (
        np.asarray(gsource_cached, dtype=float)
        if (reuse_step and gsource_cached is not None)
        else np.asarray(ctx["gsource_vmec"], dtype=float)
    )
    return runtime_next, gsource_dump


def _nestor_external_step_result(ctx_in: dict[str, Any]) -> tuple[NestorSolveResult, NestorRuntimeState]:
    """Assemble NESTOR result, runtime cache, optional trace, and debug dump."""

    ctx = dict(ctx_in)
    vac_total = ctx["vac_total"]
    bsqvac = np.asarray(vac_total.bsqvac)
    ctx["bsqvac"] = bsqvac
    solve_time = max(0.0, time.perf_counter() - ctx["ts"])
    ctx["solve_time"] = solve_time

    diagnostics = _nestor_step_diagnostics(ctx)

    trace_arrays = None
    if bool(ctx["collect_trace_arrays"]):
        trace_arrays = _nestor_trace_arrays(
            sample=ctx["sample"],
            vac_total=vac_total,
            gsource_vmec=ctx["gsource_vmec"],
            potvac=ctx["potvac"],
            bvec_mode=ctx["bvec_mode"],
            bvec_mode_nonsing=ctx["bvec_mode_nonsing"],
            bvec_mode_analytic=ctx["bvec_mode_analytic"],
            grpmn_nonsing=ctx["grpmn_nonsing"],
            grpmn_analytic=ctx["grpmn_analytic"],
            grpmn_total=ctx["grpmn_total"],
            amatrix_mode_from_grpmn=ctx["amatrix_mode_from_grpmn"],
            cache=ctx["cache"],
        )

    res = NestorSolveResult(
        vac_total=vac_total,
        phi=ctx["phi"],
        reused=bool(ctx["reuse_step"]),
        solve_time_s=solve_time,
        sample_time_s=ctx["sample_time"],
        model=ctx["used_mode"],
        diagnostics=diagnostics,
        trace_arrays=trace_arrays,
    )
    runtime_next, gsource_dump = _nestor_runtime_next_and_dump_source(ctx)
    static = ctx["static"]
    _maybe_dump_scalpot_jax(
        iter_idx=ctx["iter_idx"],
        ivac=int(ctx["ivac"]),
        reused=bool(ctx["reuse_step"]),
        mode=ctx["used_mode"],
        rhs=np.asarray(ctx["rhs"], dtype=float),
        phi=np.asarray(ctx["phi"], dtype=float),
        vac=vac_total,
        cache=ctx["cache"],
        sample=ctx["sample"],
        mf=max(0, int(getattr(static.cfg, "mpol", 1)) + 1),
        nf=max(0, int(getattr(static.cfg, "ntor", 0))),
        nfp=max(1, int(getattr(static.cfg, "nfp", 1))),
        lasym=bool(getattr(static.cfg, "lasym", False)),
        wint_vmec=np.asarray(ctx["wint_vmec"], dtype=float),
        gsource_vmec=gsource_dump,
        potvac=None if ctx["potvac"] is None else np.asarray(ctx["potvac"], dtype=float),
        bvec_mode=None if ctx["bvec_mode"] is None else np.asarray(ctx["bvec_mode"], dtype=float),
        bvec_mode_nonsing=None
        if ctx["bvec_mode_nonsing"] is None
        else np.asarray(ctx["bvec_mode_nonsing"], dtype=float),
        bvec_mode_analytic=None
        if ctx["bvec_mode_analytic"] is None
        else np.asarray(ctx["bvec_mode_analytic"], dtype=float),
        source_cache_iter=int(runtime_next.source_cache_iter),
        matrix_override_applied=bool(ctx["matrix_override_applied"]),
        amatrix_mode_pre=None
        if ctx["amatrix_mode_pre"] is None
        else np.asarray(ctx["amatrix_mode_pre"], dtype=float),
        amatrix_mode_from_grpmn=None
        if ctx["amatrix_mode_from_grpmn"] is None
        else np.asarray(ctx["amatrix_mode_from_grpmn"], dtype=float),
        grpmn_nonsing=None if ctx["grpmn_nonsing"] is None else np.asarray(ctx["grpmn_nonsing"], dtype=float),
        grpmn_analytic=None if ctx["grpmn_analytic"] is None else np.asarray(ctx["grpmn_analytic"], dtype=float),
        grpmn_total=None if ctx["grpmn_total"] is None else np.asarray(ctx["grpmn_total"], dtype=float),
        plascur=float(ctx["plascur"]),
    )
    return res, runtime_next


def _nestor_runtime_context(runtime: NestorRuntimeState | None) -> dict[str, Any]:
    """Decode current NESTOR runtime cache with backward-compatible field names."""

    runtime_cache = None if runtime is None else getattr(runtime, "operator_cache", None)
    runtime_mode = (
        "spectral_poisson_external_only"
        if runtime is None
        else str(getattr(runtime, "mode", "spectral_poisson_external_only"))
    )
    if runtime_cache is None and runtime is not None and hasattr(runtime, "poisson"):
        # Backward compatibility with older runtime state shape.
        runtime_cache = getattr(runtime, "poisson")
    return {
        "runtime_cache": runtime_cache,
        "runtime_mode": runtime_mode,
        "runtime_source_cache_iter": -1 if runtime is None else int(getattr(runtime, "source_cache_iter", -1)),
        "runtime_gsource_cached": None if runtime is None else getattr(runtime, "gsource_cached", None),
        "runtime_source_sym_cached": None if runtime is None else getattr(runtime, "source_sym_cached", None),
        "runtime_bvec_nonsing_cached": None if runtime is None else getattr(runtime, "bvec_nonsing_cached", None),
    }


def _nestor_provider_allows_source_reuse(
    *,
    provider_kind: str,
    provider_static: Any,
) -> bool:
    return provider_kind in ("", "mgrid", "legacy_mgrid") or (
        isinstance(provider_static, dict) and bool(provider_static.get("allow_source_reuse", False))
    )


def _nestor_reuse_step(
    *,
    ivac: int,
    ivacskip: int | None,
    runtime: NestorRuntimeState | None,
) -> bool:
    if ivacskip is not None:
        return int(ivacskip) != 0 and runtime is not None
    return int(ivac) != 1 and runtime is not None


def _nestor_rhs_and_source(
    *,
    sample: ExternalBoundarySample,
    static: Any,
    reuse_step: bool,
    provider_allows_source_reuse: bool,
    runtime_gsource_cached: Any,
) -> dict[str, Any]:
    ntheta, nzeta = sample.R.shape
    rhs_mode = os.getenv("VMEC_JAX_FREEB_RHS_MODE", "bnormal_unit").strip().lower()
    wint_vmec = _vmec_boundary_wint(static=static, ntheta=int(ntheta), nzeta=int(nzeta))
    gsource_bexni = -np.asarray(sample.vac_ext.bnormal, dtype=float) * np.asarray(wint_vmec, dtype=float) * (
        (2.0 * np.pi) ** 2
    )
    gsource_vmec = np.asarray(gsource_bexni, dtype=float)
    source_reused = bool(reuse_step and provider_allows_source_reuse and runtime_gsource_cached is not None)
    if source_reused:
        gsource_vmec = np.asarray(runtime_gsource_cached, dtype=float)

    if rhs_mode in ("unit", "unit_normal", "bnormal_unit"):
        rhs = -np.asarray(sample.vac_ext.bnormal_unit, dtype=float)
        rhs_mode = "bnormal_unit"
    elif rhs_mode in ("bexni", "vmec_bexni", "bnormal_wint"):
        rhs = np.asarray(gsource_bexni, dtype=float)
        rhs_mode = "bexni"
    else:
        # VMEC scalpot source uses B·dS (non-unit normal) channels.
        rhs = -np.asarray(sample.vac_ext.bnormal, dtype=float)
        rhs_mode = "bnormal"

    return {
        "rhs_mode": rhs_mode,
        "wint_vmec": wint_vmec,
        "gsource_bexni": gsource_bexni,
        "gsource_vmec": gsource_vmec,
        "source_reused": source_reused,
        "rhs": rhs,
    }


def _initial_nestor_solve_context() -> dict[str, Any]:
    return {
        "potvac": None,
        "bvec_mode": None,
        "bvec_mode_nonsing": None,
        "bvec_mode_analytic": None,
        "grpmn_nonsing": None,
        "grpmn_analytic": None,
        "grpmn_total": None,
        "amatrix_mode_pre": None,
        "amatrix_mode_from_grpmn": None,
        "matrix_override_applied": False,
        "jax_nestor_operator_applied": False,
        "jax_nestor_operator_reason": "disabled",
        "jax_nestor_operator_jitted": False,
        "jax_nestor_operator_cache_hit": False,
        "jax_nestor_operator_time_s": 0.0,
        "cache_build_time_s": 0.0,
        "source_time_s": 0.0,
        "bvec_time_s": 0.0,
        "matrix_time_s": 0.0,
        "linear_solve_time_s": 0.0,
        "vacuum_channels_time_s": 0.0,
    }


def _prepare_nestor_external_step_context(
    *,
    state: Any,
    static: Any,
    ivac: int,
    ivacskip: int | None = None,
    iter_idx: int | None = None,
    runtime: NestorRuntimeState | None = None,
    extcur: tuple[float, ...] | None = None,
    plascur: float = 0.0,
    axis_override: tuple[np.ndarray, np.ndarray] | None = None,
    external_field_provider_kind: str | None = None,
    external_field_provider_static: Any = None,
    external_field_provider_params: Any = None,
    collect_trace_arrays: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    sample = _sample_external_boundary_arrays(
        state=state,
        static=static,
        extcur=extcur,
        plascur=float(plascur),
        axis_override=axis_override,
        external_field_provider_kind=external_field_provider_kind,
        external_field_provider_static=external_field_provider_static,
        external_field_provider_params=external_field_provider_params,
    )
    sample_time = max(0.0, time.perf_counter() - t0)
    ntheta, nzeta = sample.R.shape
    selected_mode, mode_reason = _select_nestor_mode(ntheta=ntheta, nzeta=nzeta)
    provider_kind = "mgrid" if external_field_provider_kind is None else str(external_field_provider_kind).strip().lower()
    provider_allows_source_reuse = _nestor_provider_allows_source_reuse(
        provider_kind=provider_kind,
        provider_static=external_field_provider_static,
    )
    runtime_ctx = _nestor_runtime_context(runtime)
    reuse_step = _nestor_reuse_step(ivac=int(ivac), ivacskip=ivacskip, runtime=runtime)
    source_ctx = _nestor_rhs_and_source(
        sample=sample,
        static=static,
        reuse_step=reuse_step,
        provider_allows_source_reuse=provider_allows_source_reuse,
        runtime_gsource_cached=runtime_ctx["runtime_gsource_cached"],
    )
    used_mode = selected_mode
    if mode_reason not in ("forced_fast", "forced_vmec_like", "auto_vmec_like"):
        used_mode = f"{selected_mode}_fallback:{mode_reason}"

    # On ivacskip reuse, emulate VMEC2000 scalpot behavior by reusing the cached
    # operator while refreshing the source term / solve.
    mode_for_step = runtime_ctx["runtime_mode"] if reuse_step else used_mode
    if reuse_step and runtime_ctx["runtime_cache"] is None:
        mode_for_step = used_mode

    ctx = {
        **runtime_ctx,
        **source_ctx,
        **_initial_nestor_solve_context(),
        "state": state,
        "static": static,
        "ivac": int(ivac),
        "iter_idx": iter_idx,
        "runtime": runtime,
        "plascur": float(plascur),
        "external_field_provider_static": external_field_provider_static,
        "external_field_provider_params": external_field_provider_params,
        "collect_trace_arrays": bool(collect_trace_arrays),
        "sample": sample,
        "sample_time": sample_time,
        "ntheta": int(ntheta),
        "nzeta": int(nzeta),
        "provider_kind": provider_kind,
        "provider_allows_source_reuse": provider_allows_source_reuse,
        "reuse_step": reuse_step,
        "used_mode": used_mode,
        "mode_for_step": mode_for_step,
        "cache": runtime_ctx["runtime_cache"],
        "ts": time.perf_counter(),
    }
    return ctx


def _ensure_vmec_like_cache_for_nestor_step(ctx: dict[str, Any]) -> dict[str, Any]:
    sample = ctx["sample"]
    static = ctx["static"]
    cache = ctx["cache"]
    ntheta = int(ctx["ntheta"])
    nzeta = int(ctx["nzeta"])
    reuse_step = bool(ctx["reuse_step"])
    dense_solve_mode = os.getenv("VMEC_JAX_FREEB_DENSE_SOLVE_MODE", "mode").strip().lower()
    refresh_operator_on_reuse = bool(reuse_step and not ctx["provider_allows_source_reuse"])
    if (
        not isinstance(cache, NestorVmecLikeCache)
        or int(cache.ntheta) != ntheta
        or int(cache.nzeta) != nzeta
        or not reuse_step
        or refresh_operator_on_reuse
    ):
        t_phase = time.perf_counter()
        cache = _build_vmec_like_cache(
            sample,
            alpha=_as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_ALPHA", 1.0),
            dist_eps=_as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_DIST_EPS", 1.0e-8),
            rhs_floor=_as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_RHS_FLOOR", 1.0e-14),
            diag_coeff=_as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_DIAG_COEFF", 0.5),
            row_sum_zero=_as_int_env("VMEC_JAX_FREEB_VMEC_LIKE_ROW_SUM_ZERO", 1) != 0,
            singular_diag_scale=_as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_SINGULAR_DIAG_SCALE", 1.0),
            nfp=max(1, int(getattr(static.cfg, "nfp", 1))),
            mf=max(0, int(getattr(static.cfg, "mpol", 1)) + 1),
            nf=max(0, int(getattr(static.cfg, "ntor", 0))),
            lasym=bool(getattr(static.cfg, "lasym", False)),
            wint_vmec=np.asarray(ctx["wint_vmec"], dtype=float),
            factor_physical_matrix=dense_solve_mode not in ("mode", "vmec_mode", "fouri_mode"),
        )
        ctx["cache_build_time_s"] += max(0.0, time.perf_counter() - t_phase)
    return {
        "cache": cache,
        "dense_solve_mode": dense_solve_mode,
        "refresh_operator_on_reuse": refresh_operator_on_reuse,
    }


def _vmec_like_greenf_source_terms(ctx: dict[str, Any]) -> dict[str, Any]:
    sample = ctx["sample"]
    static = ctx["static"]
    cache = ctx["cache"]
    use_greenf_source = _freeb_use_greenf_source(int(getattr(static.cfg, "ntor", 0)))
    experimental_fouri_matrix = _env_truthy("VMEC_JAX_FREEB_EXPERIMENTAL_FOURI_MATRIX", default=True)
    refresh_source_on_reuse = bool(ctx["reuse_step"] and not ctx["provider_allows_source_reuse"])
    nzeta_surf = int(np.asarray(sample.R).shape[1])
    nvper_greenf = 64 if nzeta_surf == 1 else max(1, int(getattr(static.cfg, "nfp", 1)))
    if use_greenf_source and ((not ctx["reuse_step"]) or refresh_source_on_reuse) and cache.mode_basis is not None:
        try:
            t_phase = time.perf_counter()
            if experimental_fouri_matrix:
                ctx["gsource_vmec"], ctx["grpmn_nonsing"] = _vmec_nonsingular_terms_from_bexni(
                    sample=sample,
                    basis=cache.mode_basis,
                    bexni=np.asarray(ctx["gsource_bexni"], dtype=float),
                    signgs=int(getattr(static, "signgs", -1)),
                    nvper=nvper_greenf,
                )
            else:
                ctx["gsource_vmec"] = _vmec_nonsingular_gsource_from_bexni(
                    sample=sample,
                    basis=cache.mode_basis,
                    bexni=np.asarray(ctx["gsource_bexni"], dtype=float),
                    signgs=int(getattr(static, "signgs", -1)),
                    nvper=nvper_greenf,
                )
                ctx["grpmn_nonsing"] = None
            ctx["source_time_s"] += max(0.0, time.perf_counter() - t_phase)
        except Exception:
            ctx["gsource_vmec"] = np.asarray(ctx["gsource_bexni"], dtype=float)
            ctx["grpmn_nonsing"] = None
    return {
        "use_greenf_source": use_greenf_source,
        "experimental_fouri_matrix": experimental_fouri_matrix,
        "refresh_source_on_reuse": refresh_source_on_reuse,
        "nvper_greenf": nvper_greenf,
    }


def _maybe_replace_mode_matrix_from_grpmn(
    *,
    ctx: dict[str, Any],
    experimental_fouri_matrix: bool,
    refresh_operator_on_reuse: bool,
) -> None:
    cache = ctx["cache"]
    if not (((not ctx["reuse_step"]) or refresh_operator_on_reuse) and experimental_fouri_matrix and ctx["grpmn_nonsing"] is not None):
        return
    ctx["grpmn_total"] = np.asarray(ctx["grpmn_nonsing"], dtype=float)
    if ctx["grpmn_analytic"] is not None:
        ctx["grpmn_total"] = ctx["grpmn_total"] + np.asarray(ctx["grpmn_analytic"], dtype=float)
    try:
        ctx["amatrix_mode_pre"] = None if cache.mode_matrix is None else np.asarray(cache.mode_matrix, dtype=float)
        t_phase = time.perf_counter()
        ctx["amatrix_mode_from_grpmn"] = _vmec_mode_matrix_from_grpmn(
            grpmn=ctx["grpmn_total"],
            basis=cache.mode_basis,
        )
        ctx["cache"] = replace(
            cache,
            mode_matrix=np.asarray(ctx["amatrix_mode_from_grpmn"], dtype=float),
            mode_matrix_lu=_dense_lu_factor(np.asarray(ctx["amatrix_mode_from_grpmn"], dtype=float)),
        )
        ctx["matrix_time_s"] += max(0.0, time.perf_counter() - t_phase)
        ctx["matrix_override_applied"] = True
    except Exception:
        pass


def _maybe_solve_with_jax_nestor_operator(
    *,
    ctx: dict[str, Any],
    use_greenf_source: bool,
    experimental_fouri_matrix: bool,
    add_analytic: bool,
    nvper_greenf: int,
) -> bool:
    if not _env_truthy("VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR", False):
        return False
    sample = ctx["sample"]
    static = ctx["static"]
    cache = ctx["cache"]
    ctx["jax_nestor_operator_reason"] = "requested"
    if not (use_greenf_source and experimental_fouri_matrix):
        ctx["jax_nestor_operator_reason"] = "requires_greenf_fouri_matrix"
        return False
    if ctx["reuse_step"] and ctx["provider_allows_source_reuse"]:
        ctx["jax_nestor_operator_reason"] = "skip_cached_reuse_step"
        return False
    ok, reason = _jax_nestor_operator_guard(sample=sample, basis=cache.mode_basis)
    ctx["jax_nestor_operator_reason"] = reason
    if not ok:
        return False
    try:
        t_phase = time.perf_counter()
        (
            phi,
            potvac,
            rhs_mode_eff,
            amatrix_mode_from_grpmn,
            grpmn_total,
            gsource_vmec,
            jax_operator_jitted,
            jax_operator_cache_hit,
        ) = _solve_vmec_like_mode_with_jax_nestor_operator(
            sample=sample,
            basis=cache.mode_basis,
            bexni=np.asarray(ctx["gsource_bexni"], dtype=float),
            signgs=int(getattr(static, "signgs", -1)),
            nvper=nvper_greenf,
            include_analytic=add_analytic,
        )
        ctx["jax_nestor_operator_time_s"] += max(0.0, time.perf_counter() - t_phase)
        ctx["phi"] = phi
        ctx["potvac"] = potvac
        ctx["bvec_mode"] = np.asarray(rhs_mode_eff, dtype=float)
        ctx["amatrix_mode_pre"] = None if cache.mode_matrix is None else np.asarray(cache.mode_matrix, dtype=float)
        ctx["amatrix_mode_from_grpmn"] = amatrix_mode_from_grpmn
        ctx["grpmn_total"] = grpmn_total
        ctx["gsource_vmec"] = gsource_vmec
        ctx["cache"] = replace(
            cache,
            mode_matrix=np.asarray(amatrix_mode_from_grpmn, dtype=float),
            mode_matrix_lu=_dense_lu_factor(np.asarray(amatrix_mode_from_grpmn, dtype=float)),
        )
        ctx["matrix_override_applied"] = True
        ctx["jax_nestor_operator_applied"] = True
        ctx["jax_nestor_operator_jitted"] = bool(jax_operator_jitted)
        ctx["jax_nestor_operator_cache_hit"] = bool(jax_operator_cache_hit)
        ctx["jax_nestor_operator_reason"] = "applied"
        return True
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        ctx["jax_nestor_operator_reason"] = f"failed:{detail}"
        return False


def _solve_vmec_like_mode_nestor_step(
    *,
    ctx: dict[str, Any],
    use_greenf_source: bool,
    experimental_fouri_matrix: bool,
    refresh_operator_on_reuse: bool,
    nvper_greenf: int,
) -> None:
    cache = ctx["cache"]
    rhs_mode_eff = None
    if cache.mode_basis is not None:
        if ctx["reuse_step"] and ctx["provider_allows_source_reuse"] and ctx["runtime_bvec_nonsing_cached"] is not None:
            ctx["bvec_mode_nonsing"] = np.asarray(ctx["runtime_bvec_nonsing_cached"], dtype=float)
        else:
            t_phase = time.perf_counter()
            ctx["bvec_mode_nonsing"] = _vmec_bvec_from_gsource(
                gsource=np.asarray(ctx["gsource_vmec"], dtype=float),
                basis=cache.mode_basis,
            )
            ctx["bvec_time_s"] += max(0.0, time.perf_counter() - t_phase)
        rhs_mode_eff = np.asarray(ctx["bvec_mode_nonsing"], dtype=float)
        add_analytic = _env_truthy("VMEC_JAX_FREEB_ADD_ANALYTIC_BVEC", default=True)
        if add_analytic:
            t_phase = time.perf_counter()
            ctx["bvec_mode_analytic"], ctx["grpmn_analytic"] = _vmec_analytic_terms_from_geometry(
                sample=ctx["sample"],
                basis=cache.mode_basis,
                bexni=np.asarray(ctx["gsource_bexni"], dtype=float),
                signgs=int(getattr(ctx["static"], "signgs", -1)),
            )
            rhs_mode_eff = rhs_mode_eff + np.asarray(ctx["bvec_mode_analytic"], dtype=float)
            ctx["bvec_time_s"] += max(0.0, time.perf_counter() - t_phase)
        _maybe_replace_mode_matrix_from_grpmn(
            ctx=ctx,
            experimental_fouri_matrix=experimental_fouri_matrix,
            refresh_operator_on_reuse=refresh_operator_on_reuse,
        )
        if _maybe_solve_with_jax_nestor_operator(
            ctx=ctx,
            use_greenf_source=use_greenf_source,
            experimental_fouri_matrix=experimental_fouri_matrix,
            add_analytic=add_analytic,
            nvper_greenf=nvper_greenf,
        ):
            return
    t_phase = time.perf_counter()
    ctx["phi"], ctx["potvac"], ctx["bvec_mode"] = _solve_vmec_like_mode_from_gsource(
        cache=ctx["cache"],
        gsource=np.asarray(ctx["gsource_vmec"], dtype=float),
        rhs_mode=rhs_mode_eff,
    )
    ctx["linear_solve_time_s"] += max(0.0, time.perf_counter() - t_phase)


def _solve_vmec_like_nestor_step(ctx: dict[str, Any]) -> None:
    cache_ctx = _ensure_vmec_like_cache_for_nestor_step(ctx)
    ctx.update(cache_ctx)
    cache = ctx["cache"]
    source_policy = _vmec_like_greenf_source_terms(ctx)
    if ctx["dense_solve_mode"] in ("mode", "vmec_mode", "fouri_mode"):
        _solve_vmec_like_mode_nestor_step(
            ctx=ctx,
            use_greenf_source=source_policy["use_greenf_source"],
            experimental_fouri_matrix=source_policy["experimental_fouri_matrix"],
            refresh_operator_on_reuse=ctx["refresh_operator_on_reuse"],
            nvper_greenf=source_policy["nvper_greenf"],
        )
        if cache.mode_basis is not None:
            t_phase = time.perf_counter()
            ctx["vac_total"] = _vacuum_channels_from_sample_potvac(
                sample=ctx["sample"],
                basis=ctx["cache"].mode_basis,
                potvac=np.asarray(ctx["potvac"], dtype=float),
            )
            ctx["vacuum_channels_time_s"] += max(0.0, time.perf_counter() - t_phase)
        else:
            _solve_nestor_vacuum_channels_from_phi(ctx)
    else:
        t_phase = time.perf_counter()
        ctx["phi"] = _solve_vmec_like_dense(ctx["rhs"], ctx["cache"])
        ctx["linear_solve_time_s"] += max(0.0, time.perf_counter() - t_phase)
        _solve_nestor_vacuum_channels_from_phi(ctx)
    ctx["used_mode"] = ctx["mode_for_step"]


def _solve_nestor_vacuum_channels_from_phi(ctx: dict[str, Any]) -> None:
    t_phase = time.perf_counter()
    ctx["vac_total"] = _vacuum_channels_from_sample_phi(ctx["sample"], ctx["phi"])
    ctx["vacuum_channels_time_s"] += max(0.0, time.perf_counter() - t_phase)


def _solve_nestor_fft_step(ctx: dict[str, Any]) -> None:
    cache = ctx["cache"]
    if (
        not isinstance(cache, NestorPoissonCache)
        or int(cache.ntheta) != int(ctx["ntheta"])
        or int(cache.nzeta) != int(ctx["nzeta"])
    ):
        t_phase = time.perf_counter()
        cache = _build_poisson_cache(ntheta=int(ctx["ntheta"]), nzeta=int(ctx["nzeta"]))
        ctx["cache_build_time_s"] += max(0.0, time.perf_counter() - t_phase)
    ctx["cache"] = cache
    t_phase = time.perf_counter()
    ctx["phi"] = _solve_periodic_poisson_fft(ctx["rhs"], cache)
    ctx["linear_solve_time_s"] += max(0.0, time.perf_counter() - t_phase)
    _solve_nestor_vacuum_channels_from_phi(ctx)
    ctx["used_mode"] = ctx["mode_for_step"]


def _solve_nestor_external_step(ctx: dict[str, Any]) -> None:
    if _is_dense_mode(ctx["mode_for_step"]):
        try:
            _solve_vmec_like_nestor_step(ctx)
        except Exception:
            t_phase = time.perf_counter()
            ctx["cache"] = _build_poisson_cache(ntheta=int(ctx["ntheta"]), nzeta=int(ctx["nzeta"]))
            ctx["cache_build_time_s"] += max(0.0, time.perf_counter() - t_phase)
            t_phase = time.perf_counter()
            ctx["phi"] = _solve_periodic_poisson_fft(ctx["rhs"], ctx["cache"])
            ctx["linear_solve_time_s"] += max(0.0, time.perf_counter() - t_phase)
            _solve_nestor_vacuum_channels_from_phi(ctx)
            ctx["used_mode"] = "spectral_poisson_external_only_fallback:dense_failed"
    else:
        _solve_nestor_fft_step(ctx)


def nestor_external_only_step(
    *,
    state: Any,
    static: Any,
    ivac: int,
    ivacskip: int | None = None,
    iter_idx: int | None = None,
    runtime: NestorRuntimeState | None = None,
    extcur: tuple[float, ...] | None = None,
    plascur: float = 0.0,
    axis_override: tuple[np.ndarray, np.ndarray] | None = None,
    external_field_provider_kind: str | None = None,
    external_field_provider_static: Any = None,
    external_field_provider_params: Any = None,
    collect_trace_arrays: bool = False,
) -> tuple[NestorSolveResult, NestorRuntimeState]:
    """Simplified NESTOR-style update/reuse with ivacskip-compatible behavior."""

    runtime_ctx = _nestor_runtime_context(runtime)
    force_rhs_reuse = _env_truthy("VMEC_JAX_FREEB_REUSE_RHS_UPDATE", default=True)
    if int(ivac) != 1 and runtime is not None and not force_rhs_reuse:
        return _nestor_legacy_fast_reuse(
            runtime=runtime,
            runtime_cache=runtime_ctx["runtime_cache"],
            runtime_mode=runtime_ctx["runtime_mode"],
        )

    ctx = _prepare_nestor_external_step_context(
        state=state,
        static=static,
        ivac=int(ivac),
        ivacskip=ivacskip,
        iter_idx=iter_idx,
        runtime=runtime,
        extcur=extcur,
        plascur=float(plascur),
        axis_override=axis_override,
        external_field_provider_kind=external_field_provider_kind,
        external_field_provider_static=external_field_provider_static,
        external_field_provider_params=external_field_provider_params,
        collect_trace_arrays=collect_trace_arrays,
    )
    _solve_nestor_external_step(ctx)
    return _nestor_external_step_result(ctx)


def sample_external_vacuum_diagnostics(
    *,
    state: Any,
    static: Any,
    extcur: tuple[float, ...] | None = None,
    plascur: float = 0.0,
) -> dict[str, Any]:
    """Sample external mgrid field on plasma boundary and derive vacuum channels.

    This is a WP2 diagnostic scaffold: it computes boundary geometry and
    VMEC-style surface field projections (`Bu/Bv/B^u/B^v/bsqvac`) using the
    external field only. NESTOR scalar-potential coupling is still pending.
    """

    out: dict[str, Any] = {
        "enabled": False,
        "available": False,
        "vacuum_stub": True,
    }

    t0 = time.perf_counter()
    try:
        sample = _sample_external_boundary_arrays(
            state=state,
            static=static,
            extcur=extcur,
            plascur=float(plascur),
        )
        vac = sample.vac_ext
        br = sample.br
        bp = sample.bp
        bz = sample.bz
        bmag = np.sqrt(br * br + bp * bp + bz * bz)
        out.update(
            {
                "enabled": True,
                "available": True,
                "mgrid_path": sample.mgrid_path,
                "n_samples": int(br.size),
                "br_rms": float(np.sqrt(np.mean(br * br))),
                "bp_rms": float(np.sqrt(np.mean(bp * bp))),
                "bz_rms": float(np.sqrt(np.mean(bz * bz))),
                "br_axis_rms": float(np.sqrt(np.mean(sample.br_axis * sample.br_axis))),
                "bp_axis_rms": float(np.sqrt(np.mean(sample.bp_axis * sample.bp_axis))),
                "bz_axis_rms": float(np.sqrt(np.mean(sample.bz_axis * sample.bz_axis))),
                "bmag_mean": float(np.mean(bmag)),
                "bmag_max": float(np.max(bmag)),
                "bu_rms": float(np.sqrt(np.mean(vac.bu * vac.bu))),
                "bv_rms": float(np.sqrt(np.mean(vac.bv * vac.bv))),
                "bsupu_rms": float(np.sqrt(np.mean(vac.bsupu * vac.bsupu))),
                "bsupv_rms": float(np.sqrt(np.mean(vac.bsupv * vac.bsupv))),
                "bsqvac_mean": float(np.mean(vac.bsqvac)),
                "bsqvac_max": float(np.max(vac.bsqvac)),
                "bnormal_rms": float(np.sqrt(np.mean(vac.bnormal * vac.bnormal))),
                "bnormal_unit_rms": float(np.sqrt(np.mean(vac.bnormal_unit * vac.bnormal_unit))),
                "det_guv_min": float(np.min(vac.det_guv)),
                "det_guv_max": float(np.max(vac.det_guv)),
            }
        )
    except Exception as exc:
        out["reason"] = "sample_failed"
        out["error"] = str(exc)
    out["sample_time_s"] = float(max(0.0, time.perf_counter() - t0))
    return out


def prepare_mgrid_for_config(
    cfg: VMECConfig,
    *,
    load_fields: bool = False,
    strict: bool = True,
) -> PreparedMGrid | MGridData | None:
    """Load and validate mgrid against VMECConfig.

    This wrapper intentionally uses the currently bound facade `load_mgrid`
    symbol so tests/downstream users can monkeypatch `vmec_jax.free_boundary`
    without reaching into the internal helper module.
    """

    return _prepare_mgrid_for_config_impl(
        cfg,
        load_fields=load_fields,
        strict=strict,
        load_mgrid_fn=load_mgrid,
        validate_config_fn=validate_free_boundary_config,
    )
