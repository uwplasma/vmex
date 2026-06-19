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
    boundary_metric_from_rz,  # noqa: F401 - public compatibility facade
    contravariant_boundary_field_from_covariant,
    covariant_boundary_field_from_cylindrical,  # noqa: F401 - public compatibility facade
    sample_free_boundary_external_field,  # noqa: F401 - public compatibility facade
    vacuum_boundary_fields_from_cylindrical,
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
    compact_jax_nestor_basis as _compact_jax_nestor_basis,  # noqa: F401 - compatibility facade
    digest_array_for_cache as _digest_array_for_cache,  # noqa: F401 - compatibility facade
    jax_nestor_input_signature as _jax_nestor_input_signature,  # noqa: F401 - compatibility facade
    jax_nestor_operator_cache_key as _jax_nestor_operator_cache_key,  # noqa: F401 - compatibility facade
    jax_nestor_operator_guard as _jax_nestor_operator_guard,
    jitted_jax_nestor_operator as _jitted_jax_nestor_operator,  # noqa: F401 - compatibility facade
    mapping_cache_signature as _mapping_cache_signature,  # noqa: F401 - compatibility facade
    solve_vmec_like_mode_with_jax_nestor_operator as _solve_vmec_like_mode_with_jax_nestor_operator_impl,
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

try:  # pragma: no cover - optional dependency
    from scipy.linalg import lu_factor as _SCIPY_LU_FACTOR  # type: ignore
    from scipy.linalg import lu_solve as _SCIPY_LU_SOLVE  # type: ignore
except Exception:  # pragma: no cover - SciPy is optional at runtime
    _SCIPY_LU_FACTOR = None
    _SCIPY_LU_SOLVE = None

_FREEB_HOST_PHASE_CACHE: dict[tuple[int, int, tuple[str, ...]], np.ndarray] = {}
_FREEB_TRIG_CACHE: dict[tuple[int, int, int, int, int, bool], Any] = {}
_FREEB_BOUNDARY_SETUP_CACHE: dict[tuple[int, int], Any] = {}
_FREEB_WINT_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _freeb_host_phase_stack(*, modes: Any, trig: Any, derivs: tuple[str, ...]) -> np.ndarray:
    """Return cached NumPy phase stacks for host-side boundary synthesis."""

    key = (id(modes), id(trig), tuple(derivs))
    cached = _FREEB_HOST_PHASE_CACHE.get(key)
    if cached is not None:
        # Guard against Python id-reuse: a GC'd (modes, trig) pair may share the
        # same id as a new pair with different shapes.  Validate the cached array.
        # Expected shape: (len(derivs), 2*K, ntheta3, nzeta).
        try:
            K = int(np.asarray(modes.m).shape[0])
            nzeta = int(np.asarray(trig.cosnv).shape[0])
            ntheta3 = int(np.asarray(trig.cosmu).shape[0])
            if (
                int(cached.shape[1]) != 2 * K
                or int(cached.shape[2]) != ntheta3
                or int(cached.shape[3]) != nzeta
            ):
                cached = None
        except Exception:
            cached = None
    if cached is not None:
        return cached

    m = np.asarray(modes.m, dtype=np.int32)
    n = np.asarray(modes.n, dtype=np.int32)
    n1 = np.abs(n)
    sgn = np.where(n < 0, -1.0, 1.0)

    cosmu = np.asarray(trig.cosmu, dtype=float)
    sinmu = np.asarray(trig.sinmu, dtype=float)
    cosmum = np.asarray(trig.cosmum, dtype=float)
    sinmum = np.asarray(trig.sinmum, dtype=float)
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)
    cosnvn = np.asarray(trig.cosnvn, dtype=float)
    sinnvn = np.asarray(trig.sinnvn, dtype=float)

    cosmu_m = cosmu[:, m].T
    sinmu_m = sinmu[:, m].T
    cosmum_m = cosmum[:, m].T
    sinmum_m = sinmum[:, m].T
    cosnv_n = cosnv[:, n1].T
    sinnv_n = sinnv[:, n1].T
    cosnvn_n = cosnvn[:, n1].T
    sinnvn_n = sinnvn[:, n1].T

    phase_blocks: list[np.ndarray] = []
    for deriv in derivs:
        if deriv == "base":
            cos_phase = cosmu_m[:, :, None] * cosnv_n[:, None, :] + sgn[:, None, None] * sinmu_m[:, :, None] * sinnv_n[:, None, :]
            sin_phase = sinmu_m[:, :, None] * cosnv_n[:, None, :] - sgn[:, None, None] * cosmu_m[:, :, None] * sinnv_n[:, None, :]
        elif deriv == "dtheta":
            cos_phase = sinmum_m[:, :, None] * cosnv_n[:, None, :] + sgn[:, None, None] * cosmum_m[:, :, None] * sinnv_n[:, None, :]
            sin_phase = cosmum_m[:, :, None] * cosnv_n[:, None, :] - sgn[:, None, None] * sinmum_m[:, :, None] * sinnv_n[:, None, :]
        elif deriv == "dzeta":
            cos_phase = cosmu_m[:, :, None] * sinnvn_n[:, None, :] + sgn[:, None, None] * sinmu_m[:, :, None] * cosnvn_n[:, None, :]
            sin_phase = sinmu_m[:, :, None] * sinnvn_n[:, None, :] - sgn[:, None, None] * cosmu_m[:, :, None] * cosnvn_n[:, None, :]
        else:  # pragma: no cover
            raise ValueError(f"Unknown deriv={deriv!r}")
        phase_blocks.append(np.concatenate([cos_phase, sin_phase], axis=0))

    phase_all = np.stack(phase_blocks, axis=0)
    _FREEB_HOST_PHASE_CACHE[key] = phase_all
    return phase_all


def _freeb_boundary_trig(*, cfg: VMECConfig, nzeta: int) -> Any:
    """Return cached trig tables for free-boundary boundary sampling."""

    from .vmec_tomnsp import vmec_trig_tables

    key = (
        int(cfg.ntheta),
        int(nzeta),
        int(cfg.nfp),
        int(cfg.mpol) - 1,
        int(cfg.ntor),
        bool(cfg.lasym),
    )
    cached = _FREEB_TRIG_CACHE.get(key)
    if cached is not None:
        return cached
    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(nzeta),
        nfp=int(cfg.nfp),
        mmax=int(cfg.mpol) - 1,
        nmax=int(cfg.ntor),
        lasym=bool(cfg.lasym),
    )
    _FREEB_TRIG_CACHE[key] = trig
    return trig


def _vmec_realspace_synthesis_multi_host(
    *,
    coeff_cos: np.ndarray,
    coeff_sin: np.ndarray,
    modes: Any,
    trig: Any,
    derivs: tuple[str, ...] = ("base",),
) -> tuple[np.ndarray, ...]:
    """Host-side VMEC synthesis for external boundary sampling."""

    coeff_cos = np.asarray(coeff_cos, dtype=float)
    coeff_sin = np.asarray(coeff_sin, dtype=float)
    if coeff_cos.shape != coeff_sin.shape:
        raise ValueError("coeff_cos and coeff_sin must have the same shape")
    phase_all = _freeb_host_phase_stack(modes=modes, trig=trig, derivs=tuple(derivs))
    coeff = np.concatenate([coeff_cos, coeff_sin], axis=-1)
    out = np.einsum("...k,tkij->t...ij", coeff, phase_all, optimize=True)
    return tuple(np.asarray(out[i], dtype=float) for i in range(len(derivs)))


def _freeb_boundary_sample_setup(*, static: Any, sample_nzeta: int) -> _FreeBoundarySampleSetup:
    """Return cached static data used by host-side free-boundary sampling."""

    _trig_hint = getattr(static, "trig_vmec", None)
    _ntheta3_hint = int(_trig_hint.ntheta3) if _trig_hint is not None else -1
    key = (id(static), _ntheta3_hint, int(sample_nzeta))
    cached = _FREEB_BOUNDARY_SETUP_CACHE.get(key)
    if cached is not None:
        return cached

    trig = getattr(static, "trig_vmec", None)
    if trig is None or int(sample_nzeta) != int(static.cfg.nzeta):
        trig = _freeb_boundary_trig(cfg=static.cfg, nzeta=int(sample_nzeta))

    m_arr = np.asarray(static.modes.m, dtype=float)
    n_arr = np.asarray(static.modes.n, dtype=float) * float(int(static.cfg.nfp))
    second_facs = np.stack(
        [
            (-(m_arr**2)).reshape((1, -1)),
            (m_arr * n_arr).reshape((1, -1)),
            (-(n_arr**2)).reshape((1, -1)),
        ],
        axis=0,
    )

    ntheta = int(np.asarray(trig.cosmu).shape[0])
    nzeta = max(1, int(sample_nzeta))
    phi = (2.0 * np.pi / float(nzeta)) * np.arange(nzeta, dtype=float)
    phi /= float(max(1, int(static.cfg.nfp)))
    phi_grid = np.broadcast_to(phi[None, :], (ntheta, nzeta))

    if getattr(static, "m_is_even", None) is not None:
        even_m_mask = np.asarray(static.m_is_even, dtype=float).reshape((1, 1, -1))
    else:
        even_m_mask = (np.asarray(static.modes.m, dtype=int) % 2 == 0).astype(float).reshape((1, 1, -1))

    setup = _FreeBoundarySampleSetup(
        trig=trig,
        second_facs=second_facs,
        phi_grid=phi_grid,
        even_m_mask=even_m_mask,
        wint_vmec=_vmec_boundary_wint(static=static, ntheta=ntheta, nzeta=nzeta, trig=trig),
    )
    _FREEB_BOUNDARY_SETUP_CACHE[key] = setup
    return setup


def _dense_lu_factor(matrix: np.ndarray) -> Any | None:
    if _SCIPY_LU_FACTOR is None:
        return None
    try:
        return _SCIPY_LU_FACTOR(np.asarray(matrix, dtype=float))
    except Exception:
        return None


def _dense_lu_solve(lu_fac: Any | None, matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    rhs_arr = np.asarray(rhs, dtype=float)
    if lu_fac is not None and _SCIPY_LU_SOLVE is not None:
        try:
            return np.asarray(_SCIPY_LU_SOLVE(lu_fac, rhs_arr), dtype=float)
        except Exception:
            pass
    return np.asarray(np.linalg.solve(np.asarray(matrix, dtype=float), rhs_arr), dtype=float)


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


def _vmec_boundary_wint(*, static: Any, ntheta: int, nzeta: int, trig: Any | None = None) -> np.ndarray:
    """Return VMEC angular weights on the free-boundary mesh."""

    key = (id(static), int(ntheta), int(nzeta))
    cached = _FREEB_WINT_CACHE.get(key)
    if cached is not None:
        return cached

    trig = getattr(static, "trig_vmec", None) if trig is None else trig
    if trig is not None:
        try:
            from .vmec_residue import vmec_wint_from_trig

            w = np.asarray(vmec_wint_from_trig(trig, nzeta=int(nzeta)), dtype=float)
            if w.shape == (int(ntheta), int(nzeta)):
                _FREEB_WINT_CACHE[key] = w
                return w
        except Exception:
            pass
    w = np.full((int(ntheta), int(nzeta)), 1.0 / float(max(1, int(ntheta) * int(nzeta))), dtype=float)
    _FREEB_WINT_CACHE[key] = w
    return w


def _build_vmec_cmns(*, mf: int, nf: int, onp: float) -> np.ndarray:
    """VMEC precal.f cmns(l,m,n) coefficients (n>=0 block)."""

    mf = max(0, int(mf))
    nf = max(0, int(nf))
    lmax = mf + nf
    cmn = np.zeros((lmax + 1, mf + 1, nf + 1), dtype=float)
    for m in range(mf + 1):
        for n in range(nf + 1):
            jmn = m + n
            imn = m - n
            kmn = abs(imn)
            smn = (jmn + kmn) // 2
            f1 = 1.0
            f2 = 1.0
            f3 = 1.0
            for i in range(1, kmn + 1):
                f1 *= float(smn + 1 - i)
                f2 *= float(i)
            for l in range(kmn, jmn + 1, 2):
                cmn[l, m, n] = (f1 / (f2 * f3)) * ((-1.0) ** ((l - imn) // 2))
                f1 = f1 * 0.25 * float((jmn + l + 2) * (jmn - l))
                f2 = f2 * 0.5 * float(l + 2 + kmn)
                f3 = f3 * 0.5 * float(l + 2 - kmn)

    alp = 2.0 * np.pi * float(onp)
    cmns = np.zeros_like(cmn)
    if mf >= 1 and nf >= 1:
        cmns[:, 1 : mf + 1, 1 : nf + 1] = (
            0.5
            * alp
            * (
                cmn[:, 1 : mf + 1, 1 : nf + 1]
                + cmn[:, :mf, 1 : nf + 1]
                + cmn[:, 1 : mf + 1, :nf]
                + cmn[:, :mf, :nf]
            )
        )
    if mf >= 1:
        cmns[:, 1 : mf + 1, 0] = 0.5 * alp * (cmn[:, 1 : mf + 1, 0] + cmn[:, :mf, 0])
    if nf >= 1:
        cmns[:, 0, 1 : nf + 1] = 0.5 * alp * (cmn[:, 0, 1 : nf + 1] + cmn[:, 0, :nf])
    cmns[:, 0, 0] = 0.5 * alp * (cmn[:, 0, 0] + cmn[:, 0, 0])
    return cmns


def _build_vmec_mode_basis(
    *,
    ntheta: int,
    nzeta: int,
    nfp: int,
    mf: int,
    nf: int,
    lasym: bool,
    wint: np.ndarray,
) -> dict[str, Any]:
    """Build VMEC-like mode tables and weighted sin/cos basis arrays."""

    ntheta = int(ntheta)
    nzeta = int(nzeta)
    nfp = max(1, int(nfp))
    mf = max(0, int(mf))
    nf = max(0, int(nf))
    lasym = bool(lasym)

    pi2 = 2.0 * np.pi
    # VMEC vacuum precal uses a full poloidal grid `nu`; for stellarator-symmetric
    # solves (`lasym=F`) the active storage is only the first `nu3 = nu/2 + 1` rows.
    # Our free-boundary boundary arrays live on that reduced grid, so recover `nu`
    # from `nu3` to match precal sinmni/cosmni phases exactly.
    if lasym:
        nu_full = int(ntheta)
    else:
        nu_full = max(int(ntheta), 2 * (int(ntheta) - 1))
    theta = (pi2 / float(max(1, nu_full))) * np.arange(ntheta, dtype=float)
    zeta = (pi2 / float(max(1, nzeta))) * np.arange(nzeta, dtype=float)
    th_grid = np.broadcast_to(theta[:, None], (ntheta, nzeta))
    ze_grid = np.broadcast_to(zeta[None, :], (ntheta, nzeta))
    th = th_grid.reshape(-1)
    ze = ze_grid.reshape(-1)

    w = np.asarray(wint, dtype=float).reshape(-1)
    if w.size != th.size:
        w = np.full((th.size,), 1.0 / float(max(1, th.size)), dtype=float)

    mvals: list[int] = []
    nvals: list[int] = []
    for n in range(-nf, nf + 1):
        for m in range(0, mf + 1):
            mvals.append(int(m))
            nvals.append(int(n))
    xmpot = np.asarray(mvals, dtype=np.int64)
    n_raw = np.asarray(nvals, dtype=np.int64)
    xnpot = np.asarray(n_raw * nfp, dtype=np.int64)
    mnpd = int(xmpot.size)
    mnpd2 = int(mnpd * (2 if lasym else 1))

    phase = (xmpot[None, :] * th[:, None]) - (n_raw[None, :] * ze[:, None])
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)
    weight = ((pi2 * pi2) * w)[:, None]
    sinmni = weight * sin_phase
    cosmni = weight * cos_phase

    idx = np.arange(th.size, dtype=np.int64)
    lt = idx // max(1, nzeta)
    lz = idx % max(1, nzeta)
    if lasym or (nu_full == ntheta):
        lt_m = (ntheta - lt) % max(1, ntheta)
    else:
        lt_m_full = (nu_full - lt) % max(1, nu_full)
        lt_m = np.minimum(lt_m_full, (nu_full - lt_m_full) % max(1, nu_full))
    lz_m = (nzeta - lz) % max(1, nzeta)
    imirr = (lt_m * nzeta + lz_m).astype(np.int64)
    nuv_full = int(max(1, nu_full) * max(1, nzeta))
    idx_full = np.arange(nuv_full, dtype=np.int64)
    ku_full = idx_full // max(1, nzeta)
    kv_full = idx_full % max(1, nzeta)
    ku_m_full = (nu_full - ku_full) % max(1, nu_full)
    kv_m_full = (nzeta - kv_full) % max(1, nzeta)
    imirr_full = (ku_m_full * nzeta + kv_m_full).astype(np.int64)

    mn0 = 0
    for j in range(mnpd):
        if int(xmpot[j]) == 0 and int(n_raw[j]) == 0:
            mn0 = int(j)
            break

    return {
        "xmpot": xmpot,
        "xnpot": xnpot,
        "n_raw": n_raw,
        "sin_phase": sin_phase,
        "cos_phase": cos_phase,
        "sinmni": sinmni,
        "cosmni": cosmni,
        "wint": w,
        "imirr": imirr,
        "imirr_full": imirr_full,
        "mnpd": mnpd,
        "mnpd2": mnpd2,
        "nuv3": int(th.size),
        "nuv_full": nuv_full,
        "mn0": mn0,
        "onp": 1.0 / float(nfp),
        "nfp": nfp,
        "mf": mf,
        "nf": nf,
        "nu_full": int(nu_full),
        "lasym": lasym,
        "theta": th,
        "zeta": ze,
        "cmns": _build_vmec_cmns(mf=mf, nf=nf, onp=1.0 / float(nfp)),
    }


def _build_poisson_cache(*, ntheta: int, nzeta: int) -> NestorPoissonCache:
    """Build spectral Laplacian eigenvalues on periodic `(theta,zeta)` grid."""

    ntheta = int(ntheta)
    nzeta = int(nzeta)
    ku = 2.0 * np.pi * np.fft.fftfreq(ntheta)
    kv = 2.0 * np.pi * np.fft.fftfreq(nzeta)
    ku2 = ku[:, None] * ku[:, None]
    kv2 = kv[None, :] * kv[None, :]
    lam = ku2 + kv2
    lam[0, 0] = 1.0
    return NestorPoissonCache(ntheta=ntheta, nzeta=nzeta, lam=lam)


def _build_vmec_like_cache(
    sample: ExternalBoundarySample,
    *,
    alpha: float,
    dist_eps: float,
    rhs_floor: float,
    diag_coeff: float,
    row_sum_zero: bool,
    singular_diag_scale: float,
    nfp: int,
    mf: int,
    nf: int,
    lasym: bool,
    wint_vmec: np.ndarray | None = None,
    factor_physical_matrix: bool = True,
) -> NestorVmecLikeCache:
    """Build a dense boundary-integral-like operator on the VMEC angular grid.

    This mirrors NESTOR's matrix-assembly style (dense Green-function operator
    over boundary samples), while remaining a compact NumPy implementation.
    """

    R = np.asarray(sample.R, dtype=float)
    Z = np.asarray(sample.Z, dtype=float)
    ntheta, nzeta = R.shape
    npts = int(ntheta * nzeta)
    phi_grid = np.asarray(sample.phi, dtype=float)
    if phi_grid.shape != R.shape:
        phi_grid = np.broadcast_to(phi_grid, R.shape)
    x = R * np.cos(phi_grid)
    y = R * np.sin(phi_grid)
    coords = np.stack([x, y, Z], axis=-1).reshape(npts, 3)
    det = np.asarray(sample.vac_ext.det_guv, dtype=float)
    # Surface-area-like quadrature weights; normalization keeps conditioning sane.
    w = np.sqrt(np.maximum(np.abs(det), 0.0)).reshape(npts)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= rhs_floor:
        w = np.full((npts,), 1.0 / float(max(1, npts)), dtype=float)
    else:
        w = w / w_sum

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + float(dist_eps) ** 2)
    invdist = np.where(dist > 0.0, 1.0 / dist, 0.0)
    np.fill_diagonal(invdist, 0.0)

    kernel = (invdist * w[None, :]) / (4.0 * np.pi)
    if bool(row_sum_zero):
        # Principal-value-inspired stabilization: remove each row mean contribution
        # from the diagonal so constant offsets do not spuriously dominate.
        row_sum = np.sum(kernel, axis=1)
        kernel[np.arange(npts), np.arange(npts)] -= row_sum

    diag_extra = np.zeros((npts,), dtype=float)
    if float(singular_diag_scale) != 0.0:
        # Local nearest-neighbor scale as a compact proxy for the singular
        # self-panel contribution in VMEC/NESTOR diagonal treatment.
        dist_nodiag = np.asarray(dist, dtype=float).copy()
        np.fill_diagonal(dist_nodiag, np.inf)
        h = np.minimum(np.min(dist_nodiag, axis=1), 1.0 / float(max(1, npts)))
        h = np.maximum(h, float(dist_eps))
        diag_extra = (float(singular_diag_scale) / (4.0 * np.pi)) * (w / h)

    matrix = float(alpha) * kernel
    matrix[np.arange(npts), np.arange(npts)] += float(diag_coeff) + diag_extra
    rhs_scale = np.where(w > rhs_floor, w, rhs_floor)

    wint_use = np.asarray(wint_vmec, dtype=float) if wint_vmec is not None else np.asarray(w, dtype=float).reshape(ntheta, nzeta)
    mode_basis = _build_vmec_mode_basis(
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=int(nfp),
        mf=int(mf),
        nf=int(nf),
        lasym=bool(lasym),
        wint=np.asarray(wint_use, dtype=float),
    )
    sinmni = np.asarray(mode_basis["sinmni"], dtype=float)
    cosmni = np.asarray(mode_basis["cosmni"], dtype=float)
    if bool(lasym):
        B = np.concatenate([sinmni, cosmni], axis=1)
    else:
        B = sinmni
    mode_matrix = B.T @ (matrix @ B)
    mnpd = int(mode_basis["mnpd"])
    if mnpd > 0:
        # VMEC/NESTOR `pi3` from precal.f: p5*pi2**3 = 4*pi**3.
        pi3 = float(4.0 * (np.pi**3))
        mode_matrix[:mnpd, :mnpd][np.diag_indices(mnpd)] += pi3
        if bool(lasym):
            mode_matrix[mnpd:, mnpd:][np.diag_indices(mnpd)] += pi3
            mn0 = int(mode_basis["mn0"])
            if 0 <= mn0 < mnpd:
                mode_matrix[mnpd + mn0, mnpd + mn0] += pi3

    return NestorVmecLikeCache(
        ntheta=ntheta,
        nzeta=nzeta,
        matrix=matrix,
        rhs_scale=rhs_scale,
        mode_basis=mode_basis,
        mode_matrix=mode_matrix,
        matrix_lu=_dense_lu_factor(matrix) if bool(factor_physical_matrix) else None,
        mode_matrix_lu=_dense_lu_factor(mode_matrix),
    )


def _solve_vmec_like_dense(rhs: np.ndarray, cache: NestorVmecLikeCache) -> np.ndarray:
    rhs_flat = np.asarray(rhs, dtype=float).reshape(-1) * np.asarray(cache.rhs_scale, dtype=float)
    phi_flat = _dense_lu_solve(cache.matrix_lu, np.asarray(cache.matrix, dtype=float), rhs_flat)
    phi = phi_flat.reshape(int(cache.ntheta), int(cache.nzeta))
    phi = phi - float(np.mean(phi))
    return phi


def _vmec_source_from_gsource(*, gsource: np.ndarray, basis: dict[str, Any]) -> np.ndarray:
    """VMEC fouri.f source symmetrization from gsource = -(2π)^2*B·dS*wint."""

    gsrc = np.asarray(gsource, dtype=float).reshape(-1)
    onp = float(basis["onp"])
    nuv3 = int(basis.get("nuv3", gsrc.size))
    nuv_full = int(basis.get("nuv_full", nuv3))
    if bool(basis["lasym"]):
        src = onp * gsrc[:nuv3]
    else:
        if gsrc.size >= nuv_full and "imirr_full" in basis:
            imirr_full = np.asarray(basis["imirr_full"], dtype=np.int64)
            src = 0.5 * onp * (gsrc[:nuv3] - gsrc[imirr_full[:nuv3]])
        else:
            imirr = np.asarray(basis["imirr"], dtype=np.int64)
            src = 0.5 * onp * (gsrc[:nuv3] - gsrc[imirr[:nuv3]])
    return np.asarray(src, dtype=float)


def _spectral_second_derivatives_2d(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodic spectral second derivatives on a uniform `(u,v)` grid."""

    f = np.asarray(field, dtype=float)
    nu, nv = f.shape
    ku = np.fft.fftfreq(nu, d=1.0 / float(max(1, nu)))
    kv = np.fft.fftfreq(nv, d=1.0 / float(max(1, nv)))
    fh = np.fft.fftn(f)
    duu = np.fft.ifftn((-(ku[:, None] ** 2)) * fh).real
    dvv = np.fft.ifftn((-(kv[None, :] ** 2)) * fh).real
    duv = np.fft.ifftn((-(ku[:, None] * kv[None, :])) * fh).real
    return np.asarray(duu, dtype=float), np.asarray(duv, dtype=float), np.asarray(dvv, dtype=float)


def _vmec_precal_tan_tables(*, nu: int, nv: int, nvper: int) -> tuple[np.ndarray, np.ndarray]:
    """VMEC ``precal.f`` tan tables used by ``greenf.f``."""

    nu = max(1, int(nu))
    nv = max(1, int(nv))
    nvper = max(1, int(nvper))
    kp_count = int(nvper) if int(nv) == 1 else 1
    nuv_tan = int(2 * nu * nv * kp_count)
    tanu = np.zeros((nuv_tan,), dtype=float)
    tanv = np.zeros((nuv_tan,), dtype=float)
    alu = 2.0 * np.pi / float(nu)
    alv = 2.0 * np.pi / float(nv)
    alp_per = 2.0 * np.pi / float(nvper)
    epstan = np.finfo(float).eps
    bigno = 1.0e50
    i = 0
    for kp in range(1, kp_count + 1):
        argp = 0.5 * alp_per * float(kp - 1)
        for ku in range(1, 2 * nu + 1):
            argu = 0.5 * alu * float(ku - 1)
            near_qpi = abs(argu - 0.25 * 2.0 * np.pi) < epstan
            near_3qpi = abs(argu - 0.75 * 2.0 * np.pi) < epstan
            for kv in range(1, nv + 1):
                argv = 0.5 * alv * float(kv - 1) + argp
                if near_qpi or near_3qpi:
                    tanu[i] = bigno
                else:
                    tanu[i] = 2.0 * np.tan(argu)
                if abs(argv - 0.25 * 2.0 * np.pi) < epstan:
                    tanv[i] = bigno
                else:
                    tanv[i] = 2.0 * np.tan(argv)
                i += 1
    return np.asarray(tanu, dtype=float), np.asarray(tanv, dtype=float)


def _ensure_vmec_nonsingular_kernel_tables(*, basis: dict[str, Any], nv: int, nvper: int) -> dict[str, np.ndarray]:
    """Cache VMEC nonsingular Green-function helper tables on the mode basis."""

    nv = max(1, int(nv))
    nvper = max(1, int(nvper))
    cache = basis.get("_nonsingular_kernel_tables")
    if (
        isinstance(cache, dict)
        and int(cache.get("nv", -1)) == nv
        and int(cache.get("nvper", -1)) == nvper
    ):
        return cache

    nu = int(basis["nu_full"])
    mf = int(basis["mf"])
    nf = int(basis["nf"])
    onp = float(basis["onp"])
    nuv_full = int(basis["nuv_full"])

    tanu, tanv = _vmec_precal_tan_tables(nu=nu, nv=nv, nvper=nvper)

    alv = 2.0 * np.pi / float(max(1, nv))
    alvp = onp * alv
    kv = np.arange(nv, dtype=np.int64)
    cos_v = np.cos(alvp * kv)
    sin_v = np.sin(alvp * kv)
    cosuv = np.broadcast_to(cos_v[None, :], (nu, nv)).reshape(-1)
    sinuv = np.broadcast_to(sin_v[None, :], (nu, nv)).reshape(-1)

    alp_per = 2.0 * np.pi / float(max(1, nvper))
    cosper = np.cos(alp_per * np.arange(nvper, dtype=float))
    sinper = np.sin(alp_per * np.arange(nvper, dtype=float))

    cosv_tab = np.zeros((nf + 1, nv), dtype=float)
    sinv_tab = np.zeros((nf + 1, nv), dtype=float)
    kv_idx = np.arange(nv, dtype=float)
    for n in range(0, nf + 1):
        dn1 = alv * float(n)
        cosv_tab[n, :] = np.cos(dn1 * kv_idx)
        sinv_tab[n, :] = np.sin(dn1 * kv_idx)

    alu = 2.0 * np.pi / float(max(1, nu))
    nu_fourp = int(nu // 2 + 1)
    cosui = np.zeros((mf + 1, nu_fourp), dtype=float)
    sinui = np.zeros((mf + 1, nu_fourp), dtype=float)
    ku_idx = np.arange(nu_fourp, dtype=float)
    for m in range(0, mf + 1):
        c = np.cos(alu * float(m) * ku_idx)
        s = np.sin(alu * float(m) * ku_idx)
        cosui[m, :] = c * alu * alv * 2.0
        sinui[m, :] = s * alu * alv * 2.0
        cosui[m, 0] *= 0.5
        cosui[m, -1] *= 0.5

    cache = {
        "nv": np.asarray(nv, dtype=np.int64),
        "nvper": np.asarray(nvper, dtype=np.int64),
        "idx_all": np.arange(nuv_full, dtype=np.int64),
        "tanu": np.asarray(tanu, dtype=float),
        "tanv": np.asarray(tanv, dtype=float),
        "cosuv": np.asarray(cosuv, dtype=float),
        "sinuv": np.asarray(sinuv, dtype=float),
        "cosper": np.asarray(cosper, dtype=float),
        "sinper": np.asarray(sinper, dtype=float),
        "cosv_tab": np.asarray(cosv_tab, dtype=float),
        "sinv_tab": np.asarray(sinv_tab, dtype=float),
        "cosui": np.asarray(cosui, dtype=float),
        "sinui": np.asarray(sinui, dtype=float),
    }
    basis["_nonsingular_kernel_tables"] = cache
    return cache


def _vmec_nonsingular_gsource_from_bexni(
    *,
    sample: ExternalBoundarySample,
    basis: dict[str, Any],
    bexni: np.ndarray,
    signgs: int,
    nvper: int,
) -> np.ndarray:
    """Approximate VMEC greenf+gstore source assembly on one boundary period.

    This ports the key numerics of ``greenf.f``/``scalpot.f`` source accumulation:
    ``gstore(i) = sum_ip bexni(ip) * delgr(i;ip)``, where ``delgr`` is the
    non-singular Green-function remainder over field periods.
    """

    ntheta3, nzeta = sample.R.shape
    nu = int(basis.get("nu_full", ntheta3))
    nv = int(nzeta)
    nuv_full = int(nu * nv)
    nuv3 = int(ntheta3 * nv)
    if nuv_full <= 0:
        return np.zeros((0,), dtype=float)

    onp = float(basis["onp"])
    onp2 = onp * onp
    signgs = int(signgs)
    nvper = max(1, int(nvper))

    R_red = np.asarray(sample.R, dtype=float)
    Z_red = np.asarray(sample.Z, dtype=float)
    Ru_red = np.asarray(sample.Ru, dtype=float)
    Zu_red = np.asarray(sample.Zu, dtype=float)
    Rv_red = np.asarray(sample.Rv, dtype=float)
    Zv_red = np.asarray(sample.Zv, dtype=float)

    if (nu == ntheta3) or bool(basis.get("lasym", False)):
        R2 = np.asarray(R_red, dtype=float)
        Z2 = np.asarray(Z_red, dtype=float)
        Ru2 = np.asarray(Ru_red, dtype=float)
        Zu2 = np.asarray(Zu_red, dtype=float)
        Rv2 = np.asarray(Rv_red, dtype=float)
        Zv2 = np.asarray(Zv_red, dtype=float)
    else:
        # Rebuild full `nu` surface arrays from stellarator-symmetric half grid.
        R2 = np.zeros((nu, nv), dtype=float)
        Z2 = np.zeros((nu, nv), dtype=float)
        Ru2 = np.zeros((nu, nv), dtype=float)
        Zu2 = np.zeros((nu, nv), dtype=float)
        Rv2 = np.zeros((nu, nv), dtype=float)
        Zv2 = np.zeros((nu, nv), dtype=float)
        R2[:ntheta3, :] = R_red
        Z2[:ntheta3, :] = Z_red
        Ru2[:ntheta3, :] = Ru_red
        Zu2[:ntheta3, :] = Zu_red
        Rv2[:ntheta3, :] = Rv_red
        Zv2[:ntheta3, :] = Zv_red
        kv_m = (nv - np.arange(nv, dtype=np.int64)) % max(1, nv)
        for ku in range(1, max(1, ntheta3 - 1)):
            km = (nu - ku) % max(1, nu)
            if km < ntheta3:
                continue
            # Stellarator symmetry for missing half-grid rows:
            # (u,v) -> (-u,+v) maps to source rows sampled at (+u,-v).
            R2[km, :] = R_red[ku, kv_m]
            Z2[km, :] = -Z_red[ku, kv_m]
            Ru2[km, :] = -Ru_red[ku, kv_m]
            Zu2[km, :] = Zu_red[ku, kv_m]
            Rv2[km, :] = -Rv_red[ku, kv_m]
            Zv2[km, :] = Zv_red[ku, kv_m]

    # Prefer exact modal second derivatives from surface sampling (VMEC surface.f).
    # For stellarator-symmetric runs, source derivatives are only needed on the
    # reduced `ntheta3` rows (primed mesh), so we embed them into full arrays.
    have_second = (
        sample.ruu is not None
        and sample.ruv is not None
        and sample.rvv is not None
        and sample.zuu is not None
        and sample.zuv is not None
        and sample.zvv is not None
    )
    if have_second:
        ruu_s = np.asarray(sample.ruu, dtype=float)
        ruv_s = np.asarray(sample.ruv, dtype=float)
        rvv_s = np.asarray(sample.rvv, dtype=float)
        zuu_s = np.asarray(sample.zuu, dtype=float)
        zuv_s = np.asarray(sample.zuv, dtype=float)
        zvv_s = np.asarray(sample.zvv, dtype=float)
        if ruu_s.shape == R2.shape:
            ruu, ruv, rvv = ruu_s, ruv_s, rvv_s
            zuu, zuv, zvv = zuu_s, zuv_s, zvv_s
        elif ruu_s.shape == (ntheta3, nv):
            ruu = np.zeros_like(R2)
            ruv = np.zeros_like(R2)
            rvv = np.zeros_like(R2)
            zuu = np.zeros_like(R2)
            zuv = np.zeros_like(R2)
            zvv = np.zeros_like(R2)
            ruu[:ntheta3, :] = ruu_s
            ruv[:ntheta3, :] = ruv_s
            rvv[:ntheta3, :] = rvv_s
            zuu[:ntheta3, :] = zuu_s
            zuv[:ntheta3, :] = zuv_s
            zvv[:ntheta3, :] = zvv_s
        else:
            ruu, ruv, rvv = _spectral_second_derivatives_2d(R2)
            zuu, zuv, zvv = _spectral_second_derivatives_2d(Z2)
    else:
        ruu, ruv, rvv = _spectral_second_derivatives_2d(R2)
        zuu, zuv, zvv = _spectral_second_derivatives_2d(Z2)
    R = R2.reshape(-1)
    Z = Z2.reshape(-1)
    Ru = Ru2.reshape(-1)
    Zu = Zu2.reshape(-1)
    Rv = Rv2.reshape(-1)
    Zv = Zv2.reshape(-1)
    ruu = ruu.reshape(-1)
    rvv = rvv.reshape(-1)
    ruv = ruv.reshape(-1)
    zuu = zuu.reshape(-1)
    zvv = zvv.reshape(-1)
    zuv = zuv.reshape(-1)

    snr = float(signgs) * R * Zu
    snv = float(signgs) * (Ru * Zv - Rv * Zu)
    snz = -float(signgs) * R * Ru
    guu_b = Ru * Ru + Zu * Zu
    guv_b = (Ru * Rv + Zu * Zv) * onp * 2.0
    gvv_b = (Rv * Rv + Zv * Zv + R * R) * onp2
    auu = 0.5 * (snr * ruu + snz * zuu)
    auv = (snr * ruv + snv * Ru + snz * zuv) * onp
    avv = (snv * Rv + 0.5 * (snr * (rvv - R) + snz * zvv)) * onp2
    rzb2 = R * R + Z * Z

    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=nv, nvper=nvper)
    idx_all = np.asarray(tables["idx_all"], dtype=np.int64)
    tanu = np.asarray(tables["tanu"], dtype=float)
    tanv = np.asarray(tables["tanv"], dtype=float)
    cosuv = np.asarray(tables["cosuv"], dtype=float)
    sinuv = np.asarray(tables["sinuv"], dtype=float)
    cosper = np.asarray(tables["cosper"], dtype=float)
    sinper = np.asarray(tables["sinper"], dtype=float)
    rcosuv = R * cosuv
    rsinuv = R * sinuv

    bex = np.asarray(bexni, dtype=float).reshape(-1)
    if bex.size < nuv3:
        bex = np.resize(bex, (nuv3,))
    else:
        bex = bex[:nuv3]

    gstore = np.zeros((nuv_full,), dtype=float)
    for ip in range(nuv3):
        xip = rcosuv[ip]
        yip = rsinuv[ip]
        ivoff = nuv_full - ip
        iskip = ip // nv
        iuoff = nuv_full - nv * iskip

        gsave = rzb2[ip] + rzb2 - 2.0 * Z[ip] * Z
        delgr = np.zeros((nuv_full,), dtype=float)
        for kp in range(nvper):
            xper = xip * cosper[kp] - yip * sinper[kp]
            yper = yip * cosper[kp] + xip * sinper[kp]
            base = gsave - 2.0 * (xper * rcosuv + yper * rsinuv)
            if kp == 0 or nv == 1:
                tidx_u = idx_all + iuoff
                ivoff_k = ivoff + (2 * nu * kp if nv == 1 else 0)
                tidx_v = idx_all + ivoff_k
                ga1 = tanu[tidx_u] * (guu_b[ip] * tanu[tidx_u] + guv_b[ip] * tanv[tidx_v]) + gvv_b[ip] * tanv[tidx_v] * tanv[tidx_v]
                ga2 = tanu[tidx_u] * (auu[ip] * tanu[tidx_u] + auv[ip] * tanv[tidx_v]) + avv[ip] * tanv[tidx_v] * tanv[tidx_v]
                ga2 = ga2 / ga1
                ga1s = 1.0 / np.sqrt(ga1)
                mask = (idx_all != ip) if kp == 0 else np.ones_like(idx_all, dtype=bool)
                if np.any(mask):
                    base_m = base[mask]
                    htemp_m = np.sqrt(1.0 / base_m)
                    delgr[mask] += htemp_m - ga1s[mask]
            else:
                htemp = np.sqrt(1.0 / base)
                delgr += htemp
        # VMEC greenf.f: when nv==1, normalize the field-period sum by nvper.
        if nv == 1 and nvper > 1:
            delgr /= float(nvper)
        gstore += bex[ip] * delgr

    return np.asarray(gstore, dtype=float)


def _vmec_mode_matrix_from_grpmn(
    *,
    grpmn: np.ndarray,
    basis: dict[str, Any],
) -> np.ndarray:
    """Build VMEC mode-space matrix from `grpmn` using ``fouri.f`` formulas."""

    g = np.asarray(grpmn, dtype=float)
    mnpd = int(basis["mnpd"])
    lasym = bool(basis["lasym"])
    sinmni = np.asarray(basis["sinmni"], dtype=float)
    cosmni = np.asarray(basis["cosmni"], dtype=float)
    # VMEC/NESTOR `pi3` from precal.f: p5*pi2**3 = 4*pi**3.
    pi3 = float(4.0 * (np.pi**3))
    mn0 = int(basis.get("mn0", 0))

    if g.ndim != 2 or g.shape[0] < mnpd:
        raise ValueError("invalid_grpmn_shape")
    xmpot = np.asarray(basis["xmpot"], dtype=np.int64)
    n_raw = np.asarray(basis["n_raw"], dtype=np.int64)
    skip_col = np.logical_and(xmpot == 0, n_raw < 0)
    gsin = g[:mnpd, :]
    a11 = gsin @ sinmni
    a11 = np.asarray(a11, dtype=float)
    if np.any(skip_col):
        # fouri.f skips m=0,n<0 in the primed-mesh loop: these are column-only skips.
        a11[:, skip_col] = 0.0
    a11[np.diag_indices(mnpd)] += pi3

    if not lasym:
        return a11

    if g.shape[0] < 2 * mnpd:
        raise ValueError("invalid_grpmn_shape_lasym")
    gcos = g[mnpd : 2 * mnpd, :]
    a12 = gsin @ cosmni
    a21 = gcos @ sinmni
    a22 = gcos @ cosmni
    if np.any(skip_col):
        a12 = np.asarray(a12, dtype=float)
        a21 = np.asarray(a21, dtype=float)
        a22 = np.asarray(a22, dtype=float)
        a12[:, skip_col] = 0.0
        a21[:, skip_col] = 0.0
        a22[:, skip_col] = 0.0
    a22 = np.asarray(a22, dtype=float)
    a22[np.diag_indices(mnpd)] += pi3
    if 0 <= mn0 < mnpd:
        a22[mn0, mn0] += pi3

    out = np.zeros((2 * mnpd, 2 * mnpd), dtype=float)
    out[:mnpd, :mnpd] = a11
    out[:mnpd, mnpd:] = a12
    out[mnpd:, :mnpd] = a21
    out[mnpd:, mnpd:] = a22
    return out


def _vmec_nonsingular_terms_from_bexni(
    *,
    sample: ExternalBoundarySample,
    basis: dict[str, Any],
    bexni: np.ndarray,
    signgs: int,
    nvper: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute VMEC-like non-singular source and matrix kernel terms.

    Returns:
      - `gstore` (`gsource_full`) on the full `nu*nv` grid.
      - `grpmn_nonsing` Fourier-kernel contribution in mode space (`mnpd2,nuv3`).
    """

    ntheta3, nzeta = sample.R.shape
    nu = int(basis.get("nu_full", ntheta3))
    nv = int(nzeta)
    nuv_full = int(nu * nv)
    nuv3 = int(ntheta3 * nv)
    if nuv_full <= 0 or nuv3 <= 0:
        return np.zeros((0,), dtype=float), np.zeros((0, 0), dtype=float)

    mf = int(basis["mf"])
    nf = int(basis["nf"])
    mnpd = int(basis["mnpd"])
    lasym = bool(basis["lasym"])
    mnpd2 = int(basis["mnpd2"])
    onp = float(basis["onp"])
    signgs = int(signgs)
    nvper = max(1, int(nvper))

    R_red = np.asarray(sample.R, dtype=float)
    Z_red = np.asarray(sample.Z, dtype=float)
    Ru_red = np.asarray(sample.Ru, dtype=float)
    Zu_red = np.asarray(sample.Zu, dtype=float)
    Rv_red = np.asarray(sample.Rv, dtype=float)
    Zv_red = np.asarray(sample.Zv, dtype=float)

    if (nu == ntheta3) or lasym:
        R2 = np.asarray(R_red, dtype=float)
        Z2 = np.asarray(Z_red, dtype=float)
        Ru2 = np.asarray(Ru_red, dtype=float)
        Zu2 = np.asarray(Zu_red, dtype=float)
        Rv2 = np.asarray(Rv_red, dtype=float)
        Zv2 = np.asarray(Zv_red, dtype=float)
    else:
        R2 = np.zeros((nu, nv), dtype=float)
        Z2 = np.zeros((nu, nv), dtype=float)
        Ru2 = np.zeros((nu, nv), dtype=float)
        Zu2 = np.zeros((nu, nv), dtype=float)
        Rv2 = np.zeros((nu, nv), dtype=float)
        Zv2 = np.zeros((nu, nv), dtype=float)
        R2[:ntheta3, :] = R_red
        Z2[:ntheta3, :] = Z_red
        Ru2[:ntheta3, :] = Ru_red
        Zu2[:ntheta3, :] = Zu_red
        Rv2[:ntheta3, :] = Rv_red
        Zv2[:ntheta3, :] = Zv_red
        kv_m = (nv - np.arange(nv, dtype=np.int64)) % max(1, nv)
        for ku in range(1, max(1, ntheta3 - 1)):
            km = (nu - ku) % max(1, nu)
            if km < ntheta3:
                continue
            R2[km, :] = R_red[ku, kv_m]
            Z2[km, :] = -Z_red[ku, kv_m]
            Ru2[km, :] = -Ru_red[ku, kv_m]
            Zu2[km, :] = Zu_red[ku, kv_m]
            Rv2[km, :] = -Rv_red[ku, kv_m]
            Zv2[km, :] = Zv_red[ku, kv_m]

    have_second = (
        sample.ruu is not None
        and sample.ruv is not None
        and sample.rvv is not None
        and sample.zuu is not None
        and sample.zuv is not None
        and sample.zvv is not None
    )
    if have_second:
        ruu_s = np.asarray(sample.ruu, dtype=float)
        ruv_s = np.asarray(sample.ruv, dtype=float)
        rvv_s = np.asarray(sample.rvv, dtype=float)
        zuu_s = np.asarray(sample.zuu, dtype=float)
        zuv_s = np.asarray(sample.zuv, dtype=float)
        zvv_s = np.asarray(sample.zvv, dtype=float)
        if ruu_s.shape == R2.shape:
            ruu, ruv, rvv = ruu_s, ruv_s, rvv_s
            zuu, zuv, zvv = zuu_s, zuv_s, zvv_s
        elif ruu_s.shape == (ntheta3, nv):
            ruu = np.zeros_like(R2)
            ruv = np.zeros_like(R2)
            rvv = np.zeros_like(R2)
            zuu = np.zeros_like(R2)
            zuv = np.zeros_like(R2)
            zvv = np.zeros_like(R2)
            ruu[:ntheta3, :] = ruu_s
            ruv[:ntheta3, :] = ruv_s
            rvv[:ntheta3, :] = rvv_s
            zuu[:ntheta3, :] = zuu_s
            zuv[:ntheta3, :] = zuv_s
            zvv[:ntheta3, :] = zvv_s
        else:
            ruu, ruv, rvv = _spectral_second_derivatives_2d(R2)
            zuu, zuv, zvv = _spectral_second_derivatives_2d(Z2)
    else:
        ruu, ruv, rvv = _spectral_second_derivatives_2d(R2)
        zuu, zuv, zvv = _spectral_second_derivatives_2d(Z2)
    R = R2.reshape(-1)
    Z = Z2.reshape(-1)
    Ru = Ru2.reshape(-1)
    Zu = Zu2.reshape(-1)
    Rv = Rv2.reshape(-1)
    Zv = Zv2.reshape(-1)
    ruu = ruu.reshape(-1)
    rvv = rvv.reshape(-1)
    ruv = ruv.reshape(-1)
    zuu = zuu.reshape(-1)
    zvv = zvv.reshape(-1)
    zuv = zuv.reshape(-1)

    snr = float(signgs) * R * Zu
    snv = float(signgs) * (Ru * Zv - Rv * Zu)
    snz = -float(signgs) * R * Ru
    drv = -(R * snr + Z * snz)
    guu_b = Ru * Ru + Zu * Zu
    guv_b = (Ru * Rv + Zu * Zv) * onp * 2.0
    gvv_b = (Rv * Rv + Zv * Zv + R * R) * (onp * onp)
    auu = 0.5 * (snr * ruu + snz * zuu)
    auv = (snr * ruv + snv * Ru + snz * zuv) * onp
    avv = (snv * Rv + 0.5 * (snr * (rvv - R) + snz * zvv)) * (onp * onp)
    rzb2 = R * R + Z * Z

    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=nv, nvper=nvper)
    idx_all = np.asarray(tables["idx_all"], dtype=np.int64)
    tanu = np.asarray(tables["tanu"], dtype=float)
    tanv = np.asarray(tables["tanv"], dtype=float)
    cosuv = np.asarray(tables["cosuv"], dtype=float)
    sinuv = np.asarray(tables["sinuv"], dtype=float)
    cosper = np.asarray(tables["cosper"], dtype=float)
    sinper = np.asarray(tables["sinper"], dtype=float)
    cosv_tab = np.asarray(tables["cosv_tab"], dtype=float)
    sinv_tab = np.asarray(tables["sinv_tab"], dtype=float)
    cosui = np.asarray(tables["cosui"], dtype=float)
    sinui = np.asarray(tables["sinui"], dtype=float)
    nu_fourp = int(cosui.shape[1])
    rcosuv = R * cosuv
    rsinuv = R * sinuv

    bex = np.asarray(bexni, dtype=float).reshape(-1)
    if bex.size < nuv3:
        bex = np.resize(bex, (nuv3,))
    else:
        bex = bex[:nuv3]

    imirr_full = np.asarray(basis["imirr_full"], dtype=np.int64)
    grpmn_nonsing = np.zeros((mnpd2, nuv3), dtype=float)
    mf1 = mf + 1
    ndim = 2 if lasym else 1
    iuv_grid = (np.arange(int(nu_fourp), dtype=np.int64)[:, None] * int(nv)) + np.arange(int(nv), dtype=np.int64)[
        None, :
    ]
    iuv_grid = np.asarray(iuv_grid, dtype=np.int64)
    iref_grid = np.asarray(imirr_full[iuv_grid], dtype=np.int64)
    cosv_modes = 0.5 * onp * np.asarray(cosv_tab[: nf + 1, :], dtype=float)
    sinv_modes = 0.5 * onp * np.asarray(sinv_tab[: nf + 1, :], dtype=float)
    m_idx = np.arange(mf + 1, dtype=np.int64)
    n_idx = np.arange(nf + 1, dtype=np.int64)
    idx_p_grid = m_idx[:, None] + (n_idx[None, :] + nf) * mf1
    idx_m_grid = m_idx[:, None] + ((-n_idx[None, :]) + nf) * mf1
    add_negative_n = (n_idx[None, :] != 0) & (m_idx[:, None] != 0)
    idx_p_flat = idx_p_grid.reshape(-1)
    idx_m_flat = idx_m_grid.reshape(-1)
    negative_n_flat = np.asarray(add_negative_n.reshape(-1), dtype=bool)
    sinm_sym = np.asarray(sinui[: mf + 1, :], dtype=float)
    cosm_sym = -np.asarray(cosui[: mf + 1, :], dtype=float)
    sinm_asym = np.asarray(cosui[: mf + 1, :], dtype=float) if lasym else None
    cosm_asym = np.asarray(sinui[: mf + 1, :], dtype=float) if lasym else None

    try:
        ip_chunk = int(os.getenv("VMEC_JAX_FREEB_NONSINGULAR_IP_CHUNK", "64"))
    except Exception:
        ip_chunk = 64
    ip_chunk = max(1, min(int(ip_chunk), int(nuv3)))

    gstore = np.zeros((nuv_full,), dtype=float)
    idx_all_b = idx_all[None, :]
    rcosuv_b = rcosuv[None, :]
    rsinuv_b = rsinuv[None, :]
    z_b = Z[None, :]
    for ip0 in range(0, nuv3, ip_chunk):
        ip1 = min(nuv3, ip0 + ip_chunk)
        ip_idx = np.arange(ip0, ip1, dtype=np.int64)
        n_chunk = int(ip_idx.size)

        xip = rcosuv[ip_idx]
        yip = rsinuv[ip_idx]
        ivoff = nuv_full - ip_idx
        iskip = ip_idx // nv
        iuoff = nuv_full - nv * iskip
        gsave = rzb2[ip_idx, None] + rzb2[None, :] - 2.0 * Z[ip_idx, None] * z_b
        dsave = drv[ip_idx, None] + z_b * snz[ip_idx, None]
        delgr = np.zeros((n_chunk, nuv_full), dtype=float)
        delgrp = np.zeros((n_chunk, nuv_full), dtype=float)

        for kp in range(nvper):
            xper = xip * cosper[kp] - yip * sinper[kp]
            yper = yip * cosper[kp] + xip * sinper[kp]
            sxsave = (snr[ip_idx] * xper - snv[ip_idx] * yper) / R[ip_idx]
            sysave = (snr[ip_idx] * yper + snv[ip_idx] * xper) / R[ip_idx]
            base = gsave - 2.0 * (xper[:, None] * rcosuv_b + yper[:, None] * rsinuv_b)
            deriv_num = rcosuv_b * sxsave[:, None] + rsinuv_b * sysave[:, None] + dsave

            if kp == 0 or nv == 1:
                tidx_u = idx_all_b + iuoff[:, None]
                ivoff_k = ivoff + (2 * nu * kp if nv == 1 else 0)
                tidx_v = idx_all_b + ivoff_k[:, None]
                tanu_use = tanu[tidx_u]
                tanv_use = tanv[tidx_v]
                ga1 = tanu_use * (
                    guu_b[ip_idx, None] * tanu_use + guv_b[ip_idx, None] * tanv_use
                ) + gvv_b[ip_idx, None] * tanv_use * tanv_use
                ga2 = tanu_use * (
                    auu[ip_idx, None] * tanu_use + auv[ip_idx, None] * tanv_use
                ) + avv[ip_idx, None] * tanv_use * tanv_use
                ga2 = ga2 / ga1
                ga1s = 1.0 / np.sqrt(ga1)
                if kp == 0:
                    mask = np.ones((n_chunk, nuv_full), dtype=bool)
                    mask[np.arange(n_chunk, dtype=np.int64), ip_idx] = False
                else:
                    mask = np.ones((n_chunk, nuv_full), dtype=bool)
                safe_base = np.where(mask, base, 1.0)
                ftemp = 1.0 / safe_base
                htemp = np.sqrt(ftemp)
                deriv = ftemp * htemp * deriv_num
                delgr += np.where(mask, htemp - ga1s, 0.0)
                delgrp += np.where(mask, deriv - ga2 * ga1s, 0.0)
            else:
                ftemp = 1.0 / base
                htemp = np.sqrt(ftemp)
                deriv = ftemp * htemp * deriv_num
                delgr += htemp
                delgrp += deriv

        # VMEC greenf.f: when nv==1, normalize both non-singular sums by nvper.
        if nv == 1 and nvper > 1:
            scale = 1.0 / float(nvper)
            delgr *= scale
            delgrp *= scale

        # Keep the gstore accumulation order explicit for close parity with the
        # scalar Fortran-style formulation while still vectorizing the expensive
        # kernel construction above.
        for loc, ip in enumerate(ip_idx):
            gstore += bex[int(ip)] * delgr[loc]

        del_iuv = delgrp[:, iuv_grid]
        del_ref = delgrp[:, iref_grid]
        ka_grid = del_iuv - del_ref
        g1_sym = np.einsum("cuv,fv->cuf", ka_grid, cosv_modes, optimize=True)
        g2_sym = np.einsum("cuv,fv->cuf", ka_grid, sinv_modes, optimize=True)

        for isym in range(ndim):
            if isym == 0:
                g1_use = g1_sym
                g2_use = g2_sym
                sinm_table = sinm_sym
                cosm_table = cosm_sym
                row_off = 0
            else:
                ks_grid = del_iuv + del_ref
                g1_use = np.einsum("cuv,fv->cuf", ks_grid, cosv_modes, optimize=True)
                g2_use = np.einsum("cuv,fv->cuf", ks_grid, sinv_modes, optimize=True)
                sinm_table = sinm_asym
                cosm_table = cosm_asym
                row_off = mnpd

            gcos = np.einsum("mu,cuf->cmf", sinm_table, g1_use, optimize=True)
            gsin = np.einsum("mu,cuf->cmf", cosm_table, g2_use, optimize=True)
            total_plus = (gcos + gsin).reshape(n_chunk, -1)
            total_minus = (gcos - gsin).reshape(n_chunk, -1)
            rows_plus = row_off + idx_p_flat
            rows_minus = row_off + idx_m_flat[negative_n_flat]
            grpmn_nonsing[np.ix_(rows_plus, ip_idx)] += total_plus.T
            grpmn_nonsing[np.ix_(rows_minus, ip_idx)] += total_minus[:, negative_n_flat].T

    # Keep raw fourp accumulation scale; any legacy scale experiments are
    # handled upstream in diagnostics, not in the core assembly path.

    return np.asarray(gstore, dtype=float), np.asarray(grpmn_nonsing, dtype=float)


def _vmec_bvec_from_gsource(*, gsource: np.ndarray, basis: dict[str, Any]) -> np.ndarray:
    src = _vmec_source_from_gsource(gsource=gsource, basis=basis)
    sinmni = np.asarray(basis["sinmni"], dtype=float)
    bsin = sinmni.T @ src
    xmpot = np.asarray(basis["xmpot"], dtype=np.int64)
    n_raw = np.asarray(basis["n_raw"], dtype=np.int64)
    skip_mask = np.logical_and(xmpot == 0, n_raw < 0)
    if np.any(skip_mask):
        bsin = np.asarray(bsin, dtype=float)
        bsin[skip_mask] = 0.0
    if bool(basis["lasym"]):
        cosmni = np.asarray(basis["cosmni"], dtype=float)
        bcos = cosmni.T @ src
        if np.any(skip_mask):
            bcos = np.asarray(bcos, dtype=float)
            bcos[skip_mask] = 0.0
        return np.concatenate([bsin, bcos], axis=0)
    return bsin


def _vmec_analytic_terms_from_geometry(
    *,
    sample: ExternalBoundarySample,
    basis: dict[str, Any],
    bexni: np.ndarray,
    signgs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic VMEC terms from ``analyt.f``: `(bvec_analytic, grpmn_analytic)`."""

    mnpd = int(basis["mnpd"])
    lasym = bool(basis["lasym"])
    mf = int(basis["mf"])
    nf = int(basis["nf"])
    onp = float(basis["onp"])
    signgs = int(signgs)
    cmns = np.asarray(basis["cmns"], dtype=float)
    theta = np.asarray(basis["theta"], dtype=float).reshape(-1)
    zeta = np.asarray(basis["zeta"], dtype=float).reshape(-1)
    npts = int(theta.size)
    bex = np.asarray(bexni, dtype=float).reshape(-1)
    if bex.size < npts:
        bex = np.resize(bex, (npts,))
    else:
        bex = bex[:npts]

    R = np.asarray(sample.R, dtype=float).reshape(-1)[:npts]
    Ru = np.asarray(sample.Ru, dtype=float).reshape(-1)[:npts]
    Rv = np.asarray(sample.Rv, dtype=float).reshape(-1)[:npts]
    Zu = np.asarray(sample.Zu, dtype=float).reshape(-1)[:npts]
    Zv = np.asarray(sample.Zv, dtype=float).reshape(-1)[:npts]

    guu_b = Ru * Ru + Zu * Zu
    guv_b = (Ru * Rv + Zu * Zv) * (2.0 * onp)
    gvv_b = (Rv * Rv + Zv * Zv + R * R) * (onp * onp)

    adp = guu_b + guv_b + gvv_b
    adm = guu_b - guv_b + gvv_b
    cma = gvv_b - guu_b
    sqrtc = 2.0 * np.sqrt(gvv_b)
    sqrta = 2.0 * np.sqrt(guu_b)
    sqad1 = np.sqrt(adp)
    sqad2 = np.sqrt(adm)

    tlp = (1.0 / sqad1) * np.log((sqad1 * sqrtc + adp + cma) / (sqad1 * sqrta - adp + cma))
    tlm = (1.0 / sqad2) * np.log((sqad2 * sqrtc + adm + cma) / (sqad2 * sqrta - adm + cma))
    tlp_prev = np.zeros_like(tlp)
    tlm_prev = np.zeros_like(tlm)
    tlpm = tlp + tlm

    bsin = np.zeros((mf + 1, 2 * nf + 1), dtype=float)
    bcos = np.zeros((mf + 1, 2 * nf + 1), dtype=float) if lasym else None
    gsin = np.zeros((mf + 1, 2 * nf + 1, npts), dtype=float)
    gcos = np.zeros((mf + 1, 2 * nf + 1, npts), dtype=float) if lasym else None

    delt1u = adp * adm - cma * cma
    azp1u = np.zeros_like(adp)
    azm1u = np.zeros_like(adm)
    cma11u = np.zeros_like(cma)
    r1p = np.zeros_like(adp)
    r1m = np.zeros_like(adm)
    r0p = np.zeros_like(adp)
    r0m = np.zeros_like(adm)
    ra1p = np.zeros_like(adp)
    ra1m = np.zeros_like(adm)
    azp1u[:] = 0.0
    azm1u[:] = 0.0
    cma11u[:] = 0.0

    # Second-derivative geometry terms (surface.f).
    ntheta3, nzeta = sample.R.shape
    nu_full = int(basis.get("nu_full", ntheta3))
    if ntheta3 * nzeta == npts and ntheta3 > 0 and nzeta > 0:
        R_red = np.asarray(sample.R, dtype=float)
        Z_red = np.asarray(sample.Z, dtype=float)
        Ru_red = np.asarray(sample.Ru, dtype=float)
        Zu_red = np.asarray(sample.Zu, dtype=float)
        Rv_red = np.asarray(sample.Rv, dtype=float)
        Zv_red = np.asarray(sample.Zv, dtype=float)
        nv = int(nzeta)
        have_second = (
            sample.ruu is not None
            and sample.ruv is not None
            and sample.rvv is not None
            and sample.zuu is not None
            and sample.zuv is not None
            and sample.zvv is not None
        )
        if have_second and np.asarray(sample.ruu).shape == (ntheta3, nv):
            # Preferred VMEC-equivalent path: second derivatives synthesized
            # directly from modal coefficients on the reduced surface grid.
            R_eval = np.asarray(R_red, dtype=float)
            Ru_eval = np.asarray(Ru_red, dtype=float)
            Rv_eval = np.asarray(Rv_red, dtype=float)
            Zu_eval = np.asarray(Zu_red, dtype=float)
            Zv_eval = np.asarray(Zv_red, dtype=float)
            ruu = np.asarray(sample.ruu, dtype=float)
            ruv = np.asarray(sample.ruv, dtype=float)
            rvv = np.asarray(sample.rvv, dtype=float)
            zuu = np.asarray(sample.zuu, dtype=float)
            zuv = np.asarray(sample.zuv, dtype=float)
            zvv = np.asarray(sample.zvv, dtype=float)
        else:
            if (nu_full == ntheta3) or lasym:
                R2 = np.asarray(R_red, dtype=float)
                Z2 = np.asarray(Z_red, dtype=float)
                Ru2 = np.asarray(Ru_red, dtype=float)
                Zu2 = np.asarray(Zu_red, dtype=float)
                Rv2 = np.asarray(Rv_red, dtype=float)
                Zv2 = np.asarray(Zv_red, dtype=float)
            else:
                R2 = np.zeros((nu_full, nv), dtype=float)
                Z2 = np.zeros((nu_full, nv), dtype=float)
                Ru2 = np.zeros((nu_full, nv), dtype=float)
                Zu2 = np.zeros((nu_full, nv), dtype=float)
                Rv2 = np.zeros((nu_full, nv), dtype=float)
                Zv2 = np.zeros((nu_full, nv), dtype=float)
                R2[:ntheta3, :] = R_red
                Z2[:ntheta3, :] = Z_red
                Ru2[:ntheta3, :] = Ru_red
                Zu2[:ntheta3, :] = Zu_red
                Rv2[:ntheta3, :] = Rv_red
                Zv2[:ntheta3, :] = Zv_red
                kv_m = (nv - np.arange(nv, dtype=np.int64)) % max(1, nv)
                for ku in range(1, max(1, ntheta3 - 1)):
                    km = (nu_full - ku) % max(1, nu_full)
                    if km < ntheta3:
                        continue
                    R2[km, :] = R_red[ku, kv_m]
                    Z2[km, :] = -Z_red[ku, kv_m]
                    Ru2[km, :] = -Ru_red[ku, kv_m]
                    Zu2[km, :] = Zu_red[ku, kv_m]
                    Rv2[km, :] = -Rv_red[ku, kv_m]
                    Zv2[km, :] = Zv_red[ku, kv_m]

            ruu, ruv, rvv = _spectral_second_derivatives_2d(R2)
            zuu, zuv, zvv = _spectral_second_derivatives_2d(Z2)
            if (nu_full != ntheta3) and (not lasym):
                sl = slice(0, ntheta3)
                R_eval = R2[sl, :]
                Ru_eval = Ru2[sl, :]
                Rv_eval = Rv2[sl, :]
                Zu_eval = Zu2[sl, :]
                Zv_eval = Zv2[sl, :]
                ruu = ruu[sl, :]
                rvv = rvv[sl, :]
                ruv = ruv[sl, :]
                zuu = zuu[sl, :]
                zvv = zvv[sl, :]
                zuv = zuv[sl, :]
            else:
                R_eval = R2
                Ru_eval = Ru2
                Rv_eval = Rv2
                Zu_eval = Zu2
                Zv_eval = Zv2

        sgn = float(signgs)
        snr = sgn * R_eval * Zu_eval
        snv = sgn * (Ru_eval * Zv_eval - Rv_eval * Zu_eval)
        snz = -sgn * R_eval * Ru_eval
        auu = 0.5 * (snr * ruu + snz * zuu)
        auv = (snr * ruv + snv * Ru_eval + snz * zuv) * onp
        avv = (snv * Rv_eval + 0.5 * (snr * (rvv - R_eval) + snz * zvv)) * (onp * onp)
        auu = auu.reshape(-1)
        auv = auv.reshape(-1)
        avv = avv.reshape(-1)
        azp1u = auu + auv + avv
        azm1u = auu - auv + avv
        cma11u = avv - auu
        r1p = (azp1u * (delt1u - cma * cma) / adp - azm1u * adp + 2.0 * cma11u * cma) / delt1u
        r1m = (azm1u * (delt1u - cma * cma) / adm - azp1u * adm + 2.0 * cma11u * cma) / delt1u
        r0p = (-azp1u * adm * cma / adp - azm1u * cma + 2.0 * cma11u * adm) / delt1u
        r0m = (-azm1u * adp * cma / adm - azp1u * cma + 2.0 * cma11u * adp) / delt1u
        ra1p = azp1u / adp
        ra1m = azm1u / adm

    sign1 = 1.0
    fl1 = 0.0
    for l in range(0, mf + nf + 1):
        fl = fl1
        slp = (r1p * fl + ra1p) * tlp + r0p * fl * tlp_prev - (r1p + r0p) / sqrtc + sign1 * (r0p - r1p) / sqrta
        slm = (r1m * fl + ra1m) * tlm + r0m * fl * tlm_prev - (r1m + r0m) / sqrtc + sign1 * (r0m - r1m) / sqrta
        slpm = slp + slm
        for nabs in range(0, nf + 1):
            zv = float(nabs) * zeta
            cosv = np.cos(zv)
            sinv = np.sin(zv)
            for m in range(0, mf + 1):
                cm = float(cmns[l, m, nabs])
                if cm == 0.0:
                    continue
                mu = float(m) * theta
                sinu = np.sin(mu)
                cosu = np.cos(mu)
                col_p = nabs + nf
                col_m = (-nabs) + nf
                if nabs == 0 or m == 0:
                    sinp = (sinu * cosv - sinv * cosu) * cm
                    bsin[m, col_p] += np.sum(tlpm * bex * sinp)
                    gsin[m, col_p, :] += slpm * sinp
                    if lasym and bcos is not None:
                        cosp = (cosu * cosv + sinv * sinu) * cm
                        bcos[m, col_p] += np.sum(tlpm * bex * cosp)
                        if gcos is not None:
                            gcos[m, col_p, :] += slpm * cosp
                else:
                    sinp0 = sinu * cosv * cm
                    temp = -cosu * sinv * cm
                    sinm = sinp0 - temp
                    sinp = sinp0 + temp
                    # VMEC analyt.f calls analysesum2 with swapped argument
                    # order: (slm, tlm, slp, tlp). Preserve this Fortran quirk
                    # for exact matrix-side parity.
                    bsin[m, col_p] += np.sum(tlm * bex * sinp)
                    bsin[m, col_m] += np.sum(tlp * bex * sinm)
                    gsin[m, col_p, :] += slm * sinp
                    gsin[m, col_m, :] += slp * sinm
                    if lasym and bcos is not None:
                        cosp0 = cosu * cosv * cm
                        temp2 = sinu * sinv * cm
                        cosm = cosp0 - temp2
                        cosp = cosp0 + temp2
                        bcos[m, col_p] += np.sum(tlm * bex * cosp)
                        bcos[m, col_m] += np.sum(tlp * bex * cosm)
                        if gcos is not None:
                            gcos[m, col_p, :] += slm * cosp
                            gcos[m, col_m, :] += slp * cosm

        fl1 = fl1 + 1.0
        fl2 = 2.0 * fl1 - 1.0
        sign1 = -sign1

        tlp_next = ((sqrtc + sign1 * sqrta) - fl2 * cma * tlp - fl * adm * tlp_prev) / (adp * fl1)
        tlm_next = ((sqrtc + sign1 * sqrta) - fl2 * cma * tlm - fl * adp * tlm_prev) / (adm * fl1)
        tlp_prev = tlp
        tlm_prev = tlm
        tlp = tlp_next
        tlm = tlm_next
        tlpm = tlp + tlm

    out_s = np.zeros((mnpd,), dtype=float)
    out_c = np.zeros((mnpd,), dtype=float) if lasym else None
    gr_s = np.zeros((mnpd, npts), dtype=float)
    gr_c = np.zeros((mnpd, npts), dtype=float) if lasym else None
    xmpot = np.asarray(basis["xmpot"], dtype=np.int64)
    n_raw = np.asarray(basis["n_raw"], dtype=np.int64)
    for j in range(mnpd):
        m = int(xmpot[j])
        n = int(n_raw[j])
        out_s[j] = bsin[m, n + nf]
        gr_s[j, :] = gsin[m, n + nf, :]
        if lasym and out_c is not None:
            out_c[j] = bcos[m, n + nf]
            if gr_c is not None and gcos is not None:
                gr_c[j, :] = gcos[m, n + nf, :]
    if lasym and out_c is not None and gr_c is not None:
        return np.concatenate([out_s, out_c], axis=0), np.concatenate([gr_s, gr_c], axis=0)
    return out_s, gr_s


def _vmec_analytic_bvec_from_geometry(
    *,
    sample: ExternalBoundarySample,
    basis: dict[str, Any],
    bexni: np.ndarray,
    signgs: int,
) -> np.ndarray:
    """Analytic-source bvec term from VMEC ``analyt.f`` (bvec branch)."""

    bvec, _ = _vmec_analytic_terms_from_geometry(sample=sample, basis=basis, bexni=bexni, signgs=signgs)
    return bvec


def _solve_vmec_like_mode_from_gsource(
    *,
    cache: NestorVmecLikeCache,
    gsource: np.ndarray,
    rhs_mode: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve VMEC-like dense integral in mode space and return (phi, potvac)."""

    basis = cache.mode_basis
    amod = cache.mode_matrix
    if basis is None or amod is None:
        raise ValueError("missing_mode_cache")

    rhs_eff = np.asarray(rhs_mode, dtype=float) if rhs_mode is not None else _vmec_bvec_from_gsource(gsource=gsource, basis=basis)
    potvac = _dense_lu_solve(cache.mode_matrix_lu, np.asarray(amod, dtype=float), np.asarray(rhs_eff, dtype=float))

    sin_phase = np.asarray(basis["sin_phase"], dtype=float)
    cos_phase = np.asarray(basis["cos_phase"], dtype=float)
    mnpd = int(basis["mnpd"])
    if bool(basis["lasym"]):
        potsin = np.asarray(potvac[:mnpd], dtype=float)
        potcos = np.asarray(potvac[mnpd : 2 * mnpd], dtype=float)
        phi_flat = sin_phase @ potsin + cos_phase @ potcos
    else:
        potsin = np.asarray(potvac[:mnpd], dtype=float)
        phi_flat = sin_phase @ potsin
    phi = phi_flat.reshape(int(cache.ntheta), int(cache.nzeta))
    phi = phi - float(np.mean(phi))
    return np.asarray(phi, dtype=float), np.asarray(potvac, dtype=float), np.asarray(rhs_eff, dtype=float)


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
            if bvec_mode_nonsing is not None:
                bvn = np.asarray(bvec_mode_nonsing, dtype=float).reshape(-1)
                out["bvec_mode_nonsing_sin"] = np.asarray(bvn[:mnpd], dtype=float)
                if bool(basis["lasym"]) and bvn.size >= 2 * mnpd:
                    out["bvec_mode_nonsing_cos"] = np.asarray(bvn[mnpd : 2 * mnpd], dtype=float)
            if bvec_mode_analytic is not None:
                bva = np.asarray(bvec_mode_analytic, dtype=float).reshape(-1)
                out["bvec_mode_analytic_sin"] = np.asarray(bva[:mnpd], dtype=float)
                if bool(basis["lasym"]) and bva.size >= 2 * mnpd:
                    out["bvec_mode_analytic_cos"] = np.asarray(bva[mnpd : 2 * mnpd], dtype=float)

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
            if amatrix_mode_pre is not None:
                out["amatrix_mode_pre"] = np.asarray(amatrix_mode_pre, dtype=float)
            if amatrix_mode_from_grpmn is not None:
                out["amatrix_mode_from_grpmn"] = np.asarray(amatrix_mode_from_grpmn, dtype=float)
            if grpmn_nonsing is not None:
                out["grpmn_nonsing"] = np.asarray(grpmn_nonsing, dtype=float)
            if grpmn_analytic is not None:
                out["grpmn_analytic"] = np.asarray(grpmn_analytic, dtype=float)
            if grpmn_total is not None:
                out["grpmn_total"] = np.asarray(grpmn_total, dtype=float)
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
    """Simplified NESTOR-style update/reuse with ivacskip-compatible behavior.

    - `ivac==1`: full update (sample + spectral Poisson solve)
    - `ivac!=1`: reuse previous solution if available
    """

    runtime_cache = None if runtime is None else getattr(runtime, "operator_cache", None)
    runtime_mode = "spectral_poisson_external_only" if runtime is None else str(
        getattr(runtime, "mode", "spectral_poisson_external_only")
    )
    runtime_source_cache_iter = -1 if runtime is None else int(getattr(runtime, "source_cache_iter", -1))
    runtime_gsource_cached = None if runtime is None else getattr(runtime, "gsource_cached", None)
    runtime_source_sym_cached = None if runtime is None else getattr(runtime, "source_sym_cached", None)
    runtime_bvec_nonsing_cached = None if runtime is None else getattr(runtime, "bvec_nonsing_cached", None)
    if runtime_cache is None and runtime is not None and hasattr(runtime, "poisson"):
        # Backward compatibility with older runtime state shape.
        runtime_cache = getattr(runtime, "poisson")

    force_rhs_reuse = _env_truthy("VMEC_JAX_FREEB_REUSE_RHS_UPDATE", default=True)

    if int(ivac) != 1 and runtime is not None and not force_rhs_reuse:
        return _nestor_legacy_fast_reuse(
            runtime=runtime,
            runtime_cache=runtime_cache,
            runtime_mode=runtime_mode,
        )

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
    provider_allows_source_reuse = provider_kind in ("", "mgrid", "legacy_mgrid") or (
        isinstance(external_field_provider_static, dict)
        and bool(external_field_provider_static.get("allow_source_reuse", False))
    )

    if ivacskip is not None:
        reuse_step = (int(ivacskip) != 0 and runtime is not None)
    else:
        reuse_step = (int(ivac) != 1 and runtime is not None)

    rhs_mode = os.getenv("VMEC_JAX_FREEB_RHS_MODE", "bnormal_unit").strip().lower()
    wint_vmec = _vmec_boundary_wint(static=static, ntheta=int(ntheta), nzeta=int(nzeta))
    gsource_bexni = -np.asarray(sample.vac_ext.bnormal, dtype=float) * np.asarray(wint_vmec, dtype=float) * ((2.0 * np.pi) ** 2)
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
    ts = time.perf_counter()
    used_mode = selected_mode
    if mode_reason not in ("forced_fast", "forced_vmec_like", "auto_vmec_like"):
        used_mode = f"{selected_mode}_fallback:{mode_reason}"
    cache: Any = runtime_cache

    # On ivacskip reuse, emulate VMEC2000 scalpot behavior by reusing the cached
    # operator while refreshing the source term / solve.
    mode_for_step = runtime_mode if reuse_step else used_mode
    if reuse_step and runtime_cache is None:
        mode_for_step = used_mode
    potvac = bvec_mode = bvec_mode_nonsing = bvec_mode_analytic = None
    grpmn_nonsing = grpmn_analytic = grpmn_total = None
    amatrix_mode_pre = amatrix_mode_from_grpmn = None
    matrix_override_applied = jax_nestor_operator_applied = False
    jax_nestor_operator_reason = "disabled"
    jax_nestor_operator_jitted = jax_nestor_operator_cache_hit = False
    jax_nestor_operator_time_s = 0.0
    cache_build_time_s = source_time_s = bvec_time_s = matrix_time_s = 0.0
    linear_solve_time_s = vacuum_channels_time_s = 0.0

    if _is_dense_mode(mode_for_step):
        alpha = _as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_ALPHA", 1.0)
        dist_eps = _as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_DIST_EPS", 1.0e-8)
        rhs_floor = _as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_RHS_FLOOR", 1.0e-14)
        diag_coeff = _as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_DIAG_COEFF", 0.5)
        row_sum_zero = _as_int_env("VMEC_JAX_FREEB_VMEC_LIKE_ROW_SUM_ZERO", 1) != 0
        singular_diag_scale = _as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_SINGULAR_DIAG_SCALE", 1.0)
        dense_solve_mode = os.getenv("VMEC_JAX_FREEB_DENSE_SOLVE_MODE", "mode").strip().lower()
        refresh_operator_on_reuse = bool(reuse_step and not provider_allows_source_reuse)
        try:
            if (
                not isinstance(cache, NestorVmecLikeCache)
                or int(cache.ntheta) != int(ntheta)
                or int(cache.nzeta) != int(nzeta)
                or not reuse_step
                or refresh_operator_on_reuse
            ):
                t_phase = time.perf_counter()
                cache = _build_vmec_like_cache(
                    sample,
                    alpha=alpha,
                    dist_eps=dist_eps,
                    rhs_floor=rhs_floor,
                    diag_coeff=diag_coeff,
                    row_sum_zero=row_sum_zero,
                    singular_diag_scale=singular_diag_scale,
                    nfp=max(1, int(getattr(static.cfg, "nfp", 1))),
                    mf=max(0, int(getattr(static.cfg, "mpol", 1)) + 1),
                    nf=max(0, int(getattr(static.cfg, "ntor", 0))),
                    lasym=bool(getattr(static.cfg, "lasym", False)),
                    wint_vmec=np.asarray(wint_vmec, dtype=float),
                    factor_physical_matrix=dense_solve_mode not in ("mode", "vmec_mode", "fouri_mode"),
                )
                cache_build_time_s += max(0.0, time.perf_counter() - t_phase)
            use_greenf_source = _freeb_use_greenf_source(int(getattr(static.cfg, "ntor", 0)))
            # Default to Fortran-equivalent matrix assembly from grpmn (fouri
            # path). Can be disabled via VMEC_JAX_FREEB_EXPERIMENTAL_FOURI_MATRIX=0
            # for diagnostics.
            experimental_fouri_matrix = _env_truthy("VMEC_JAX_FREEB_EXPERIMENTAL_FOURI_MATRIX", default=True)
            refresh_source_on_reuse = bool(reuse_step and not provider_allows_source_reuse)
            if use_greenf_source and ((not reuse_step) or refresh_source_on_reuse) and cache.mode_basis is not None:
                nzeta_surf = int(np.asarray(sample.R).shape[1])
                nvper_greenf = 64 if nzeta_surf == 1 else max(1, int(getattr(static.cfg, "nfp", 1)))
                try:
                    t_phase = time.perf_counter()
                    if experimental_fouri_matrix:
                        gsource_vmec, grpmn_nonsing = _vmec_nonsingular_terms_from_bexni(
                            sample=sample,
                            basis=cache.mode_basis,
                            bexni=np.asarray(gsource_bexni, dtype=float),
                            signgs=int(getattr(static, "signgs", -1)),
                            nvper=nvper_greenf,
                        )
                    else:
                        gsource_vmec = _vmec_nonsingular_gsource_from_bexni(
                            sample=sample,
                            basis=cache.mode_basis,
                            bexni=np.asarray(gsource_bexni, dtype=float),
                            signgs=int(getattr(static, "signgs", -1)),
                            nvper=nvper_greenf,
                        )
                        grpmn_nonsing = None
                    source_time_s += max(0.0, time.perf_counter() - t_phase)
                except Exception:
                    gsource_vmec = np.asarray(gsource_bexni, dtype=float)
                    grpmn_nonsing = None
            if dense_solve_mode in ("mode", "vmec_mode", "fouri_mode"):
                rhs_mode_eff = None
                jax_operator_solved = False
                if cache.mode_basis is not None:
                    if reuse_step and provider_allows_source_reuse and runtime_bvec_nonsing_cached is not None:
                        bvec_mode_nonsing = np.asarray(runtime_bvec_nonsing_cached, dtype=float)
                    else:
                        t_phase = time.perf_counter()
                        bvec_mode_nonsing = _vmec_bvec_from_gsource(
                            gsource=np.asarray(gsource_vmec, dtype=float),
                            basis=cache.mode_basis,
                        )
                        bvec_time_s += max(0.0, time.perf_counter() - t_phase)
                    rhs_mode_eff = np.asarray(bvec_mode_nonsing, dtype=float)
                    add_analytic = _env_truthy("VMEC_JAX_FREEB_ADD_ANALYTIC_BVEC", default=True)
                    if add_analytic:
                        t_phase = time.perf_counter()
                        bvec_mode_analytic, grpmn_analytic = _vmec_analytic_terms_from_geometry(
                            sample=sample,
                            basis=cache.mode_basis,
                            bexni=np.asarray(gsource_bexni, dtype=float),
                            signgs=int(getattr(static, "signgs", -1)),
                        )
                        rhs_mode_eff = rhs_mode_eff + np.asarray(bvec_mode_analytic, dtype=float)
                        bvec_time_s += max(0.0, time.perf_counter() - t_phase)
                    if ((not reuse_step) or refresh_operator_on_reuse) and experimental_fouri_matrix and (grpmn_nonsing is not None):
                        grpmn_total = np.asarray(grpmn_nonsing, dtype=float)
                        if grpmn_analytic is not None:
                            grpmn_total = grpmn_total + np.asarray(grpmn_analytic, dtype=float)
                        try:
                            amatrix_mode_pre = (
                                None if cache.mode_matrix is None else np.asarray(cache.mode_matrix, dtype=float)
                            )
                            t_phase = time.perf_counter()
                            amatrix_mode_from_grpmn = _vmec_mode_matrix_from_grpmn(
                                grpmn=grpmn_total,
                                basis=cache.mode_basis,
                            )
                            cache = replace(
                                cache,
                                mode_matrix=np.asarray(amatrix_mode_from_grpmn, dtype=float),
                                mode_matrix_lu=_dense_lu_factor(np.asarray(amatrix_mode_from_grpmn, dtype=float)),
                            )
                            matrix_time_s += max(0.0, time.perf_counter() - t_phase)
                            matrix_override_applied = True
                        except Exception:
                            pass
                    if _env_truthy("VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR", False):
                        jax_nestor_operator_reason = "requested"
                        if not (use_greenf_source and experimental_fouri_matrix):
                            jax_nestor_operator_reason = "requires_greenf_fouri_matrix"
                        elif reuse_step and provider_allows_source_reuse:
                            jax_nestor_operator_reason = "skip_cached_reuse_step"
                        else:
                            ok, reason = _jax_nestor_operator_guard(sample=sample, basis=cache.mode_basis)
                            jax_nestor_operator_reason = reason
                            if ok:
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
                                        bexni=np.asarray(gsource_bexni, dtype=float),
                                        signgs=int(getattr(static, "signgs", -1)),
                                        nvper=nvper_greenf,
                                        include_analytic=add_analytic,
                                    )
                                    jax_nestor_operator_time_s += max(0.0, time.perf_counter() - t_phase)
                                    bvec_mode = np.asarray(rhs_mode_eff, dtype=float)
                                    amatrix_mode_pre = (
                                        None if cache.mode_matrix is None else np.asarray(cache.mode_matrix, dtype=float)
                                    )
                                    cache = replace(
                                        cache,
                                        mode_matrix=np.asarray(amatrix_mode_from_grpmn, dtype=float),
                                        mode_matrix_lu=_dense_lu_factor(np.asarray(amatrix_mode_from_grpmn, dtype=float)),
                                    )
                                    matrix_override_applied = True
                                    jax_nestor_operator_applied = True
                                    jax_nestor_operator_jitted = bool(jax_operator_jitted)
                                    jax_nestor_operator_cache_hit = bool(jax_operator_cache_hit)
                                    jax_nestor_operator_reason = "applied"
                                    jax_operator_solved = True
                                except Exception as exc:
                                    detail = str(exc).strip() or type(exc).__name__
                                    jax_nestor_operator_reason = f"failed:{detail}"
                if not jax_operator_solved:
                    t_phase = time.perf_counter()
                    phi, potvac, bvec_mode = _solve_vmec_like_mode_from_gsource(
                        cache=cache,
                        gsource=np.asarray(gsource_vmec, dtype=float),
                        rhs_mode=rhs_mode_eff,
                    )
                    linear_solve_time_s += max(0.0, time.perf_counter() - t_phase)
                if cache.mode_basis is not None:
                    t_phase = time.perf_counter()
                    vac_total = _vacuum_channels_from_sample_potvac(
                        sample=sample,
                        basis=cache.mode_basis,
                        potvac=np.asarray(potvac, dtype=float),
                    )
                    vacuum_channels_time_s += max(0.0, time.perf_counter() - t_phase)
                else:
                    t_phase = time.perf_counter()
                    vac_total = _vacuum_channels_from_sample_phi(sample, phi)
                    vacuum_channels_time_s += max(0.0, time.perf_counter() - t_phase)
            else:
                t_phase = time.perf_counter()
                phi = _solve_vmec_like_dense(rhs, cache)
                linear_solve_time_s += max(0.0, time.perf_counter() - t_phase)
                t_phase = time.perf_counter()
                vac_total = _vacuum_channels_from_sample_phi(sample, phi)
                vacuum_channels_time_s += max(0.0, time.perf_counter() - t_phase)
            used_mode = mode_for_step
        except Exception:
            t_phase = time.perf_counter()
            cache = _build_poisson_cache(ntheta=ntheta, nzeta=nzeta)
            cache_build_time_s += max(0.0, time.perf_counter() - t_phase)
            t_phase = time.perf_counter()
            phi = _solve_periodic_poisson_fft(rhs, cache)
            linear_solve_time_s += max(0.0, time.perf_counter() - t_phase)
            t_phase = time.perf_counter()
            vac_total = _vacuum_channels_from_sample_phi(sample, phi)
            vacuum_channels_time_s += max(0.0, time.perf_counter() - t_phase)
            used_mode = "spectral_poisson_external_only_fallback:dense_failed"
    else:
        if (
            not isinstance(cache, NestorPoissonCache)
            or int(cache.ntheta) != int(ntheta)
            or int(cache.nzeta) != int(nzeta)
        ):
            t_phase = time.perf_counter()
            cache = _build_poisson_cache(ntheta=ntheta, nzeta=nzeta)
            cache_build_time_s += max(0.0, time.perf_counter() - t_phase)
        t_phase = time.perf_counter()
        phi = _solve_periodic_poisson_fft(rhs, cache)
        linear_solve_time_s += max(0.0, time.perf_counter() - t_phase)
        t_phase = time.perf_counter()
        vac_total = _vacuum_channels_from_sample_phi(sample, phi)
        vacuum_channels_time_s += max(0.0, time.perf_counter() - t_phase)
        used_mode = mode_for_step

    bsqvac = np.asarray(vac_total.bsqvac)
    solve_time = max(0.0, time.perf_counter() - ts)

    diagnostics = _nestor_step_diagnostics(locals())

    trace_arrays = None
    if bool(collect_trace_arrays):
        trace_arrays = _nestor_trace_arrays(
            sample=sample,
            vac_total=vac_total,
            gsource_vmec=gsource_vmec,
            potvac=potvac,
            bvec_mode=bvec_mode,
            bvec_mode_nonsing=bvec_mode_nonsing,
            bvec_mode_analytic=bvec_mode_analytic,
            grpmn_nonsing=grpmn_nonsing,
            grpmn_analytic=grpmn_analytic,
            grpmn_total=grpmn_total,
            amatrix_mode_from_grpmn=amatrix_mode_from_grpmn,
            cache=cache,
        )

    res = NestorSolveResult(
        vac_total=vac_total,
        phi=phi,
        reused=bool(reuse_step),
        solve_time_s=solve_time,
        sample_time_s=sample_time,
        model=used_mode,
        diagnostics=diagnostics,
        trace_arrays=trace_arrays,
    )
    source_sym_cached, bvec_nonsing_cached, gsource_cached, source_cache_iter = (
        runtime_source_sym_cached,
        runtime_bvec_nonsing_cached,
        runtime_gsource_cached,
        runtime_source_cache_iter,
    )
    if isinstance(cache, NestorVmecLikeCache) and (cache.mode_basis is not None):
        basis = cache.mode_basis
        if (not reuse_step) or (not provider_allows_source_reuse) or (gsource_cached is None):
            gsource_cached = np.asarray(gsource_vmec, dtype=float)
        if (not reuse_step) or (not provider_allows_source_reuse) or (source_sym_cached is None):
            try:
                source_sym_cached = _vmec_source_from_gsource(
                    gsource=np.asarray(gsource_cached, dtype=float),
                    basis=basis,
                )
            except Exception:
                source_sym_cached = runtime_source_sym_cached
        if (not reuse_step) or (not provider_allows_source_reuse) or (bvec_nonsing_cached is None):
            if bvec_mode_nonsing is not None:
                bvec_nonsing_cached = np.asarray(bvec_mode_nonsing, dtype=float)
            else:
                try:
                    bvec_nonsing_cached = _vmec_bvec_from_gsource(
                        gsource=np.asarray(gsource_cached, dtype=float),
                        basis=basis,
                    )
                except Exception:
                    bvec_nonsing_cached = runtime_bvec_nonsing_cached
        if (not reuse_step) and (iter_idx is not None):
            source_cache_iter = int(iter_idx)
    runtime_next = NestorRuntimeState(
        operator_cache=cache,
        phi=np.asarray(phi),
        bsqvac=np.asarray(bsqvac),
        mode=used_mode,
        update_count=(0 if runtime is None else int(runtime.update_count)) + (0 if reuse_step else 1),
        reuse_count=(0 if runtime is None else int(runtime.reuse_count)) + (1 if reuse_step else 0),
        source_cache_iter=int(source_cache_iter),
        gsource_cached=None if gsource_cached is None else np.asarray(gsource_cached, dtype=float),
        source_sym_cached=None if source_sym_cached is None else np.asarray(source_sym_cached, dtype=float),
        bvec_nonsing_cached=None
        if bvec_nonsing_cached is None
        else np.asarray(bvec_nonsing_cached, dtype=float),
    )
    gsource_dump = (
        np.asarray(gsource_cached, dtype=float)
        if (reuse_step and gsource_cached is not None)
        else np.asarray(gsource_vmec, dtype=float)
    )
    _maybe_dump_scalpot_jax(
        iter_idx=iter_idx,
        ivac=int(ivac),
        reused=bool(reuse_step),
        mode=used_mode,
        rhs=np.asarray(rhs, dtype=float),
        phi=np.asarray(phi, dtype=float),
        vac=vac_total,
        cache=cache,
        sample=sample,
        mf=max(0, int(getattr(static.cfg, "mpol", 1)) + 1),
        nf=max(0, int(getattr(static.cfg, "ntor", 0))),
        nfp=max(1, int(getattr(static.cfg, "nfp", 1))),
        lasym=bool(getattr(static.cfg, "lasym", False)),
        wint_vmec=np.asarray(wint_vmec, dtype=float),
        gsource_vmec=gsource_dump,
        potvac=None if potvac is None else np.asarray(potvac, dtype=float),
        bvec_mode=None if bvec_mode is None else np.asarray(bvec_mode, dtype=float),
        bvec_mode_nonsing=None if bvec_mode_nonsing is None else np.asarray(bvec_mode_nonsing, dtype=float),
        bvec_mode_analytic=None if bvec_mode_analytic is None else np.asarray(bvec_mode_analytic, dtype=float),
        source_cache_iter=int(source_cache_iter),
        matrix_override_applied=bool(matrix_override_applied),
        amatrix_mode_pre=None if amatrix_mode_pre is None else np.asarray(amatrix_mode_pre, dtype=float),
        amatrix_mode_from_grpmn=None
        if amatrix_mode_from_grpmn is None
        else np.asarray(amatrix_mode_from_grpmn, dtype=float),
        grpmn_nonsing=None if grpmn_nonsing is None else np.asarray(grpmn_nonsing, dtype=float),
        grpmn_analytic=None if grpmn_analytic is None else np.asarray(grpmn_analytic, dtype=float),
        grpmn_total=None if grpmn_total is None else np.asarray(grpmn_total, dtype=float),
        plascur=float(plascur),
    )
    return res, runtime_next


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
