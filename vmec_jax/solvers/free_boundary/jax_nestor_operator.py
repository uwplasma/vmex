"""Opt-in JAX NESTOR operator cache used by free-boundary validation paths."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np


JAX_NESTOR_BASIS_KEYS = (
    "lasym",
    "mf",
    "nf",
    "mn0",
    "mnpd",
    "mnpd2",
    "nu_full",
    "nuv3",
    "nuv_full",
    "onp",
    "cmns",
    "cos_phase",
    "cosmni",
    "imirr",
    "imirr_full",
    "n_raw",
    "sin_phase",
    "sinmni",
    "theta",
    "wint",
    "xmpot",
    "zeta",
)

FREEB_JAX_NESTOR_OPERATOR_FN_CACHE: dict[tuple[Any, ...], Any] = {}


def env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in ("", "0", "false", "no")


def digest_array_for_cache(value: Any) -> tuple[tuple[int, ...], str, str]:
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
    return tuple(int(i) for i in arr.shape), str(arr.dtype), digest


def mapping_cache_signature(mapping: dict[str, Any], keys: tuple[str, ...] | None = None) -> tuple[Any, ...]:
    selected = tuple(sorted(mapping)) if keys is None else tuple(key for key in keys if key in mapping)
    signature: list[Any] = []
    for key in selected:
        value = mapping[key]
        if isinstance(value, dict):
            continue
        signature.append((key, digest_array_for_cache(value)))
    return tuple(signature)


def compact_jax_nestor_basis(basis: dict[str, Any]) -> dict[str, Any]:
    return {key: basis[key] for key in JAX_NESTOR_BASIS_KEYS if key in basis}


def jax_nestor_operator_cache_key(
    *,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    include_analytic: bool,
    symmetric: bool,
    input_signature: tuple[Any, ...] = (),
) -> tuple[Any, ...]:
    return (
        int(signgs),
        int(nvper),
        bool(include_analytic),
        bool(symmetric),
        tuple(input_signature),
        mapping_cache_signature(basis, JAX_NESTOR_BASIS_KEYS),
        mapping_cache_signature(tables),
    )


def jax_nestor_input_signature(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple((tuple(int(i) for i in np.asarray(arg).shape), str(np.asarray(arg).dtype)) for arg in args)


def jitted_jax_nestor_operator(
    *,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    include_analytic: bool,
    symmetric: bool = False,
    example_args: tuple[Any, ...] = (),
) -> tuple[Any | None, bool]:
    """Return a cached compiled dense JAX NESTOR operator closure.

    The closure bakes mode-basis and kernel-table arrays as static constants so
    the active free-boundary update does not execute the JAX operator as many
    small eager dispatches. This cache is intentionally used only by the opt-in
    research path selected with ``VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR=1``.
    """

    try:
        from ..._compat import jax as _jax
        from ...free_boundary_adjoint import dense_vmec_nestor_mode_solve_jax
    except Exception:
        return None, False
    if _jax is None:
        return None, False
    if bool(getattr(_jax.config, "jax_disable_jit", False)):
        return None, False

    key = jax_nestor_operator_cache_key(
        basis=basis,
        tables=tables,
        signgs=int(signgs),
        nvper=int(nvper),
        include_analytic=bool(include_analytic),
        symmetric=bool(symmetric),
        input_signature=jax_nestor_input_signature(tuple(example_args)),
    )
    cached = FREEB_JAX_NESTOR_OPERATOR_FN_CACHE.get(key)
    if cached is not None:
        return cached, True

    if len(FREEB_JAX_NESTOR_OPERATOR_FN_CACHE) >= 32:
        FREEB_JAX_NESTOR_OPERATOR_FN_CACHE.clear()

    basis_static = compact_jax_nestor_basis(basis)
    tables_static = {key: tables[key] for key in sorted(tables)}

    def _compiled(
        R: Any,
        Z: Any,
        Ru: Any,
        Zu: Any,
        Rv: Any,
        Zv: Any,
        ruu: Any,
        ruv: Any,
        rvv: Any,
        zuu: Any,
        zuv: Any,
        zvv: Any,
        bexni: Any,
    ) -> dict[str, Any]:
        return dense_vmec_nestor_mode_solve_jax(
            R=R,
            Z=Z,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
            ruu=ruu,
            ruv=ruv,
            rvv=rvv,
            zuu=zuu,
            zuv=zuv,
            zvv=zvv,
            bexni=bexni,
            basis=basis_static,
            tables=tables_static,
            signgs=int(signgs),
            nvper=int(nvper),
            include_analytic=bool(include_analytic),
            symmetric=bool(symmetric),
        )

    jitted = _jax.jit(_compiled)
    compiled = jitted.lower(*example_args).compile() if example_args else jitted
    FREEB_JAX_NESTOR_OPERATOR_FN_CACHE[key] = compiled
    return compiled, False


def jax_nestor_operator_guard(
    *,
    sample: Any,
    basis: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Return whether the experimental JAX VMEC/NESTOR operator can run safely."""

    if basis is None:
        return False, "missing_mode_basis"
    try:
        from ..._compat import has_jax, x64_enabled

        if not has_jax():
            return False, "jax_unavailable"
        if not x64_enabled():
            return False, "jax_x64_disabled"
    except Exception:
        return False, "jax_unavailable"
    if sample.R.ndim != 2:
        return False, "sample_R_not_2d"
    if int(sample.R.size) != int(basis.get("nuv3", sample.R.size)):
        return False, "requires_active_vmec_grid_points"
    if bool(basis.get("lasym", False)) and int(sample.R.size) != int(basis.get("nuv_full", sample.R.size)):
        return False, "requires_lasym_full_vmec_grid_points"
    if int(sample.R.shape[0]) > int(basis.get("nu_full", sample.R.shape[0])):
        return False, "active_grid_exceeds_full_grid"
    for name in ("Z", "Ru", "Zu", "Rv", "Zv"):
        arr = np.asarray(getattr(sample, name), dtype=float)
        if arr.shape != sample.R.shape:
            return False, f"{name}_shape_mismatch"
    for name in ("ruu", "ruv", "rvv", "zuu", "zuv", "zvv"):
        arr = getattr(sample, name)
        if arr is None:
            return False, f"missing_{name}"
        if np.asarray(arr).shape != sample.R.shape:
            return False, f"{name}_shape_mismatch"
    return True, "enabled"


def solve_vmec_like_mode_with_jax_nestor_operator(
    *,
    sample: Any,
    basis: dict[str, Any],
    tables: dict[str, Any],
    bexni: np.ndarray,
    signgs: int,
    nvper: int,
    include_analytic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """Run the experimental dense JAX VMEC/NESTOR mode operator."""

    from ...free_boundary_adjoint import dense_vmec_nestor_mode_solve_jax

    R = np.asarray(sample.R, dtype=float)
    Z = np.asarray(sample.Z, dtype=float)
    Ru = np.asarray(sample.Ru, dtype=float)
    Zu = np.asarray(sample.Zu, dtype=float)
    Rv = np.asarray(sample.Rv, dtype=float)
    Zv = np.asarray(sample.Zv, dtype=float)
    ruu = np.asarray(sample.ruu, dtype=float)
    ruv = np.asarray(sample.ruv, dtype=float)
    rvv = np.asarray(sample.rvv, dtype=float)
    zuu = np.asarray(sample.zuu, dtype=float)
    zuv = np.asarray(sample.zuv, dtype=float)
    zvv = np.asarray(sample.zvv, dtype=float)
    bexni_arr = np.asarray(bexni, dtype=float)
    operator_args = (R, Z, Ru, Zu, Rv, Zv, ruu, ruv, rvv, zuu, zuv, zvv, bexni_arr)
    compiled = None
    cache_hit = False
    if env_truthy("VMEC_JAX_FREEB_JAX_NESTOR_JIT_OPERATOR", True):
        compiled, cache_hit = jitted_jax_nestor_operator(
            basis=basis,
            tables=tables,
            signgs=int(signgs),
            nvper=max(1, int(nvper)),
            include_analytic=bool(include_analytic),
            example_args=operator_args,
        )
    if compiled is None:
        out = dense_vmec_nestor_mode_solve_jax(
            R=R,
            Z=Z,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
            ruu=ruu,
            ruv=ruv,
            rvv=rvv,
            zuu=zuu,
            zuv=zuv,
            zvv=zvv,
            bexni=bexni_arr,
            basis=basis,
            tables=tables,
            signgs=int(signgs),
            nvper=max(1, int(nvper)),
            include_analytic=bool(include_analytic),
        )
        jit_used = False
    else:
        out = compiled(*operator_args)
        jit_used = True
    potvac = np.asarray(out["mode_coeffs"], dtype=float)
    rhs_mode = np.asarray(out["rhs_mode"], dtype=float)
    mode_matrix = np.asarray(out["mode_matrix"], dtype=float)
    grpmn = np.asarray(out["grpmn"], dtype=float)
    gsource_nonsing = np.asarray(out["gsource_nonsing"], dtype=float)
    mnpd2 = int(basis["mnpd2"])
    if mode_matrix.shape != (mnpd2, mnpd2):
        raise ValueError("jax_nestor_mode_matrix_shape")
    if rhs_mode.shape != (mnpd2,) or potvac.shape != (mnpd2,):
        raise ValueError("jax_nestor_mode_vector_shape")
    for name, arr in (
        ("rhs_mode", rhs_mode),
        ("mode_matrix", mode_matrix),
        ("mode_coeffs", potvac),
        ("grpmn", grpmn),
        ("gsource_nonsing", gsource_nonsing),
    ):
        if not np.isfinite(arr).all():
            raise ValueError(f"jax_nestor_nonfinite_{name}")
    residual = mode_matrix @ potvac - rhs_mode
    residual_tol = 1.0e-8 * (1.0 + float(np.linalg.norm(rhs_mode)))
    if float(np.linalg.norm(residual)) > residual_tol:
        raise ValueError("jax_nestor_linear_residual")
    mnpd = int(basis["mnpd"])
    sin_phase = np.asarray(basis["sin_phase"], dtype=float)
    cos_phase = np.asarray(basis["cos_phase"], dtype=float)
    if bool(basis["lasym"]) and potvac.size >= 2 * mnpd:
        phi_flat = sin_phase @ potvac[:mnpd] + cos_phase @ potvac[mnpd : 2 * mnpd]
    else:
        phi_flat = sin_phase @ potvac[:mnpd]
    phi = np.asarray(phi_flat, dtype=float).reshape(np.asarray(sample.R).shape)
    phi = phi - float(np.mean(phi))
    return (
        phi,
        potvac,
        rhs_mode,
        mode_matrix,
        grpmn,
        gsource_nonsing,
        jit_used,
        cache_hit,
    )


__all__ = [
    "FREEB_JAX_NESTOR_OPERATOR_FN_CACHE",
    "JAX_NESTOR_BASIS_KEYS",
    "compact_jax_nestor_basis",
    "digest_array_for_cache",
    "env_truthy",
    "jax_nestor_input_signature",
    "jax_nestor_operator_cache_key",
    "jax_nestor_operator_guard",
    "jitted_jax_nestor_operator",
    "mapping_cache_signature",
    "solve_vmec_like_mode_with_jax_nestor_operator",
]
