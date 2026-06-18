"""VMEC-style real-space synthesis for parity utilities.

This module provides a VMEC-compatible inverse transform on the VMEC internal
angle grid using the trig/weight tables from ``fixaray``. It is intended for
parity diagnostics where matching VMEC's reduced theta grid and scaling
conventions matters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._compat import jnp, has_jax, einsum
from .modes import ModeTable
from .vmec_residue import vmec_scalxc_from_s
from .vmec_tomnsp import VmecTrigTables
from .vmec_parity import vmec_m1_internal_to_physical_signed


_PHASE_CACHE: dict[tuple[int, int], tuple[Any, Any]] = {}
_PHASE_DTHETA_CACHE: dict[tuple[int, int], tuple[Any, Any]] = {}
_PHASE_DZETA_CACHE: dict[tuple[int, int], tuple[Any, Any]] = {}
_PHASE_STACK_CACHE: dict[tuple[int, int], Any] = {}
_PHASE_DTHETA_STACK_CACHE: dict[tuple[int, int], Any] = {}
_PHASE_DZETA_STACK_CACHE: dict[tuple[int, int], Any] = {}
_SCALXC_MN_CACHE: dict[tuple[int, int, str, int, int], Any] = {}
_SCALXC_MN_CACHE_LIMIT = 128


def _phase_cache_key(modes: ModeTable, trig: VmecTrigTables) -> tuple[int, int]:
    return (id(modes), id(trig))


def _phase_cache_valid(cached: Any, modes: ModeTable, trig: VmecTrigTables) -> bool:
    """Return True when a cached phase pair (cos_phase, sin_phase) is valid.

    Python reuses object ids when objects are garbage-collected.  Two distinct
    (modes, trig) pairs from different tests may therefore share the same id-
    based cache key.  We guard against stale hits by checking that the cached
    array has the expected shape for the *current* modes/trig pair.

    Expected shape: (K, ntheta3, nzeta).
    """
    try:
        cos_phase, sin_phase = cached
        K = int(np.asarray(modes.m).shape[0])
        nzeta = int(np.asarray(trig.cosnv).shape[0])
        ntheta3 = int(trig.ntheta3)
        return (
            int(cos_phase.shape[0]) == K
            and int(cos_phase.shape[1]) == ntheta3
            and int(cos_phase.shape[2]) == nzeta
        )
    except Exception:
        return False


def _cache_allowed() -> bool:
    if not has_jax():
        return True
    try:
        from jax import core

        return bool(core.trace_ctx.is_top_level())
    except Exception:
        return False


def _mode_index_arrays(*, m: Any, n: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m_idx = np.asarray(m, dtype=np.int32)
    n_idx = np.asarray(n, dtype=np.int32)
    n_abs = np.abs(n_idx)
    sgn = np.where(n_idx < 0, -1.0, 1.0)
    return m_idx, n_abs, sgn


def _take_mode_columns(table: Any, indices: np.ndarray) -> Any:
    return jnp.take(jnp.asarray(table), indices, axis=1).T


def _vmec_mode_scaling(*, m: Any, n: Any, trig: VmecTrigTables) -> Any:
    """Return 1/(mscale* nscale) for each (m,n)."""
    m_idx, n1_idx, _ = _mode_index_arrays(m=m, n=n)
    mscale = jnp.asarray(trig.mscale)
    nscale = jnp.asarray(trig.nscale)
    return 1.0 / (jnp.take(mscale, m_idx, axis=0) * jnp.take(nscale, n1_idx, axis=0))


def _scalxc_mn_for_s(*, s: Any, modes: ModeTable, m_np: np.ndarray, dtype: Any) -> Any:
    """Return ``vmec_scalxc_from_s(s)[:, modes.m]`` with a top-level cache."""
    s_arr = jnp.asarray(s)
    ns = int(s_arr.shape[0])
    key = None
    if _cache_allowed():
        try:
            key = (id(s), id(modes), str(np.dtype(dtype)), ns, int(m_np.shape[0]))
            cached = _SCALXC_MN_CACHE.get(key)
            if cached is not None and tuple(cached.shape) == (ns, int(m_np.shape[0])):
                return cached
        except Exception:
            key = None
    mpol = int(np.max(m_np)) + 1
    scalxc = vmec_scalxc_from_s(s=s_arr, mpol=mpol).astype(dtype)
    scalxc_mn = scalxc[:, np.asarray(m_np, dtype=np.int32)]
    if key is not None:
        if len(_SCALXC_MN_CACHE) >= _SCALXC_MN_CACHE_LIMIT:
            _SCALXC_MN_CACHE.pop(next(iter(_SCALXC_MN_CACHE)), None)
        _SCALXC_MN_CACHE[key] = scalxc_mn
    return scalxc_mn


def _vmec_phase_tables(*, m: Any, n: Any, trig: VmecTrigTables):
    m_idx, n1_idx, sgn_np = _mode_index_arrays(m=m, n=n)
    sgn = jnp.asarray(sgn_np)

    cosmu_m = _take_mode_columns(trig.cosmu, m_idx)
    sinmu_m = _take_mode_columns(trig.sinmu, m_idx)
    cosnv_n = _take_mode_columns(trig.cosnv, n1_idx)
    sinnv_n = _take_mode_columns(trig.sinnv, n1_idx)

    cos_phase = cosmu_m[:, :, None] * cosnv_n[:, None, :] + sgn[:, None, None] * sinmu_m[:, :, None] * sinnv_n[:, None, :]
    sin_phase = sinmu_m[:, :, None] * cosnv_n[:, None, :] - sgn[:, None, None] * cosmu_m[:, :, None] * sinnv_n[:, None, :]
    return cos_phase, sin_phase


def _vmec_phase_tables_cached(*, modes: ModeTable, trig: VmecTrigTables, cache: bool = True):
    if cache and _cache_allowed():
        key = _phase_cache_key(modes, trig)
        cached = _PHASE_CACHE.get(key)
        if cached is not None and _phase_cache_valid(cached, modes, trig):
            return cached
    cos_phase, sin_phase = _vmec_phase_tables(m=modes.m, n=modes.n, trig=trig)
    if cache and _cache_allowed():
        _PHASE_CACHE[key] = (cos_phase, sin_phase)
    return cos_phase, sin_phase


def _phase_stack_cache_valid(cached: Any, modes: ModeTable, trig: VmecTrigTables) -> bool:
    """Return True when a stacked phase (2*K, ntheta3, nzeta) is valid for modes/trig."""
    try:
        K = int(np.asarray(modes.m).shape[0])
        nzeta = int(np.asarray(trig.cosnv).shape[0])
        ntheta3 = int(trig.ntheta3)
        return (
            int(cached.shape[0]) == 2 * K
            and int(cached.shape[1]) == ntheta3
            and int(cached.shape[2]) == nzeta
        )
    except Exception:
        return False


def _vmec_phase_tables_stacked_cached(*, modes: ModeTable, trig: VmecTrigTables, cache: bool = True):
    if cache and _cache_allowed():
        key = _phase_cache_key(modes, trig)
        cached = _PHASE_STACK_CACHE.get(key)
        if cached is not None and _phase_stack_cache_valid(cached, modes, trig):
            return cached
    cos_phase, sin_phase = _vmec_phase_tables_cached(modes=modes, trig=trig, cache=cache)
    phase = jnp.concatenate([cos_phase, sin_phase], axis=0)
    if cache and _cache_allowed():
        _PHASE_STACK_CACHE[key] = phase
    return phase


def _same_mode_array(candidate: Any, expected: Any) -> bool:
    if candidate is expected:
        return True
    try:
        candidate_np = np.asarray(candidate)
        expected_np = np.asarray(expected)
        return candidate_np.shape == expected_np.shape and bool(np.array_equal(candidate_np, expected_np))
    except Exception:
        return False


def _phase_stack_from_trig(modes: ModeTable, trig: VmecTrigTables, attr: str) -> Any | None:
    phase = getattr(trig, attr, None)
    if phase is None:
        return None
    if not _same_mode_array(getattr(trig, "phase_stack_m", None), modes.m):
        return None
    if not _same_mode_array(getattr(trig, "phase_stack_n", None), modes.n):
        return None
    return phase


def _vmec_phase_tables_dtheta(*, m: Any, n: Any, trig: VmecTrigTables):
    m_idx, n1_idx, sgn_np = _mode_index_arrays(m=m, n=n)
    sgn = jnp.asarray(sgn_np)

    cosmum_m = _take_mode_columns(trig.cosmum, m_idx)
    sinmum_m = _take_mode_columns(trig.sinmum, m_idx)
    cosnv_n = _take_mode_columns(trig.cosnv, n1_idx)
    sinnv_n = _take_mode_columns(trig.sinnv, n1_idx)

    dcos_phase = sinmum_m[:, :, None] * cosnv_n[:, None, :] + sgn[:, None, None] * cosmum_m[:, :, None] * sinnv_n[:, None, :]
    dsin_phase = cosmum_m[:, :, None] * cosnv_n[:, None, :] - sgn[:, None, None] * sinmum_m[:, :, None] * sinnv_n[:, None, :]
    return dcos_phase, dsin_phase


def _vmec_phase_tables_dtheta_cached(*, modes: ModeTable, trig: VmecTrigTables, cache: bool = True):
    if cache and _cache_allowed():
        key = _phase_cache_key(modes, trig)
        cached = _PHASE_DTHETA_CACHE.get(key)
        if cached is not None and _phase_cache_valid(cached, modes, trig):
            return cached
    dcos_phase, dsin_phase = _vmec_phase_tables_dtheta(m=modes.m, n=modes.n, trig=trig)
    if cache and _cache_allowed():
        _PHASE_DTHETA_CACHE[key] = (dcos_phase, dsin_phase)
    return dcos_phase, dsin_phase


def _vmec_phase_tables_dtheta_stacked_cached(*, modes: ModeTable, trig: VmecTrigTables, cache: bool = True):
    if cache and _cache_allowed():
        key = _phase_cache_key(modes, trig)
        cached = _PHASE_DTHETA_STACK_CACHE.get(key)
        if cached is not None and _phase_stack_cache_valid(cached, modes, trig):
            return cached
    dcos_phase, dsin_phase = _vmec_phase_tables_dtheta_cached(modes=modes, trig=trig, cache=cache)
    phase = jnp.concatenate([dcos_phase, dsin_phase], axis=0)
    if cache and _cache_allowed():
        _PHASE_DTHETA_STACK_CACHE[key] = phase
    return phase


def _vmec_phase_tables_dzeta(*, m: Any, n: Any, trig: VmecTrigTables):
    m_idx, n1_idx, sgn_np = _mode_index_arrays(m=m, n=n)
    sgn = jnp.asarray(sgn_np)

    cosmu_m = _take_mode_columns(trig.cosmu, m_idx)
    sinmu_m = _take_mode_columns(trig.sinmu, m_idx)
    cosnvn_n = _take_mode_columns(trig.cosnvn, n1_idx)
    sinnvn_n = _take_mode_columns(trig.sinnvn, n1_idx)

    dcos_phase = cosmu_m[:, :, None] * sinnvn_n[:, None, :] + sgn[:, None, None] * sinmu_m[:, :, None] * cosnvn_n[:, None, :]
    dsin_phase = sinmu_m[:, :, None] * sinnvn_n[:, None, :] - sgn[:, None, None] * cosmu_m[:, :, None] * cosnvn_n[:, None, :]
    return dcos_phase, dsin_phase


def _vmec_phase_tables_dzeta_cached(*, modes: ModeTable, trig: VmecTrigTables, cache: bool = True):
    if cache and _cache_allowed():
        key = _phase_cache_key(modes, trig)
        cached = _PHASE_DZETA_CACHE.get(key)
        if cached is not None and _phase_cache_valid(cached, modes, trig):
            return cached
    dcos_phase, dsin_phase = _vmec_phase_tables_dzeta(m=modes.m, n=modes.n, trig=trig)
    if cache and _cache_allowed():
        _PHASE_DZETA_CACHE[key] = (dcos_phase, dsin_phase)
    return dcos_phase, dsin_phase


def _vmec_phase_tables_dzeta_stacked_cached(*, modes: ModeTable, trig: VmecTrigTables, cache: bool = True):
    if cache and _cache_allowed():
        key = _phase_cache_key(modes, trig)
        cached = _PHASE_DZETA_STACK_CACHE.get(key)
        if cached is not None and _phase_stack_cache_valid(cached, modes, trig):
            return cached
    dcos_phase, dsin_phase = _vmec_phase_tables_dzeta_cached(modes=modes, trig=trig, cache=cache)
    phase = jnp.concatenate([dcos_phase, dsin_phase], axis=0)
    if cache and _cache_allowed():
        _PHASE_DZETA_STACK_CACHE[key] = phase
    return phase


def vmec_realspace_synthesis(
    *,
    coeff_cos: Any,
    coeff_sin: Any,
    modes: ModeTable,
    trig: VmecTrigTables,
    coeffs_internal: bool = False,
    apply_scalxc: bool = False,
    s: Any | None = None,
    use_stacked_dot: bool = True,
) -> Any:
    """Synthesize a real-space field on the VMEC internal grid.

    This implements the same trigonometric synthesis as VMEC's ``totzsp``
    using the precomputed ``fixaray`` tables. The input coefficients are
    assumed to be in the **wout/physical** convention by default. Set
    ``coeffs_internal=True`` when passing VMEC *internal* coefficients.

    Parameters
    ----------
    coeff_cos, coeff_sin:
        Arrays of shape (ns, K) with Fourier coefficients for cos/sin.
    modes:
        Mode table with arrays ``m`` and ``n`` (n is *not* multiplied by nfp).
    trig:
        VMEC trig tables from ``vmec_trig_tables``.

    Returns
    -------
    f:
        Real-space field of shape (ns, ntheta3, nzeta).
    """
    coeff_cos = jnp.asarray(coeff_cos)
    coeff_sin = jnp.asarray(coeff_sin)
    m_np = np.asarray(modes.m)
    m = jnp.asarray(m_np).astype(jnp.int32)
    n = jnp.asarray(modes.n).astype(jnp.int32)

    if coeff_cos.ndim < 2 or coeff_sin.ndim < 2:
        raise ValueError("Expected coeff arrays with shape (..., ns, K)")
    if coeff_cos.shape != coeff_sin.shape:
        raise ValueError("coeff_cos and coeff_sin must have the same shape")
    if coeff_cos.shape[1] != m.shape[0]:
        if coeff_cos.shape[-1] != m.shape[0]:
            raise ValueError("Mode count mismatch between coefficients and modes")

    # VMEC internal scaling: coefficients are stored divided by mscale*nscale.
    if not bool(coeffs_internal):
        scale = _vmec_mode_scaling(m=m, n=n, trig=trig).astype(coeff_cos.dtype)
        scale_shape = (1,) * (coeff_cos.ndim - 1) + (m.shape[0],)
        coeff_cos = coeff_cos * scale.reshape(scale_shape)
        coeff_sin = coeff_sin * scale.reshape(scale_shape)

    # Optional odd-m scaling hook. VMEC's `totzsp` uses coefficients that already
    # include the 1/sqrt(s) odd-m scaling; apply this when synthesizing from
    # internal VMEC coefficients. For external (wout-style) coefficients, keep
    # this False.
    if bool(apply_scalxc):
        ns = int(coeff_cos.shape[-2])
        if s is None:
            if ns < 2:
                s = jnp.asarray([0.0], dtype=coeff_cos.dtype)
            else:
                s = jnp.linspace(0.0, 1.0, ns, dtype=coeff_cos.dtype)
        scalxc_mn = _scalxc_mn_for_s(s=s, modes=modes, m_np=m_np, dtype=coeff_cos.dtype)
        scalxc_shape = (1,) * (coeff_cos.ndim - 2) + scalxc_mn.shape
        coeff_cos = coeff_cos * scalxc_mn.reshape(scalxc_shape)
        coeff_sin = coeff_sin * scalxc_mn.reshape(scalxc_shape)

    if bool(use_stacked_dot):
        phase = _phase_stack_from_trig(modes, trig, "phase_stack")
        if phase is None:
            phase = _vmec_phase_tables_stacked_cached(modes=modes, trig=trig, cache=True)
        coeff = jnp.concatenate([coeff_cos, coeff_sin], axis=-1)
        f = einsum("...k,kij->...ij", coeff, phase)
    else:
        cos_phase, sin_phase = _vmec_phase_tables_cached(modes=modes, trig=trig, cache=True)
        f = einsum("...k,kij->...ij", coeff_cos, cos_phase) + einsum(
            "...k,kij->...ij", coeff_sin, sin_phase
        )
    return f


def vmec_realspace_synthesis_multi(
    *,
    coeff_cos: Any,
    coeff_sin: Any,
    modes: ModeTable,
    trig: VmecTrigTables,
    coeffs_internal: bool = False,
    apply_scalxc: bool = False,
    s: Any | None = None,
    derivs: tuple[str, ...] = ("base", "dtheta", "dzeta"),
    use_stacked_dot: bool = True,
) -> tuple[Any, ...]:
    """Synthesize multiple real-space fields (base/derivatives) in one batched call."""
    coeff_cos = jnp.asarray(coeff_cos)
    coeff_sin = jnp.asarray(coeff_sin)
    # In the CPU host-update force path, jnp is temporarily replaced by the
    # NumPy shim.  Full-solve profiles for the scalxc-weighted odd-channel
    # synthesis are faster with three smaller contractions than with one large
    # stacked contraction; keep the fused form for JAX/XLA backends.
    if bool(apply_scalxc) and bool(use_stacked_dot) and type(jnp).__name__ == "_NpModule":
        use_stacked_dot = False
    m_np = np.asarray(modes.m)
    m = jnp.asarray(m_np).astype(jnp.int32)
    n = jnp.asarray(modes.n).astype(jnp.int32)

    if coeff_cos.ndim < 2 or coeff_sin.ndim < 2:
        raise ValueError("Expected coeff arrays with shape (..., ns, K)")
    if coeff_cos.shape != coeff_sin.shape:
        raise ValueError("coeff_cos and coeff_sin must have the same shape")
    if coeff_cos.shape[-1] != m.shape[0]:
        raise ValueError("Mode count mismatch between coefficients and modes")

    if not bool(coeffs_internal):
        scale = _vmec_mode_scaling(m=m, n=n, trig=trig).astype(coeff_cos.dtype)
        scale_shape = (1,) * (coeff_cos.ndim - 1) + (m.shape[0],)
        coeff_cos = coeff_cos * scale.reshape(scale_shape)
        coeff_sin = coeff_sin * scale.reshape(scale_shape)

    if bool(apply_scalxc):
        ns = int(coeff_cos.shape[-2])
        if s is None:
            if ns < 2:
                s = jnp.asarray([0.0], dtype=coeff_cos.dtype)
            else:
                s = jnp.linspace(0.0, 1.0, ns, dtype=coeff_cos.dtype)
        scalxc_mn = _scalxc_mn_for_s(s=s, modes=modes, m_np=m_np, dtype=coeff_cos.dtype)
        scalxc_shape = (1,) * (coeff_cos.ndim - 2) + scalxc_mn.shape
        coeff_cos = coeff_cos * scalxc_mn.reshape(scalxc_shape)
        coeff_sin = coeff_sin * scalxc_mn.reshape(scalxc_shape)

    if not bool(use_stacked_dot):
        out = []
        for deriv in derivs:
            if deriv == "base":
                out.append(
                    vmec_realspace_synthesis(
                        coeff_cos=coeff_cos,
                        coeff_sin=coeff_sin,
                        modes=modes,
                        trig=trig,
                        coeffs_internal=True,
                        apply_scalxc=False,
                        s=s,
                        use_stacked_dot=False,
                    )
                )
            elif deriv == "dtheta":
                out.append(
                    vmec_realspace_synthesis_dtheta(
                        coeff_cos=coeff_cos,
                        coeff_sin=coeff_sin,
                        modes=modes,
                        trig=trig,
                        coeffs_internal=True,
                        apply_scalxc=False,
                        s=s,
                        use_stacked_dot=False,
                    )
                )
            elif deriv == "dzeta":
                out.append(
                    vmec_realspace_synthesis_dzeta_phys(
                        coeff_cos=coeff_cos,
                        coeff_sin=coeff_sin,
                        modes=modes,
                        trig=trig,
                        coeffs_internal=True,
                        apply_scalxc=False,
                        s=s,
                        use_stacked_dot=False,
                    )
                )
            else:
                raise ValueError(f"Unknown deriv={deriv!r}")
        return tuple(out)

    coeff = jnp.concatenate([coeff_cos, coeff_sin], axis=-1)
    phases = []
    for deriv in derivs:
        if deriv == "base":
            phase = _phase_stack_from_trig(modes, trig, "phase_stack")
            if phase is None:
                phase = _vmec_phase_tables_stacked_cached(modes=modes, trig=trig, cache=True)
        elif deriv == "dtheta":
            phase = _phase_stack_from_trig(modes, trig, "phase_dtheta_stack")
            if phase is None:
                phase = _vmec_phase_tables_dtheta_stacked_cached(modes=modes, trig=trig, cache=True)
        elif deriv == "dzeta":
            phase = _phase_stack_from_trig(modes, trig, "phase_dzeta_stack")
            if phase is None:
                phase = _vmec_phase_tables_dzeta_stacked_cached(modes=modes, trig=trig, cache=True)
        else:
            raise ValueError(f"Unknown deriv={deriv!r}")
        phases.append(phase)
    phase_all = jnp.stack(phases, axis=0)
    f_all = einsum("...k,tkij->t...ij", coeff, phase_all)
    return tuple(f_all[i] for i in range(len(derivs)))


def vmec_realspace_analysis(
    *,
    f: Any,
    modes: ModeTable,
    trig: VmecTrigTables,
    parity: str = "both",
) -> tuple[Any, Any]:
    """Project a VMEC real-space field back to Fourier coefficients.

    This is the VMEC-grid counterpart to :func:`vmec_realspace_synthesis`.
    It uses the same `fixaray`-style integration weights (via `dnorm` and
    the theta endpoint half-weights) so that a synth->analyze round-trip
    is numerically stable for *stellarator-symmetric* fields on the VMEC
    internal grid (lasym=False).

    Parameters
    ----------
    f:
        Real-space field on the VMEC internal grid, shape ``(ns, ntheta3, nzeta)``.
    modes:
        Mode table with arrays ``m`` and ``n`` (n is *not* multiplied by nfp).
    trig:
        VMEC trig tables from ``vmec_trig_tables``.

    parity:
        Which parity block to keep on a symmetric grid:
        - ``"cos"``: return cos coefficients and zero the sin block
        - ``"sin"``: return sin coefficients and zero the cos block
        - ``"both"``: return both (note: cross-talk can occur on reduced grids)

    Returns
    -------
    (coeff_cos, coeff_sin):
        Fourier coefficients in the **wout/physical** convention, shape (ns, K).
    """
    f = jnp.asarray(f)
    m_np = np.asarray(modes.m)
    m = jnp.asarray(m_np).astype(jnp.int32)
    n = jnp.asarray(modes.n).astype(jnp.int32)

    if f.ndim != 3:
        raise ValueError(f"Expected f with shape (ns,ntheta,nzeta), got {f.shape}")
    use_full_theta = int(trig.ntheta3) != int(trig.ntheta2)
    if use_full_theta:
        if int(f.shape[1]) < int(trig.ntheta3):
            raise ValueError("Input theta grid is smaller than VMEC ntheta3")
    else:
        if int(f.shape[1]) < int(trig.ntheta2):
            raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    if int(f.shape[2]) != int(trig.cosnv.shape[0]):
        raise ValueError("Input zeta grid does not match trig tables")

    if use_full_theta:
        # LASYM=True: integrate over full poloidal interval [0,2π) using dnorm3.
        nt3 = int(trig.ntheta3)
        f = f[:, :nt3, :]
        dnorm = float(trig.dnorm3)
        w = jnp.full((nt3,), dnorm, dtype=f.dtype)
        f_w = f * w[None, :, None]
    else:
        # LASYM=False: integrate over reduced grid [0,π] with endpoint half-weights.
        nt2 = int(trig.ntheta2)
        f = f[:, :nt2, :]
        dnorm = float(trig.dnorm)
        w = jnp.full((nt2,), dnorm, dtype=f.dtype)
        if hasattr(w, "at"):
            w = w.at[0].set(0.5 * dnorm)
            w = w.at[nt2 - 1].set(0.5 * dnorm)
        else:  # numpy fallback
            w = w.copy()
            w[0] = 0.5 * dnorm
            w[nt2 - 1] = 0.5 * dnorm
        f_w = f * w[None, :, None]

    phase = _phase_stack_from_trig(modes, trig, "phase_stack")
    if phase is not None and phase.shape[0] == 2 * int(m.shape[0]):
        cos_phase = phase[: int(m.shape[0])]
        sin_phase = phase[int(m.shape[0]) :]
    else:
        cos_phase, sin_phase = _vmec_phase_tables_cached(modes=modes, trig=trig, cache=True)
    if not use_full_theta:
        cos_phase = cos_phase[:, : int(trig.ntheta2), :]
        sin_phase = sin_phase[:, : int(trig.ntheta2), :]

    # Convert to *unscaled* helical basis functions: cos(mθ - nζ), sin(mθ - nζ).
    mscale = jnp.asarray(trig.mscale)
    nscale = jnp.asarray(trig.nscale)
    scale = (mscale[m] * nscale[jnp.abs(n)]).astype(f.dtype)
    cos_unscaled = cos_phase / scale[:, None, None]
    sin_unscaled = sin_phase / scale[:, None, None]

    inner_cos = einsum("sij,kij->sk", f_w, cos_unscaled)
    inner_sin = einsum("sij,kij->sk", f_w, sin_unscaled)

    # Norms of the unscaled basis on the reduced VMEC grid:
    # - (m,n) = (0,0) has norm 1
    # - all other modes have norm 1/2
    norm = jnp.where((m == 0) & (n == 0), 1.0, 0.5).astype(f.dtype)
    coeff_cos = inner_cos / norm[None, :]
    coeff_sin = inner_sin / norm[None, :]
    # sin(mθ - nζ) is identically zero for m=n=0; enforce that explicitly.
    coeff_sin = jnp.where((m == 0) & (n == 0), 0.0, coeff_sin)

    parity = str(parity).lower()
    if parity == "cos":
        coeff_sin = jnp.zeros_like(coeff_sin)
    elif parity == "sin":
        coeff_cos = jnp.zeros_like(coeff_cos)
    elif parity != "both":
        raise ValueError("parity must be one of {'cos','sin','both'}")

    return coeff_cos, coeff_sin


def vmec_realspace_synthesis_dtheta(
    *,
    coeff_cos: Any,
    coeff_sin: Any,
    modes: ModeTable,
    trig: VmecTrigTables,
    coeffs_internal: bool = False,
    apply_scalxc: bool = False,
    s: Any | None = None,
    use_stacked_dot: bool = True,
) -> Any:
    """Theta derivative of the VMEC real-space synthesis."""
    coeff_cos = jnp.asarray(coeff_cos)
    coeff_sin = jnp.asarray(coeff_sin)
    m_np = np.asarray(modes.m)
    m = jnp.asarray(m_np).astype(jnp.int32)
    n = jnp.asarray(modes.n).astype(jnp.int32)

    if coeff_cos.ndim < 2 or coeff_sin.ndim < 2:
        raise ValueError("Expected coeff arrays with shape (..., ns, K)")
    if coeff_cos.shape != coeff_sin.shape:
        raise ValueError("coeff_cos and coeff_sin must have the same shape")
    if coeff_cos.shape[-1] != m.shape[0]:
        raise ValueError("Mode count mismatch between coefficients and modes")

    if not bool(coeffs_internal):
        scale = _vmec_mode_scaling(m=m, n=n, trig=trig).astype(coeff_cos.dtype)
        scale_shape = (1,) * (coeff_cos.ndim - 1) + (m.shape[0],)
        coeff_cos = coeff_cos * scale.reshape(scale_shape)
        coeff_sin = coeff_sin * scale.reshape(scale_shape)

    if bool(apply_scalxc):
        ns = int(coeff_cos.shape[-2])
        if s is None:
            if ns < 2:
                s = jnp.asarray([0.0], dtype=coeff_cos.dtype)
            else:
                s = jnp.linspace(0.0, 1.0, ns, dtype=coeff_cos.dtype)
        scalxc_mn = _scalxc_mn_for_s(s=s, modes=modes, m_np=m_np, dtype=coeff_cos.dtype)
        scalxc_shape = (1,) * (coeff_cos.ndim - 2) + scalxc_mn.shape
        coeff_cos = coeff_cos * scalxc_mn.reshape(scalxc_shape)
        coeff_sin = coeff_sin * scalxc_mn.reshape(scalxc_shape)

    if bool(use_stacked_dot):
        phase = _phase_stack_from_trig(modes, trig, "phase_dtheta_stack")
        if phase is None:
            phase = _vmec_phase_tables_dtheta_stacked_cached(modes=modes, trig=trig, cache=True)
        coeff = jnp.concatenate([coeff_cos, coeff_sin], axis=-1)
        f = einsum("...k,kij->...ij", coeff, phase)
    else:
        dcos_phase, dsin_phase = _vmec_phase_tables_dtheta_cached(modes=modes, trig=trig, cache=True)
        f = einsum("...k,kij->...ij", coeff_cos, dcos_phase) + einsum(
            "...k,kij->...ij", coeff_sin, dsin_phase
        )
    return f


def vmec_realspace_synthesis_dzeta_phys(
    *,
    coeff_cos: Any,
    coeff_sin: Any,
    modes: ModeTable,
    trig: VmecTrigTables,
    coeffs_internal: bool = False,
    apply_scalxc: bool = False,
    s: Any | None = None,
    use_stacked_dot: bool = True,
) -> Any:
    """Zeta(physical) derivative of the VMEC real-space synthesis."""
    coeff_cos = jnp.asarray(coeff_cos)
    coeff_sin = jnp.asarray(coeff_sin)
    m_np = np.asarray(modes.m)
    m = jnp.asarray(m_np).astype(jnp.int32)
    n = jnp.asarray(modes.n).astype(jnp.int32)

    if coeff_cos.ndim < 2 or coeff_sin.ndim < 2:
        raise ValueError("Expected coeff arrays with shape (..., ns, K)")
    if coeff_cos.shape != coeff_sin.shape:
        raise ValueError("coeff_cos and coeff_sin must have the same shape")
    if coeff_cos.shape[-1] != m.shape[0]:
        raise ValueError("Mode count mismatch between coefficients and modes")

    if not bool(coeffs_internal):
        scale = _vmec_mode_scaling(m=m, n=n, trig=trig).astype(coeff_cos.dtype)
        scale_shape = (1,) * (coeff_cos.ndim - 1) + (m.shape[0],)
        coeff_cos = coeff_cos * scale.reshape(scale_shape)
        coeff_sin = coeff_sin * scale.reshape(scale_shape)

    if bool(apply_scalxc):
        ns = int(coeff_cos.shape[-2])
        if s is None:
            if ns < 2:
                s = jnp.asarray([0.0], dtype=coeff_cos.dtype)
            else:
                s = jnp.linspace(0.0, 1.0, ns, dtype=coeff_cos.dtype)
        scalxc_mn = _scalxc_mn_for_s(s=s, modes=modes, m_np=m_np, dtype=coeff_cos.dtype)
        scalxc_shape = (1,) * (coeff_cos.ndim - 2) + scalxc_mn.shape
        coeff_cos = coeff_cos * scalxc_mn.reshape(scalxc_shape)
        coeff_sin = coeff_sin * scalxc_mn.reshape(scalxc_shape)

    if bool(use_stacked_dot):
        phase = _phase_stack_from_trig(modes, trig, "phase_dzeta_stack")
        if phase is None:
            phase = _vmec_phase_tables_dzeta_stacked_cached(modes=modes, trig=trig, cache=True)
        coeff = jnp.concatenate([coeff_cos, coeff_sin], axis=-1)
        f = einsum("...k,kij->...ij", coeff, phase)
    else:
        dcos_phase, dsin_phase = _vmec_phase_tables_dzeta_cached(modes=modes, trig=trig, cache=True)
        f = einsum("...k,kij->...ij", coeff_cos, dcos_phase) + einsum(
            "...k,kij->...ij", coeff_sin, dsin_phase
        )
    return f


def vmec_realspace_geom_from_state(
    *,
    state,
    modes: ModeTable,
    trig: VmecTrigTables,
) -> dict[str, Any]:
    """Compute VMEC real-space geometry fields on the internal grid."""
    Rcos = jnp.asarray(state.Rcos)
    Rsin = jnp.asarray(state.Rsin)
    Zcos = jnp.asarray(state.Zcos)
    Zsin = jnp.asarray(state.Zsin)
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
    has_lambda = hasattr(state, "Lcos") and hasattr(state, "Lsin")
    if has_lambda:
        coeff_cos_stack = jnp.stack([Rcos, Zcos, jnp.asarray(state.Lcos)], axis=0)
        coeff_sin_stack = jnp.stack([Rsin, Zsin, jnp.asarray(state.Lsin)], axis=0)
    else:
        coeff_cos_stack = jnp.stack([Rcos, Zcos], axis=0)
        coeff_sin_stack = jnp.stack([Rsin, Zsin], axis=0)
    rz = vmec_realspace_synthesis(
        coeff_cos=coeff_cos_stack,
        coeff_sin=coeff_sin_stack,
        modes=modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
    )
    rz_t = vmec_realspace_synthesis_dtheta(
        coeff_cos=coeff_cos_stack,
        coeff_sin=coeff_sin_stack,
        modes=modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
    )
    rz_p = vmec_realspace_synthesis_dzeta_phys(
        coeff_cos=coeff_cos_stack,
        coeff_sin=coeff_sin_stack,
        modes=modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
    )
    R, Z = rz[0], rz[1]
    Ru, Zu = rz_t[0], rz_t[1]
    Rv, Zv = rz_p[0], rz_p[1]
    if has_lambda:
        Lu = rz_t[2]
        Lv = rz_p[2]
    else:
        Lu = None
        Lv = None
    return {
        "R": R,
        "Z": Z,
        "Ru": Ru,
        "Zu": Zu,
        "Rv": Rv,
        "Zv": Zv,
        "Lu": Lu,
        "Lv": Lv,
    }
