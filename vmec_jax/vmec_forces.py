"""VMEC force/residue kernels for parity work.

This module implements a direct, array-based port of VMEC2000's ``forces`` core
for the **R/Z** equations, operating on:

- VMEC even/odd-m real-space decomposition (odd stored in 1/sqrt(s) form),
- half-mesh quantities from :mod:`vmec_jax.vmec_bcovar`.

Scope
-----
This is a parity/debug kernel used to validate the algebra and staggering.
It is *not* yet the full VMEC solver pipeline (no vacuum/free boundary, no 2D
preconditioner, and no full lambda residue parity), but it *does* include the
VMEC constraint-force pipeline (`tcon` + `alias`) for fixed-boundary parity.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from contextlib import contextmanager
import time

import numpy as np

from ._compat import jnp, has_jax, jax, tree_util
from ._solve_runtime import _parse_iter_list
from .fourier import project_to_modes
from .fourier import eval_fourier, eval_fourier_dtheta, eval_fourier_dzeta_phys
from .field import lamscale_from_phips
from .grids import AngleGrid
from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from .vmec_constraints import (
    alias_gcon,
    precondn_diag_axd1_from_bcovar,
    tcon_from_cached_precondn_diag,
    tcon_from_tcon0_heuristic,
)
from .vmec_tomnsp import (
    TomnspsMasks,
    TomnspsRZL,
    VmecTrigTables,
    tomnsps_rzl,
    tomnspa_rzl,
    vmec_trig_tables,
)
from .nyquist import nyquist_basis_from_wout
from .vmec_parity import (
    internal_odd_from_physical_vmec_jlam,
    internal_odd_from_physical_vmec_m1,
    split_rzl_even_odd_m,
    vmec_m1_internal_to_physical_signed,
)

_PRODUCTION_VMEC_BCOVAR_HALF_MESH_FROM_WOUT = vmec_bcovar_half_mesh_from_wout


@contextmanager
def _optional_jax_context(factory):
    if has_jax():
        try:
            context = factory()
        except Exception:
            context = None
        if context is not None:
            try:
                context.__enter__()
            except Exception:
                pass
            else:
                try:
                    yield
                except BaseException as exc:
                    suppress = context.__exit__(type(exc), exc, exc.__traceback__)
                    if not suppress:
                        raise
                else:
                    context.__exit__(None, None, None)
                return
    yield


def _named_scope(name: str):
    return _optional_jax_context(lambda: jax.named_scope(name))


def _trace(name: str):
    return _optional_jax_context(lambda: jax.profiler.TraceAnnotation(name))


def _vmec_force_profile_enabled() -> bool:
    value = os.environ.get("VMEC_JAX_PROFILE_FORCE", "")
    return value.strip().lower() not in ("", "0", "false", "no")


def _vmec_force_profile_log(stage: str, start: float | None = None, **extra) -> None:
    if not _vmec_force_profile_enabled():
        return
    payload = {"stage": stage}
    if start is not None:
        payload["elapsed_s"] = time.perf_counter() - start
    payload.update(extra)
    print(f"[vmec_jax force] {payload}", flush=True)


def _bcovar_with_parity_aux(result: Any):
    """Normalize production and synthetic bcovar return shapes.

    Production calls with ``return_parity_aux=True`` return ``(bcovar, aux)``.
    Some focused tests and user diagnostics monkeypatch a lighter bcovar-like
    object; synthesize the parity channels needed by the force algebra for
    those non-production paths instead of requiring every mock to duplicate the
    full auxiliary object.
    """

    if isinstance(result, tuple) and len(result) == 2:
        return result

    bc = result.bc if _looks_like_force_kernel_payload(result) else result
    bc = _complete_synthetic_bcovar_payload(bc)
    jac = getattr(bc, "jac", SimpleNamespace())
    base = getattr(jac, "sqrtg", getattr(bc, "bsq", jnp.asarray(0.0)))
    zeros = jnp.zeros_like(jnp.asarray(base))
    parity = SimpleNamespace(
        pr1_even=getattr(jac, "r12", zeros),
        pr1_odd=zeros,
        pz1_even=zeros,
        pz1_odd=zeros,
        pru_even=getattr(jac, "ru12", zeros),
        pru_odd=zeros,
        pzu_even=getattr(jac, "zu12", zeros),
        pzu_odd=zeros,
        prv_even=zeros,
        prv_odd=zeros,
        pzv_even=zeros,
        pzv_odd=zeros,
        lu_odd=zeros,
        lv_odd=zeros,
    )
    return bc, parity


def _looks_like_force_kernel_payload(value: Any) -> bool:
    """Detect a force-kernel mock accidentally routed through bcovar."""

    return hasattr(value, "state") and hasattr(value, "bc") and hasattr(value, "tcon")


def _bcovar_payload_for_shape(result: Any) -> Any:
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    if _looks_like_force_kernel_payload(result):
        return result.bc
    return result


def _bcovar_matches_radial_grid(result: Any, s: Any) -> bool:
    """Return whether a bcovar-like payload is compatible with this solve grid."""

    bc = _bcovar_payload_for_shape(result)
    expected = int(np.shape(s)[0])
    for attr in ("gij_b_uu", "guu", "lu_e", "bsq"):
        if hasattr(bc, attr):
            shape = np.shape(getattr(bc, attr))
            if shape:
                return int(shape[0]) == expected
    jac = getattr(bc, "jac", None)
    if jac is not None and hasattr(jac, "sqrtg"):
        shape = np.shape(jac.sqrtg)
        if shape:
            return int(shape[0]) == expected
    return True


def _complete_synthetic_bcovar_payload(bc: Any) -> Any:
    """Fill minimal force fields for intentionally lightweight bcovar mocks."""

    if hasattr(bc, "lu_e") and hasattr(bc, "gij_b_uu"):
        return bc

    jac = getattr(bc, "jac", SimpleNamespace())
    base = getattr(jac, "sqrtg", getattr(bc, "bsq", jnp.asarray(0.0)))
    zeros = jnp.zeros_like(jnp.asarray(base))
    jac_fields = dict(getattr(jac, "__dict__", {}))
    jac_fields.update(
        sqrtg=getattr(jac, "sqrtg", base),
        r12=getattr(jac, "r12", zeros),
        ru12=getattr(jac, "ru12", zeros),
        zu12=getattr(jac, "zu12", zeros),
        rs=getattr(jac, "rs", zeros),
        zs=getattr(jac, "zs", zeros),
        tau=getattr(jac, "tau", zeros),
    )
    jac = SimpleNamespace(**jac_fields)
    fields = dict(getattr(bc, "__dict__", {}))
    fields.update(
        jac=jac,
        bsq=getattr(bc, "bsq", zeros),
        lu_e=getattr(bc, "lu_e", getattr(bc, "bsubu", zeros)),
        lv_e=getattr(bc, "lv_e", getattr(bc, "bsubv", zeros)),
        gij_b_uu=getattr(bc, "gij_b_uu", getattr(bc, "guu", zeros)),
        gij_b_uv=getattr(bc, "gij_b_uv", getattr(bc, "guv", zeros)),
        gij_b_vv=getattr(bc, "gij_b_vv", getattr(bc, "gvv", zeros)),
    )
    return SimpleNamespace(**fields)


def _production_bcovar_half_mesh_from_wout(*, expected_s: Any | None = None, **kwargs):
    """Call the real bcovar implementation after an impossible mock leak."""

    try:
        result = _PRODUCTION_VMEC_BCOVAR_HALF_MESH_FROM_WOUT(**kwargs)
        if not _looks_like_force_kernel_payload(result) and (
            expected_s is None or _bcovar_matches_radial_grid(result, expected_s)
        ):
            return result
    except Exception:
        pass
    from . import vmec_bcovar as _vmec_bcovar_module

    reloaded = importlib.reload(_vmec_bcovar_module)
    return reloaded.vmec_bcovar_half_mesh_from_wout(**kwargs)


class _WoutProfileProxy:
    """WOUT-like view with input-deck flux/profile overrides."""

    __slots__ = ("_base", "_overrides")

    def __init__(self, base, overrides):
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


def _resolve_force_wout_and_pressure(*, wout: Any, indata: Any | None, s: Any):
    """Fill missing WOUT flux functions from input profiles for solver diagnostics."""

    need_fill = (indata is not None) and any(not hasattr(wout, name) for name in ("phipf", "phips", "chipf", "pres"))
    if not need_fill:
        return wout, None

    from .energy import flux_profiles_from_indata
    from .profiles import eval_profiles

    signgs = int(getattr(wout, "signgs", 1))
    flux = flux_profiles_from_indata(indata, s, signgs=signgs)
    s_half = s if int(s.shape[0]) < 2 else jnp.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])], axis=0)
    return (
        _WoutProfileProxy(
            wout,
            {
                "phipf": flux.phipf,
                "phips": flux.phips,
                "chipf": jnp.asarray(flux.chipf),
                "signgs": int(signgs),
            },
        ),
        eval_profiles(indata, s_half).get("pressure", None),
    )


def _apply_freeb_edge_forcing(ctx):
    """Apply VMEC free-boundary edge pressure/vacuum forcing to A-kernels."""

    ctx = SimpleNamespace(**ctx)
    armn_e, armn_o, azmn_e, azmn_o = ctx.armn_e, ctx.armn_o, ctx.azmn_e, ctx.azmn_o
    pr1_0, pr1_1 = ctx.pr1_0, ctx.pr1_1
    pru_0, pru_1, pzu_0, pzu_1 = ctx.pru_0, ctx.pru_1, ctx.pzu_0, ctx.pzu_1
    freeb_bsqvac_half = ctx.freeb_bsqvac_half
    if freeb_bsqvac_half is None:
        return armn_e, armn_o, azmn_e, azmn_o
    vac_full = jnp.asarray(freeb_bsqvac_half)
    if vac_full.shape == pr1_0[-1].shape:
        vac_edge = vac_full
    elif vac_full.shape == pr1_0.shape:
        vac_edge = vac_full[-1]
    else:
        raise ValueError(
            "freeb_bsqvac_half shape mismatch: "
            f"expected edge {pr1_0[-1].shape} or full {pr1_0.shape}, got {vac_full.shape}"
        )
    if vac_edge.shape != pr1_0[-1].shape:
        raise ValueError(f"freeb_bsqvac_half edge shape mismatch: expected {pr1_0[-1].shape}, got {vac_edge.shape}")
    pres = jnp.asarray(getattr(ctx.wout, "pres", jnp.zeros((int(ctx.s.shape[0]),), dtype=vac_edge.dtype)))
    pres_edge = jnp.asarray(pres[-1], dtype=vac_edge.dtype) if pres.ndim > 0 else jnp.asarray(pres, dtype=vac_edge.dtype)
    if ctx.freeb_pres_scale is not None:
        pres_edge = jnp.asarray(pres_edge) * jnp.asarray(ctx.freeb_pres_scale, dtype=vac_edge.dtype)
    elif (ctx.indata is not None) and int(ctx.s.shape[0]) >= 2:
        try:
            from .profiles import eval_profiles

            hs_f = float(np.asarray(ctx.s[1] - ctx.s[0], dtype=float))
            sedge = hs_f * (float(int(ctx.s.shape[0])) - 1.5)
            p_edge_prof = eval_profiles(ctx.indata, jnp.asarray([sedge], dtype=jnp.asarray(ctx.s).dtype)).get("pressure", None)
            p_one_prof = eval_profiles(ctx.indata, jnp.asarray([1.0], dtype=jnp.asarray(ctx.s).dtype)).get("pressure", None)
            if p_edge_prof is not None and p_one_prof is not None:
                p_edge_val = float(np.asarray(p_edge_prof, dtype=float).reshape(-1)[0])
                p_one_val = float(np.asarray(p_one_prof, dtype=float).reshape(-1)[0])
                pres_edge = (
                    jnp.asarray((p_one_val / p_edge_val) * float(np.asarray(pres_edge, dtype=float)), dtype=vac_edge.dtype)
                    if p_edge_val != 0.0
                    else jnp.asarray(p_edge_val, dtype=vac_edge.dtype)
                )
        except Exception:
            pass
    gcon_edge = vac_edge + pres_edge
    rbsq_edge = (
        gcon_edge
        * (pr1_0[-1] + pr1_1[-1])
        * jnp.asarray(ctx.ohs, dtype=vac_edge.dtype)
        * jnp.asarray(float(os.getenv("VMEC_JAX_FREEB_RBSQ_SCALE", "1.0") or 1.0), dtype=vac_edge.dtype)
    )
    ru0_edge = jnp.asarray(pru_0[-1]) + jnp.asarray(pru_1[-1])
    zu0_edge = jnp.asarray(pzu_0[-1]) + jnp.asarray(pzu_1[-1])
    if ctx.iter_idx is not None:
        env = os.getenv("VMEC_JAX_DUMP_FREEB_COUPLING", "").strip().lower()
        if env not in ("", "0", "false", "no"):
            outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
            outdir.mkdir(parents=True, exist_ok=True)
            plasma_bsq_edge = jnp.asarray(ctx.bc.bsq[-1], dtype=vac_edge.dtype)
            plasma_bsq_edge_extrap = (
                1.5 * jnp.asarray(ctx.bc.bsq[-1], dtype=vac_edge.dtype) - 0.5 * jnp.asarray(ctx.bc.bsq[-2], dtype=vac_edge.dtype)
                if int(ctx.bc.bsq.shape[0]) >= 2
                else plasma_bsq_edge
            )
            np.savez(
                outdir / f"freeb_coupling_iter{int(ctx.iter_idx)}.npz",
                gcon_edge=np.asarray(gcon_edge),
                rbsq_edge=np.asarray(rbsq_edge),
                bsqvac_edge=np.asarray(vac_edge),
                pres_edge=np.asarray(pres_edge),
                plasma_bsq_edge=np.asarray(plasma_bsq_edge),
                plasma_bsq_edge_extrap=np.asarray(plasma_bsq_edge_extrap),
                dbsq_edge_proxy=np.asarray(jnp.abs(gcon_edge - plasma_bsq_edge_extrap)),
                pr1_even_edge=np.asarray(pr1_0[-1]),
                pr1_odd_edge=np.asarray(pr1_1[-1]),
                pzu0_edge=np.asarray(zu0_edge),
                pru0_edge=np.asarray(ru0_edge),
                pzu0_even_edge=np.asarray(pzu_0[-1]),
                pru0_even_edge=np.asarray(pru_0[-1]),
                zu0_phys_edge=np.asarray(zu0_edge),
                ru0_phys_edge=np.asarray(ru0_edge),
            )
    return (
        _add_edge_row(armn_e, zu0_edge * rbsq_edge),
        _add_edge_row(armn_o, zu0_edge * rbsq_edge),
        _add_edge_row(azmn_e, -ru0_edge * rbsq_edge),
        _add_edge_row(azmn_o, -ru0_edge * rbsq_edge),
    )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VmecRZForceKernels:
    """Force kernels on the full angular grid, split by m-parity.

    Array fields are on the ``(ns, ntheta, nzeta)`` grid unless noted.
    Geometry parity fields use VMEC's internal decomposition
    ``X = X_even + sqrt(s) * X_odd_internal``.
    """

    armn_e: Any  # (ns, ntheta, nzeta)
    armn_o: Any  # (ns, ntheta, nzeta)
    brmn_e: Any  # (ns, ntheta, nzeta)
    brmn_o: Any  # (ns, ntheta, nzeta)
    crmn_e: Any  # (ns, ntheta, nzeta)
    crmn_o: Any  # (ns, ntheta, nzeta)
    azmn_e: Any  # (ns, ntheta, nzeta)
    azmn_o: Any  # (ns, ntheta, nzeta)
    bzmn_e: Any  # (ns, ntheta, nzeta)
    bzmn_o: Any  # (ns, ntheta, nzeta)
    czmn_e: Any  # (ns, ntheta, nzeta)
    czmn_o: Any  # (ns, ntheta, nzeta)
    bc: Any
    arcon_e: Any  # (ns, ntheta, nzeta)
    arcon_o: Any  # (ns, ntheta, nzeta)
    azcon_e: Any  # (ns, ntheta, nzeta)
    azcon_o: Any  # (ns, ntheta, nzeta)
    gcon: Any  # (ns, ntheta, nzeta)
    pr1_even: Any  # (ns, ntheta, nzeta)
    pr1_odd: Any  # (ns, ntheta, nzeta)
    pz1_even: Any  # (ns, ntheta, nzeta)
    pz1_odd: Any  # (ns, ntheta, nzeta)
    pru_even: Any  # (ns, ntheta, nzeta)
    pru_odd: Any  # (ns, ntheta, nzeta)
    pzu_even: Any  # (ns, ntheta, nzeta)
    pzu_odd: Any  # (ns, ntheta, nzeta)
    prv_even: Any  # (ns, ntheta, nzeta)
    prv_odd: Any  # (ns, ntheta, nzeta)
    pzv_even: Any  # (ns, ntheta, nzeta)
    pzv_odd: Any  # (ns, ntheta, nzeta)
    tcon: Any | None = None  # (ns,)
    constraint_rcon0: Any | None = None  # (ns, ntheta, nzeta)
    constraint_zcon0: Any | None = None  # (ns, ntheta, nzeta)

    def tree_flatten(self):
        return tuple(getattr(self, field.name) for field in fields(self)), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@dataclass(frozen=True)
class VmecConstraintKernels:
    """Constraint-force kernels produced by the `alias` pipeline."""

    rcon_force: Any  # (ns, ntheta, nzeta)
    zcon_force: Any  # (ns, ntheta, nzeta)
    arcon_e: Any  # (ns, ntheta, nzeta)
    arcon_o: Any  # (ns, ntheta, nzeta)
    azcon_e: Any  # (ns, ntheta, nzeta)
    azcon_o: Any  # (ns, ntheta, nzeta)
    gcon: Any  # (ns, ntheta, nzeta)
    tcon: Any  # (ns,)
    ard1: Any  # (ns,)
    azd1: Any  # (ns,)
    rcon0: Any  # (ns, ntheta, nzeta)
    zcon0: Any  # (ns, ntheta, nzeta)


def _constraint_preconditioner_and_tcon(
    *,
    s,
    ns: int,
    dtype,
    trig: VmecTrigTables,
    wout,
    bc,
    ru0,
    zu0,
    constraint_tcon0: float | None,
    tcon_override: Any | None,
    precond_diag_override: tuple[Any, Any] | None,
    precond_active: Any | None,
    tcon_active: Any | None,
):
    """Select VMEC constraint preconditioner diagonals and `tcon` profile."""
    tcon0_val = jnp.asarray(0.0 if constraint_tcon0 is None else constraint_tcon0, dtype=dtype)

    def _diag_from_bc():
        return precondn_diag_axd1_from_bcovar(
            trig=trig,
            s=s,
            bsq=bc.bsq,
            r12=bc.jac.r12,
            sqrtg=bc.jac.sqrtg,
            ru12=bc.jac.ru12,
            zu12=bc.jac.zu12,
        )

    def _safe_tcon_from_diag(ard1, azd1):
        tcon_tmp = tcon_from_cached_precondn_diag(
            tcon0=tcon0_val,
            trig=trig,
            s=s,
            lasym=bool(wout.lasym),
            ard1=ard1,
            azd1=azd1,
            ru0=ru0,
            zu0=zu0,
        )
        tcon_heur = tcon_from_tcon0_heuristic(
            tcon0=tcon0_val,
            s=s,
            trig=trig,
            lasym=bool(wout.lasym),
        )
        return jnp.where(jnp.all(jnp.isfinite(tcon_tmp)), tcon_tmp, tcon_heur)

    use_dynamic = (precond_active is not None) or (tcon_active is not None)
    if use_dynamic:
        if jax is None:
            raise RuntimeError("Dynamic constraint overrides require JAX.")
        precond_override = precond_diag_override
        if precond_override is None:
            precond_override = (
                jnp.zeros((ns,), dtype=dtype),
                jnp.zeros((ns,), dtype=dtype),
            )
        tcon_override_arr = tcon_override
        if tcon_override_arr is None:
            tcon_override_arr = jnp.zeros((ns,), dtype=dtype)

        use_precond = precond_active if precond_active is not None else bool(precond_diag_override is not None)
        use_tcon = tcon_active if tcon_active is not None else bool(tcon_override is not None)
        use_precond = jnp.asarray(use_precond, dtype=bool)
        use_tcon = jnp.asarray(use_tcon, dtype=bool)

        def _diag_override(_):
            return precond_override

        def _diag_from_bc_cond(_):
            return _diag_from_bc()

        ard1, azd1 = jax.lax.cond(use_precond, _diag_override, _diag_from_bc_cond, operand=None)

        def _tcon_from_diag(_):
            return _safe_tcon_from_diag(ard1, azd1)

        def _tcon_override(_):
            return jnp.asarray(tcon_override_arr, dtype=dtype)

        tcon = jax.lax.cond(use_tcon, _tcon_override, _tcon_from_diag, operand=None)
        return ard1, azd1, tcon

    if tcon_override is None:
        if precond_diag_override is None:
            ard1, azd1 = _diag_from_bc()
        else:
            ard1, azd1 = precond_diag_override

        # VMEC2000 updates `tcon(js)` only when refreshing the 1D
        # preconditioner blocks; between refreshes, callers may reuse it.
        return ard1, azd1, _safe_tcon_from_diag(ard1, azd1)

    tcon = jnp.asarray(tcon_override, dtype=dtype)
    if precond_diag_override is None:
        ard1 = jnp.zeros((ns,), dtype=dtype)
        azd1 = jnp.zeros((ns,), dtype=dtype)
    else:
        ard1, azd1 = precond_diag_override
    return ard1, azd1, tcon


def _maybe_dump_iter_npz(env_name: str, *, iter_idx: int | None, stem: str, arrays) -> None:
    env = os.getenv(env_name, "")
    if not env or env == "0" or iter_idx is None:
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if callable(arrays):
        arrays = arrays()
    np.savez(outdir / f"{stem}_iter{int(iter_idx)}.npz", **arrays)


def _pshalf_from_s(s: Any) -> Any:
    s = jnp.asarray(s)
    if s.shape[0] < 2:
        return jnp.sqrt(jnp.maximum(s, 0.0))
    sh = 0.5 * (s[1:] + s[:-1])
    p = jnp.concatenate([sh[:1], sh], axis=0)
    return jnp.sqrt(jnp.maximum(p, 0.0))


def _with_axis_zero(a):
    a = jnp.asarray(a)
    if a.shape[0] == 0:
        return a
    return jnp.concatenate([jnp.zeros_like(a[:1]), a[1:]], axis=0)


def _avg_forward_half_to_int(a):
    """VMEC's forward-average from half mesh to integer mesh along s."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    body = 0.5 * (a[:-1] + a[1:])
    tail = 0.5 * a[-1:]
    return jnp.concatenate([body, tail], axis=0)


def _sum_forward_half(a):
    """VMEC's forward-sum (a(js) <- a(js) + a(js+1)) along s."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    body = a[:-1] + a[1:]
    tail = a[-1:]
    return jnp.concatenate([body, tail], axis=0)


def _diff_forward_half(a, b):
    """VMEC's forward difference a(js) <- a(js+1) - a(js) plus a 2-term average b."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.shape[0] < 2:
        return a
    body = a[1:] - a[:-1] + 0.5 * (b[:-1] + b[1:])
    tail = -a[-1:] + 0.5 * b[-1:]
    return jnp.concatenate([body, tail], axis=0)


def _constraint_kernels_from_state(
    *,
    state,
    static,
    wout,
    bc,
    pru_0,
    pru_1,
    pzu_0,
    pzu_1,
    constraint_tcon0: float | None,
    tcon_override: Any | None = None,
    precond_diag_override: tuple[Any, Any] | None = None,
    precond_active: Any | None = None,
    tcon_active: Any | None = None,
    rcon0_override: Any | None = None,
    zcon0_override: Any | None = None,
    trig: VmecTrigTables | None = None,
    iter_idx: int | None = None,
) -> VmecConstraintKernels:
    """Compute VMEC constraint-force kernels from state/parity fields.

    This follows the fixed-boundary pipeline in `funct3d` -> `alias` -> `forces`.
    """
    s = jnp.asarray(static.s)
    ns = int(s.shape[0])
    dtype = jnp.asarray(state.Rcos).dtype

    use_dynamic = (precond_active is not None) or (tcon_active is not None)
    if not use_dynamic:
        if (constraint_tcon0 is None or float(constraint_tcon0) == 0.0) and (tcon_override is None):
            z = jnp.zeros_like(pru_0)
            tcon = jnp.zeros((ns,), dtype=dtype)
            z1 = jnp.zeros((ns,), dtype=dtype)
            return VmecConstraintKernels(
                rcon_force=z,
                zcon_force=z,
                arcon_e=z,
                arcon_o=z,
                azcon_e=z,
                azcon_o=z,
                gcon=z,
                tcon=tcon,
                ard1=z1,
                azd1=z1,
                rcon0=z,
                zcon0=z,
            )

    # xmpq(m,1) = m*(m-1).
    if getattr(static, "m_xmpq1", None) is not None:
        xmpq1 = jnp.asarray(static.m_xmpq1, dtype=dtype)
    else:
        m_k = jnp.asarray(static.modes.m, dtype=dtype)
        xmpq1 = m_k * (m_k - 1.0)

    if getattr(static, "m_is_even", None) is not None:
        mask_even = jnp.asarray(static.m_is_even, dtype=dtype)
        mask_m1 = jnp.asarray(static.m_is_m1, dtype=dtype)
        mask_odd_rest = jnp.asarray(static.m_is_odd_rest, dtype=dtype)
    else:
        m_modes = np.asarray(static.modes.m, dtype=int)
        mask_even = jnp.asarray((m_modes % 2) == 0, dtype=dtype)
        mask_m1 = jnp.asarray(m_modes == 1, dtype=dtype)
        mask_odd_rest = jnp.asarray((m_modes % 2 == 1) & (m_modes != 1), dtype=dtype)

    coeff_cos_stack = jnp.stack(
        [
            state.Rcos * xmpq1 * mask_even,
            state.Zcos * xmpq1 * mask_even,
            state.Rcos * xmpq1 * mask_m1,
            state.Rcos * xmpq1 * mask_odd_rest,
            state.Zcos * xmpq1 * mask_m1,
            state.Zcos * xmpq1 * mask_odd_rest,
        ],
        axis=0,
    )
    coeff_sin_stack = jnp.stack(
        [
            state.Rsin * xmpq1 * mask_even,
            state.Zsin * xmpq1 * mask_even,
            state.Rsin * xmpq1 * mask_m1,
            state.Rsin * xmpq1 * mask_odd_rest,
            state.Zsin * xmpq1 * mask_m1,
            state.Zsin * xmpq1 * mask_odd_rest,
        ],
        axis=0,
    )
    if trig is not None:
        from .vmec_realspace import vmec_realspace_synthesis

        eval_stack = vmec_realspace_synthesis(
            coeff_cos=coeff_cos_stack,
            coeff_sin=coeff_sin_stack,
            modes=static.modes,
            trig=trig,
            coeffs_internal=True,
            apply_scalxc=False,
            s=s,
        )
    else:
        eval_stack = eval_fourier(
            coeff_cos_stack,
            coeff_sin_stack,
            static.basis,
            coeffs_internal=True,
        )
    rcon_even, zcon_even, rcon_odd_m1, rcon_odd_rest, zcon_odd_m1, zcon_odd_rest = eval_stack

    rcon_odd_int = internal_odd_from_physical_vmec_m1(odd_m1_phys=rcon_odd_m1, odd_mge2_phys=rcon_odd_rest, s=s)
    zcon_odd_int = internal_odd_from_physical_vmec_m1(odd_m1_phys=zcon_odd_m1, odd_mge2_phys=zcon_odd_rest, s=s)

    psqrts = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    rcon_phys = jnp.asarray(rcon_even) + psqrts * jnp.asarray(rcon_odd_int)
    zcon_phys = jnp.asarray(zcon_even) + psqrts * jnp.asarray(zcon_odd_int)

    # Constraint baseline (VMEC `rcon0/zcon0`):
    # initialize from edge profile scaled by s, then persist/update in caller.
    if (rcon0_override is None) or (zcon0_override is None):
        rcon0 = (s[:, None, None] * jnp.asarray(rcon_phys[-1])[None, :, :]).astype(jnp.asarray(rcon_phys).dtype)
        zcon0 = (s[:, None, None] * jnp.asarray(zcon_phys[-1])[None, :, :]).astype(jnp.asarray(zcon_phys).dtype)
    else:
        rcon0 = jnp.asarray(rcon0_override, dtype=jnp.asarray(rcon_phys).dtype)
        zcon0 = jnp.asarray(zcon0_override, dtype=jnp.asarray(zcon_phys).dtype)
        if rcon0.shape != rcon_phys.shape:
            raise ValueError(
                f"rcon0_override shape mismatch: expected {rcon_phys.shape}, got {rcon0.shape}"
            )
        if zcon0.shape != zcon_phys.shape:
            raise ValueError(
                f"zcon0_override shape mismatch: expected {zcon_phys.shape}, got {zcon0.shape}"
            )

    # Physical ru0/zu0 for ztemp formation.
    ru0 = jnp.asarray(pru_0) + psqrts * jnp.asarray(pru_1)
    zu0 = jnp.asarray(pzu_0) + psqrts * jnp.asarray(pzu_1)

    ztemp = (rcon_phys - rcon0) * ru0 + (zcon_phys - zcon0) * zu0

    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(wout.nfp),
            mmax=int(wout.mpol) - 1,
            nmax=int(wout.ntor),
            lasym=bool(wout.lasym),
            dtype=jnp.asarray(ztemp).dtype,
        )

    ard1, azd1, tcon = _constraint_preconditioner_and_tcon(
        s=s,
        ns=ns,
        dtype=dtype,
        trig=trig,
        wout=wout,
        bc=bc,
        ru0=ru0,
        zu0=zu0,
        constraint_tcon0=constraint_tcon0,
        tcon_override=tcon_override,
        precond_diag_override=precond_diag_override,
        precond_active=precond_active,
        tcon_active=tcon_active,
    )

    gcon = alias_gcon(
        ztemp=ztemp,
        trig=trig,
        ntor=int(wout.ntor),
        mpol=int(wout.mpol),
        signgs=int(wout.signgs),
        tcon=tcon,
        lasym=bool(wout.lasym),
    )

    # Optional debug dump of the constraint pipeline for parity work.
    _maybe_dump_iter_npz(
        "VMEC_JAX_DUMP_CONSTRAINTS",
        iter_idx=iter_idx,
        stem="constraints_raw",
        arrays=lambda: {
            "gcon": np.asarray(gcon),
            "ztemp": np.asarray(ztemp),
            "ru0": np.asarray(ru0),
            "zu0": np.asarray(zu0),
            "rcon0": np.asarray(rcon0),
            "zcon0": np.asarray(zcon0),
            "rcon": np.asarray(rcon_phys),
            "zcon": np.asarray(zcon_phys),
            "tcon": np.asarray(tcon),
            "ard1": np.asarray(ard1),
            "azd1": np.asarray(azd1),
            "ns": int(ns),
            "ntheta3": int(trig.ntheta3),
            "nzeta": int(trig.cosnv.shape[0]),
        },
    )

    env_bcovar = os.getenv("VMEC_JAX_DUMP_BCOVAR", "")
    if env_bcovar and env_bcovar != "0" and iter_idx is not None:
        sqrtg = np.asarray(bc.jac.sqrtg, dtype=float)
        twopi = float(2.0 * np.pi)
        denom = float(int(wout.signgs)) * sqrtg * twopi
        overg = np.where(denom != 0.0, 1.0 / denom, 0.0)
        phipog_vmec = np.where(sqrtg != 0.0, 1.0 / sqrtg, 0.0)
        w_theta = np.asarray(trig.cosmui3[:, 0], dtype=float) / float(np.asarray(trig.mscale[0]))
        wint2 = w_theta[:, None] * np.ones((int(np.asarray(trig.cosnv).shape[0]),), dtype=float)[None, :]
        wint3 = np.broadcast_to(wint2[None, :, :], sqrtg.shape).copy()
        if wint3.shape[0] > 0:
            wint3[0, :, :] = 0.0
        bsupu = np.asarray(bc.bsupu, dtype=float)
        bsupv = np.asarray(bc.bsupv, dtype=float)
        bsubu = np.asarray(bc.bsubu, dtype=float)
        bsubv = np.asarray(bc.bsubv, dtype=float)
        _maybe_dump_iter_npz(
            "VMEC_JAX_DUMP_BCOVAR",
            iter_idx=iter_idx,
            stem="bcovar_raw",
            arrays={
                "r12": np.asarray(bc.jac.r12, dtype=float),
                "sqrtg": sqrtg,
                "tau": np.asarray(bc.jac.tau, dtype=float),
                "ru12": np.asarray(bc.jac.ru12, dtype=float),
                "zu12": np.asarray(bc.jac.zu12, dtype=float),
                "bsupu": bsupu,
                "bsupv": bsupv,
                "bsubu": bsubu,
                "bsubv": bsubv,
                "b2": (bsupu * bsubu) + (bsupv * bsubv),
                "bsq": np.asarray(bc.bsq, dtype=float),
                "overg": overg,
                "phipog_vmec": phipog_vmec,
                "wint": wint3,
                "ns": int(ns),
                "ntheta3": int(trig.ntheta3),
                "nzeta": int(trig.cosnv.shape[0]),
            },
        )

    rcon_force = (rcon_phys - rcon0) * gcon
    zcon_force = (zcon_phys - zcon0) * gcon

    arcon_e = ru0 * gcon
    azcon_e = zu0 * gcon
    arcon_o = arcon_e * psqrts
    azcon_o = azcon_e * psqrts

    return VmecConstraintKernels(
        rcon_force=rcon_force,
        zcon_force=zcon_force,
        arcon_e=arcon_e,
        arcon_o=arcon_o,
        azcon_e=azcon_e,
        azcon_o=azcon_o,
        gcon=gcon,
        tcon=tcon,
        ard1=ard1,
        azd1=azd1,
        rcon0=rcon0,
        zcon0=zcon0,
    )


def _diff_forward_half_noavg(a):
    """VMEC's forward difference a(js) <- a(js+1) - a(js)."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    body = a[1:] - a[:-1]
    tail = -a[-1:]
    return jnp.concatenate([body, tail], axis=0)


def _avg_forward_half(a):
    """VMEC's forward average a(js) <- 0.5*(a(js)+a(js+1))."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    body = 0.5 * (a[:-1] + a[1:])
    tail = 0.5 * a[-1:]
    return jnp.concatenate([body, tail], axis=0)


def _add_edge_row(a, delta):
    a = jnp.asarray(a)
    if a.shape[0] == 0:
        return a
    return jnp.concatenate([a[:-1], a[-1:] + jnp.asarray(delta)[None, ...]], axis=0)


def _avg_forward_half_to_int_or_zero(a):
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return jnp.zeros_like(a)
    body = 0.5 * (a[:-1] + a[1:])
    tail = 0.5 * a[-1:]
    return jnp.concatenate([body, tail], axis=0)


def _scale_lambda_full_mesh(a, lamscale):
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    return jnp.concatenate([a[:1], -lamscale * a[1:]], axis=0)


def _scale_lambda_full_mesh_zero_axis(a, lamscale):
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return jnp.zeros_like(a)
    return jnp.concatenate([jnp.zeros_like(a[:1]), -lamscale * a[1:]], axis=0)


def _odd_force_radial_updates(*, armn_o, azmn_o, brmn_o, bzmn_o, lu_o, pzu_0, pru_0, bsqr_s, lv_es):
    """Build odd-parity radial updates without repeated whole-array .at copies."""
    if armn_o.shape[0] == 0:
        return armn_o, azmn_o, brmn_o, bzmn_o, lu_o

    armn_tail = -armn_o[-1:] - pzu_0[-1:] * bsqr_s[-1:] + 0.5 * lv_es[-1:]
    azmn_tail = -azmn_o[-1:] + pru_0[-1:] * bsqr_s[-1:]
    brmn_tail = 0.5 * brmn_o[-1:]
    bzmn_tail = 0.5 * bzmn_o[-1:]

    if armn_o.shape[0] < 2:
        return armn_tail, azmn_tail, brmn_tail, bzmn_tail, lu_o

    armn_body = armn_o[1:] - armn_o[:-1] - pzu_0[:-1] * bsqr_s[:-1] + 0.5 * (lv_es[:-1] + lv_es[1:])
    azmn_body = azmn_o[1:] - azmn_o[:-1] + pru_0[:-1] * bsqr_s[:-1]
    brmn_body = 0.5 * (brmn_o[:-1] + brmn_o[1:])
    bzmn_body = 0.5 * (bzmn_o[:-1] + bzmn_o[1:])
    lu_body = lu_o[:-1] + lu_o[1:]

    return (
        jnp.concatenate([armn_body, armn_tail], axis=0),
        jnp.concatenate([azmn_body, azmn_tail], axis=0),
        jnp.concatenate([brmn_body, brmn_tail], axis=0),
        jnp.concatenate([bzmn_body, bzmn_tail], axis=0),
        jnp.concatenate([lu_body, lu_o[-1:]], axis=0),
    )


def _assemble_vmec_rz_radial_forces(
    *,
    s,
    ohs,
    dshalfds,
    pshalf,
    psqrts,
    phips,
    lu_e,
    lv_e,
    guu,
    guv,
    gvv,
    jac,
    pr1_0,
    pr1_1,
    pz1_1,
    pru_0,
    pru_1,
    pzu_0,
    pzu_1,
    prv_0,
    prv_1,
    pzv_0,
    pzv_1,
    Lv1,
    lthreed: bool,
) -> tuple[Any, ...]:
    """Apply VMEC radial staggering to R/Z force kernels."""

    guus = guu * pshalf
    guvs = guv * pshalf
    gvvs = gvv * pshalf

    armn_e = ohs * jac.zu12 * lu_e
    azmn_e = -ohs * jac.ru12 * lu_e
    brmn_e = jac.zs * lu_e
    bzmn_e = -jac.rs * lu_e
    bsqr = dshalfds * lu_e / jnp.where(pshalf != 0, pshalf, 1.0)

    armn_o = armn_e * pshalf
    azmn_o = azmn_e * pshalf
    brmn_o = brmn_e * pshalf
    bzmn_o = bzmn_e * pshalf

    guu_i = _avg_forward_half_to_int(guu)
    gvv_i = _avg_forward_half_to_int(gvv)
    guus_i = _avg_forward_half_to_int(guus)
    gvvs_i = _avg_forward_half_to_int(gvvs)
    guv_i = _avg_forward_half_to_int(guv)
    guvs_i = _avg_forward_half_to_int(guvs)
    bsqr_s = _sum_forward_half(bsqr)

    armn_e = _diff_forward_half(armn_e, lv_e)
    azmn_e = _diff_forward_half_noavg(azmn_e)
    brmn_e = _avg_forward_half(brmn_e)
    bzmn_e = _avg_forward_half(bzmn_e)

    armn_e = armn_e - (gvvs_i * pr1_1 + gvv_i * pr1_0)
    brmn_e = brmn_e + bsqr_s * pz1_1 - (guus_i * pru_1 + guu_i * pru_0)
    bzmn_e = bzmn_e - (bsqr_s * pr1_1 + guus_i * pzu_1 + guu_i * pzu_0)

    lv_es = lv_e * pshalf
    lu_o = dshalfds * lu_e
    armn_o, azmn_o, brmn_o, bzmn_o, lu_o = _odd_force_radial_updates(
        armn_o=armn_o,
        azmn_o=azmn_o,
        brmn_o=brmn_o,
        bzmn_o=bzmn_o,
        lu_o=lu_o,
        pzu_0=pzu_0,
        pru_0=pru_0,
        bsqr_s=bsqr_s,
        lv_es=lv_es,
    )

    ss = (psqrts * psqrts).astype(guu_i.dtype)
    guu_s = guu_i * ss
    gvv_s = gvv_i * ss
    armn_o = armn_o - (pzu_1 * lu_o + gvv_s * pr1_1 + gvvs_i * pr1_0)
    azmn_o = azmn_o + pru_1 * lu_o
    brmn_o = brmn_o + pz1_1 * lu_o - (guu_s * pru_1 + guus_i * pru_0)
    bzmn_o = bzmn_o - (pr1_1 * lu_o + guu_s * pzu_1 + guus_i * pzu_0)

    if bool(lthreed):
        brmn_e = brmn_e - (guv_i * prv_0 + guvs_i * prv_1)
        bzmn_e = bzmn_e - (guv_i * pzv_0 + guvs_i * pzv_1)
        crmn_e = guv_i * pru_0 + gvv_i * prv_0 + gvvs_i * prv_1 + guvs_i * pru_1
        czmn_e = guv_i * pzu_0 + gvv_i * pzv_0 + gvvs_i * pzv_1 + guvs_i * pzu_1
        guv_s = guv_i * ss
        brmn_o = brmn_o - (guvs_i * prv_0 + guv_s * prv_1)
        bzmn_o = bzmn_o - (guvs_i * pzv_0 + guv_s * pzv_1)
        crmn_o = guvs_i * pru_0 + gvvs_i * prv_0 + gvv_s * prv_1 + guv_s * pru_1
        czmn_o = guvs_i * pzu_0 + gvvs_i * pzv_0 + gvv_s * pzv_1 + guv_s * pzu_1
    else:
        lamscale = jnp.asarray(lamscale_from_phips(phips, s))
        if lamscale.ndim == 0:
            lamscale = jnp.full_like(s, lamscale)
        lamscale = lamscale[:, None, None]
        crmn_e = jnp.asarray(lv_es)
        czmn_e = jnp.asarray(lu_e)
        crmn_o = -lamscale * jnp.asarray(Lv1)
        czmn_o = jnp.asarray(lu_o)

    return (
        armn_e,
        armn_o,
        brmn_e,
        brmn_o,
        crmn_e,
        crmn_o,
        azmn_e,
        azmn_o,
        bzmn_e,
        bzmn_o,
        czmn_e,
        czmn_o,
    )


def vmec_forces_rz_from_wout(
    *,
    state,
    static,
    wout,
    indata=None,
    constraint_tcon0: float | None = None,
    constraint_tcon: Any | None = None,
    constraint_precond_diag: tuple[Any, Any] | None = None,
    constraint_precond_active: Any | None = None,
    constraint_tcon_active: Any | None = None,
    constraint_rcon0: Any | None = None,
    constraint_zcon0: Any | None = None,
    freeb_bsqvac_half: Any | None = None,
    freeb_pres_scale: Any | None = None,
    use_wout_bsup: bool = False,
    use_vmec_synthesis: bool = False,
    trig: VmecTrigTables | None = None,
    iter_idx: int | None = None,
) -> VmecRZForceKernels:
    """Compute VMEC R/Z force kernels (armn/brmn/...) from a `wout` equilibrium.

    Parameters
    ----------
    use_wout_bsup:
        If True, use the Nyquist `bsup*` fields stored in the `wout` file when
        forming the B-product tensors inside `bcovar`. This isolates the forces
        algebra from small differences in the derived contravariant field. In
        this parity mode, lambda-force kernels (`blmn/clmn`) are also formed
        from averaged `wout` `bsub*` fields.
    freeb_bsqvac_half:
        Optional free-boundary vacuum ``0.5*|B|^2`` proxy. This may be either
        the full half-mesh field ``(ns, ntheta, nzeta)`` or just the edge
        slice ``(ntheta, nzeta)``. In both cases only the edge slice is used
        to override the edge constraint-pressure channel ``gcon`` (VMEC
        funct3d-style coupling) while keeping `bcovar` unchanged.
    freeb_pres_scale:
        Optional VMEC free-boundary pressure scale ``pmass(1)/pmass(s_edge)``
        used in ``presf_ns``. When provided, this is multiplied by the current
        edge pressure ``pres(ns)`` before forming ``gcon``.
    """
    force_start = time.perf_counter()
    s = jnp.asarray(static.s)
    ohs = jnp.asarray(1.0 / (s[1] - s[0])) if s.shape[0] >= 2 else jnp.asarray(0.0)
    dshalfds = jnp.asarray(0.25, dtype=s.dtype)

    # Solver diagnostics may start from an input deck plus a lightweight WOUT
    # shell. Normalize that case once before the force and bcovar paths.
    wout_eff, pres_half = _resolve_force_wout_and_pressure(wout=wout, indata=indata, s=s)

    phips = wout_eff.phips
    # This conversion is needed both by bcovar and by the constraint kernels.
    # Compute it once per force evaluation and pass it through to bcovar.
    geom_start = time.perf_counter()
    Rcos_int, Zsin_int, Rsin_int, Zcos_int = vmec_m1_internal_to_physical_signed(
        Rcos=state.Rcos,
        Zsin=state.Zsin,
        Rsin=state.Rsin,
        Zcos=state.Zcos,
        modes=static.modes,
        lthreed=bool(getattr(static.cfg, "lthreed", True)),
        lasym=bool(getattr(static.cfg, "lasym", False)),
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
    )
    bcovar_start = time.perf_counter()
    with _trace("bcovar"):
        bcovar_kwargs = dict(
            state=state,
            static=static,
            wout=wout_eff,
            pres=pres_half,
            freeb_bsqvac_edge=None,
            use_wout_bsup=use_wout_bsup,
            use_wout_bsub_for_lambda=use_wout_bsup,
            use_wout_bmag_for_bsq=use_wout_bsup,
            use_vmec_synthesis=use_vmec_synthesis,
            trig=trig,
            return_parity_aux=True,
            state_physical_signed=(Rcos_int, Zsin_int, Rsin_int, Zcos_int),
        )
        bcovar_result = vmec_bcovar_half_mesh_from_wout(**bcovar_kwargs)
        if _looks_like_force_kernel_payload(bcovar_result) or not _bcovar_matches_radial_grid(bcovar_result, s):
            bcovar_result = _production_bcovar_half_mesh_from_wout(expected_s=s, **bcovar_kwargs)
        bc, bc_parity = _bcovar_with_parity_aux(bcovar_result)
    _vmec_force_profile_log("bcovar_done", bcovar_start)

    # VMEC stores internal coefficients; undo the m=1 internal constraint for
    # R/Z before real-space synthesis.
    state_geom = SimpleNamespace(
        Rcos=jnp.asarray(Rcos_int),
        Rsin=jnp.asarray(Rsin_int),
        Zcos=jnp.asarray(Zcos_int),
        Zsin=jnp.asarray(Zsin_int),
        Lcos=jnp.asarray(state.Lcos),
        Lsin=jnp.asarray(state.Lsin),
    )

    pr1_0 = jnp.asarray(bc_parity.pr1_even)
    pr1_1 = jnp.asarray(bc_parity.pr1_odd)
    pz1_0 = jnp.asarray(bc_parity.pz1_even)
    pz1_1 = jnp.asarray(bc_parity.pz1_odd)
    pru_0 = jnp.asarray(bc_parity.pru_even)
    pru_1 = jnp.asarray(bc_parity.pru_odd)
    pzu_0 = jnp.asarray(bc_parity.pzu_even)
    pzu_1 = jnp.asarray(bc_parity.pzu_odd)
    prv_0 = jnp.asarray(bc_parity.prv_even)
    prv_1 = jnp.asarray(bc_parity.prv_odd)
    pzv_0 = jnp.asarray(bc_parity.pzv_even)
    pzv_1 = jnp.asarray(bc_parity.pzv_odd)
    Lu1 = jnp.asarray(bc_parity.lu_odd)
    Lv1 = jnp.asarray(bc_parity.lv_odd)

    _vmec_force_profile_log("geometry_done", geom_start)

    # Half-mesh sqrt(s) and full-mesh sqrt(s).
    pshalf = _pshalf_from_s(s)[:, None, None]
    psqrts = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]

    # Inputs `forces.f` expects after `bcovar` (half mesh).
    #
    # Important: by the time `forces.f` runs, VMEC has overwritten `guu/guv/gvv`
    # with the B-product tensors:
    #   GIJ = (B^i B^j) * sqrt(g)   for i,j ∈ {u,v}
    # (see `bcovar.f` "STORE LU * LV COMBINATIONS USED IN FORCES").
    lu_e = _with_axis_zero(bc.lu_e)
    lv_e = _with_axis_zero(bc.lv_e)
    guu = _with_axis_zero(bc.gij_b_uu)
    guv = _with_axis_zero(bc.gij_b_uv)
    gvv = _with_axis_zero(bc.gij_b_vv)

    assembly_start = time.perf_counter()
    lthreed = bool(np.any(np.asarray(static.modes.n) != 0))
    radial = _assemble_vmec_rz_radial_forces(
        s=s,
        ohs=ohs,
        dshalfds=dshalfds,
        pshalf=pshalf,
        psqrts=psqrts,
        phips=phips,
        lu_e=lu_e,
        lv_e=lv_e,
        guu=guu,
        guv=guv,
        gvv=gvv,
        jac=bc.jac,
        pr1_0=pr1_0,
        pr1_1=pr1_1,
        pz1_1=pz1_1,
        pru_0=pru_0,
        pru_1=pru_1,
        pzu_0=pzu_0,
        pzu_1=pzu_1,
        prv_0=prv_0,
        prv_1=prv_1,
        pzv_0=pzv_0,
        pzv_1=pzv_1,
        Lv1=Lv1,
        lthreed=bool(lthreed),
    )
    (
        armn_e, armn_o, brmn_e, brmn_o, crmn_e, crmn_o,
        azmn_e, azmn_o, bzmn_e, bzmn_o, czmn_e, czmn_o,
    ) = radial

    _vmec_force_profile_log("assembly_done", assembly_start)

    armn_e, armn_o, azmn_e, azmn_o = _apply_freeb_edge_forcing(locals())

    # ---------------------------------------------------------------------
    # Constraint force pipeline: compute gcon from ztemp via alias and apply
    # the constraint force kernels to B-terms (forces.f "CONSTRAINT FORCE").
    # ---------------------------------------------------------------------
    # VMEC default: `tcon0 = 1` (see `readin.f`).
    # If caller passed an explicit value, do not override it from `indata`.
    if indata is not None and constraint_tcon0 is None:
        constraint_tcon0 = float(indata.get_float("TCON0", 1.0))
    constraint_start = time.perf_counter()
    con = _constraint_kernels_from_state(
        state=state_geom,
        static=static,
        wout=wout,
        bc=bc,
        pru_0=pru_0,
        pru_1=pru_1,
        pzu_0=pzu_0,
        pzu_1=pzu_1,
        constraint_tcon0=constraint_tcon0,
        tcon_override=constraint_tcon,
        precond_diag_override=constraint_precond_diag,
        precond_active=constraint_precond_active,
        tcon_active=constraint_tcon_active,
        rcon0_override=constraint_rcon0,
        zcon0_override=constraint_zcon0,
        trig=trig,
        iter_idx=iter_idx,
    )
    _vmec_force_profile_log("constraint_done", constraint_start)

    brmn_e = brmn_e + con.rcon_force
    bzmn_e = bzmn_e + con.zcon_force
    brmn_o = brmn_o + con.rcon_force * psqrts
    bzmn_o = bzmn_o + con.zcon_force * psqrts

    arcon_e = con.arcon_e
    arcon_o = con.arcon_o
    azcon_e = con.azcon_e
    azcon_o = con.azcon_o
    gcon = con.gcon

    result = VmecRZForceKernels(
        armn_e=armn_e,
        armn_o=armn_o,
        brmn_e=brmn_e,
        brmn_o=brmn_o,
        crmn_e=crmn_e,
        crmn_o=crmn_o,
        azmn_e=azmn_e,
        azmn_o=azmn_o,
        bzmn_e=bzmn_e,
        bzmn_o=bzmn_o,
        czmn_e=czmn_e,
        czmn_o=czmn_o,
        bc=bc,
        arcon_e=arcon_e,
        arcon_o=arcon_o,
        azcon_e=azcon_e,
        azcon_o=azcon_o,
        gcon=gcon,
        tcon=con.tcon,
        pr1_even=pr1_0,
        pr1_odd=pr1_1,
        pz1_even=pz1_0,
        pz1_odd=pz1_1,
        pru_even=pru_0,
        pru_odd=pru_1,
        pzu_even=pzu_0,
        pzu_odd=pzu_1,
        prv_even=prv_0,
        prv_odd=prv_1,
        pzv_even=pzv_0,
        pzv_odd=pzv_1,
        constraint_rcon0=con.rcon0,
        constraint_zcon0=con.zcon0,
    )
    _vmec_force_profile_log("force_done", force_start)
    return result


def vmec_forces_rz_from_wout_reference_fields(
    *,
    state,
    static,
    wout,
    indata=None,
    constraint_tcon0: float | None = None,
) -> VmecRZForceKernels:
    """Compute VMEC R/Z force kernels using `wout`'s stored (sqrtg, bsup, ``|B|``).

    This is a parity/debug variant that reduces the number of derived quantities
    computed by vmec_jax, making it easier to validate the *forces* algebra in
    isolation. If `constraint_tcon0` (or `indata.TCON0`) is provided, the VMEC
    constraint-force pipeline is also applied.
    """
    s = jnp.asarray(static.s)
    ohs = jnp.asarray(1.0 / (s[1] - s[0])) if s.shape[0] >= 2 else jnp.asarray(0.0)
    dshalfds = jnp.asarray(0.25, dtype=s.dtype)

    # Geometry parity arrays.
    parity = split_rzl_even_odd_m(state, static.basis, static.modes.m)

    m_modes = np.asarray(static.modes.m, dtype=int)
    dtype = jnp.asarray(state.Rcos).dtype
    mask_m1 = jnp.asarray(m_modes == 1, dtype=dtype)
    mask_odd_rest = jnp.asarray((m_modes % 2 == 1) & (m_modes != 1), dtype=dtype)

    def _odd_internal_vmec(*, coeff_cos, coeff_sin, eval_fn):
        phys_m1 = eval_fn(coeff_cos * mask_m1, coeff_sin * mask_m1, static.basis, coeffs_internal=True)
        phys_rest = eval_fn(coeff_cos * mask_odd_rest, coeff_sin * mask_odd_rest, static.basis, coeffs_internal=True)
        return internal_odd_from_physical_vmec_m1(odd_m1_phys=phys_m1, odd_mge2_phys=phys_rest, s=s)

    def _odd_internal_vmec_lambda(*, coeff_cos, coeff_sin, eval_fn):
        phys_m1 = eval_fn(coeff_cos * mask_m1, coeff_sin * mask_m1, static.basis, coeffs_internal=True)
        phys_rest = eval_fn(coeff_cos * mask_odd_rest, coeff_sin * mask_odd_rest, static.basis, coeffs_internal=True)
        return internal_odd_from_physical_vmec_jlam(
            odd_m1_phys=phys_m1,
            odd_mge2_phys=phys_rest,
            s=s,
        )

    R1 = _odd_internal_vmec(coeff_cos=state.Rcos, coeff_sin=state.Rsin, eval_fn=eval_fourier)
    Z1 = _odd_internal_vmec(coeff_cos=state.Zcos, coeff_sin=state.Zsin, eval_fn=eval_fourier)
    Ru1 = _odd_internal_vmec(coeff_cos=state.Rcos, coeff_sin=state.Rsin, eval_fn=eval_fourier_dtheta)
    Zu1 = _odd_internal_vmec(coeff_cos=state.Zcos, coeff_sin=state.Zsin, eval_fn=eval_fourier_dtheta)
    Rv1 = _odd_internal_vmec(coeff_cos=state.Rcos, coeff_sin=state.Rsin, eval_fn=eval_fourier_dzeta_phys)
    Zv1 = _odd_internal_vmec(coeff_cos=state.Zcos, coeff_sin=state.Zsin, eval_fn=eval_fourier_dzeta_phys)
    Lv1 = _odd_internal_vmec_lambda(coeff_cos=state.Lcos, coeff_sin=state.Lsin, eval_fn=eval_fourier_dzeta_phys)

    pr1_0, pr1_1 = jnp.asarray(parity.R_even), jnp.asarray(R1)
    pz1_0, pz1_1 = jnp.asarray(parity.Z_even), jnp.asarray(Z1)
    pru_0, pru_1 = jnp.asarray(parity.Rt_even), jnp.asarray(Ru1)
    pzu_0, pzu_1 = jnp.asarray(parity.Zt_even), jnp.asarray(Zu1)
    prv_0, prv_1 = jnp.asarray(parity.Rp_even), jnp.asarray(Rv1)
    pzv_0, pzv_1 = jnp.asarray(parity.Zp_even), jnp.asarray(Zv1)

    # Half-mesh Jacobian-like quantities (r12/rs/zs/ru12/zu12) from our parity kernel.
    from .vmec_jacobian import jacobian_half_mesh_from_parity

    jac = jacobian_half_mesh_from_parity(
        pr1_even=pr1_0,
        pr1_odd=pr1_1,
        pz1_even=pz1_0,
        pz1_odd=pz1_1,
        pru_even=pru_0,
        pru_odd=pru_1,
        pzu_even=pzu_0,
        pzu_odd=pzu_1,
        s=s,
    )

    # Evaluate stored wout Nyquist fields on our angular grid.
    if int(getattr(static.grid, "nfp", 0)) == int(wout.nfp):
        grid = static.grid
    else:
        grid = AngleGrid(theta=np.asarray(static.grid.theta), zeta=np.asarray(static.grid.zeta), nfp=int(wout.nfp))
    basis_nyq = nyquist_basis_from_wout(wout=wout, grid=grid)

    sqrtg = jnp.asarray(eval_fourier(wout.gmnc, wout.gmns, basis_nyq))
    bsupu = jnp.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv = jnp.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
    bsubu = jnp.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
    bsubv = jnp.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))
    bmag = jnp.asarray(eval_fourier(wout.bmnc, wout.bmns, basis_nyq))
    phips = wout.phips

    # bsq = |B|^2/2 + p (half mesh).
    pres_h = jnp.asarray(wout.pres)[:, None, None]
    bsq = 0.5 * (bmag * bmag) + pres_h

    # Use sqrtg from wout to define tau; r12 from our parity half-mesh construction.
    r12 = jnp.asarray(jac.r12)
    tau = jnp.where(r12 != 0, sqrtg / r12, 0.0)

    lu_e = _with_axis_zero(bsq * r12)
    lv_e = _with_axis_zero(bsq * tau)

    # Metric elements on the half mesh (bcovar.f convention). We keep these for
    # scaling diagnostics, but the *forces* kernel below uses GIJ (B-products).
    def _half_mesh_from_even_odd(even, odd_int, *, s):
        even = jnp.asarray(even)
        odd_int = jnp.asarray(odd_int)
        s = jnp.asarray(s)
        ns_ = int(s.shape[0])
        if ns_ < 2:
            return even
        psh = _pshalf_from_s(s)[:, None, None]
        inner = 0.5 * (even[1:] + even[:-1] + psh[1:] * (odd_int[1:] + odd_int[:-1]))
        return jnp.concatenate([inner[:1], inner], axis=0)

    ss0 = s[:, None, None]
    guu_e = pru_0 * pru_0 + pzu_0 * pzu_0 + ss0 * (pru_1 * pru_1 + pzu_1 * pzu_1)
    guu_o = 2.0 * (pru_0 * pru_1 + pzu_0 * pzu_1)
    guv_e = prv_0 * pru_0 + pzv_0 * pzu_0 + ss0 * (prv_1 * pru_1 + pzv_1 * pzu_1)
    guv_o = prv_0 * pru_1 + prv_1 * pru_0 + pzv_0 * pzu_1 + pzv_1 * pzu_0
    gvv_e = prv_0 * prv_0 + pzv_0 * pzv_0 + ss0 * (prv_1 * prv_1 + pzv_1 * pzv_1)
    gvv_o = 2.0 * (prv_0 * prv_1 + pzv_0 * pzv_1)

    # Add R^2 term to gvv in cylindrical coordinates.
    r2_e = pr1_0 * pr1_0 + ss0 * (pr1_1 * pr1_1)
    r2_o = 2.0 * (pr1_0 * pr1_1)

    guu_metric = _with_axis_zero(_half_mesh_from_even_odd(guu_e, guu_o, s=s))
    guv_metric = _with_axis_zero(_half_mesh_from_even_odd(guv_e, guv_o, s=s))
    gvv_metric = _with_axis_zero(
        _half_mesh_from_even_odd(gvv_e, gvv_o, s=s) + _half_mesh_from_even_odd(r2_e, r2_o, s=s)
    )

    # GIJ = (B^i B^j)*sqrt(g) used in forces.f.
    guu = _with_axis_zero((bsupu * bsupu) * sqrtg)
    guv = _with_axis_zero((bsupu * bsupv) * sqrtg)
    gvv = _with_axis_zero((bsupv * bsupv) * sqrtg)

    # Half-mesh sqrt(s) and full-mesh sqrt(s).
    pshalf = _pshalf_from_s(s)[:, None, None]
    psqrts = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]

    lthreed = bool(np.any(np.asarray(static.modes.n) != 0))
    radial = _assemble_vmec_rz_radial_forces(
        s=s,
        ohs=ohs,
        dshalfds=dshalfds,
        pshalf=pshalf,
        psqrts=psqrts,
        phips=phips,
        lu_e=lu_e,
        lv_e=lv_e,
        guu=guu,
        guv=guv,
        gvv=gvv,
        jac=jac,
        pr1_0=pr1_0,
        pr1_1=pr1_1,
        pz1_1=pz1_1,
        pru_0=pru_0,
        pru_1=pru_1,
        pzu_0=pzu_0,
        pzu_1=pzu_1,
        prv_0=prv_0,
        prv_1=prv_1,
        pzv_0=pzv_0,
        pzv_1=pzv_1,
        Lv1=Lv1,
        lthreed=bool(lthreed),
    )
    (
        armn_e, armn_o, brmn_e, brmn_o, crmn_e, crmn_o,
        azmn_e, azmn_o, bzmn_e, bzmn_o, czmn_e, czmn_o,
    ) = radial

    # Build lambda-force kernels (blmn/clmn) using the VMEC formulas but with
    # reference-field inputs.
    lamscale = lamscale_from_phips(phips, s)

    # For reference-field parity we form the lambda-force kernels from the
    # stored wout bsubu/bsubv fields by averaging to the full mesh. This avoids
    # re-deriving bsubv_e from lambda derivatives, which can amplify small
    # discrepancies in the reference path.
    bsubu_e = _avg_forward_half_to_int_or_zero(bsubu)
    bsubv_e = _avg_forward_half_to_int_or_zero(bsubv)

    # Scale for tomnsps (skip axis surface).
    clmn_even = _scale_lambda_full_mesh_zero_axis(bsubu_e, lamscale)
    blmn_even = _scale_lambda_full_mesh_zero_axis(bsubv_e, lamscale)
    clmn_odd = psqrts * clmn_even
    blmn_odd = psqrts * blmn_even

    # `bc` object is used only for downstream scaling helpers; provide the pieces we need.
    class _BC:
        pass

    bc_obj = _BC()
    from .vmec_jacobian import VmecHalfMeshJacobian

    bc_obj.jac = VmecHalfMeshJacobian(
        r12=jac.r12,
        rs=jac.rs,
        zs=jac.zs,
        ru12=jac.ru12,
        zu12=jac.zu12,
        tau=tau,
        sqrtg=sqrtg,
    )
    bc_obj.guu = guu_metric
    bc_obj.guv = guv_metric
    bc_obj.gvv = gvv_metric
    bc_obj.bsubu = bsubu
    bc_obj.bsubv = bsubv
    bc_obj.lamscale = lamscale
    bc_obj.bsq = bsq
    bc_obj.clmn_even = clmn_even
    bc_obj.clmn_odd = clmn_odd
    bc_obj.blmn_even = blmn_even
    bc_obj.blmn_odd = blmn_odd
    bc_obj.bsubu_e = bsubu_e
    bc_obj.bsubv_e = bsubv_e
    bc_obj.bsubu_e_scaled = clmn_even
    bc_obj.bsubv_e_scaled = blmn_even

    # VMEC default: `tcon0 = 1` (see `readin.f`).
    # If caller passed an explicit value, do not override it from `indata`.
    if indata is not None and constraint_tcon0 is None:
        constraint_tcon0 = float(indata.get_float("TCON0", 1.0))
    con = _constraint_kernels_from_state(
        state=state,
        static=static,
        wout=wout,
        bc=bc_obj,
        pru_0=pru_0,
        pru_1=pru_1,
        pzu_0=pzu_0,
        pzu_1=pzu_1,
        constraint_tcon0=constraint_tcon0,
        tcon_override=None,
    )

    brmn_e = brmn_e + con.rcon_force
    bzmn_e = bzmn_e + con.zcon_force
    brmn_o = brmn_o + con.rcon_force * psqrts
    bzmn_o = bzmn_o + con.zcon_force * psqrts

    return VmecRZForceKernels(
        armn_e=armn_e,
        armn_o=armn_o,
        brmn_e=brmn_e,
        brmn_o=brmn_o,
        crmn_e=crmn_e,
        crmn_o=crmn_o,
        azmn_e=azmn_e,
        azmn_o=azmn_o,
        bzmn_e=bzmn_e,
        bzmn_o=bzmn_o,
        czmn_e=czmn_e,
        czmn_o=czmn_o,
        bc=bc_obj,
        arcon_e=con.arcon_e,
        arcon_o=con.arcon_o,
        azcon_e=con.azcon_e,
        azcon_o=con.azcon_o,
        gcon=con.gcon,
        tcon=con.tcon,
        pr1_even=pr1_0,
        pr1_odd=pr1_1,
        pz1_even=pz1_0,
        pz1_odd=pz1_1,
        pru_even=pru_0,
        pru_odd=pru_1,
        pzu_even=pzu_0,
        pzu_odd=pzu_1,
        prv_even=prv_0,
        prv_odd=prv_1,
        pzv_even=pzv_0,
        pzv_odd=pzv_1,
    )


@dataclass(frozen=True)
class VmecRZResidualCoeffs:
    gcr_cos: Any  # (ns, K)
    gcr_sin: Any  # (ns, K)
    gcz_cos: Any  # (ns, K)
    gcz_sin: Any  # (ns, K)


def _select_parity_coeffs(*, coeff_even, coeff_odd, m):
    mask_even = (m % 2) == 0
    return jnp.where(mask_even[None, :], coeff_even, coeff_odd)


def rz_residual_coeffs_from_kernels(k: VmecRZForceKernels, *, static) -> VmecRZResidualCoeffs:
    """Compute Fourier-space residual coefficients gcr/gcz from force kernels.

    This mirrors VMEC's ``tomnsps`` combination:
        FR = A - dB/du + dC/dv
    using coefficient-space differentiation (no finite differences).
    """
    m = jnp.asarray(static.modes.m, dtype=jnp.asarray(k.armn_e).dtype)
    n = jnp.asarray(static.modes.n, dtype=m.dtype)
    n_phys = n * int(static.grid.nfp)

    # Project each parity field to helical coefficients.
    aR_e_c, aR_e_s = project_to_modes(k.armn_e, static.basis)
    aR_o_c, aR_o_s = project_to_modes(k.armn_o, static.basis)
    bR_e_c, bR_e_s = project_to_modes(k.brmn_e, static.basis)
    bR_o_c, bR_o_s = project_to_modes(k.brmn_o, static.basis)
    cR_e_c, cR_e_s = project_to_modes(k.crmn_e, static.basis)
    cR_o_c, cR_o_s = project_to_modes(k.crmn_o, static.basis)

    aZ_e_c, aZ_e_s = project_to_modes(k.azmn_e, static.basis)
    aZ_o_c, aZ_o_s = project_to_modes(k.azmn_o, static.basis)
    bZ_e_c, bZ_e_s = project_to_modes(k.bzmn_e, static.basis)
    bZ_o_c, bZ_o_s = project_to_modes(k.bzmn_o, static.basis)
    cZ_e_c, cZ_e_s = project_to_modes(k.czmn_e, static.basis)
    cZ_o_c, cZ_o_s = project_to_modes(k.czmn_o, static.basis)

    aR_c = _select_parity_coeffs(coeff_even=aR_e_c, coeff_odd=aR_o_c, m=m)
    aR_s = _select_parity_coeffs(coeff_even=aR_e_s, coeff_odd=aR_o_s, m=m)
    bR_c = _select_parity_coeffs(coeff_even=bR_e_c, coeff_odd=bR_o_c, m=m)
    bR_s = _select_parity_coeffs(coeff_even=bR_e_s, coeff_odd=bR_o_s, m=m)
    cR_c = _select_parity_coeffs(coeff_even=cR_e_c, coeff_odd=cR_o_c, m=m)
    cR_s = _select_parity_coeffs(coeff_even=cR_e_s, coeff_odd=cR_o_s, m=m)

    aZ_c = _select_parity_coeffs(coeff_even=aZ_e_c, coeff_odd=aZ_o_c, m=m)
    aZ_s = _select_parity_coeffs(coeff_even=aZ_e_s, coeff_odd=aZ_o_s, m=m)
    bZ_c = _select_parity_coeffs(coeff_even=bZ_e_c, coeff_odd=bZ_o_c, m=m)
    bZ_s = _select_parity_coeffs(coeff_even=bZ_e_s, coeff_odd=bZ_o_s, m=m)
    cZ_c = _select_parity_coeffs(coeff_even=cZ_e_c, coeff_odd=cZ_o_c, m=m)
    cZ_s = _select_parity_coeffs(coeff_even=cZ_e_s, coeff_odd=cZ_o_s, m=m)

    # Derivatives in coefficient space.
    dBdu_R_c = m[None, :] * bR_s
    dBdu_R_s = -m[None, :] * bR_c
    dCdv_R_c = -(n_phys[None, :]) * cR_s
    dCdv_R_s = (n_phys[None, :]) * cR_c

    dBdu_Z_c = m[None, :] * bZ_s
    dBdu_Z_s = -m[None, :] * bZ_c
    dCdv_Z_c = -(n_phys[None, :]) * cZ_s
    dCdv_Z_s = (n_phys[None, :]) * cZ_c

    gcr_cos = aR_c - dBdu_R_c + dCdv_R_c
    gcr_sin = aR_s - dBdu_R_s + dCdv_R_s
    gcz_cos = aZ_c - dBdu_Z_c + dCdv_Z_c
    gcz_sin = aZ_s - dBdu_Z_s + dCdv_Z_s

    return VmecRZResidualCoeffs(gcr_cos=gcr_cos, gcr_sin=gcr_sin, gcz_cos=gcz_cos, gcz_sin=gcz_sin)


@dataclass(frozen=True)
class VmecInternalResidualRZL:
    """Internal VMEC-style residual arrays produced by `tomnsps` (+ `tomnspa` when `lasym=True`)."""

    frcc: Any
    frss: Any | None
    fzsc: Any
    fzcs: Any | None
    flsc: Any
    flcs: Any | None

    # Asymmetric components from `tomnspa` (lasym=True only).
    frsc: Any | None = None
    frcs: Any | None = None
    fzcc: Any | None = None
    fzss: Any | None = None
    flcc: Any | None = None
    flss: Any | None = None


def vmec_residual_internal_from_kernels(
    k: VmecRZForceKernels,
    *,
    cfg_ntheta: int,
    cfg_nzeta: int,
    wout,
    trig: VmecTrigTables | None = None,
    apply_lforbal: bool = False,
    include_edge: bool = False,
    masks: TomnspsMasks | None = None,
) -> VmecInternalResidualRZL:
    """Compute internal residual coefficient arrays using VMEC's `tomnsps` conventions."""
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(cfg_ntheta),
            nzeta=int(cfg_nzeta),
            nfp=int(wout.nfp),
            mmax=int(wout.mpol) - 1,
            nmax=int(wout.ntor),
            lasym=bool(wout.lasym),
        )

    # Lambda kernels are optional for early parity work.
    z = jnp.zeros_like(k.armn_e)
    blmn_even = getattr(k.bc, "blmn_even", z)
    blmn_odd = getattr(k.bc, "blmn_odd", z)
    clmn_even = getattr(k.bc, "clmn_even", z)
    clmn_odd = getattr(k.bc, "clmn_odd", z)

    lasym = bool(wout.lasym)

    if os.getenv("VMEC_JAX_SCAN_DEBUG_FORCE", "") not in ("", "0"):
        try:
            from jax import debug as _jax_debug  # type: ignore
        except Exception:
            _jax_debug = None  # type: ignore
        if _jax_debug is not None:
            azmn_e2 = jnp.sum(k.azmn_e * k.azmn_e)
            bzmn_e2 = jnp.sum(k.bzmn_e * k.bzmn_e)
            azmn_o2 = jnp.sum(k.azmn_o * k.azmn_o)
            bzmn_o2 = jnp.sum(k.bzmn_o * k.bzmn_o)
            _jax_debug.print(
                "[tomnsps-debug] azmn_e2={aze:.6e} bzmn_e2={bze:.6e} azmn_o2={azo:.6e} bzmn_o2={bzo:.6e}",
                aze=azmn_e2,
                bze=bzmn_e2,
                azo=azmn_o2,
                bzo=bzmn_o2,
            )

    def _symforce_split_one(
        a,
        *,
        trig: VmecTrigTables,
        kind: str,
    ):
        """Split a field into VMEC symmetric/antisymmetric parts for lasym transforms.

        VMEC's `tomnsps`/`tomnspa` always integrate on the restricted interval
        u∈[0,π] (i=1..ntheta2). For `lasym=True`, VMEC first decomposes each
        kernel into a "symmetric" piece (paired with cos(mu±nv)) and an
        "antisymmetric" piece (paired with sin(mu±nv)) using `symforce.f`.

        The mapping is not uniform across kernels (some have reversed dominant
        symmetry); see `VMEC2000/Sources/General/symforce.f`.
        """
        a = jnp.asarray(a)
        ns, ntheta3, nzeta = a.shape
        nt2 = int(trig.ntheta2)
        nt1 = int(trig.ntheta1)
        if int(trig.ntheta3) != int(ntheta3):
            raise ValueError("symforce: theta size mismatch")
        if nt2 <= 0 or nt2 > ntheta3:
            raise ValueError("symforce: invalid ntheta2")

        # Reflection map (0-based) for i=1..ntheta2:
        #   ir = ntheta1 + 2 - i, with i==1 -> ir=1   (Fortran, 1-based)
        i0 = jnp.arange(nt2, dtype=jnp.int32)
        ir0 = jnp.where(i0 == 0, 0, nt1 - i0)
        kk = (nzeta - jnp.arange(nzeta, dtype=jnp.int32)) % nzeta

        a_half = a[:, :nt2, :]
        a_ref = a[:, ir0, :][:, :, kk]

        if kind in ("ars", "bzs", "bls", "rcs", "czs", "cls"):
            a_sym_half = 0.5 * (a_half + a_ref)
            a_asym_half = 0.5 * (a_half - a_ref)
        elif kind in ("brs", "azs", "zcs", "crs"):
            # Reversed dominant symmetry (see `symforce.f`).
            a_sym_half = 0.5 * (a_half - a_ref)
            a_asym_half = 0.5 * (a_half + a_ref)
        else:  # pragma: no cover
            raise ValueError(f"symforce: unknown kind {kind!r}")

        # VMEC updates only i<=ntheta2; values for i>ntheta2 are retained on the
        # symmetric arrays and unused for the antisymmetric ones (tomnspa uses
        # the restricted interval). Preserve the original data to match VMEC.
        a_sym = jnp.concatenate([a_sym_half, a[:, nt2:, :]], axis=1)
        a_asym = jnp.concatenate([a_asym_half, jnp.zeros_like(a[:, nt2:, :])], axis=1)
        return a_sym, a_asym

    mask_pack = masks

    if lasym:
        # Decompose each kernel before calling tomnsps/tomnspa.
        armn_e_s, armn_e_a = _symforce_split_one(k.armn_e, trig=trig, kind="ars")
        armn_o_s, armn_o_a = _symforce_split_one(k.armn_o, trig=trig, kind="ars")
        brmn_e_s, brmn_e_a = _symforce_split_one(k.brmn_e, trig=trig, kind="brs")
        brmn_o_s, brmn_o_a = _symforce_split_one(k.brmn_o, trig=trig, kind="brs")
        crmn_e_s, crmn_e_a = _symforce_split_one(k.crmn_e, trig=trig, kind="crs")
        crmn_o_s, crmn_o_a = _symforce_split_one(k.crmn_o, trig=trig, kind="crs")

        azmn_e_s, azmn_e_a = _symforce_split_one(k.azmn_e, trig=trig, kind="azs")
        azmn_o_s, azmn_o_a = _symforce_split_one(k.azmn_o, trig=trig, kind="azs")
        bzmn_e_s, bzmn_e_a = _symforce_split_one(k.bzmn_e, trig=trig, kind="bzs")
        bzmn_o_s, bzmn_o_a = _symforce_split_one(k.bzmn_o, trig=trig, kind="bzs")
        czmn_e_s, czmn_e_a = _symforce_split_one(k.czmn_e, trig=trig, kind="czs")
        czmn_o_s, czmn_o_a = _symforce_split_one(k.czmn_o, trig=trig, kind="czs")

        blmn_e_s, blmn_e_a = _symforce_split_one(blmn_even, trig=trig, kind="bls")
        blmn_o_s, blmn_o_a = _symforce_split_one(blmn_odd, trig=trig, kind="bls")
        clmn_e_s, clmn_e_a = _symforce_split_one(clmn_even, trig=trig, kind="cls")
        clmn_o_s, clmn_o_a = _symforce_split_one(clmn_odd, trig=trig, kind="cls")

        arcon_e_s, arcon_e_a = _symforce_split_one(k.arcon_e, trig=trig, kind="rcs")
        arcon_o_s, arcon_o_a = _symforce_split_one(k.arcon_o, trig=trig, kind="rcs")
        azcon_e_s, azcon_e_a = _symforce_split_one(k.azcon_e, trig=trig, kind="zcs")
        azcon_o_s, azcon_o_a = _symforce_split_one(k.azcon_o, trig=trig, kind="zcs")

        with _trace("tomnsps_rzl"), _named_scope("tomnsps_rzl"):
            out_sym = tomnsps_rzl(
                armn_even=armn_e_s,
                armn_odd=armn_o_s,
                brmn_even=brmn_e_s,
                brmn_odd=brmn_o_s,
                crmn_even=crmn_e_s,
                crmn_odd=crmn_o_s,
                azmn_even=azmn_e_s,
                azmn_odd=azmn_o_s,
                bzmn_even=bzmn_e_s,
                bzmn_odd=bzmn_o_s,
                czmn_even=czmn_e_s,
                czmn_odd=czmn_o_s,
                blmn_even=blmn_e_s,
                blmn_odd=blmn_o_s,
                clmn_even=clmn_e_s,
                clmn_odd=clmn_o_s,
                arcon_even=arcon_e_s,
                arcon_odd=arcon_o_s,
                azcon_even=azcon_e_s,
                azcon_odd=azcon_o_s,
                mpol=int(wout.mpol),
                ntor=int(wout.ntor),
                nfp=int(wout.nfp),
                lasym=True,
                trig=trig,
                include_edge=bool(include_edge),
                masks=mask_pack,
            )

        with _trace("tomnspa_rzl"), _named_scope("tomnspa_rzl"):
            out_asym = tomnspa_rzl(
                armn_even=armn_e_a,
                armn_odd=armn_o_a,
                brmn_even=brmn_e_a,
                brmn_odd=brmn_o_a,
                crmn_even=crmn_e_a,
                crmn_odd=crmn_o_a,
                azmn_even=azmn_e_a,
                azmn_odd=azmn_o_a,
                bzmn_even=bzmn_e_a,
                bzmn_odd=bzmn_o_a,
                czmn_even=czmn_e_a,
                czmn_odd=czmn_o_a,
                blmn_even=blmn_e_a,
                blmn_odd=blmn_o_a,
                clmn_even=clmn_e_a,
                clmn_odd=clmn_o_a,
                arcon_even=arcon_e_a,
                arcon_odd=arcon_o_a,
                azcon_even=azcon_e_a,
                azcon_odd=azcon_o_a,
                mpol=int(wout.mpol),
                ntor=int(wout.ntor),
                nfp=int(wout.nfp),
                lasym=True,
                trig=trig,
                include_edge=bool(include_edge),
                masks=mask_pack,
            )
    else:
        with _trace("tomnsps_rzl"), _named_scope("tomnsps_rzl"):
            out_sym = tomnsps_rzl(
                armn_even=k.armn_e,
                armn_odd=k.armn_o,
                brmn_even=k.brmn_e,
                brmn_odd=k.brmn_o,
                crmn_even=k.crmn_e,
                crmn_odd=k.crmn_o,
                azmn_even=k.azmn_e,
                azmn_odd=k.azmn_o,
                bzmn_even=k.bzmn_e,
                bzmn_odd=k.bzmn_o,
                czmn_even=k.czmn_e,
                czmn_odd=k.czmn_o,
                blmn_even=blmn_even,
                blmn_odd=blmn_odd,
                clmn_even=clmn_even,
                clmn_odd=clmn_odd,
                arcon_even=k.arcon_e,
                arcon_odd=k.arcon_o,
                azcon_even=k.azcon_e,
                azcon_odd=k.azcon_o,
                mpol=int(wout.mpol),
                ntor=int(wout.ntor),
                nfp=int(wout.nfp),
                lasym=False,
                trig=trig,
                include_edge=bool(include_edge),
                masks=mask_pack,
            )
        out_asym = None

    # VMEC `lforbal` modifies the (m=1,n=0) symmetric forces to satisfy the
    # flux-surface-averaged force balance exactly. This primarily affects
    # the scalar residuals `fsqr/fsqz`. See `VMEC2000/Sources/General/tomnsp_mod.f`.
    if bool(apply_lforbal):
        from .vmec_lforbal import apply_lforbal_to_tomnsps, lforbal_factors_from_state

        ns = int(jnp.asarray(out_sym.frcc).shape[0])
        s_grid = jnp.linspace(0.0, 1.0, ns, dtype=jnp.asarray(out_sym.frcc).dtype)
        factors = lforbal_factors_from_state(
            bc=k.bc,
            trig=trig,
            wout=wout,
            s=s_grid,
            pru_even=k.pru_even,
            pru_odd=k.pru_odd,
            pzu_even=k.pzu_even,
            pzu_odd=k.pzu_odd,
            pr1_odd=k.pr1_odd,
            pz1_odd=k.pz1_odd,
        )
        frcc2, fzsc2 = apply_lforbal_to_tomnsps(frcc=out_sym.frcc, fzsc=out_sym.fzsc, factors=factors, trig=trig)
        out_sym = TomnspsRZL(
            frcc=frcc2,
            frss=out_sym.frss,
            fzsc=fzsc2,
            fzcs=out_sym.fzcs,
            flsc=out_sym.flsc,
            flcs=out_sym.flcs,
        )

    return VmecInternalResidualRZL(
        frcc=out_sym.frcc,
        frss=out_sym.frss,
        fzsc=out_sym.fzsc,
        fzcs=out_sym.fzcs,
        flsc=out_sym.flsc,
        flcs=out_sym.flcs,
        frsc=None if out_asym is None else out_asym.frsc,
        frcs=None if out_asym is None else out_asym.frcs,
        fzcc=None if out_asym is None else out_asym.fzcc,
        fzss=None if out_asym is None else out_asym.fzss,
        flcc=None if out_asym is None else out_asym.flcc,
        flss=None if out_asym is None else out_asym.flss,
    )


@dataclass(frozen=True)
class VmecRZResidualScalars:
    fsqr_like: float
    fsqz_like: float


def rz_residual_scalars_like_vmec(
    coeffs: VmecRZResidualCoeffs,
    *,
    bc,
    wout,
    s,
) -> VmecRZResidualScalars:
    """Compute VMEC-like invariant scalars for the R/Z residuals.

    This uses VMEC's documented structure:
        fsqr = gnorm * sum(gcr^2),  with gnorm = r1*fnorm and r1 = 1/(2*r0scale)^2 = 1/4.

    We approximate the missing VMEC angular weighting with a uniform tensor grid.
    """
    s = np.asarray(s)
    if s.size < 2:
        return VmecRZResidualScalars(fsqr_like=float("nan"), fsqz_like=float("nan"))

    # VMEC's r0scale from fixaray defaults to 1 (mscale(0)*nscale(0)).
    r1 = 0.25

    # VMEC uses volume and energies normalized by (2π)^2 in its internal scaling.
    vol_norm = float(wout.volume_p / (4.0 * np.pi**2))
    e_norm = float(max(wout.wb, wout.wp))
    r2 = e_norm / vol_norm if vol_norm != 0.0 else float("inf")

    # Approximate <guu * R^2> with a uniform angular average of the half-mesh field.
    r12 = np.asarray(bc.jac.r12)
    guu = np.asarray(bc.gij_b_uu)
    guu_r2 = guu * (r12 * r12)
    avg_guu_r2 = float(np.mean(guu_r2[1:]))  # exclude axis surface

    fnorm = 1.0 / (avg_guu_r2 * (r2 * r2)) if avg_guu_r2 != 0.0 else float("inf")
    gnorm = r1 * fnorm

    gcr2 = float(np.sum(np.asarray(coeffs.gcr_cos)[1:] ** 2 + np.asarray(coeffs.gcr_sin)[1:] ** 2))
    gcz2 = float(np.sum(np.asarray(coeffs.gcz_cos)[1:] ** 2 + np.asarray(coeffs.gcz_sin)[1:] ** 2))
    return VmecRZResidualScalars(fsqr_like=gnorm * gcr2, fsqz_like=gnorm * gcz2)
