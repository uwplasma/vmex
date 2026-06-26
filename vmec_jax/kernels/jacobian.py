"""VMEC-style half-mesh Jacobian construction.

This module ports the core logic of VMEC2000's ``jacobian.f`` / ``jacobian_par``
into a small, dependency-light implementation.

Motivation
----------
VMEC uses an internal representation in which *odd-m* Fourier content is stored
in a ``1/sqrt(s)`` form for axis regularity. In real space, many quantities are
represented as:

    X(s,θ,ζ) = X_even(s,θ,ζ) + sqrt(s) * X_odd(s,θ,ζ)

VMEC then constructs several derivatives and the Jacobian on the **radial half
mesh** with explicit correction terms arising from ``d/ds sqrt(s)``.

The direct Cartesian cross-product Jacobian in :mod:`vmec_jax.geom` is fine for
many uses, but does not match VMEC's discrete half-mesh convention used for
Nyquist ``wout`` fields like ``gmnc/gmns``. This module exists specifically for
parity work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from .._compat import jnp, tree_util
from .realspace import vmec_realspace_synthesis, vmec_realspace_synthesis_dtheta
from .parity import vmec_m1_internal_to_physical_signed
from .tomnsp import VmecTrigTables


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VmecHalfMeshJacobian:
    """Half-mesh Jacobian outputs (VMEC conventions)."""

    # R on half mesh.
    r12: Any  # (ns, ntheta, nzeta)
    # Rs and Zs on half mesh.
    rs: Any  # (ns, ntheta, nzeta)
    zs: Any  # (ns, ntheta, nzeta)
    # Ru and Zu on half mesh.
    ru12: Any  # (ns, ntheta, nzeta)
    zu12: Any  # (ns, ntheta, nzeta)
    # tau = sqrt(g)/R on half mesh (VMEC name).
    tau: Any  # (ns, ntheta, nzeta)
    # sqrt(g) on half mesh.
    sqrtg: Any  # (ns, ntheta, nzeta)

    def tree_flatten(self):
        """Return JAX pytree leaves and static metadata for transformations."""
        children = (self.r12, self.rs, self.zs, self.ru12, self.zu12, self.tau, self.sqrtg)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild the object from JAX pytree metadata and leaves."""
        return cls(*children)


def _safe_divide(x, y, *, eps: float = 1e-14):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    mask = jnp.abs(y) > eps
    y_safe = jnp.where(mask, y, jnp.ones_like(y))
    return mask.astype(x.dtype) * (x / y_safe)


def _pshalf_from_s(s: Any) -> Any:
    """Compute VMEC-like sqrt(s) on the half mesh."""
    s = jnp.asarray(s)
    if s.shape[0] < 2:
        return jnp.sqrt(jnp.maximum(s, 0.0))
    sh = 0.5 * (s[1:] + s[:-1])
    p = jnp.concatenate([sh[:1], sh], axis=0)
    return jnp.sqrt(jnp.maximum(p, 0.0))


def _apply_vmec_axis_rules(coeff: Any, m: Any) -> Any:
    """Apply VMEC's jmin1 axis rules to Fourier coefficients.

    VMEC's `totzsp_mod` enforces:
      - m=1: copy js=2 to js=1 (origin extrapolation),
      - m>=2: axis contributions are zero (jmin1=2).
    """
    coeff = jnp.asarray(coeff)
    if coeff.shape[0] < 2:
        return coeff
    m = jnp.asarray(m)
    c0 = coeff[0]
    c1 = coeff[1]
    m1_mask = (m == 1)
    mge2_mask = (m >= 2)
    c0 = jnp.where(m1_mask, c1, c0)
    c0 = jnp.where(mge2_mask, jnp.zeros_like(c0), c0)
    return coeff.at[0].set(c0)


def jacobian_half_mesh_from_parity(
    *,
    pr1_even,
    pr1_odd,
    pz1_even,
    pz1_odd,
    pru_even,
    pru_odd,
    pzu_even,
    pzu_odd,
    s,
) -> VmecHalfMeshJacobian:
    """Compute half-mesh Jacobian quantities using VMEC's discrete formula.

    Parameters
    ----------
    pr1_even, pr1_odd, ... :
        Real-space fields representing the internal VMEC decomposition:

            X = X_even + sqrt(s)*X_odd

        Each array has shape ``(ns, ntheta, nzeta)``.
    s:
        Radial grid (ns,), assumed uniform.
    """
    pr1_even = jnp.asarray(pr1_even)
    pr1_odd = jnp.asarray(pr1_odd)
    pz1_even = jnp.asarray(pz1_even)
    pz1_odd = jnp.asarray(pz1_odd)
    pru_even = jnp.asarray(pru_even)
    pru_odd = jnp.asarray(pru_odd)
    pzu_even = jnp.asarray(pzu_even)
    pzu_odd = jnp.asarray(pzu_odd)
    s = jnp.asarray(s)

    ns = int(s.shape[0])
    if ns < 2:
        z = jnp.zeros_like(pr1_even)
        return VmecHalfMeshJacobian(r12=pr1_even, rs=z, zs=z, ru12=z, zu12=z, tau=z, sqrtg=z)

    hs = s[1] - s[0]
    ohs = _safe_divide(1.0, hs)
    # This is exactly VMEC's `p25 = (0.5)^2`.
    dshalfds = 0.25

    psqrts = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    pshalf = _pshalf_from_s(s)[:, None, None]

    # Slices for js>=1.
    sl = slice(1, ns)
    sm1 = slice(0, ns - 1)

    ru12_inner = 0.5 * (
        pru_even[sl]
        + pru_even[sm1]
        + pshalf[sl] * (pru_odd[sl] + pru_odd[sm1])
    )
    zs_inner = ohs * (
        (pz1_even[sl] - pz1_even[sm1]) + pshalf[sl] * (pz1_odd[sl] - pz1_odd[sm1])
    )
    tau_inner = ru12_inner * zs_inner + dshalfds * (
        pru_odd[sl] * pz1_odd[sl]
        + pru_odd[sm1] * pz1_odd[sm1]
        + _safe_divide(
            pru_even[sl] * pz1_odd[sl] + pru_even[sm1] * pz1_odd[sm1],
            pshalf[sl],
        )
    )

    zu12_inner = 0.5 * (
        pzu_even[sl]
        + pzu_even[sm1]
        + pshalf[sl] * (pzu_odd[sl] + pzu_odd[sm1])
    )
    rs_inner = ohs * (
        (pr1_even[sl] - pr1_even[sm1]) + pshalf[sl] * (pr1_odd[sl] - pr1_odd[sm1])
    )
    r12_inner = 0.5 * (
        pr1_even[sl]
        + pr1_even[sm1]
        + pshalf[sl] * (pr1_odd[sl] + pr1_odd[sm1])
    )
    tau_inner = tau_inner - rs_inner * zu12_inner - dshalfds * (
        pzu_odd[sl] * pr1_odd[sl]
        + pzu_odd[sm1] * pr1_odd[sm1]
        + _safe_divide(
            pzu_even[sl] * pr1_odd[sl] + pzu_even[sm1] * pr1_odd[sm1],
            pshalf[sl],
        )
    )

    # VMEC copies js=1 to js=0 for tau/r12 in the serial routine.
    ru12 = jnp.concatenate([ru12_inner[:1], ru12_inner], axis=0)
    zu12 = jnp.concatenate([zu12_inner[:1], zu12_inner], axis=0)
    rs = jnp.concatenate([rs_inner[:1], rs_inner], axis=0)
    zs = jnp.concatenate([zs_inner[:1], zs_inner], axis=0)
    r12 = jnp.concatenate([r12_inner[:1], r12_inner], axis=0)
    tau = jnp.concatenate([tau_inner[:1], tau_inner], axis=0)

    sqrtg = r12 * tau
    # Avoid NaNs on axis.
    sqrtg = jnp.where(psqrts == 0, 0.0, sqrtg)
    return VmecHalfMeshJacobian(r12=r12, rs=rs, zs=zs, ru12=ru12, zu12=zu12, tau=tau, sqrtg=sqrtg)


def vmec_half_mesh_jacobian_from_state(
    *,
    state,
    modes,
    trig: VmecTrigTables,
    s,
    lconm1: bool = True,
    lthreed: bool = True,
    apply_m1_constraint: bool = True,
    apply_scalxc: bool = True,
    mask_even: Any | None = None,
    mask_odd: Any | None = None,
) -> VmecHalfMeshJacobian:
    """Compute VMEC half-mesh Jacobian directly from Fourier coefficients."""
    m = jnp.asarray(modes.m)
    if mask_even is None or mask_odd is None:
        mask_even = (m % 2) == 0
        mask_odd = jnp.logical_not(mask_even)
    else:
        mask_even = jnp.asarray(mask_even)
        mask_odd = jnp.asarray(mask_odd)

    Rcos = jnp.asarray(state.Rcos)
    Rsin = jnp.asarray(state.Rsin)
    Zcos = jnp.asarray(state.Zcos)
    Zsin = jnp.asarray(state.Zsin)

    if bool(apply_m1_constraint):
        # VMEC stores internal coefficients; undo the m=1 internal constraint before
        # synthesis when physical coefficients are required.
        lasym_state = bool(getattr(getattr(state, "layout", None), "lasym", False))
        Rcos_int, Zsin_int, Rsin_int, Zcos_int = vmec_m1_internal_to_physical_signed(
            Rcos=Rcos,
            Zsin=Zsin,
            Rsin=Rsin,
            Zcos=Zcos,
            modes=modes,
            lthreed=bool(lthreed),
            lasym=lasym_state,
            lconm1=bool(lconm1),
        )
        Rcos = jnp.asarray(Rcos_int)
        Rsin = jnp.asarray(Rsin_int)
        Zcos = jnp.asarray(Zcos_int)
        Zsin = jnp.asarray(Zsin_int)

    # Apply VMEC's axis rules (jmin1) before real-space synthesis.
    Rcos = _apply_vmec_axis_rules(Rcos, m)
    Rsin = _apply_vmec_axis_rules(Rsin, m)
    Zcos = _apply_vmec_axis_rules(Zcos, m)
    Zsin = _apply_vmec_axis_rules(Zsin, m)

    mask_even_f = jnp.asarray(mask_even, dtype=Rcos.dtype)
    mask_odd_f = jnp.asarray(mask_odd, dtype=Rcos.dtype)

    coeff_cos_stack = jnp.stack([Rcos, Zcos], axis=0)
    coeff_sin_stack = jnp.stack([Rsin, Zsin], axis=0)

    def _eval_stack(mask_stack):
        coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
        coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]
        return vmec_realspace_synthesis(
            coeff_cos=coeff_cos,
            coeff_sin=coeff_sin,
            modes=modes,
            trig=trig,
            coeffs_internal=True,
            apply_scalxc=bool(apply_scalxc),
            s=s,
        )

    def _eval_stack_dtheta(mask_stack):
        coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
        coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]
        return vmec_realspace_synthesis_dtheta(
            coeff_cos=coeff_cos,
            coeff_sin=coeff_sin,
            modes=modes,
            trig=trig,
            coeffs_internal=True,
            apply_scalxc=bool(apply_scalxc),
            s=s,
        )

    mask_stack = jnp.stack([mask_even_f, mask_odd_f], axis=0)
    stack = _eval_stack(mask_stack)
    stack_t = _eval_stack_dtheta(mask_stack)

    even = stack[0]
    odd = stack[1]
    even_t = stack_t[0]
    odd_t = stack_t[1]

    pr1_even = even[0]
    pr1_odd = odd[0]
    pz1_even = even[1]
    pz1_odd = odd[1]
    pru_even = even_t[0]
    pru_odd = odd_t[0]
    pzu_even = even_t[1]
    pzu_odd = odd_t[1]

    return jacobian_half_mesh_from_parity(
        pr1_even=pr1_even,
        pr1_odd=pr1_odd,
        pz1_even=pz1_even,
        pz1_odd=pz1_odd,
        pru_even=pru_even,
        pru_odd=pru_odd,
        pzu_even=pzu_even,
        pzu_odd=pzu_odd,
        s=s,
    )
