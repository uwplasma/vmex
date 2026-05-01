"""Differentiable quasi-isodynamic residuals.

The reference QI scripts in ``omnigenity_optimization`` measure whether the
bounce width and field-line profile of magnetic wells are independent of
field-line label.  Their implementation uses branch finding and splines, which
are useful diagnostics but not a smooth objective for JAX autodiff.  This
module keeps the same target and replaces hard branch extraction with smooth
level-set widths:

``width(alpha, B_j) = integral H(B_j - B(alpha, phi)) dphi``

where ``H`` is a logistic approximation to the step function.  A QI surface has
widths that are independent of ``alpha`` for every level ``B_j``.  The residual
also includes a smooth profile-consistency term at fixed toroidal angle.  This
prevents single-helicity QH-like fields from scoring artificially well just
because changing ``alpha`` phase-shifts a well without changing its width.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ._compat import jnp

__all__ = [
    "boundary_max_elongation_from_rz",
    "max_elongation_penalty_from_state",
    "mirror_ratio_penalty_from_boozer_modes",
    "mirror_ratio_penalty_from_boozer_output",
    "quasi_isodynamic_residual_from_boozer_modes",
    "quasi_isodynamic_residual_from_boozer_output",
    "quasi_isodynamic_residual_from_state",
]


def _require_jax():
    if jnp is None:
        raise ImportError("vmec_jax.quasi_isodynamic requires JAX (jax + jaxlib).")


def _as_surface_array(surfaces) -> jnp.ndarray:
    try:
        values = list(surfaces)  # type: ignore[arg-type]
    except Exception:
        values = [surfaces]
    return jnp.asarray(values, dtype=jnp.float64)


def _as_surface_list(surfaces) -> list[float]:
    try:
        values = list(surfaces)  # type: ignore[arg-type]
    except Exception:
        values = [surfaces]
    return [float(value) for value in values]


def _as_weight_array(weights, nsurf: int) -> jnp.ndarray:
    if weights is None:
        return jnp.ones((nsurf,), dtype=jnp.float64)
    return jnp.asarray(list(weights), dtype=jnp.float64)


def _smooth_reduce_max(values, *, axis=None, softness: float = 0.0):
    """Return a hard or smooth maximum with stable normalization."""
    values = jnp.asarray(values)
    if float(softness) <= 0.0:
        return jnp.max(values, axis=axis)
    eps = jnp.asarray(float(softness), dtype=values.dtype)
    vmax = jnp.max(values, axis=axis, keepdims=True)
    shifted = jnp.exp((values - vmax) / eps)
    out = jnp.squeeze(vmax, axis=axis) + eps * jnp.log(jnp.mean(shifted, axis=axis))
    return out


def _smooth_reduce_min(values, *, axis=None, softness: float = 0.0):
    return -_smooth_reduce_max(-jnp.asarray(values), axis=axis, softness=softness)


def _positive_part(value, *, softness: float = 0.0):
    value = jnp.asarray(value)
    if float(softness) <= 0.0:
        return jnp.maximum(value, jnp.asarray(0.0, dtype=value.dtype))
    eps = jnp.asarray(float(softness), dtype=value.dtype)
    return eps * jnp.log1p(jnp.exp(value / eps))


def _nearest_half_mesh_indices(surfaces: Iterable[float], *, n_half: int) -> np.ndarray:
    if int(n_half) < 1:
        raise ValueError("QI residual requires at least one half-mesh Boozer surface")
    surf = np.asarray(list(surfaces), dtype=float)
    s_half = 0.5 * (np.arange(int(n_half), dtype=float) + np.arange(1, int(n_half) + 1, dtype=float)) / float(n_half)
    return np.asarray([int(np.argmin(np.abs(s_half - value))) for value in surf], dtype=np.int32)


def mirror_ratio_penalty_from_boozer_modes(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    nfp: int,
    bmns_b=None,
    threshold: float = 0.21,
    weights: Iterable[float] | None = None,
    ntheta: int = 128,
    nphi: int = 128,
    phimin: float = 0.0,
    smooth_extrema: float = 0.0,
    smooth_penalty: float = 0.0,
):
    """Penalize the maximum mirror ratio from Boozer ``|B|`` modes.

    This is the JAX-native analogue of the ``MirrorRatioPen`` diagnostic used
    in the reference ``omnigenity_optimization`` QI script.  For every supplied
    Boozer surface it evaluates ``|B|(theta_B, phi_B)`` on a uniform grid and
    computes

    ``M = (Bmax - Bmin) / (Bmax + Bmin)``.

    The least-squares residual is ``max(0, M - threshold)`` per surface.  Set
    ``smooth_extrema`` and/or ``smooth_penalty`` to positive values when a fully
    smooth softmax/softplus surrogate is preferred.
    """
    _require_jax()

    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    if bmnc_b.ndim == 1:
        bmnc_b = bmnc_b[None, :]
    bmns_b = jnp.zeros_like(bmnc_b) if bmns_b is None else jnp.asarray(bmns_b, dtype=jnp.float64)
    if bmns_b.ndim == 1:
        bmns_b = bmns_b[None, :]
    xm_b = jnp.asarray(xm_b, dtype=jnp.float64)
    xn_b = jnp.asarray(xn_b, dtype=jnp.float64)
    if int(bmnc_b.shape[1]) != int(xm_b.shape[0]) or int(xm_b.shape[0]) != int(xn_b.shape[0]):
        raise ValueError("Boozer mode arrays must have the same mode dimension as bmnc_b")
    if bmns_b.shape != bmnc_b.shape:
        raise ValueError("bmns_b must have the same shape as bmnc_b")
    if int(ntheta) < 4 or int(nphi) < 4:
        raise ValueError("mirror-ratio penalty requires ntheta >= 4 and nphi >= 4")

    nsurf = int(bmnc_b.shape[0])
    weights_arr = _as_weight_array(weights, nsurf)
    if int(weights_arr.shape[0]) != nsurf:
        raise ValueError("weights must have the same length as the number of Boozer surfaces")

    dtype = bmnc_b.dtype
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False, dtype=dtype)
    phi = jnp.linspace(
        float(phimin),
        float(phimin) + 2.0 * jnp.pi / float(nfp),
        int(nphi),
        endpoint=False,
        dtype=dtype,
    )
    angle = theta[None, :, None, None] * xm_b[None, None, None, :] - phi[None, None, :, None] * xn_b[
        None, None, None, :
    ]
    bmag = jnp.sum(
        bmnc_b[:, None, None, :] * jnp.cos(angle) + bmns_b[:, None, None, :] * jnp.sin(angle),
        axis=-1,
    )
    bmax = _smooth_reduce_max(bmag, axis=(1, 2), softness=float(smooth_extrema))
    bmin = _smooth_reduce_min(bmag, axis=(1, 2), softness=float(smooth_extrema))
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    # Truncated Boozer |B| spectra can undershoot and produce nonpositive
    # pointwise values during intermediate optimizer steps.  The physical
    # mirror-ratio target should then become a large finite penalty, not an
    # Inf/NaN that breaks the trust-region linear algebra.
    bmin_positive = jnp.maximum(bmin, tiny)
    denom = jnp.maximum(bmax + bmin_positive, tiny)
    mirror_ratio = (bmax - bmin_positive) / denom
    penalty = _positive_part(mirror_ratio - float(threshold), softness=float(smooth_penalty))
    residuals1d = penalty * jnp.sqrt(weights_arr)
    total = jnp.sum(residuals1d * residuals1d)
    return {
        "residuals1d": residuals1d,
        "total": total,
        "penalty": penalty,
        "mirror_ratio": mirror_ratio,
        "bmax": bmax,
        "bmin": bmin,
        "bmag": bmag,
        "theta": theta,
        "phi": phi,
        "threshold": jnp.asarray(float(threshold), dtype=dtype),
    }


def mirror_ratio_penalty_from_boozer_output(
    booz,
    *,
    nfp: int | None = None,
    threshold: float = 0.21,
    weights: Iterable[float] | None = None,
    ntheta: int = 128,
    nphi: int = 128,
    phimin: float = 0.0,
    smooth_extrema: float = 0.0,
    smooth_penalty: float = 0.0,
):
    """Evaluate :func:`mirror_ratio_penalty_from_boozer_modes` from Boozer output."""
    nfp_local = int(nfp) if nfp is not None else int(np.asarray(booz["nfp_b"]))
    bmns_b = booz.get("bmns_b")
    return mirror_ratio_penalty_from_boozer_modes(
        bmnc_b=booz["bmnc_b"],
        bmns_b=bmns_b,
        xm_b=booz["ixm_b"],
        xn_b=booz["ixn_b"],
        nfp=nfp_local,
        threshold=threshold,
        weights=weights,
        ntheta=ntheta,
        nphi=nphi,
        phimin=phimin,
        smooth_extrema=smooth_extrema,
        smooth_penalty=smooth_penalty,
    )


def boundary_max_elongation_from_rz(
    R,
    Z,
    *,
    phi=None,
    smooth_extrema: float = 0.0,
):
    """Estimate maximum LCFS cross-section elongation from ``R(theta, phi), Z``.

    The reference SIMSOPT QI script computes an effective elongation by solving
    for normal-plane cross-sections and fitting an ellipse from perimeter and
    area.  That path uses root finding and is not a good JAX objective.  This
    differentiable proxy computes, for each fixed toroidal angle, the covariance
    of the 3-D boundary curve and returns ``sqrt(lambda_major/lambda_minor)``
    using the two largest covariance eigenvalues.  It tracks the same geometric
    failure mode (very stretched boundary cross-sections) while remaining cheap
    and compatible with autodiff.
    """
    _require_jax()

    R = jnp.asarray(R, dtype=jnp.float64)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    if R.shape != Z.shape or R.ndim != 2:
        raise ValueError("R and Z must both have shape (ntheta, nphi)")
    nphi = int(R.shape[1])
    dtype = R.dtype
    if phi is None:
        phi_arr = jnp.linspace(0.0, 2.0 * jnp.pi, nphi, endpoint=False, dtype=dtype)
    else:
        phi_arr = jnp.asarray(phi, dtype=dtype)
        if int(phi_arr.shape[0]) != nphi:
            raise ValueError("phi must have one value per toroidal grid point")

    X = R * jnp.cos(phi_arr)[None, :]
    Y = R * jnp.sin(phi_arr)[None, :]
    points = jnp.stack([X, Y, Z], axis=-1)  # (ntheta, nphi, 3)
    points = jnp.swapaxes(points, 0, 1)  # (nphi, ntheta, 3)
    centered = points - jnp.mean(points, axis=1, keepdims=True)
    cov = jnp.einsum("pti,ptj->pij", centered, centered) / jnp.asarray(R.shape[0], dtype=dtype)
    eigvals = jnp.linalg.eigvalsh(cov)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    elongation = jnp.sqrt(jnp.maximum(eigvals[:, -1], tiny) / jnp.maximum(eigvals[:, -2], tiny))
    max_elongation = _smooth_reduce_max(elongation, axis=0, softness=float(smooth_extrema))
    return {
        "max_elongation": max_elongation,
        "elongation": elongation,
        "phi": phi_arr,
    }


def max_elongation_penalty_from_state(
    *,
    state,
    static,
    threshold: float = 8.0,
    ntheta: int = 64,
    nphi: int = 24,
    s_index: int = -1,
    smooth_extrema: float = 0.0,
    smooth_penalty: float = 0.0,
):
    """Penalize excessive LCFS elongation from a solved VMEC state."""
    _require_jax()
    if int(ntheta) < 4 or int(nphi) < 3:
        raise ValueError("elongation penalty requires ntheta >= 4 and nphi >= 3")

    from .fourier import build_helical_basis, eval_fourier
    from .grids import AngleGrid
    from .state import VMECState
    from .vmec_parity import vmec_m1_internal_to_physical_signed

    nfp = int(static.cfg.nfp)
    # Keep the angular grid host-static. ``build_helical_basis`` intentionally
    # uses NumPy to cache static Fourier tables; feeding it traced JAX arrays
    # would fail when this objective is evaluated inside the optimizer JIT.
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False, dtype=float)
    phi = np.linspace(0.0, 2.0 * np.pi / float(nfp), int(nphi), endpoint=False, dtype=float)
    zeta = phi * float(nfp)
    grid = AngleGrid(theta=theta, zeta=zeta, nfp=nfp)
    basis = build_helical_basis(static.modes, grid, cache=False)

    cfg = static.cfg
    lconm1 = bool(getattr(cfg, "lconm1", True))
    lthreed = bool(getattr(cfg, "lthreed", int(getattr(cfg, "ntor", 0)) > 0))
    lasym = bool(getattr(cfg, "lasym", False))
    state_use = state
    if lconm1 and (lthreed or lasym) and int(getattr(cfg, "mpol", 0)) > 1:
        Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
            Rcos=state.Rcos,
            Zsin=state.Zsin,
            Rsin=state.Rsin,
            Zcos=state.Zcos,
            modes=static.modes,
            lthreed=lthreed,
            lasym=lasym,
            lconm1=lconm1,
        )
        state_use = VMECState(
            layout=state.layout,
            Rcos=Rcos,
            Rsin=Rsin,
            Zcos=Zcos,
            Zsin=Zsin,
            Lcos=state.Lcos,
            Lsin=state.Lsin,
        )

    s_idx = int(s_index)
    R = eval_fourier(state_use.Rcos[s_idx], state_use.Rsin[s_idx], basis, coeffs_internal=True)
    Z = eval_fourier(state_use.Zcos[s_idx], state_use.Zsin[s_idx], basis, coeffs_internal=True)
    out = boundary_max_elongation_from_rz(R, Z, phi=phi, smooth_extrema=smooth_extrema)
    penalty = _positive_part(out["max_elongation"] - float(threshold), softness=float(smooth_penalty))
    residuals1d = jnp.asarray([penalty], dtype=jnp.float64)
    return {
        **out,
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        "penalty": penalty,
        "threshold": jnp.asarray(float(threshold), dtype=jnp.float64),
    }


def quasi_isodynamic_residual_from_boozer_modes(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    nfp: int,
    weights: Iterable[float] | None = None,
    nphi: int = 151,
    nalpha: int = 31,
    n_bounce: int = 51,
    softness: float = 2.0e-2,
    profile_weight: float = 1.0,
    phimin: float = 0.0,
):
    """Evaluate a smooth QI residual from Boozer ``|B|`` Fourier modes.

    Parameters
    ----------
    bmnc_b:
        Cosine Boozer ``|B|`` coefficients with shape ``(nsurf, nmodes)``.
    xm_b, xn_b:
        Boozer mode numbers. ``xn_b`` should use the physical toroidal mode
        convention, matching ``booz_xform_jax`` and BOOZ_XFORM.
    iota_b:
        Rotational transform on the same Boozer surfaces.
    nfp:
        Number of field periods.
    weights:
        Optional surface weights.  They are applied as square roots to the
        least-squares residual vector.
    nphi, nalpha, n_bounce:
        Sampling resolution along toroidal angle, field-line label, and
        normalized bounce level.
    softness:
        Logistic smoothing width in normalized ``|B|`` units. Smaller values
        approach hard branch widths but increase stiffness.
    profile_weight:
        Relative weight for the field-line profile consistency residual.  Set
        to 0 to recover the width-only surrogate.
    phimin:
        Start of the toroidal interval. The interval length is one field period.

    Returns
    -------
    dict
        ``residuals1d`` is suitable for least-squares optimization. ``total``
        is its squared norm.
    """
    _require_jax()

    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    xm_b = jnp.asarray(xm_b, dtype=jnp.float64)
    xn_b = jnp.asarray(xn_b, dtype=jnp.float64)
    iota_b = jnp.asarray(iota_b, dtype=jnp.float64)
    nfp = int(nfp)
    nphi = int(nphi)
    nalpha = int(nalpha)
    n_bounce = int(n_bounce)
    if bmnc_b.ndim != 2:
        raise ValueError(f"bmnc_b must have shape (nsurf, nmodes), got {bmnc_b.shape}")
    if int(bmnc_b.shape[1]) != int(xm_b.shape[0]) or int(xm_b.shape[0]) != int(xn_b.shape[0]):
        raise ValueError("Boozer mode arrays must have the same mode dimension as bmnc_b")
    if int(iota_b.shape[0]) != int(bmnc_b.shape[0]):
        raise ValueError("iota_b must have one value per Boozer surface")
    if nphi < 4 or nalpha < 2 or n_bounce < 2:
        raise ValueError("QI residual requires nphi >= 4, nalpha >= 2, and n_bounce >= 2")

    nsurf = int(bmnc_b.shape[0])
    weights_arr = _as_weight_array(weights, nsurf)
    if int(weights_arr.shape[0]) != nsurf:
        raise ValueError("weights must have the same length as the number of Boozer surfaces")

    dtype = bmnc_b.dtype
    phi0 = jnp.asarray(float(phimin), dtype=dtype)
    phi1 = phi0 + jnp.asarray(2.0 * np.pi / nfp, dtype=dtype)
    phi = jnp.linspace(phi0, phi1, nphi, endpoint=True, dtype=dtype)
    alpha = jnp.linspace(0.0, 2.0 * jnp.pi, nalpha, endpoint=False, dtype=dtype)

    theta = alpha[None, None, :] + iota_b[:, None, None] * phi[None, :, None]
    angle = theta[:, :, :, None] * xm_b[None, None, None, :] - phi[None, :, None, None] * xn_b[None, None, None, :]
    bmag = jnp.sum(bmnc_b[:, None, None, :] * jnp.cos(angle), axis=-1)

    bmin = jnp.min(bmag, axis=(1, 2), keepdims=True)
    bmax = jnp.max(bmag, axis=(1, 2), keepdims=True)
    denom = jnp.maximum(bmax - bmin, jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype))
    bnorm = (bmag - bmin) / denom

    levels = jnp.linspace(0.0, 1.0, n_bounce + 2, endpoint=True, dtype=dtype)[1:-1]
    eps = jnp.maximum(jnp.asarray(float(softness), dtype=dtype), jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
    occupancy = jax_sigmoid((levels[None, None, None, :] - bnorm[:, :, :, None]) / eps)

    # Mean over the uniformly sampled toroidal interval. This is proportional
    # to the bounce width used in the branch-based diagnostic.
    widths = jnp.mean(occupancy, axis=1)
    widths_mean = jnp.mean(widths, axis=1, keepdims=True)
    width_residuals3d = (widths - widths_mean) * jnp.sqrt(weights_arr)[:, None, None]
    width_residuals1d = jnp.ravel(width_residuals3d) / jnp.sqrt(jnp.asarray(nalpha * n_bounce, dtype=dtype))

    profile_mean = jnp.mean(bnorm, axis=2, keepdims=True)
    profile_residuals3d = (
        (bnorm - profile_mean)
        * jnp.sqrt(weights_arr)[:, None, None]
        * jnp.asarray(float(profile_weight), dtype=dtype)
    )
    profile_residuals1d = jnp.ravel(profile_residuals3d) / jnp.sqrt(jnp.asarray(nalpha * nphi, dtype=dtype))
    residuals1d = jnp.concatenate([width_residuals1d, profile_residuals1d])
    total = jnp.sum(residuals1d * residuals1d)
    return {
        "residuals1d": residuals1d,
        "residuals3d": width_residuals3d,
        "width_residuals1d": width_residuals1d,
        "width_residuals3d": width_residuals3d,
        "profile_residuals1d": profile_residuals1d,
        "profile_residuals3d": profile_residuals3d,
        "total": total,
        "widths": widths,
        "widths_mean": widths_mean,
        "profile_mean": profile_mean,
        "bmag": bmag,
        "bnorm": bnorm,
        "levels": levels,
        "phi": phi,
        "alpha": alpha,
        "iota": iota_b,
    }


def jax_sigmoid(x):
    """Small wrapper to keep this module importable without importing jax eagerly."""
    return 1.0 / (1.0 + jnp.exp(-x))


def quasi_isodynamic_residual_from_boozer_output(
    booz,
    *,
    nfp: int | None = None,
    weights: Iterable[float] | None = None,
    nphi: int = 151,
    nalpha: int = 31,
    n_bounce: int = 51,
    softness: float = 2.0e-2,
    profile_weight: float = 1.0,
    phimin: float = 0.0,
):
    """Evaluate the smooth QI residual from a ``booz_xform_jax`` output dict."""
    nfp_local = int(nfp) if nfp is not None else int(np.asarray(booz["nfp_b"]))
    return quasi_isodynamic_residual_from_boozer_modes(
        bmnc_b=booz["bmnc_b"],
        xm_b=booz["ixm_b"],
        xn_b=booz["ixn_b"],
        iota_b=booz["iota_b"],
        nfp=nfp_local,
        weights=weights,
        nphi=nphi,
        nalpha=nalpha,
        n_bounce=n_bounce,
        softness=softness,
        profile_weight=profile_weight,
        phimin=phimin,
    )


def quasi_isodynamic_residual_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    surfaces,
    weights: Iterable[float] | None = None,
    mboz: int = 12,
    nboz: int = 12,
    nphi: int = 151,
    nalpha: int = 31,
    n_bounce: int = 51,
    softness: float = 2.0e-2,
    profile_weight: float = 1.0,
    phimin: float = 0.0,
    flux_local=None,
    prof_local=None,
    pressure_local=None,
    jit_booz: bool = False,
    booz_constants=None,
    booz_grids=None,
    surface_indices=None,
):
    """Evaluate a differentiable QI residual directly from a solved VMEC state.

    This uses ``vmec_jax.booz_xform_inputs_from_state`` followed by the
    functional ``booz_xform_jax`` API.  ``booz_xform_jax`` is an optional
    runtime dependency; install it to use this state-level objective.
    """
    _require_jax()
    try:
        from booz_xform_jax import booz_xform_from_inputs, prepare_booz_xform_constants_from_inputs
    except Exception as exc:  # pragma: no cover - exercised only when optional dep missing
        raise ImportError(
            "quasi_isodynamic_residual_from_state requires booz_xform_jax. "
            "Install it with `pip install booz_xform_jax` or from github.com/uwplasma/booz_xform_jax."
        ) from exc

    from .booz_input import booz_xform_inputs_from_state

    surface_list = _as_surface_list(surfaces)
    surface_values = jnp.asarray(surface_list, dtype=jnp.float64)
    inputs = booz_xform_inputs_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
        flux=flux_local,
        profiles_half=prof_local,
    )
    if pressure_local is not None:
        # The argument is accepted for symmetry with quasisymmetry helpers; the
        # Boozer input adapter already consumes ``prof_local``/``flux_local``.
        del pressure_local

    if booz_constants is None or booz_grids is None:
        constants, grids = prepare_booz_xform_constants_from_inputs(
            inputs=inputs,
            mboz=int(mboz),
            nboz=int(nboz),
            asym=bool(static.cfg.lasym),
        )
    else:
        constants, grids = booz_constants, booz_grids
    if surface_indices is None:
        surface_indices = _nearest_half_mesh_indices(surface_list, n_half=int(inputs.rmnc.shape[0]))
    booz = booz_xform_from_inputs(
        inputs=inputs,
        constants=constants,
        grids=grids,
        surface_indices=jnp.asarray(surface_indices, dtype=jnp.int32),
        jit=bool(jit_booz),
    )
    out = quasi_isodynamic_residual_from_boozer_output(
        booz,
        nfp=int(inputs.nfp),
        weights=weights,
        nphi=nphi,
        nalpha=nalpha,
        n_bounce=n_bounce,
        softness=softness,
        profile_weight=profile_weight,
        phimin=phimin,
    )
    return {
        **out,
        "booz": booz,
        "surfaces": surface_values,
        "surface_indices": jnp.asarray(surface_indices),
    }
