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


def _nearest_half_mesh_indices(surfaces: Iterable[float], *, n_half: int) -> np.ndarray:
    if int(n_half) < 1:
        raise ValueError("QI residual requires at least one half-mesh Boozer surface")
    surf = np.asarray(list(surfaces), dtype=float)
    s_half = 0.5 * (np.arange(int(n_half), dtype=float) + np.arange(1, int(n_half) + 1, dtype=float)) / float(n_half)
    return np.asarray([int(np.argmin(np.abs(s_half - value))) for value in surf], dtype=np.int32)


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
    return {**out, "surfaces": surface_values, "surface_indices": jnp.asarray(surface_indices)}
