"""JAX-native quasisymmetry diagnostics from VMEC-JAX states."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable

import numpy as np

from ._compat import jnp


def _require_jax():
    if jnp is None:
        raise ImportError("vmec_jax.quasisymmetry requires JAX (jax + jaxlib).")


def _as_surface_array(surfaces) -> jnp.ndarray:
    try:
        values = list(surfaces)  # type: ignore[arg-type]
    except Exception:
        values = [surfaces]
    return jnp.asarray(values, dtype=jnp.float64)


def _as_weight_array(weights, nsurf: int) -> jnp.ndarray:
    if weights is None:
        return jnp.ones((nsurf,), dtype=jnp.float64)
    return jnp.asarray(list(weights), dtype=jnp.float64)


def _quasisymmetry_angle_cache(
    *,
    nfp: int,
    xm_nyq,
    xn_nyq,
    ntheta: int = 63,
    nphi: int = 64,
) -> dict[str, object]:
    """Precompute the fixed angular grid used by QS residuals.

    Optimisation callbacks evaluate the same QS grid many times at different
    VMEC states.  Keeping these arrays outside the residual trace avoids
    rebuilding the mesh and trigonometric tables for every residual/Jacobian
    callback while preserving the public uncached API.
    """
    ntheta = int(ntheta)
    nphi = int(nphi)
    nfp = int(nfp)
    theta1d = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False, dtype=jnp.float64)
    phi1d = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, nphi, endpoint=False, dtype=jnp.float64)
    theta2d, phi2d = jnp.meshgrid(theta1d, phi1d, indexing="ij")
    xm_nyq = _as_jax_array(xm_nyq, dtype=np.float64)
    xn_nyq = _as_jax_array(xn_nyq, dtype=np.float64)
    angle = theta2d[:, :, None] * xm_nyq[None, None, :] - phi2d[:, :, None] * xn_nyq[None, None, :]
    return {
        "ntheta": ntheta,
        "nphi": nphi,
        "nfp": nfp,
        "theta1d": theta1d,
        "phi1d": phi1d,
        "theta2d": theta2d,
        "phi2d": phi2d,
        "dtheta": theta1d[1] - theta1d[0],
        "dphi": phi1d[1] - phi1d[0],
        "cosangle": jnp.cos(angle),
        "sinangle": jnp.sin(angle),
    }


def quasisymmetry_angle_cache(
    *,
    nfp: int,
    xm_nyq,
    xn_nyq,
    ntheta: int = 63,
    nphi: int = 64,
) -> dict[str, object]:
    """Precompute angular tables for repeated QS residual evaluations."""

    return _quasisymmetry_angle_cache(
        nfp=int(nfp),
        xm_nyq=xm_nyq,
        xn_nyq=xn_nyq,
        ntheta=int(ntheta),
        nphi=int(nphi),
    )


def _boozer_array(booz: dict, key: str):
    value = booz.get(key)
    if value is None:
        return None
    return _as_jax_array(value, dtype=np.float64)


def quasisymmetry_boozer_mode_residual_from_boozer_output(
    booz: dict,
    *,
    helicity_m: int = 1,
    helicity_n: int = 0,
    nfp: int | None = None,
    weights: Iterable[float] | None = None,
    normalize: bool = True,
    include_b00: bool = False,
) -> dict[str, object]:
    """Return a differentiable Boozer-coefficient quasisymmetry metric.

    A Boozer ``|B|`` spectrum is quasisymmetric when only modes satisfying
    ``helicity_m * n - helicity_n * nfp * m == 0`` remain.  This helper
    penalizes the complementary modes in ``bmnc_b`` and, when present, the
    asymmetric ``bmns_b`` spectrum.  The result is intentionally lightweight:
    it is suitable for validation gates and optimization diagnostics that need
    the Boozer-space content without constructing a full field-line residual.
    """

    _require_jax()
    bmnc = _boozer_array(booz, "bmnc_b")
    if bmnc is None:
        raise KeyError("Boozer output must contain bmnc_b")
    if bmnc.ndim != 2:
        raise ValueError(f"bmnc_b must have shape (nsurf, mnmax), got {bmnc.shape}")
    xm = _boozer_array(booz, "ixm_b")
    xn = _boozer_array(booz, "ixn_b")
    if xm is None or xn is None:
        raise KeyError("Boozer output must contain ixm_b and ixn_b")
    xm = jnp.asarray(xm, dtype=jnp.float64).reshape((-1,))
    xn = jnp.asarray(xn, dtype=jnp.float64).reshape((-1,))
    if int(xm.shape[0]) != int(bmnc.shape[1]) or int(xn.shape[0]) != int(bmnc.shape[1]):
        raise ValueError("Boozer mode arrays must match the second dimension of bmnc_b")

    if nfp is None:
        nfp_values = _boozer_array(booz, "nfp_b")
        nfp = 1 if nfp_values is None or int(jnp.size(nfp_values)) == 0 else int(np.asarray(nfp_values).reshape(-1)[0])
    nn = int(helicity_n) * int(nfp)
    target = (int(helicity_m) * xn - nn * xm) == 0
    if not include_b00:
        target = jnp.logical_or(target, jnp.logical_and(xm == 0, xn == 0))
    non_qs = jnp.logical_not(target)

    weights_arr = _as_weight_array(weights, int(bmnc.shape[0]))
    weights_arr = weights_arr.reshape((-1, 1))
    cos_residual = bmnc * non_qs[None, :]
    numerator = jnp.sum(weights_arr * cos_residual * cos_residual)
    denominator = jnp.sum(weights_arr * bmnc * bmnc)

    bmns = _boozer_array(booz, "bmns_b")
    if bmns is not None:
        if bmns.shape != bmnc.shape:
            raise ValueError(f"bmns_b must match bmnc_b shape {bmnc.shape}, got {bmns.shape}")
        sin_residual = bmns * non_qs[None, :]
        numerator = numerator + jnp.sum(weights_arr * sin_residual * sin_residual)
        denominator = denominator + jnp.sum(weights_arr * bmns * bmns)

    total = numerator
    if normalize:
        eps = jnp.asarray(jnp.finfo(bmnc.dtype).tiny, dtype=bmnc.dtype)
        total = numerator / jnp.maximum(denominator, eps)

    profile = jnp.sum(cos_residual * cos_residual, axis=1)
    if bmns is not None:
        profile = profile + jnp.sum((bmns * non_qs[None, :]) ** 2, axis=1)
    if normalize:
        profile_denom = jnp.sum(bmnc * bmnc, axis=1)
        if bmns is not None:
            profile_denom = profile_denom + jnp.sum(bmns * bmns, axis=1)
        profile = profile / jnp.maximum(profile_denom, jnp.asarray(jnp.finfo(bmnc.dtype).tiny, dtype=bmnc.dtype))

    return {
        "total": total,
        "profile": profile,
        "residual_bmnc": cos_residual,
        "target_mask": target,
        "non_qs_mask": non_qs,
        "weights": weights_arr.reshape((-1,)),
        "helicity_m": jnp.asarray(int(helicity_m)),
        "helicity_n": jnp.asarray(int(helicity_n)),
        "nfp": jnp.asarray(int(nfp)),
        "normalized": bool(normalize),
    }


def quasisymmetry_angle_cache_from_static(
    static,
    *,
    ntheta: int = 63,
    nphi: int = 64,
) -> dict[str, object]:
    """Precompute QS angular tables from VMEC static grid metadata."""

    from .modes import nyquist_mode_table_from_grid

    cfg = static.cfg
    nyq_modes = nyquist_mode_table_from_grid(
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
    )
    return _quasisymmetry_angle_cache(
        nfp=int(cfg.nfp),
        xm_nyq=jnp.asarray(nyq_modes.m, dtype=jnp.float64),
        xn_nyq=jnp.asarray(nyq_modes.n * int(cfg.nfp), dtype=jnp.float64),
        ntheta=int(ntheta),
        nphi=int(nphi),
    )


def _half_grid(radial_count: int, dtype) -> jnp.ndarray:
    if radial_count < 2:
        return jnp.zeros((0,), dtype=dtype)
    s_full = jnp.linspace(0.0, 1.0, radial_count, dtype=dtype)
    return 0.5 * (s_full[:-1] + s_full[1:])


def _interp_half_grid(samples, surfaces, s_half):
    samples = jnp.asarray(samples)
    surfaces = jnp.asarray(surfaces, dtype=samples.dtype)
    s_half = jnp.asarray(s_half, dtype=samples.dtype)
    if s_half.shape[0] == 0:
        raise ValueError("half-grid interpolation requires at least one radial point")
    if s_half.shape[0] == 1:
        return jnp.broadcast_to(samples[:1], (surfaces.shape[0],) + samples.shape[1:])

    idx_hi = jnp.searchsorted(s_half, surfaces, side="left")
    idx_hi = jnp.clip(idx_hi, 1, s_half.shape[0] - 1)
    idx_lo = idx_hi - 1
    x0 = s_half[idx_lo]
    x1 = s_half[idx_hi]
    denom = jnp.where(x1 != x0, x1 - x0, jnp.ones_like(x1))
    t = ((surfaces - x0) / denom).reshape((surfaces.shape[0],) + (1,) * (samples.ndim - 1))
    y0 = samples[idx_lo]
    y1 = samples[idx_hi]
    return y0 + t * (y1 - y0)


def _as_jax_array(values, *, dtype=None):
    try:
        arr = np.asarray(values)
    except Exception:
        return jnp.asarray(values, dtype=dtype)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    elif arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
    return jnp.asarray(arr)


def _radial_mode_matrix(values, *, radial_count: int, mode_count: int) -> jnp.ndarray:
    arr = _as_jax_array(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a rank-2 coefficient array, got shape {arr.shape}")
    if arr.shape == (radial_count, mode_count):
        return arr
    if arr.shape == (mode_count, radial_count):
        return jnp.swapaxes(arr, 0, 1)
    raise ValueError(
        f"unexpected coefficient shape {arr.shape}; expected {(radial_count, mode_count)} "
        f"or {(mode_count, radial_count)}"
    )


def _optional_radial_mode_matrix(wout, name: str, *, radial_count: int, mode_count: int, like) -> jnp.ndarray:
    if not bool(getattr(wout, "lasym", False)) or getattr(wout, name, None) is None:
        return jnp.zeros_like(jnp.asarray(like))
    return _radial_mode_matrix(getattr(wout, name), radial_count=radial_count, mode_count=mode_count)


def _vmec_wrout_nyquist_cos_coeffs_jax(*, f, modes, trig):
    f = jnp.asarray(f)
    if f.ndim != 3:
        raise ValueError(f"Expected f with shape (ns, ntheta, nzeta), got {f.shape}")

    m = jnp.asarray(modes.m, dtype=jnp.int32)
    n = jnp.asarray(modes.n, dtype=jnp.int32)
    if int(m.shape[0]) == 0:
        return jnp.zeros((int(f.shape[0]), 0), dtype=f.dtype)

    nt2 = int(trig.ntheta2)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    f = f[:, :nt2, :]

    cosmui = jnp.asarray(trig.cosmui, dtype=f.dtype)[:nt2, :]
    sinmui = jnp.asarray(trig.sinmui, dtype=f.dtype)[:nt2, :]
    cosnv = jnp.asarray(trig.cosnv, dtype=f.dtype)
    sinnv = jnp.asarray(trig.sinnv, dtype=f.dtype)

    mnyq = int(cosmui.shape[1] - 1)
    if mnyq > 0:
        cosmui = cosmui.at[:, mnyq].multiply(jnp.asarray(0.5, dtype=f.dtype))
    nnyq = int(cosnv.shape[1] - 1)
    if nnyq > 0:
        cosnv = cosnv.at[:, nnyq].multiply(jnp.asarray(0.5, dtype=f.dtype))

    f_theta_cos = jnp.einsum("sik,im->smk", f, cosmui, optimize=True)
    f_theta_sin = jnp.einsum("sik,im->smk", f, sinmui, optimize=True)
    cos_zeta = jnp.einsum("smk,kn->smn", f_theta_cos, cosnv, optimize=True)
    sin_zeta = jnp.einsum("smk,kn->smn", f_theta_sin, sinnv, optimize=True)

    n_abs = jnp.abs(n)
    sgn = jnp.where(n < 0, -1.0, 1.0).astype(f.dtype)
    coeff = cos_zeta[:, m, n_abs] + sgn[None, :] * sin_zeta[:, m, n_abs]

    mscale = jnp.asarray(trig.mscale, dtype=f.dtype)
    nscale = jnp.asarray(trig.nscale, dtype=f.dtype)
    dmult = mscale[m] * nscale[n_abs] * jnp.asarray(0.5 / float(getattr(trig, "r0scale", 1.0)) ** 2, dtype=f.dtype)
    dmult = jnp.where((m == 0) | (n == 0), 2.0 * dmult, dmult)
    return coeff * dmult[None, :]


def _vmec_wrout_nyquist_sin_coeffs_jax(*, f, modes, trig):
    f = jnp.asarray(f)
    if f.ndim != 3:
        raise ValueError(f"Expected f with shape (ns, ntheta, nzeta), got {f.shape}")

    m = jnp.asarray(modes.m, dtype=jnp.int32)
    n = jnp.asarray(modes.n, dtype=jnp.int32)
    if int(m.shape[0]) == 0:
        return jnp.zeros((int(f.shape[0]), 0), dtype=f.dtype)

    nt2 = int(trig.ntheta2)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    f = f[:, :nt2, :]

    cosmui = jnp.asarray(trig.cosmui, dtype=f.dtype)[:nt2, :]
    sinmui = jnp.asarray(trig.sinmui, dtype=f.dtype)[:nt2, :]
    cosnv = jnp.asarray(trig.cosnv, dtype=f.dtype)
    sinnv = jnp.asarray(trig.sinnv, dtype=f.dtype)

    mnyq = int(cosmui.shape[1] - 1)
    if mnyq > 0:
        cosmui = cosmui.at[:, mnyq].multiply(jnp.asarray(0.5, dtype=f.dtype))
    nnyq = int(cosnv.shape[1] - 1)
    if nnyq > 0:
        cosnv = cosnv.at[:, nnyq].multiply(jnp.asarray(0.5, dtype=f.dtype))

    f_theta_sin = jnp.einsum("sik,im->smk", f, sinmui, optimize=True)
    f_theta_cos = jnp.einsum("sik,im->smk", f, cosmui, optimize=True)
    cos_zeta = jnp.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True)
    sin_zeta = jnp.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True)

    n_abs = jnp.abs(n)
    sgn = jnp.where(n < 0, -1.0, 1.0).astype(f.dtype)
    coeff = cos_zeta[:, m, n_abs] - sgn[None, :] * sin_zeta[:, m, n_abs]

    mscale = jnp.asarray(trig.mscale, dtype=f.dtype)
    nscale = jnp.asarray(trig.nscale, dtype=f.dtype)
    dmult = mscale[m] * nscale[n_abs] * jnp.asarray(0.5 / float(getattr(trig, "r0scale", 1.0)) ** 2, dtype=f.dtype)
    dmult = jnp.where((m == 0) | (n == 0), 2.0 * dmult, dmult)
    return coeff * dmult[None, :]


def _vmec_symoutput_split_jax(*, f, trig, reversed_sym: bool = False):
    f = jnp.asarray(f)
    if f.ndim != 3:
        raise ValueError(f"Expected f with shape (ns, ntheta, nzeta), got {f.shape}")
    nt2 = int(trig.ntheta2)
    nt1 = int(trig.ntheta1)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    nzeta = int(f.shape[2])

    i0 = jnp.arange(nt2, dtype=jnp.int32)
    ir0 = jnp.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - jnp.arange(nzeta, dtype=jnp.int32)) % nzeta
    f_half = f[:, :nt2, :]
    f_ref = jnp.take(jnp.take(f, ir0, axis=1), kk, axis=2)
    if bool(reversed_sym):
        sym = 0.5 * (f_half - f_ref)
        asym = 0.5 * (f_half + f_ref)
    else:
        sym = 0.5 * (f_half + f_ref)
        asym = 0.5 * (f_half - f_ref)
    return sym, asym


def _zero_like_coeffs(coeffs):
    return jnp.zeros_like(jnp.asarray(coeffs))


def quasisymmetry_diagnostics_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    flux_local=None,
    prof_local=None,
    pressure_local=None,
):
    """Build the VMEC-only QS channels directly from a solved state."""
    _require_jax()

    from .booz_input import _filter_bsubuv_jxbforce_parity_jax
    from .driver import _final_flux_profiles_from_state
    from .energy import _iotaf_from_iotas
    from .integrals import cumrect_s_halfmesh
    from .modes import nyquist_mode_table_from_grid
    from .energy import flux_profiles_from_indata
    from .profiles import eval_profiles
    from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout
    from .vmec_lforbal import currents_from_bcovar
    from .vmec_tomnsp import vmec_trig_tables

    if pressure_local is None:
        prof_seed = eval_profiles(indata, np.asarray(static.s))
        pressure_local = jnp.asarray(prof_seed.get("pressure", jnp.zeros_like(jnp.asarray(static.s))))
    if prof_local is None:
        prof_local = {"pressure": pressure_local}
    if flux_local is None:
        flux_local = flux_profiles_from_indata(indata, np.asarray(static.s), signgs=int(signgs))

    flux, prof = _final_flux_profiles_from_state(
        indata=indata,
        static_in=static,
        state=state,
        signgs=int(signgs),
        flux_local=flux_local,
        prof_local=prof_local,
        pressure_local=pressure_local,
    )

    s_full = jnp.asarray(static.s)
    pres = jnp.asarray(prof.get("pressure", pressure_local))
    if int(pres.shape[0]) > 0:
        pres = pres.at[0].set(0.0)

    iotas = prof.get("iota", None)
    if iotas is None:
        phips = jnp.asarray(flux.phips)
        chipf = jnp.asarray(flux.chipf)
        chips = jnp.concatenate([chipf[:1], 0.5 * (chipf[1:] + chipf[:-1])], axis=0)
        safe_phips = jnp.where(phips != 0.0, phips, 1.0)
        iotas = jnp.where(phips != 0.0, chips / safe_phips, 0.0)
    iotas = jnp.asarray(iotas)
    if int(iotas.shape[0]) > 0:
        iotas = iotas.at[0].set(0.0)
    iotaf = jnp.asarray(prof.get("iotaf", _iotaf_from_iotas(iotas, lrfp=bool(indata.get_bool("LRFP", False)))))

    cfg = static.cfg
    wout_like = SimpleNamespace(
        phipf=jnp.asarray(flux.phipf),
        chipf=jnp.asarray(flux.chipf),
        phips=jnp.asarray(flux.phips),
        iotas=iotas,
        iotaf=iotaf,
        nfp=int(cfg.nfp),
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        lasym=bool(cfg.lasym),
        signgs=int(signgs),
        ncurr=int(indata.get_int("NCURR", 0)),
        lcurrent=bool(int(indata.get_int("NCURR", 0)) == 1),
        flux_is_internal=True,
    )

    nyq_modes = nyquist_mode_table_from_grid(
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
    )
    mmax_nyq = int(np.max(np.asarray(nyq_modes.m))) if int(nyq_modes.K) > 0 else 0
    nmax_nyq = int(np.max(np.abs(np.asarray(nyq_modes.n)))) if int(nyq_modes.K) > 0 else 0
    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        mmax=max(int(cfg.mpol) - 1, mmax_nyq),
        nmax=max(int(cfg.ntor), nmax_nyq),
        lasym=bool(cfg.lasym),
        dtype=jnp.asarray(state.Rcos).dtype,
    )

    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        pres=pres,
        use_wout_bsup=False,
        use_wout_bsub_for_lambda=False,
        use_wout_bmag_for_bsq=False,
        use_vmec_synthesis=True,
        trig=trig,
    )
    buco, bvco, _, _ = currents_from_bcovar(bc=bc, trig=trig, wout=wout_like, s=s_full)

    bsq = jnp.asarray(bc.bsq)
    bmod_sq = jnp.maximum(
        2.0 * (bsq - pres[:, None, None]),
        jnp.asarray(jnp.finfo(bsq.dtype).tiny, dtype=bsq.dtype),
    )
    bmod = jnp.sqrt(bmod_sq)
    bsubu = jnp.asarray(bc.bsubu)
    bsubv = jnp.asarray(bc.bsubv)
    if not bool(cfg.lasym):
        pshalf = jnp.sqrt(jnp.maximum(0.5 * (s_full[1:] + s_full[:-1]), 0.0))
        pshalf = jnp.concatenate([pshalf[:1], pshalf], axis=0)
        if int(pshalf.shape[0]) > 1:
            pshalf = pshalf.at[0].set(pshalf[1])
        pshalf = pshalf[:, None, None]
        bsubu, bsubv = _filter_bsubuv_jxbforce_parity_jax(
            bsubu_even=bsubu,
            bsubu_odd=pshalf * bsubu,
            bsubv_even=bsubv,
            bsubv_odd=pshalf * bsubv,
            trig=trig,
            mmax_force=max(int(cfg.mpol) - 1, 0),
            nmax_force=int(cfg.ntor),
            s=s_full,
        )

    if bool(cfg.lasym):
        sqrtg_sym, sqrtg_asym = _vmec_symoutput_split_jax(f=jnp.asarray(bc.jac.sqrtg), trig=trig)
        bmod_sym, bmod_asym = _vmec_symoutput_split_jax(f=bmod, trig=trig)
        bsubu_sym, bsubu_asym = _vmec_symoutput_split_jax(f=bsubu, trig=trig)
        bsubv_sym, bsubv_asym = _vmec_symoutput_split_jax(f=bsubv, trig=trig)
        bsupu_sym, bsupu_asym = _vmec_symoutput_split_jax(f=jnp.asarray(bc.bsupu), trig=trig)
        bsupv_sym, bsupv_asym = _vmec_symoutput_split_jax(f=jnp.asarray(bc.bsupv), trig=trig)
        gmnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=sqrtg_sym, modes=nyq_modes, trig=trig)
        bmnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=bmod_sym, modes=nyq_modes, trig=trig)
        bsubumnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=bsubu_sym, modes=nyq_modes, trig=trig)
        bsubvmnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=bsubv_sym, modes=nyq_modes, trig=trig)
        bsupumnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=bsupu_sym, modes=nyq_modes, trig=trig)
        bsupvmnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=bsupv_sym, modes=nyq_modes, trig=trig)
        gmns = _vmec_wrout_nyquist_sin_coeffs_jax(f=sqrtg_asym, modes=nyq_modes, trig=trig)
        bmns = _vmec_wrout_nyquist_sin_coeffs_jax(f=bmod_asym, modes=nyq_modes, trig=trig)
        bsubumns = _vmec_wrout_nyquist_sin_coeffs_jax(f=bsubu_asym, modes=nyq_modes, trig=trig)
        bsubvmns = _vmec_wrout_nyquist_sin_coeffs_jax(f=bsubv_asym, modes=nyq_modes, trig=trig)
        bsupumns = _vmec_wrout_nyquist_sin_coeffs_jax(f=bsupu_asym, modes=nyq_modes, trig=trig)
        bsupvmns = _vmec_wrout_nyquist_sin_coeffs_jax(f=bsupv_asym, modes=nyq_modes, trig=trig)
    else:
        gmnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=jnp.asarray(bc.jac.sqrtg), modes=nyq_modes, trig=trig)
        bmnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=bmod, modes=nyq_modes, trig=trig)
        bsubumnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=bsubu, modes=nyq_modes, trig=trig)
        bsubvmnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=bsubv, modes=nyq_modes, trig=trig)
        bsupumnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=jnp.asarray(bc.bsupu), modes=nyq_modes, trig=trig)
        bsupvmnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=jnp.asarray(bc.bsupv), modes=nyq_modes, trig=trig)
        gmns = _zero_like_coeffs(gmnc)
        bmns = _zero_like_coeffs(bmnc)
        bsubumns = _zero_like_coeffs(bsubumnc)
        bsubvmns = _zero_like_coeffs(bsubvmnc)
        bsupumns = _zero_like_coeffs(bsupumnc)
        bsupvmns = _zero_like_coeffs(bsupvmnc)

    if not bool(cfg.lasym):
        mask_bsub = (jnp.asarray(nyq_modes.m) >= int(cfg.mpol)) | (jnp.abs(jnp.asarray(nyq_modes.n)) > int(cfg.ntor))
        bsubumnc = jnp.where(mask_bsub[None, :], 0.0, jnp.asarray(bsubumnc))
        bsubvmnc = jnp.where(mask_bsub[None, :], 0.0, jnp.asarray(bsubvmnc))
    else:
        mask_bsub = (jnp.asarray(nyq_modes.m) >= int(cfg.mpol)) | (jnp.abs(jnp.asarray(nyq_modes.n)) > int(cfg.ntor))
        bsubumnc = jnp.where(mask_bsub[None, :], 0.0, jnp.asarray(bsubumnc))
        bsubumns = jnp.where(mask_bsub[None, :], 0.0, jnp.asarray(bsubumns))
        bsubvmnc = jnp.where(mask_bsub[None, :], 0.0, jnp.asarray(bsubvmnc))
        bsubvmns = jnp.where(mask_bsub[None, :], 0.0, jnp.asarray(bsubvmns))

    phi = cumrect_s_halfmesh(jnp.asarray(flux.phipf) * float(2.0 * np.pi * int(signgs)), s_full)
    return SimpleNamespace(
        lasym=bool(cfg.lasym),
        nfp=int(cfg.nfp),
        iotas=iotas,
        buco=jnp.asarray(buco),
        bvco=jnp.asarray(bvco),
        gmnc=jnp.asarray(gmnc),
        gmns=jnp.asarray(gmns),
        bmnc=jnp.asarray(bmnc),
        bmns=jnp.asarray(bmns),
        bsubumnc=jnp.asarray(bsubumnc),
        bsubumns=jnp.asarray(bsubumns),
        bsubvmnc=jnp.asarray(bsubvmnc),
        bsubvmns=jnp.asarray(bsubvmns),
        bsupumnc=jnp.asarray(bsupumnc),
        bsupumns=jnp.asarray(bsupumns),
        bsupvmnc=jnp.asarray(bsupvmnc),
        bsupvmns=jnp.asarray(bsupvmns),
        xm_nyq=jnp.asarray(nyq_modes.m, dtype=jnp.float64),
        xn_nyq=jnp.asarray(nyq_modes.n * int(cfg.nfp), dtype=jnp.float64),
        phi=jnp.asarray(phi),
    )


def quasisymmetry_ratio_residual_from_wout(
    wout,
    *,
    surfaces,
    helicity_m: int = 1,
    helicity_n: int = 0,
    weights: Iterable[float] | None = None,
    ntheta: int = 63,
    nphi: int = 64,
    angle_cache: dict[str, object] | None = None,
):
    """Evaluate the VMEC-only quasisymmetry residual from wout-like data."""
    _require_jax()

    surfaces = _as_surface_array(surfaces)
    weights = _as_weight_array(weights, int(surfaces.shape[0]))
    if weights.shape[0] != surfaces.shape[0]:
        raise ValueError("weights must have the same length as surfaces")

    ntheta = int(ntheta)
    nphi = int(nphi)
    nfp = int(getattr(wout, "nfp"))
    helicity_m = int(helicity_m)
    helicity_n = int(helicity_n)

    iotas_full = _as_jax_array(getattr(wout, "iotas"), dtype=np.float64)
    radial_count = int(iotas_full.shape[0])
    s_half = _half_grid(radial_count, iotas_full.dtype)
    xm_nyq = _as_jax_array(getattr(wout, "xm_nyq"), dtype=np.float64)
    xn_nyq = _as_jax_array(getattr(wout, "xn_nyq"), dtype=np.float64)
    mode_count = int(xm_nyq.shape[0])

    iota = _interp_half_grid(iotas_full[1:], surfaces, s_half)
    G = _interp_half_grid(_as_jax_array(getattr(wout, "bvco"), dtype=np.float64)[1:], surfaces, s_half)
    I = _interp_half_grid(_as_jax_array(getattr(wout, "buco"), dtype=np.float64)[1:], surfaces, s_half)

    gmnc_full = _radial_mode_matrix(getattr(wout, "gmnc"), radial_count=radial_count, mode_count=mode_count)
    bmnc_full = _radial_mode_matrix(getattr(wout, "bmnc"), radial_count=radial_count, mode_count=mode_count)
    bsubumnc_full = _radial_mode_matrix(getattr(wout, "bsubumnc"), radial_count=radial_count, mode_count=mode_count)
    bsubvmnc_full = _radial_mode_matrix(getattr(wout, "bsubvmnc"), radial_count=radial_count, mode_count=mode_count)
    bsupumnc_full = _radial_mode_matrix(getattr(wout, "bsupumnc"), radial_count=radial_count, mode_count=mode_count)
    bsupvmnc_full = _radial_mode_matrix(getattr(wout, "bsupvmnc"), radial_count=radial_count, mode_count=mode_count)
    gmns_full = _optional_radial_mode_matrix(
        wout, "gmns", radial_count=radial_count, mode_count=mode_count, like=gmnc_full
    )
    bmns_full = _optional_radial_mode_matrix(
        wout, "bmns", radial_count=radial_count, mode_count=mode_count, like=bmnc_full
    )
    bsubumns_full = _optional_radial_mode_matrix(
        wout, "bsubumns", radial_count=radial_count, mode_count=mode_count, like=bsubumnc_full
    )
    bsubvmns_full = _optional_radial_mode_matrix(
        wout, "bsubvmns", radial_count=radial_count, mode_count=mode_count, like=bsubvmnc_full
    )
    bsupumns_full = _optional_radial_mode_matrix(
        wout, "bsupumns", radial_count=radial_count, mode_count=mode_count, like=bsupumnc_full
    )
    bsupvmns_full = _optional_radial_mode_matrix(
        wout, "bsupvmns", radial_count=radial_count, mode_count=mode_count, like=bsupvmnc_full
    )

    gmnc = _interp_half_grid(
        gmnc_full[1:],
        surfaces,
        s_half,
    )
    gmns = _interp_half_grid(
        gmns_full[1:],
        surfaces,
        s_half,
    )
    bmnc = _interp_half_grid(
        bmnc_full[1:],
        surfaces,
        s_half,
    )
    bmns = _interp_half_grid(
        bmns_full[1:],
        surfaces,
        s_half,
    )
    bsubumnc = _interp_half_grid(
        bsubumnc_full[1:],
        surfaces,
        s_half,
    )
    bsubumns = _interp_half_grid(
        bsubumns_full[1:],
        surfaces,
        s_half,
    )
    bsubvmnc = _interp_half_grid(
        bsubvmnc_full[1:],
        surfaces,
        s_half,
    )
    bsubvmns = _interp_half_grid(
        bsubvmns_full[1:],
        surfaces,
        s_half,
    )
    bsupumnc = _interp_half_grid(
        bsupumnc_full[1:],
        surfaces,
        s_half,
    )
    bsupumns = _interp_half_grid(
        bsupumns_full[1:],
        surfaces,
        s_half,
    )
    bsupvmnc = _interp_half_grid(
        bsupvmnc_full[1:],
        surfaces,
        s_half,
    )
    bsupvmns = _interp_half_grid(
        bsupvmns_full[1:],
        surfaces,
        s_half,
    )

    if angle_cache is None:
        angle_cache = _quasisymmetry_angle_cache(
            nfp=nfp,
            xm_nyq=xm_nyq,
            xn_nyq=xn_nyq,
            ntheta=ntheta,
            nphi=nphi,
        )
    else:
        ntheta = int(angle_cache["ntheta"])
        nphi = int(angle_cache["nphi"])
    theta1d = angle_cache["theta1d"]
    phi1d = angle_cache["phi1d"]
    theta2d = angle_cache["theta2d"]
    phi2d = angle_cache["phi2d"]
    dtheta = angle_cache["dtheta"]
    dphi = angle_cache["dphi"]
    cosangle = angle_cache["cosangle"]
    sinangle = angle_cache["sinangle"]

    modB = jnp.einsum("sm,tpm->stp", bmnc, cosangle) + jnp.einsum("sm,tpm->stp", bmns, sinangle)
    d_B_d_theta = jnp.einsum("sm,tpm,m->stp", bmnc, -sinangle, xm_nyq) + jnp.einsum(
        "sm,tpm,m->stp", bmns, cosangle, xm_nyq
    )
    d_B_d_phi = jnp.einsum("sm,tpm,m->stp", bmnc, sinangle, xn_nyq) + jnp.einsum(
        "sm,tpm,m->stp", bmns, -cosangle, xn_nyq
    )
    sqrtg = jnp.einsum("sm,tpm->stp", gmnc, cosangle) + jnp.einsum("sm,tpm->stp", gmns, sinangle)
    bsubu = jnp.einsum("sm,tpm->stp", bsubumnc, cosangle) + jnp.einsum(
        "sm,tpm->stp", bsubumns, sinangle
    )
    bsubv = jnp.einsum("sm,tpm->stp", bsubvmnc, cosangle) + jnp.einsum(
        "sm,tpm->stp", bsubvmns, sinangle
    )
    bsupu = jnp.einsum("sm,tpm->stp", bsupumnc, cosangle) + jnp.einsum(
        "sm,tpm->stp", bsupumns, sinangle
    )
    bsupv = jnp.einsum("sm,tpm->stp", bsupvmnc, cosangle) + jnp.einsum(
        "sm,tpm->stp", bsupvmns, sinangle
    )

    d_psi_d_s = -_as_jax_array(getattr(wout, "phi"), dtype=np.float64)[-1] / (2.0 * jnp.pi)
    sqrtg_safe = jnp.where(sqrtg != 0.0, sqrtg, jnp.ones_like(sqrtg))
    B_dot_grad_B = bsupu * d_B_d_theta + bsupv * d_B_d_phi
    B_cross_grad_B_dot_grad_psi = d_psi_d_s * (bsubu * d_B_d_phi - bsubv * d_B_d_theta) / sqrtg_safe

    sqrtg_abs = jnp.abs(sqrtg)
    sqrtg_abs_safe = jnp.maximum(sqrtg_abs, jnp.asarray(jnp.finfo(sqrtg.dtype).tiny, dtype=sqrtg.dtype))
    modB_safe = jnp.maximum(jnp.abs(modB), jnp.asarray(jnp.finfo(modB.dtype).tiny, dtype=modB.dtype))
    V_prime = nfp * dtheta * dphi * jnp.sum(sqrtg_abs_safe, axis=(1, 2))

    nn = helicity_n * nfp
    prefactor = jnp.sqrt(
        weights[:, None, None] * nfp * dtheta * dphi / V_prime[:, None, None] * sqrtg_abs_safe
    )
    residuals3d = prefactor * (
        B_cross_grad_B_dot_grad_psi * (nn - iota[:, None, None] * helicity_m)
        - B_dot_grad_B * (helicity_m * G[:, None, None] + nn * I[:, None, None])
    ) / (modB_safe**3)

    residuals1d = jnp.ravel(residuals3d)
    profile = jnp.sum(residuals3d * residuals3d, axis=(1, 2))
    total = jnp.sum(residuals1d * residuals1d)
    return {
        "surfaces": surfaces,
        "weights": weights,
        "ntheta": jnp.asarray(ntheta),
        "nphi": jnp.asarray(nphi),
        "nfp": jnp.asarray(nfp),
        "dtheta": dtheta,
        "dphi": dphi,
        "theta1d": theta1d,
        "phi1d": phi1d,
        "theta2d": theta2d,
        "phi2d": phi2d,
        "d_psi_d_s": d_psi_d_s,
        "V_prime": V_prime,
        "residuals3d": residuals3d,
        "residuals1d": residuals1d,
        "profile": profile,
        "total": total,
        "modB": modB,
        "d_B_d_theta": d_B_d_theta,
        "d_B_d_phi": d_B_d_phi,
        "sqrtg": sqrtg,
        "bsubu": bsubu,
        "bsubv": bsubv,
        "bsupu": bsupu,
        "bsupv": bsupv,
        "B_dot_grad_B": B_dot_grad_B,
        "B_cross_grad_B_dot_grad_psi": B_cross_grad_B_dot_grad_psi,
        "iota": iota,
        "G": G,
        "I": I,
    }


def quasisymmetry_ratio_residual_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    surfaces,
    helicity_m: int = 1,
    helicity_n: int = 0,
    weights: Iterable[float] | None = None,
    ntheta: int = 63,
    nphi: int = 64,
    flux_local=None,
    prof_local=None,
    pressure_local=None,
    angle_cache: dict[str, object] | None = None,
):
    """Evaluate the VMEC-only QS residual directly from a solved state."""
    data = quasisymmetry_diagnostics_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
        flux_local=flux_local,
        prof_local=prof_local,
        pressure_local=pressure_local,
    )
    return quasisymmetry_ratio_residual_from_wout(
        data,
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        weights=weights,
        ntheta=ntheta,
        nphi=nphi,
        angle_cache=angle_cache,
    )
