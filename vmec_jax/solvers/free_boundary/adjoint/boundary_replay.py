"""Accepted-boundary geometry and vacuum-field replay helpers."""

from __future__ import annotations

from typing import Any

from vmec_jax._compat import jnp


def vacuum_boundary_fields_from_cylindrical_jax(
    *,
    br: Any,
    bp: Any,
    bz: Any,
    R: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    det_floor: float = 1.0e-30,
    include_bnormal_unit: bool = True,
    include_contravariant: bool = True,
) -> dict[str, Any]:
    """Project cylindrical vacuum-field samples onto VMEC boundary channels.

    This mirrors ``free_boundary.vacuum_boundary_fields_from_cylindrical`` for
    derivative tests, but returns a transformable JAX dictionary rather than
    the NumPy dataclass used by the production bridge.
    """

    br_arr = jnp.asarray(br)
    bp_arr = jnp.asarray(bp)
    bz_arr = jnp.asarray(bz)
    R_arr = jnp.asarray(R)
    Ru_arr = jnp.asarray(Ru)
    Zu_arr = jnp.asarray(Zu)
    Rv_arr = jnp.asarray(Rv)
    Zv_arr = jnp.asarray(Zv)

    g_uu = Ru_arr * Ru_arr + Zu_arr * Zu_arr
    g_uv = Ru_arr * Rv_arr + Zu_arr * Zv_arr
    g_vv = R_arr * R_arr + Rv_arr * Rv_arr + Zv_arr * Zv_arr
    det = g_uu * g_vv - g_uv * g_uv

    bu = br_arr * Ru_arr + bz_arr * Zu_arr
    bv = br_arr * Rv_arr + bp_arr * R_arr + bz_arr * Zv_arr
    n_r = -R_arr * Zu_arr
    n_phi = Zu_arr * Rv_arr - Ru_arr * Zv_arr
    n_z = R_arr * Ru_arr
    bnormal = br_arr * n_r + bp_arr * n_phi + bz_arr * n_z

    result = {
        "bu": bu,
        "bv": bv,
        "bnormal": bnormal,
        "g_uu": g_uu,
        "g_uv": g_uv,
        "g_vv": g_vv,
        "det_guv": det,
    }
    if bool(include_contravariant):
        det_safe = jnp.where(
            jnp.abs(det) >= float(det_floor),
            det,
            jnp.sign(det + 1.0e-300) * float(det_floor),
        )
        bsupu = (g_vv * bu - g_uv * bv) / det_safe
        bsupv = (g_uu * bv - g_uv * bu) / det_safe
        bsqvac = 0.5 * (bu * bsupu + bv * bsupv)
        result.update(
            {
                "bsupu": bsupu,
                "bsupv": bsupv,
                "bsqvac": bsqvac,
            }
        )
    if bool(include_bnormal_unit):
        n_norm = jnp.sqrt(n_r * n_r + n_phi * n_phi + n_z * n_z)
        result["bnormal_unit"] = bnormal / jnp.where(n_norm > 0.0, n_norm, 1.0)
    return result


def vacuum_boundary_fields_from_mode_coeffs_jax(
    mode_coeffs: Any,
    *,
    basis: dict[str, Any],
    bu_ext: Any,
    bv_ext: Any,
    g_uu: Any,
    g_uv: Any,
    g_vv: Any,
) -> dict[str, Any]:
    """Replay VMEC vacuum channels from JAX NESTOR mode coefficients."""

    pot = jnp.ravel(jnp.asarray(mode_coeffs))
    mnpd = int(basis["mnpd"])
    if int(pot.shape[0]) < mnpd:
        raise ValueError("mode_coeffs_too_small")
    potsin = pot[:mnpd]
    if bool(basis["lasym"]) and int(pot.shape[0]) >= 2 * mnpd:
        potcos = pot[mnpd : 2 * mnpd]
    else:
        potcos = jnp.zeros((mnpd,), dtype=pot.dtype)

    xmpot = jnp.asarray(basis["xmpot"], dtype=pot.dtype)
    n_raw = jnp.asarray(basis["n_raw"], dtype=pot.dtype)
    nfp = jnp.asarray(float(int(basis["nfp"])), dtype=pot.dtype)
    cos_phase = jnp.asarray(basis["cos_phase"], dtype=pot.dtype)
    sin_phase = jnp.asarray(basis["sin_phase"], dtype=pot.dtype)

    potu = cos_phase @ (xmpot * potsin)
    potv = cos_phase @ ((-n_raw * nfp) * potsin)
    if bool(basis["lasym"]):
        potu = potu - (sin_phase @ (xmpot * potcos))
        potv = potv - (sin_phase @ ((-n_raw * nfp) * potcos))

    bu_ext = jnp.asarray(bu_ext)
    bv_ext = jnp.asarray(bv_ext)
    potu = jnp.reshape(potu, bu_ext.shape)
    potv = jnp.reshape(potv, bv_ext.shape)
    bu = bu_ext + potu
    bv = bv_ext + potv
    g_uu = jnp.asarray(g_uu, dtype=bu.dtype)
    g_uv = jnp.asarray(g_uv, dtype=bu.dtype)
    g_vv = jnp.asarray(g_vv, dtype=bu.dtype)
    det = g_uu * g_vv - g_uv * g_uv
    det_safe = jnp.where(jnp.abs(det) > 1.0e-30, det, jnp.sign(det + 1.0e-300) * 1.0e-30)
    bsupu = (g_vv * bu - g_uv * bv) / det_safe
    bsupv = (g_uu * bv - g_uv * bu) / det_safe
    bsqvac = 0.5 * (bu * bsupu + bv * bsupv)
    return {
        "bu": bu,
        "bv": bv,
        "bsupu": bsupu,
        "bsupv": bsupv,
        "bsqvac": bsqvac,
        "det_guv": det,
    }


def direct_coil_boundary_bnormal_rms_jax(
    params: Any,
    *,
    R: Any,
    Z: Any,
    phi: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    br_add: Any = 0.0,
    bp_add: Any = 0.0,
    bz_add: Any = 0.0,
) -> Any:
    """Replay the accepted-boundary direct-coil normal-field RMS in JAX."""

    from vmec_jax.external_fields import sample_coil_field_cylindrical

    br, bp, bz = sample_coil_field_cylindrical(
        params,
        jnp.asarray(R),
        jnp.asarray(Z),
        jnp.asarray(phi),
    )
    br = br + jnp.asarray(br_add, dtype=br.dtype)
    bp = bp + jnp.asarray(bp_add, dtype=bp.dtype)
    bz = bz + jnp.asarray(bz_add, dtype=bz.dtype)
    vac = vacuum_boundary_fields_from_cylindrical_jax(
        br=br,
        bp=bp,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        include_contravariant=False,
    )
    bnormal = jnp.ravel(jnp.asarray(vac["bnormal"]))
    return jnp.sqrt(jnp.mean(bnormal * bnormal))


def free_boundary_boundary_geometry_jax(
    state: Any,
    static: Any,
    *,
    sample_nzeta: int | None = None,
) -> dict[str, Any]:
    """Synthesize accepted free-boundary geometry through JAX."""

    from vmec_jax.free_boundary import _freeb_boundary_sample_setup
    from vmec_jax.vmec_parity import vmec_m1_internal_to_physical_signed
    from vmec_jax.vmec_realspace import vmec_realspace_synthesis_multi

    cfg = static.cfg
    if sample_nzeta is None:
        sample_nzeta = 1 if (not bool(getattr(cfg, "lthreed", True))) else int(cfg.nzeta)
    setup = _freeb_boundary_sample_setup(static=static, sample_nzeta=int(sample_nzeta))
    trig = setup.trig

    Rcos = jnp.asarray(state.Rcos)
    Rsin = jnp.asarray(state.Rsin)
    Zcos = jnp.asarray(state.Zcos)
    Zsin = jnp.asarray(state.Zsin)
    Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
        Rcos=Rcos,
        Zsin=Zsin,
        Rsin=Rsin,
        Zcos=Zcos,
        modes=static.modes,
        lthreed=bool(getattr(cfg, "lthreed", True)),
        lasym=bool(getattr(cfg, "lasym", False)),
        lconm1=bool(getattr(cfg, "lconm1", True)),
    )

    coeff_cos = jnp.stack([Rcos[-1:, :], Zcos[-1:, :]], axis=0)
    coeff_sin = jnp.stack([Rsin[-1:, :], Zsin[-1:, :]], axis=0)
    base, dtheta, dzeta = vmec_realspace_synthesis_multi(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=static.modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=False,
        derivs=("base", "dtheta", "dzeta"),
    )

    second_facs = jnp.asarray(setup.second_facs, dtype=coeff_cos.dtype)
    second_cos = jnp.stack([Rcos[-1:, :], Zcos[-1:, :]], axis=0)[:, None, :, :] * second_facs[None, :, :, :]
    second_sin = jnp.stack([Rsin[-1:, :], Zsin[-1:, :]], axis=0)[:, None, :, :] * second_facs[None, :, :, :]
    second_base = vmec_realspace_synthesis_multi(
        coeff_cos=second_cos,
        coeff_sin=second_sin,
        modes=static.modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=False,
        derivs=("base",),
    )[0]

    R = base[0, 0]
    Z = base[1, 0]
    return {
        "R": R,
        "Z": Z,
        "phi": jnp.asarray(setup.phi_grid, dtype=R.dtype),
        "Ru": dtheta[0, 0],
        "Zu": dtheta[1, 0],
        "Rv": dzeta[0, 0],
        "Zv": dzeta[1, 0],
        "ruu": second_base[0, 0, 0],
        "ruv": second_base[0, 1, 0],
        "rvv": second_base[0, 2, 0],
        "zuu": second_base[1, 0, 0],
        "zuv": second_base[1, 1, 0],
        "zvv": second_base[1, 2, 0],
    }


__all__ = [
    "direct_coil_boundary_bnormal_rms_jax",
    "free_boundary_boundary_geometry_jax",
    "vacuum_boundary_fields_from_cylindrical_jax",
    "vacuum_boundary_fields_from_mode_coeffs_jax",
]
