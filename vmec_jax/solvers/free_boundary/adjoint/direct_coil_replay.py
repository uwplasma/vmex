"""Direct-coil accepted-boundary vacuum replay helpers.

These functions are the JAX-visible bridge between coil parameters and the
accepted free-boundary ``bsqvac`` arrays used by the branch-local adjoint
validation ladder.  They deliberately replay a fixed accepted boundary; they
do not claim derivatives through the host adaptive controller that selected
that boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

from vmec_jax._compat import jax, jnp
from vmec_jax.solvers.free_boundary.adjoint.boundary_replay import (
    vacuum_boundary_fields_from_cylindrical_jax,
    vacuum_boundary_fields_from_mode_coeffs_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.replay_context import (
    direct_coil_trace_vacuum_field_override as _direct_coil_trace_vacuum_field_override,
)
from vmec_jax.solvers.free_boundary.adjoint.runtime import jax_named_scope as _runtime_jax_named_scope
from vmec_jax.solvers.free_boundary.adjoint.vmec_nestor import dense_vmec_nestor_mode_solve_jax


def _jax_named_scope(name: str) -> Any:
    return _runtime_jax_named_scope(name, jax_module=jax, nullcontext_factory=nullcontext)


def direct_coil_boundary_bsqvac_jax(
    params: Any,
    *,
    R: Any,
    Z: Any,
    phi: Any,
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
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    br_add: Any = 0.0,
    bp_add: Any = 0.0,
    bz_add: Any = 0.0,
    wint: Any | None = None,
    include_analytic: bool = True,
    include_diagnostics: bool = True,
    include_mode_diagnostics: bool = True,
    vac_override: Mapping[str, Any] | None = None,
    coil_geometry: Any | None = None,
    nestor_solve_mode: str = "dense",
    nestor_operator_solver: str = "gmres",
    nestor_operator_tol: float = 1.0e-11,
    nestor_operator_atol: float = 1.0e-13,
    nestor_operator_maxiter: int | None = None,
    nestor_operator_restart: int | None = None,
) -> dict[str, Any]:
    """Replay accepted-boundary direct-coil ``bsqvac`` through JAX NESTOR.

    This is the reusable phase-2 validation primitive for the production
    accepted-output ladder.  It holds a VMEC plasma boundary fixed, samples the
    differentiable direct-coil Biot-Savart field on that boundary, projects the
    normal field into VMEC/NESTOR source space, solves the dense JAX mode-space
    vacuum response, and reconstructs ``bsqvac`` on the boundary.

    The helper validates and exposes the differentiable accepted-boundary
    replay contract.  It intentionally does **not** differentiate through the
    outer host-controlled nonlinear VMEC iteration loop.
    """

    from vmec_jax.external_fields import sample_coil_field_cylindrical, sample_coil_field_cylindrical_from_geometry

    R_j = jnp.asarray(R)
    if vac_override is None:
        with _jax_named_scope("vmec_jax.free_boundary.direct_coil_sample"):
            if coil_geometry is None:
                br, bp, bz = sample_coil_field_cylindrical(
                    params,
                    R_j,
                    jnp.asarray(Z),
                    jnp.asarray(phi),
                )
            else:
                br, bp, bz = sample_coil_field_cylindrical_from_geometry(
                    coil_geometry,
                    R_j,
                    jnp.asarray(Z),
                    jnp.asarray(phi),
                    regularization_epsilon=float(getattr(params, "regularization_epsilon", 0.0)),
                    chunk_size=getattr(params, "chunk_size", None),
                )
            br = br + jnp.asarray(br_add, dtype=br.dtype)
            bp = bp + jnp.asarray(bp_add, dtype=bp.dtype)
            bz = bz + jnp.asarray(bz_add, dtype=bz.dtype)
        with _jax_named_scope("vmec_jax.free_boundary.vacuum_boundary_projection"):
            vac = vacuum_boundary_fields_from_cylindrical_jax(
                br=br,
                bp=bp,
                bz=bz,
                R=R_j,
                Ru=Ru,
                Zu=Zu,
                Rv=Rv,
                Zv=Zv,
                include_bnormal_unit=False,
                include_contravariant=False,
            )
    else:
        vac = {
            "bu": jnp.asarray(vac_override["bu"]),
            "bv": jnp.asarray(vac_override["bv"]),
            "bnormal": jnp.asarray(vac_override["bnormal"]),
            "g_uu": jnp.asarray(vac_override["g_uu"]),
            "g_uv": jnp.asarray(vac_override["g_uv"]),
            "g_vv": jnp.asarray(vac_override["g_vv"]),
        }
    if wint is None:
        wint_j = jnp.ones_like(R_j)
    else:
        wint_j = jnp.asarray(wint, dtype=jnp.asarray(vac["bnormal"]).dtype)
    bexni = -jnp.asarray(vac["bnormal"]) * wint_j * ((2.0 * jnp.pi) ** 2)
    with _jax_named_scope("vmec_jax.free_boundary.dense_nestor_mode_solve"):
        mode_solution = dense_vmec_nestor_mode_solve_jax(
            R=R_j,
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
            bexni=jnp.ravel(bexni),
            basis=basis,
            tables=tables,
            signgs=int(signgs),
            nvper=int(nvper),
            include_analytic=bool(include_analytic),
            include_phi_flat=bool(include_mode_diagnostics),
            include_residual=bool(include_mode_diagnostics),
            solve_mode=str(nestor_solve_mode),
            operator_solver=str(nestor_operator_solver),
            operator_tol=float(nestor_operator_tol),
            operator_atol=float(nestor_operator_atol),
            operator_maxiter=nestor_operator_maxiter,
            operator_restart=nestor_operator_restart,
        )
    with _jax_named_scope("vmec_jax.free_boundary.mode_field_reconstruction"):
        channels = vacuum_boundary_fields_from_mode_coeffs_jax(
            mode_solution["mode_coeffs"],
            basis=basis,
            bu_ext=vac["bu"],
            bv_ext=vac["bv"],
            g_uu=vac["g_uu"],
            g_uv=vac["g_uv"],
            g_vv=vac["g_vv"],
        )
    out = {"bsqvac": channels["bsqvac"]}
    if bool(include_diagnostics):
        out.update(
            {
                "channels": channels,
                "mode_solution": mode_solution,
                "vac": vac,
                "bexni": bexni,
            }
        )
    return out


def direct_coil_boundary_bsqvac_from_trace_jax(
    params: Any,
    geometry: dict[str, Any],
    trace: dict[str, Any],
    *,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    wint: Any,
    include_analytic: bool = True,
    include_diagnostics: bool = True,
    include_mode_diagnostics: bool = True,
    freeze_vacuum_field: bool = False,
    coil_geometry: Any | None = None,
    nestor_solve_mode: str = "dense",
    nestor_operator_solver: str = "gmres",
    nestor_operator_tol: float = 1.0e-11,
    nestor_operator_atol: float = 1.0e-13,
    nestor_operator_maxiter: int | None = None,
    nestor_operator_restart: int | None = None,
) -> dict[str, Any]:
    """Replay direct-coil ``bsqvac`` on accepted geometry using trace metadata.

    ``trace`` may be either a full residual-step trace containing
    ``freeb_nestor_trace`` or the nested NESTOR trace itself.  This keeps the
    production validation ladder from duplicating trace-to-replay plumbing in
    every test while keeping the differentiated path explicit: accepted
    geometry and direct-coil parameters remain JAX-visible, while basis/tables
    and axis-additive fields are captured trace data.
    """

    nestor_trace = trace.get("freeb_nestor_trace", trace)
    if not isinstance(nestor_trace, dict):
        raise ValueError("trace must be a NESTOR trace or contain 'freeb_nestor_trace'")

    vac_override = _direct_coil_trace_vacuum_field_override(trace) if bool(freeze_vacuum_field) else None
    return direct_coil_boundary_bsqvac_jax(
        params,
        R=geometry["R"],
        Z=geometry["Z"],
        phi=geometry["phi"],
        Ru=geometry["Ru"],
        Zu=geometry["Zu"],
        Rv=geometry["Rv"],
        Zv=geometry["Zv"],
        ruu=geometry["ruu"],
        ruv=geometry["ruv"],
        rvv=geometry["rvv"],
        zuu=geometry["zuu"],
        zuv=geometry["zuv"],
        zvv=geometry["zvv"],
        basis=basis,
        tables=tables,
        signgs=int(signgs),
        nvper=int(nvper),
        br_add=jnp.asarray(nestor_trace["br_axis"]),
        bp_add=jnp.asarray(nestor_trace["bp_axis"]),
        bz_add=jnp.asarray(nestor_trace["bz_axis"]),
        wint=jnp.asarray(wint),
        include_analytic=bool(include_analytic),
        include_diagnostics=bool(include_diagnostics),
        include_mode_diagnostics=bool(include_mode_diagnostics),
        vac_override=vac_override,
        coil_geometry=coil_geometry,
        nestor_solve_mode=nestor_solve_mode,
        nestor_operator_solver=nestor_operator_solver,
        nestor_operator_tol=nestor_operator_tol,
        nestor_operator_atol=nestor_operator_atol,
        nestor_operator_maxiter=nestor_operator_maxiter,
        nestor_operator_restart=nestor_operator_restart,
    )
