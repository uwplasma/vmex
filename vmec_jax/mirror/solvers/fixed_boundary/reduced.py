"""Reduced fixed-boundary coordinates for mirror solves."""

from __future__ import annotations

import numpy as np

from vmec_jax._compat import jax, jnp

from ...core.boundary import MirrorBoundary
from ...core.grids import MirrorGrid
from ...core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ...core.state import MirrorState3D, MirrorStateAxisym
from ...kernels.constraints import project_axisym_state, project_state_3d
from ...kernels.forces import (
    axisym_energy_value_and_gradient,
    axisym_total_energy_jax,
    energy_value_and_gradient_3d,
)


def _scaling_key(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key in {"none", "identity", "off", "false"}:
        return "none"
    if key in {"geometry", "vmec", "vmec_like", "diagonal"}:
        return "geometry"
    raise ValueError(f"unsupported mirror reduced-coordinate scaling {value!r}")


def _sanitize_scale(scale, *, expected_size: int) -> np.ndarray:
    scale = np.asarray(scale, dtype=float).reshape(-1)
    if scale.size != int(expected_size):
        raise ValueError(f"scale vector has size {scale.size}, expected {int(expected_size)}")
    scale = np.abs(scale)
    scale[(~np.isfinite(scale)) | (scale <= np.finfo(float).tiny)] = 1.0
    return scale


def _require_jax() -> None:
    if jax is None:
        raise RuntimeError("JAX is required for differentiable reduced mirror residuals")


def axisym_reduced_a_mask(grid: MirrorGrid) -> np.ndarray:
    """Return the independent ``a`` nodes for fixed-boundary axisymmetric solves."""
    mask = np.zeros((grid.ns, grid.nxi), dtype=bool)
    if grid.ns > 2 and grid.nxi > 2:
        mask[1:-1, 1:-1] = True
    return mask


def axisym_reduced_coordinate_scale(
    state: MirrorStateAxisym,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    mode: str = "geometry",
) -> np.ndarray:
    """Return the diagonal scale for axisymmetric reduced coordinates.

    The reduced vector contains the interior radius nodes followed by one
    gauge-fixed lambda block per radial surface.  Scaling keeps radius and
    lambda coordinates near comparable sizes before SciPy sees them.
    """
    vector_size = pack_axisym_reduced_state(state, grid, boundary).size
    if _scaling_key(mode) == "none":
        return np.ones(vector_size, dtype=float)

    boundary_radius = np.asarray(boundary.radius_on_grid(grid), dtype=float)
    radius_scale = _sanitize_scale(boundary_radius, expected_size=grid.nxi)
    a_scale = np.broadcast_to(radius_scale[None, :], (grid.ns, grid.nxi))[axisym_reduced_a_mask(grid)]
    lambda_scale = np.full(grid.ns * (grid.nxi - 1), float(np.median(radius_scale)), dtype=float)
    return _sanitize_scale(np.concatenate([a_scale, lambda_scale]), expected_size=vector_size)


def reduced_a_mask_3d(grid: MirrorGrid) -> np.ndarray:
    """Return the independent ``a`` nodes for fixed-boundary 3D solves."""
    mask = np.zeros((grid.ns, grid.ntheta, grid.nxi), dtype=bool)
    if grid.ns > 2 and grid.nxi > 2:
        mask[1:-1, :, 1:-1] = True
    return mask


def reduced_coordinate_scale_3d(
    state: MirrorState3D,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    mode: str = "geometry",
) -> np.ndarray:
    """Return the diagonal reduced-coordinate scale used by 3D mirror L-BFGS-B."""
    vector_size = pack_reduced_state_3d(state, grid, boundary).size
    if _scaling_key(mode) == "none":
        return np.ones(vector_size, dtype=float)

    boundary_radius = np.asarray(boundary.radius_on_grid_3d(grid), dtype=float)
    radius_scale = _sanitize_scale(boundary_radius, expected_size=grid.ntheta * grid.nxi).reshape(
        grid.ntheta,
        grid.nxi,
    )
    a_scale = np.broadcast_to(radius_scale[None, :, :], (grid.ns, grid.ntheta, grid.nxi))[reduced_a_mask_3d(grid)]
    lambda_scale = np.full(grid.ns * (grid.ntheta * grid.nxi - 1), float(np.median(radius_scale)), dtype=float)
    return _sanitize_scale(np.concatenate([a_scale, lambda_scale]), expected_size=vector_size)


def pack_axisym_reduced_state(state: MirrorStateAxisym, grid: MirrorGrid, boundary: MirrorBoundary) -> np.ndarray:
    """Pack independent ``a`` nodes and gauge-fixed ``lambda`` nodes."""
    projected = project_axisym_state(state, grid, boundary)
    a_values = projected.a[axisym_reduced_a_mask(grid)]
    lam_values = np.asarray(projected.lam[:, :-1], dtype=float).ravel()
    return np.concatenate([a_values, lam_values])


def axisym_reduced_bounds(grid: MirrorGrid, *, a_floor: float = 1.0e-10) -> list[tuple[float | None, float | None]]:
    """Return L-BFGS-B bounds for axisymmetric reduced coordinates."""
    num_a = int(np.count_nonzero(axisym_reduced_a_mask(grid)))
    num_lam = grid.ns * (grid.nxi - 1)
    return [(float(a_floor), None)] * num_a + [(None, None)] * num_lam


def scale_reduced_bounds(
    bounds: list[tuple[float | None, float | None]], scale: np.ndarray
) -> list[tuple[float | None, float | None]]:
    """Convert reduced-coordinate bounds into scaled optimizer coordinates."""
    scale = _sanitize_scale(scale, expected_size=len(bounds))
    scaled: list[tuple[float | None, float | None]] = []
    for (lower, upper), item_scale in zip(bounds, scale, strict=True):
        scaled.append(
            (
                None if lower is None else float(lower) / float(item_scale),
                None if upper is None else float(upper) / float(item_scale),
            )
        )
    return scaled


def _scaled_bounds(
    bounds: list[tuple[float | None, float | None]], scale: np.ndarray
) -> list[tuple[float | None, float | None]]:
    return scale_reduced_bounds(bounds, scale)


def unpack_axisym_reduced_state(vector, grid: MirrorGrid, boundary: MirrorBoundary) -> MirrorStateAxisym:
    """Reconstruct a projected axisymmetric state from reduced coordinates."""
    vector = np.asarray(vector, dtype=float)
    mask = axisym_reduced_a_mask(grid)
    num_a = int(np.count_nonzero(mask))
    expected = num_a + grid.ns * (grid.nxi - 1)
    if vector.size != expected:
        raise ValueError(f"reduced vector has size {vector.size}, expected {expected}")

    boundary_radius = boundary.radius_on_grid(grid)
    a = np.broadcast_to(boundary_radius[None, :], (grid.ns, grid.nxi)).copy()
    a[mask] = vector[:num_a]

    lam = np.zeros((grid.ns, grid.nxi), dtype=float)
    lam[:, :-1] = vector[num_a:].reshape(grid.ns, grid.nxi - 1)
    lam[:, -1] = -np.einsum("j,ij->i", grid.w_xi[:-1], lam[:, :-1]) / float(grid.w_xi[-1])
    return project_axisym_state(MirrorStateAxisym(a=a, lam=lam), grid, boundary)


def _unpack_axisym_reduced_state_jax(vector, grid: MirrorGrid, boundary: MirrorBoundary):
    boundary_radius = jnp.asarray(boundary.radius_on_grid(grid), dtype=jnp.asarray(vector).dtype)
    mask_i, mask_j = np.nonzero(axisym_reduced_a_mask(grid))
    num_a = int(mask_i.size)

    a = jnp.broadcast_to(boundary_radius[None, :], (grid.ns, grid.nxi))
    a = a.at[(mask_i, mask_j)].set(vector[:num_a])
    a = a.at[0, :].set(a[1, :])
    a = a.at[0, 0].set(boundary_radius[0])
    a = a.at[0, -1].set(boundary_radius[-1])

    lam_inner = jnp.reshape(vector[num_a:], (grid.ns, grid.nxi - 1))
    w_xi = jnp.asarray(grid.w_xi, dtype=jnp.asarray(vector).dtype)
    lam_last = -jnp.einsum("j,ij->i", w_xi[:-1], lam_inner) / w_xi[-1]
    lam = jnp.concatenate([lam_inner, lam_last[:, None]], axis=1)
    return a, lam


def _axisym_reduced_energy_jax(
    vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    mu0: float,
):
    a, lam = _unpack_axisym_reduced_state_jax(vector, grid, boundary)
    return axisym_total_energy_jax(
        a,
        lam,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )


def _axisym_reduced_objective_jax(
    vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    source_vector=None,
    state_ridge: float = 0.0,
    reference_vector=None,
    mu0: float,
):
    energy = _axisym_reduced_energy_jax(
        vector,
        grid,
        boundary,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    if source_vector is None:
        sourced_energy = energy
    else:
        source_vector = jnp.asarray(source_vector, dtype=jnp.asarray(vector).dtype)
        if source_vector.shape != jnp.asarray(vector).shape:
            raise ValueError(
                f"source_vector shape {source_vector.shape} does not match vector shape {jnp.asarray(vector).shape}"
            )
        sourced_energy = energy - jnp.vdot(source_vector, vector)
    if float(state_ridge) == 0.0:
        return sourced_energy
    if reference_vector is None:
        reference_vector = jnp.zeros_like(vector)
    reference_vector = jnp.asarray(reference_vector, dtype=jnp.asarray(vector).dtype)
    if reference_vector.shape != jnp.asarray(vector).shape:
        raise ValueError(
            f"reference_vector shape {reference_vector.shape} does not match vector shape {jnp.asarray(vector).shape}"
        )
    delta = jnp.asarray(vector) - reference_vector
    return sourced_energy + 0.5 * float(state_ridge) * jnp.vdot(delta, delta)


def axisym_reduced_residual_jax(
    vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    source_vector=None,
    state_ridge: float = 0.0,
    reference_vector=None,
    mu0: float = 4.0e-7 * np.pi,
):
    """Return the differentiable reduced fixed-boundary residual.

    The residual is the gradient of the axisymmetric reduced energy with
    respect to the independent fixed-boundary coordinates.  It is the equation
    ``F(x, p) = 0`` used by implicit differentiation of a converged mirror
    state.
    """
    _require_jax()
    vector = jnp.asarray(vector)

    def objective(items):
        return _axisym_reduced_objective_jax(
            items,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            source_vector=source_vector,
            state_ridge=state_ridge,
            reference_vector=reference_vector,
            mu0=mu0,
        )

    return jax.grad(objective)(vector)


def axisym_reduced_residual_jacobian_jax(
    vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    source_vector=None,
    state_ridge: float = 0.0,
    reference_vector=None,
    derivative: str = "hessian",
    mu0: float = 4.0e-7 * np.pi,
):
    """Return the JAX linearization of the reduced residual.

    ``derivative="hessian"`` uses the fact that the current residual is an
    energy gradient.  ``"forward"`` and ``"reverse"`` are available as explicit
    Jacobian modes for method comparisons before introducing a custom implicit
    derivative rule.
    """
    _require_jax()
    vector = jnp.asarray(vector)
    key = str(derivative).strip().lower().replace("-", "_")

    def objective(items):
        return _axisym_reduced_objective_jax(
            items,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            source_vector=source_vector,
            state_ridge=state_ridge,
            reference_vector=reference_vector,
            mu0=mu0,
        )

    if key in {"hessian", "energy_hessian"}:
        return jax.hessian(objective)(vector)

    def residual(items):
        return jax.grad(objective)(items)

    if key in {"forward", "fwd", "jacfwd"}:
        return jax.jacfwd(residual)(vector)
    if key in {"reverse", "rev", "jacrev"}:
        return jax.jacrev(residual)(vector)
    raise ValueError("derivative must be 'hessian', 'forward', or 'reverse'")


def axisym_reduced_residual_matvec_jax(
    vector,
    direction,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    source_vector=None,
    state_ridge: float = 0.0,
    reference_vector=None,
    transpose: bool = False,
    ridge: float = 0.0,
    mu0: float = 4.0e-7 * np.pi,
):
    """Apply the reduced residual Hessian without forming a dense matrix.

    The current reduced residual is the gradient of a scalar energy, so the
    linearization is symmetric.  ``transpose`` is accepted to match adjoint
    solve call sites; for this energy-Hessian gate it uses the same operator.
    """
    _require_jax()
    vector = jnp.asarray(vector)
    direction = jnp.asarray(direction, dtype=vector.dtype)
    if direction.shape != vector.shape:
        raise ValueError(f"direction shape {direction.shape} does not match vector shape {vector.shape}")

    def residual(items):
        return axisym_reduced_residual_jax(
            items,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            source_vector=source_vector,
            state_ridge=state_ridge,
            reference_vector=reference_vector,
            mu0=mu0,
        )

    del transpose
    _, product = jax.jvp(residual, (vector,), (direction,))
    if float(ridge) != 0.0:
        product = product + float(ridge) * direction
    return product


def axisym_reduced_residual_linear_solve_jax(
    vector,
    rhs,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    source_vector=None,
    state_ridge: float = 0.0,
    reference_vector=None,
    derivative: str = "hessian",
    transpose: bool = False,
    ridge: float = 0.0,
    method: str = "dense",
    cg_tol: float = 1.0e-8,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
    initial_guess=None,
    mu0: float = 4.0e-7 * np.pi,
):
    """Solve a reduced residual linear system for implicit-differentiation gates.

    ``method="dense"`` forms the dense Jacobian and is the tiny-grid correctness
    reference.  ``method="matrix_free_cg"`` uses JAX CG with Hessian-vector
    products and keeps the same forward/transpose call shape for larger
    validation grids.
    """
    key = str(method).strip().lower().replace("-", "_")
    if key in {"matrix_free", "matrix_free_cg", "cg", "jax_cg"}:
        derivative_key = str(derivative).strip().lower().replace("-", "_")
        if derivative_key not in {"hessian", "energy_hessian"}:
            raise ValueError("matrix_free_cg requires derivative='hessian'")
        from jax.scipy.sparse.linalg import cg as jax_cg

        vector = jnp.asarray(vector)
        rhs = jnp.asarray(rhs, dtype=vector.dtype)
        if rhs.shape != vector.shape:
            raise ValueError(f"rhs shape {rhs.shape} does not match vector shape {vector.shape}")
        x0 = None if initial_guess is None else jnp.asarray(initial_guess, dtype=rhs.dtype)
        if x0 is not None and x0.shape != rhs.shape:
            raise ValueError(f"initial_guess shape {x0.shape} does not match rhs shape {rhs.shape}")

        def matvec(direction):
            return axisym_reduced_residual_matvec_jax(
                vector,
                direction,
                grid,
                boundary,
                psi_prime=psi_prime,
                i_prime=i_prime,
                pressure=pressure,
                source_vector=source_vector,
                state_ridge=state_ridge,
                reference_vector=reference_vector,
                transpose=transpose,
                ridge=ridge,
                mu0=mu0,
            )

        solution, _ = jax_cg(
            matvec,
            rhs,
            x0=x0,
            tol=float(cg_tol),
            atol=float(cg_atol),
            maxiter=None if cg_maxiter is None else int(cg_maxiter),
        )
        return solution
    if key not in {"dense", "direct", "dense_direct"}:
        raise ValueError("method must be 'dense' or 'matrix_free_cg'")

    jacobian = axisym_reduced_residual_jacobian_jax(
        vector,
        grid,
        boundary,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        source_vector=source_vector,
        state_ridge=state_ridge,
        reference_vector=reference_vector,
        derivative=derivative,
        mu0=mu0,
    )
    matrix = jacobian.T if bool(transpose) else jacobian
    rhs = jnp.asarray(rhs, dtype=matrix.dtype)
    if float(ridge) != 0.0:
        matrix = matrix + float(ridge) * jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    return jnp.linalg.solve(matrix, rhs)


def axisym_reduced_implicit_state_sensitivity_jax(
    vector,
    residual_parameter_derivative,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    source_vector=None,
    state_ridge: float = 0.0,
    reference_vector=None,
    derivative: str = "hessian",
    ridge: float = 0.0,
    solve_method: str = "dense",
    cg_tol: float = 1.0e-8,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
    initial_guess=None,
    mu0: float = 4.0e-7 * np.pi,
):
    """Return ``dx/dp`` from the implicit equation ``F_x dx/dp = -F_p``."""
    _require_jax()
    vector = jnp.asarray(vector)
    residual_parameter_derivative = jnp.asarray(residual_parameter_derivative, dtype=vector.dtype)
    if residual_parameter_derivative.shape != vector.shape:
        raise ValueError(
            "residual_parameter_derivative shape "
            f"{residual_parameter_derivative.shape} does not match vector shape {vector.shape}"
        )
    return axisym_reduced_residual_linear_solve_jax(
        vector,
        -residual_parameter_derivative,
        grid,
        boundary,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        source_vector=source_vector,
        state_ridge=state_ridge,
        reference_vector=reference_vector,
        derivative=derivative,
        ridge=ridge,
        method=solve_method,
        cg_tol=cg_tol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
        initial_guess=initial_guess,
        mu0=mu0,
    )


def axisym_reduced_implicit_adjoint_jax(
    vector,
    loss_state_gradient,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    source_vector=None,
    state_ridge: float = 0.0,
    reference_vector=None,
    derivative: str = "hessian",
    ridge: float = 0.0,
    solve_method: str = "dense",
    cg_tol: float = 1.0e-8,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
    initial_guess=None,
    mu0: float = 4.0e-7 * np.pi,
):
    """Return the implicit adjoint solving ``F_x.T adjoint = dL/dx``."""
    _require_jax()
    vector = jnp.asarray(vector)
    loss_state_gradient = jnp.asarray(loss_state_gradient, dtype=vector.dtype)
    if loss_state_gradient.shape != vector.shape:
        raise ValueError(
            f"loss_state_gradient shape {loss_state_gradient.shape} does not match vector shape {vector.shape}"
        )
    return axisym_reduced_residual_linear_solve_jax(
        vector,
        loss_state_gradient,
        grid,
        boundary,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        source_vector=source_vector,
        state_ridge=state_ridge,
        reference_vector=reference_vector,
        derivative=derivative,
        transpose=True,
        ridge=ridge,
        method=solve_method,
        cg_tol=cg_tol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
        initial_guess=initial_guess,
        mu0=mu0,
    )


def axisym_reduced_implicit_source_state_jax(
    solved_vector,
    source_vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    state_ridge: float = 0.0,
    reference_vector=None,
    derivative: str = "hessian",
    ridge: float = 0.0,
    solve_method: str = "dense",
    cg_tol: float = 1.0e-8,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
    initial_guess=None,
    mu0: float = 4.0e-7 * np.pi,
):
    """Return a solved reduced state with an implicit VJP for its source.

    ``solved_vector`` is expected to already satisfy
    ``F(solved_vector, source_vector) = 0``.  The primal call returns that
    vector unchanged; reverse-mode differentiation with respect to
    ``source_vector`` uses the adjoint equation
    ``F_x.T adjoint = dL/dx``.  This avoids differentiating through a host-side
    nonlinear optimizer while keeping downstream source-gradient checks in JAX.

    Gradients with respect to ``solved_vector`` are intentionally zero because
    it acts as a cached result of the root solve.  Use
    ``axisym_reduced_implicit_state_sensitivity_jax`` when an explicit forward
    tangent is needed.
    """
    _require_jax()
    solved_vector = jnp.asarray(solved_vector)
    source_vector = jnp.asarray(source_vector, dtype=solved_vector.dtype)
    if source_vector.shape != solved_vector.shape:
        raise ValueError(
            f"source_vector shape {source_vector.shape} does not match solved_vector shape {solved_vector.shape}"
        )

    @jax.custom_vjp
    def solved_state_from_source(root, source):
        del source
        return root

    def solved_state_from_source_fwd(root, source):
        return root, (root, source)

    def solved_state_from_source_bwd(residual_data, cotangent):
        root, source = residual_data
        adjoint = axisym_reduced_implicit_adjoint_jax(
            root,
            cotangent,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            source_vector=source,
            state_ridge=state_ridge,
            reference_vector=reference_vector,
            derivative=derivative,
            ridge=ridge,
            solve_method=solve_method,
            cg_tol=cg_tol,
            cg_atol=cg_atol,
            cg_maxiter=cg_maxiter,
            initial_guess=initial_guess,
            mu0=mu0,
        )
        return jnp.zeros_like(root), adjoint

    solved_state_from_source.defvjp(solved_state_from_source_fwd, solved_state_from_source_bwd)
    return solved_state_from_source(solved_vector, source_vector)


def axisym_reduced_residual_pressure_jacobian_jax(
    vector,
    pressure_coefficients,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure_gamma: float = 5.0 / 3.0,
    source_vector=None,
    state_ridge: float = 0.0,
    reference_vector=None,
    derivative: str = "forward",
    mu0: float = 4.0e-7 * np.pi,
):
    """Return ``dF/dp_coeffs`` for the reduced residual."""
    _require_jax()
    vector = jnp.asarray(vector)
    pressure_coefficients = jnp.asarray(pressure_coefficients, dtype=vector.dtype)
    if pressure_coefficients.ndim != 1 or pressure_coefficients.size < 1:
        raise ValueError("pressure_coefficients must be a nonempty vector")
    key = str(derivative).strip().lower().replace("-", "_")

    def residual_for_pressure(coefficients):
        pressure = PressureProfile(coefficients=coefficients, gamma=float(pressure_gamma))
        return axisym_reduced_residual_jax(
            vector,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            source_vector=source_vector,
            state_ridge=state_ridge,
            reference_vector=reference_vector,
            mu0=mu0,
        )

    if key in {"forward", "fwd", "jacfwd"}:
        return jax.jacfwd(residual_for_pressure)(pressure_coefficients)
    if key in {"reverse", "rev", "jacrev"}:
        return jax.jacrev(residual_for_pressure)(pressure_coefficients)
    raise ValueError("derivative must be 'forward' or 'reverse'")


def axisym_reduced_implicit_pressure_sensitivity_jax(
    vector,
    pressure_coefficients,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure_gamma: float = 5.0 / 3.0,
    source_vector=None,
    state_ridge: float = 0.0,
    reference_vector=None,
    derivative: str = "hessian",
    parameter_derivative: str = "forward",
    ridge: float = 0.0,
    solve_method: str = "dense",
    cg_tol: float = 1.0e-8,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
    mu0: float = 4.0e-7 * np.pi,
):
    """Return ``dx/dp_coeffs`` for pressure-profile coefficients."""
    _require_jax()
    vector = jnp.asarray(vector)
    pressure_coefficients = jnp.asarray(pressure_coefficients, dtype=vector.dtype)
    pressure = PressureProfile(coefficients=pressure_coefficients, gamma=float(pressure_gamma))
    pressure_jacobian = axisym_reduced_residual_pressure_jacobian_jax(
        vector,
        pressure_coefficients,
        grid,
        boundary,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure_gamma=pressure_gamma,
        source_vector=source_vector,
        state_ridge=state_ridge,
        reference_vector=reference_vector,
        derivative=parameter_derivative,
        mu0=mu0,
    )
    columns = []
    for idx in range(int(pressure_coefficients.size)):
        columns.append(
            axisym_reduced_residual_linear_solve_jax(
                vector,
                -pressure_jacobian[:, idx],
                grid,
                boundary,
                psi_prime=psi_prime,
                i_prime=i_prime,
                pressure=pressure,
                source_vector=source_vector,
                state_ridge=state_ridge,
                reference_vector=reference_vector,
                derivative=derivative,
                ridge=ridge,
                method=solve_method,
                cg_tol=cg_tol,
                cg_atol=cg_atol,
                cg_maxiter=cg_maxiter,
                mu0=mu0,
            )
        )
    return jnp.stack(columns, axis=1)


def axisym_reduced_implicit_pressure_state_jax(
    solved_vector,
    pressure_coefficients,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure_gamma: float = 5.0 / 3.0,
    source_vector=None,
    state_ridge: float = 0.0,
    reference_vector=None,
    derivative: str = "hessian",
    ridge: float = 0.0,
    solve_method: str = "dense",
    cg_tol: float = 1.0e-8,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
    initial_guess=None,
    mu0: float = 4.0e-7 * np.pi,
):
    """Return a solved reduced state with an implicit VJP for pressure coefficients.

    As with ``axisym_reduced_implicit_source_state_jax``, the primal value is a
    cached converged reduced state.  The reverse pass differentiates a scalar
    objective with respect to pressure polynomial coefficients using
    ``-adjoint.T @ dF/dp_coeffs``.
    """
    _require_jax()
    solved_vector = jnp.asarray(solved_vector)
    pressure_coefficients = jnp.asarray(pressure_coefficients, dtype=solved_vector.dtype)
    if pressure_coefficients.ndim != 1 or pressure_coefficients.size < 1:
        raise ValueError("pressure_coefficients must be a nonempty vector")

    @jax.custom_vjp
    def solved_state_from_pressure(root, coefficients):
        del coefficients
        return root

    def solved_state_from_pressure_fwd(root, coefficients):
        return root, (root, coefficients)

    def solved_state_from_pressure_bwd(residual_data, cotangent):
        root, coefficients = residual_data
        pressure = PressureProfile(coefficients=coefficients, gamma=float(pressure_gamma))
        adjoint = axisym_reduced_implicit_adjoint_jax(
            root,
            cotangent,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            source_vector=source_vector,
            state_ridge=state_ridge,
            reference_vector=reference_vector,
            derivative=derivative,
            ridge=ridge,
            solve_method=solve_method,
            cg_tol=cg_tol,
            cg_atol=cg_atol,
            cg_maxiter=cg_maxiter,
            initial_guess=initial_guess,
            mu0=mu0,
        )

        def residual_for_pressure(items):
            pressure_for_items = PressureProfile(coefficients=items, gamma=float(pressure_gamma))
            return axisym_reduced_residual_jax(
                root,
                grid,
                boundary,
                psi_prime=psi_prime,
                i_prime=i_prime,
                pressure=pressure_for_items,
                source_vector=source_vector,
                state_ridge=state_ridge,
                reference_vector=reference_vector,
                mu0=mu0,
            )

        _, pullback = jax.vjp(residual_for_pressure, coefficients)
        pressure_bar = -pullback(adjoint)[0]
        return jnp.zeros_like(root), pressure_bar

    solved_state_from_pressure.defvjp(solved_state_from_pressure_fwd, solved_state_from_pressure_bwd)
    return solved_state_from_pressure(solved_vector, pressure_coefficients)


def pack_reduced_state_3d(state: MirrorState3D, grid: MirrorGrid, boundary: MirrorBoundary) -> np.ndarray:
    """Pack independent 3D ``a`` nodes and gauge-fixed ``lambda`` nodes."""
    projected = project_state_3d(state, grid, boundary)
    a_values = projected.a[reduced_a_mask_3d(grid)]
    lam_values = np.asarray(projected.lam[:, :, :], dtype=float).reshape(grid.ns, -1)[:, :-1].ravel()
    return np.concatenate([a_values, lam_values])


def reduced_bounds_3d(grid: MirrorGrid, *, a_floor: float = 1.0e-10) -> list[tuple[float | None, float | None]]:
    """Return L-BFGS-B bounds for 3D reduced coordinates."""
    num_a = int(np.count_nonzero(reduced_a_mask_3d(grid)))
    num_lam = grid.ns * (grid.ntheta * grid.nxi - 1)
    return [(float(a_floor), None)] * num_a + [(None, None)] * num_lam


def unpack_reduced_state_3d(vector, grid: MirrorGrid, boundary: MirrorBoundary) -> MirrorState3D:
    """Reconstruct a projected 3D state from reduced coordinates."""
    vector = np.asarray(vector, dtype=float)
    mask = reduced_a_mask_3d(grid)
    num_a = int(np.count_nonzero(mask))
    num_lam_surface = grid.ntheta * grid.nxi - 1
    expected = num_a + grid.ns * num_lam_surface
    if vector.size != expected:
        raise ValueError(f"reduced vector has size {vector.size}, expected {expected}")

    boundary_radius = boundary.radius_on_grid_3d(grid)
    a = np.broadcast_to(boundary_radius[None, :, :], (grid.ns, grid.ntheta, grid.nxi)).copy()
    a[mask] = vector[:num_a]

    lam = np.zeros((grid.ns, grid.ntheta * grid.nxi), dtype=float)
    lam[:, :-1] = vector[num_a:].reshape(grid.ns, num_lam_surface)
    flat_weights = (grid.w_theta[:, None] * grid.w_xi[None, :]).ravel()
    lam[:, -1] = -np.einsum("j,ij->i", flat_weights[:-1], lam[:, :-1]) / float(flat_weights[-1])
    lam = lam.reshape(grid.ns, grid.ntheta, grid.nxi)
    return project_state_3d(MirrorState3D(a=a, lam=lam), grid, boundary)


def pack_axisym_reduced_gradient_components(grad_a, grad_lam, grid: MirrorGrid) -> np.ndarray:
    """Pack full-state axisymmetric gradients into reduced fixed-boundary coordinates."""
    mask = axisym_reduced_a_mask(grid)
    grad_a = np.asarray(grad_a, dtype=float).copy()
    if grid.ns > 2:
        grad_a[1, :] += grad_a[0, :]
    a_values = grad_a[mask]

    grad_lam = np.asarray(grad_lam, dtype=float)
    lam_values = grad_lam[:, :-1] - (grid.w_xi[:-1] / grid.w_xi[-1])[None, :] * grad_lam[:, -1:]
    return np.concatenate([a_values, lam_values.ravel()])


def _pack_axisym_reduced_gradient(gradient, grid: MirrorGrid) -> np.ndarray:
    return pack_axisym_reduced_gradient_components(gradient.grad_a, gradient.grad_lam, grid)


def _pack_reduced_gradient_3d(gradient, grid: MirrorGrid) -> np.ndarray:
    mask = reduced_a_mask_3d(grid)
    grad_a = np.asarray(gradient.grad_a, dtype=float).copy()
    if grid.ns > 2:
        grad_a[1, :, :] += grad_a[0, :, :]
    a_values = grad_a[mask]

    grad_lam = np.asarray(gradient.grad_lam, dtype=float).reshape(grid.ns, -1)
    flat_weights = (grid.w_theta[:, None] * grid.w_xi[None, :]).ravel()
    lam_values = grad_lam[:, :-1] - (flat_weights[:-1] / flat_weights[-1])[None, :] * grad_lam[:, -1:]
    return np.concatenate([a_values, lam_values.ravel()])


def reduced_axisym_energy_and_gradient(
    vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    source_a=None,
    source_lam=None,
    mu0: float = 4.0e-7 * np.pi,
) -> tuple[float, np.ndarray]:
    """Return energy and exact reduced-coordinate gradient."""
    state = unpack_axisym_reduced_state(vector, grid, boundary)
    gradient = axisym_energy_value_and_gradient(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    energy = float(gradient.energy)
    grad_a = np.asarray(gradient.grad_a, dtype=float)
    grad_lam = np.asarray(gradient.grad_lam, dtype=float)
    if source_a is not None:
        source_a = np.asarray(source_a, dtype=float)
        if source_a.shape != state.a.shape:
            raise ValueError(f"source_a shape {source_a.shape} does not match state shape {state.a.shape}")
        energy -= float(np.sum(source_a * state.a))
        grad_a = grad_a - source_a
    if source_lam is not None:
        source_lam = np.asarray(source_lam, dtype=float)
        if source_lam.shape != state.lam.shape:
            raise ValueError(f"source_lam shape {source_lam.shape} does not match state shape {state.lam.shape}")
        energy -= float(np.sum(source_lam * state.lam))
        grad_lam = grad_lam - source_lam
    return energy, pack_axisym_reduced_gradient_components(grad_a, grad_lam, grid)


def reduced_3d_energy_and_gradient(
    vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    mu0: float = 4.0e-7 * np.pi,
) -> tuple[float, np.ndarray]:
    """Return 3D energy and exact reduced-coordinate gradient."""
    state = unpack_reduced_state_3d(vector, grid, boundary)
    gradient = energy_value_and_gradient_3d(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    return gradient.energy, _pack_reduced_gradient_3d(gradient, grid)
