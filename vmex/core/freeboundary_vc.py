"""Free-boundary equilibria from the three plasma-vacuum interface conditions.

At a free boundary ``S`` without a sheet current the interface conditions are
(Conlin et al. 2024, arXiv:2412.05680)

1. ``B_out . n = 0``,
2. ``|B_out|^2 = |B_in|^2 + 2 mu0 p``,
3. ``n x (B_out - B_in) = mu0 K = 0``,

with ``B_out = B_coil + B_plasma`` and ``B_plasma`` the plasma's own field from
the virtual-casing principle (:mod:`vmex.core.virtual_casing`).  NESTOR
(:mod:`vmex.core.freeboundary`) builds the vacuum field that satisfies (1) by
construction and balances only ``|B|`` in (2), so it accepts any tangential
field jump: a VMEC/NESTOR equilibrium may carry an edge sheet current.  Here
the boundary is the unknown instead.  Every trial boundary is a fixed-boundary
equilibrium, and a Gauss-Newton least-squares solve minimizes the stacked
residual of all three conditions with exact implicit derivatives
(:class:`ThreeTermFreeBoundaryModel`).  At an exact solution all three vanish
together; at finite resolution they stop at a floor set mainly by ``mpol``
(about 1e-3 relative at ``mpol = 5``, 5e-4 at 7), where the relative
``weights`` decide the balance.

Condition (1) alone leaves one direction open: the flux of ``B_out`` through
the closed surface vanishes identically, and a mismatch of the net poloidal
current ``G`` adds a field tangent to ``S``.  Conditions (2) and (3) close it in
principle; numerically a row ``(G_plasma - G_coil) / G_coil`` closes it directly
(``net_current_weight``).  With ``p = 0`` on the boundary a solution satisfies
all three exactly; a nonzero edge pressure needs a sheet current and is
rejected.

Entry points, from the most to the least packaged:

- a free-boundary deck with ``!@VMEX BOUNDARY_CONDITION = THREE_TERM``
  (``vmex input.case``, :func:`~vmex.core.multigrid.solve_file`), the CLI flag
  ``--boundary-condition three-term``, or
  ``solve_free_boundary_multigrid(..., boundary_condition="three_term")``;
- :func:`solve_free_boundary_three_term`: one solve with every control, and
  ``previous=`` to restart from an earlier result after the coils change;
- ``FreeBoundaryProblem.from_loss(..., boundary_condition="three_term")``
  (:class:`ThreeTermFreeBoundaryProblem`): single-stage optimization, a loss
  and constraint rows of coils and plasma parameters at their free boundary,
  for :func:`vmex.core.optimize.minimize`;
- :class:`ThreeTermFreeBoundaryModel`: the residual, its Jacobian, state
  tangents and pullbacks for one deck and any external field or plasma
  parameters, without recompiling -- the pieces an optimizer needs;
- :func:`boundary_residual`: the three conditions on any equilibrium, e.g. to
  check a NESTOR result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import time
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from . import virtual_casing as vc
from .errors import TrialRejected, VmecConvergenceError, VmecError
from .input import VmecInput
from .problem import FunctionProblem


@dataclass(frozen=True)
class BoundaryResidual:
    """Area-weighted RMS of the three interface conditions, relative to ``|B_in|``.

    ``normal`` is ``B_out . n / |B_in|``, ``pressure`` is
    ``(|B_out|^2 - |B_in|^2 - 2 mu0 p) / (2 |B_in|^2)`` and ``sheet_current``
    is ``|n x (B_out - B_in)| / |B_in| = mu0 |K| / |B_in|``.
    """

    normal: float
    pressure: float
    sheet_current: float


def _fingerprint(*trees) -> str:
    """Hash of the concrete arrays of ``trees`` (cache keys of linearizations)."""
    digest = hashlib.sha1()
    for leaf in jax.tree.leaves(trees):
        digest.update(np.asarray(leaf).tobytes())
    return digest.hexdigest()


def _interface_terms(inp, state, external_field, *, runtime, nphi, ntheta, digits, precision, p_edge):
    """Rows and weights of :func:`boundary_residual`, plus the boundary points ``(3, nphi, ntheta)``."""
    data = vc.surface_field_data_from_state(inp, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
    interface = vc.PlasmaVacuumInterface.from_surface_data(
        data, p_edge=p_edge, digits=digits, precision=precision)
    B_in = jnp.asarray(data.B_total)
    B_out = interface.total_B_out(external_field)
    B2 = interface.Bin_mag2
    scale = jnp.sqrt(B2)
    normal = jnp.sum(B_out * interface.normal, axis=0) / scale
    pressure = interface.pressure_balance_residual(external_field) / (2.0 * B2)
    sheet = jnp.cross(interface.normal, B_out - B_in, axis=0) / scale
    return jnp.concatenate([normal[None], pressure[None], sheet]), interface.weights, interface.gamma


def boundary_residual(inp: VmecInput, state, external_field: Any, *, runtime=None,
                      nphi: int, ntheta: int, digits: int = 4, precision=None,
                      p_edge: float = 0.0):
    """Pointwise interface conditions on the boundary of a live ``state``.

    Returns ``(rows, weights)``: ``rows`` has shape ``(5, nphi, ntheta)`` with
    ``B_out . n``, the pressure-balance jump and the three Cartesian components
    of ``n x (B_out - B_in)``, each scaled as in :class:`BoundaryResidual`;
    ``weights`` are the area weights (summing to one).  Traceable in ``state``
    and the external-field parameters when ``precision`` is supplied (see
    :func:`~vmex.core.virtual_casing.plan_vc_precision`).
    """
    rows, weights, _ = _interface_terms(inp, state, external_field, runtime=runtime, nphi=nphi,
                                        ntheta=ntheta, digits=digits, precision=precision, p_edge=p_edge)
    return rows, weights


def summarize_boundary_residual(rows, weights) -> BoundaryResidual:
    """RMS of each condition from :func:`boundary_residual`'s output."""
    rows, weights = np.asarray(rows), np.asarray(weights)

    def rms(value):
        return float(np.sqrt(np.sum(weights * value**2)))

    return BoundaryResidual(normal=rms(rows[0]), pressure=rms(rows[1]),
                            sheet_current=rms(np.linalg.norm(rows[2:], axis=0)))


def _edge_pressure(inp: VmecInput) -> tuple[float, float]:
    """Edge and peak pressure of ``inp`` in Pa."""
    from .profiles import pressure

    p = np.asarray(pressure(inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f, np.linspace(0.0, 1.0, 101),
                            pres_scale=inp.pres_scale, bloat=inp.bloat, spres_ped=inp.spres_ped))
    return float(p[-1]), float(np.max(np.abs(p)))


def _axis_loop(wout):
    """Points ``(3, n)``, tangent and spacing of the magnetic axis of ``wout``: a closed curve inside the coils."""
    phi = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    harmonics = np.arange(np.asarray(wout.raxis_cc).size) * int(wout.nfp)
    R = np.cos(np.outer(phi, harmonics)) @ np.asarray(wout.raxis_cc, dtype=float)
    Z = -np.sin(np.outer(phi, harmonics)) @ np.asarray(wout.zaxis_cs, dtype=float)
    points = np.stack([R * np.cos(phi), R * np.sin(phi), Z])
    return jnp.asarray(points), jnp.asarray(np.gradient(points, phi, axis=1)), float(phi[1] - phi[0])


def _net_current(field, points, tangent, dphi):
    """Traceable ``G = (1/2 pi)`` circulation of ``field`` along a closed curve (:func:`_axis_loop`)."""
    B = vc.external_B_cartesian(field, points[:, :, None])[:, :, 0]
    return jnp.sum(B * tangent) * dphi / (2.0 * jnp.pi)


def _coil_net_current(external_field, wout) -> float:
    """``G = (1/2 pi)`` times the external field's circulation around the magnetic axis of ``wout``."""
    return float(_net_current(external_field, *_axis_loop(wout)))


def solve_free_boundary_three_term(
    inp: VmecInput,
    *,
    external_field: Any = None,
    mgrid_path=None,
    initial_boundary: VmecInput | None = None,
    max_mode: int | None = None,
    nphi: int | None = None,
    ntheta: int | None = None,
    digits: int = 4,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    net_current_weight: float = 1.0,
    label_weight: float = 1e-2,
    ftol: float = 1e-4,
    jacobian_ftol: float | None = 1e-2,
    max_nfev: int = 60,
    quadrature: tuple[int, int] | None = None,
    trial_ftol: float | None = None,
    chunk: int = 8,
    bootstrap=None,
    bootstrap_helicity: int = 0,
    bootstrap_weight: float = 100.0,
    previous=None,
    verbose: int = 0,
):
    """Free-boundary equilibrium of ``inp`` from the three interface conditions.

    ``inp`` is a free-boundary deck (``PHIEDGE``, profiles, and the external
    field through ``external_field``, ``mgrid_path`` or ``MGRID_FILE``/``EXTCUR``);
    its boundary, or that of ``initial_boundary``, is the initial guess, so a
    fixed-boundary design and the coils fitted to it make a natural start.  The
    boundary Fourier coefficients up to ``max_mode`` (default: all) and
    ``RBC(0,0)`` are solved for at fixed toroidal flux, pressure and current or
    iota, every trial being a fixed-boundary equilibrium on the deck's
    ``NS_ARRAY`` ladder.  ``external_field`` is anything
    :func:`~vmex.core.virtual_casing.external_B_cartesian` accepts (an
    :class:`~vmex.core.mgrid.MgridField`, a ``xyz -> B`` callable such as an
    ESSOS Biot-Savart field, or a coil field with ``b_cyl``).

    The residual is ``sqrt(area weight)`` times ``weights = (normal, pressure,
    sheet current)`` times the rows of :func:`boundary_residual` on an
    ``nphi x ntheta`` grid over one field period (default 48 x 48), with the
    virtual-casing quadrature planned once on the initial boundary to
    ``digits``.  The grid, not ``digits``, limits the virtual-casing field:
    for a current-free state it returns ``|B_plasma| / |B|`` of about 2e-4 at
    48 x 48 and 5e-5 at 64 x 64 (whose quadrature needs several times the
    memory), below the residual floor that ``mpol`` sets (module docstring).
    Rows of the boundary points' displacement along the surface in ``theta``,
    relative to the initial boundary and scaled by ``label_weight``, fix the
    boundary's poloidal labelling, which the interface conditions leave free.
    ``net_current_weight`` scales the row ``(G_plasma - G_coil) / G_coil`` of
    the net poloidal current (edge ``rbtor`` against the external field's
    circulation around the seed's magnetic axis), the direction the
    normal-field rows cannot see.

    A trust region with exact implicit Jacobians
    (:class:`ThreeTermFreeBoundaryModel`) runs to a relative cost change of
    ``jacobian_ftol``; Levenberg-Marquardt steps with its last Jacobian then go
    on to ``ftol``, the floor set by the resolution (``None``: the trust region
    runs to ``ftol``).  ``previous``, an earlier result for the same deck
    (typically before a coil change), starts from its boundary with those steps
    and its Jacobian, and from its model: neither new coils nor a repeat solve
    compile anything.  ``quadrature``, ``trial_ftol`` and ``chunk`` are
    :class:`ThreeTermFreeBoundaryModel`'s (fixed singular quadrature, looser
    trial equilibria, Jacobian columns per batch).
    ``bootstrap`` (kinetic profiles), ``bootstrap_helicity`` and
    ``bootstrap_weight`` solve for a Redl-self-consistent current profile
    together with the boundary (:class:`ThreeTermFreeBoundaryModel`).

    Returns a :class:`scipy.optimize.OptimizeResult` with ``x`` (the boundary
    coordinates, then any bootstrap-current values), ``fun`` (the residual),
    ``jac`` (its last Jacobian), ``cost``, ``nfev``, ``njev``, ``input`` (the
    free boundary, and the current, as a fixed-boundary deck),
    ``equilibrium``, ``model``, ``aux`` (``(state, mask, params, tight)`` of
    the solution, for ``previous=``), ``boundary_residual`` and
    ``initial_boundary_residual`` (:class:`BoundaryResidual` at the start).
    """
    from scipy.optimize import OptimizeResult

    from .freeboundary import _external_field_from_input
    from .implicit import input_with_params
    from .optimize import solve_equilibrium

    if bool(inp.lasym):
        raise NotImplementedError("the virtual-casing free boundary supports lasym = False only")
    p_edge, p_max = _edge_pressure(inp)
    if abs(p_edge) > 1e-8 * p_max:
        raise ValueError(f"edge pressure is {p_edge:g} Pa: without a sheet current the "
                         "boundary pressure must vanish")
    if external_field is None:
        external_field = _external_field_from_input(inp, mgrid_path)
    if previous is not None:
        model, x0, jacobian = previous.model, previous.x, previous.jac
        seed_state, seed_params = previous.aux[0], previous.aux[2]
    else:
        start = inp if initial_boundary is None else replace(
            inp, rbc=initial_boundary.rbc, zbs=initial_boundary.zbs,
            raxis_c=initial_boundary.raxis_c, zaxis_s=initial_boundary.zaxis_s)
        model = ThreeTermFreeBoundaryModel(
            start, max_mode=max_mode, nphi=int(nphi or 48), ntheta=int(ntheta or 48), digits=digits,
            weights=weights, net_current_weight=net_current_weight, label_weight=label_weight,
            quadrature=quadrature, trial_ftol=trial_ftol, chunk=chunk, bootstrap=bootstrap,
            bootstrap_helicity=bootstrap_helicity, bootstrap_weight=bootstrap_weight)
        x0 = jacobian = None
        seed_state, seed_params = model.seed[0], model.params0
    initial = model.boundary_residual(seed_state, seed_params, external_field)
    out = model.solve_boundary(model.params0, external_field, x0=x0, jacobian=jacobian, ftol=ftol,
                               jacobian_ftol=ftol if jacobian_ftol is None else jacobian_ftol,
                               max_nfev=max_nfev, verbose=verbose)
    state, _, params, _ = out["aux"]
    boundary = input_with_params(model.fixed, params)
    return OptimizeResult(
        x=out["x"], fun=out["rows"], jac=out["jacobian"], cost=0.5 * out["rows"] @ out["rows"], nfev=out["nfev"],
        njev=out["njev"], success=True, message=f"{out['accepted']} Levenberg-Marquardt steps after the trust region",
        input=boundary, equilibrium=solve_equilibrium(boundary, initial_state=state), model=model, aux=out["aux"],
        boundary_residual=model.boundary_residual(state, params, external_field), initial_boundary_residual=initial)


class ThreeTermFreeBoundaryModel:
    """The free boundary of one deck for many external fields and plasma parameters.

    The fixed-boundary equilibria and their derivatives come from
    :mod:`vmex.core.implicit`, with the boundary, PHIEDGE and current profile
    as traced :class:`~vmex.core.implicit.ImplicitParams`, and the external
    field enters the compiled interface rows as an argument: neither new coils
    nor new plasma parameters recompile anything, which is what an optimizer
    that solves the free boundary at every step needs.

    Coordinates: ``x`` are the boundary coefficients up to ``max_mode`` and
    ``RBC(0,0)`` (:func:`~vmex.core.optimize.pack_boundary`; ``x0`` the
    deck's, ``x_scale`` their ESS scales), ``params`` an ``ImplicitParams``
    (``params0`` the deck's) whose boundary ``with_boundary(params, x)``
    replaces.  The rows are those of :func:`solve_free_boundary_three_term`,
    with the quadrature plan and the labelling reference fixed on the deck's
    boundary (``seed``: its ``(state, mask)``).

    Methods: :meth:`solve_boundary` (the K = 0 boundary, cold or warm),
    :meth:`evaluate` (rows at a boundary), :meth:`linearize` (their Jacobian
    and the state responses), :meth:`push` (forward-mode columns of other
    functions of the equilibrium), :meth:`pullback` (their reverse-mode
    gradients), :meth:`boundary_residual`, :meth:`bootstrap_residual` and
    :meth:`solve`; ``rows(state, params, field)`` and ``rows_at(state,
    runtime, field)`` are the traceable rows themselves.

    ``trial_ftol`` solves the trial equilibria of the boundary steps to a looser
    force residual (each is re-solved to the deck's before it is
    differentiated); ``quadrature = (quad_nt, quad_np)`` fixes the singular
    quadrature instead of planning it to ``digits`` (``quad_nt`` a multiple of
    ``nfp * nphi``), whose error estimate alone can exhaust a GPU on a strongly
    shaped boundary.  Use ``(4 nfp nphi, 2 ntheta)``: with ``quad_np = ntheta``
    the tangential plasma field on a vacuum surface is off by ~3e-4 ``|B|``, the
    fit absorbs it in the boundary, and its near-resonant B.n harmonics grow
    several-fold (8 x 8 vacuum benchmark: field lines leave the LCFS by 7 mm
    instead of 1 mm), at no extra memory.  ``chunk`` bounds the Jacobian
    columns per batch.

    ``bootstrap`` (:class:`~vmex.core.bootstrap.KineticProfiles`, with
    ``bootstrap_helicity`` the quasisymmetry ``N`` of the Redl model) makes the
    current profile Redl's bootstrap current in the same solve: ``x`` gains its
    knot values (three inside the first surface, then one on every half-mesh
    surface) after the boundary coordinates (``n_boundary`` of them), and the
    rows gain one self-consistency row per knot, scaled by
    ``bootstrap_weight`` (large by default: the current is no interface
    condition to trade against the others, so its block is solved far below
    their floor; see :class:`~vmex.core.bootstrap.HalfMeshCurrent`;
    ``ns <= 97``, a prescribed-current deck).  The current then follows the
    equilibrium instead of being an input, and :meth:`bootstrap_residual`
    reports how far it is from Redl's.  ``fixed_boundary=True`` (with
    ``bootstrap``) holds the boundary at that of ``params`` and solves only the
    current: ``x`` is the current values alone, the rows are the bootstrap ones
    and no external field or virtual casing enters -- the fixed-boundary
    single stage's equilibrium, with the same Jacobians and pullbacks.
    """

    def __init__(self, inp: VmecInput, *, max_mode=None, nphi=48, ntheta=48, digits=4,
                 weights=(1.0, 1.0, 1.0), net_current_weight=1.0, label_weight=1e-2, chunk=8, device=None,
                 trial_ftol=None, quadrature=None, bootstrap=None, bootstrap_helicity=0, bootstrap_weight=100.0,
                 fixed_boundary=False):
        from . import implicit as im
        from .bootstrap import HalfMeshCurrent
        from .optimize import _ess_scale, boundary_arrays_from_x, pack_boundary

        self._im = im
        self.fixed = fixed = replace(inp, lfreeb=False, mgrid_file="NONE")
        self.bootstrap = None if bootstrap is None else HalfMeshCurrent(fixed, bootstrap, bootstrap_helicity,
                                                                       bootstrap_weight)
        self.fixed_boundary = bool(fixed_boundary)
        if self.fixed_boundary and self.bootstrap is None:
            raise ValueError("fixed_boundary=True solves only the bootstrap current: pass bootstrap=")
        if self.bootstrap is not None:
            self.fixed = fixed = self.bootstrap.deck
        self.max_mode = int(max(fixed.mpol - 1, fixed.ntor) if max_mode is None else max_mode)
        device = jax.devices()[0] if device is None else device
        cfg = im.make_config(fixed, multigrid=True, hot_restart=True)
        self.cfg = cfg = im._canonical_config(replace(cfg, device=device))
        # Trial points (the boundary steps) may converge to a looser force residual: their rows move by far less
        # than the interface floor, and a state is solved to the deck's tolerance before it is differentiated.
        self.cfg_trial = cfg
        if trial_ftol is not None:
            ftols = np.array(fixed.ftol_array, dtype=float)
            ftols[-1] = max(float(trial_ftol), ftols[-1])
            loose = im.make_config(replace(fixed, ftol_array=ftols), multigrid=True, hot_restart=True)
            self.cfg_trial = im._canonical_config(replace(loose, device=device))
            im._template_runtime(self.cfg_trial)
        self.params0 = im.params_from_input(fixed, device=device)
        im._template_runtime(cfg)
        self.x0 = pack_boundary(fixed, self.max_mode, vary_major_radius=True)
        self.x_scale = _ess_scale(fixed, self.max_mode, 1.2, vary_major_radius=True)
        if self.fixed_boundary:
            self.x0, self.x_scale = self.x0[:0], self.x_scale[:0]
        self.n_boundary = nb = self.x0.size
        if self.bootstrap is not None:
            self.x0 = np.r_[self.x0, self.bootstrap.x0]
            self.x_scale = np.r_[self.x_scale, self.bootstrap.x_scale]
        self.chunk = int(chunk)
        self._linearization = None
        self._last_linearize = None  # (key, extra key, result, _linearization) of the latest linearize
        self._radius = None  # the boundary steps' trust radius, carried from one solve to the next

        def with_boundary(params, x):
            if not self.fixed_boundary:
                rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], self.max_mode, vary_major_radius=True)
                params = replace(params, rbc=rbc, zbs=zbs)
            return params if self.bootstrap is None else self.bootstrap.apply(params, x[nb:])

        self.with_boundary = with_boundary
        seed = self.solve(self.params0)
        if seed is None:
            raise VmecConvergenceError("the initial boundary has no converged fixed-boundary equilibrium")
        state, mask = seed[:2]
        self.seed = (state, mask)
        if not self.fixed_boundary:
            self._interface_setup(state, nphi, ntheta, digits, quadrature, weights, label_weight, net_current_weight)
        self._implicit_setup()

    def _interface_setup(self, state, nphi, ntheta, digits, quadrature, weights, label_weight, net_current_weight):
        """The interface rows, with the quadrature plan and labelling reference of the seed ``state``."""
        from .freeboundary import _vacuum_scalars
        from .optimize import solve_equilibrium

        im, cfg, fixed = self._im, self.cfg, self.fixed
        runtime = im.runtime_from_params(self.params0, cfg)
        surface = vc.surface_field_data_from_state(fixed, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
        quad_nt, quad_np = (None, None) if quadrature is None else map(int, quadrature)
        precision = vc.plan_vc_precision(surface, digits=digits, quad_nt=quad_nt, quad_np=quad_np)
        gamma0 = jnp.asarray(surface.gamma)
        wavenumber = jnp.fft.fftfreq(ntheta, 1.0 / ntheta)
        e_theta = jnp.real(jnp.fft.ifft(1j * wavenumber * jnp.fft.fft(gamma0, axis=-1), axis=-1))
        length = jnp.linalg.norm(e_theta, axis=0)
        tangent = e_theta / (length * jnp.mean(length))[None]
        scale = jnp.repeat(jnp.asarray(weights, dtype=float), jnp.asarray([1, 1, 3]))[:, None, None]
        # A closed curve inside the coils for G_coil: the seed's magnetic axis.
        axis = _axis_loop(solve_equilibrium(fixed, initial_state=state).wout)

        def rows_at(state, runtime, field):
            values, area, gamma = _interface_terms(fixed, state, field, runtime=runtime, nphi=nphi, ntheta=ntheta,
                                                   digits=digits, precision=precision, p_edge=0.0)
            label = label_weight * jnp.sum((gamma - gamma0) * tangent, axis=0)
            interface = (jnp.sqrt(area)[None] * jnp.concatenate([scale * values, label[None]])).ravel()
            if not net_current_weight:
                return interface
            G_coil = jnp.abs(_net_current(field, *axis))
            G = jnp.abs(_vacuum_scalars(state, runtime)[1])
            return jnp.concatenate([interface, (net_current_weight * (G - G_coil) / G_coil)[None]])

        self.rows_at = rows_at
        self._terms = jax.jit(lambda state, params, field: _interface_terms(
            fixed, state, field, runtime=im.runtime_from_params(params, cfg), nphi=nphi, ntheta=ntheta, digits=digits,
            precision=precision, p_edge=0.0)[:2])

    def _implicit_setup(self):
        """The rows (interface, then bootstrap) and their implicit tangents, pushes and pullbacks."""
        im, cfg = self._im, self.cfg

        def rows(state, params, field):
            runtime = im.runtime_from_params(params, cfg)
            parts = [] if self.fixed_boundary else [self.rows_at(state, runtime, field)]
            if self.bootstrap is not None:
                parts.append(self.bootstrap.rows(state, runtime, params))
            return jnp.concatenate(parts)

        self.rows = rows
        self._rows = jax.jit(rows)
        active = im._active_state_fields(cfg)
        edge = im._edge_mask(cfg)

        def tangents(params, state, mask, batch):
            """Projected state responses to the stacked parameter tangents: one block factorization, chunked."""
            return im._implicit_evolved_tangent_multi_rhs(
                params, cfg, jax.lax.stop_gradient(state), mask, batch, active_fields=active,
                probe_chunk_size=self.chunk, response_chunk_size=self.chunk,
                certify_rtol=float(cfg.jacobian_adjoint_tol), certify_maxiter=int(cfg.jacobian_adjoint_maxiter))[0]

        def push(fun, state, mask, params, dz_batch, params_batch):
            """Columns of ``fun(state, params)`` along the stacked responses, ``chunk`` at a time.

            As upstream's implicit Jacobian: the state of a column is assembled from its projected response and
            its parameters inside the map, so no batch of full states is ever built.
            """
            frozen = jax.lax.stop_gradient(state)
            P = im._dof_projector(cfg, mask)

            def at(z, prm):
                return fun(im._assemble(z, im.runtime_from_params(prm, cfg), frozen, P, edge), prm)

            return jax.lax.map(lambda a: jax.jvp(at, (P(frozen), params), (P(a[0]), a[1]))[1],
                               (dz_batch, params_batch), batch_size=self.chunk)

        def predict(state, mask, dz_batch, dx, params):
            """First-order trial state: the anchor plus the responses times ``dx``, on the trial boundary."""
            frozen = jax.lax.stop_gradient(state)
            P = im._dof_projector(cfg, mask)
            z = jax.tree.map(lambda a, d: a + jnp.tensordot(dx, d, axes=1), P(frozen), dz_batch)
            return im._assemble(z, im.runtime_from_params(params, cfg), frozen, P, edge)

        self._tangents = jax.jit(tangents)
        self.push = push
        self._push_rows = jax.jit(lambda state, mask, params, dz, pb, field: push(
            lambda s, p: rows(s, p, field), state, mask, params, dz, pb))
        # First-order trial state from the last linearization (upstream's perturbation warm start): it keeps
        # the trial on the path the Jacobian was taken along, instead of a hot restart that may drift.
        self._predict = jax.jit(predict)

        def pullback(fun, state, mask, params, cotangents, *args):
            """``ImplicitParams`` gradients of ``cotangents[i] . fun(state, params, *args)`` through the equilibrium.

            Reverse mode with one block factorization shared by every row (upstream's implicit pullback): for a few
            scalar rows of an expensive function this costs a few reverse passes, not one forward pass per
            parameter direction.  ``fun`` may be a tuple of functions with ``cotangents`` a tuple of their rows'
            cotangents: each row then runs only its own function's reverse pass, and the rows come out in order.
            """
            frozen = jax.lax.stop_gradient(state)
            back = im._block_state_pullback(params, cfg, frozen, mask, active_fields=active,
                                            probe_chunk_size=self.chunk)
            funs, blocks = (fun, cotangents) if isinstance(fun, tuple) else ((fun,), (cotangents,))
            parts = []
            for f, c in zip(funs, blocks):
                _, vjp = jax.vjp(lambda s, p, f=f: f(s, p, *args), frozen, params)

                def row(c, vjp=vjp):
                    state_bar, params_bar = vjp(c)
                    return jax.tree.map(jnp.add, params_bar, back(state_bar)[0])

                parts.append(jax.lax.map(row, c))
            return jax.tree.map(lambda *a: jnp.concatenate(a), *parts)

        self.pullback = pullback

    def solve(self, params, seed=None, tight=True):
        """Hot-restarted fixed-boundary equilibrium at ``params``: ``(state, mask)``, or ``None`` if not certified.

        ``tight=False`` solves to the trial tolerance, without the Newton refinement that certifies a state for
        differentiation (:meth:`linearize` re-solves tight first).
        """
        im, cfg = self._im, (self.cfg if tight else self.cfg_trial)
        if seed is not None:
            im._PERTURB_SEED[cfg] = seed
        params_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64), params)
        if tight:
            state, mask, status, _, _ = im._host_solve_and_mask_status(cfg, params_np)
            if int(status) != 0:
                return None
        else:  # a trial point is not differentiated: the forward solve alone, without the Newton refinement
            with im._device_context(cfg):
                try:
                    state, mask = im._host_solve_and_mask_impl(cfg, params_np, refine=False)
                except (VmecError, RuntimeError):  # RuntimeError: the relayed sentinel of a VmecError
                    return None
            result = im._LAST_SOLVE[cfg][1]
            fsq = float(result.fsqr) + float(result.fsqz) + float(result.fsql)
            if not (bool(result.converged) or fsq / cfg.ftol <= cfg.max_fsq_ratio):
                return None
        return jax.device_put(jax.tree.map(jnp.asarray, (state, mask)), cfg.device)

    def boundary_residual(self, state, params, field) -> BoundaryResidual:
        """:class:`BoundaryResidual` of a solved state."""
        return summarize_boundary_residual(*self._terms(state, params, field))

    def bootstrap_residual(self, state, params) -> float:
        """``max_j |I'(s_j) - I'_Redl(s_j)| / max_j |I'_Redl(s_j)|`` of a solved state (with ``bootstrap``)."""
        difference, target = map(np.asarray, self.bootstrap.mismatch(
            state, self._im.runtime_from_params(params, self.cfg), params))
        return float(np.max(np.abs(difference)) / np.max(np.abs(target)))

    def evaluate(self, x, params, field, seed=None, tight=False):
        """Rows at boundary ``x``, or ``None`` if the solve failed.

        Returns ``(rows, (state, mask, params_x, tight))``; ``tight`` marks a state solved to the deck's
        tolerance rather than the trial one.
        """
        params_x = self.with_boundary(params, jnp.asarray(x))
        if seed is None and self._linearization is not None:
            x_ref, state_ref, mask_ref, dz = self._linearization
            seed = self._predict(state_ref, mask_ref, dz, jnp.asarray(np.asarray(x) - x_ref), params_x)
        tight = tight or self.cfg_trial is self.cfg
        solved = self.solve(params_x, seed, tight)
        if solved is None:
            return None
        state, mask = solved
        return np.asarray(self._rows(state, params_x, field)), (state, mask, params_x, tight)

    def boundary_directions(self, params, x):
        """``ImplicitParams`` tangents of the boundary coordinates, stacked."""
        eye = jnp.eye(np.size(x))
        return jax.vmap(lambda e: jax.jvp(lambda xx: self.with_boundary(params, xx), (jnp.asarray(x),), (e,))[1])(eye)

    def linearize(self, x, aux, field, extra=None):
        """Columns of the rows along the model coordinates and the ``extra`` parameter tangents.

        The model coordinates are the boundary, then the bootstrap current.  A trial state is first solved to
        the deck's tolerance (from itself).  Returns ``(J, dz, params_batch, aux)``: the Jacobian, the projected
        state responses and parameter tangents of its columns, and the (tight) state they belong to.  The
        responses also seed the next trials (first-order predicted states).  A call at the boundary, plasma
        parameters and field of the latest one returns its result (without ``extra``'s columns when not asked
        for): a polish that stalls asks again for the Jacobian it was given.
        """
        state, mask, params_x, tight = aux
        key, extra_key = _fingerprint(x, params_x, field), None if extra is None else _fingerprint(extra)
        last = self._last_linearize
        if last is not None and last[0] == key and extra_key in (None, last[1]):
            J, dz, batch, aux = last[2]
            self._linearization = last[3]
            if extra_key is None and last[1] is not None:  # drop the extra columns
                n = self.x0.size
                J = J[:, :n]
                dz, batch = jax.tree.map(lambda a: a[:n], dz), jax.tree.map(lambda a: a[:n], batch)
            return J, dz, batch, aux
        if not tight:
            solved = self.solve(params_x, seed=state)
            if solved is None:
                raise RuntimeError("the trial state does not converge to the deck's tolerance")
            state, mask = solved
            aux = (state, mask, params_x, True)
        directions = self.boundary_directions(params_x, x)
        nb = self.n_boundary
        # Boundary, bootstrap-current and extra directions each in a program of its own: one batch of every
        # direction can exhaust the GPU.
        groups = [jax.tree.map(lambda a: a[:nb], directions)] if nb else []
        if self.bootstrap is not None:
            groups.append(jax.tree.map(lambda a: a[nb:], directions))
        n_coordinates = len(groups)
        if extra is not None:
            groups.append(extra)
        join = lambda *a: jnp.concatenate(a)  # noqa: E731
        # One tangent solve (one block factorization) for every group; the pushes stay per group.
        dz_all = self._tangents(params_x, state, mask, jax.tree.map(join, *groups))
        ends = np.cumsum([jax.tree.leaves(batch)[0].shape[0] for batch in groups])
        parts = [(jax.tree.map(lambda a: a[lo:hi], dz_all), batch)
                 for lo, hi, batch in zip(np.r_[0, ends[:-1]], ends, groups)]
        columns = np.hstack([np.asarray(self._push_rows(state, mask, params_x, dz, batch, field)).T
                             for dz, batch in parts])
        dz = jax.tree.map(join, *[dz for dz, _ in parts[:n_coordinates]])
        self._linearization = (np.asarray(x, dtype=float).copy(), state, mask, dz)
        if extra is not None:
            dz = jax.tree.map(join, dz, parts[-1][0])
        batch = jax.tree.map(join, *[batch for _, batch in parts])
        self._last_linearize = (key, extra_key, (columns, dz, batch, aux), self._linearization)
        return columns, dz, batch, aux

    def solve_boundary(self, params, field, *, x0=None, jacobian=None, ftol=1e-4, jacobian_ftol=1e-2, max_nfev=60,
                       target_cost=None, verbose=0):
        """K = 0 boundary for ``params`` and ``field``.

        With ``jacobian`` (from a nearby solve) Levenberg-Marquardt steps reuse it from ``x0``; otherwise, or if no
        step reduces the cost, or they end above ``target_cost``, a trust region with fresh Jacobians runs to
        ``jacobian_ftol`` and the reuse steps finish to ``ftol`` (see :func:`solve_free_boundary_three_term`).
        Returns a dict with ``x``, ``rows``, ``jacobian``, ``aux`` (state, mask, params, tight), ``nfev``,
        ``njev``, ``accepted`` and ``converged``.
        """
        import scipy.optimize

        x = np.asarray(self.x0 if x0 is None else x0, dtype=float)
        nfev = njev = 0
        if jacobian is not None:
            first = self.evaluate(x, params, field)
            if first is None:
                raise RuntimeError("the warm-start boundary has no certified equilibrium")
            out = _levenberg_marquardt(lambda z: self.evaluate(z, params, field), x, *first, jacobian,
                                       ftol=ftol, max_nfev=max_nfev, verbose=verbose, radius=self._radius)
            self._radius = out["radius"]
            close = target_cost is None or 0.5 * out["rows"] @ out["rows"] <= target_cost
            if (out["accepted"] or out["converged"]) and close:
                return dict(out, jacobian=jacobian, njev=0)
            # One fresh Jacobian where the steps stalled, and the same steps again, before a full trust region.
            J1, _, _, aux1 = self.linearize(out["x"], out["aux"], field)
            again = _levenberg_marquardt(lambda z: self.evaluate(z, params, field), out["x"], out["rows"], aux1, J1,
                                         ftol=ftol, max_nfev=max_nfev, verbose=verbose, radius=self._radius)
            self._radius = again["radius"]
            again["nfev"] += out["nfev"]
            if target_cost is None or 0.5 * again["rows"] @ again["rows"] <= target_cost:
                return dict(again, jacobian=J1, njev=1)
            x, nfev = again["x"], again["nfev"]
        memo = {}

        def fun(z):
            got = self.evaluate(z, params, field)
            if got is None:
                return np.full(memo.get("size", 1), 1e6)
            memo.update(key=z.tobytes(), size=got[0].size, rows=got[0], aux=got[1])
            return got[0]

        def jac(z):
            nonlocal njev
            if memo.get("key") != z.tobytes():
                fun(z)
            njev += 1
            J, _, _, memo["aux"] = self.linearize(z, memo["aux"], field)
            memo.update(jacobian=J, jkey=z.tobytes())
            return J

        fit = scipy.optimize.least_squares(fun, x, jac=jac, x_scale=self.x_scale, ftol=max(ftol, jacobian_ftol),
                                           xtol=1e-5, gtol=1e-10, max_nfev=max_nfev, verbose=verbose)
        nfev += int(fit.nfev)
        if memo.get("key") != fit.x.tobytes():
            fun(fit.x)
        J = jac(fit.x) if memo.get("jacobian") is None or memo.get("jkey") != fit.x.tobytes() else memo["jacobian"]
        out = _levenberg_marquardt(lambda z: self.evaluate(z, params, field), fit.x, memo["rows"], memo["aux"], J,
                                   ftol=ftol, max_nfev=max_nfev, verbose=verbose, radius=self._radius)
        self._radius = out["radius"]
        return dict(out, jacobian=J, nfev=nfev + out["nfev"], njev=njev)


@dataclass(frozen=True)
class _AcceptedPoint:
    """The design the optimizer last accepted (``ThreeTermFreeBoundaryProblem.accepted``)."""

    parameters: np.ndarray


class ThreeTermFreeBoundaryProblem(FunctionProblem):
    """A scalar loss and constraint rows of a design ``x`` at its three-term free boundary, for SLSQP.

    Build with ``FreeBoundaryProblem.from_loss(..., boundary_condition="three_term")``
    (:meth:`~vmex.core.freeboundary_problem.FreeBoundaryProblem.from_loss`); pass it to
    :func:`vmex.core.optimize.minimize`.  ``loss``, ``quantities``, ``parameter_quantities``,
    ``field_from_parameters`` and ``plasma_from_parameters`` are as there.  Every trial is the K = 0 boundary
    of the field and plasma parameters of ``x`` (:class:`ThreeTermFreeBoundaryModel`), warm-started from the
    first-order prediction of the latest linearized solution; a trial without a certified solution raises
    :class:`~vmex.core.errors.TrialRejected`.  A row ``h`` (loss or quantity) has the design gradient of the
    implicit function theorem of the boundary least squares,

        dh/dx = dh/dx|_b - lambda^T dr/dx,   lambda = J_b (J_b^T J_b)^-1 dh/db,

    with ``r`` the interface rows, ``J_b`` their Jacobian in the model's coordinates ``b`` (boundary, and the
    bootstrap current with ``bootstrap``) and ``dr/dx`` through the plasma parameters and the field; every
    row's ``dh/db`` and plasma term come from one reverse pass through the equilibrium.  Nothing recompiles
    when ``x`` changes.  ``model`` is the :class:`ThreeTermFreeBoundaryModel`; :meth:`boundary_residual` and
    ``stats`` (counts of trials, failures, boundary evaluations and linearizations) report progress.
    """

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None, scales=None, names=None,
                  quantities=(), parameter_quantities=(), boundary_ftol=1e-3, boundary_max_nfev=40,
                  boundary_max_residual=None, **model_options):
        """Solve the seed design's free boundary.

        ``model_options`` are :class:`ThreeTermFreeBoundaryModel` keywords (``nphi``, ``ntheta``,
        ``quadrature``, ``chunk``, ``trial_ftol``, ``bootstrap``, ...).  A trial's boundary fit stops at a
        relative cost change of ``boundary_ftol`` or after ``boundary_max_nfev`` evaluations; each gradient
        first polishes its trial with its own Jacobian.  With ``boundary_max_residual`` a fit that ends with
        any :class:`BoundaryResidual` term above it is no solution: the trial tries its next start, or is
        rejected.
        """
        functions = (loss, field_from_parameters, *quantities, *parameter_quantities)
        if plasma_from_parameters is not None:
            functions += (plasma_from_parameters,)
        if not all(map(callable, functions)):
            raise TypeError("loss, parameter maps and quantities must be callable")
        if not jax.config.x64_enabled:
            raise ValueError("free-boundary optimization requires JAX_ENABLE_X64=1")
        model = ThreeTermFreeBoundaryModel(inp, **model_options)
        return cls(model, loss, x0, field_from_parameters=field_from_parameters,
                   plasma_from_parameters=plasma_from_parameters, scales=scales, names=names,
                   quantities=tuple(quantities), parameter_quantities=tuple(parameter_quantities),
                   boundary_ftol=boundary_ftol, boundary_max_nfev=boundary_max_nfev,
                   boundary_max_residual=boundary_max_residual)

    def __init__(self, model, loss, x0, *, field_from_parameters, plasma_from_parameters, scales, names, quantities,
                 parameter_quantities, boundary_ftol, boundary_max_nfev, boundary_max_residual=None):
        from .implicit import runtime_from_params

        self.model, self.field = model, field_from_parameters
        self.boundary_ftol, self.boundary_max_nfev = float(boundary_ftol), int(boundary_max_nfev)
        self.boundary_max_residual = None if boundary_max_residual is None else float(boundary_max_residual)
        x0 = np.asarray(x0, dtype=float)
        params_at = (lambda params, x: params) if plasma_from_parameters is None else plasma_from_parameters

        def seed_params_at(x):
            return params_at(model.params0, x)

        self._params_at = jax.jit(seed_params_at)

        def part(f, with_x):
            def value(state, params, x):
                runtime = runtime_from_params(params, model.cfg)
                return jnp.ravel(jnp.asarray(f(state, runtime, x) if with_x else f(state, runtime)))
            return value

        parts = (part(loss, True), *(part(f, False) for f in quantities),
                 *(part(f, True) for f in parameter_quantities))

        def rows(state, params, x):
            return jnp.concatenate([f(state, params, x) for f in parts])

        self._rows = jax.jit(rows)
        # The coordinates of x that move plasma parameters: their columns of the interface rows enter the gradient.
        tangent = jax.jacfwd(seed_params_at)(jnp.asarray(x0))
        moved = sum(np.any(np.asarray(a).reshape(-1, x0.size) != 0, axis=0) for a in jax.tree.leaves(tangent))
        self._plasma = np.flatnonzero(moved)
        eye = jnp.eye(x0.size)[self._plasma]
        self._plasma_directions = jax.jit(lambda params, x: jax.vmap(lambda e: jax.jvp(
            lambda z: params_at(params, z), (x,), (e,))[1])(eye))

        def pullback(state, mask, params, x):
            """Every row's ``ImplicitParams`` gradient, each through its own function's reverse pass only."""
            sizes = [jax.eval_shape(f, state, params, x).size for f in parts]
            return model.pullback(parts, state, mask, params, tuple(jnp.eye(k) for k in sizes), x)

        self._pullback = jax.jit(pullback)

        @jax.jit
        def coordinates(params, b, x, params_bar):
            """``ImplicitParams`` cotangents onto the model coordinates ``b`` and onto ``x`` (plasma parameters)."""
            _, b_vjp = jax.vjp(lambda z: model.with_boundary(params, z), b)
            _, x_vjp = jax.vjp(lambda z: params_at(params, z), x)
            return jax.vmap(lambda bar: (b_vjp(bar)[0], x_vjp(bar)[0]))(params_bar)

        self._coordinates = coordinates
        self._direct = jax.jit(jax.jacrev(rows, argnums=2))
        self._field_vjp = jax.jit(lambda state, params, x, lam: jax.vjp(
            lambda z: model.rows(state, params, field_from_parameters(z)), x)[1](lam)[0])
        self._field_jvp = jax.jit(lambda state, params, x, dx: jax.jvp(
            lambda z: model.rows(state, params, field_from_parameters(z)), (x,), (dx,))[1])
        self._cache, self._anchor = {}, {}
        self.stats = dict(trials=0, failed=0, boundary_evaluations=0, linearizations=0)
        if self._solve(x0) is None:
            raise VmecError("the seed design has no three-term free boundary")
        self.accepted, self.accepted_step = _AcceptedPoint(x0.copy()), 0
        super().__init__(x0, names=names, scales=scales, fun=self._value, value_and_grad=self._value_gradient,
                         metadata={"holder": {"failed_trials": 0}})

    def _solve(self, x):
        """The trial at ``x`` (cached): boundary fit, state and rows, or ``None``."""
        x = np.asarray(x, dtype=float)
        key = x.tobytes()
        if key in self._cache:
            return self._cache[key]
        self.stats["trials"] += 1
        params, field = self._params_at(jnp.asarray(x)), self.field(jnp.asarray(x))
        anchor, starts, J = self._anchor, [None], None
        if anchor:  # the anchor's first-order boundary prediction, then the anchor's own boundary
            dx = x - anchor["x"]
            dr = anchor["J_p"] @ dx[self._plasma] + np.asarray(self._field_jvp(
                anchor["state"], anchor["params"], jnp.asarray(anchor["x"]), jnp.asarray(dx)))
            starts, J = [anchor["b"] - np.linalg.lstsq(anchor["J_b"], dr, rcond=None)[0], anchor["b"]], anchor["J_b"]
        fit = None
        for start in starts:
            try:
                fit = self.model.solve_boundary(params, field, x0=start, jacobian=J, ftol=self.boundary_ftol,
                                                max_nfev=self.boundary_max_nfev,
                                                target_cost=4 * anchor["cost"] if anchor else None)
            except (VmecError, RuntimeError):  # an uncertified equilibrium: try the next start
                continue
            if self.boundary_max_residual is not None:
                state, _, params_b, _ = fit["aux"]
                residual = self.model.boundary_residual(state, params_b, field)
                if max(residual.normal, residual.pressure, residual.sheet_current) > self.boundary_max_residual:
                    self.stats["boundary_evaluations"] += fit["nfev"]
                    fit = None  # the fit stalled away from the free boundary: try the next start
                    continue
            break
        if fit is None:
            self.stats["failed"] += 1
            self._cache[key] = None
            return None
        self.stats["boundary_evaluations"] += fit["nfev"]
        state, mask, params_b, _ = fit["aux"]
        self._cache[key] = dict(fit, state=state, mask=mask, params=params_b,
                                values=np.asarray(self._rows(state, params_b, jnp.asarray(x))))
        return self._cache[key]

    def _record(self, x):
        sol = self._solve(self._x(x))
        if sol is None:
            self.metadata["holder"]["failed_trials"] += 1
            raise TrialRejected("no three-term free boundary at this design point")
        return sol

    def _linearize(self, x, sol):
        """Every row's design gradient at ``sol`` (cached), re-solved tight and polished with its own Jacobian."""
        if "gradients" in sol:
            return sol["gradients"]
        model, xj = self.model, jnp.asarray(x)
        field, nb = self.field(xj), model.x0.size

        def relinearize():
            extra = self._plasma_directions(sol["params"], xj) if self._plasma.size else None
            try:
                J, _, _, aux = model.linearize(sol["x"], sol["aux"], field, extra=extra)
            except RuntimeError as error:  # the trial's state does not tighten to the deck's tolerance
                raise TrialRejected(str(error)) from error
            state, mask, params, _ = aux
            sol.update(aux=aux, state=state, mask=mask, params=params,
                       rows=np.asarray(model._rows(state, params, field)),
                       values=np.asarray(self._rows(state, params, xj)))
            self.stats["linearizations"] += 1
            return J

        J = relinearize()
        polished = model.solve_boundary(self._params_at(xj), field, x0=sol["x"], jacobian=J[:, :nb], ftol=1e-4,
                                        max_nfev=20)
        self.stats["boundary_evaluations"] += polished["nfev"]
        if polished["rows"] @ polished["rows"] < 0.9 * sol["rows"] @ sol["rows"]:
            sol.update(x=polished["x"], aux=polished["aux"])
            J = relinearize()
        J_b, J_p = J[:, :nb], J[:, nb:]
        state, params = sol["state"], sol["params"]
        params_bar = self._pullback(state, sol["mask"], params, xj)
        G_b, G_x = map(np.asarray, self._coordinates(params, jnp.asarray(sol["x"]), xj, params_bar))
        Q, R = np.linalg.qr(J_b)
        lam = Q @ np.linalg.solve(R.T, G_b.T)  # residual-space multipliers, a column per row
        gradients = np.asarray(self._direct(state, params, xj)) + G_x
        gradients -= np.stack([np.asarray(self._field_vjp(state, params, xj, jnp.asarray(column))) for column in lam.T])
        gradients[:, self._plasma] -= lam.T @ J_p
        self._anchor = dict(x=self._x(x).copy(), b=sol["x"], J_b=J_b, J_p=J_p, state=state,
                            params=params, cost=0.5 * sol["rows"] @ sol["rows"])
        sol["gradients"] = gradients
        return gradients

    def _value(self, x):
        return float(self._record(x)["values"][0])

    def _value_gradient(self, x):
        sol = self._record(x)
        gradients = self._linearize(x, sol)
        return float(sol["values"][0]), gradients[0].copy()

    def constraint_values(self, x):
        """The quantities' values, in their units."""
        return self._record(x)["values"][1:].copy()

    def constraint_jac(self, x):
        """The quantities' design gradients."""
        return self._linearize(x, self._record(x))[1:].copy()

    def nonlinear_constraint(self, lower, upper, *, scales=1.0):
        """Bound the quantities supplied to ``from_loss``, in their units."""
        from .freeboundary_problem import _nonlinear_constraint

        return _nonlinear_constraint(self.constraint_values, self.constraint_jac, lower, upper, scales)

    def accept_x(self, x):
        """Promote an evaluated point after the optimizer accepts it; drop the other trials."""
        x = self._x(x)
        key = x.tobytes()
        if self._cache.get(key) is None:
            raise ValueError("evaluate the candidate before accepting it")
        self.accepted, self.accepted_step = _AcceptedPoint(x.copy()), self.accepted_step + 1
        self._cache = {key: self._cache[key]}

    def state(self, x):
        """``(state, params)`` of ``x``'s free boundary (``params`` the ``ImplicitParams`` it was solved at)."""
        sol = self._record(x)
        return sol["state"], sol["params"]

    def boundary_residual(self, x) -> BoundaryResidual:
        """The three interface conditions at ``x``'s free boundary."""
        sol = self._record(x)
        return self.model.boundary_residual(sol["state"], sol["params"], self.field(jnp.asarray(x)))

    def equilibrium_from_x(self, x):
        """The free-boundary equilibrium of ``x`` as an ``Equilibrium`` (fixed-boundary solve on its boundary)."""
        from .implicit import input_with_params
        from .optimize import solve_equilibrium

        sol = self._record(x)
        return solve_equilibrium(input_with_params(self.model.fixed, sol["params"]),
                                 initial_state=sol["state"])

    def close(self):
        """Drop every cached trial except the accepted one."""
        key = self.accepted.parameters.tobytes()
        self._cache = {k: v for k, v in self._cache.items() if k == key}


def _levenberg_marquardt(evaluate, x, r, aux, J, *, ftol, max_nfev, verbose, radius=None):
    """Trust-region Gauss-Newton steps with a fixed Jacobian ``J`` from ``x`` (``evaluate(x) -> (r, aux)`` or ``None``).

    Each step costs one evaluation.  In Marquardt-scaled coordinates the step is the Levenberg-Marquardt step
    whose length is the trust radius (the Gauss-Newton step if shorter), from one SVD of ``J``.  The radius
    shrinks to a quarter of a step that gained less than a quarter of its predicted reduction, or that failed
    to solve, and doubles after a step at the radius that gained three quarters of it.  ``radius`` carries the
    last solve's radius over (at least 1/20 of the first Gauss-Newton step; default: that step's length).
    Stops at a relative cost change of ``ftol`` that the linear model predicted, after three steps in a row
    that each gained under 5% (the Jacobian is stale there), or when the radius falls below 1e-4 of the
    Gauss-Newton step.
    """
    J = np.asarray(J)
    norms = np.linalg.norm(J, axis=0)
    norms[norms == 0.0] = 1.0
    U, sv, Vt = np.linalg.svd(J / norms, full_matrices=False)
    nfev, accepted, converged, slow = 1, 0, False, 0  # nfev counts the caller's evaluation at x
    started = time.perf_counter()

    def step(g, lam):
        return -(sv / (sv**2 + lam)) * g  # in the singular basis of the scaled Jacobian

    while nfev < max_nfev:
        g = U.T @ r
        cost = 0.5 * r @ r
        if 0.5 * g @ g <= ftol * cost:  # the Gauss-Newton step's predicted gain
            converged = True
            break
        full = np.linalg.norm(step(g, 0.0))
        if nfev == 1:  # a carried radius, but not far below this Jacobian's own step
            radius = full if radius is None else max(radius, 0.05 * full)
        if radius < 1e-4 * full:
            break
        lam = 0.0
        if full > radius:  # the damping whose step has the radius' length (bisection in log lam)
            lo, hi = np.log(sv[-1] ** 2 * 1e-12 + 1e-300), np.log(sv[0] ** 2 * (full / radius) + 1e-300)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if np.linalg.norm(step(g, np.exp(mid))) > radius else (lo, mid)
            lam = np.exp(hi)
        z = step(g, lam)
        length = np.linalg.norm(z)
        predicted = cost - 0.5 * (r @ r - g @ g + np.sum((g + sv * z) ** 2))
        x_new = x + (Vt.T @ z) / norms
        got = evaluate(x_new)
        nfev += 1
        cost_new = np.inf if got is None else 0.5 * got[0] @ got[0]
        rho = (cost - cost_new) / predicted if predicted > 0 else -np.inf
        if verbose:
            print(f"Levenberg-Marquardt step {nfev - 1}: |step| {length:.2e} (radius {radius:.2e}), cost "
                  f"{cost:.6e} -> {cost_new:.6e}, gain ratio {rho:.2f} ({time.perf_counter() - started:.1f} s)",
                  flush=True)
        if rho < 0.25:
            radius = 0.25 * length
        elif rho > 0.75 and length > 0.99 * radius:
            radius = 2.0 * radius
        slow = slow + 1 if cost_new > 0.95 * cost else 0
        if cost_new < cost:
            x, (r, aux), accepted = x_new, got, accepted + 1
            if cost - cost_new <= ftol * cost and rho > 0.25:  # a small gain the linear model foresaw
                converged = True
                break
        if slow == 3:  # three steps in a row gained under 5%: the Jacobian no longer describes the rows here
            break
    return dict(x=x, rows=r, aux=aux, nfev=nfev, accepted=accepted, converged=converged, radius=radius)
