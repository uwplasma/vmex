Optimization API
================

VMEX separates equilibrium physics and derivatives from the optimization
algorithm. :class:`vmex.core.problem.VmecProblem` contains a decision vector,
callables, metadata, and immutable input conversions; it does not own an
optimizer.

Problem construction
--------------------

Weighted objective tuples are the shortest interface:

.. code-block:: python

   problem = opt.VmecProblem.from_tuples(
       inp,
       [(qi, 0.0, 1.0),
        (opt.aspect_ratio, 5.0, 0.01),
        (iota_floor, 0.0, 10.0)],
       max_mode=5,
       use_ess=True,
   )

Each tuple is ``(function, target, weight)``. By default the row is
``sqrt(weight) * (function - target)``, so ``weight`` multiplies the squared
cost. Negative cost weights are rejected. Set
``weight_semantics="residual"`` only when importing a definition in which the
weight itself multiplies each residual row.

``weight`` may also be a one-dimensional array with one entry per residual
row. This is the general radial-weight interface: for a profile sampled on
``s``, an edge-emphasized cost can use, for example,
``weight = w0 * (1 + 9*s**4)``. The same mechanism applies to Mercier,
Glasser, magnetic-well, QS/QI, and user-defined vector objectives; no
objective-specific weighting class is needed.

Use :meth:`~vmex.core.problem.VmecProblem.from_loss` for one traceable scalar
``loss(equilibrium_state, solver_context)``. ``equilibrium_state`` contains
the solved VMEC spectral coefficients; ``solver_context`` contains its grids,
profiles, and transforms (it is not elapsed run time). Use
:meth:`~vmex.core.problem.FunctionProblem.from_functions` when the user already
has decision-vector-level functions and derivatives.

Shared scalar optimizer
-----------------------

``opt.minimize(problem, ...)`` accepts ``VmecProblem``,
``FreeBoundaryProblem`` or ``FunctionProblem``. BFGS, L-BFGS-B and SLSQP
share the same call; equilibrium and adjoint controls belong in the problem
constructor. The older ``opt.minimize(objective_terms, inp, ...)`` call remains
available.

.. code-block:: python

   problem = opt.FreeBoundaryProblem.from_loss(
       inp, loss, coils=coils, coil_current_dofs=(),
       quantities=(opt.min_abs_iota, opt.major_radius),
       solver_options=dict(ftol=1e-11, edge_force_tolerance=1e-11,
                           adjoint_residual_rtol=1e-9))
   limits = problem.nonlinear_constraint(
       [0.19, 0.99], [float("inf"), 1.01], scales=[0.19, 0.01])
   result = opt.minimize(problem, method="SLSQP", constraints=limits,
                         options=dict(maxiter=100, ftol=1e-11))

The loss remains an ordinary ``loss(state, runtime, coils)`` function in the
example. Coordinates, bounds, callbacks, returned gradients and ``result.x``
use the problem's units. ``problem.scales`` conditions internal optimizer
steps about the initial point. ``nonlinear_constraint`` normalizes constraint
rows, including their analytic Jacobian. For ``FunctionProblem`` and
``VmecProblem`` it bounds residual rows; for scalar ``FreeBoundaryProblem`` it
bounds the supplied physical ``quantities``. Explicit SciPy ``Bounds``,
``LinearConstraint`` and ``NonlinearConstraint`` objects are also accepted;
nonlinear constraints must supply an analytic Jacobian.

Free-boundary promotion belongs to this adapter: L-BFGS-B/BFGS accept iteration
callbacks, while SLSQP accepts the next Jacobian request after its line search.
A rejected probe never becomes the warm-start anchor. Its ``maxiter`` limits
accepted steps; a budget stop is not convergence. An unrecoverable trial returns
``success=False``, ``stop_reason="equilibrium_trial_rejected"`` and the last
accepted point. Callbacks run after promotion. ``StopIteration`` from a callback
also returns the last accepted point. Inspect success, stop reason and physical
endpoint checks separately.

``coil_current_dofs`` explicitly selects coil currents in the free-boundary
constructors (``()`` fixes them); the legacy ``current_dofs`` spelling remains
supported. Fixed-boundary ``VmecProblem.current_dofs`` instead selects the
plasma-current profile. Free-boundary pressure/current profiles currently come
from ``inp`` and are held fixed by this coil parameterization.

Vector least squares or one scalar adjoint
-------------------------------------------

The residual and scalar interfaces can represent the same mathematical cost
while asking for different derivatives. Let ``z`` be the converged equilibrium,
``x`` the boundary/current decision vector, and

.. math::

   g(z,x)=0, \qquad
   \Phi(z,x)=\tfrac12 r(z,x)^T r(z,x).

The residual/Jacobian route builds

.. math::

   J_r = \frac{d r}{d x}
       = r_x-r_z g_z^{-1}g_x,
   \qquad \nabla_x\Phi=J_r^T r.

This is what ``VmecProblem.from_tuples`` plus SciPy ``least_squares`` needs.
The optimizer receives every residual row and the complete
``n_residuals``-by-``n_dofs`` Jacobian. It can form a Gauss--Newton trust-region
model, report individual rows, and stop on residual/Jacobian criteria. VMEX
uses its block response algorithm to amortize the equilibrium linear algebra,
but the complete residual Jacobian is still computed and materialized.

When the optimizer only needs a scalar value and gradient, form ``Phi`` before
differentiating. The adjoint is

.. math::

   g_z^T\lambda=\Phi_z^T= r_z^T r,
   \qquad
   \nabla_x\Phi=\Phi_x-\lambda^T g_x.

There is one equilibrium-adjoint right-hand side per scalar gradient,
independent of the number of residual rows and decision variables. ``One``
describes the number of adjoint solves; its cost still depends on equilibrium
resolution, linear conditioning, and the work needed to evaluate the objective
VJP. VMEX does not tape the nonlinear equilibrium iterations.

The explicit scalar construction is:

.. code-block:: python

   def loss(state, runtime):
       rows = opt.residuals_from_tuples(state, runtime, terms)
       return 0.5 * jnp.vdot(rows, rows)

   problem = opt.VmecProblem.from_loss(
       inp, loss, max_mode=5, use_ess=True)
   problem.compile_value_and_gradient()

   result = scipy.optimize.minimize(
       problem.value_and_grad, problem.x0,
       jac=True, method="L-BFGS-B", bounds=problem.bounds)

For a single tuple-defined stage, :func:`vmex.core.optimize.minimize` is the
short form of the same route:

.. code-block:: python

   result = opt.minimize(
       terms, inp, max_mode=5, method="L-BFGS-B",
       options={"maxiter": 100})

The longer ``from_loss`` form is useful when the scalar is not a sum of tuple
rows, or when a script owns stage-specific scaled coordinates and monitoring.

The two constructions use the same rows, targets, weights, and scalar cost.
They do **not** use the same optimization algorithm. ``least_squares`` uses
the explicit residual Jacobian and Gauss--Newton curvature; L-BFGS-B sees only
``Phi`` and its gradient and builds limited-memory curvature from accepted
steps. Their iterates, stopping tests, and possibly the local minimum reached
can differ. Always compare final physical terms and held-out validation
metrics, not iteration counts alone.

Choose deliberately:

* Keep the vector route when residual-level diagnostics, a least-squares trust
  region, or Gauss--Newton curvature is valuable.
* Prefer the scalar route when a large pointwise objective makes cold Jacobian
  compilation or materialization the bottleneck and the optimizer only needs
  ``(value, gradient)``.
* A scalar loss must be fully JAX-traceable. Opaque WOUT/host callbacks require
  a traceable implementation or the finite-difference derivative lane.
* ``compile_value_and_gradient`` makes the first scalar compile explicit;
  it does not turn a cold timing into a warm timing.

The committed QA startup measurements (VMEX 0.7.0, Apple M4, before
``jacobian_batch_size="auto"`` became the default) make the tradeoff concrete
for 6,723 rows and 48 boundary degrees of freedom. Cold startup
dropped from 44.7 s and 2,965 MiB peak RSS for the residual Jacobian to 32.2 s
and 2,574 MiB for the scalar adjoint. Warm value/gradient medians were 16.6 s
and 17.4 s, respectively, so the scalar path did not improve warm throughput
in that measurement. The raw records are
``benchmarks/qa_optimization_startup_least_squares_m4.json`` and
``benchmarks/qa_optimization_startup_scalar_m4.json``; neither
record is a GPU or persistent-cache claim.

The canonical ``QA_optimization.py`` remains the residual/Jacobian tutorial.
The eight ``{QA,QH,QP,QI}_optimization[_finite_beta]_scalar.py`` companions
show the scalar route in vacuum and at finite beta without replacing the
least-squares examples.

Callable contracts
------------------

.. list-table::
   :header-rows: 1

   * - Consumer
     - Value
     - Derivative
   * - ``scipy.optimize.least_squares``
     - ``problem.residual``
     - ``problem.residual_jac``
   * - ``scipy.optimize.minimize``
     - ``problem.fun``
     - ``problem.grad`` or ``problem.value_and_grad`` with ``jac=True``
   * - JAXopt / Optax
     - ``problem.jax_fun``
     - ``problem.jax_value_and_grad``
   * - SIMSOPT-style user code
     - ``problem.J``
     - ``problem.dJ``

For tuple problems, every scalar interface is defined by the certified
least-squares pair

.. math::

   \Phi(x) = \tfrac12 r(x)^T r(x), \qquad \nabla\Phi(x) = J(x)^T r(x).

The SciPy and JAX callables therefore return the same value and gradient.
VMEX maintains one exact-key host cache to avoid repeated work when an
optimizer requests the value and derivative separately.

For a custom vector diagnostic, use
:meth:`~vmex.core.problem.VmecProblem.jax_quantity_from_state`. It returns the
floating-point quantity and the equilibrium status from one implicit solve;
rejected trials return NaNs rather than a plausible value. Form only the
contraction an algorithm needs with ``jax.vjp`` instead of materializing a
full residual Jacobian.

``problem.dof_names`` is ordered exactly like ``problem.x0`` and every
optimizer vector passed to the problem. For example,
``dict(zip(problem.dof_names, result.x))`` labels an optimized SciPy result.
The older ``problem.names`` spelling remains available as the underlying
immutable tuple.

``RBC(0,0)`` is fixed by default because changing it mainly changes the major
radius. Pass ``vary_major_radius=True`` to release that coefficient explicitly;
VMEX does not add the identically-zero ``ZBS(0,0)`` direction.

Derivative methods
------------------

``derivative_method="implicit"`` is the default. It differentiates the
converged fixed point, requires traceable
``(equilibrium_state, solver_context)`` objectives, and
normally costs far less than one equilibrium solve per decision variable.
``implicit_jacobian_method="auto"`` uses a reverse adjoint for one residual row
and the block-tridiagonal forward response for a residual vector. Advanced
choices are ``"block_tridiagonal"``, ``"forward_gmres"``, and
``"reverse_adjoint"``.

VMEX checks each Jacobian response against the linearized residual:

======================  ================================================
Interface               Failed response
======================  ================================================
``auto``                 recompute with the reverse adjoint
explicit host method    raise :class:`vmex.core.errors.AdjointSolveError`
``jax_residual_jac``     select the reverse branch inside the JAX graph
======================  ================================================

``jacobian_batch_size="auto"`` is the default. It sizes from the available
memory the batch of probe rows the block system assembles at once, which is
where a block Jacobian spends its time; the batch width changes the cost, not
the Jacobian. Set
``jacobian_batch_size=1`` for the serial pass when peak memory, not
throughput, is the binding constraint; ``None`` is the widest and the most
memory-hungry. ``adjoint_tol`` and ``adjoint_maxiter`` control the certified
Krylov solves.

``derivative_method="finite_difference"`` accepts opaque host objectives. It
uses independent equilibrium probes and ``workers=None`` automatically uses
the CPUs available to the process. Select ``fd_method="2-point"`` or
``"3-point"`` and set ``workers=1`` for a serial reference.

Forward solves and FSQ certification
------------------------------------

The input's ``NS_ARRAY``, ``FTOL_ARRAY``, and ``NITER_ARRAY`` define the
multigrid solve. The same schedule is used by implicit and finite-difference
problems. ``forward_ftol`` and ``forward_max_iterations`` are concise
overrides for the final stage. :func:`vmex.core.optimize.solve_equilibrium`
accepts the same two names for one-off forward solves:

.. code-block:: python

   problem = opt.VmecProblem.from_tuples(
       inp, terms, max_mode=5,
       forward_ftol=1e-12,
       forward_max_iterations=5500,
       max_fsq_ratio=1e2,
   )

VMEC reports ``FSQ = fsqr + fsqz + fsql``. A converged trial is always
derivative-certified. If a trial exhausts its iteration budget, VMEX only
differentiates it when ``FSQ / forward_ftol <= max_fsq_ratio``; otherwise all
scalar interfaces return the same smooth rejection wall. The default
``1e2`` of the optimization factories (``make_problem``, ``least_squares``,
``minimize``) keeps the implicit derivative, which assumes ``F = 0``, away from
trials whose residual is far above the deck tolerance. Tighten it for stricter
studies.

Inspect the policy instead of guessing:

.. code-block:: python

   evaluation = problem.evaluate(x)
   print(evaluation.status, evaluation.diagnostics)

Diagnostics include ``fsq``, ``fsq_ratio``, ``max_fsq_ratio``,
``derivative_certified``, solve/iteration totals, rejected trials, and
derivative fallbacks. ``benchmarks/optimization.py`` profiles the QI, QA, QH,
QP, and scalar contracts over NFP 1--5 and accepts ``--max-fsq-ratio`` without
turning machine-specific results into a package default.

SciPy
-----

.. code-block:: python

   result = scipy.optimize.least_squares(
       problem.residual, problem.x0,
       jac=problem.residual_jac,
       x_scale=problem.scales,
       max_nfev=50,
       verbose=2,
   )

   result = scipy.optimize.minimize(
       problem.value_and_grad, problem.x0,
       jac=True, method="L-BFGS-B",
       bounds=problem.bounds,
       options={"maxiter": 100},
   )

``BFGS`` and ``L-BFGS-B`` use the same smooth rejected-trial scalar pair as
the least-squares-derived objective. Bounds and line-search options remain
ordinary SciPy choices. :class:`vmex.core.monitoring.OptimizationMonitor`
records accepted iterations without changing the objective. For a tuple
problem it also separates the weighted cost by term; saving and plotting the
history needs only the callback plus two output calls:

The SciPy examples optimize normalized variables ``u`` with
``x = x0 + step * u``. A bound of ``[-1, 1]`` therefore means one declared
parameter step, not a universal physical limit. The QA/QP boundary examples
use that conservative box; the QI and coil-shape examples use ``[-3, 3]``
because the narrower box changed the line-search path or pinned necessary coil
motion in representative scans. Keep ``PARAMETER_BOUND`` next to the other
driver inputs and validate it for a new normalization.

.. code-block:: python

   monitor = opt.OptimizationMonitor(problem, stream=None)
   result = scipy.optimize.least_squares(
       problem.residual, problem.x0, jac=problem.residual_jac,
       callback=monitor)
   monitor.save("objectives.csv")
   monitor.plot("objectives.png")

Continuation stages may restart the optimizer iteration counter; the monitor
keeps the combined saved history strictly increasing. Exact-zero terms are
drawn at a relative numerical display floor instead of forcing a meaningless
``1e-308`` axis. For joint surface/coil objectives, accepted vectors are also
available as ``monitor.x_history``. For a joint normalized surface/coil driver,
the optional movie call applies ``x = x0 + step*u`` and can color the surface::

   monitor.movie_surface_coils(
       "optimization.gif", objects_from_x, x0=x0, scales=step,
       surface_color="B.n/B", plasma_problem=problem,
       external_field=lambda objects: coil_field(objects[1]), max_frames=50)

Pass ``color_factory=`` to color the first surface by a scalar field at every
accepted iterate. The single-stage examples expose one top-level choice:
``None``, ``"absB"``, ``"B.n/B"``, or a user callable (for example a
bootstrap diagnostic). The movie uses one color scale across all frames.

For a custom JAX scalar objective, keep differentiation visible in the driver.
Return ``(cost, {name: term_cost})`` as auxiliary data, call
``jax.value_and_grad`` directly, and cache that already-computed evaluation for
the plotting callback:

.. code-block:: python

   value_and_grad = jax.value_and_grad(objective, has_aux=True)

   def scipy_value_and_grad(x):
       (value, terms), gradient = value_and_grad(jnp.asarray(x))
       monitor.cache_evaluation(x, value, gradient, terms)
       return float(value), np.asarray(gradient)

   result = scipy.optimize.minimize(
       scipy_value_and_grad, x0, jac=True, method="BFGS", callback=monitor)

The objective, its exact derivative, and SciPy's ``jac=True`` contract are all
explicit. Rejected line-search evaluations remain outside the accepted-iterate
history because only SciPy calls ``monitor`` as the callback. Large VMEC,
virtual-casing, and coil terms may likewise be differentiated separately and
their values and gradients added before ``cache_evaluation``; this produces
smaller XLA executables without changing the optimizer contract.

Joint VMEX--ESSOS objectives
----------------------------

ESSOS owns coil geometry and its variable convention. ``coils.dofs`` and
``coils.dof_names`` have identical ordering, while ``coils.with_dofs(x)``
constructs a traceable trial set without mutation. The same package provides
``surfacerzfourier_from_boundary(rbc, zbs, nfp, ...)``,
``loss_coil_separation``, ``loss_coil_surface_distance``, and
``Coils.from_simsopt``. VMEX therefore needs no duplicate coil indexing or
distance implementation.

.. warning::

   The joint coil, exterior VJP, and field-line tracing examples need ESSOS
   0.17 or newer: ``pip install "vmex[coils]"``. Earlier releases lack the
   ``Coils.from_json``/``Coils.with_dofs`` API these examples use.

For a finite-beta prescribed boundary, virtual casing gives the field of the
enclosed plasma currents. The physical exterior field is that contribution
plus the actual ESSOS coil field. The two interface residuals used by the
example are

.. math::

   \mathbf B_{\rm out}\!\cdot\mathbf n,
   \qquad
   \frac{|\mathbf B_{\rm out}|^2-|\mathbf B_{\rm in}|^2
   -2\mu_0p_{\rm edge}}{B_{\rm ref}^2}.

The second is a normalized total-pressure jump, not a pressure-profile error.
It supplies the tangential-field magnitude condition that ``B.n/B`` alone
does not constrain, even when the input pressure vanishes at the LCFS. The
fixed-boundary examples vary boundary and coil variables together; they do
not call NESTOR. A coil-only free-boundary optimization must instead
differentiate the fully reconverged NESTOR root. The experimental public path
keeps the construction visible in the driver::

   from vmex.core import implicit as im

   params = im.params_from_input(inp)
   config = vj.make_free_boundary_config(
       inp, BiotSavart(coils0), field_from_parameters=field_from_u)
   solver_context = im.runtime_from_params(params, config.implicit)

   def objective(u):
       equilibrium_state, status, _, _ = vj.solve_free_boundary_implicit_status(
           params, u, config)

       def accepted(_):
           residual = opt.residuals_from_tuples(
               equilibrium_state, solver_context, tuples)
           return 0.5 * jnp.vdot(residual, residual)

       # This visible wall lets a line search backtrack from an invalid trial.
       rejected = lambda _: 1e3 * (1 + jnp.linalg.norm(u))**2
       return jax.lax.cond(status == 0, accepted, rejected, None)

   value_and_grad = jax.value_and_grad(objective)
   result = scipy.optimize.minimize(
       value_and_grad, u0, jac=True, method="L-BFGS-B")

NESTOR moves the LCFS and only the coil vector is optimized. The vacuum example
adds coil geometry terms; the finite-beta example adds beta and Redl bootstrap
terms. This path is reverse-mode only. Status 0 enters the adjoint; status 1
means a failed solve, status 2 an under-converged solve, and status 3 a solve
that met ``ftol`` but could not be Newton-anchored on the coupled
plasma--vacuum root. Every status-0 state is anchored there (to
``refine_tol``, 1e-10 by default), so its value and its adjoint refer to the
same point; VMEC's ``ftol`` alone leaves the state off that root along weakly
damped directions.
The certified whole-state GCROT transpose remains the default.
``adjoint_solver="boundary_schur"`` selects the advanced boundary-Schur lane, which
eliminates the block-tridiagonal radial bulk and solves only the evolved-edge
correction before checking the full coupled residual. Local three-surface
rows and a pivoted sparse band solve reduce its cold cost substantially, but
local-row compilation still keeps it opt-in. It is a host lane: under an
outer ``jax.jit`` the pullback warns and uses the staged coupled GCROT solve
instead, so call the objective eagerly to keep it. ``device="auto"`` uses the CPU
for this response on accelerator hosts; an explicit ``device="gpu"`` or
process-wide JAX placement overrides that measured lower-memory default.

Several objectives at one free-boundary root
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced callers that retain an accepted state, its DOF mask, and its
``rcon0``/``zcon0`` constraint baselines can use
``vmex.free_boundary_state_pullback_multi_rhs``. Stack state cotangents on a
leading RHS axis and pass the same profile parameters, field parameters,
configuration, state, mask, and baselines that define that root. The returned
pair contains profile-parameter and field-parameter cotangents with the same
leading RHS axis. Add explicit objective derivatives with respect to those
parameters separately; this helper supplies only the implicit state response.

For the Krylov backends, the helper prepares one state transpose and one
parameter pullback for all rows. It runs independent sequential GCROT
solves, retaining the scalar
tolerances and independently checking every adjoint residual. It does not
carry recycle spaces between objectives. A projected preconditioned root
residual check (``root_residual_atol``, default ``1e-5``) precedes the solves;
this numerical gate does not establish physical gradient accuracy.

This interface is host-eager and supports all five free-boundary adjoint
backends, including upstream ``boundary_schur`` and ``edge_response``.
For Schur, both the adjoint and parameter pullback use the raw residual;
other methods use their existing preconditioned residual. Schur currently
factors each cotangent row independently.
The existing scalar custom VJP, traced solver path, and ``boundary_schur``
interface retain their behavior. Use independent re-solve finite differences
and root-convergence studies to qualify a new physical response.

Continuation from an accepted free-boundary anchor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``make_free_boundary_continuation_config`` takes an existing coupled solver
configuration, fixed ``ImplicitParams``, and a rank-one vector of field
coordinates. It solves a cold anchor with the solver's FTOL and iteration
limit. Each target follows a straight path from that fixed anchor, with step
sizes bounded by the infinity norm of the change divided by
``parameter_scales``. Every intermediate solve must converge, satisfy FTOL
in all three reported force components, and pass the explicitly supplied
projected residual limit before its state can seed the next point.

For example, after constructing a field chart and its reference field::

    solver = vmex.make_free_boundary_config(
        inp, field, field_from_parameters=field_from_parameters,
        ftol=1e-17, adjoint_tol=1e-10, device="gpu")
    continuation = vmex.make_free_boundary_continuation_config(
        solver, params, parameter_anchor=p0, parameter_scales=scales,
        continuation_step=0.1, max_continuation_steps=32,
        root_residual_atol=1e-8)
    accepted = vmex.free_boundary_continuation_result(p_trial, continuation)
    state = accepted.state
    complete_forward_output = accepted.result

The tolerances in this example are starting choices, not universal physical
accuracy guarantees. Use force/root convergence studies and independent
gradient checks for the observables and parameter directions of interest.
This path uses the ordinary forward solver; it does not invoke a root
polisher. Constraint baselines are carried between points, while the vacuum
solution is recomputed for the changed field.

The returned ``FreeBoundaryContinuationResult`` binds the state to its exact
parameter vector, DOF mask and constraint baselines. For several objectives,
pass that record to
``free_boundary_continuation_state_pullback(accepted, continuation, rhs_batch)``.
This reuses the shared coupled-GCROT pullback and returns implicit
field-coordinate derivatives. Add explicit objective derivatives separately.
For scalar JAX objectives, use
``solve_free_boundary_continuation(parameters, continuation)``; its custom VJP
delegates to the existing scalar adjoint at the saved endpoint. Profiles stay
fixed in this initial continuation API. Solver iterations and anchor selection
are not differentiated.

.. _free-boundary-seed-lu:

Reusing a dense seed as a matrix-free preconditioner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an explicit continuation loop, ``forward_dense_jax`` with
``adjoint_fail="error"`` can retain one dense seed LU and solve subsequent
adjoints and predictors with current-root matrix-vector products::

    dense = vmex.free_boundary_continuation_state_pullback(
        accepted, continuation, rhs_batch, return_linearization=True)
    seed = dense.preconditioner(
        rtol=1e-11, tangent_rtol=1e-11, restart=100, max_restarts=3,
        rhs_batch_size=3, require_adjoint_convergence=False)
    dense.close()  # the seed owns an independent float64 host copy
    try:
        current = vmex.free_boundary_continuation_state_pullback(
            accepted, continuation, rhs_batch, preconditioner=seed,
            return_linearization=True)
        try:
            implicit_gradient_rows = current.field_jacobian
            response = current.tangent(accepted, continuation, coil_direction)
        finally:
            current.close()
    finally:
        seed.close()

Repeat the inner pullback at each newly certified root, with that root's
cotangents, using the same seed. Add the objective's explicit field derivatives
to ``field_jacobian``. The seed only preconditions the solve: it never replaces
the current Jacobian. Identical or opposite cotangents reuse a solution within
one batch; arbitrary row counts are supported. Predictor ownership and full
residual checks remain the same as in the dense path.

``FreeBoundaryLUPreconditioner`` is bound to one continuation configuration,
state layout and active basis. A different resolution or mask requires a new
seed, as does reanchoring into a new continuation configuration. There is no
global cache, automatic factor refresh or dense fallback in this core API.
The requested Krylov tolerance is distinct from ``adjoint_residual_rtol``,
which checks the actual full projected equation. Failure raises instead of
returning an unchecked gradient. Set the full residual tolerance explicitly;
the vacuum scalar pilot used ``adjoint_residual_rtol=1e-9``.
``require_adjoint_convergence=True`` additionally requires Krylov's stopping
flag; otherwise acceptance depends on the finite, full residual.
``tangent_rtol`` independently controls predictor stopping and inherits
``rtol`` when omitted. ``rhs_batch_size`` groups one to four independent
adjoints, each with its own convergence and full-residual checks.
Closing a seed prevents further pullbacks;
already-created predictor objects retain their own factor references.

This removes repeated dense assembly, not the initial dense setup or storage.
The caller may also use ``linearization.offload_factors()`` on the ordinary
dense path to keep its factors in host memory between predictor calls. Both
paths are host-eager and require float64. The scalar custom VJP is unchanged.
Qualification still requires independent, reconverged finite differences at
representative equilibria. A correct adjoint of an under-resolved equilibrium
does not establish physical accuracy at higher resolution.

Dense remains the default. This low-level option does not change
``FreeBoundaryProblem`` or its projected optimizer. Callers implementing dense
recovery must check the same certified trial and promote replacement factors
only after optimizer acceptance. A short vacuum pilot does not qualify long
optimization runs or finite-beta equilibria.

Exact targets use a bounded memo (eight endpoints by default). A failed path
does not update the memo or anchor. ``force_recompute=True`` requests a fresh
path from the same anchor; failure preserves a previously accepted endpoint.
Each new target starts from the common anchor, irrespective of trial order.
The optional parameter validator runs on memo hits and all path points.
Keep the solver configuration and field chart immutable during use.

After the optimizer accepts a result under its physical constraints and
objective criteria, explicitly call
``reanchor_free_boundary_continuation_config(continuation, accepted)`` to obtain
a new context. Merely evaluating a trial never promotes it. Use
``free_boundary_continuation_stats`` for solve/iteration counts, memo hits,
failures and the last path's accepted-point residuals. This API supersedes the
local continuation implementation's dependency on the separate projected-root
response engine; historical design/scalar aliases are not imported.

Use :class:`vmex.core.monitoring.EquilibriumReporter` for the compact physics
summary shared by the examples.  Each entry accepts either VMEX's
``function(equilibrium_state, solver_context)`` convention or a host
``function(equilibrium)``;
the call prints one line and returns the same values by label::

   report = opt.EquilibriumReporter(
       ("QS total", qs.total, ".6e"),
       ("aspect", opt.aspect_ratio, ".4f"),
       ("mean iota", opt.mean_iota, ".4f"))
   values = report("final", equilibrium)

JAXopt and Optax
----------------

Install ``vmex[optimizers]`` and pass the JAX pair directly:

.. code-block:: python

   solver = jaxopt.LBFGS(
       problem.jax_value_and_grad,
       value_and_grad=True,
       jit=False,
       maxiter=100,
   )
   result = solver.run(jnp.asarray(problem.x0))

   transform = optax.adam(1e-2)
   x, state = jnp.asarray(problem.x0), transform.init(problem.x0)
   for _ in range(100):
       value, gradient = problem.jax_value_and_grad(x)
       updates, state = transform.update(gradient, state, x)
       x = optax.apply_updates(x, updates)

The equilibrium uses a host callback, so an outer JAXopt solver should use
``jit=False``; VMEX still JIT-compiles the numerical kernels. See the three
``QI_optimization_{scipy,jaxopt,optax}.py`` examples, which share one problem
definition.

Resolution, continuation, and ESS
---------------------------------

Optimization scripts should show their numerical resolution explicitly:

.. code-block:: python

   mpol = max(max_mode + 2, minimum_mpol)
   inp = replace(inp, delt=0.5).change_resolution(
       mpol=mpol, ntor=mpol,
       ntheta=2 * mpol + 6,
       nzeta=2 * mpol + 4,
   )

``max_mode`` selects decision variables; ``mpol`` and ``ntor`` select the
equilibrium representation. They are related but not interchangeable.
Real-space grids must resolve the retained spectrum. Converge representative
results in radial and angular resolution rather than treating one formula as
a proof of adequacy.

``use_ess=True`` supplies exponential spectral scales to the optimizer. It
allows high modes to be present while low modes take larger steps. ESS is a
scaling policy, not a global optimizer: a mode ladder can reach a different
basin because every stage solves a different restricted problem. Carry a
stage forward with ``inp = problem.input_from_x(result.x)`` and construct the
next problem from that input.

``ess_alpha`` controls that separation explicitly. A mode of level
``k=max(|m|, |n|)`` is scaled by ``exp(-ess_alpha*(k-1))``. The default
``ess_alpha=1.2`` is a conservative starting point for crude seeds; values near
``0.7``--``0.9`` allow modes 3--5 to move more during basin exploration, while
larger values suppress them more strongly. There is no configuration-independent
best value: compare candidates at equal solve budgets and recheck the winner at
the final VMEC resolution. ``QA_optimization_global.py`` shows bounded basin
hopping with ``ess_alpha=0.7`` followed by exact least-squares polishing.

Hot restart and final output
----------------------------

Optimization trials hot-restart by default. The exact accepted state is
available without another cold solve:

.. code-block:: python

   inp = problem.input_from_x(result.x)
   equilibrium = problem.equilibrium_from_x(result.x)

   final_input = replace(
       inp,
       ns_array=np.array([101]),
       ftol_array=np.array([1e-14]),
       niter_array=np.array([8000]),
   )
   final_equilibrium = opt.solve_equilibrium(
       final_input,
       initial_state=equilibrium.state,
       verbose=True,
       raise_on_max_iterations=True,
   )
   final_input.to_indata("input.optimized")
   vj.write_wout("wout_optimized.nc", final_equilibrium.wout)
   vj.plot_wout("wout_optimized.nc", "figures")

``verbose=True`` shows whether the final run needs a larger iteration budget.
The hot seed is especially important for strongly shaped boundaries whose
cold magnetic-axis guess may be poor.

Pointwise fields and VJPs
-------------------------

Use :func:`~vmex.core.optimize.solve_equilibrium` for ordinary field queries.
When parameter VJPs are needed,
:meth:`~vmex.core.problem.VmecProblem.from_input` supplies the boundary/current
parameterization without inventing an optimization objective, and
``problem.equilibrium_from_x`` retains it. Set Cartesian or VMEC flux points
once and evaluate the field inside the LCFS:

.. code-block:: python

   final_equilibrium = problem.equilibrium_from_x(result.x)
   final_equilibrium.set_points_xyz([[x, y, z]])
   # Or: final_equilibrium.set_points_flux([[s, theta, phi]])

   B = final_equilibrium.B()
   absB = final_equilibrium.absB()
   gradB = final_equilibrium.gradB()
   gradgradB = final_equilibrium.gradgradB()
   gradgradgradB = final_equilibrium.gradgradgradB()

   dBdx = final_equilibrium.B_vjp(jnp.ones_like(B))
   dgradBdx = final_equilibrium.gradB_vjp(jnp.ones_like(gradB))
   d2Bdx = final_equilibrium.gradgradB_vjp(jnp.ones_like(gradgradB))
   d3Bdx = final_equilibrium.gradgradgradB_vjp(
       jnp.ones_like(gradgradgradB))

``B`` and all derivative components/axes are Cartesian for either point-input
route: ``gradB[..., i, j] = d B_i / d x_j``. VJPs return one value per
``problem.dof_names`` and include selected current parameters as well as
boundary modes; a flux-coordinate point is first mapped to Cartesian space,
and that physical point is held fixed during the parameter VJP. VMEX inverts
Cartesian points to ``(s, theta, phi)`` with a differentiable Newton solve,
evaluates angular dependence spectrally, and interpolates the radial mesh.
Points outside the LCFS return NaNs; use ``problem.exterior_field`` or
``equilibrium.exterior_field`` there. Equilibria returned by
``problem.equilibrium_from_x`` retain the same exterior-field VJPs. High
spatial derivative orders are
substantially more expensive and radial derivatives are piecewise smooth at
the VMEC mesh surfaces, so converge them in ``NS_ARRAY``.

``examples/vmex_get_B_gradB.py`` demonstrates the stable interior field API,
including Cartesian and flux-coordinate queries, three spatial derivatives,
and parameter VJPs. The exterior VJP and field-line tracing scripts need
ESSOS 0.17 or newer (``pip install "vmex[coils]"``).

To include coil parameters in an exterior-field VJP, pass the same functional
ESSOS update used by an optimization:

.. code-block:: python

   field = problem.exterior_field(
       result.x,
       external_parameters=coils.dofs,
       external_field_from_parameters=lambda x: coil_field(coils.with_dofs(x)),
       external_dof_names=coils.dof_names,
   )

``field.dof_names`` then lists VMEX variables followed by ESSOS variables.
For unusually large surfaces or target arrays, ``chunk_size`` and
``target_chunk_size`` cap virtual-casing batches; keep the default ``"auto"``
unless memory measurements justify an override.
The factored reverse pass differentiates the equilibrium and coil data once,
rather than nesting the implicit equilibrium solve inside each Cartesian
spatial derivative. Third spatial derivatives and their parameter VJPs remain
expensive; the examples print progress before each order and use compact grids
for a bounded cold run.

Resources and reproducibility
-----------------------------

A single equilibrium uses XLA threading. Parallel finite differences and
ensembles use process-available CPUs by default, respecting scheduler
affinity; set ``workers`` explicitly when sharing a node. Device
selection is controlled by ``device=`` and the policies in
:doc:`/howto/run-on-gpu`.

JAX compilation is structural. A new resolution or objective shape compiles
new executables; repeated equal-shape stages reuse them and the persistent
machine-local cache. Optional ``compile_residual_and_jacobian`` and
``compile_value_and_gradient`` calls merely make the first compilation
visible with elapsed-time heartbeats.

API summary
-----------

The main entry points are :func:`vmex.core.optimize.make_problem`,
:class:`vmex.core.problem.FunctionProblem`,
:class:`vmex.core.problem.VmecProblem`,
:class:`vmex.core.problem.Evaluation`, and
:class:`vmex.core.monitoring.OptimizationMonitor`, and
:class:`vmex.core.monitoring.EquilibriumReporter`.


Optimizer and adjoint choices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The outer optimizer and equilibrium linear solver are independent choices.
Use ``opt.minimize(problem, method="SLSQP")`` for nonlinear constraints or
``method="L-BFGS-B"`` for box bounds; BFGS is also available for unconstrained
problems. Both upstream free-boundary single-stage examples use this same
adapter. ``minimize_projected`` remains a separate research algorithm because
its target restoration and stationarity criterion differ from SLSQP.

Free-boundary adjoint backends are ``coupled_gcrot`` (host iterations),
``boundary_schur`` (bulk elimination),
``edge_response`` (cached vacuum response), ``forward_dense`` (SciPy LU), and
``forward_dense_jax`` (JAX LU). They share acceptance/reporting conventions and
one accepted-root continuation interface. Dense methods retain their factors
for prediction; the other methods use the existing ``gcrot_tangent``
forward solve. A Schur adjoint does not yet retain Schur factors for its tangent.
Changing methods still requires independent derivative and progress checks.

The three-method production examples keep ``forward_dense_jax`` plus
``problem.enable_matrix_free()``: current-root GMRES with a seed-LU
preconditioner, and one checked dense retry. ``problem.solver_info`` reports the configured
adjoint, active reuse policy, recovery and predictor; event rows identify the
actual solver and true residual, including unsuccessful attempts before a
recovery. The optimization summary also saves this policy. The shared
linearized-transpose helper prepares an operator; its use does not select GCROT.

Seed-LU reuse remains specific to ``forward_dense_jax``. Other adjoint choices
are opt-in API alternatives, not qualified replacements for the production
configuration. Compare gradient and predictor accuracy, correction iterations,
and total gradient-plus-predictor time including factor construction before
changing the default. No timing advantage is implied by the common interface.

GCROT free-boundary prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Non-dense adjoint methods use the existing device GCROT forward tangent,
reported as ``gcrot_tangent``. This solves the parameter-direction response
at the accepted root; it is not a separate adjoint backend. It retains the
same root, mask, constraint baselines, iteration controls and independent
true-residual check. Dense and seed-LU methods use their own retained factors
for prediction.

The former ``adjoint_solver="reverse_gcrot"`` option has been removed.
Configurations using it now raise ``ValueError``. Select a supported method
explicitly and qualify its gradients before changing a production run.


Dense free-boundary adjoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``make_free_boundary_config(..., adjoint_solver="forward_dense")`` assembles
an explicit active-state Jacobian with forward JVPs and solves its transpose
using SciPy LU on the host. ``adjoint_solver="forward_dense_jax"`` keeps
matrix assembly and LU on the configured JAX device. Neither option
changes the nonlinear equilibrium or requires a root polisher.

The sparse orthonormal active basis removes masked coefficients and the
redundant m=1 Z-sine pairs; asymmetric Z-cosine pairs carry opposite signs,
as required by main's projector. For A = Q.T F_z Q, both methods solve
A.T lambda_active = Q.T objective_state and lift lambda = Q lambda_active.
A shared multi-RHS call assembles and factors A once for all rows.
Parameter derivatives use main's existing residual pullback. Add explicit
objective dependence on coil/profile parameters to state-only pullbacks.

Both interfaces require float64 and host-eager inputs. An ordinary scalar
``jax.grad`` and the continuation state pullback are supported; wrapping
the scalar gradient in an outer ``jax.jit`` raises a clear error because
the runtime mask determines the active dimension. The JAX backend name
describes matrix placement, not a fully traced control plane. No Krylov
fallback is silently substituted.

``adjoint_dense_batch_size`` (default 4) bounds the number of simultaneous
JVP columns. ``adjoint_dense_max_dofs`` (default 4096) rejects larger active
systems before allocating the quadratic matrix; increase it explicitly
only after budgeting matrix, factorization, and JVP working memory. A
4096-by-4096 float64 matrix alone occupies 128 MiB, excluding workspace.
The parameter count is independent of this active-state dimension.

After factorization, each actual transpose equation is checked against
main's existing acceptance threshold (10 * adjoint_tol * norm(rhs)), or
``adjoint_residual_rtol * norm(rhs)`` when explicitly configured. Shared
pullbacks append per-row residual certificates to an optional ``diagnostics``
list, matching the strict-continuation interface.
Singular or nonfinite solutions raise AdjointSolveError. Finite residual
failures follow the explicit adjoint_fail policy. The root-residual gate
and independent finite-difference qualification remain necessary. Dense
methods are optional alternatives; coupled_gcrot remains the default.


Scalar free-boundary problems
-----------------------------

``opt.FreeBoundaryProblem.from_loss(inp, loss, ...)`` follows the scalar
``VmecProblem.from_loss`` convention. The JAX-traceable loss receives
``(state, runtime, coils)`` and returns one scalar. Its gradient includes the
implicit equilibrium response and explicit coil derivative. Optional
``quantities=(function, ...)`` use the usual ``(state, runtime)`` signature;
``constraint_values(x)`` and ``constraint_jac(x)`` expose physical values and
rows for an external optimizer to bound. Values alone do not compute adjoints.

Use ``opt.boundary_from_state(state, runtime)`` for differentiable physical
``(RBC, ZBS, RBS, ZBC)`` edge arrays, indexed by ``(n+NTOR, m)``. This undoes
the m=1 constraint and Fourier normalization; moving-surface coil terms must
be inside the scalar loss to include their equilibrium response.
``opt.boundary_from_wout(wout, mpol=..., ntor=...)`` supplies the same physical
edge arrays from WOUT, retaining/truncating/padding Fourier modes as requested.
Use the matching input deck for profiles and ``vmex.state_from_wout`` for the
spectral restart.

Ordinary evaluations never promote a trial. Call ``problem.accept_x(x)`` only
after optimizer acceptance. Scalar problems retain the accepted tangent and
use one bounded correction from that state. Residual ``from_tuples`` problems
retain their existing continuation and projected-optimizer behavior.

``problem.enable_matrix_free(...)`` constructs a checked dense seed at the
accepted root and retains its host LU for production. Passing an explicit
``direction`` additionally compares matrix-free gradients and that tangent
against dense, returning a qualification report. Omitting the direction skips
only that comparison; it never disables actual residual checks.
The solver's ``adjoint_residual_rtol`` still checks full residuals independently
of the Krylov request. Failed trial adjoints get one dense retry; the seed is
refreshed only after accepting that recovered trial. ``close()`` releases
retained factors. Independent derivative checks can call
``evaluate_trial(delta, predict=False, ftol=...)`` to bypass the predictor and
correct from the accepted state at a tighter force tolerance.

Adaptive refresh is opt-in: ``problem.enable_matrix_free(refresh_horizon=10)``
compares the latest three-step median of adjoint, predictor and root-polishing time with the
best warm median since the seed was built. It rebuilds when the estimated
savings over ten further accepted steps exceed the measured dense rebuild
cost. The first two accepted steps after each seed are excluded to avoid
compilation-driven refreshes. Rejected trials do not contribute samples.
The rebuild uses ``forward_dense_jax`` at the optimizer-accepted candidate;
checked factors replace the old seed only after the rebuild succeeds.
The rebuilt total-derivative rows must agree with the current rows within
``parity_rtol`` (default ``1e-6``). A failed rebuild or disagreement preserves
the accepted root, retained factors and refresh policy. Set
``refresh_max_steps`` to the remaining accepted-step budget to limit the
payback horizon and skip a cost-triggered rebuild on the last step.
All force, root and full linear residual tolerances remain unchanged.
The horizon is a cost estimate, not a guaranteed speedup or a fixed rebuild
interval. Omit it to retain refresh-on-failure only. Events report the refresh
reason and measured cost; production setup does not add gradient experiments.

Coupled-root polishing is also opt-in, for scalar losses using
``forward_dense_jax``. Before enabling matrix-free reuse or starting the
optimizer, call::

    problem.enable_root_polishing(tolerance=1e-12)
    problem.enable_matrix_free(refresh_horizon=10, refresh_max_steps=100)

The first call refines the initial equilibrium if necessary and enables the
same refinement for subsequent candidates, before values or gradients are
returned. It takes at most three Newton steps and eight damping probes per
step, using the retained LU as a preconditioner when available. A failed
matrix-free refinement receives one temporary dense retry. Temporary trial
factors never replace the accepted seed. Changed states receive fresh vacuum,
force and coupled-residual certification, retaining inactive coordinates and
constraint baselines. Initialization failures leave the original root usable;
trial failures are rejected. Already-polished checkpoints retain their exact
state, and enabling the option does not count as an optimizer step.

``evaluate_trial(..., predict=False)`` still starts from the last accepted
equilibrium and applies the same root refinement; it does not use a tangent
predictor. If an explicit tighter ``ftol`` is requested, the refined force
diagnostics must also satisfy it. Polishing and residual gates do not replace
independent derivative or resolution tests.

See ``examples/single-stage-benchmarks/free_boundary_single_stage_optimization_scalar.py`` for
SLSQP with fixed currents, stage-two coil fitting, scalar plasma/coil penalties
and final verification. ``free_boundary_single_stage_optimization.py`` uses
the same workflow with bounded L-BFGS-B. It optimizes the same weighted loss
without SLSQP's nonlinear inequalities; physical iota/radius limits remain
endpoint checks. Only its accepted-iteration callback promotes the warm-start
state, never its line-search gradient evaluations.
``verify_free_boundary_single_stage.py`` imports the
same problem builder for independent derivative checks and saves a qualified
initial checkpoint. Production can start directly from input/WOUT and coils,
or optionally restore a matching report and re-certify its checkpoint. Neither
route runs finite differences or parity experiments. Runs without a supplied
report record ``derivative_qualified=False``; residual checks and the final
equilibrium solve remain part of production.
A matched GPU qualification and pilot are still needed before claiming
production timing or physical convergence.

The example exposes ``build_problem``, ``run_optimizer``, ``verify_endpoint``
and ``postprocess``. ``opt.OptimizationQualification.signature/read`` binds
numerical reports to their code, configuration, runtime and saved artifacts.
``problem.state_from_checkpoint`` authenticates accepted snapshots for replay
without another equilibrium solve; it does not re-certify them.
``opt.CoilDiagnostics`` reports physical coil motion and current changes without
an equilibrium/field evaluation. Expensive numerical qualification stays in the
separate verification programs. Both benchmark folders use shared production
support under ``examples/single_stage_support/``. The fixed builder is shared
between both folders and uses public accepted-state ownership and
``opt.minimize``. ``verify_single_stage.py`` independently checks its objective
and enabled constraints without starting joint optimization.

Default production force tolerance and endpoint verification use ``1e-15``.
Qualification signatures can bind explicit numerical functions through
``functions=``. Their executable syntax excludes comments and docstrings;
callers still bind numerical parameters, helpers, input, core and runtime.
Reporting and budget edits outside those functions do not require repeated
qualification. Numerical changes require a new standalone report before
claiming derivative qualification; old whole-file contracts remain strict.



Accepted-state certification
----------------------------

``make_free_boundary_continuation_config_from_state`` imports an equilibrium
with explicit constraint baselines and freshly certifies it without invoking
an equilibrium solver. ``certify_free_boundary_continuation_state`` returns
an owner-bound record for explicit promotion. Failed certification never
changes the old anchor or launches a recovery solve.

The free-boundary solver accepts opt-in ``include_edge_in_convergence`` and
``edge_force_tolerance`` controls; ``SolveResult.fedge`` reports the spectral
edge-force residual. Fresh certification checks the exact returned state.

For every free-boundary backend, ``adjoint_residual_rtol``
optionally specifies a strict
true-residual acceptance threshold independently of ``adjoint_tol``. Its
``None`` default retains the existing tenfold acceptance margin. The shared
pullback accepts an optional ``diagnostics`` list populated with each
row's residual norm, RHS norm, threshold and iterations. No extra equilibrium
or linear solve is performed for this reporting.

Free-boundary coil variables
----------------------------

``opt.FreeBoundaryProblem`` uses the same ``FunctionProblem`` interface as
``VmecProblem``. Its design vector contains coil currents and Cartesian
Fourier coefficients; the LCFS is the result of the free-boundary equilibrium.
The class owns no files, command-line arguments, or signal handlers.

For an input with ``lfreeb=True`` and a confining initial coil set:

.. code-block:: python

   qs = opt.QuasisymmetryRatioResidual(
       (0.25, 0.5, 0.75, 1.0), helicity_m=1, helicity_n=0)
   problem = opt.FreeBoundaryProblem.from_tuples(
       inp, [(qs, 0.0, 1.0)],
       coils=coils, current_dofs=(1, 2, 3), max_coil_mode=4,
       restart_from=initial_state,
       constraints=[
           opt.TargetBand(opt.mean_iota, 0.2, rtol=0.01, scale=0.005),
           opt.TargetBand(opt.aspect_ratio, 5.0, rtol=0.01, scale=0.05),
       ],
       solver_options=dict(
           ftol=1e-11, edge_force_tolerance=1e-11,
           max_iterations=12000, adjoint_solver="forward_dense_jax",
           adjoint_tol=2e-5, adjoint_residual_rtol=2e-5),
   )
   monitor = opt.OptimizationMonitor(problem)
   result = opt.minimize_projected(problem, maxiter=10, callback=monitor)
   coils_final = problem.coils_from_x(result.x)
   equilibrium = problem.equilibrium_from_x(result.x)
   problem.close()

The scalar production examples in ``examples/single-stage-benchmarks/``
use ``from_loss`` with direct coil penalties. The residual interface above
remains available for objectives expressed as tuples.

Coordinates and constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``CoilParameters`` places selected relative current changes first, followed
by additive coefficients in metres in ``(coil, Cartesian coordinate, Fourier
coefficient)`` order. Fourier order is constant, sine-1, cosine-1, sine-2,
cosine-2, and so on. Unselected currents and higher modes stay fixed. Base
currents are physical amperes, not ESSOS's scaled current DOFs. Symmetry copies
are not independent variables. ``dof_names`` and ``scales`` follow this exact
ordering; pass an explicit ``parameterization`` to retain an existing chart.

ESSOS compatibility
~~~~~~~~~~~~~~~~~~~

This free-boundary interface supports unmodified ESSOS ``main`` (validated at
``c9b41222e06aed62c246ca3b35349e427a3ea239``) and the research branch. Its ESSOS
adapter is contained in ``vmex/core/coil_parameters.py``; no ESSOS patches or
version-specific branches are needed. ``from_coils`` reads physical currents
through ``dofs_currents_raw`` and removes ``curves.scaling`` from public Fourier
DOFs once, when constructing the parameterization. Non-default ESSOS scaling
therefore preserves the physical geometry and currents.

The traced field path uses standard ``Curves``/``Coils`` construction and passes
geometry/current arrays to ``DirectCoilField``. It does not pass a ``Coils``
pytree through the equilibrium solver, so it does not require the research
branch's dynamic current-scale metadata fix. The projected optimizer, compact
derivatives, retained tangent factors and batching policy are unchanged.

``tests/test_coil_parameters.py`` covers scaled inputs, standard JSON round
trips, and nested JIT forward/reverse derivatives. Run it with the selected
ESSOS source root first on ``PYTHONPATH``. ``benchmarks/coil_parameters.py``
compares field values, both Jacobians, compiler input hashes, compilation time
and synchronized warm timings. These local coil checks do not measure a whole
GPU equilibrium or optimizer step.

Use ``Coils.to_json`` / ``Coils.from_json`` for new exports. Legacy fork JSON
with normalized ``dofs_currents`` has different loader semantics on upstream;
convert such inputs to explicit physical arrays before using this interface.
The maintained example uses standard ESSOS JSON with physical currents.

Target bands
~~~~~~~~~~~~

``TargetBand`` is a physical constraint, not an objective penalty. Its
acceptance tolerance is ``atol + rtol * abs(target)``; its independent
``scale`` conditions the target error and its Jacobian. A zero target requires
positive ``atol``. The initial normalization is
``max(norm(weighted_objective_residuals), 0.001)``. Pass the saved positive
numeric normalization when resuming; it is never reset after promotion.

Derivatives and evaluation ownership
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``value_and_grad`` and ``constraint_jac`` share a compact pullback with one
objective-norm row plus one row per physical constraint. The full pointwise
objective Jacobian is built only by ``residual_jac`` / ``residual_and_jac``.
``compile_value_and_gradient`` warms the compact path;
``compile_residual_and_jacobian`` explicitly requests the full path.

Every trial uses tangent prediction, ordinary correction, and strict force,
edge, and projected-root certification. Evaluation does not move the accepted
anchor. ``accept(candidate)`` is called only after the optimizer's nonlinear
acceptance checks, and invalidates old derivative factors and evaluation caches.
The retained linearization belongs to one root and configuration.

``coils_from_x`` performs no equilibrium solve. ``equilibrium_from_x`` reuses
the most recent evaluation or solves/certifies the requested parameters without
promoting them. Its WOUT property computes vacuum data on that exact geometry;
it does not reconverge or replace the accepted equilibrium.

Limits and results
~~~~~~~~~~~~~~~~~~

This interface requires float64 JAX and ``adjoint_fail="error"``. Choose
any supported backend through ``solver_options["adjoint_solver"]``.
Residual ``from_tuples`` objectives and constraints depend on
``(state, runtime)``; scalar ``from_loss`` additionally includes explicit
coil-objective derivatives. Calls stay eager
outside the compiled state functions; do not wrap the problem in ``jax.jit``.

``minimize_projected`` preserves projected descent, adaptive target restoration,
physical coil/current step caps and finite backtracking. ``maxiter`` counts
new accepted steps. Its result includes ``x``, ``fun``, ``constraints``,
``feasible``, ``accepted``, ``nit``, ``nfev`` (trial evaluations), ``njev``
(linearization requests), and the original projected-gradient reference.
The callback receives an ``OptimizeResult`` after each accepted step; raising
``StopIteration`` returns the last accepted state with ``callback_stopped``.

``success=True`` means feasible equality-tangent stationarity. It is not a
full inequality KKT certificate or independent physical qualification.
``step_budget_reached`` and ``stagnated`` both have ``success=False``.
The caller owns checkpoint authentication and numerical qualification.


Checkpoint restoration and batch tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``restart_from`` accepts a lossless spectral seed path and performs one ordinary
solve and fresh certification. It does not treat the seed as a certified root.

For an exact optimizer resume, supply ``checkpoint``, ``checkpoint_sha256`` and
``checkpoint_identity`` to ``FreeBoundaryProblem.from_tuples``. The identity
must describe the objective definitions and optimizer options. The library also
binds the input, coil chart, target bands, solver options and continuation gates.
Checkpoint storage requires construction with ``solver_options``; attaching an
existing ``continuation`` remains a separate in-memory interface.
Changed numerical settings are rejected before certification. Certification
must preserve the saved state, masks and constraint baselines exactly and never
launches an ordinary solve. The problem exposes ``accepted_step`` and
``initial_gradient_norm`` and writes accepted states with ``save_checkpoint``.
Pass the saved gradient reference to ``minimize_projected`` on resume.

The default dense batch size remains 32. ``problem.tune_adjoint_batch()`` is
optional: it compares 32/64 at the unchanged root, checks gradient agreement,
and retains only the selected certified factors.

Coil geometry constraints
-------------------------

``FreeBoundaryProblem.from_loss`` accepts ``coil_quantities``, scalar callables
with signature ``function(state, runtime, coils)``. These rows follow the
ordinary ``quantities`` in ``constraint_values`` and ``constraint_jac``. Their
derivatives include both explicit coil dependence and the equilibrium response.
For example, a coil-to-plasma clearance must move both the coils and the plasma
boundary. Coil-only quantities should use direct optimizer constraints to avoid
adding unnecessary equilibrium adjoint right-hand sides.

``examples/coil-constraints-benchmarks/`` provides matched fixed/free-boundary
SLSQP drivers and a shared ``parameters.py``. The filament bounds include
per-coil length, maximum and arc-length-averaged squared curvature, intercoil
distance including symmetry copies, and clearance to the moving plasma surface.
Nonadjacent segment intersection and parametrization speed checks supplement
these bounds. The seed is infeasible; reducing QA alone is not a constrained
solution.

Geometry uses denser sampling than field quadrature. Endpoint checks refine
curvature peaks and compare mean-squared curvature on two grids. The surface
clearance check uses multiple continuous-coordinate searches, which do not
certify a global minimum or finite-winding-pack manufacturability. The example
README records the sampling, numerical guards and separate qualification steps.


Single-stage example layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both scalar formulations use the same short builder/optimizer/endpoint entry
structure. ``examples/single_stage_support/`` contains the shared file handling,
free-boundary setup, fixed-boundary setup and diagnostics. Its ``data/`` directory
holds one copy of the common input and coil files; ``coils.fitted.json`` is the
unchanged former timestamp-named fitted coil file. Geometry tests are in
``tests/test_single_stage_coil_constraints.py``. Historical validation records
remain unchanged in ``benchmarks/single_stage_provenance/``.

Production never repeats derivative qualification. Separate verifiers call the
same builders, and free-boundary qualification fingerprints include the shared
implementation. Reorganized source therefore requires a new matching report;
old reports remain valid only for their original source snapshot. This layout
change does not alter the objective weights, physical constraints or solvers.
