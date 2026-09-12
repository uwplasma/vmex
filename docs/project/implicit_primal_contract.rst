Implicit primal eligibility
===========================

The ordinary fixed-boundary implicit APIs require a finite residual tolerance
``ImplicitConfig.primal_tol`` (default ``1e-10``). This is independent of
``refine_tol``: setting the latter to infinity disables Newton refinement,
but does not qualify an approximate root for differentiation.

Admission measures the assembled state actually returned to the objective.
A fresh force evaluation must have a finite projected preconditioned residual
norm below ``primal_tol``, finite nonnegative normalized raw FSQ no larger
than ``max_fsq_ratio * ftol``, finite state coefficients, and no Jacobian sign
change. The host solver's historical FSQ and stopping flag cannot substitute
for these checks. These tolerances refer to different residual normalizations;
they are not interchangeable or estimates of physical force error.

``solve_implicit_status`` returns status 2 and a zero pullback for an
uncertified primal, allowing optimizer callers to use their value-only or
penalty branch. Its returned FSQ fields retain their historical host meaning.
``VmecProblem.evaluate`` additionally reports ``primal_residual_norm``,
``primal_tol``, ``primal_fsq``, ``primal_fsq_ratio`` and
``primal_geometry_valid`` when the matching status certificate is available.
A missing certificate cannot claim derivative eligibility. Matching checks both
the exact parameters and a hash of the refined coefficients. Materialized
equilibria and first-order predictor seeds also use the matching refined
anchor; their host FSQ fields retain the original stopping diagnostics.

The ordinary scalar reverse API and batched tangent/pullback APIs raise on an
uncertified concrete primal. Under tracing, they produce nonfinite derivatives;
the tangent report also marks convergence false. Value-only solves remain
available. Linear-response residual certification remains a separate check.

This contract does not certify conditioning, a unique equilibrium, agreement
between cold and warm starts, or physical force convergence under resolution
refinement. Re-solved Taylor tests, tangent/adjoint duality, and independent
physical force checks are still necessary. The certificate is for the ordinary
fixed-boundary force equations; it does not replace the coupled NESTOR
free-boundary derivative contract.

Derivative caches retain the certificate captured when their arrays were
computed. A later global solve does not invalidate an already certified cached
pair. Diagnostics expose ``derivative_anchor_matches_current_refinement`` when
such a pair is revisited; mixed or missing cache provenance cannot certify the
combined evaluation. A certified derivative at one root does not establish
uniqueness or that a later warm/cold solve found that same root.

Validation checkpoint (2026-09-12)
----------------------------------

The focused status, tolerance, state identity, cache provenance and
materialization tests passed (23 tests). One small Solovev force pass confirms
that the measured norm equals the residual actually linearized. The existing
converged Solovev scalar-gradient test passed with a fresh finite-difference
cache: primal norm ``3.35468e-14``, fresh raw FSQ ``5.43259e-24``, valid geometry,
and maximum relative error ``5.09e-8`` across four scalar/parameter comparisons
against a ``1e-6`` threshold. This is CPU evidence for the ordinary positive
path, not a QA Taylor-study or a performance benchmark. GPU parity and
production optimization qualification remain separate work.

A concrete rejected primal stops before the batched linear solves. Under
tracing, the batched APIs currently evaluate their linear-response graph and
then mask rejected outputs. Rejection therefore prevents a usable derivative
but does not promise to avoid that computation. Moving these branches behind
a runtime conditional and measuring its compile/runtime effects is still a
performance task.


Materializing cached equilibria
------------------------------

Each scalar-gradient and residual-Jacobian cache retains a read-only host copy
of its certified coefficients and the source solve result. Materializing that
cached point uses this copy even if a later global solve changes or discards
its anchor. Historical host FSQ remains attached to the original source solve.
Storage is bounded by the existing one-entry cache for each derivative form;
replacing an entry releases its previous anchor.

If both derivative caches match the requested point but certify different
anchors, materialization raises an explicit error and evaluation reports an
uncertified state. It does not silently choose one branch. When no certified
cached derivative exists, materialization retains the ordinary current
host/refined-state path.

The bound equilibrium's interior and exterior fields are snapshots of those
same coefficients, including their spatial derivatives. They do not expose
optimization-parameter VJPs: requesting one raises the field API's explicit
"not constructed with optimizable parameters" error. Snapshot exterior
construction accepts an evaluated ``external_field`` and rejects parameterized
external-field factory arguments. The uncached materialization path retains
its existing live parameterized fields.

This is a deliberate restriction on snapshot fields, not completion of field
optimization differentiability. A future parameterized continuation anchored
at the saved equilibrium needs branch-consistency and re-solved derivative
validation; switching to a different live root only away from the saved point
would not define a continuous function. Cache/provenance updates and reads
share the existing per-problem reentrant lock.
