# Optimizers

Reusable optimization routines live here.  Problem-specific example scripts
should assemble objectives in `examples/optimization/` and call these helpers
instead of hiding the workflow in a large wrapper.

- `fixed_boundary/`: fixed-boundary QS/QI residual construction, scalar-trust,
  matrix-free, and exact-gradient helpers.

Keep objective-specific physics terms in their physics domain modules when
possible; optimizer modules should focus on stepping, scaling, and derivative
strategy.
