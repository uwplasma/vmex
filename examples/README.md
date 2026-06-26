# Examples

All runnable examples live under this single `examples/` tree.

- Top-level scripts demonstrate common workflows.
- `optimization/`: fixed-boundary, QI/QS, finite-beta, and coil optimization
  examples.
- `diagnostics/`: small scripts for plotting or inspecting solver outputs.
- `validation/`: optional parity or external-code comparison examples.
- `data/`: bundled input decks and small checked-in fixtures.
- `data/single_grid/`: fixed-boundary single-grid benchmark inputs and optional
  fetched reference assets.

Generated outputs should go to ignored `results/`, `outputs/`, or a user-chosen
directory.  Do not commit generated WOUT, mgrid, Boozer, PDF, or plot files
unless they are compact reviewed documentation artifacts.
