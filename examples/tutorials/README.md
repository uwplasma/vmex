# Tutorials

These scripts are intended to be read/run in order. They provide a guided path from
input parsing and geometry reconstruction to simple fixed-boundary solves.

Run from the repo root:

```bash
python examples/tutorials/00_parse_and_boundary.py examples/data/input.circular_tokamak --out boundary_step0.npz --verbose
```

Outputs are typically `.npz` files suitable for quick plotting or regression comparisons.
