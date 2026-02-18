# Optimization tutorials

Autodiff and optimization examples (requires JAX).

- `grad_bmag_wrt_boundary.py`: differentiate through a tiny fixed-boundary solve
  and compute the gradient of mean(|B|) on the LCFS with respect to boundary
  coefficients.

Tip: enable 64-bit in JAX for parity work (`vmec_jax._compat.enable_x64(True)`).
