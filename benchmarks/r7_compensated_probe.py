"""R7 probe: is the stationarity floor input rounding or internal arithmetic?

For stored physical coefficients B, form hi = one-ulp neighbour of B on a random
half of entries and the exact remainder lo = B - hi (Sterbenz-exact).  Compare
  plain:       g(hi)                      (input-rounded evaluation)
  compensated: g(hi) + D_B g(hi)[lo]      (first-order remainder restored)
against g(B).  If the compensated error is far below the plain error, the floor
is input representation, curable by a declared hi+lo coefficient state.  The
remaining compensated error measures internal float64 evaluation noise.
Read-only; writes one JSON.
"""
import argparse
import functools
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import polish_recovery_sparse as prs  # noqa: E402

FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")

p = argparse.ArgumentParser(); p.add_argument("--state", type=Path, required=True)
p.add_argument("--output", type=Path, required=True); p.add_argument("--trials", type=int, default=3)
p.add_argument("--fields", default=",".join(FIELDS))
p.add_argument("--stable-derivatives", action="store_true")
a = p.parse_args(); t0 = time.perf_counter()
if a.stable_derivatives:
    prs.make_variational_plan = functools.partial(prs.make_variational_plan, stable_derivatives=True)
base, accepted, plan, layout, gauge, scale, coords, _ = prs._load_original_problem(a.state)
# Re-anchor at the accepted state with c=0 so the stored state is exactly B.
zero = jnp.zeros_like(coords)
C = prs.native_tangential_gauge_matrix(base, layout, gauge, scale).tocsr()
n, q = C.shape[1], C.shape[0]
lu = splu(sparse.bmat([[sparse.eye(n, format="csc"), C.T], [C, None]], format="csc"))


def proj(v):
    return lu.solve(np.r_[np.asarray(v), np.zeros(q)])[:n]


def grad_of(*arrays):
    st = replace(accepted, **dict(zip(FIELDS, arrays)))
    def obj(c):
        r = prs.native_physical_force_residual(c, st, layout, gauge, scale, prs.FORCE_SCALE, prs.VOLUME_SCALE)
        return 0.5 * jnp.vdot(r, r)
    return jax.grad(obj)(zero)
grad_j = jax.jit(grad_of)
jvp_j = jax.jit(lambda prim, tan: jax.jvp(grad_of, prim, tan))
B = tuple(jnp.asarray(getattr(accepted, f)) for f in FIELDS)
gB = np.asarray(grad_j(*B)); pB = proj(gB)
rng = np.random.default_rng(17); rows = []
chosen = set(a.fields.split(","))
for t in range(a.trials):
    hi, lo = [], []
    for f, b in zip(FIELDS, B):
        b = np.asarray(b)
        if f in chosen:
            m = rng.random(b.shape) < 0.5
            d = np.where(rng.random(b.shape) < 0.5, -np.inf, np.inf)
            h = np.where(m, np.nextafter(b, d), b)
        else:
            h = b.copy()
        hi.append(jnp.asarray(h)); lo.append(jnp.asarray(b - h))
    g_hi, dg = jvp_j(tuple(hi), tuple(lo))
    g_hi = np.asarray(g_hi); g_c = g_hi + np.asarray(dg)
    rows.append({"plain_projected_error": float(np.linalg.norm(proj(g_hi - gB))),
                 "compensated_projected_error": float(np.linalg.norm(proj(g_c - gB))),
                 "remainder_linear_term": float(np.linalg.norm(proj(np.asarray(dg))))})
# Internal noise reference: same exact input evaluated eagerly (different fusion).
with jax.disable_jit():
    g_eager = np.asarray(grad_of(*B))
out = {"schema": "vmex-r7-compensated-probe/1", "state_sha256": hashlib.sha256(a.state.read_bytes()).hexdigest(),
       "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "fields": sorted(chosen), "stable_derivatives": a.stable_derivatives,
       "projected_gradient_norm": float(np.linalg.norm(pB)), "gate_threshold_projected": 5.56307046831159e-08,
       "trials": rows, "jit_vs_eager_projected_difference": float(np.linalg.norm(proj(g_eager - gB))),
       "elapsed_seconds": time.perf_counter() - t0}
a.output.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
