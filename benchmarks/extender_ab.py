"""A/B timing of VmecExtender spatial derivatives and near-surface queries.

One fresh process per arm prints one ``RESULT {json}`` line: first-call and
best-of-three warm seconds for ``B`` through ``gradgradgradB`` at 16 and 128
targets one minor radius off the 2.5 % beta QA boundary (direct path, no
accuracy check), and for ``B`` at 16 targets 0.05 minor radii off at the
default settings.  ``benchmarks/extender_ab_20260923.json`` records an
A/B/A/B run of main against the graded-rule branch.

    PYTHONPATH=<tree> python benchmarks/extender_ab.py <label> \
        examples/data/input.LandremanPaul2021_QA_beta2p5_bootstrap
"""
import json
import os
import sys
import time
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import jax
import jax.numpy as jnp
import warnings
warnings.simplefilter("ignore")
import vmex as vj
from vmex.core.multigrid import solve_multigrid
from vmex.core.extender import VmecExtender
from vmex.core import virtual_casing as vc

label = sys.argv[1]
deck = sys.argv[2]
inp = vj.VmecInput.from_file(deck)
res = solve_multigrid(inp, ns_array=[9, 17, 31], ftol_array=[1e-10, 1e-10, 1e-12], niter_array=[4000, 4000, 20000])
state = res.state
data = vc.surface_field_data_from_state(inp, state, nphi=64, ntheta=64)
g = np.moveaxis(np.asarray(data.gamma), 0, -1).reshape(-1, 3)
n = np.moveaxis(np.asarray(data.normal), 0, -1).reshape(-1, 3)
R = np.hypot(g[:64, 0], g[:64, 1]); a = 0.5 * np.ptp(np.hypot(g[:, 0], g[:, 1]).reshape(64, 64)[0])
rng = np.random.default_rng(0)
idx = rng.choice(len(g), 256, replace=False)
far = jnp.asarray(g[idx[:128]] + a * n[idx[:128]])        # d = a: resolved by the direct path
near = jnp.asarray(g[idx[128:144]] + 0.05 * a * n[idx[128:144]])  # d = 0.05 a

def timed(fn, *args):
    t = time.perf_counter(); out = jax.block_until_ready(fn(*args)); return time.perf_counter() - t, out

row = {"label": label}
field = VmecExtender.from_state(inp, state, accuracy_check="off")
if hasattr(field, "near_surface"):
    field.near_surface = "direct"
for npts in (16, 128):
    X = far[:npts]
    for name in ("B", "gradB", "gradgradB", "gradgradgradB"):
        first, _ = timed(getattr(field, name), X)
        warm = min(timed(getattr(field, name), X)[0] for _ in range(3))
        row[f"{name}_{npts}_first"] = round(first, 3); row[f"{name}_{npts}_warm"] = round(warm, 4)
# near-surface B at default settings (main: direct with warning; branch: auto -> graded)
near_field = VmecExtender.from_state(inp, state, accuracy_check="off")
first, B = timed(near_field.B, near)
warm = min(timed(near_field.B, near)[0] for _ in range(3))
row["near_B_16_first"] = round(first, 3); row["near_B_16_warm"] = round(warm, 4)
row["near_B_values_sum"] = float(np.sum(np.abs(np.asarray(B))))
print("RESULT " + json.dumps(row), flush=True)
